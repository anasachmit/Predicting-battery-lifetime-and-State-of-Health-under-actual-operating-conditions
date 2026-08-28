# -*- coding: utf-8 -*-
"""T1 - evaluation leave-one-cell-out du modele d'ancrage contre le baseline
officiel REFITE dans les memes conditions (jamais contre son score publie).

Rappel du contexte : la metrique officielle n'est pas publiee. On rapporte donc
nos deux proxies (`uniforme_cycle` et `uniforme_soh`) et, conformement a la
regle de selection, leur MAXIMUM - jamais leur moyenne - ainsi que l'ecart
entre les deux comme indicateur de fragilite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .anchor_model import (fit_joint, fit_condition_map, trajectory, b_at,
                           N_MAX_PREDICT)
from .metrics import evaluate_cell, evaluate_cell_full, relative_to_baseline
from .design import loco_qualification

TEMPLATE = (ROOT / "Predicting-battery-lifetime-and-State-of-Health-under-"
                   "actual-operating-conditions" / "techarena-2026-topic1-template")


def load_target_cells(data_dir=None):
    if str(TEMPLATE) not in sys.path:
        sys.path.insert(0, str(TEMPLATE))
    from framework.data import load_cells
    data_dir = data_dir or (ROOT / "data" / "forfinetune")
    out = []
    for c in load_cells(str(data_dir), verbose=False):
        d = c.soh.dropna()
        n = d["cycle_number"].to_numpy(float)
        y = d["soh_percent"].to_numpy(float)
        out.append(dict(cell_id=c.cell_id, T=float(c.temperature_degC),
                        C=float(c.c_rate), n=n, y=y,
                        s0=float(np.median(y[n <= 10]))))
    return out


# ------------------------------------------------- baseline officiel refit --
def fit_baseline(cells):
    """Reimplementation fidele de my_model/model_example.py."""
    rows = []
    for cl in cells:
        n, y = cl["n"], cl["y"]
        X = np.column_stack([np.ones_like(n), n ** 0.8])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(-coef[1])
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            rows.append((a, b, cl["T"], cl["C"]))
    A = np.array(rows)
    a_ = float(A[:, 0].mean())
    Tk = A[:, 2] + 273.15
    X = np.column_stack([np.ones(len(A)), 1000.0 / Tk, np.log(A[:, 3])])
    w_ = (np.linalg.lstsq(X, np.log(A[:, 1]), rcond=None)[0] if len(A) >= 3
          else np.array([float(np.log(A[:, 1]).mean()), 0.0, 0.0]))
    return a_, w_


def predict_baseline(a_, w_, T, C):
    b = float(np.exp(w_[0] + w_[1] * 1000.0 / (T + 273.15) + w_[2] * np.log(C)))
    n = np.arange(1, N_MAX_PREDICT + 1, dtype=float)
    return np.clip(a_ - b * n ** 0.8, 0.5, 119.9)


# ----------------------------------------------------------------- LOCO -----
def run_loco(cells=None, free_c_exponent=False, return_traj=False):
    cells = cells or load_target_cells()
    rows, params, trajs = [], [], {}
    for i, held in enumerate(cells):
        train = [c for j, c in enumerate(cells) if j != i]

        r = fit_joint(train)
        w, r2_b, _ = fit_condition_map(train, r["b"],
                                       free_c_exponent=free_c_exponent)
        s0 = float(np.mean([c["s0"] for c in train]))
        traj_anchor = trajectory(s0, b_at(w, held["T"], held["C"]),
                                 r["soh_k"], r["c"], r["beta"])

        a_, w_b = fit_baseline(train)
        traj_base = predict_baseline(a_, w_b, held["T"], held["C"])

        trajs[held["cell_id"]] = dict(
            n=held["n"], y=held["y"], T=held["T"], C=held["C"],
            ancrage=traj_anchor, baseline=traj_base,
            soh_k=r["soh_k"], b=b_at(w, held["T"], held["C"]), s0=s0)

        m_a = evaluate_cell(traj_anchor, held["n"], held["y"])
        m_b = evaluate_cell(traj_base, held["n"], held["y"])
        rel = relative_to_baseline(m_a, m_b)

        # regle de selection : on retient le PIRE des deux grilles
        pire_a = max(m_a["rmse_uniforme_cycle"], m_a["rmse_uniforme_soh"])
        pire_b = max(m_b["rmse_uniforme_cycle"], m_b["rmse_uniforme_soh"])

        rows.append(dict(
            cell_id=held["cell_id"], T=held["T"], C=held["C"],
            rmse_cycle_ancrage=m_a["rmse_uniforme_cycle"],
            rmse_soh_ancrage=m_a["rmse_uniforme_soh"],
            rmse_pire_ancrage=pire_a,
            ecart_entre_grilles=abs(m_a["rmse_uniforme_cycle"]
                                    - m_a["rmse_uniforme_soh"]),
            rmse_sous80_ancrage=m_a["rmse_sous_80"],
            rmse_cycle_baseline=m_b["rmse_uniforme_cycle"],
            rmse_soh_baseline=m_b["rmse_uniforme_soh"],
            rmse_pire_baseline=pire_b,
            rel_pire=pire_a / pire_b if pire_b else np.nan,
            rel_rmse_cycle=rel["rel_rmse_uniforme_cycle"],
            rel_rmse_soh=rel["rel_rmse_uniforme_soh"],
            n70_vrai=m_a["n70_true"], n70_ancrage=m_a["n70_pred"],
            n70_baseline=m_b["n70_pred"],
            eol_ape_ancrage=m_a["eol_ape"], eol_ape_baseline=m_b["eol_ape"],
            eol_observe=m_a["eol_observe"],
            violation_monotonie=m_a["taux_violation_monotonie"],
        ))
        params.append(dict(fold=held["cell_id"], soh_k=r["soh_k"], c=r["c"],
                           beta=r["beta"], w0=w[0], w1=w[1], w2=w[2],
                           r2_b=r2_b, s0=s0))
    if return_traj:
        return pd.DataFrame(rows), pd.DataFrame(params), trajs
    return pd.DataFrame(rows), pd.DataFrame(params)


def summarise(res, par, summ_target=None):
    """Synthese : par cellule, jamais en mediane sur n@70."""
    obs = res[res["eol_observe"]]
    out = {
        "n_folds": int(len(res)),
        "n_folds_eol_observe": int(len(obs)),
        "rmse_pire_ancrage_median": float(res["rmse_pire_ancrage"].median()),
        "rmse_pire_baseline_median": float(res["rmse_pire_baseline"].median()),
        "rel_pire_median": float(res["rel_pire"].median()),
        "rel_pire_max": float(res["rel_pire"].max()),
        "ecart_entre_grilles_max": float(res["ecart_entre_grilles"].max()),
        "eol_ape_ancrage_par_cellule": {r.cell_id: round(float(r.eol_ape_ancrage), 2)
                                        for r in obs.itertuples()},
        "eol_ape_baseline_par_cellule": {r.cell_id: round(float(r.eol_ape_baseline), 2)
                                         for r in obs.itertuples()},
        "violation_monotonie_max": float(res["violation_monotonie"].max()),
    }
    # dispersion des parametres entre folds : diagnostic d'identifiabilite
    for k in ("soh_k", "c", "beta", "w1"):
        v = par[k].to_numpy(float)
        out[f"dispersion_{k}"] = float(np.nanmax(np.abs(v)) /
                                       max(np.nanmin(np.abs(v)), 1e-12))
    return out
