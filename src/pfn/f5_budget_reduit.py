# -*- coding: utf-8 -*-
"""F.5 - le PFN en regime de budget reduit. ELIMINATOIRE.

Le pipeline officiel re-entraine chaque soumission avec moins de cellules et des
cycles de debut de vie seulement, et ce run porte le bonus de data-efficiency.
C'est le regime qui a tue L4 : excellent sur trajectoires completes, 154 % d'APE
mediane a 95 % de troncature.

Le PFN n'y a jamais ete teste. Il a vu 5 cellules de contexte COMPLETES ; en run
reduit il recevra 3 cellules tronquees.

Ce qui est degrade, c'est le CONTEXTE - pas la verite. La cellule retiree reste
evaluee sur sa trajectoire complete, parce que c'est bien la trajectoire
complete que le challenge note.

Critere, pose dans VERDICT.md AVANT execution : le PFN remplace v3 si, dans TOUS
les regimes, son RMSE median est <= a celui de v3, son RMSE pire cas est < a
celui de v3, et aucun fold ne depasse 100 % d'APE.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch

from ..config import RESULTS
from ..metrics import evaluate_cell_scoped
from .real_theta import load_real_cells
from .l4_retest import fit_joint, carte_conditions, predicteur
from .f2_reel import predit_pfn, fit_baseline, predit_baseline

OUT = RESULTS / "pfn"

# (troncature du contexte, nombre de cellules de contexte)
REGIMES = [(None, 5), (95.0, 5), (90.0, 5), (85.0, 5), (None, 3), (95.0, 3)]
N_COMBOS = 4        # combinaisons de 3 cellules explorees, quand n_ctx_cells=3


def tronque(c, seuil):
    """Coupe la trajectoire au premier franchissement de `seuil`."""
    if seuil is None:
        return c
    env = np.minimum.accumulate(c["y"])
    k = np.flatnonzero(env < seuil)
    stop = int(k[0]) if k.size else len(c["y"])
    if stop < 30:
        stop = min(len(c["y"]), 30)
    d = dict(c)
    d["n"], d["y"] = c["n"][:stop], c["y"][:stop]
    d["soh_min"] = float(np.minimum.accumulate(d["y"]).min())
    return d


def run(modele, n_ctx=32, seed=0):
    cellules = [c for c in load_real_cells() if c["source"] == "target"]
    lignes = []

    for seuil, n_cells in REGIMES:
        for i, held in enumerate(cellules):
            reste = [j for j in range(len(cellules)) if j != i]
            if n_cells >= len(reste):
                combos = [tuple(reste)]
            else:
                combos = list(itertools.islice(
                    itertools.combinations(reste, n_cells), N_COMBOS))

            for rep, combo in enumerate(combos):
                ctx = [tronque(cellules[j], seuil) for j in combo]

                t_pfn = predit_pfn(modele, ctx, held["T"], held["C"],
                                   n_ctx, seed)
                try:
                    f = fit_joint(ctx, p_free=False, seed=seed * 100 + i)
                    w3 = carte_conditions(ctx, f["b"])
                    t_v3 = predicteur(f, w3)(held["T"], held["C"])
                except Exception:
                    t_v3 = None
                try:
                    a_, wb = fit_baseline(ctx)
                    t_bl = predit_baseline(a_, wb, held["T"], held["C"])
                except Exception:
                    t_bl = None

                ligne = dict(regime=f"{'complet' if seuil is None else f'{seuil:.0f}%'}"
                                    f" / {n_cells} cellules",
                             troncature=(seuil if seuil else 999.0),
                             n_cellules=n_cells, fold=held["cell_id"],
                             rep=rep)
                for nom, traj in (("pfn", t_pfn), ("v3", t_v3),
                                  ("baseline", t_bl)):
                    if traj is None:
                        ligne[f"rmse_{nom}"] = np.nan
                        ligne[f"ape_{nom}"] = np.nan
                        continue
                    ev = evaluate_cell_scoped(traj, held["n"], held["y"],
                                              soh_floor=70.0)
                    ligne[f"rmse_{nom}"] = max(
                        ev.get("rmse_uniforme_cycle", np.nan),
                        ev.get("rmse_uniforme_soh", np.nan))
                    ligne[f"ape_{nom}"] = ev.get("eol_ape", np.nan)
                ligne["n70_observe"] = bool(ev.get("eol_observe", False))
                lignes.append(ligne)

    d = pd.DataFrame(lignes)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "f5_budget_reduit.csv", index=False)
    return d


def resume(d):
    """Un bloc par regime, avec les trois conditions du critere."""
    rows = []
    for reg, g in d.groupby("regime", sort=False):
        obs = g[g.n70_observe]
        r = dict(regime=reg, n_cas=len(g))
        for nom in ("pfn", "v3", "baseline"):
            r[f"rmse_med_{nom}"] = round(float(g[f"rmse_{nom}"].median()), 2)
            r[f"rmse_pire_{nom}"] = round(float(g[f"rmse_{nom}"].max()), 2)
            r[f"ape_med_{nom}"] = round(float(obs[f"ape_{nom}"].median()), 2)
            r[f"folds_sup_100_{nom}"] = int((obs[f"ape_{nom}"] > 100).sum())
        r["c1_rmse_med"] = bool(r["rmse_med_pfn"] <= r["rmse_med_v3"] + 1e-9)
        r["c2_rmse_pire"] = bool(r["rmse_pire_pfn"] < r["rmse_pire_v3"])
        r["c3_aucun_ape_100"] = bool(r["folds_sup_100_pfn"] == 0)
        r["regime_passe"] = bool(r["c1_rmse_med"] and r["c2_rmse_pire"]
                                 and r["c3_aucun_ape_100"])
        rows.append(r)
    return pd.DataFrame(rows)


def verdict(res):
    return dict(regimes_passes=int(res.regime_passe.sum()),
                regimes_total=int(len(res)),
                regimes_echoues=list(res[~res.regime_passe].regime),
                F5_passe=bool(res.regime_passe.all()))
