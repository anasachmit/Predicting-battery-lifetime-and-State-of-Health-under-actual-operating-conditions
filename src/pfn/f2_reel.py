# -*- coding: utf-8 -*-
"""F.2 - leave-one-cell-out sur les SIX CELLULES REELLES du challenge.

C'est le seul test de la branche qui porte sur les cellules cible et non sur des
laboratoires synthetiques. Trois modeles y sont compares dans des conditions
strictement identiques :

    PFN       le reseau pre-entraine sur le prior, qui ne voit les 5 cellules
              d'entrainement que comme CONTEXTE - aucun ajustement, un seul
              passage avant
    v3        le modele soumis (L3, exposant fige a 0.8, carte d'Arrhenius),
              reajuste sur les 5 memes cellules
    baseline  le baseline officiel refite sur les 5 memes cellules, jamais son
              score publie

Protocole, identique pour les trois
-----------------------------------
* leave-one-cell-out sur les 6 cellules cible, trajectoires completes ;
* metriques restreintes au perimetre note, SOH vrai >= 70 % ;
* selection sur le MAXIMUM des deux grilles de cycles, jamais leur moyenne ;
* les 4 APE individuelles sur n@70 sont rapportees - jamais la seule mediane,
  parce que 4 observations ne font pas une distribution.

Le PFN ne recoit AUCUN point de la cellule retiree, exactement comme a
l'inference reelle ou `predict_soh` ne recoit qu'un couple (T, C).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import RESULTS
from ..metrics import evaluate_cell_scoped
from .data import normalise, denormalise_soh
from .evaluate import GRILLE_EVAL
from .real_theta import load_real_cells
from .l4_retest import fit_joint, carte_conditions, predicteur

OUT = RESULTS / "pfn"
N_MAX = 12000


# ------------------------------------------------------------------- PFN ---
@torch.no_grad()
def predit_pfn(modele, cellules_contexte, T_req, C_req, n_ctx=32, seed=0,
               device="cpu"):
    """Trajectoire SOH 1..N_MAX predite par le PFN pour un couple (T, C).

    Le contexte est constitue des cellules REELLES d'entrainement, ramenees a
    `n_ctx` points chacune sur une grille geometrique - la meme construction
    que celle vue a l'entrainement.
    """
    modele.eval()
    ctx = []
    for c in cellules_contexte:
        n, y = c["n"], c["y"]
        if len(n) > n_ctx:
            idx = np.unique(np.round(np.geomspace(1, len(n), n_ctx)).astype(int) - 1)
            idx = np.clip(idx, 0, len(n) - 1)
            idx[-1] = len(n) - 1
            n, y = n[idx], y[idx]
        ctx.append(normalise(np.full(len(n), c["T"]), np.full(len(n), c["C"]),
                             n, y))
    Xc = torch.tensor(np.concatenate(ctx)[None], dtype=torch.float32, device=device)
    Xq = torch.tensor(
        normalise(np.full(len(GRILLE_EVAL), T_req),
                  np.full(len(GRILLE_EVAL), C_req), GRILLE_EVAL)[None],
        dtype=torch.float32, device=device)

    q = modele(Xc, Xq)[0].cpu().numpy()
    i_med = modele.quantiles.index(0.50)
    soh_grille = denormalise_soh(q[:, i_med])

    # Le scoring interroge les cycles 1..N_MAX un par un : on interpole la
    # sortie du reseau, puis on impose la monotonie decroissante par enveloppe
    # cumulative - une trajectoire de SOH ne remonte pas.
    n_plein = np.arange(1, N_MAX + 1, dtype=float)
    traj = np.interp(n_plein, GRILLE_EVAL, soh_grille)
    return np.minimum.accumulate(np.clip(traj, 0.5, 119.9))


# -------------------------------------------------------------- baseline ---
def fit_baseline(cellules):
    """Reimplementation fidele de my_model/model_example.py."""
    lignes = []
    for c in cellules:
        n, y = c["n"], c["y"]
        X = np.column_stack([np.ones_like(n), n ** 0.8])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(-coef[1])
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            lignes.append((a, b, c["T"], c["C"]))
    A = np.array(lignes)
    a_ = float(A[:, 0].mean())
    Tk = A[:, 2] + 273.15
    X = np.column_stack([np.ones(len(A)), 1000.0 / Tk, np.log(A[:, 3])])
    w = (np.linalg.lstsq(X, np.log(A[:, 1]), rcond=None)[0] if len(A) >= 3
         else np.array([float(np.log(A[:, 1]).mean()), 0.0, 0.0]))
    return a_, w


def predit_baseline(a_, w, T, C):
    b = float(np.exp(w[0] + w[1] * 1000.0 / (T + 273.15) + w[2] * np.log(C)))
    n = np.arange(1, N_MAX + 1, dtype=float)
    return np.clip(a_ - b * n ** 0.8, 0.5, 119.9)


# ------------------------------------------------------------------ LOCO ---
def run(modele, n_ctx=32, seed=0):
    cellules = [c for c in load_real_cells() if c["source"] == "target"]
    lignes, trajs = [], {}

    for i, held in enumerate(cellules):
        train = [c for j, c in enumerate(cellules) if j != i]

        t_pfn = predit_pfn(modele, train, held["T"], held["C"], n_ctx, seed)

        f = fit_joint(train, p_free=False, seed=seed * 100 + i)
        w3 = carte_conditions(train, f["b"])
        t_v3 = predicteur(f, w3)(held["T"], held["C"])

        a_, wb = fit_baseline(train)
        t_bl = predit_baseline(a_, wb, held["T"], held["C"])

        trajs[held["cell_id"]] = dict(n=held["n"], y=held["y"], T=held["T"],
                                      C=held["C"], pfn=t_pfn, v3=t_v3,
                                      baseline=t_bl)
        ligne = dict(fold=held["cell_id"], T=held["T"], C=held["C"])
        for nom, traj in (("pfn", t_pfn), ("v3", t_v3), ("baseline", t_bl)):
            ev = evaluate_cell_scoped(traj, held["n"], held["y"], soh_floor=70.0)
            # selection sur le MAXIMUM des deux grilles, jamais la moyenne
            ligne[f"rmse_{nom}"] = max(ev.get("rmse_uniforme_cycle", np.nan),
                                       ev.get("rmse_uniforme_soh", np.nan))
            ligne[f"ape_{nom}"] = ev.get("eol_ape", np.nan)
            ligne[f"n70_predit_{nom}"] = ev.get("n70_pred", np.nan)
        ligne["n70_vrai"] = ev.get("n70_true", np.nan)
        ligne["n70_observe"] = bool(ev.get("eol_observe", False))
        lignes.append(ligne)

    d = pd.DataFrame(lignes)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "f2_loco_reel.csv", index=False)
    return d, trajs


def resume(d):
    """Les 4 APE individuelles, jamais la seule mediane."""
    obs = d[d.n70_observe]
    out = {}
    for nom in ("pfn", "v3", "baseline"):
        out[nom] = dict(
            ape_par_fold={r.fold: (round(r[f"ape_{nom}"], 2)
                                   if np.isfinite(r[f"ape_{nom}"]) else None)
                          for _, r in obs.iterrows()},
            ape_mediane=round(float(obs[f"ape_{nom}"].median()), 2),
            ape_pire=round(float(obs[f"ape_{nom}"].max()), 2),
            rmse_mediane=round(float(d[f"rmse_{nom}"].median()), 2),
            rmse_pire=round(float(d[f"rmse_{nom}"].max()), 2),
            folds_gagnes_ape=int(sum(
                obs[f"ape_{nom}"] <= obs[[f"ape_{m}" for m in
                                          ("pfn", "v3", "baseline")]].min(axis=1)
                + 1e-9)))
    return out
