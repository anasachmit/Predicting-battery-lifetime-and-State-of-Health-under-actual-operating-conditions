# -*- coding: utf-8 -*-
"""F.4 - audit de plausibilite physique du PFN. ELIMINATOIRE.

Un reseau n'a AUCUNE garantie physique. La monotonie qu'affiche `f2_reel` vient
d'un `minimum.accumulate` applique apres coup : c'est un pansement, pas une
propriete du modele. Cet audit mesure donc ce que le reseau produit REELLEMENT,
sans correction, en plus des criteres appliques a v3.

Criteres, identiques a ceux de v3 (`src/plausibility.py`) sauf mention :

  1. trajectoires finies, SOH dans (0, 103]
  2. monotonie de SOH(n) -- rapportee DEUX FOIS : taux de violation BRUT du
     reseau (diagnostic a part entiere) et apres enveloppe cumulative
  3. n@70 strictement decroissant en temperature
  4. rapport n@70(0.5C) / n@70(1C) dans [0.80, 1.25]
  5. aucun knee avant 30 cycles
  6. comportement au COIN VIDE 55 degC / 0.5C, ou aucune cellule n'existe :
     trajectoire complete, n@70, extrapolation profonde comparee a v3
  7. largeur de l'intervalle q10-q90 au coin vide, convertie en CYCLES et
     comparee a notre analyse de sensibilite manuelle (734 cycles pour w1).
     Un intervalle nettement plus etroit signalerait un reseau SUR-CONFIANT -
     c'est le defaut le plus dangereux d'un modele appris, parce qu'il ne se
     voit pas sur les metriques d'erreur ponctuelle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import RESULTS
from ..plausibility import sweep, audit, T_GRID, C_GRID
from .data import normalise, denormalise_soh
from .evaluate import GRILLE_EVAL
from .real_theta import load_real_cells
from .l4_retest import fit_joint, carte_conditions, predicteur

OUT = RESULTS / "pfn"
N_MAX = 12000

# Reference : sensibilite manuelle de v3 a la pente d'Arrhenius w1.
SENSIBILITE_W1_CYCLES = 734.0

# v3 au coin vide 55 degC / 0.5C, mesure precedemment.
V3_COIN_VIDE = {2000: 82.1, 4000: 50.6, 6000: 14.4, 8000: 0.5}
V3_COIN_VIDE_N70 = 2876.0


@torch.no_grad()
def _contexte_reel(modele, n_ctx=32, device="cpu"):
    """Les SIX cellules reelles comme contexte - le regime de deploiement."""
    cellules = [c for c in load_real_cells() if c["source"] == "target"]
    ctx = []
    for c in cellules:
        n, y = c["n"], c["y"]
        if len(n) > n_ctx:
            idx = np.unique(np.round(np.geomspace(1, len(n), n_ctx)).astype(int) - 1)
            idx = np.clip(idx, 0, len(n) - 1)
            idx[-1] = len(n) - 1
            n, y = n[idx], y[idx]
        ctx.append(normalise(np.full(len(n), c["T"]), np.full(len(n), c["C"]),
                             n, y))
    return torch.tensor(np.concatenate(ctx)[None], dtype=torch.float32,
                        device=device)


@torch.no_grad()
def quantiles_pfn(modele, Xc, T, C, device="cpu"):
    """Tous les quantiles sur GRILLE_EVAL, en points de SOH."""
    Xq = torch.tensor(
        normalise(np.full(len(GRILLE_EVAL), T), np.full(len(GRILLE_EVAL), C),
                  GRILLE_EVAL)[None], dtype=torch.float32, device=device)
    q = modele(Xc, Xq)[0].cpu().numpy()
    return denormalise_soh(q)          # (len(GRILLE_EVAL), n_quantiles)


def _plein(soh_grille, monotone):
    """Interpole sur 1..N_MAX ; `monotone` applique l'enveloppe cumulative."""
    n = np.arange(1, N_MAX + 1, dtype=float)
    t = np.interp(n, GRILLE_EVAL, soh_grille)
    if monotone:
        t = np.minimum.accumulate(t)
    return np.clip(t, 0.5, 119.9)


def predicteur_pfn(modele, Xc, monotone=True, i_q=None):
    i = modele.quantiles.index(0.50) if i_q is None else i_q

    def f(T, C):
        return _plein(quantiles_pfn(modele, Xc, T, C)[:, i], monotone)
    return f


def _n_at(traj, seuil=70.0):
    below = np.flatnonzero(np.asarray(traj, float) < seuil)
    if not below.size or below[0] == 0:
        return np.nan
    j = int(below[0])
    return float(np.interp(seuil, [traj[j], traj[j - 1]], [j + 1, j]))


def run(modele, n_ctx=32):
    Xc = _contexte_reel(modele, n_ctx)
    res = {}

    # --- 2. monotonie BRUTE, sans enveloppe --------------------------------
    sw_brut = sweep(predicteur_pfn(modele, Xc, monotone=False))
    sw_env = sweep(predicteur_pfn(modele, Xc, monotone=True))
    res["monotonie_brute"] = dict(
        taux_violation_median=float(sw_brut.taux_violation_monotonie.median()),
        taux_violation_max=float(sw_brut.taux_violation_monotonie.max()),
        conditions_sans_violation=int(
            (sw_brut.taux_violation_monotonie == 0).sum()),
        conditions_total=int(len(sw_brut)))

    # --- 1, 3, 4, 5 : criteres de v3, sur la trajectoire deployee ---------
    checks, piv = audit(sw_env)
    res["audit"] = checks
    res["n70_par_condition"] = piv

    # --- 6. coin vide 55 degC / 0.5C --------------------------------------
    q_coin = quantiles_pfn(modele, Xc, 55.0, 0.5)
    i_med = modele.quantiles.index(0.50)
    traj_coin = _plein(q_coin[:, i_med], monotone=True)
    res["coin_vide"] = dict(
        n70_pfn=_n_at(traj_coin), n70_v3=V3_COIN_VIDE_N70,
        profond={c: dict(pfn=round(float(traj_coin[c - 1]), 1),
                         v3=V3_COIN_VIDE[c]) for c in V3_COIN_VIDE})

    # --- 7. largeur q10-q90 au coin vide, convertie en cycles --------------
    i_lo = modele.quantiles.index(0.10)
    i_hi = modele.quantiles.index(0.90)
    n70_lo = _n_at(_plein(q_coin[:, i_lo], True))    # SOH bas -> franchit tot
    n70_hi = _n_at(_plein(q_coin[:, i_hi], True))    # SOH haut -> franchit tard
    largeur = (n70_hi - n70_lo) if np.isfinite(n70_hi) and np.isfinite(n70_lo) \
        else np.nan
    res["incertitude_coin_vide"] = dict(
        n70_q10=n70_lo, n70_q90=n70_hi, largeur_cycles=largeur,
        reference_sensibilite_w1=SENSIBILITE_W1_CYCLES,
        rapport=(round(largeur / SENSIBILITE_W1_CYCLES, 2)
                 if np.isfinite(largeur) else np.nan),
        # Un rapport nettement < 1 signale un reseau sur-confiant.
        surconfiant=bool(np.isfinite(largeur)
                         and largeur < 0.5 * SENSIBILITE_W1_CYCLES))

    sw_brut.to_csv(OUT / "f4_sweep_brut.csv", index=False)
    sw_env.to_csv(OUT / "f4_sweep_enveloppe.csv", index=False)
    return res, sw_brut, sw_env


def verdict(res):
    """F.4 est ELIMINATOIRE : tout critere faux arrete la discussion."""
    ok = all(res["audit"].values())
    return dict(audit_complet=ok,
                surconfiant=res["incertitude_coin_vide"]["surconfiant"],
                F4_passe=bool(ok and not res["incertitude_coin_vide"]["surconfiant"]))
