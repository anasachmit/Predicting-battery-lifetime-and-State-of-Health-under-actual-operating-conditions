# -*- coding: utf-8 -*-
"""Prior sur la forme du knee, estime sur les 20 cellules Wheeler.

Pourquoi Wheeler et pas les cellules cible : 20/20 des cellules Wheeler
descendent sous 70 % de SOH et 18/20 sous 60 %, donc leurs knees sont
COMPLETS. Les cellules cible n'en ont que 4 sur 6, dont deux tres partiels.
Estimer la forme du knee sur les cibles puis s'en servir pour predire les
cibles serait circulaire ; l'estimer sur Wheeler ne l'est pas.

Ce qui est transferable, et ce qui ne l'est pas. Wheeler ce sont des 18650
1.1 Ah a 50 degC, les cibles des prismatiques 102 Ah de 25 a 55 degC : les
grandeurs qui portent une echelle (nombre de cycles, capacite) ne se
transferent pas. On ne retient donc que des grandeurs SANS dimension ou
exprimees en points de SOH :

    SOH_k     niveau de SOH ou le knee demarre        (points de SOH)
    tau_frac  tau / n_k                               (sans dimension)
    g_scaled  g * n_k, perte supplementaire par n_k   (points de SOH)

`b` et `n_k` restent propres a chaque cellule et ne sont jamais transferes.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .anchor_model import FADE_P, _nk_from_sohk
from .config import RESULTS


def _knee(n, nk, tau, g):
    z = np.clip((n - nk) / max(tau, 1e-6), -50.0, 50.0)
    return np.where(n > nk, g * max(tau, 1e-6) * np.expm1(z), 0.0)


def fit_cell_g(n, y, s0=None, seed=0, n_restarts=6):
    """Ajuste (b, SOH_k, g, tau_frac) sur une cellule a knee complet."""
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    s0 = float(np.median(y[n <= max(10, n.min())])) if s0 is None else float(s0)
    nm = float(n.max())
    p0 = np.array([max(s0 - y.min(), 1.0) / nm ** FADE_P, 82.0, 0.005, 0.25])
    lo = np.array([1e-7, 60.0, 0.0, 0.02])
    hi = np.array([5.0, 98.0, 1.0, 5.0])

    def traj(p, nn):
        nk = _nk_from_sohk(s0, p[0], p[1])
        return (s0 - p[0] * np.power(nn, FADE_P)
                - _knee(nn, nk, max(p[3] * nk, 1.0), p[2]))

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.4, 2.5, 4), lo, hi)
        try:
            r = least_squares(lambda p: traj(p, n) - y, st, bounds=(lo, hi),
                              max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    b, soh_k, g, tau_frac = best.x
    nk = _nk_from_sohk(s0, b, soh_k)
    return dict(b=float(b), soh_k=float(soh_k), g=float(g),
                tau_frac=float(tau_frac), n_k=float(nk),
                g_scaled=float(g * nk), s0=s0,
                rmse=float(np.sqrt(np.mean((traj(best.x, n) - y) ** 2))))


def fit_wheeler(unified, min_span=25.0):
    """Ajuste chaque cellule Wheeler et renvoie le tableau des parametres.

    `min_span` : on n'utilise que les cellules qui parcourent au moins 25
    points de SOH, sans quoi le knee n'est pas contraint.
    """
    w = unified[unified["source"] == "wheeler"]
    rows = []
    for cid, g in w.groupby("cell_id"):
        g = g.sort_values("cycle_n")
        n = g["cycle_n"].to_numpy(float)
        y = g["SOH"].to_numpy(float) * 100.0
        ok = np.isfinite(y) & (n > 0)
        n, y = n[ok], y[ok]
        if len(n) < 20 or (y.max() - y.min()) < min_span:
            continue
        r = fit_cell_g(n, y, s0=100.0)
        if r is None:
            continue
        r.update(cell_id=cid, n_points=len(n), soh_min=float(y.min()))
        rows.append(r)
    return pd.DataFrame(rows)


def build_prior(par_wheeler, save=True):
    """Prior robuste : mediane et ecart interquartile des grandeurs
    transferables. On prend la mediane plutot que la moyenne parce que trois
    cellules Wheeler ont un knee tres precoce qui tire la moyenne."""
    q = {}
    for k in ("soh_k", "tau_frac", "g_scaled"):
        v = par_wheeler[k].to_numpy(float)
        v = v[np.isfinite(v)]
        q[k] = dict(median=float(np.median(v)),
                    q25=float(np.percentile(v, 25)),
                    q75=float(np.percentile(v, 75)),
                    min=float(v.min()), max=float(v.max()),
                    dispersion=float(np.max(np.abs(v)) /
                                     max(np.min(np.abs(v)), 1e-12)))
    q["n_cellules"] = int(len(par_wheeler))
    if save:
        import json
        (RESULTS / "wheeler_knee_prior.json").write_text(
            json.dumps(q, indent=1), encoding="utf-8")
    return q
