# -*- coding: utf-8 -*-
"""Sensibilite de la prediction au coin vide 55 degC / 0.5C a la valeur de SOH_k.

SOH_k est estime conjointement sur les cellules d'entrainement, mais les
valeurs par cellule s'etalent de 78.0 a 85.0 (echelle corrigee), et la plus
atypique est celle de 55 degC - precisement la temperature du coin sans
donnee. On mesure donc ce que devient n@70 a ce coin quand SOH_k est FIGE a
differentes valeurs et que tout le reste est reajuste.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

FADE_P = 0.8
N_MAX = 12000


def _nk(s0, b, soh_k):
    return float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / FADE_P))


def _traj(n, s0, b, soh_k, g, tau_frac):
    n = np.asarray(n, float)
    y = s0 - b * np.power(n, FADE_P)
    if g > 0:
        nk = _nk(s0, b, soh_k)
        tau = max(tau_frac * nk, 1.0)
        z = np.clip((n - nk) / tau, -50.0, 50.0)
        y = y - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    return np.clip(y, 0.5, 119.9)


def fit_with_fixed_sohk(cells, soh_k, seed=20260101):
    """(g, tau_frac) communs + un b par cellule, SOH_k FIGE."""
    m = len(cells)
    b0 = [max(c["s0"] - c["y"].min(), 1.0) / c["n"].max() ** FADE_P for c in cells]
    p0 = np.array([0.005, 0.25] + b0, float)
    lo = np.array([0.0, 0.02] + [1e-7] * m, float)
    hi = np.array([1.0, 5.0] + [5.0] * m, float)

    def resid(p):
        out = []
        for i, c in enumerate(cells):
            pred = _traj(c["n"], c["s0"], p[2 + i], soh_k, p[0], p[1])
            out.append((pred - c["y"]) / np.sqrt(len(c["n"])))
        return np.concatenate(out)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(6):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 2.0, len(p0)), lo, hi)
        try:
            r = least_squares(resid, st, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    b_cell = np.asarray(p[2:], float)
    Tk = np.array([c["T"] for c in cells]) + 273.15
    X = np.column_stack([np.ones(m), 1000.0 / Tk])
    w, *_ = np.linalg.lstsq(X, np.log(b_cell), rcond=None)
    s0 = float(np.mean([c["s0"] for c in cells]))
    rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    return dict(soh_k=soh_k, g=float(p[0]), tau_frac=float(p[1]),
                w=[float(w[0]), float(w[1])], s0=s0, rmse=rmse)


def n_at(traj, thr=70.0):
    b = np.where(np.asarray(traj, float) < thr)[0]
    if not b.size or b[0] == 0:
        return np.nan
    j = int(b[0])
    return float(np.interp(thr, [traj[j], traj[j - 1]], [j + 1, j]))


def predict(fit, T, C=None):
    b = float(np.exp(fit["w"][0] + fit["w"][1] * 1000.0 / (T + 273.15)))
    n = np.arange(1, N_MAX + 1, dtype=float)
    return _traj(n, fit["s0"], b, fit["soh_k"], fit["g"], fit["tau_frac"])


def sweep(cells, grille=(76.0, 77.8, 79.0, 80.4, 82.0, 83.0, 85.0)):
    rows = []
    for sk in grille:
        f = fit_with_fixed_sohk(cells, sk)
        r = dict(SOH_k=sk, g=f["g"], tau_frac=f["tau_frac"], rmse_ajust=f["rmse"])
        for T, lab in ((55, "n70_55C"), (45, "n70_45C"), (25, "n70_25C")):
            r[lab] = n_at(predict(f, T))
        rows.append(r)
    return pd.DataFrame(rows)
