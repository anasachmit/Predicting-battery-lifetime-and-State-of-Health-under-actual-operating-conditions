# -*- coding: utf-8 -*-
"""T1 - ajustement du modele d'ancrage : baseline officiel + terme de knee.

Loi : SOH(n) = s0 - b*n^0.8 - c*(exp((n-nk)/tau) - 1)_+
Carte conditions -> parametres : ln(b) et ln(nk) lineaires en (1000/T_K, ln C),
comme le baseline officiel ; c et tau mis en commun (pooling) faute de contraste
suffisant sur 6 cellules.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .laws import FADE_P_DEFAULT, _relu_expm1


def law_knee(n, s0, b, c, nk, tau):
    n = np.asarray(n, float)
    return s0 - b * np.power(n, FADE_P_DEFAULT) - c * _relu_expm1(n, nk, tau)


def fit_cell(n, y, seed=0, n_restarts=6):
    nm, s0g = float(n.max()), float(np.median(y[:20]))
    drop = max(float(y[0] - y.min()), 1.0)
    lo = [95.0, 0.0, 0.0, 0.05 * nm, 1.0]
    hi = [110.0, 5.0, 60.0, 4.0 * nm, 30 * nm]
    p0 = [s0g, drop / nm ** FADE_P_DEFAULT, 1e-3, 0.8 * nm, 0.25 * nm]
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else list(np.clip(np.array(p0) * rng.uniform(0.3, 3.0, 5), lo, hi))
        try:
            pp, _ = curve_fit(law_knee, n, y, p0=st, bounds=(lo, hi), maxfev=100000)
        except Exception:
            continue
        sse = float(((y - law_knee(n, *pp)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, pp)
    if best is None:
        return None, np.inf
    return best[1], float(np.sqrt(best[0] / len(n)))


def fit_all(cells_soh):
    """cells_soh : dict cell_id -> (T, C, n, soh_percent)."""
    rows = []
    for cid, (T, C, n, y) in cells_soh.items():
        pp, rmse = fit_cell(np.asarray(n, float), np.asarray(y, float))
        if pp is None:
            continue
        s0, b, c, nk, tau = pp
        rows.append(dict(cell_id=cid, T_amb=T, C_rate=C, n_max=float(n.max()),
                         s0=s0, b=b, c=c, nk=nk, tau=tau, rmse=rmse,
                         knee_visible=bool(nk < 1.05 * n.max())))
    return pd.DataFrame(rows)


def build_maps(par):
    """Regressions (1000/T_K, ln C) -> ln b et ln nk ; c et tau mis en commun."""
    Tk = par["T_amb"].to_numpy(float) + 273.15
    X = np.column_stack([np.ones(len(par)), 1000.0 / Tk,
                         np.log(par["C_rate"].to_numpy(float))])
    out = {"s0": float(par["s0"].mean())}
    for k in ("b", "nk"):
        y = np.log(par[k].to_numpy(float))
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ w
        out[f"w_{k}"] = w.tolist()
        out[f"r2_{k}"] = float(1 - ((y - pred) ** 2).sum()
                               / max(((y - y.mean()) ** 2).sum(), 1e-12))
    vis = par[par["knee_visible"]]
    src = vis if len(vis) >= 2 else par
    out["c"] = float(np.median(src["c"]))
    out["tau"] = float(np.median(src["tau"]))
    out["n_knee_visible"] = int(len(vis))
    return out
