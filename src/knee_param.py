# -*- coding: utf-8 -*-
"""Reparametrage du knee par le rapport identifiable, et verification.

Constat a l'origine : dans la parametrisation (c, beta), `beta` tape sa borne
basse dans 5 folds LOCO sur 6 et `c` varie d'un facteur 7. Seul leur RAPPORT
est determine - c'est visible sur la forme du terme, puisque pour z petit

    c * (exp(z) - 1) ~ c * z = (c / tau) * (n - n_k)

et donc seule la pente initiale du knee, g = c / tau, agit tant que la
trajectoire ne s'enfonce pas loin sous n_k.

Nouvelle parametrisation : (g, tau) avec c = g * tau. La limite tau -> infini
donne une rampe lineaire g * (n - n_k), qui est un comportement propre, alors
que dans l'ancienne parametrisation la meme limite exigeait c -> infini.

`SOH_k` reste l'ancrage : c'est le parametre le plus stable du modele
(dispersion x1.02 entre folds), et c'est lui qui fixe n_k.
"""
import numpy as np
from scipy.optimize import least_squares

from .anchor_model import FADE_P, N_MAX_PREDICT, _nk_from_sohk

SOH_LO, SOH_HI = 0.5, 119.9


def _knee(n, nk, tau, g):
    """Terme de knee, parametre par sa pente initiale g = c/tau."""
    z = np.clip((n - nk) / max(tau, 1e-6), -50.0, 50.0)
    return np.where(n > nk, g * max(tau, 1e-6) * np.expm1(z), 0.0)


def trajectory_g(s0, b, soh_k, g, tau, n=None):
    n = (np.arange(1, N_MAX_PREDICT + 1, dtype=float) if n is None
         else np.asarray(n, float))
    base = s0 - b * np.power(n, FADE_P)
    nk = _nk_from_sohk(s0, b, soh_k)
    return np.clip(base - _knee(n, nk, tau, g), SOH_LO, SOH_HI)


def fit_joint_g(cells, seed=0, n_restarts=6, tau_frac_bounds=(0.02, 5.0)):
    """Ajustement conjoint : (SOH_k, g, tau_frac) communs, un b par cellule.

    `tau_frac` = tau / n_k, sans dimension, pour que la borne ait le meme sens
    quelle que soit la duree de vie de la cellule.
    """
    m = len(cells)
    nref = float(np.median([c["n"].max() for c in cells]))
    b0 = [max(c["s0"] - c["y"].min(), 1.0) / c["n"].max() ** FADE_P
          for c in cells]
    p0 = np.array([82.0, 0.005, 0.25] + b0, float)
    lo = np.array([60.0, 0.0, tau_frac_bounds[0]] + [1e-6] * m, float)
    hi = np.array([98.0, 1.0, tau_frac_bounds[1]] + [5.0] * m, float)

    def traj_for(p, i, nn):
        s0 = cells[i]["s0"]
        b = p[3 + i]
        nk = _nk_from_sohk(s0, b, p[0])
        return (s0 - b * np.power(nn, FADE_P)
                - _knee(nn, nk, max(p[2] * nk, 1.0), p[1]))

    def resid(p):
        out = []
        for i, cl in enumerate(cells):
            out.append((traj_for(p, i, cl["n"]) - cl["y"])
                       / np.sqrt(len(cl["n"])))
        return np.concatenate(out)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 2.0, len(p0)),
                                       lo, hi)
        try:
            r = least_squares(resid, st, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("ajustement conjoint (g, tau) echoue")
    p = best.x
    return dict(soh_k=float(p[0]), g=float(p[1]), tau_frac=float(p[2]),
                b=[float(v) for v in p[3:]], res=best,
                noms=["SOH_k", "g", "tau_frac"] + [f"b_{i}" for i in range(m)])


def param_correlation(res, noms=None):
    """Matrice de correlation des parametres, depuis le jacobien a l'optimum."""
    J = np.asarray(res.jac, float)
    n, p = J.shape
    try:
        JTJ_inv = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        return None, np.nan
    sigma2 = float(res.fun @ res.fun) / max(n - p, 1)
    cov = JTJ_inv * sigma2
    sd = np.sqrt(np.abs(np.diag(cov)))
    with np.errstate(all="ignore"):
        corr = cov / np.outer(sd, sd)
    off = corr[~np.eye(p, dtype=bool)]
    return corr, float(np.nanmax(np.abs(off)))


def knee_corr_pair(corr, i, j):
    """|correlation| entre les deux parametres de forme du knee."""
    if corr is None:
        return np.nan
    return float(abs(corr[i, j]))
