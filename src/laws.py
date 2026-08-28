# -*- coding: utf-8 -*-
"""Lois de degradation candidates et ajustement par cellule.

Convention : SOH en POURCENT (convention nominale du scoring), n en cycles.
Toutes les lois sont ecrites `s0 - (termes de perte)`, tous les termes de perte
etant contraints >= 0 pour garantir la decroissance.
"""
import numpy as np
from scipy.optimize import curve_fit

FADE_P_DEFAULT = 0.8  # exposant du baseline officiel


def _relu_expm1(n, nk, tau, cap=50.0):
    """(exp((n-nk)/tau) - 1)_+ , sature pour eviter les debordements."""
    z = np.clip((np.asarray(n, float) - nk) / max(tau, 1e-6), -cap, cap)
    return np.where(np.asarray(n, float) > nk, np.expm1(z), 0.0)


# ---------------------------------------------------------------- lois -----
def L0_baseline(n, s0, b):
    """Baseline officiel : s0 - b*n^0.8 (exposant fige)."""
    return s0 - b * np.power(np.asarray(n, float), FADE_P_DEFAULT)


def L1_power(n, s0, a, p):
    """Puissance a exposant libre."""
    return s0 - a * np.power(np.asarray(n, float), p)


def L2_biexp(n, s0, a1, k1, a2, k2):
    """Double exponentielle (saturation + acceleration)."""
    n = np.asarray(n, float)
    return s0 - a1 * (-np.expm1(-np.clip(k1 * n, 0, 50))) \
              - a2 * np.expm1(np.clip(k2 * n, -50, 50))


def L3_user(n, s0, a, b, c, nk, tau):
    """Loi proposee : sqrt + lineaire + knee exponentiel."""
    n = np.asarray(n, float)
    return s0 - a * np.sqrt(n) - b * n - c * _relu_expm1(n, nk, tau)


def L4_user_freep(n, s0, a, p, c, nk, tau):
    """L3 avec exposant libre en remplacement du couple sqrt/lineaire.
    PAS de terme calendaire : exposant libre et sqrt(t) ne coexistent jamais."""
    n = np.asarray(n, float)
    return s0 - a * np.power(n, p) - c * _relu_expm1(n, nk, tau)


def L5_calendar(n, s0, a, k_cal, c, nk, tau, *, sqrt_t, p=FADE_P_DEFAULT):
    """L4 a exposant FIXE, plus un terme calendaire en sqrt(t).

    `sqrt_t` est precalcule (tableau de meme longueur que n) : c'est la racine
    du temps physique ecoule, qui depend de C-rate et de la trajectoire.
    """
    n = np.asarray(n, float)
    return s0 - a * np.power(n, p) - k_cal * np.asarray(sqrt_t, float) \
             - c * _relu_expm1(n, nk, tau)


# --------------------------------------------------- bornes et departs -----
def bounds_and_p0(law, n, y):
    """Bornes physiques et point de depart, mis a l'echelle de la cellule."""
    nm = float(np.max(n))
    s0g = float(np.median(y[:max(3, len(y) // 50)]))
    drop = max(float(y[0] - y.min()), 1.0)
    if law is L0_baseline:
        return ([80, 0], [120, 50]), [s0g, drop / nm ** FADE_P_DEFAULT]
    if law is L1_power:
        return ([80, 0, 0.2], [120, 50, 3.0]), [s0g, drop / nm ** 0.8, 0.8]
    if law is L2_biexp:
        return (([80, 0, 1e-7, 0, 1e-7], [120, 60, 1e-1, 60, 1e-2]),
                [s0g, drop * 0.7, 3.0 / nm, drop * 0.05, 1.0 / nm])
    if law is L3_user:
        return (([80, 0, 0, 0, 0.05 * nm, 1.0],
                 [120, 10, 1, 50, 3.0 * nm, 20 * nm]),
                [s0g, drop / np.sqrt(nm), 1e-4, 1e-3, 0.75 * nm, 0.25 * nm])
    if law is L4_user_freep:
        return (([80, 0, 0.2, 0, 0.05 * nm, 1.0],
                 [120, 50, 3.0, 50, 3.0 * nm, 20 * nm]),
                [s0g, drop / nm ** 0.8, 0.8, 1e-3, 0.75 * nm, 0.25 * nm])
    raise ValueError(f"loi inconnue : {law}")


def fit_law(law, n, y, p0=None, bounds=None, n_restarts=4, seed=0):
    """Moindres carres non lineaires avec redemarrages multiples.

    Renvoie (params, info). `info` porte le diagnostic d'identifiabilite :
    conditionnement du jacobien et matrice de correlation des parametres.
    """
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    if bounds is None or p0 is None:
        bounds, p0 = bounds_and_p0(law, n, y)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    rng = np.random.default_rng(seed)

    best = None
    for k in range(n_restarts):
        start = np.asarray(p0, float) if k == 0 else np.clip(
            np.asarray(p0, float) * rng.uniform(0.4, 2.5, len(p0)), lo, hi)
        try:
            pp, cov = curve_fit(law, n, y, p0=start, bounds=(lo, hi),
                                maxfev=100000)
        except Exception:
            continue
        res = y - law(n, *pp)
        sse = float(res @ res)
        if best is None or sse < best[0]:
            best = (sse, pp, cov)
    if best is None:
        return None, dict(ok=False)

    sse, pp, cov = best
    dof = max(len(n) - len(pp), 1)
    rmse = float(np.sqrt(sse / len(n)))
    # diagnostic d'identifiabilite
    with np.errstate(all="ignore"):
        sd = np.sqrt(np.abs(np.diag(cov)))
        corr = cov / np.outer(sd, sd) if np.all(sd > 0) else np.full_like(cov, np.nan)
        cond = float(np.linalg.cond(cov)) if np.all(np.isfinite(cov)) else np.inf
    off = corr[~np.eye(len(pp), dtype=bool)] if corr.size else np.array([np.nan])
    return pp, dict(ok=True, rmse=rmse, sse=sse, dof=dof, cov=cov, corr=corr,
                    cond_cov=cond, sd=sd,
                    corr_abs_max=float(np.nanmax(np.abs(off))) if off.size else np.nan,
                    aic=float(len(n) * np.log(sse / len(n)) + 2 * len(pp)))


# ==========================================================================
#  L5 - loi a terme calendaire, et ajustement sur trajectoire tronquee.
# ==========================================================================
from .protocol import A_DUR_H_C, B_DUR_H  # noqa: E402


def elapsed_hours(soh_pct_traj, c_rate, integral=True, a=A_DUR_H_C, b=B_DUR_H):
    """Temps physique ecoule (h) apres chaque cycle 1..N.

    `integral=True`  : duree(n) = (a/C)*SOH(n) + b, forme mesuree.
    `integral=False` : duree constante, forme lineaire de reference (ablation).
    """
    soh = np.asarray(soh_pct_traj, float) / 100.0
    if integral:
        return np.cumsum(a / float(c_rate) * soh + b)
    return np.arange(1, len(soh) + 1, dtype=float) * (1.9017 / float(c_rate)
                                                      + 0.939 + 0.5)


def L5_trajectory(n_max, s0, a, k_cal, c, nk, tau, c_rate,
                  p=FADE_P_DEFAULT, integral=True, iters=5):
    """Trajectoire L5 sur les cycles 1..n_max.

    Le terme calendaire depend du temps ecoule, qui depend lui-meme de la
    trajectoire (la duree de cycle decroit avec le SOH). On resout par
    iteration de point fixe : cinq passes suffisent, l'ecart entre la
    quatrieme et la cinquieme etant inferieur au millieme de point de SOH.
    """
    n = np.arange(1, int(n_max) + 1, dtype=float)
    cyc = a * np.power(n, p) if False else None  # garde-fou lisibilite
    base_cycle = a * np.power(n, p)
    knee = c * _relu_expm1(n, nk, tau)
    soh = np.full(len(n), s0, float)
    for _ in range(int(iters)):
        t = elapsed_hours(soh, c_rate, integral=integral)
        soh = np.clip(s0 - base_cycle - k_cal * np.sqrt(t) - knee, 20.0, 120.0)
    return soh


def fit_L5(n, y, c_rate, p=FADE_P_DEFAULT, integral=True, seed=0, n_restarts=5):
    """Ajuste L5 sur (n, y). L'exposant p est FIXE : sqrt(t) et n^p sont
    colineaires des que p approche 0.5, les laisser libres tous les deux rend
    le probleme non identifiable."""
    from scipy.optimize import least_squares
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    nm = float(n.max())
    idx = (n.astype(int) - 1)
    s0g = float(np.median(y[n <= 10])) if (n <= 10).any() else float(y[0])
    drop = max(float(s0g - y.min()), 1.0)

    lo = np.array([95.0, 0.0, 0.0, 0.0, 0.05 * nm, 1.0])
    hi = np.array([110.0, 5.0, 5.0, 200.0, 4.0 * nm, 30 * nm])
    p0 = np.array([s0g, 0.5 * drop / nm ** p, 0.02, 1.0, 0.8 * nm, 0.25 * nm])

    def resid(q):
        traj = L5_trajectory(int(nm), q[0], q[1], q[2], q[3], q[4], q[5],
                             c_rate, p=p, integral=integral)
        return traj[idx] - y

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.4, 2.5, 6), lo, hi)
        try:
            r = least_squares(resid, st, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, dict(ok=False)
    q = best.x
    res = best.fun
    return q, dict(ok=True, rmse=float(np.sqrt(np.mean(res ** 2))),
                   integral=integral, p=p)


def truncate_at(n, y, soh_trunc):
    """Restreint (n, y) aux cycles precedant le franchissement de `soh_trunc`.

    C'est le geste central de l'etape 1 : on n'ajuste que sur ce qu'on aurait
    observe si l'essai s'etait arrete a ce niveau de SOH.
    """
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    env = np.minimum.accumulate(y)
    below = np.where(env < soh_trunc)[0]
    stop = below[0] if len(below) else len(n)
    return n[:stop], y[:stop], (float(n[stop - 1]) if stop else np.nan)


def predict_law(law, params, n_max, c_rate=None, **kw):
    """Trajectoire 1..n_max pour n'importe laquelle des lois candidates."""
    if law is L5_trajectory:
        return L5_trajectory(n_max, *params, c_rate, **kw)
    n = np.arange(1, int(n_max) + 1, dtype=float)
    return law(n, *params)


CANDIDATES = {
    "L0 baseline s0-b*n^0.8": L0_baseline,
    "L1 puissance p libre": L1_power,
    "L2 double exponentielle": L2_biexp,
    "L3 sqrt+lineaire+knee": L3_user,
    "L4 puissance p libre+knee": L4_user_freep,
    "L5 calendaire (p fixe)": L5_trajectory,
}


def fit_all_laws(n, y, c_rate, n_max, integral=True, seed=0):
    """Ajuste toutes les lois candidates sur (n, y) et renvoie
    {label: (trajectoire 1..n_max, info)}."""
    out = {}
    for lab, law in CANDIDATES.items():
        if law is L5_trajectory:
            q, info = fit_L5(n, y, c_rate, integral=integral, seed=seed)
            traj = (L5_trajectory(n_max, *q, c_rate, integral=integral)
                    if q is not None else None)
        else:
            q, info = fit_law(law, n, y, seed=seed)
            traj = (law(np.arange(1, int(n_max) + 1, dtype=float), *q)
                    if q is not None else None)
        if traj is None:
            continue
        out[lab] = (np.clip(traj, 0.5, 119.9), dict(info, params=q))
    return out


def fit_anchor_single(n, y, soh_k=80.4, seed=0, knee_bounds=None,
                      knee_init=None):
    """Loi du modele d'ancrage ajustee sur UNE cellule.

    `s0` est mesure sur les premiers cycles et `SOH_k` est fixe a la valeur
    estimee conjointement sur les 6 cellules : sur une trajectoire tronquee a
    95 % il ne reste aucune information pour les estimer, et les laisser libres
    rend l'extrapolation arbitraire.
    """
    from scipy.optimize import least_squares
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    s0 = float(np.median(y[n <= 10])) if (n <= 10).any() else float(y[0])
    nm = float(n.max())
    c0, beta0 = knee_init if knee_init else (1.0, 4.0)
    (c_lo, c_hi), (b_lo, b_hi) = (knee_bounds if knee_bounds
                                  else ((0.0, 200.0), (0.2, 60.0)))
    p0 = np.array([max(s0 - y.min(), 1.0) / nm ** FADE_P_DEFAULT, c0, beta0])
    lo = np.array([1e-6, c_lo, b_lo])
    hi = np.array([5.0, c_hi, b_hi])

    def traj_of(q, nn):
        base = s0 - q[0] * np.power(nn, FADE_P_DEFAULT)
        nk = float(np.power(max((s0 - soh_k) / max(q[0], 1e-12), 1e-9),
                            1.0 / FADE_P_DEFAULT))
        tau = nk / max(q[2], 1e-3)
        return base - _relu_expm1(nn, nk, tau) * q[1]

    rng = np.random.default_rng(seed)
    best = None
    for k in range(5):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.4, 2.5, 3), lo, hi)
        try:
            r = least_squares(lambda q: traj_of(q, n) - y, st, bounds=(lo, hi),
                              max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, None
    return best.x, (lambda n_max: np.clip(
        traj_of(best.x, np.arange(1, int(n_max) + 1, dtype=float)), 0.5, 119.9))
