# -*- coding: utf-8 -*-
"""T1 - modele d'ancrage : baseline officiel refite + terme de knee ancre en SOH.

Forme :
    base(n)  = s0 - b * n^0.8                         (baseline officiel)
    n_k      = cycle ou base(n) franchit SOH_k        (ancrage en SOH, pas en n)
    SOH(n)   = base(n) - c * (exp((n - n_k) / (n_k/beta)) - 1)_+

Pourquoi ancrer le knee en SOH et non en nombre de cycles : ajuste librement,
`n_k` se disperse d'un facteur 6 entre cellules et sa carte (T, C) donne
R2 = 0.31 ; le NIVEAU de SOH auquel le knee demarre, lui, tient dans
78-85 % sur les 6 cellules, sans dependance visible a T ni a C. C'est le
parametre identifiable, donc c'est celui qu'on estime.

`SOH_k`, `c` et `beta` sont ajustes CONJOINTEMENT sur toutes les cellules
(un jeu commun), et seul `b` est propre a chaque cellule : mettre en commun un
ajustement conjoint est bien plus stable que moyenner des estimations
individuelles instables. `b` est ensuite envoye sur (T, C) par
ln(b) = w0 + w1*1000/T_K + w2*ln(C), comme le baseline officiel.

`s0` est FIXE a la valeur mesuree (mediane des cycles <= 10), pas ajuste :
l'etendue de s0 sur les 6 cellules est de 0.84 point de SOH, sans dependance a
T (p = 0.16), et le laisser libre lui fait absorber une partie du knee.
"""
import numpy as np
from scipy.optimize import least_squares

FADE_P = 0.8
N_MAX_PREDICT = 12000


def _knee_loss(n, nk, tau, c):
    z = np.clip((n - nk) / max(tau, 1e-6), -50.0, 50.0)
    return np.where(n > nk, c * np.expm1(z), 0.0)


def _nk_from_sohk(s0, b, soh_k):
    """Cycle ou la loi de base franchit SOH_k."""
    d = (s0 - soh_k) / max(b, 1e-12)
    return float(np.power(max(d, 1e-9), 1.0 / FADE_P))


def trajectory(s0, b, soh_k, c, beta, n=None):
    """Trajectoire SOH% pour les cycles 1..N."""
    n = np.arange(1, N_MAX_PREDICT + 1, dtype=float) if n is None else np.asarray(n, float)
    base = s0 - b * np.power(n, FADE_P)
    nk = _nk_from_sohk(s0, b, soh_k)
    tau = nk / max(beta, 1e-3)
    soh = base - _knee_loss(n, nk, tau, c)
    # bornes exigees par framework/io.py : valeurs finies dans (0, 120]
    return np.clip(soh, 0.5, 119.9)


def fit_joint(cells, seed=0):
    """Ajustement conjoint : (SOH_k, c, beta) communs, un b par cellule.

    cells : liste de dicts {cell_id, T, C, n, y, s0}
    """
    m = len(cells)
    b0 = []
    for cl in cells:
        drop = max(float(cl["s0"] - cl["y"].min()), 1.0)
        b0.append(drop / float(cl["n"].max()) ** FADE_P)
    p0 = np.array([82.0, 1.0, 4.0] + b0, float)
    lo = np.array([60.0, 0.0, 0.2] + [1e-6] * m, float)
    hi = np.array([98.0, 200.0, 60.0] + [5.0] * m, float)

    def resid(p):
        soh_k, c, beta = p[0], p[1], p[2]
        out = []
        for i, cl in enumerate(cells):
            pred = trajectory(cl["s0"], p[3 + i], soh_k, c, beta, cl["n"])
            # ponderation : chaque cellule pese autant, quel que soit son nb de cycles
            out.append((pred - cl["y"]) / np.sqrt(len(cl["n"])))
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
    if best is None:
        raise RuntimeError("ajustement conjoint echoue")

    p = best.x
    return dict(soh_k=float(p[0]), c=float(p[1]), beta=float(p[2]),
                b=[float(v) for v in p[3:]],
                cost=float(best.cost), jac=best.jac, x=p)


def fit_condition_map(cells, b_list, free_c_exponent=False):
    """ln(b) = w0 + w1*1000/T_K + w2*ln(C).

    `free_c_exponent=False` impose w2 = 0. Justification empirique : a nombre de
    cycles egal, les paires 0.5C / 1C a 25 et 45 degC different de moins de
    1.5 point de SOH, et l'ecart CHANGE DE SIGNE au cours de la vie. La pente
    en ln(C) n'est donc pas identifiable sur 2 contrastes, et la laisser libre
    produit ici un w2 negatif (= vieillir plus vite a 0.5C qu'a 1C, par cycle),
    qui s'extrapole en aberration au coin vide 55 degC / 0.5C.
    """
    Tk = np.array([c["T"] for c in cells], float) + 273.15
    C = np.array([c["C"] for c in cells], float)
    cols = [np.ones(len(cells)), 1000.0 / Tk]
    if free_c_exponent:
        cols.append(np.log(C))
    X = np.column_stack(cols)
    y = np.log(np.asarray(b_list, float))
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    r2 = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    w = list(w) + ([0.0] if not free_c_exponent else [])
    return w, r2, float(np.sqrt(np.mean((np.exp(pred) - np.exp(y)) ** 2)))


def b_at(w, T_degC, c_rate):
    return float(np.exp(w[0] + w[1] * 1000.0 / (float(T_degC) + 273.15)
                        + w[2] * np.log(float(c_rate))))
