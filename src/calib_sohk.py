# -*- coding: utf-8 -*-
"""Reconcilier SOH_k (ajustement conjoint) et le SOH au knee (Bacon-Watts).

Les deux nombres ne mesurent pas la meme chose, et il faut le prouver plutot
que de l'affirmer :

  * `SOH_k` est le niveau ou le terme exponentiel du modele COMMENCE a agir.
    Par construction, sa contribution y est nulle et croit ensuite.
  * Bacon-Watts localise le centre du basculement entre deux droites, donc
    quelque part APRES le debut de l'effet - et donc a un SOH plus BAS.

Un decalage systematique est attendu. Le test consiste a le mesurer : on
genere des trajectoires avec NOTRE propre loi, `SOH_k` connu et balaye, bruit
et densite de points comparables aux cellules cible, puis on fait tourner
Bacon-Watts dessus. Si l'ecart est stable, les deux estimations sont
reconciliees par un facteur de calibration documente. S'il est erratique,
l'une des deux est fausse.
"""
import numpy as np
import pandas as pd

from .knee_detect import bacon_watts

FADE_P = 0.8


def synth(soh_k, b, s0=102.47, g=0.00896, tau_frac=5.0, n_max=5000,
          bruit=0.30, pas=1, seed=0, soh_stop=42.0):
    """Trajectoire engendree par la loi du modele v3, SOH_k connu."""
    rng = np.random.default_rng(seed)
    n = np.arange(1, int(n_max) + 1, dtype=float)
    nk = float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / FADE_P))
    tau = max(tau_frac * nk, 1.0)
    z = np.clip((n - nk) / tau, -50.0, 50.0)
    y = s0 - b * np.power(n, FADE_P) - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    keep = y >= soh_stop
    n, y = n[keep], y[keep]
    if pas > 1:
        n, y = n[::pas], y[::pas]
    return n, y + rng.normal(0.0, bruit, len(y)), nk


def sweep(soh_k_grid=np.arange(76.0, 84.01, 1.0),
          b_grid=(0.028, 0.038, 0.054), n_reps=4, bruit=0.30, seed=0):
    """Balayage : pour chaque SOH_k vrai, que renvoie Bacon-Watts ?"""
    rows = []
    for sk in soh_k_grid:
        for b in b_grid:
            for r in range(n_reps):
                n, y, nk = synth(sk, b, bruit=bruit, seed=seed + 1000 * r + int(sk))
                if len(n) < 200:
                    continue
                bw = bacon_watts(n, y)
                if bw is None:
                    continue
                rows.append(dict(soh_k_vrai=float(sk), b=b, rep=r,
                                 n_k_vrai=nk, x1_bw=bw["x1"],
                                 soh_bw=bw["soh_au_knee"],
                                 ecart=bw["soh_au_knee"] - float(sk),
                                 rapport_pentes=bw["rapport_pentes"],
                                 profondeur=float(y.max() - y.min()),
                                 n_points=len(n)))
    return pd.DataFrame(rows)


def calibration(df):
    """L'ecart est-il stable ? Regression soh_bw ~ soh_k_vrai."""
    d = df.dropna(subset=["soh_bw"])
    x = d["soh_k_vrai"].to_numpy(float)
    y = d["soh_bw"].to_numpy(float)
    A = np.column_stack([np.ones(len(x)), x])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ w
    r2 = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    e = d["ecart"].to_numpy(float)
    return dict(pente=float(w[1]), ordonnee=float(w[0]), r2=r2,
                ecart_median=float(np.median(e)), ecart_moyen=float(np.mean(e)),
                ecart_sd=float(np.std(e)),
                ecart_min=float(np.min(e)), ecart_max=float(np.max(e)),
                n=len(d))


def invert(soh_bw, calib):
    """Remonter de la mesure Bacon-Watts au SOH_k du modele."""
    return (float(soh_bw) - calib["ordonnee"]) / max(calib["pente"], 1e-9)
