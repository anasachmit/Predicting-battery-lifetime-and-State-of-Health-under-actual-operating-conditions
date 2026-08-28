# -*- coding: utf-8 -*-
"""Ajustement de L3 sur les 44 cellules reelles -> table des theta observes.

C'est la SEULE porte d'entree des donnees reelles dans la branche PFN. Le
reseau, lui, ne verra jamais ces trajectoires : elles ne servent qu'a poser les
bornes du prior (etape A.2) et a le tester (etape B). Un PFN dont le prior est
regle a l'oeil est decoratif ; un PFN dont le prior est regle sur des theta
mesures est une hypothese testable.

Loi L3 (identique a celle du modele soumis v3, cf. my_model/model_template.py) :

    base(n) = s0 - b * n^0.8
    n_k     = ((s0 - SOH_k) / b)^(1/0.8)
    tau     = tau_frac * n_k
    SOH(n)  = base(n) - g * tau * (exp((n - n_k)/tau) - 1)_+

Strategie d'ajustement, par source, dictee par ce que chaque source identifie :

* cible (6 cellules) - ajustement CONJOINT : (SOH_k, g, tau_frac) communs, un b
  par cellule. C'est le protocole du modele soumis, et sur 6 cellules dont 4
  seulement franchissent 80 %, un knee par cellule serait du bruit.
* SNL (18 cellules) - ajustement conjoint identique. Aucune ne descend sous
  69 %, donc le knee y est faible : ces cellules renseignent la plage de `b`,
  pas celle du knee.
* Wheeler (20 cellules) - ajustement INDIVIDUEL : 20/20 descendent sous 70 %,
  chacune porte un knee complet. C'est la seule source qui donne une VRAIE
  dispersion inter-cellules de (SOH_k, g, tau_frac).

Convention SOH : `SOH_nom` (capacite / capacite nominale), en pourcent - la
convention du scoring officiel. Elle porte le decalage de formation propre a
chaque source (cible ~102.5 %, SNL ~93-97 %, Wheeler ~91 %) ; c'est voulu, et
c'est traite explicitement au moment du test de couverture.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..config import DATA, RESULTS

FADE_P = 0.8
OUT_CSV = RESULTS / "pfn" / "theta_real.csv"

# Profondeur en dessous de laquelle une cellule porte assez de signal pour
# qu'un knee individuel ait un sens. 85 % = SOH_k median + ~5 points.
DEEP_SOH = 85.0


# ------------------------------------------------------------------- loi ---
def nk_from_sohk(s0, b, soh_k, p=FADE_P):
    return float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / p))


def l3(n, s0, b, soh_k=None, g=0.0, tau_frac=1.0, p=FADE_P):
    """Trajectoire L3. `g = 0` desactive le knee (forme de base seule).

    `p` reste a 0.8 par defaut : ce module ajuste les cellules REELLES avec le
    meme exposant que le modele soumis, pour que les theta produits soient
    comparables a ceux du banc. Le prior, lui, tire p dans [0.40, 1.00].
    """
    n = np.asarray(n, float)
    soh = s0 - b * np.power(n, p)
    if soh_k is not None and g > 0.0:
        nk = nk_from_sohk(s0, b, soh_k, p)
        tau = max(tau_frac * nk, 1.0)
        z = np.clip((n - nk) / tau, -50.0, 50.0)
        soh = soh - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    return soh


# -------------------------------------------------------------- lecture ---
def load_real_cells(parquet=None):
    """Une entree par cellule : (cell_id, source, T, C, n, y, s0)."""
    d = pd.read_parquet(parquet or (DATA / "unified.parquet"))
    d = d.dropna(subset=["SOH_nom", "cycle_n"]).copy()
    d["y"] = d["SOH_nom"] * 100.0
    out = []
    for cid, g in d.groupby("cell_id"):
        g = g.sort_values("cycle_n")
        n = g["cycle_n"].to_numpy(float)
        y = g["y"].to_numpy(float)
        if len(n) < 15:
            continue
        early = y[n <= 10]
        s0 = float(np.median(early)) if early.size else float(y[0])
        out.append(dict(cell_id=cid, source=str(g["source"].iloc[0]),
                        T=float(g["T_amb"].iloc[0]),
                        C=float(g["C_rate_dchg"].iloc[0]),
                        n=n, y=y, s0=s0,
                        soh_min=float(np.minimum.accumulate(y).min())))
    return sorted(out, key=lambda c: (c["source"], c["T"], c["C"], c["cell_id"]))


# ------------------------------------------------------------ ajustements ---
def _best(fun, p0, lo, hi, seed=0, n_restarts=8):
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 2.0, len(p0)), lo, hi)
        try:
            r = least_squares(fun, st, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def fit_joint(cells, seed=0):
    """(SOH_k, g, tau_frac) communs + un b par cellule."""
    m = len(cells)
    b0 = [max(c["s0"] - c["y"].min(), 1.0) / c["n"].max() ** FADE_P for c in cells]
    p0 = np.array([82.0, 0.005, 0.25] + b0, float)
    lo = np.array([60.0, 0.0, 0.02] + [1e-7] * m, float)
    hi = np.array([98.0, 1.0, 5.0] + [5.0] * m, float)

    def resid(p):
        return np.concatenate([
            (l3(c["n"], c["s0"], p[3 + i], p[0], p[1], p[2]) - c["y"])
            / np.sqrt(len(c["n"])) for i, c in enumerate(cells)])

    r = _best(resid, p0, lo, hi, seed=seed)
    if r is None:
        raise RuntimeError("ajustement conjoint echoue")
    p = r.x
    return dict(soh_k=float(p[0]), g=float(p[1]), tau_frac=float(p[2]),
                b=[float(v) for v in p[3:]],
                rmse=float(np.sqrt(2 * r.cost / m)))


def fit_single(cell, seed=0):
    """(b, SOH_k, g, tau_frac) sur une seule cellule descendue assez bas."""
    nm = float(cell["n"].max())
    b0 = max(cell["s0"] - cell["y"].min(), 1.0) / nm ** FADE_P
    p0 = np.array([b0, 80.0, 0.005, 0.5], float)
    lo = np.array([1e-7, 60.0, 1e-6, 0.02], float)
    hi = np.array([5.0, 95.0, 1.0, 5.0], float)
    r = _best(lambda p: l3(cell["n"], cell["s0"], p[0], p[1], p[2], p[3]) - cell["y"],
              p0, lo, hi, seed=seed)
    if r is None:
        return None
    p = r.x
    return dict(b=float(p[0]), soh_k=float(p[1]), g=float(p[2]),
                tau_frac=float(p[3]),
                rmse=float(np.sqrt(np.mean(r.fun ** 2))))


def run(parquet=None, out_csv=OUT_CSV, seed=0):
    """Produit la table des theta reels et l'ecrit sur disque."""
    cells = load_real_cells(parquet)
    rows = []

    for src in ("target", "snl"):
        sub = [c for c in cells if c["source"] == src]
        if not sub:
            continue
        j = fit_joint(sub, seed=seed)
        for c, b in zip(sub, j["b"]):
            rows.append(dict(cell_id=c["cell_id"], source=src, T=c["T"], C=c["C"],
                             n_max=float(c["n"].max()), n_points=len(c["n"]),
                             s0=c["s0"], soh_min=c["soh_min"], b=b,
                             soh_k=j["soh_k"], g=j["g"], tau_frac=j["tau_frac"],
                             knee_individuel=False, rmse=j["rmse"]))

    for c in [c for c in cells if c["source"] == "wheeler"]:
        f = fit_single(c, seed=seed)
        if f is None:
            continue
        rows.append(dict(cell_id=c["cell_id"], source="wheeler", T=c["T"], C=c["C"],
                         n_max=float(c["n"].max()), n_points=len(c["n"]),
                         s0=c["s0"], soh_min=c["soh_min"],
                         knee_individuel=True, **f))

    t = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_csv, index=False)
    return t


if __name__ == "__main__":
    t = run()
    pd.set_option("display.width", 200)
    print(t.round(5).to_string())
