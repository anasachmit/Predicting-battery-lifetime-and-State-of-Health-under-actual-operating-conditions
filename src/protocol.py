"""Duree de cycle reelle et temps ecoule, pour la loi L5 (terme calendaire).

L5 a besoin de t(n, C, T) = temps physique ecoule apres n cycles. Plutot que
de le modeliser, on le MESURE sur les cellules cible via `absolute_time`, puis
on ajuste une forme simple utilisable a la prediction (ou seuls T et C sont
connus).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from .config import FIGURES, TARGET_DIR

_TGT_RE = re.compile(r"102Ah_(\d+)degC_(0p5C|1C)_cell(\d+)$")


def cycle_durations(data_dir=None):
    """Duree horloge de chaque cycle (h), par cellule, depuis absolute_time."""
    data_dir = data_dir or TARGET_DIR
    out = []
    for folder in sorted(glob.glob(os.path.join(str(data_dir), "*"))):
        m = _TGT_RE.search(os.path.basename(folder))
        if m is None or not os.path.isdir(folder):
            continue
        parts = sorted(glob.glob(os.path.join(folder, "*_time_series*.csv")))
        dfs = [pd.read_csv(p, usecols=["cycle_number", "absolute_time"],
                           parse_dates=["absolute_time"]) for p in parts]
        ts = pd.concat(dfs, ignore_index=True)
        g = ts.groupby("cycle_number")["absolute_time"].agg(["min", "max"])
        g = g.sort_index()
        # duree = debut du cycle suivant - debut du cycle courant (inclut le repos)
        dur_h = (g["min"].shift(-1) - g["min"]).dt.total_seconds() / 3600.0
        span_h = (g["max"] - g["min"]).dt.total_seconds() / 3600.0
        out.append(pd.DataFrame({
            "cell_id": os.path.basename(folder),
            "T_amb": float(m.group(1)),
            "C_rate": 0.5 if m.group(2) == "0p5C" else 1.0,
            "cycle_n": g.index.astype(float),
            "dur_h": dur_h.to_numpy(),
            "span_h": span_h.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def duration_summary(dur):
    """Duree mediane par cellule, et derive au cours du vieillissement."""
    rows = []
    for cid, g in dur.groupby("cell_id"):
        g = g.dropna(subset=["dur_h"])
        # on ecarte les interruptions d'essai (pauses machine) : au-dela de 3x
        # la mediane ce n'est plus un cycle, c'est un arret de banc.
        med = g["dur_h"].median()
        ok = g[(g["dur_h"] > 0) & (g["dur_h"] < 3 * med)]
        n = ok["cycle_n"].to_numpy()
        d = ok["dur_h"].to_numpy()
        early = d[n <= np.percentile(n, 10)].mean() if len(n) else np.nan
        late = d[n >= np.percentile(n, 90)].mean() if len(n) else np.nan
        rows.append(dict(cell_id=cid, T_amb=g["T_amb"].iloc[0],
                         C_rate=g["C_rate"].iloc[0], n_cycles=len(ok),
                         dur_med_h=med, dur_early_h=early, dur_late_h=late,
                         derive_pct=100 * (late - early) / early,
                         frac_ecartee=1 - len(ok) / max(len(g), 1)))
    return pd.DataFrame(rows).sort_values(["C_rate", "T_amb"])


def fit_duration_model(summ):
    """duree ~ a/C + b  : temps CC proportionnel a 1/C, plus un forfait fixe
    (CV + repos). C'est la forme qui rend t(n, C) non proportionnel a n."""
    C = summ["C_rate"].to_numpy(float)
    y = summ["dur_med_h"].to_numpy(float)
    A = np.column_stack([1.0 / C, np.ones_like(C)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return dict(a_over_C=float(coef[0]), b_const=float(coef[1]),
                rmse_h=float(np.sqrt(np.mean((y - pred) ** 2))),
                r2=float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def plot_duration(dur, summ, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIGURES / "step0_cycle_duration.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    for cid, g in dur.groupby("cell_id"):
        g = g.dropna(subset=["dur_h"])
        med = g["dur_h"].median()
        g = g[(g["dur_h"] > 0) & (g["dur_h"] < 3 * med)]
        ax.plot(g["cycle_n"], g["dur_h"].rolling(51, min_periods=5).median(),
                lw=1.2, label=cid.replace("102Ah_", "").replace("_cell", " c"))
    ax.set_xlabel("cycle"); ax.set_ylabel("duree de cycle (h, mediane glissante)")
    ax.set_title("Duree horloge par cycle (repos inclus)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    ax = axes[1]
    for c, mk in [(0.5, "o"), (1.0, "s")]:
        s = summ[summ["C_rate"] == c]
        ax.scatter(s["T_amb"], s["dur_med_h"], marker=mk, s=110, ec="k",
                   label=f"{c}C")
    ax.set_xlabel("T ambiante (degC)"); ax.set_ylabel("duree mediane (h)")
    ax.set_title("Duree mediane par condition", fontsize=10)
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Modele de temps calendaire retenu pour L5 (voir audit T0).
# Forme integrale : duree(n) = (a/C) * SOH(n) + b, ajustee CONJOINTEMENT sur
# les 6 cellules et tous leurs cycles. Le `a` mesure (2.08 h.C) est proche de
# sa valeur physique : charge CC + decharge CC a la meme C-rate consomment
# 2*SOH/C heures. Le `b` couvre le palier CV et le repos.
# --------------------------------------------------------------------------
A_DUR_H_C = 2.0834   # h.C   coefficient du terme proportionnel au SOH
B_DUR_H = 1.0883     # h     forfait fixe (CV + repos)


def fit_integral_duration_model(dur, unified):
    """Ajustement conjoint duree = (a/C)*SOH + b sur toutes les cellules."""
    rows = []
    for cid, g in dur.groupby("cell_id"):
        med = g["dur_h"].median()
        g = g[(g["dur_h"] > 0) & (g["dur_h"] < 3 * med)]
        u = unified[unified["cell_id"] == cid][["cycle_n", "SOH"]]
        rows.append(g.merge(u, on="cycle_n").dropna(subset=["SOH", "dur_h"]))
    m = pd.concat(rows, ignore_index=True)
    y = m["dur_h"].to_numpy(float)
    for name, X in [("integral", np.column_stack([m["SOH"] / m["C_rate"],
                                                  np.ones(len(m))])),
                    ("constant", np.column_stack([1.0 / m["C_rate"],
                                                  np.ones(len(m))]))]:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        yield dict(modele=name, a_over_C=float(coef[0]), b_const=float(coef[1]),
                   r2=float(1 - ((y - pred) ** 2).sum()
                            / ((y - y.mean()) ** 2).sum()),
                   rmse_h=float(np.sqrt(np.mean((y - pred) ** 2))))


def elapsed_time_h(soh_traj, c_rate, a=A_DUR_H_C, b=B_DUR_H):
    """Temps physique ecoule (h) apres chaque cycle, forme integrale.

    soh_traj : SOH en fraction (pas en %), pour les cycles 1..N.
    Utilisable a la prediction : ne depend que de la trajectoire predite.
    """
    soh = np.asarray(soh_traj, float)
    return np.cumsum(a / float(c_rate) * soh + b)


def elapsed_time_linear_h(n, c_rate, dur_h=None, rest_h=0.5):
    """Forme lineaire de reference : t = n * (duree(C) + repos).
    Sert d'ablation face a `elapsed_time_h`."""
    n = np.asarray(n, float)
    if dur_h is None:
        dur_h = 1.9017 / float(c_rate) + 0.939
    return n * (dur_h + rest_h)
