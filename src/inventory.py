"""Etape 0 : construction de data/unified.parquet et inventaire chiffre."""
import numpy as np
import pandas as pd

from .config import DATA, FIGURES, UNIFIED_SCHEMA
from .loaders import load_all
from .runlog import log_run

UNIFIED_PATH = DATA / "unified.parquet"


def build_unified(save=True):
    df = load_all()
    cols = UNIFIED_SCHEMA + ["Q_ref", "SOH_nom"]
    df = df[cols].sort_values(["source", "cell_id", "cycle_n"]).reset_index(drop=True)
    if save:
        df.to_parquet(UNIFIED_PATH, index=False)
    return df


def _cross(n, soh, thr):
    """Premier cycle ou SOH passe sous `thr` (interpole). -1 si jamais atteint."""
    below = np.where(soh < thr)[0]
    if not len(below) or below[0] == 0:
        return -1
    j = below[0]
    return float(np.interp(thr, [soh[j], soh[j - 1]], [n[j], n[j - 1]]))


def per_cell_summary(df):
    rows = []
    for cid, g in df.groupby("cell_id", sort=False):
        g = g.sort_values("cycle_n")
        n = g["cycle_n"].to_numpy(float)
        s = g["SOH"].to_numpy(float)
        ok = np.isfinite(s)
        n, s = n[ok], s[ok]
        # enveloppe monotone decroissante : robuste au bruit de mesure
        s_env = np.minimum.accumulate(s)
        rows.append(dict(
            cell_id=cid, source=g["source"].iloc[0],
            T_amb=g["T_amb"].iloc[0], C_rate_chg=g["C_rate_chg"].iloc[0],
            C_rate_dchg=g["C_rate_dchg"].iloc[0], DoD=g["DoD"].iloc[0],
            format=g["format"].iloc[0], Q_nom=g["Q_nom"].iloc[0],
            Q_ref=g["Q_ref"].iloc[0], n_points=len(n), n_max=n.max(),
            SOH_min=s.min(),
            n_at_90=_cross(n, s_env, 0.90), n_at_80=_cross(n, s_env, 0.80),
            n_at_70=_cross(n, s_env, 0.70), n_at_60=_cross(n, s_env, 0.60),
            frac_below_80=float((s < 0.80).mean()),
        ))
    return pd.DataFrame(rows)


def coverage_table(summ):
    g = summ.groupby("source")
    out = pd.DataFrame({
        "n_cells": g.size(),
        "n_T_levels": g["T_amb"].nunique(),
        "T_range": g["T_amb"].agg(lambda x: f"{np.nanmin(x):.0f}-{np.nanmax(x):.0f}"),
        "n_Cdchg_levels": g["C_rate_dchg"].nunique(),
        "cells_le_80": g["SOH_min"].apply(lambda x: int((x <= 0.80).sum())),
        "cells_le_70": g["SOH_min"].apply(lambda x: int((x <= 0.70).sum())),
        "cells_le_60": g["SOH_min"].apply(lambda x: int((x <= 0.60).sum())),
        "pts_below_80pct": g["frac_below_80"].mean().round(3),
    })
    return out.reset_index()


def plot_inventory(df, summ, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIGURES / "step0_inventory.png")
    srcs = ["wheeler", "snl", "target"]
    titles = {"wheeler": "Wheeler 2025 (20 cell., 18650 LFP, 50 degC)",
              "snl": "SNL LFP (18 cell., 15/25/35 degC)",
              "target": "Cible challenge (6 cell., 102 Ah prismatique)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, src in zip(axes, srcs):
        sub = df[df["source"] == src]
        for cid, g in sub.groupby("cell_id"):
            g = g.sort_values("cycle_n")
            ax.plot(g["cycle_n"], g["SOH"] * 100, lw=0.9, alpha=0.8)
        ax.axhline(80, color="0.4", ls=":", lw=1)
        ax.axhline(70, color="crimson", ls="--", lw=1.2)
        ax.set_title(titles[src], fontsize=10)
        ax.set_xlabel("cycle")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("SOH (%, ref. cellule)")
    axes[0].set_ylim(30, 105)
    fig.suptitle("Etape 0 - trajectoires SOH par source ; rouge = seuil 70 %", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run():
    df = build_unified()
    summ = per_cell_summary(df)
    cov = coverage_table(summ)
    fig = plot_inventory(df, summ)
    summ.to_csv(DATA / "cell_summary.csv", index=False)
    log_run("step0_inventory",
            params={"sources": sorted(df["source"].unique().tolist())},
            metrics={"n_cells": int(summ.shape[0]),
                     "n_rows": int(df.shape[0]),
                     "cells_le_70": int((summ["SOH_min"] <= 0.70).sum()),
                     "cells_le_80": int((summ["SOH_min"] <= 0.80).sum())},
            notes=f"figure={fig.name}")
    return df, summ, cov
