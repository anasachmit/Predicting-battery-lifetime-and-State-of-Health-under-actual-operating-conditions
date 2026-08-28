# -*- coding: utf-8 -*-
"""T0 - Audit des bases de temps sur l'INTEGRALITE des cellules cible.

Question : `time_in_cycle_s` et `absolute_time` sont-ils coherents, et laquelle
fait foi ? La branche calendaire de L5 en depend entierement.

Trois tests, tous sur toutes les lignes de toutes les cellules :

  A. offset = absolute_time - time_in_cycle_s, par (cellule, cycle).
     Si les deux bases sont coherentes, l'offset est CONSTANT a l'interieur
     d'un cycle (c'est l'instant d'origine du cycle). On mesure sa dispersion.

  B. coherence des increments : apres tri chronologique, ratio
     diff(absolute_time) / diff(time_in_cycle_s), par cellule et step_type.

  C. arbitrage physique : capacite recalculee par integration du courant sur
     chaque base de temps, comparee a `step_capacity_Ah` du fabricant.
     La base de temps physique est celle qui reproduit la capacite.

Puis : duree de cycle recalculee depuis `time_in_cycle_s` seul, a comparer a
la mesure `1.90/C + 0.94 h` obtenue depuis `absolute_time`.

ATTENTION : les lignes des CSV ne sont PAS stockees dans l'ordre
chronologique (blocs decroissants d'environ 8 lignes). Tout calcul de
difference exige un tri prealable ; ne pas trier produit un ratio aberrant
(c'est l'artefact "facteur 12").
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from .config import FIGURES, TARGET_DIR

_TGT_RE = re.compile(r"102Ah_(\d+)degC_(0p5C|1C)_cell(\d+)$")
_COLS = ["cycle_number", "step_type", "time_in_cycle_s", "current_A",
         "step_capacity_Ah", "absolute_time"]


def _cell_folders(data_dir=None):
    data_dir = data_dir or TARGET_DIR
    out = []
    for folder in sorted(glob.glob(os.path.join(str(data_dir), "*"))):
        m = _TGT_RE.search(os.path.basename(folder))
        if m and os.path.isdir(folder):
            out.append((os.path.basename(folder), folder, float(m.group(1)),
                        0.5 if m.group(2) == "0p5C" else 1.0))
    return out


def _read_cell(folder):
    parts = sorted(glob.glob(os.path.join(folder, "*_time_series*.csv")))
    dfs = [pd.read_csv(p, usecols=_COLS, parse_dates=["absolute_time"])
           for p in parts]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------- test A ----
def offset_consistency(ts):
    """offset = absolute_time - time_in_cycle_s, doit etre constant par cycle."""
    t_abs = ts["absolute_time"].astype("int64") / 1e9      # secondes epoch
    off = t_abs - ts["time_in_cycle_s"]
    g = off.groupby(ts["cycle_number"])
    spread = (g.max() - g.min())                            # etendue, secondes
    return dict(
        n_cycles=int(spread.size),
        offset_spread_median_s=float(spread.median()),
        offset_spread_p99_s=float(spread.quantile(0.99)),
        offset_spread_max_s=float(spread.max()),
        frac_cycles_spread_gt_2s=float((spread > 2).mean()),
        frac_cycles_spread_gt_60s=float((spread > 60).mean()),
    )


# ---------------------------------------------------------------- test B ----
def increment_ratio(ts):
    """Ratio diff(absolute_time)/diff(time_in_cycle_s) APRES tri, par step."""
    rows = []
    for step, g in ts.groupby("step_type"):
        g = g.sort_values(["cycle_number", "time_in_cycle_s"])
        d_rel = g["time_in_cycle_s"].diff()
        d_abs = g["absolute_time"].diff().dt.total_seconds()
        same_cycle = g["cycle_number"].diff() == 0
        m = same_cycle & (d_rel > 0)
        if m.sum() < 100:
            continue
        r = (d_abs[m] / d_rel[m]).to_numpy()
        rows.append(dict(step_type=step, n=int(m.sum()),
                         ratio_median=float(np.median(r)),
                         ratio_p01=float(np.percentile(r, 1)),
                         ratio_p99=float(np.percentile(r, 99)),
                         frac_exactement_1=float(np.mean(np.abs(r - 1) < 1e-6))))
    # sans tri : reproduit l'artefact observe
    d_rel = ts["time_in_cycle_s"].diff()
    d_abs = ts["absolute_time"].diff().dt.total_seconds()
    m = (ts["cycle_number"].diff() == 0) & (d_rel.abs() > 0)
    r_ns = (d_abs[m] / d_rel[m]).to_numpy()
    rows.append(dict(step_type="__NON TRIE (artefact)__", n=int(m.sum()),
                     ratio_median=float(np.median(r_ns)),
                     ratio_p01=float(np.percentile(r_ns, 1)),
                     ratio_p99=float(np.percentile(r_ns, 99)),
                     frac_exactement_1=float(np.mean(np.abs(r_ns - 1) < 1e-6))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test C ----
def capacity_reconstruction(ts, n_cycles=60):
    """Capacite = integrale |I| dt sur chaque base de temps, vs step_capacity_Ah.

    On integre sur la decharge CC, ou `step_capacity_Ah` est cumulatif et ou le
    courant est constant : la comparaison est alors sans ambiguite.
    """
    cyc = np.sort(ts["cycle_number"].unique())
    cyc = cyc[np.linspace(0, len(cyc) - 1, min(n_cycles, len(cyc))).astype(int)]
    sub = ts[(ts["cycle_number"].isin(cyc)) & (ts["step_type"] == "cc_discharge")]
    rows = []
    for n, g in sub.groupby("cycle_number"):
        g = g.sort_values("time_in_cycle_s")
        q_ref = float(g["step_capacity_Ah"].max())
        if q_ref <= 0 or len(g) < 10:
            continue
        i = g["current_A"].abs().to_numpy()
        t_rel = g["time_in_cycle_s"].to_numpy()
        t_abs = g["absolute_time"].astype("int64").to_numpy() / 1e9
        t_abs = t_abs - t_abs[0]
        q_rel = np.trapezoid(i, t_rel) / 3600.0
        q_abs = np.trapezoid(i, t_abs) / 3600.0
        rows.append(dict(cycle_number=int(n), q_declaree_Ah=q_ref,
                         q_integ_time_in_cycle_Ah=q_rel,
                         q_integ_absolute_time_Ah=q_abs))
    d = pd.DataFrame(rows)
    if d.empty:
        return d, {}
    stat = {}
    for col, lab in [("q_integ_time_in_cycle_Ah", "time_in_cycle_s"),
                     ("q_integ_absolute_time_Ah", "absolute_time")]:
        err = (d[col] - d["q_declaree_Ah"]) / d["q_declaree_Ah"] * 100
        stat[lab] = dict(biais_median_pct=float(err.median()),
                         mae_pct=float(err.abs().mean()),
                         p95_abs_pct=float(err.abs().quantile(0.95)))
    return d, stat


# ------------------------------------------------------- duree de cycle ----
def durations_both_bases(ts):
    """Duree de cycle selon chaque base de temps (h)."""
    g = ts.groupby("cycle_number")
    dur_rel = (g["time_in_cycle_s"].max() - g["time_in_cycle_s"].min()) / 3600.0
    a_min = g["absolute_time"].min()
    a_max = g["absolute_time"].max()
    dur_abs = (a_max - a_min).dt.total_seconds() / 3600.0
    # duree "horloge" = debut du cycle suivant - debut du cycle courant
    dur_wall = (a_min.shift(-1) - a_min).dt.total_seconds() / 3600.0
    return pd.DataFrame({"cycle_number": dur_rel.index,
                         "dur_time_in_cycle_h": dur_rel.to_numpy(),
                         "dur_absolute_span_h": dur_abs.to_numpy(),
                         "dur_wall_h": dur_wall.to_numpy()})


def _robust_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x):
        return np.nan
    med = np.median(x)
    return float(np.median(x[x < 3 * med]))


def run(data_dir=None, verbose=True):
    cells = _cell_folders(data_dir)
    off_rows, ratio_rows, cap_rows, dur_rows, capdet = [], [], [], [], {}
    for cid, folder, T, C in cells:
        ts = _read_cell(folder)
        if verbose:
            print(f"  {cid}: {len(ts):,} lignes")
        o = offset_consistency(ts); o.update(cell_id=cid, T_amb=T, C_rate=C)
        off_rows.append(o)

        r = increment_ratio(ts); r.insert(0, "cell_id", cid)
        ratio_rows.append(r)

        d, stat = capacity_reconstruction(ts)
        capdet[cid] = d
        for base, s in stat.items():
            cap_rows.append(dict(cell_id=cid, base_de_temps=base, **s))

        du = durations_both_bases(ts)
        dur_rows.append(dict(
            cell_id=cid, T_amb=T, C_rate=C,
            dur_time_in_cycle_h=_robust_median(du["dur_time_in_cycle_h"]),
            dur_absolute_span_h=_robust_median(du["dur_absolute_span_h"]),
            dur_wall_h=_robust_median(du["dur_wall_h"])))
    return (pd.DataFrame(off_rows), pd.concat(ratio_rows, ignore_index=True),
            pd.DataFrame(cap_rows), pd.DataFrame(dur_rows), capdet)


def fit_duration(summ, col):
    """duree ~ a/C + b sur la colonne demandee."""
    C = summ["C_rate"].to_numpy(float)
    y = summ[col].to_numpy(float)
    A = np.column_stack([1.0 / C, np.ones_like(C)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return dict(base=col, a_over_C=float(coef[0]), b_const=float(coef[1]),
                rmse_h=float(np.sqrt(np.mean((y - pred) ** 2))),
                r2=float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def plot_audit(off, ratio, capdet, dur, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIGURES / "T0_audit_timestamps.png")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    ax.bar(range(len(off)), off["offset_spread_max_s"], color="#1f77b4")
    ax.set_xticks(range(len(off)))
    ax.set_xticklabels([c.replace("102Ah_", "").replace("_cell", "\nc")
                        for c in off["cell_id"]], fontsize=7, rotation=45)
    ax.set_ylabel("etendue max de l'offset (s)")
    ax.set_title("A. offset = absolute_time - time_in_cycle_s\n"
                 "(0 = bases parfaitement coherentes)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    tri = ratio[ratio["step_type"] != "__NON TRIE (artefact)__"]
    art = ratio[ratio["step_type"] == "__NON TRIE (artefact)__"]
    ax.scatter(range(len(tri)), tri["ratio_median"], s=45, c="#1f77b4",
               label="apres tri chronologique")
    ax.scatter(range(len(art)), art["ratio_median"], s=45, c="crimson",
               marker="x", label="sans tri (artefact)")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_ylabel("mediane du ratio d(abs)/d(rel)")
    ax.set_xlabel("cellule x step_type")
    ax.set_title("B. coherence des increments", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    cid = list(capdet)[0]
    d = capdet[cid]
    ax.plot(d["q_declaree_Ah"], d["q_integ_time_in_cycle_Ah"], "o", ms=4,
            label="int. sur time_in_cycle_s")
    ax.plot(d["q_declaree_Ah"], d["q_integ_absolute_time_Ah"], "s", ms=4,
            mfc="none", label="int. sur absolute_time")
    lim = [d["q_declaree_Ah"].min() * 0.95, d["q_declaree_Ah"].max() * 1.05]
    ax.plot(lim, lim, "0.4", ls="--", lw=1)
    ax.set_xlabel("step_capacity_Ah declaree"); ax.set_ylabel("capacite integree (Ah)")
    ax.set_title(f"C. arbitrage physique\n{cid}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
