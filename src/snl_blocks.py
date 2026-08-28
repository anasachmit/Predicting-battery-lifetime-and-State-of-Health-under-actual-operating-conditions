# -*- coding: utf-8 -*-
"""SNL : separer les cycles de vieillissement des controles de capacite.

Le protocole SNL alterne de longs blocs de cyclage a la C-rate d'essai et de
rares controles de capacite a une C-rate plus faible. Exemple mesure sur
`SNL_18650_LFP_25C_0-100_0.5-1C_a` : 3483 cycles de decharge a 1.099 A (1C,
91 points par cycle) et 62 cycles a 0.550 A (0.5C, 1438 points par cycle).

Consequence : prendre `max(discharge_capacity)` sur TOUS les cycles melange
deux grandeurs physiques distinctes - la capacite a 1C et la capacite a 0.5C,
qui differe de plusieurs pourcents. La trajectoire obtenue est en escalier, et
un escalier produit un ecart a la corde sans qu'aucun knee ne soit present.

On separe donc par la C-rate de decharge effectivement mesuree, et on mesure
sur chaque famille isolement.
"""
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

from .config import RAW

SNL_DIR = RAW / "snl" / "pkl"
_RE = re.compile(r"SNL_18650_LFP_(\d+)C_(\d+)-(\d+)_([\d.]+)-([\d.]+)C_(\w)")


def _cycle_rows(d):
    """Une ligne par cycle : capacite de decharge et courant median de decharge."""
    rows = []
    for c in d.get("cycle_data") or []:
        i = np.asarray(c.get("current_in_A") or [], float)
        q = np.asarray(c.get("discharge_capacity_in_Ah") or [], float)
        idis = i[i < -1e-3]
        if not idis.size or not q.size:
            continue
        qq = np.nanmax(q)
        if not np.isfinite(qq) or qq <= 0.1:
            continue
        rows.append((float(c["cycle_number"]), float(np.median(idis)), float(qq),
                     int(i.size)))
    return pd.DataFrame(rows, columns=["cycle_n", "i_dis", "q", "n_pts"])


def split_cell(path):
    """Renvoie (meta, df) ou df porte une colonne `famille` : aging ou rpt."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    m = _RE.search(os.path.basename(path))
    df = _cycle_rows(d)
    if df.empty:
        return None, df
    # la C-rate de vieillissement est celle qui domine en nombre de cycles
    rate = df["i_dis"].round(2)
    principal = rate.value_counts().idxmax()
    df["famille"] = np.where(np.isclose(rate, principal), "aging", "rpt")
    # doublons consecutifs : le pretraitement repete certains cycles
    df["doublon"] = df["q"].eq(df["q"].shift()) & df["cycle_n"].eq(df["cycle_n"].shift() + 1)
    meta = dict(cell_id=d["cell_id"], T=float(m.group(1)),
                C_chg=float(m.group(4)), C_dchg=float(m.group(5)),
                i_aging=float(principal),
                n_aging=int((df["famille"] == "aging").sum()),
                n_rpt=int((df["famille"] == "rpt").sum()),
                q_nom=float(d.get("nominal_capacity_in_Ah", 1.1)))
    return meta, df


def trajectories(pkl_dir=None, famille="aging", min_pts=40):
    """Trajectoires SOH par famille de cycles, normalisees dans la famille."""
    pkl_dir = pkl_dir or SNL_DIR
    trajs, metas = {}, []
    for f in sorted(glob.glob(os.path.join(str(pkl_dir), "*LFP*.pkl"))):
        meta, df = split_cell(f)
        if meta is None:
            continue
        sub = df[(df["famille"] == famille) & (~df["doublon"])].sort_values("cycle_n")
        if len(sub) < min_pts:
            continue
        n = sub["cycle_n"].to_numpy(float)
        q = sub["q"].to_numpy(float)
        early = q[n <= max(20, n.min() + 10)]
        q_ref = float(np.median(early)) if early.size else float(np.median(q[:5]))
        if not np.isfinite(q_ref) or q_ref <= 0:
            continue
        trajs[meta["cell_id"]] = (n, q / q_ref * 100.0)
        metas.append(dict(meta, famille=famille, n_utilises=len(sub),
                          q_ref=q_ref, soh_min=float((q / q_ref * 100).min())))
    return trajs, pd.DataFrame(metas)


def knee_vs_blocks(pkl_dir=None, tol=15):
    """Le coude tombe-t-il sur une frontiere de bloc ?

    Pour chaque cellule on compare le cycle du coude (mesure sur la trajectoire
    MELANGEE, celle qui contient l'artefact) a la position du controle de
    capacite le plus proche. Si l'ecart est systematiquement inferieur a `tol`,
    le coude est un artefact de protocole.
    """
    from .knee_detect import chord_deviation
    pkl_dir = pkl_dir or SNL_DIR
    rows = []
    for f in sorted(glob.glob(os.path.join(str(pkl_dir), "*LFP*.pkl"))):
        meta, df = split_cell(f)
        if meta is None or df.empty:
            continue
        d = df.sort_values("cycle_n")
        dev, n_knee, span = chord_deviation(d["cycle_n"], d["q"] / d["q"].iloc[:20].median() * 100)
        rpt = d.loc[d["famille"] == "rpt", "cycle_n"].to_numpy(float)
        dist = (float(np.min(np.abs(rpt - n_knee))) if rpt.size and np.isfinite(n_knee)
                else np.nan)
        rows.append(dict(cell_id=meta["cell_id"], T=meta["T"], C_dchg=meta["C_dchg"],
                         ecart_corde_melange=dev, cycle_knee=n_knee,
                         dist_au_controle=dist,
                         sur_frontiere=bool(np.isfinite(dist) and dist <= tol),
                         n_rpt=meta["n_rpt"], n_aging=meta["n_aging"]))
    return pd.DataFrame(rows)
