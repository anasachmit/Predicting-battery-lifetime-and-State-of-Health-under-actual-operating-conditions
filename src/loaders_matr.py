# -*- coding: utf-8 -*-
"""Loader MATR / Severson-Attia, via le pretraitement BatteryLife.

169 cellules LFP 18650 1.1 Ah, toutes a 30 degC, protocoles de charge rapide
varies, decharge identique. Source : Zenodo 19688272 (CC BY 4.0), qui reprend
data.matr.io.

On n'extrait que la capacite de decharge par cycle : c'est tout ce qu'il faut
pour tester la presence d'un knee, et ca evite de garder 10 Go en memoire.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

from .config import RAW

MATR_DIR = RAW / "matr" / "pkl"
NOMINAL_AH = 1.1


def _cycle_capacity(cycle):
    q = cycle.get("discharge_capacity_in_Ah")
    if q is None or (hasattr(q, "__len__") and len(q) == 0):
        return np.nan
    v = np.asarray(q, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else np.nan


def load_one(path, q_min=0.1):
    """Renvoie (cell_id, cycles, capacite_Ah, metadonnees)."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    cd = d.get("cycle_data") or []
    n = np.array([c.get("cycle_number", np.nan) for c in cd], float)
    q = np.array([_cycle_capacity(c) for c in cd], float)
    ok = np.isfinite(n) & np.isfinite(q) & (q > q_min)
    meta = dict(nominal=d.get("nominal_capacity_in_Ah", NOMINAL_AH),
                anode=d.get("anode_material"), cathode=d.get("cathode_material"),
                form=d.get("form_factor"),
                chg=d.get("charge_protocol"), dchg=d.get("discharge_protocol"))
    return str(d.get("cell_id", os.path.basename(path))), n[ok], q[ok], meta


def load_all(pkl_dir=None, limit=None, verbose=True):
    """Trajectoires SOH de toutes les cellules MATR trouvees.

    SOH normalise par la capacite de reference DE LA CELLULE (mediane des
    cycles <= 10), comme pour les autres sources du banc.
    """
    pkl_dir = pkl_dir or MATR_DIR
    files = sorted(glob.glob(os.path.join(str(pkl_dir), "*.pkl")))
    if limit:
        files = files[:limit]
    trajs, rows = {}, []
    for i, f in enumerate(files):
        try:
            cid, n, q, meta = load_one(f)
        except Exception as e:
            if verbose:
                print(f"  [skip] {os.path.basename(f)} : {e}")
            continue
        if len(n) < 20:
            continue
        o = np.argsort(n)
        n, q = n[o], q[o]
        early = q[n <= 10]
        q_ref = float(np.median(early)) if early.size else float(np.median(q[:5]))
        if not np.isfinite(q_ref) or q_ref <= 0:
            continue
        soh = q / q_ref * 100.0
        trajs[cid] = (n, soh)
        rows.append(dict(cell_id=cid, n_cycles=int(n.max()), n_points=len(n),
                         q_ref=q_ref, soh_min=float(np.min(soh)),
                         soh_depart=float(soh[0]),
                         nominal=meta["nominal"], cathode=meta["cathode"]))
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} lues")
    return trajs, pd.DataFrame(rows)
