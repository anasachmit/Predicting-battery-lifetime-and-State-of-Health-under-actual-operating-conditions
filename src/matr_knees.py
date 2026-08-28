# -*- coding: utf-8 -*-
"""MATR : ou se situent les knees, et sont-ils du meme mecanisme que les notres ?

Le critere de decision n'est pas la PRESENCE d'un coude mais son NIVEAU DE SOH.
Le knee des cellules cible demarre a SOH_k = 80.4 % sous un cyclage symetrique
CC-CV a 0.5-1C. Celui de MATR, s'il existe, vient d'un placage de lithium
induit par une charge rapide multi-paliers a 3.6-6C. Si les distributions de
SOH au knee ne se recouvrent pas, la forme n'est pas transferable, et 169
cellules donneraient une fausse autorite a un mecanisme etranger.

Detection : Bacon-Watts sur la trajectoire complete. Valide sur la cellule de
controle b2c12, dont le knee est documente entre les cycles 365 et 391 par cinq
methodes concordantes : on obtient 366.1 +/- 1.4.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

from .config import RAW
from .knee_detect import bacon_watts, chord_deviation, curvature_ratio

MATR_PKL = RAW / "matr" / "pkl"
NOMINAL_AH = 1.1


def charge_intensity(protocol):
    """Resume l'agressivite du protocole de charge rapide multi-paliers."""
    if not protocol:
        return dict(c_max=np.nan, c_moyen=np.nan, c_premier=np.nan, n_paliers=0)
    rates = [s.get("rate_in_C") for s in protocol
             if s.get("rate_in_C") is not None]
    if not rates:
        return dict(c_max=np.nan, c_moyen=np.nan, c_premier=np.nan, n_paliers=0)
    return dict(c_max=float(np.max(rates)), c_moyen=float(np.mean(rates)),
                c_premier=float(rates[0]), n_paliers=len(rates))


def _capacities(cell_data):
    n, q = [], []
    for c in cell_data:
        v = c.get("discharge_capacity_in_Ah")
        if v is None or not len(v):
            continue
        vv = np.asarray(v, float)
        vv = vv[np.isfinite(vv)]
        if not vv.size:
            continue
        qq = float(vv.max())
        if qq <= 0.1:
            continue
        n.append(float(c["cycle_number"]))
        q.append(qq)
    return np.asarray(n), np.asarray(q)


def scan_all(pkl_dir=None, verbose=True, min_cycles=40):
    pkl_dir = pkl_dir or MATR_PKL
    files = sorted(glob.glob(os.path.join(str(pkl_dir), "*.pkl")))
    rows = []
    for i, f in enumerate(files):
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception:
            continue
        n, q = _capacities(d.get("cycle_data") or [])
        if len(n) < min_cycles:
            continue
        o = np.argsort(n)
        n, q = n[o], q[o]
        soh = q / NOMINAL_AH * 100.0        # convention nominale, comme le papier
        env = np.minimum.accumulate(soh)

        bw = bacon_watts(n, soh)
        dev, n_chord, span = chord_deviation(n, soh)
        ci = charge_intensity(d.get("charge_protocol"))

        rows.append(dict(
            cell_id=str(d.get("cell_id", os.path.basename(f))),
            n_cycles=int(n.max()), n_points=len(n),
            soh_depart=float(env[0]), soh_min=float(env[-1]), profondeur=float(span),
            x1=(bw["x1"] if bw else np.nan),
            x1_sd=(bw["x1_sd"] if bw else np.nan),
            soh_au_knee=(bw["soh_au_knee"] if bw else np.nan),
            rapport_pentes=(bw["rapport_pentes"] if bw else np.nan),
            bw_rmse=(bw["rmse"] if bw else np.nan),
            ecart_corde=dev, acceleration=curvature_ratio(n, soh),
            **ci))
        if verbose and (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(files)}")
    return pd.DataFrame(rows)


def knees_reels(df, rapport_min=2.0, sd_max_frac=0.05, prof_min=8.0):
    """Selection des knees credibles.

    Trois conditions cumulatives, toutes necessaires :
      - le rapport des pentes apres/avant depasse `rapport_min` : il y a bien
        une rupture, pas une simple courbure ;
      - l'incertitude sur le point de rupture reste sous `sd_max_frac` de la
        duree de vie : la rupture est LOCALISEE ;
      - la trajectoire parcourt au moins `prof_min` points de SOH, sans quoi
        aucune methode ne peut distinguer une rupture d'un bruit.
    """
    d = df.dropna(subset=["x1", "rapport_pentes"]).copy()
    d["sd_rel"] = d["x1_sd"] / d["n_cycles"]
    m = ((d["rapport_pentes"] >= rapport_min)
         & (d["sd_rel"] <= sd_max_frac)
         & (d["profondeur"] >= prof_min))
    return d[m].copy()
