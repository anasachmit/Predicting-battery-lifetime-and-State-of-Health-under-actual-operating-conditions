"""Loaders des trois sources, vers le schema unifie.

Schema (une ligne = un cycle avec une capacite mesuree) :
    cell_id, source, cycle_n, capacity_Ah, SOH, T_amb,
    C_rate_chg, C_rate_dchg, DoD, format, Q_nom

Convention SOH : capacite du cycle / capacite de reference de LA CELLULE
(mediane des premiers points de controle valides), et non la capacite
nominale du catalogue -> evite le biais de formation entre cellules.
La colonne `SOH_nom` conserve la definition nominale pour comparaison,
car c'est celle qu'utilise le scoring officiel du challenge.
"""
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy.io as sio

from .config import SNL_PKL_DIR, TARGET_DIR, TARGET_Q_NOM_AH, WHEELER_MAT

# Capacite de reference : mediane des points de controle des REF_CYCLE_MAX
# premiers cycles. Chez Wheeler les RPT sont espacees de 100 cycles, il n'y a
# donc qu'un seul point dans cette fenetre (le RPT initial) -- c'est bien la
# capacite BOL, et non une moyenne deja degradee.
REF_CYCLE_MAX = 10


def _scalar(x):
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    return a[0] if a.size else np.nan


def _finalise(df):
    """Ajoute SOH (reference cellule) et trie."""
    out = []
    for cid, g in df.groupby("cell_id", sort=False):
        g = g.sort_values("cycle_n").copy()
        ok = g["capacity_Ah"].notna() & (g["capacity_Ah"] > 0)
        if ok.sum() < 3:
            continue
        early = g.loc[ok & (g["cycle_n"] <= REF_CYCLE_MAX), "capacity_Ah"]
        if early.empty:                      # aucun point tres precoce
            early = g.loc[ok, "capacity_Ah"].head(1)
        q_ref = float(early.median())
        g["Q_ref"] = q_ref
        g["SOH"] = g["capacity_Ah"] / q_ref
        g["SOH_nom"] = g["capacity_Ah"] / g["Q_nom"]
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


# --------------------------------------------------------------------------
# Wheeler et al. 2025 - 20 cellules A123 18650 Graphite/LFP 1.1 Ah, CC BY 4.0
# DOI 10.57745/OLBXKT. On lit extractedData.mat (0.46 Mo), qui contient les
# points de caracterisation (RPT) de chaque cellule : (cycle, capacite, SOH).
# Les 26 Go de zips bruts ne sont PAS necessaires pour la trajectoire SOH.
# --------------------------------------------------------------------------
def load_wheeler(path=None):
    path = path or WHEELER_MAT
    cells = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["cell"]
    rows = []
    for c in cells:
        name = str(c.name)
        # La condition d'essai est encodee dans le chemin d'origine des fichiers
        # (ex. "agingdata_50degC"); on la relit telle quelle.
        folders = {str(f.folder).split("\\")[-1]
                   for f in np.atleast_1d(c.agingfileslist).ravel()}
        temps = sorted({int(m.group(1)) for f in folders
                        if (m := re.search(r"(\d+)degC", f))})
        t_amb = float(temps[0]) if len(temps) == 1 else np.nan
        for ch in np.atleast_1d(c.charac).ravel():
            n, q = _scalar(ch.cycle), _scalar(ch.chargecapacity)
            if not np.isfinite(n) or not np.isfinite(q):
                continue
            rows.append(dict(cell_id=f"WHEELER_{name}", source="wheeler",
                             cycle_n=float(n), capacity_Ah=float(q),
                             T_amb=t_amb, C_rate_chg=np.nan, C_rate_dchg=np.nan,
                             DoD=np.nan, format="18650", Q_nom=1.1))
    return _finalise(pd.DataFrame(rows))


# --------------------------------------------------------------------------
# SNL (Sandia, via BatteryLife processed, Zenodo 19688272, CC BY 4.0)
# 18 cellules 18650 LFP 1.1 Ah, 15/25/35 degC, decharge 0.5-3C, DoD 100 %.
# --------------------------------------------------------------------------
_SNL_RE = re.compile(r"SNL_18650_LFP_(\d+)C_(\d+)-(\d+)_([\d.]+)-([\d.]+)C_(\w)")


def load_snl_lfp(pkl_dir=None):
    pkl_dir = pkl_dir or SNL_PKL_DIR
    rows = []
    for f in sorted(glob.glob(os.path.join(str(pkl_dir), "*LFP*.pkl"))):
        m = _SNL_RE.search(os.path.basename(f))
        if m is None:
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        soc_lo, soc_hi = int(m.group(2)), int(m.group(3))
        for cy in d["cycle_data"]:
            qd = cy.get("discharge_capacity_in_Ah")
            if not qd:
                continue
            q = float(np.nanmax(qd))
            if not np.isfinite(q) or q <= 0.1:
                continue
            rows.append(dict(cell_id=d["cell_id"], source="snl",
                             cycle_n=float(cy["cycle_number"]), capacity_Ah=q,
                             T_amb=float(m.group(1)),
                             C_rate_chg=float(m.group(4)),
                             C_rate_dchg=float(m.group(5)),
                             DoD=(soc_hi - soc_lo) / 100.0, format="18650",
                             Q_nom=float(d["nominal_capacity_in_Ah"])))
    return _finalise(pd.DataFrame(rows))


# --------------------------------------------------------------------------
# Cellules cible du challenge (locales) - LFP prismatique 102 Ah.
# Le label SOH officiel = Qdis(cycle) / 102 Ah ; on le recalcule exactement
# comme framework/data.py du template, puis on ajoute la version normalisee
# par la capacite de reference de la cellule.
# --------------------------------------------------------------------------
_TGT_RE = re.compile(r"102Ah_(\d+)degC_(0p5C|1C)_cell(\d+)$")


def load_target(data_dir=None):
    data_dir = data_dir or TARGET_DIR
    rows = []
    for folder in sorted(glob.glob(os.path.join(str(data_dir), "*"))):
        m = _TGT_RE.search(os.path.basename(folder))
        if m is None or not os.path.isdir(folder):
            continue
        cid = os.path.basename(folder)
        parts = sorted(glob.glob(os.path.join(folder, "*_time_series*.csv")))
        dfs = [pd.read_csv(p, usecols=["cycle_number", "step_type", "step_capacity_Ah"])
               for p in parts]
        ts = pd.concat(dfs, ignore_index=True)
        dis = ts[ts["step_type"] == "cc_discharge"]
        q = dis.groupby("cycle_number")["step_capacity_Ah"].max().sort_index()
        qv = q.astype(float)
        # Nettoyage identique au framework officiel : cycles aberrants et
        # cycles partiels (interrompus) ne portent pas de label.
        implaus = (qv > 1.15 * TARGET_Q_NOM_AH) | (qv < 0)
        med = (qv.shift(1).rolling(15, min_periods=3).median()
               .fillna(qv.rolling(9, center=True, min_periods=1).median()))
        partial = (qv < 0.6 * med) & ~implaus
        keep = ~implaus & ~partial
        c_rate = 0.5 if m.group(2) == "0p5C" else 1.0
        sub = pd.DataFrame({
            "cell_id": cid, "source": "target",
            "cycle_n": q.index.astype(float)[keep.values],
            "capacity_Ah": qv.values[keep.values],
            "T_amb": float(m.group(1)),
            "C_rate_chg": c_rate, "C_rate_dchg": c_rate, "DoD": 1.0,
            "format": "prismatic", "Q_nom": TARGET_Q_NOM_AH})
        rows.append(sub)
    return _finalise(pd.concat(rows, ignore_index=True))


def load_all():
    return pd.concat([load_wheeler(), load_snl_lfp(), load_target()],
                     ignore_index=True)
