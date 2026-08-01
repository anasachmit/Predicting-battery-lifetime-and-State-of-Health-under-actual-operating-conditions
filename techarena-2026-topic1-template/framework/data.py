# -*- coding: utf-8 -*-
# =============================================================================
#  FRAMEWORK - DO NOT EDIT.
#  Loads the dataset and hands your model a simple list of Cell objects.
# =============================================================================
"""
Each training cell is given to your model as a Cell object:

    cell.cell_id            e.g. "102Ah_45degC_1C_cell1"
    cell.temperature_degC   e.g. 45
    cell.c_rate             e.g. 1.0
    cell.soh                DataFrame: cycle_number, soh_percent
                            (the official label definition, cleaned)
    cell.time_series()      full raw signals as a DataFrame (lazy; only
                            loaded if you call it): cycle_number, step_type,
                            time_in_cycle_s, voltage_V, current_A,
                            temperature_C, step_capacity_Ah, absolute_time

SOH label definition (identical to the hidden evaluation ground truth):
SOH (%) = discharge capacity of the cycle / 102 Ah nominal x 100, where the
discharge capacity is the maximum cumulative step_capacity_Ah within the
cycle's cc_discharge step; implausible (> 1.15 x nominal) and aborted
partial cycles carry no label.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

NOMINAL_AH = 102.0
_CELL_RE = re.compile(r"102Ah_(\d+)degC_(0p5C|1C)_cell(\d+)$")


class Cell:
    def __init__(self, cell_id, folder, temperature_degC, c_rate):
        self.cell_id = cell_id
        self.folder = folder
        self.temperature_degC = temperature_degC
        self.c_rate = c_rate
        self.soh = _derive_soh(self._read(usecols=["cycle_number", "step_type",
                                                   "step_capacity_Ah"]))

    def time_series(self):
        return self._read()

    def _read(self, usecols=None):
        single = os.path.join(self.folder, f"{self.cell_id}_time_series.csv")
        parts = sorted(glob.glob(os.path.join(
            self.folder, f"{self.cell_id}_time_series_part*.csv")))
        files = [single] if os.path.exists(single) else parts
        if not files:                       # tolerant: any time series inside
            files = sorted(glob.glob(os.path.join(self.folder,
                                                  "*_time_series*.csv")))
        dfs = []
        for f in files:
            head = pd.read_csv(f, nrows=0)
            cols = [c for c in usecols if c in head.columns] if usecols else None
            dfs.append(pd.read_csv(f, usecols=cols))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def __repr__(self):
        n = int(self.soh["soh_percent"].notna().sum())
        return (f"Cell({self.cell_id}, T={self.temperature_degC} degC, "
                f"C={self.c_rate}, {n} labeled cycles)")


def _derive_soh(ts):
    dis = ts[ts["step_type"] == "cc_discharge"]
    q = dis.groupby("cycle_number")["step_capacity_Ah"].max().sort_index()
    lab = pd.DataFrame({"cycle_number": q.index.astype(int),
                        "discharge_capacity_Ah": q.values})
    qv = lab["discharge_capacity_Ah"].astype(float)
    implaus = (qv > 1.15 * NOMINAL_AH) | (qv < 0)
    med = qv.shift(1).rolling(15, min_periods=3).median() \
            .fillna(qv.rolling(9, center=True, min_periods=1).median())
    partial = (qv < 0.6 * med) & ~implaus
    ok = ~implaus & ~partial
    lab["soh_percent"] = np.where(ok, (qv / NOMINAL_AH * 100).round(3), np.nan)
    return lab[["cycle_number", "soh_percent"]]


def load_cells(data_dir, verbose=True):
    """All cells found below data_dir, as a list of Cell objects."""
    cells = []
    hits = glob.glob(os.path.join(data_dir, "**", "*_time_series*part01.csv"),
                     recursive=True)
    hits += glob.glob(os.path.join(data_dir, "**", "*_time_series.csv"),
                      recursive=True)
    for f in sorted(set(os.path.dirname(h) for h in hits)):
        cid = os.path.basename(f)
        m = _CELL_RE.search(cid)
        if not m:
            continue
        cell = Cell(cid, f, int(m.group(1)),
                    0.5 if m.group(2) == "0p5C" else 1.0)
        if verbose:
            print(f"  loaded {cell}")
        cells.append(cell)
    if not cells:
        raise FileNotFoundError(f"no cells found under {data_dir}")
    return cells
