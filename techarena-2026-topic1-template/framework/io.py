# -*- coding: utf-8 -*-
# =============================================================================
#  FRAMEWORK - DO NOT EDIT.
#  Fills SOH_model for every requested row and writes output.csv in the
#  required format:  Cycle, SOH_true, Temperature (degC),
#                    Current (C-rate), SOH_model, errSOH
# =============================================================================
import os

import numpy as np
import pandas as pd

COLUMNS = ["Cycle", "SOH_true", "Temperature (degC)", "Current (C-rate)",
           "SOH_model", "errSOH"]


def write_output(inp, model, output_dir):
    df = inp.copy()
    if "SOH_true" not in df.columns:
        df["SOH_true"] = np.nan
    df["SOH_model"] = np.nan
    for (T, C), g in df.groupby(["Temperature (degC)", "Current (C-rate)"]):
        soh = np.asarray(model.predict_soh(float(T), float(C)),
                         dtype=float).ravel()
        cyc = g["Cycle"].to_numpy(int)
        if cyc.min() < 1:
            raise ValueError("input contains Cycle < 1")
        if cyc.max() > len(soh):
            raise ValueError(
                f"predict_soh({T}, {C}) returned {len(soh)} values but "
                f"cycle {cyc.max()} was requested - return a longer "
                f"trajectory (up to 12000 values is always safe)")
        df.loc[g.index, "SOH_model"] = soh[cyc - 1]
    bad = ~np.isfinite(df["SOH_model"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} SOH_model values are NaN/inf")
    if (df["SOH_model"] <= 0).any() or (df["SOH_model"] > 120).any():
        raise ValueError("SOH_model values must be in (0, 120]")
    df["SOH_model"] = df["SOH_model"].round(3)
    df["errSOH"] = (df["SOH_model"] - df["SOH_true"]).round(3)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "output.csv")
    df[COLUMNS].to_csv(path, index=False)
    return path


def check_output_file(path, inp):
    problems = []
    df = pd.read_csv(path)
    if [c for c in COLUMNS if c not in df.columns]:
        return [f"output.csv must have columns {COLUMNS}"]
    if len(df) != len(inp):
        problems.append(f"row count {len(df)} != input rows {len(inp)}")
    if df["SOH_model"].isna().any():
        problems.append("SOH_model contains empty values")
    elif ((df["SOH_model"] <= 0) | (df["SOH_model"] > 120)).any():
        problems.append("SOH_model outside (0, 120]")
    return problems
