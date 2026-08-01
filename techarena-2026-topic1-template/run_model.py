#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  FRAMEWORK - DO NOT EDIT. The automated evaluation calls exactly this
#  (three arguments, same structure as last year):
#
#    python run_model.py --model train --input "\path\to\dataset"   --output-dir "\path\to\output"
#    python run_model.py --model test  --input "\path\to\input.csv" --output-dir "\path\to\output"
#
#  --model train : --input is the dataset FOLDER. Loads all cells, calls
#                  YourModel.fit(cells), saves the fitted model
#                  automatically to ./model_state/ (framework/persistence).
#  --model test / valid : --input is a CSV with the rows to predict
#                  (columns: Cycle, Temperature (degC), Current (C-rate),
#                  optionally SOH_true). Fills SOH_model (and errSOH where
#                  SOH_true is given) and writes output.csv to --output-dir.
# =============================================================================
import argparse
import sys
import time

import pandas as pd

from my_model import ActiveModel
from framework.data import load_cells
from framework.io import write_output
from framework.persistence import save_model, load_model

STATE_DIR = "model_state"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["train", "test", "valid"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    t0 = time.time()
    if args.model == "train":
        cells = load_cells(args.input)
        model = ActiveModel()
        model.fit(cells)
        save_model(model, STATE_DIR)
        print(f"[train] model saved to ./{STATE_DIR} "
              f"({time.time() - t0:.1f} s)")
    else:
        inp = pd.read_csv(args.input)
        need = ["Cycle", "Temperature (degC)", "Current (C-rate)"]
        missing = [c for c in need if c not in inp.columns]
        if missing:
            sys.exit(f"input csv is missing columns: {missing}")
        model = load_model(STATE_DIR, ActiveModel)
        path = write_output(inp, model, args.output_dir)
        print(f"[{args.model}] wrote {path} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
