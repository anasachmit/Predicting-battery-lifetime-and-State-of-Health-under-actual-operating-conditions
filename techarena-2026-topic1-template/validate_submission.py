#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_submission.py - run this BEFORE you submit.

Performs the same checks as the automated evaluation intake:

  1. required files and folder structure
  2. requirements.txt is readable
  3. dry run:  train on ./sample_data (time limit)
  4. dry run:  predict on ./sample_input.csv (time limit)
  5. output.csv schema and trajectory rules
  6. optional: --data-dir <real dataset> full smoke test

Prints a PASS/FAIL report and writes validation_report.txt.
Exit code 0 = ready to submit.

NOTE: sample_data/ ships only 2 tiny synthetic cells, which is enough to
smoke-test that your code runs end-to-end and produces well-formed output,
but NOT enough to judge prediction quality (e.g. the reference model needs
>= 3 training cells before it fits any temperature/C-rate dependence at
all). A PASS here means "my code is submission-ready", not "my model is
any good" - validate scientific quality against your own held-out data.
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# Local sanity timeouts so a hanging dry run does not block you - these
# are NOT the competition limits. Official scoring enforces up to 2 h for
# --model train and up to 30 min per cell for --model test (see README).
TRAIN_LIMIT_S = 900
PREDICT_LIMIT_S = 300

REQUIRED = ["run_model.py", "requirements.txt", "README.md",
            "sample_input.csv",
            os.path.join("my_model", "model_template.py"),
            os.path.join("my_model", "model_example.py"),
            os.path.join("my_model", "__init__.py"),
            os.path.join("framework", "data.py"),
            os.path.join("framework", "io.py"),
            os.path.join("framework", "persistence.py")]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def run(cmd, limit):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           timeout=limit)
        return p.returncode == 0, time.time() - t0, \
            (p.stdout + p.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, limit, f"time limit of {limit} s exceeded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="optional: real dataset folder for a full smoke run")
    args = ap.parse_args()

    print("TechArena 2026 - Submission Validation Report")
    print("=" * 60)

    print("\n[1/5] file structure")
    for f in REQUIRED:
        check(f, os.path.exists(os.path.join(HERE, f)))

    print("\n[2/5] requirements.txt")
    try:
        reqs = [l.strip() for l in
                open(os.path.join(HERE, "requirements.txt"))
                if l.strip() and not l.startswith("#")]
        check("readable, %d package(s)" % len(reqs), True)
    except Exception as e:
        check("readable", False, str(e))

    print("\n[3/5] dry run: train on sample_data")
    print("      (structural smoke test only - sample_data is 2 tiny")
    print("       synthetic cells, not enough to judge model quality)")
    ok, dt, log = run([PY, "run_model.py", "--model", "train",
                       "--input", "sample_data",
                       "--output-dir", ".val_out"], TRAIN_LIMIT_S)
    check(f"train finished ({dt:.1f} s)", ok, "" if ok else log)

    print("\n[4/5] dry run: test on sample_input.csv")
    ok2, dt, log = run([PY, "run_model.py", "--model", "test",
                        "--input", "sample_input.csv",
                        "--output-dir", ".val_out"], PREDICT_LIMIT_S)
    check(f"test finished ({dt:.1f} s)", ok2, "" if ok2 else log)

    print("\n[5/5] output.csv format")
    pred_path = os.path.join(HERE, ".val_out", "output.csv")
    if os.path.exists(pred_path):
        from framework.io import check_output_file
        inp = pd.read_csv(os.path.join(HERE, "sample_input.csv"))
        problems = check_output_file(pred_path, inp)
        check("required columns and value ranges", not problems,
              "; ".join(problems[:4]))
        if not problems:
            dfp = pd.read_csv(pred_path)
            for (T, C), g in dfp.groupby(["Temperature (degC)",
                                          "Current (C-rate)"]):
                print(f"        {T} degC / {C}C: {len(g)} rows, "
                      f"SOH_model {g['SOH_model'].iloc[0]:.1f} -> "
                      f"{g['SOH_model'].iloc[-1]:.1f}")
    else:
        check("output.csv exists", False)

    if args.data_dir:
        print("\n[extra] full smoke run on real dataset")
        ok, dt, log = run([PY, "run_model.py", "--model", "train",
                           "--input", args.data_dir,
                           "--output-dir", ".val_out_full"], 3600)
        check(f"full train finished ({dt:.0f} s)", ok, "" if ok else log)
        ok, dt, log = run([PY, "run_model.py", "--model", "test",
                           "--input", "sample_input.csv",
                           "--output-dir", ".val_out_full"], 600)
        check(f"full test finished ({dt:.0f} s)", ok, "" if ok else log)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 60)
    verdict = ("SUBMISSION READY - all checks passed" if n_fail == 0
               else f"SUBMISSION NOT READY - {n_fail} check(s) failed")
    print(verdict)
    with open(os.path.join(HERE, "validation_report.txt"), "w") as fh:
        fh.write("TechArena 2026 - Submission Validation Report\n")
        for name, ok, detail in results:
            fh.write(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}\n")
        fh.write(verdict + "\n")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
