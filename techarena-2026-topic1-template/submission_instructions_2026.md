TechArena 2026 - Challenge 1 Submission Instructions

> **Before you distribute this document, fill in / confirm the items marked
> `<<TODO: ...>>`** - these are event-logistics details (contacts, deadline,
> platform links) that aren't derivable from the codebase and were left as
> placeholders rather than guessed.

# 📋 Overview

Welcome to TechArena 2026! Challenge 1 asks you to build a **battery
State-of-Health (SOH) fade forecasting model**: given measured data from a
set of released battery cells, predict the SOH%-vs-cycle curve for *any*
operating point (temperature 25-55 °C, C-rate 0.5-1.0 C) - including
combinations you never saw during training.

Unlike a from-scratch script, this year's submission is a **fixed
framework with one file you fill in**. You are not writing an entry point,
a CLI, or an output writer - all of that is provided and frozen so every
team is scored through the exact same harness. Your job is to implement two
Python methods. Please read this document fully before you start.

# 🎯 Submission Requirements

## Required Deliverables

Your submission is the **entire provided template folder**, with these
parts filled in / updated:

1. **Model implementation**: `my_model/model_template.py` - implement
   `fit()` and `predict_soh()` (see below).
2. **Active-model switch**: `my_model/__init__.py` - flip the import to
   point at your model instead of the example.
3. **Dependencies**: `requirements.txt` - every package your model needs,
   pinned versions recommended.
4. **Documentation**: `README.md` - replace the provided one with a
   description of your approach: method, any open datasets used (with
   citations and licenses), what transferred to this task, and your own
   validation methodology (what you held out, and why you trust your
   accuracy estimate at unseen operating points).
5. Optionally, any additional files your model needs (helper modules,
   pretrained weight files) - place them inside `my_model/`.

**Do not modify** `run_model.py`, anything under `framework/`, or
`validate_submission.py`. These implement the fixed evaluation interface
and are identical across all teams; the organizers run your submission
through their own untouched copies of these files regardless of what you
submit, so local edits to them have no effect on scoring and only risk
breaking your own local testing.

## The Two Functions You Implement

```python
class MyModel:
    def fit(self, cells):
        # cells: list of Cell objects, one per training cell. Learn
        # whatever you need and store it on self.* - the framework
        # pickles your whole fitted object after this call.
        ...

    def predict_soh(self, temperature_degC, c_rate):
        # Return a 1D array of SOH% for cycles 1, 2, 3, ... at this
        # (temperature, C-rate). Returning up to 12000 values is always
        # safe - the framework indexes only the cycles it needs.
        ...
```

At prediction time your model receives **only** `(temperature_degC,
c_rate)` - no measured signals for the target point, because hidden
evaluation cells have no data available by definition. Anything you learn
from raw signals during `fit()` (voltage/temperature curves, ICA / dQ-dV,
DC resistance, CV-phase timing, pretrained encoders, ...) must be captured
into your model's parameters, since `fit()` runs once and `predict_soh()`
is all that's called afterward, possibly in a separate process.

# 📁 Project Structure

Your submission zip must preserve this exact structure:

```
<TeamName>_Challenge1.zip
|
|-- my_model/                <<< YOUR CODE LIVES HERE
|     model_template.py        your implementation (fit + predict_soh)
|     model_example.py         reference baseline - do not need to touch
|     __init__.py               ONE line selects which model runs
|     [any extra files you add: helper modules, pretrained weights, ...]
|
|-- framework/                DO NOT EDIT (data loading, output, saving)
|-- run_model.py               DO NOT EDIT - fixed evaluation interface
|-- validate_submission.py     run this before every submission
|-- sample_data/               tiny synthetic cells for local dry runs
|-- sample_input.csv           example evaluation input for local dry runs
|-- requirements.txt           your dependencies
|-- README.md                  your approach writeup
```

# 🚨 Critical Requirements

- **Only `my_model/` is yours to edit.** Everything else must stay exactly
  as provided.
- `my_model/__init__.py` must import your class as `ActiveModel` before you
  submit - the example is wired in by default so the template runs
  out-of-the-box; forgetting to flip this switch means the organizers score
  the example baseline, not your model.
- `predict_soh()` must return values in `(0, 120]`, all finite (no
  NaN/inf), and long enough to cover every cycle the evaluation might
  request (12000 values is always safe).
- No absolute or team-specific file paths - your model must run unchanged
  on the organizers' machine, not just yours.
- Pretrain on open data **offline**, in your own scripts, and ship the
  resulting weights inside `my_model/`. Don't assume your `fit()` can reach
  the internet during official scoring.
- Fix your random seeds so your submission is reproducible.

# 💻 Technical Specifications

## Python Requirements

- **Python version**: `<<TODO: confirm minimum supported version - the
  pinned dependencies (numpy>=1.24, pandas>=2.0) suggest Python 3.9+>>`.
- **Core dependencies already used by the framework**: `numpy`, `pandas`
  (already listed in `requirements.txt` - add anything else your model
  needs on top).

## Data You're Given

`fit(cells)` receives a list of `Cell` objects, one per released training
cell:

| Attribute | Meaning |
|---|---|
| `cell.cell_id` | e.g. `"102Ah_45degC_1C_cell1"` |
| `cell.temperature_degC` | e.g. `45` |
| `cell.c_rate` | e.g. `1.0` |
| `cell.soh` | DataFrame `[cycle_number, soh_percent]` - official label definition, already cleaned |
| `cell.time_series()` | full raw signals (lazy-loaded): `cycle_number, step_type, time_in_cycle_s, voltage_V, current_A, temperature_C, step_capacity_Ah, absolute_time` |

SOH label definition (identical to the hidden ground truth used for
scoring): `SOH(%) = discharge capacity of the cc_discharge step / 102 Ah
nominal * 100`; implausible (>115% of nominal) and aborted partial cycles
carry no label (`NaN`).

Both modelling styles are supported through the same interface:
label-curve models can use `cell.soh` alone; signal-based models can pull
features from `cell.time_series()`. See the worked snippets in
`my_model/model_template.py` for examples of ICA/dQ-dV, DC resistance, and
CV-phase-time extraction.

## Hardware & Runtime Limits

There is no hard cap on model complexity, but the official scoring run
enforces generous wall-clock limits per submission: **up to 2 hours** for
`--model train` on the full released dataset, and **up to 30 minutes per
cell** for `--model test`. Those limits are measured on the machine below -
size your training pipeline against it before you submit:

- **OS**: Windows Server 2022
- **GPU**: 1x NVIDIA RTX PRO (Blackwell) Workstation Edition - 94 GB
  dedicated VRAM (up to 350 GB reported total with shared system memory)
- **CPU**: 96 cores, 554 GB RAM

If your training pipeline cannot comfortably finish in under 2 hours on
hardware like this, don't retrain it from scratch inside `fit()` - pretrain
offline and ship the resulting weights inside `my_model/` instead.

# 🔧 Implementation Guidelines

## Where to Start

Read, in order:

1. `my_model/model_example.py` - a complete, working reference model (a
   power-law fade curve per cell, extrapolated across temperature/C-rate
   with an Arrhenius-style fit). Run it as-is (it's the default) to see the
   whole pipeline work end-to-end before you change anything.
2. `my_model/model_template.py` - your file, with both label-curve and
   signal-based approaches sketched in comments.
3. `my_model/__init__.py` - the one-line switch between the two.

## Best Practices

✅ **DO**

- Use relative paths only for any files you add under `my_model/`.
- Handle both training conditions (25-55 °C, 0.5-1.0 C) and points strictly
  between them - your model will be scored on unseen combinations.
- Build your **own validation scheme** - no scoring script is provided to
  teams, and how you estimate generalization to unseen operating points is
  part of the assessed work.
- Keep everything you learn on `self.*` inside your model class - that
  whole object is what gets pickled.
- Test with `python validate_submission.py` **inside a fresh virtual
  environment** installed only from `requirements.txt`, after every
  meaningful change (see [Step 2](#step-2-validate-locally)).
- Fix random seeds.

❌ **DON'T**

- Don't edit `framework/`, `run_model.py`, or `validate_submission.py`.
- Don't rely on `cell.time_series()` or `cell.soh` being available at
  *prediction* time - they're only available during `fit()`.
- Don't assume internet access is available when your submission is
  scored.
- Don't leave `raise NotImplementedError(...)` in `model_template.py` if
  you're pointing `ActiveModel` at it.
- Don't hardcode absolute or team-specific paths.

## Sample `requirements.txt`

```
numpy>=1.24
pandas>=2.0
# add anything else your model needs, e.g.:
# scikit-learn>=1.3
# torch>=2.1
```

# 📝 Submission Process

## Step 1: Implement

Fill in `my_model/model_template.py`, then point `my_model/__init__.py` at
it:

```python
# from .model_example import ExampleModel as ActiveModel   # reference baseline
from .model_template import MyModel as ActiveModel          # <-- your model
```

## Step 2: Validate Locally

Always validate inside a **fresh virtual environment** built only from
your own `requirements.txt` - not whatever environment you happened to
develop in. Official scoring runs in its own clean environment, seeded
only from `requirements.txt`; if your model quietly depends on a package
that's already installed globally on your machine but missing from
`requirements.txt`, it will pass here and fail there.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python validate_submission.py
```

This must print `SUBMISSION READY`. It checks required files, that
`requirements.txt` is readable, runs a real `train` + `test` dry run
against `sample_data/`/`sample_input.csv`, and checks `output.csv`'s
schema. It writes `validation_report.txt` with the full detail.

If it fails with an import error here, that's the fresh environment
doing its job - add the missing package to `requirements.txt`, reinstall,
and rerun, rather than installing it globally and moving on.

> Note: `sample_data/` is only 2 tiny synthetic cells - a `SUBMISSION
> READY` result confirms your code runs end-to-end and produces
> well-formed output, **not** that your model is scientifically good.
> Validate prediction quality against your own held-out data.

Optionally, run a fuller local smoke test against a real dataset:

```bash
python validate_submission.py --data-dir <path to a real dataset folder>
```

## Step 3: Package

1. Zip the entire submission folder.
2. Name it: `<TeamName>_Challenge1.zip`.
3. Extract the zip somewhere clean and re-run `validate_submission.py`
   inside it, to confirm nothing was left out.

## Step 4: Submit

`<<TODO: submission platform / upload link>>`. There is no limit on how
many times you may resubmit before the deadline
(`<<TODO: deadline date/time and timezone>>`).

# ⚡ Automated Validation & Scoring

## Local Validation (`validate_submission.py`)

Runs the same structural checks as the intake step of the official
pipeline:

- Required files and folder structure present.
- `requirements.txt` readable.
- Real `train` dry run on `sample_data/`.
- Real `test` dry run on `sample_input.csv`.
- `output.csv` schema and value-range checks (finite, in `(0, 120]`, one
  row per input row).

## Official Scoring (server-side)

Submissions that pass intake are trained from scratch and evaluated on
the organizers' machine against hidden ground-truth cells: sibling
cells of the released conditions as well as cells at further operating
points within 25-55 °C x 0.5-1.0 C. Scoring uses a fixed composite
error metric - identical for every team - reported relative to the
provided reference baseline (`model_example.py`): **baseline = 1.0,
lower is better**. This relative score is what the public leaderboard
shows. The leaderboard score, your report, and the reproducibility of
your submission together determine the final ranking.

The exact metric definitions are deliberately not published. Designing
and justifying **your own validation methodology** - what you hold
out, and how you estimate accuracy at operating points you have no
data for - is part of the challenge and is assessed through your
report. A model tuned to look good only on the released cells will not
generalize; build a validation scheme you can defend.

In addition, every submission is automatically re-trained in a
reduced-budget run (fewer cells, early-life cycles only). Models that
keep their accuracy with less proprietary data earn a data-efficiency
bonus on top of their score - nothing to implement on your side.

# 🚨 Common Issues and Solutions

**"no cells found under `<path>`"**
Your training folder's cell subfolders must match the naming pattern
`102Ah_<temperature>degC_<0p5C|1C>_cell<N>` (e.g.
`102Ah_45degC_1C_cell3`). Check your `--input` path points at the folder
*containing* those subfolders, not one of the subfolders itself.

**"input csv is missing columns: [...]"**
The `test`/`valid` input CSV must contain `Cycle`, `Temperature (degC)`,
`Current (C-rate)` (an optional `SOH_true` column is allowed for local
scoring but not required).

**"`predict_soh(T, C)` returned N values but cycle M was requested - return
a longer trajectory"**
Your `predict_soh()` array is too short for the cycles being asked for.
Return up to 12000 values (cycles 1..12000) to be safe - the framework
only ever reads the indices it needs.

**"N SOH_model values are NaN/inf"**
Your `predict_soh()` returned non-finite values somewhere in the requested
range. Check for divide-by-zero or log-of-non-positive terms at extreme
temperature/C-rate combinations near the edges of 25-55 °C / 0.5-1.0 C.

**"SOH_model values must be in (0, 120]"**
Clip your output to a sane physical range, e.g.
`np.clip(soh, 0.5, 119.9)`, as the reference model does.

**Dry run in `validate_submission.py` times out**
The 900s (train) / 300s (test) limits inside `validate_submission.py` are
**local sanity limits only, not the competition limit** - they exist so a
hanging script doesn't block you locally. The real limits used for
official scoring are 2 hours (train) / 30 minutes per cell (test) - see
[Hardware & Runtime Limits](#hardware--runtime-limits).

**"It passed on my machine but failed during scoring" / `ModuleNotFoundError`
during official evaluation**
`validate_submission.py` runs your model with whatever Python environment
you invoke it from - it does not check that `requirements.txt` is
complete. If a package is already installed globally on your machine
(common for anything ML-related) but missing from `requirements.txt`,
validation passes for you but fails in the organizers' clean environment.
Always validate inside a fresh virtual environment installed only from
your own `requirements.txt` - see [Step 2](#step-2-validate-locally).

**Model loads but predictions look identical across every temperature/
C-rate**
If your model can't distinguish operating points (e.g. because your
fit only saw 1-2 training conditions), it will degrade to a
condition-independent prediction. This is a modelling issue, not a
framework bug - make sure your `fit()` genuinely uses
`cell.temperature_degC` / `cell.c_rate` as inputs to whatever
extrapolation you build.

# 📋 Submission Checklist

Before submitting, verify:

- [ ] `my_model/__init__.py` points `ActiveModel` at your model, not the example
- [ ] `python validate_submission.py` prints `SUBMISSION READY` **when run
      inside a fresh virtual environment installed only from
      `requirements.txt`**
- [ ] `requirements.txt` lists every package your model needs
- [ ] Random seeds are fixed
- [ ] `README.md` describes your approach, datasets used, what transferred, and your validation methodology
- [ ] No absolute or team-specific file paths anywhere in `my_model/`
- [ ] `predict_soh()` returns finite values in `(0, 120]` for every point in
      25-55 °C x 0.5-1.0 C, not just the training conditions
- [ ] Any pretrained weights are shipped inside `my_model/`, not downloaded at runtime
- [ ] Zip is named `<TeamName>_Challenge1.zip` and was tested from a clean extraction

Good luck!

TechArena 2026 Organizing Committee
Last updated: 2026-07-29
