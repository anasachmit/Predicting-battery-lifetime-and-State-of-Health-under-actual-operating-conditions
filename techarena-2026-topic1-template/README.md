# TechArena 2026 - Challenge 1: Team Submission Template

 You implement your model in `my_model/` - everything
else is provided and stays untouched.

```
team_submission_template/
|
|-- my_model/                <<< YOUR CODE LIVES HERE
|     model_template.py        the skeleton you fill in (fit + predict_soh)
|     model_example.py         a complete working reference baseline
|     __init__.py              ONE line selects which of the two runs
|
|-- framework/               DO NOT EDIT (data loading, output, saving)
|-- run_model.py             DO NOT EDIT - fixed evaluation interface
|-- validate_submission.py   run this before every submission
|-- sample_data/             tiny synthetic cells for the dry run
|-- sample_input.csv         example evaluation input for the dry run
|-- requirements.txt         add your dependencies here
|-- README.md                replace with a description of your approach
```

## Run model - 3 input arguments

```
python run_model.py --model train --input "\path\to\dataset"   --output-dir "\path\to\output"
python run_model.py --model test  --input "\path\to\input.csv" --output-dir "\path\to\output"
```

(`--model valid` is also accepted and behaves identically to `test` - it is
not a separate mode, just an accepted alias.)

`--model train`: `--input` is the dataset folder. The framework loads all
cells and calls your `fit(cells)`. Your fitted model is saved
automatically to `./model_state/` (pickle - works for numpy, scipy,
scikit-learn, PyTorch and plain Python; frameworks that cannot be
pickled add `save`/`load` methods, see `framework/persistence.py`).
You never create this file yourself.

`--model test` (or `valid`): `--input` is a CSV provided by the
evaluation pipeline with the rows to predict - you do not write it
yourself, `sample_input.csv` shows the format. The framework calls your
`predict_soh(temperature_degC, c_rate)` per operating point, fills the
requested rows and writes `output.csv`:

| Cycle | SOH_true | Temperature (degC) | Current (C-rate) | SOH_model | errSOH |
|------:|---------:|-------------------:|-----------------:|----------:|-------:|
| 1     | 100.0    | 25                 | 1                | 100.2     | 0.2    |
| 2     | 98.0     | 25                 | 1                | 98.9      | 0.9    |

`SOH_true`/`errSOH` stay empty where no truth is provided. The requested
operating points can lie anywhere in 25-55 degC x 0.5-1.0 C.

## The two functions you implement

```python
class MyModel:
    def fit(self, cells):
        # cells: list of Cell objects - per cell:
        #   cell.temperature_degC, cell.c_rate
        #   cell.soh            -> DataFrame [cycle_number, soh_percent]
        #   cell.time_series()  -> raw voltage/current/temperature signals
        ...

    def predict_soh(self, temperature_degC, c_rate):
        # return SOH % for cycles 1, 2, 3, ... as a 1D array
        # (12000 values is always safe - the framework picks what it needs)
        ...
```

Both modelling styles are supported through the same interface:
label-curve models use `cell.soh`; signal-based models use
`cell.time_series()` for anything derivable from the raw data - voltage
and temperature profiles, ICA / dQ-dV, DC resistance, CV-phase times,
and so on (worked snippets in `model_template.py`). All of this happens
in `fit()`: the evaluation input only ever asks for (Cycle, Temperature,
C-rate) rows, so everything your model learned from signals must live
inside the fitted model.

## Before you submit

1. Create a **fresh virtual environment** and install only from
   `requirements.txt` - this is the only reliable way to catch missing or
   unpinned dependencies before they show up as failures during official
   scoring, which runs in its own clean environment, not yours:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
2. With that environment active: `python validate_submission.py` - must
   print SUBMISSION READY (optionally `--data-dir <real dataset>` for a
   full smoke run). If it fails here because of a missing package, add it
   to `requirements.txt` and reinstall - don't just install it globally
   and move on, or the same failure will happen again during scoring.
3. Replace this README with a short description of your approach: method,
   open datasets used (with citations and licenses), what transferred, and
   your own validation methodology.
4. `requirements.txt` lists every package you need. Fix your random seeds.
5. Zip the folder as `<TeamName>_Challenge1.zip` and submit - as often as
   you like.

## Scoring

Official scoring runs server-side against hidden ground-truth cells
(sibling cells of the released conditions plus further operating points
within 25-55 degC x 0.5-1.0 C), with a fixed composite error metric
relative to the reference baseline: **baseline = 1.0, lower is better**.
The metric definitions are not published - designing and reporting your
own validation methodology is part of the task. Leaderboard score, report
and reproducibility together determine the final ranking. Every submission
is additionally re-trained in an automatic reduced-budget run (fewer
cells, early-life cycles only) for a data-efficiency bonus - nothing to
implement on your side. Details in `submission_instructions_2026.md`.

## Runtime and size limits

There is no hard cap on model complexity, but the official scoring run
enforces generous wall-clock limits per submission: **up to 2 hours** for
`--model train` on the full released dataset, and **up to 30 minutes per
cell** for `--model test`. Those limits are measured on the machine below -
size your training pipeline against it before you submit:

- **OS**: Windows Server 2022
- **GPU**: 1x NVIDIA RTX PRO (Blackwell) Workstation Edition - 94 GB
  dedicated VRAM (up to 350 GB reported total with shared system memory)
- **CPU**: 96 cores, 554 GB RAM

If your training pipeline (e.g. heavy pretraining) cannot comfortably
finish in under 2 hours on hardware like this, don't retrain it from
scratch inside `fit()` - pretrain offline in your own scripts and ship the
resulting weights inside `my_model/` instead, as this README already
recommends.
