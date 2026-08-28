"""Journalisation append-only de chaque run dans results/runs.jsonl."""
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

from .config import RUNS_LOG, RANDOM_SEED


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def log_run(step, params=None, metrics=None, notes=None):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "seed": RANDOM_SEED,
        "git": _git_rev(),
        "python": platform.python_version(),
        "params": params or {},
        "metrics": metrics or {},
        "notes": notes,
    }
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
