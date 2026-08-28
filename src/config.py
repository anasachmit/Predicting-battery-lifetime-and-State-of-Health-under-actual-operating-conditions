"""Chemins et constantes globales du banc d'essai.

Les repertoires de sortie sont PARAMETRABLES par variable d'environnement :

    BATTERY_RESULTS_DIR   ou vont results/  (figures, CSV, checkpoints)
    BATTERY_DATA_DIR      ou vont data/     (pool, parquets)

Necessaire parce que l'entrainement tourne sur une VM Colab dont le systeme de
fichiers est distinct : les sorties doivent pouvoir aller sur un Drive monte
sans qu'aucun chemin ne soit ecrit en dur.
"""
import os
from pathlib import Path

RANDOM_SEED = 20260101

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BATTERY_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
RESULTS = Path(os.environ.get("BATTERY_RESULTS_DIR", ROOT / "results"))
FIGURES = RESULTS / "figures"
RUNS_LOG = RESULTS / "runs.jsonl"

WHEELER_MAT = RAW / "wheeler" / "extractedData.mat"
SNL_PKL_DIR = RAW / "snl" / "pkl"
TARGET_DIR = DATA / "forfinetune"

# Capacite nominale des cellules cible du challenge (Huawei TechArena 2026, topic 1).
TARGET_Q_NOM_AH = 102.0

UNIFIED_SCHEMA = [
    "cell_id", "source", "cycle_n", "capacity_Ah", "SOH",
    "T_amb", "C_rate_chg", "C_rate_dchg", "DoD", "format", "Q_nom",
]

for _d in (RESULTS, FIGURES, DATA):
    _d.mkdir(parents=True, exist_ok=True)
