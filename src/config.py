"""Caminhos e constantes compartilhadas pelo pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data set.csv"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RAW_PARQUET = DATA_DIR / "raw.parquet"

SEED = 42
TARGET = "CLASSI_FIN"
DATE_COLS = ["DT_NOTIFIC", "DT_SIN_PRI", "DT_NASC", "DT_INVEST", "DT_ENCERRA"]
