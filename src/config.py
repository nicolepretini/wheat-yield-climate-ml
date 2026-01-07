from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"

OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
MODEL_DIR = OUTPUTS_DIR / "models"
TABLE_DIR = OUTPUTS_DIR / "tables"

TARGET = "yield_kg_ha"
BASE_FEATURES = ["tas_gs_mean", "pr_gs_sum"]
RANDOM_STATE = 42
