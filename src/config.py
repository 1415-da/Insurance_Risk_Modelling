"""Project-wide configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "insurance_claims.csv"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

TARGET_COL = "claim_status"
DROP_COLS = ["policy_id"]
SENSITIVE_CANDIDATES = ["region_code", "segment", "age_group", "fuel_type"]

AGE_BINS = [34, 45, 55, 65, 76]
AGE_LABELS = ["35-45", "46-55", "56-65", "66-75"]

MIN_REGION_COUNT = 200
