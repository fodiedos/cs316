"""Paths, column names, dtypes and experiment settings in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "crop_recommendation.csv"

FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

NUTRIENTS = ["N", "P", "K"]

CLIMATE = ["temperature", "humidity", "ph", "rainfall"]

FEATURES = NUTRIENTS + CLIMATE

TARGET = "label"

# int16 fits N/P/K (0-205), float32 is precise enough for climate values,
# and category stores each crop name once instead of on every row.
OPTIMISED_DTYPES = {
    "N": "int16",
    "P": "int16",
    "K": "int16",
    "temperature": "float32",
    "humidity": "float32",
    "ph": "float32",
    "rainfall": "float32",
    "label": "category",
}

RANDOM_STATE = 42      # fixed seed for repeatable results
TEST_SIZE = 0.20
CV_FOLDS = 5

TOP_K = 3              # crops the re-ranker may choose from
MIN_PROB = 0.10        # minimum probability to count as suitable
ALPHA = 0.5            # weight of suitability vs. resource cost
