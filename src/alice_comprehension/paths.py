from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DATA_DIR = PROJECT_ROOT / "data" / "derived"

DEFAULT_RAW_ROOT = Path("/Users/yanyuwoo/Data/r")
DEFAULT_BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
DEFAULT_COMPREHENSION_SCORES = DEFAULT_RAW_ROOT / "comprehension-scores.txt"

