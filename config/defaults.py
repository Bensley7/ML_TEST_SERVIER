from yacs.config import CfgNode as CN
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]

_C = CN()

# -----------------------------------------------------------------------------
# DATA DIGEST CONFIG
# -----------------------------------------------------------------------------
_C.DATA_DIGEST = CN()
_C.DATA_DIGEST.labels_name = ["P1", "P2"] # Default: ["P1"]
_C.DATA_DIGEST.val_ratio = 0.2
_C.DATA_DIGEST.test_ratio = 0.1