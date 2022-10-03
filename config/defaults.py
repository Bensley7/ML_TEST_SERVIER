from yacs.config import CfgNode as CN
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]

_C = CN()

# -----------------------------------------------------------------------------
# DATA DIGEST CONFIG
# -----------------------------------------------------------------------------
_C.DATA_DIGEST = CN()
_C.DATA_DIGEST.smiles_name = "smiles"
_C.DATA_DIGEST.labels_name = ["P1"] # Default: ["P1"]
_C.DATA_DIGEST.val_ratio = 0.2
_C.DATA_DIGEST.test_ratio = 0.1

# -----------------------------------------------------------------------------
# MOLECULAR MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.name = "attention_lstm"
_C.MODEL.device = "cuda:0"
_C.MODEL.num_workers = 4
_C.MODEL.activation = "ReLU"
_C.MODEL.loss_function = "binary_cross_entropy"
_C.MODEL.ffn_num_layers = 2
_C.MODEL.ffn_hidden_size = 300
_C.MODEL.hidden_size = 300
_C.MODEL.dropout = 0.0
_C.MODEL.depth = 3
_C.MODEL.bias = False
_C.MODEL.number_of_molecules = 1
_C.MODEL.aggregation = "mean" # sum or norm as options too
_C.MODEL.aggregation_norm = 100

# -----------------------------------------------------------------------------
# TRAINING OPTIMIZATION CONFIG
# -----------------------------------------------------------------------------
_C.MODEL.TRAINING = CN()
_C.MODEL.TRAINING.batch_size = 120
_C.MODEL.TRAINING.nb_epochs = 30
_C.MODEL.TRAINING.class_balance = False
_C.MODEL.TRAINING.init_lr = 0.0001
_C.MODEL.TRAINING.init_lr = 0.0001
_C.MODEL.TRAINING.num_lrs = 1
_C.MODEL.TRAINING.max_lr = 0.001
_C.MODEL.TRAINING.final_lr = 0.0001
_C.MODEL.TRAINING.warmup_epochs = 2
_C.MODEL.TRAINING.log_frequency = 10
_C.MODEL.TRAINING.metric = "accuracy"
_C.MODEL.TRAINING.metrics = ["accuracy", "prc-auc", "f1"]
_C.MODEL.TRAINING.max_len = 64
_C.MODEL.TRAINING.lr = 1e-4

