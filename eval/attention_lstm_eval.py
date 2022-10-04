from typing import Dict, List
import sys
import argparse

import torch
import pytorch_lightning as pl

sys.path.append("../")
from utils.io import read_cfg, load_model
from utils.utils import get_molecule_bit_map_loader
from models.model import MoleculeAttentionLSTM

def eval_model(model_path: str, val_data_path: str, config_file, opts):
    
    #Read config file
    cfg = read_cfg(config_file, opts)
    
    print('Loading data')
    val_data_loader = get_molecule_bit_map_loader(
        path=val_data_path,
        cfg=cfg
        )

    #Load Data
    model = MoleculeAttentionLSTM(metrics = cfg.MODEL.TRAINING.metrics)
    model = load_model(model, model_path)
    model.eval()
    model.eval()
    
    trainer = pl.Trainer(logger=None)
    res = trainer.test(model, dataloaders=val_data_loader)

    print(res)

    return res


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--model_path', type=str, required = True, help='weight path of the model')
    parser.add_argument("--val_data_path", help="dataset path with labels", type=str, required = True)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def eval(opt):
    eval_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    eval(args)