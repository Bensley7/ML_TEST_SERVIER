from typing import Dict, List
import sys
import argparse

import pytorch_lightning as pl

from molecular_ml.utils.io import read_cfg, load_model
from molecular_ml.utils.utils import get_molecule_bit_map_loader
from molecular_ml.models.model import MoleculeAttentionLSTM

def eval_model(model_path: str, val_data_path: str, config_file):
    
    #Read config file
    cfg = read_cfg(config_file)
    
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

    args = parser.parse_args()
    return args


def eval(opt):
    eval_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    eval_model(args.model_path,
               args.val_data_path,
               args.config_file)