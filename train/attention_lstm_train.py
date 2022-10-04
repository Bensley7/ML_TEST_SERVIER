import argparse
from ast import arg
import sys
import os
from os import makedirs

from torch import cpu, device

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("../")
from utils.io import read_cfg
from models.model import MoleculeAttentionLSTM
from utils.utils import get_molecule_bit_map_loader


def train_model(
    train_data_path,
    val_data_path,
    test_data_path,
    config_file,
    opts,
    model_dir="./attention_lstm/",
    model_name="attention_lstm1.pt"
    ):
    #Read config file
    cfg = read_cfg(config_file, opts)
    #Get data
    print('Loading data')

    train_loader = get_molecule_bit_map_loader(
        train_data_path,
        cfg
    )
    val_loader = get_molecule_bit_map_loader(
        val_data_path,
        cfg
    )
    test_loader = get_molecule_bit_map_loader(test_data_path, cfg)

    makedirs(model_dir)

    device = 'cpu'
    if 'cuda' in cfg.MODEL.device:
        device = 'gpu'
    trainer = pl.Trainer(
        max_epochs=cfg.MODEL.TRAINING.nb_epochs,
        logger=None,
        accelerator=device,
        enable_progress_bar=True,
    )

    model = MoleculeAttentionLSTM(lr=cfg.MODEL.TRAINING.lr,
                                  str_loss_func = cfg.MODEL.loss_function,
                                  metrics = cfg.MODEL.TRAINING.metrics,   
                                 )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(os.path.join(model_dir, model_name))
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--train_data_path", help="train dataset path", type=str)
    parser.add_argument("--val_data_path", help="valid dataset path", type=str)
    parser.add_argument("--test_data_path", help="test dataset path", type=str)

    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model_dir', type=str, default="mol_models/", help='directory to save the model')
    parser.add_argument('--model_name', type=str, default = "model1.pt", help='pytorch model name')

    args = parser.parse_args()
    return args


def train(opt):
    train_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    train(args)
