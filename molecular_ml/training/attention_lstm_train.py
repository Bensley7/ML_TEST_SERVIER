import argparse
import sys
from os import makedirs

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from molecular_ml.utils.io import read_cfg
from molecular_ml.models.model import MoleculeAttentionLSTM
from molecular_ml.utils.utils import get_molecule_bit_map_loader


def train_model(
    train_data_path,
    val_data_path,
    test_data_path,
    config_file,
    model_dir="./attention_lstm/",
    model_name="attention_lstm1"
    ):
    #Read config file
    cfg = read_cfg(config_file)
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

    device = 'cpu'
    if 'cuda' in cfg.MODEL.device:
        device = 'gpu'

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=model_name,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.MODEL.TRAINING.nb_epochs,
        logger=None,
        callbacks=[checkpoint_callback],
        accelerator=device,
        enable_progress_bar=True,
    )

    model = MoleculeAttentionLSTM(lr=cfg.MODEL.TRAINING.lr,
                                  str_loss_func = cfg.MODEL.loss_function,
                                  metrics = cfg.MODEL.TRAINING.metrics,   
                                 )

    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--train_data_path", help="train dataset path", type=str)
    parser.add_argument("--val_data_path", help="valid dataset path", type=str)
    parser.add_argument("--test_data_path", help="test dataset path", type=str)

    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument('--model_dir', type=str, default="mol_models/", help='directory to save the model')
    parser.add_argument('--model_name', type=str, default = "model1", help='pytorch model name')

    args = parser.parse_args()
    return args


def train(opt):
    train_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    train(args)
