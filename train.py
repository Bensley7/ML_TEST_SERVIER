import numpy as np
import os
from os import makedirs
from tqdm import trange
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch

from torch.optim.lr_scheduler import ExponentialLR

from utils.utils import get_data
from utils.evaluator import evaluate, evaluate_predictions
from utils.predictor import predict
from utils.trainer import batch_train
from utils.io import load_checkpoint, save_checkpoint, read_cfg
from models.utils import get_loss_func, build_lr_scheduler, build_optimizer
from models.model import MoleculeModel
from dataset.loader import MoleculeDataLoader

def train_model(
    train_data_path,
    val_data_path,
    test_data_path,
    config_file,
    opts,
    model_path="mol_models/",
    model_name="model1.pt",
) -> None:

    #Read config file
    cfg = read_cfg(config_file, opts)
    #Get data
    print('Loading data')
    train_data = get_data(
        path=train_data_path,
        cfg=cfg
        )
    
    train_data.reset()

    # Run training
    
    # Set pytorch seed for random initial weights
    torch.manual_seed(0)

    test_data = get_data(path=test_data_path,
                        cfg=cfg)
    val_data = get_data(path=val_data_path,
                        cfg=cfg)

    print(f'Total size = {(len(train_data) + len(val_data) + len(test_data)):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    dataset_type = "classification"
    if len(cfg.DATA_DIGEST.labels_name) > 1:
        dataset_type = "multiclass"

    # Get loss function
    loss_func = get_loss_func(cfg.MODEL.loss_function, dataset_type)

    # Set up test set evaluation
    _, test_targets = test_data.smiles(), test_data.targets()

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=cfg.MODEL.TRAINING.batch_size,
        num_workers=cfg.MODEL.num_workers,
        class_balance=cfg.MODEL.TRAINING.class_balance,
        shuffle=True
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=cfg.MODEL.TRAINING.batch_size,
        num_workers=cfg.MODEL.num_workers,
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=cfg.MODEL.TRAINING.batch_size,
        num_workers=cfg.MODEL.num_workers,
    )

    if cfg.MODEL.TRAINING.class_balance:
        print(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)

    # Create Model dirs to save
    makedirs(model_path)

    # Load/build model
    print(f'Building model')

    model = MoleculeModel(cfg)    
    model = model.to(device)

    # Optimizers
    optimizer = build_optimizer(model, cfg.MODEL.TRAINING)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, cfg.MODEL.TRAINING, len(train_data))

    # Run training
    best_score =  -float('inf')
    best_epoch, n_iter = 0, 0
    for epoch in trange(cfg.MODEL.TRAINING.nb_epochs):
        print(f'Epoch {epoch}')
        n_iter = batch_train(
            model=model,
            data_loader=train_data_loader,
            dataset_type = dataset_type,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=cfg.MODEL.TRAINING.batch_size,
            log_frequency=cfg.MODEL.TRAINING.log_frequency,
            n_iter=n_iter,
        )
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        val_scores = evaluate(
            model=model,
            data_loader=val_data_loader,
            metrics=cfg.MODEL.TRAINING.metrics,
            dataset_type=dataset_type,
        )

        for metric, scores in val_scores.items():
            # Average validation score\
            mean_val_score = np.mean(scores, axis=None)
            print(f'Validation {metric} = {mean_val_score:.6f}')

        # Save model checkpoint if improved validation score
        mean_val_score = np.mean(val_scores[cfg.MODEL.TRAINING.metric], axis=None)
        if  mean_val_score > best_score:
            best_score, best_epoch = mean_val_score, epoch
            save_checkpoint(os.path.join(model_path, model_name), model)

    # Evaluate on test set using model with best validation score
    print(f'Model best validation {cfg.MODEL.TRAINING.metric} = {best_score:.6f} on epoch {best_epoch}')
    model = load_checkpoint(os.path.join(model_path, model_name), cfg, device=device)

    test_preds = predict(
        model=model,
        data_loader=test_data_loader,
    )

    model_scores = evaluate_predictions(
        preds=test_preds,
        targets=test_targets,
        metrics=cfg.MODEL.TRAINING.metrics,
        dataset_type=dataset_type,
    )

    for metric, scores in model_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = np.mean(scores, axis=None)
        print(f'Overall Test {metric} = {mean_ensemble_test_score:.6f}')


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--train_data_path", help="train dataset path", type=str)
    parser.add_argument("--val_data_path", help="valid dataset path", type=str)
    parser.add_argument("--test_data_path", help="test dataset path", type=str)

    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model_path', type=str, default="mol_models/", help='directory to save the model')
    parser.add_argument('--model_name', type=str, default = "model1.pt", help='pytorch model name')

    args = parser.parse_args()
    return args


def train(opt):
    train_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    train(args)
