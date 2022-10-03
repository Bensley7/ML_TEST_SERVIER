from typing import Dict, List
import sys
import argparse

import torch
import numpy as np

sys.path.append("../")
from dataset.loader import MoleculeDataLoader
from models.model import MoleculeModel
from predict.mpn_ffc_predict import predict
from utils.utils import evaluate_predictions
from utils.io import load_checkpoint, read_cfg
from utils.utils import get_data

def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             metrics: List[str],
             dataset_type: str) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`MoleculeModel`.
    :param data_loader: A :class:`MoleculeDataLoader`.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    preds = predict(
        model=model,
        data_loader=data_loader,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        metrics=metrics,
        dataset_type=dataset_type,
    )

    return results


def eval_model(model_path, val_data_path, config_file, opts):
    
    #Read config file
    cfg = read_cfg(config_file, opts)
    
    print('Loading data')
    val_data = get_data(
        path=val_data_path,
        cfg=cfg
        )
    val_data.reset()

    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)
    #Load Data
    model = load_checkpoint(model_path, cfg, device=device)
    model.eval()
    
    dataset_type = "classification"
    if len(cfg.DATA_DIGEST.labels_name) > 1:
        dataset_type = "multiclass"

    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=cfg.MODEL.TRAINING.batch_size,
        num_workers=cfg.MODEL.num_workers,
    )

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