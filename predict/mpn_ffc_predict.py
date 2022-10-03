from typing import List
import sys
import argparse

import torch
from tqdm import tqdm
import numpy as np

sys.path.append("../")
from dataset.loader import MoleculeDataLoader, MoleculeDataset
from models.model import MoleculeModel
from models.utils import activate_dropout
from utils.utils import get_data_from_smile
from utils.io import read_cfg, load_checkpoint

def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`MoleculeModel`.
    :param data_loader: A :class:`MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param dropout_prob: For use during uncertainty prediction only. The propout probability used in generating a dropout ensemble.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()
    
    # Activate dropout layers to work during inference for uncertainty estimation
    if dropout_prob > 0.0:
        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)
        model.apply(activate_dropout_)

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

def predict_from_model(model_path: str, smile: str, config_file: str, opts):
    #Read config file
    cfg = read_cfg(config_file, opts)
    
    print('Loading data')
    pred_data = get_data_from_smile(smile, cfg)
    
    pred_data.reset()

    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)
    #Load Data
    model = load_checkpoint(model_path, device=device)
    model.eval()
    
    dataset_type = "classification"
    if len(cfg.DATA_DIGEST.labels_name) > 1:
        dataset_type = "multiclass"

    pred_data_loader = MoleculeDataLoader(
        dataset=pred_data,
        batch_size=1,
        num_workers=1,
    )
    out = predict(model, pred_data_loader)
    print("Property of smile ", smile, "is ", out[0])


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--model_path', type=str, required = True, help='weight path of the model')
    parser.add_argument("--smile", help="smile of a molecule", type=str, required = True)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def pred(opt):
    predict_from_model(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    eval(args)

