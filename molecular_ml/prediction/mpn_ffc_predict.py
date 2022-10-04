from typing import List
import sys
import argparse

import torch
from tqdm import tqdm

from molecular_ml.dataset.loader import MoleculeDataLoader, MoleculeDataset
from molecular_ml.models.model import MoleculeModel
from molecular_ml.models.utils import activate_dropout
from molecular_ml.utils.utils import get_data_from_smile
from molecular_ml.utils.io import read_cfg, load_checkpoint

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

def predict_single(model_path: str, smile: str, config_file: str=""):
    #Read config file
    cfg = read_cfg(config_file)
    
    print('Loading data')
    pred_data = get_data_from_smile(smile, cfg)
    
    pred_data.reset()

    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)
    #Load Data
    model = load_checkpoint(model_path, cfg, device=device)
    model.eval()
    
    print("model loaded")

    pred_data_loader = MoleculeDataLoader(
        dataset=pred_data,
        batch_size=1,
        num_workers=1,
    )
    out = predict(model, pred_data_loader)
    
    res = None
    if isinstance(out[0][0], list):
        res = {label_name: (prob > 0.5) * 1 for label_name, prob 
                            in zip(cfg.DATA_DIGEST.labels_name, out[0][0])}
    else:
        res = {cfg.DATA_DIGEST.labels_name[0]: (out[0][0] > 0.5) * 1}

    return res


def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--model_path', type=str, help='weight path of the model')
    parser.add_argument("--smile", help="smile of a molecule", type=str)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    predict_single(args.model_path, args.smile, args.config_file)

