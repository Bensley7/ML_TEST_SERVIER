
import sys
import argparse

import torch

from molecular_ml.utils.utils import get_bit_map_loader_from_smile
from molecular_ml.utils.io import read_cfg, load_model
from molecular_ml.models.model import MoleculeAttentionLSTM


def predict_single(model_path: str, smile: str, config_file: str=""):
    #Read config file
    cfg = read_cfg(config_file)
    
    print('Loading data')
    pred_loader = get_bit_map_loader_from_smile(smile)
    x = None
    for data, _ in pred_loader:
        x = data

    #Load Data
    model = MoleculeAttentionLSTM()
    model = load_model(model, model_path)
    model.eval()
    
    print("model loaded")

    with torch.no_grad():
        out = model(x)
    
    return {cfg.DATA_DIGEST.labels_name[0]: (out[0][0].item() > 0.5) * 1}


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

