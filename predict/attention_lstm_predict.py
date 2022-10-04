
import sys
import argparse

import torch

sys.path.append("../")
from utils.utils import get_bit_map_loader_from_smile
from utils.io import read_cfg, load_model
from models.model import MoleculeAttentionLSTM


def predict_single(model_path: str, smile: str, config_file: str, opts):
    #Read config file
    cfg = read_cfg(config_file, opts)
    
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
    parser.add_argument('--model_path', type=str, required = True, help='weight path of the model')
    parser.add_argument("--smile", help="smile of a molecule", type=str, required = True)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def pred(opt):
    return predict_single(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    eval(args)

