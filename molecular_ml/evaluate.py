import argparse

from molecular_ml.evaluation.mpn_ffd_eval import eval_model as evalmpn
from molecular_ml.evaluation.attention_lstm_eval import eval_model as evallstm
from molecular_ml.utils.io import read_cfg

def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('model_path', type=str, required = True, help='weight path of the model')
    parser.add_argument("val_data_path", help="dataset path with labels", type=str, required = True)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)

    args = parser.parse_args()
    return args

def evaluate_model(model_path: str, val_data_path: str, config_file: str=""):
    #Read config file
    cfg = read_cfg(config_file)
    if cfg.MODEL.name == "attention_lstm":
        evallstm(model_path, val_data_path, config_file)
    elif cfg.MODEL.name == "mpn_ffd":
        evalmpn(model_path, val_data_path, config_file)

if __name__ == "__main__":
    args = parse_opt()
    evaluate_model(args.model_path,
                   args.val_data_path,
                   args.config_file)