import argparse

from molecular_ml.prediction.mpn_ffc_predict import predict_single as predmpn
from molecular_ml.prediction.attention_lstm_predict import predict_single as predlstm
from molecular_ml.utils.io import read_cfg

def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('model_path', type=str, help='weight path of the model')
    parser.add_argument("smile", help="smile of a molecule", type=str)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)

    args = parser.parse_args()
    return args

def predict_model(model_path: str, smile: str, config_file: str=""):
    #Read config file
    cfg = read_cfg(config_file)
    res = None
    if cfg.MODEL.name == "attention_lstm":
        res = predlstm(model_path, smile, config_file)
    elif cfg.MODEL.name == "mpn_ffd":
        res = predmpn(model_path, smile, config_file)
    print(res)
    return res

if __name__ == "__main__":
    args = parse_opt()
    predict_model(args.model_path,
                  args.smile,
                  args.config_file)