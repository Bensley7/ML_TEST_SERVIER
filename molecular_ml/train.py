import argparse

from molecular_ml.training.attention_lstm_train import train_model as trlstm
from molecular_ml.training.mpn_ffd_train import train_model as trmpn 
from molecular_ml.utils.io import read_cfg

def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("train_data_path", help="train dataset path", type=str)
    parser.add_argument("val_data_path", help="valid dataset path", type=str)
    parser.add_argument("test_data_path", help="test dataset path", type=str)

    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument('--model_dir', type=str, default="mol_models/", help='directory to save the model')
    parser.add_argument('--model_name', type=str, default = "model1", help='pytorch model name')

    args = parser.parse_args()
    return args


def train_model(train_data_path: str,
                val_data_path: str,
                test_data_path: str,
                config_file: str="",
                model_dir: str="mol_models/",
                model_name: str="model1"):
    #Read config file
    cfg = read_cfg(config_file)
    if cfg.MODEL.name == "attention_lstm":
        trlstm(train_data_path,
                val_data_path,
                test_data_path,
                config_file,
                model_dir,
                model_name)
    elif cfg.MODEL.name == "mpn_ffd":
        trmpn(train_data_path,
                val_data_path,
                test_data_path,
                config_file,
                model_dir,
                model_name)

if __name__ == "__main__":
    args = parse_opt()
    train_model(args.train_data_path,
                args.val_data_path,
                args.test_data_path,
                args.config_file,
                args.model_dir,
                args.model_name)