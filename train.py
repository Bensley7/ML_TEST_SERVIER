import argparse

from train.attention_lstm_train import train_model as trlstm
from train.mpn_ffd_train import train_model as trmpn 
from utils.io import read_cfg

def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--train_data_path", help="train dataset path", type=str)
    parser.add_argument("--val_data_path", help="valid dataset path", type=str)
    parser.add_argument("--test_data_path", help="test dataset path", type=str)

    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model_dir', type=str, default="mol_models/", help='directory to save the model')
    parser.add_argument('--model_name', type=str, default = "model1.pt", help='pytorch model name')

    args = parser.parse_args()
    return args


def train(opt):
    #Read config file
    cfg = read_cfg(opt.config_file, opt.opts)
    if cfg.MODEL.name == "attention_lstm":
        trlstm(**vars(opt))
    elif cfg.MODEL.name == "mpn_ffd":
        trmpn(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    train(args)