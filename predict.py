import argparse

from predict.mpn_ffc_predict import pred as predmpn
from utils.io import read_cfg

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
    #Read config file
    cfg = read_cfg(opt.config_file, opt.opts)
    if cfg.MODEL.name == "mpn_ffd":
        print(predmpn(opt))

if __name__ == "__main__":
    args = parse_opt()
    pred(args)