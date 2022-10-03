import argparse

from eval.mpn_ffd_eval import eval as evalmpn
from utils.io import read_cfg

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
    #Read config file
    cfg = read_cfg(opt.config_file, opt.opts)
    if cfg.MODEL.name == "mpn_ffd":
        evalmpn(**vars(opt))

if __name__ == "__main__":
    args = parse_opt()
    eval(args)