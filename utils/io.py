import sys, os
import logging
from config import cfg
import sys

import torch

sys.path.append("../")
from models.model import MoleculeModel

def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def read_cfg(config_file, opts):
    logger = setup_logger("Setup cfg", "/tmp", 0)
    if config_file != "":
        cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    if config_file != "":
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg

def save_checkpoint(
    path: str,
    model: MoleculeModel) -> None:
    """
    Saves a model checkpoint.
    :param model: A :class:`MoleculeModel`.
    :param path: Path where checkpoint will be saved.
    """
    state = {"state_dict": model.state_dict()}
    torch.save(state, path)


def load_checkpoint(
    path: str, cfg, device: torch.device = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param cfg: config file.
    :param device: Device where the model will be moved.
    :return: The loaded :class:`MoleculeModel`.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = state["state_dict"]
    device = torch.device(cfg.MODEL.device)

    # Build model
    model = MoleculeModel(cfg)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            print (
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            print(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            print(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)
    model = model.to(device)

    return model

def load_model(model, model_path):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)["state_dict"])
    else:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
        )
    return model
