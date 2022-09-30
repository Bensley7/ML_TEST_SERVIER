import sys, os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

sys.path.append("../")
from common.io import read_cfg

class DataDigest:
    """Class to split raw molecule propeties source data into train, test and val according to CFG"""
    def __init__(self, raw_data_path: str, cfg) -> None:
        self.raw_data_path = os.path.abspath(raw_data_path)
        self.explicative_cols = ["mol_id", "smiles"]
        self.labels_name = cfg.labels_name
        self.val_ratio = cfg.val_ratio
        self.test_ratio = cfg.test_ratio

    @staticmethod
    def read(path: str):
        return pd.read_csv(path)

    def save(self, data: pd.DataFrame, filename: str) -> None:
        """os.path.splitext instead of str.split used in case of windows system parsing"""
        data_path = "%s_%s.csv" % (os.path.splitext(self.raw_data_path)[0], filename)
        data.to_csv(data_path, index=False)
    
    def split(self, data: pd.DataFrame):
        train_pool, test = train_test_split(
            data,
            stratify=data[self.labels_name],
            test_size=self.test_ratio,
            random_state=0,
        )
        val_ratio_from_train_size = self.val_ratio / (1 - self.test_ratio)
        train, val = train_test_split(
            train_pool,
            stratify=train_pool[self.labels_name],
            test_size=val_ratio_from_train_size,
            random_state=0,
        )
        return train, val, test

    def run(self) -> None:
        raw_data = self.read(self.raw_data_path)
        train, val, test = self.split(raw_data)
        self.save(train, "train")
        self.save(val, "val")
        self.save(test, "test")


class MultiLabelDataDigest(DataDigest):
    def __init__(self, raw_data_path: str, cfg) -> None:
            super().__init__(raw_data_path, cfg)
    
    def create_data_frame(self, data, labels):
        return pd.DataFrame(np.hstack((labels, data)),
                            columns = self.labels_name + self.explicative_cols)

    def split(self, data: pd.DataFrame):
        """ 
        Joint European Conference on Machine Learning and Knowledge Discovery in Databases
        ECML PKDD 2011: Machine Learning and Knowledge Discovery in Databases pp 145–158Cite as
        On the Stratification of Multi-label Data 
        """
        x_train_pool, y_train_pool, x_test, y_test = iterative_train_test_split(
            np.array(data[self.explicative_cols]),
            np.array(data[self.labels_name]),
            test_size = self.test_ratio,
            )
        val_ratio_from_train_size = self.val_ratio / (1 - self.test_ratio)
        x_train, y_train, x_val, y_val = iterative_train_test_split(
            x_train_pool,
            y_train_pool,
            test_size = val_ratio_from_train_size,
        )
        train = self.create_data_frame(x_train, y_train)
        val = self.create_data_frame(x_val, y_val)
        test = self.create_data_frame(x_test, y_test)
        return train, val, test


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", help="path to the raw dataset - either single or multi")
    parser.add_argument("--config_file", default="", help="path to config file to create dataset", type=str)
    parser.add_argument("opts", help="modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def main(args):
    cfg = read_cfg(args.config_file, args.opts)
    if len(cfg.DATA_DIGEST.labels_name) == 1:
        DataDigest(args.raw_data_path, cfg.DATA_DIGEST).run()
    elif len(cfg.DATA_DIGEST.labels_name) > 1:
        MultiLabelDataDigest(args.raw_data_path, cfg.DATA_DIGEST).run()
    else:
        raise Exception("Non-valid Data Digester module - there are no labels")


if __name__ == "__main__":
    args = parse_opt()
    main(args)