import csv
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

from torch.utils.data import DataLoader

import sys
sys.path.append("..")

from dataset.loader import MoleculeDataset, MoleculeDatapoint, MoleculeBitMapDataset
from models.metrics import get_metric_func

def get_attributes(path: str, smiles_columns, target_columns):
    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)
        all_smiles, all_targets = [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]
            targets = []
            for column in target_columns:
                value = row[column]
                targets.append(float(value))

            all_smiles.append(smiles)
            all_targets.append(targets)

    return all_smiles, all_targets


def get_data_from_smile(smile: str, cfg) -> MoleculeDataset:
    all_smiles = [smile]
    all_targets = [0]
    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smiles,
            targets=targets,
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])
    return data 


def get_data(path: str, cfg, skip_invalid_smiles = True) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.
    :param path: Path to a CSV file.
    :param cfg: config data.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :return: A :class:`dataset.loader.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """

    smiles_columns = [cfg.DATA_DIGEST.smiles_name]
    target_columns = cfg.DATA_DIGEST.labels_name

    all_smiles, all_targets = get_attributes(path, smiles_columns, target_columns)
    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smiles,
            targets=targets,
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            print(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_molecule_bit_map_loader(path: str, cfg, max_len: int = 64):

    smiles_columns = [cfg.DATA_DIGEST.smiles_name]
    target_columns = cfg.DATA_DIGEST.labels_name

    all_smiles, all_targets = get_attributes(path, smiles_columns, target_columns)

    molecule_bit_map_dataset = MoleculeBitMapDataset(all_smiles, all_targets)
    
    return DataLoader(
        molecule_bit_map_dataset,
        batch_size=cfg.MODEL.TRAINING.batch_size,
        shuffle=False,
        num_workers=cfg.MODEL.num_workers,
    )

def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.
    :param data: A :class:`dataset.loader.MoleculeDataset`.
    :return: A :class:`dataset.loader.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         metrics: List[str],
                         dataset_type: str) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] for metric in metrics}

    valid_preds = []
    valid_targets = []
    for j in range(len(preds)):
        if targets[j][0] is not None:  # Skip those without targets
            valid_preds.append(preds[j][0])
            valid_targets.append(targets[j][0])

    results = defaultdict(list)
    
    if dataset_type == 'classification':
        nan = False
        if all(target == 0 for target in valid_targets) or all(target == 1 for target in valid_targets):
            nan = True
        if all(pred == 0 for pred in valid_preds) or all(pred == 1 for pred in valid_preds):
            nan = True

        if nan:
            for metric in metrics:
                results[metric].append(float('nan'))

    for metric, metric_func in metric_to_func.items():
        if dataset_type == 'multiclass' and metric == 'cross_entropy':
            results[metric].append(metric_func(valid_targets, valid_preds,
                                            labels=list(range(len(valid_preds[0])))))
        else:
            results[metric].append(metric_func(valid_targets, valid_preds))

    results = dict(results)

    return results
