import csv
from tqdm import tqdm

import sys
sys.path.append("..")

from dataset.loader import MoleculeDataset, MoleculeDatapoint


def get_data(path: str, cfg, skip_invalid_smiles = True) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.
    :param path: Path to a CSV file.
    :param cfg: config data.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :return: A :class:`dataset.loader.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """

    # Prefer explicit function arguments but default to args if not provided
    smiles_columns = [cfg.DATA_DIGEST.smiles_name]
    target_columns = cfg.DATA_DIGEST.labels_name

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
