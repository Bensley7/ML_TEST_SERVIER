import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple
import sys

import numpy as np
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem

from .sampler import MoleculeSampler
sys.path.append("../")
from features.feature_extractor import make_mole
from features.graph_featurization import BatchMolGraph, MolGraph


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and associated targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                ):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this molecule.
        """
        self.smiles = smiles
        self.targets = targets
        self.row = row
        self.raw_targets = self.targets

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = make_moles(self.smiles)
        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.
        """
        return len(self.smiles)

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.
        """
        self.targets = targets

    def reset(self) -> None:
        """Resets targets to their raw values."""
        self.targets = self.raw_targets


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:
        """
        Constructs a :class:`BatchMolGraph` with the graph featurization of all the molecules.
        :return: A list of :class:`BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []
            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for _, m in zip(d.smiles, d.mol):
                    mol_graph = MolGraph(m)
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)
            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
        return self._batch_graph

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        """
        return [d.targets for d in self._data]
    
    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each molecule and task are present.
        """
        targets = self.targets()
        return [[t is not None for t in dt] for dt in targets]

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        return [1. for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.
        """
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset(self) -> None:
        """Resets targets to their raw values."""
        for d in self._data:
            d.reset()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.
        """
        return self._data[item]

def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    """
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        """
        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        """Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()


def make_moles(smiles: List[str]):
    """
    Builds a list of RDKit molecules for a list of smiles.
    """
    return [make_mole(s) for s in smiles]