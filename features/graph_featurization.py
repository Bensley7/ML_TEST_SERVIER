from typing import List, Tuple, Union
from itertools import zip_longest
import logging

from rdkit import Chem
import torch
import numpy as np

from .feature_extractor import make_mole
from .utils import atom_features, bond_features, get_bond_fdim, get_atom_fdim


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]):
        """
        :param mol: A SMILES or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = make_mole(mol)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)
                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * `atom_fdim`: The dimensionality of the atom feature vector.
    * `bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * `a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * `b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * `max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        """
        param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        f_bonds = self.f_bonds
        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope


def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]]) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.
    :param mols: A list of SMILES or a list of RDKit molecules.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol) for mol in zip_longest(mols)])