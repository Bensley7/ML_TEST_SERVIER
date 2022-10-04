from typing import List, Union, Tuple
from functools import reduce
import sys

from rdkit import Chem
import torch
import torch.nn as nn

sys.path.append("../")
from .utils import get_activation_function, index_select_ND
from features.graph_featurization import BatchMolGraph, mol2graph
from features.utils import get_atom_fdim, get_bond_fdim

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, cfg, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension
        :param bias: Whether to add bias to linear layers
        :param depth: Number of message passing steps
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size or cfg.MODEL.hidden_size
        self.bias = bias or cfg.MODEL.bias
        self.depth = depth or cfg.MODEL.depth
        self.dropout = cfg.MODEL.dropout
        self.layers_per_message = 1
        self.device = torch.device(cfg.MODEL.device)
        self.aggregation = cfg.MODEL.aggregation
        self.aggregation_norm = cfg.MODEL.aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(cfg.MODEL.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size = self.hidden_size
        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)


    def forward(self, mol_graph: BatchMolGraph) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`features.graph_featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, _ = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for _ in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self, cfg):
        """
        :param args: object containing model arguments.
        """
        super(MPN, self).__init__()
        self.device = torch.device(cfg.MODEL.device)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.encoder = nn.ModuleList([MPNEncoder(cfg, self.atom_fdim, self.bond_fdim)
                                    for _ in range(cfg.MODEL.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]],
                       List[BatchMolGraph]]) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`features.graph_featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]
            batch = [mol2graph(b) for b in batch]

        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)
        return output
