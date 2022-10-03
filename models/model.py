from typing import List, Union, Tuple
import sys

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("../")
from .utils import get_activation_function, initialize_weights
from features.graph_featurization import BatchMolGraph
from .mpn import MPN

class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, cfg):
        """
        :param args: object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = len(cfg.DATA_DIGEST.labels_name) == 1
        self.multiclass = len(cfg.DATA_DIGEST.labels_name) > 1
        self.loss_function = cfg.MODEL.loss_function

        self.output_size = 1
        if self.multiclass:
            self.output_size *= len(cfg.DATA_DIGEST.labels_name)

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.encoder = MPN(cfg)
        self.create_ffn(cfg)
        initialize_weights(self)

    def create_ffn(self, cfg) -> None:
        """
        Creates the feed-forward layers for the model.
        :param args: object containing model arguments.
        """
        self.multiclass = len(cfg.DATA_DIGEST.labels_name) > 1
        if self.multiclass:
            self.num_classes = len(cfg.DATA_DIGEST.labels_name)
        
        first_linear_dim = cfg.MODEL.hidden_size
        dropout = nn.Dropout(cfg.MODEL.dropout)
        activation = get_activation_function(cfg.MODEL.activation)
        # Create FFN layers
        if cfg.MODEL.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, cfg.MODEL.ffn_hidden_size)
            ]
            for _ in range(cfg.MODEL.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(cfg.MODEL.ffn_hidden_size, cfg.MODEL.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(cfg.MODEL.ffn_hidden_size, self.output_size),
            ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]]) -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: The latent fingerprint vectors.
        """
        return self.encoder(batch)

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]]) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`features.graph_featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """

        output = self.ffn(self.encoder(batch))
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss
        return output


class AttentionPooling(nn.Module):
    def __init__(self, nb_features: int):
        super().__init__()
        self.in_features = nb_features
        self.d = nn.Linear(nb_features, 1)

    def forward(self, x):
        x1 = F.softmax(self.d(x), dim=1).expand(x.size())
        x2 = x1 * x
        v = x2.sum(1)
        return v


class MolLSTM(nn.Module):
    def __init__(
        self,
        d_out: int = 1,
        vocab_limit: int = 101,
        embedding_dim: int = 128,
        hidden_lstm_size: int = 128,
        p_dropout: float = 0.25,
        att_polling_size: int = 256,
    ) -> None:
        super().__init__()

        self.embeddings = torch.nn.Embedding(vocab_limit, embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_lstm_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.attn_pool = AttentionPooling(att_polling_size)
        self.out = nn.Linear(att_polling_size, d_out)

    def forward(self, x):
        x = self.dropout(self.embeddings(x))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.attn_pool(x)
        x = torch.sigmoid(self.out(x))
        return x
