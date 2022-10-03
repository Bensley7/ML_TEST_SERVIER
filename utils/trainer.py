from tqdm import tqdm
from typing import Callable

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..dataset.loader import MoleculeDataset, MoleculeDataLoader
from ..models.model import MoleculeModel
from ..models.utils import NoamLR
from ..models.metrics import compute_gnorm, compute_pnorm

def batch_train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          dataset_type: str,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          batch_size: int,
          log_frequency: int,   
          n_iter: int = 0) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`MoleculeModel`.
    :param data_loader: A :class:`MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: An  object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :return: The total number of iterations (training examples) trained on so far.
    """
    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, target_batch, mask_batch, data_weights_batch = \
            batch.batch_graph(), batch.targets(), batch.mask(), batch.data_weights()

        mask = torch.tensor(mask_batch, dtype=torch.bool) # shape(batch, tasks)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # shape(batch, tasks)

        target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
        data_weights = torch.tensor(data_weights_batch).unsqueeze(1) # shape(batch,1)

        # Run model
        model.zero_grad()
        preds = model(mol_batch)

        # Move tensors to correct device
        torch_device = preds.device
        mask = mask.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)

        if dataset_type == "multiclass":
            targets = targets.long()
            target_losses = []
            for target_index in range(preds.size(1)):
                target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                target_losses.append(target_loss)
            loss = torch.cat(target_losses, dim=1).to(torch_device) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // batch_size) % log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            print(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

    return n_iter