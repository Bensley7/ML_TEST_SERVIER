from collections import defaultdict
import logging
from typing import Dict, List
import sys

sys.path.append("../")
from dataset.loader import MoleculeDataLoader
from models.model import MoleculeModel
from models.metrics import get_metric_func

from .predict import predict

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


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             metrics: List[str],
             dataset_type: str) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`MoleculeModel`.
    :param data_loader: A :class:`MoleculeDataLoader`.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    preds = predict(
        model=model,
        data_loader=data_loader,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        metrics=metrics,
        dataset_type=dataset_type,
    )

    return results
