"""
Defines metric functions of form
    def metric_name(output_batch, labels_batch):
Where output_batch and labels_batch are torch tensors on the cpu.
"""
from collections import defaultdict
import os 
import json
import logging 

import numpy as np
import pandas as pd
import sklearn.metrics as skl
import torch

from pet_ct.util.util import hard_to_soft, flex_concat, place_on_cpu, get_batch_size, process
from pet_ct.data.report_transforms import split_impression_sections, word_tokenize


class Metrics:
    """
    """
    def __init__(self, metric_configs=[]):
        """
        args:
            metrics_fns (list, strings)   list of function names to use for eval
            break_ties (string) method to break ties (in probabilistic labels).
        """
        self.metric_configs = metric_configs

        # Dictionary mapping functions to
        self.metrics = defaultdict(dict)

        self.global_metrics = defaultdict(int)

        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

        self.info = []

        self.precomputed_keys = None
        self.total_size = 0

    def add(self, preds, targets,
            info, precomputed_metrics={}):
        """
        Add a batches preds and targets. If you've already computed metrics for the batch
        (e.g. loss) you can pass in that value via precomputed metrics and it will update
        the average value in metrics. Note: must always provide the same precomputed
        metrics.
        args:
            preds    (list(tensor))    [(batch_size, ..., k)]
            targets     (list(tensor))    [(batch_size, ...,  1)]
            precomputed_metrics (dict) A dictionary of metric  values that have already
                been computed for the batch
        """
        # convert to list if only one element is passed in
        if type(preds) != dict:
            preds = {"primary": preds}
        if type(targets) != dict:
            targets = {"primary": targets}
        if preds.keys() != targets.keys():
            raise ValueError("Predictions and targets over different tasks.")

        tasks = list(preds.keys())
        batch_size = get_batch_size(list(targets.values())[0])
        for task in tasks:
            task_targets = targets[task]
            task_preds = preds[task]
            if(get_batch_size(task_targets) != get_batch_size(task_preds)):
                raise ValueError("preds must match targets in first dim.")

            self.preds[task].append(place_on_cpu(task_preds))
            self.targets[task].append(place_on_cpu(task_targets))

        self.info.extend(info)

        # include precomputed keys in global metrics
        if self.precomputed_keys is None:
            self.precomputed_keys = set(precomputed_metrics.keys())
        elif self.precomputed_keys != set(precomputed_metrics.keys()):
            raise ValueError("must always supply same precomputed metrics.")

        for key, value in precomputed_metrics.items():
            self.global_metrics[key] = ((self.total_size * self.global_metrics[key] +
                                         batch_size * value) /
                                        (batch_size + self.total_size))
        self.total_size += batch_size

    def compute(self):
        """
        Computes metrics on all
        """
        # call all metric_fns, detach since output has require grad
        for metric_config in self.metric_configs:
            self._compute_metric(**metric_config)

    def _compute_metric(self, fn, args={}, name=None, tasks=None):
        """
        @param primary_task (deprecated) now pass primary_task to constructor
        """
        name = name if name is not None else fn

        values = []
        for task in self.preds.keys():
            if tasks is not None and task not in tasks:
                continue

            total_value = 0
            total_size = 0

            all_preds = []
            all_targets = []
            for batch_preds, batch_targets in zip(self.preds[task], self.targets[task]):
                if type(batch_preds) is torch.Tensor:
                    # flatten dimensions
                    batch_preds = batch_preds.view(-1, batch_preds.shape[-1]).squeeze(-1)
                    batch_targets = batch_targets.view(-1).squeeze(-1)
                all_preds.append(batch_preds)
                all_targets.append(batch_targets)

            all_preds = flex_concat(all_preds, dim=0)
            all_targets = flex_concat(all_targets, dim=0)

            value = globals()[fn](all_preds, all_targets, **args)

            self.metrics[task][name] = value
            values.append(value)

        self.global_metrics[name] = np.mean(values)

    def get_metric(self, metric, task=None):
        """
        """
        if task is None:
            return self.global_metrics[metric]
        else:
            return self.metrics[task][metric]

    def get_preds(self, pos_index=1):
        """
        Get a dataframe of model preds.
        args:
            pos_index   (int) The index of the class used for probability.
        return:
            df  (DataFrame) A dataframe indexed by exam and patient
        """
        index = [info["exam_id"]for info in self.info]
        task2df = {}
        for task in self.preds.keys():
            preds = []
            targets = []
            correct_probs = []

            for batch_probs, batch_targets in zip(self.preds[task], self.targets[task]):
                if type(batch_probs) is torch.Tensor:
                    for (curr_probs,
                         curr_targets) in zip(torch.split(batch_probs, 1, dim=0),
                                              torch.split(batch_targets, 1, dim=0)):
                        curr_probs = curr_probs.numpy()
                        curr_preds = np.argmax(curr_probs, axis=-1)
                        curr_targets = curr_targets.numpy().astype(int)

                        if len(curr_probs.shape) == 3:
                            # if multiple outputs per example
                            curr_correct_probs = curr_probs[0,
                                                            np.arange(curr_probs.shape[1]),
                                                            curr_targets]
                        elif len(curr_probs.shape) == 2:
                            curr_correct_probs = curr_probs[0, curr_targets]
                        else:
                            raise ValueError("Invalid number of dimensions for output \
                                              probabilities. Must be (batch_size, k) or \
                                              (batch_size, d, k).")

                        # don't include
                        curr_correct_probs = curr_correct_probs[curr_targets != -1]
                        curr_preds = curr_preds[curr_targets != -1]
                        curr_targets = curr_targets[curr_targets != -1]

                        # todo make compatible with shape
                        targets.append(curr_targets)
                        correct_probs.append(curr_correct_probs)
                        preds.append(curr_preds)
                else:
                    targets.extend(batch_targets)
                    preds.extend(batch_probs)
                    correct_probs.extend([None] * len(batch_probs))

            df = pd.DataFrame(data={"target": [str(target) for target in targets],
                                    "pred": [str(pred) for pred in preds],
                                    "correct_prob": [str(correct_prob)
                                                     for correct_prob in correct_probs]},
                              index=index)

            task2df[task] = df

        return pd.concat(task2df.values(), axis=1, keys=task2df.keys())


def read_preds(preds_path):
    """
    """
    preds_df = pd.read_csv(preds_path, index_col=0, header=[0, 1])
    preds_df = preds_df.applymap(lambda x: float(x.strip("[]")))
    for task in preds_df.columns.get_level_values(0):
        pred_probs = {}
        for idx, row in preds_df[task].iterrows():
            pred_probs[idx] = row["correct_prob"] if row["target"] == 1.0 else 1 - row["correct_prob"]
        preds_df[task, "pred_prob"] = pd.Series(pred_probs)
    return preds_df
    

def accuracy(probs, targets):
    """
    Computes accuracy between output and labels for k classes. Targets with class -1 are
    ignored.
    args:
        probs    (tensor)    (size, k)  2-d array of class probabilities
        labels     (tensor)    (size, 1) 1-d array of correct class indices
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    targets = targets.numpy()

    pred = np.argmax(probs, axis=1)

    # ignore -1
    pred = pred[(targets != -1).squeeze()]
    targets = targets[targets != -1]

    return np.sum(pred == targets) / targets.size


def roc_auc(probs, labels):
    """
    Computes the area under the receiving operator characteristic between output probs
    and labels for k classes.
    Source: https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    If only one class present, returns 0 instead of crashing.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = probs[(labels != -1).squeeze()]
    labels = labels[labels != -1]
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    # Convert labels to one-hot indicator format, using the k inferred from probs
    if len(np.unique(labels)) <= 1:
        return 0
    labels = hard_to_soft(labels, k=probs.shape[1]).numpy()
    return skl.roc_auc_score(labels, probs)


def precision(probs, labels):
    """
    Computes the precision score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.precision_score(labels, pred)


def recall(probs, labels):
    """
    Computes the recall score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)

    return skl.recall_score(labels, pred)


def f1_score(probs, labels):
    """
    Computes the f1 score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.f1_score(labels, pred, pos_label=1)


def negative_precision(probs, labels):
    """
    Computes the precision score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.precision_score(labels, pred, pos_label=0)


def negative_recall(probs, labels):
    """
    Computes the recall score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)

    return skl.recall_score(labels, pred, pos_label=0)


def negative_f1_score(probs, labels):
    """
    Computes the f1 score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.f1_score(labels, pred, pos_label=0)


def compute_subset_metrics(experiment_dir, splits=None, epoch="best", exam_ids=None):
    if splits is None:
        splits = ["test"]
    
    for split in splits:  
        preds_df = pd.read_csv(os.path.join(experiment_dir, f"{epoch}/{split}_preds.csv"), skiprows=[0])
        preds_df = preds_df.rename(columns={"Unnamed: 0": "exam_id"})
        preds_df.target = preds_df.target.apply(lambda x: int(x.strip("[]")))
        preds_df.pred = preds_df.pred.apply(lambda x: int(x.strip("[]")))
        preds_df.correct_prob = preds_df.correct_prob.apply(lambda x: float(x.strip("[]")))
        preds_df["prob"] = preds_df.apply(lambda x: x.correct_prob if x.target == 1 else 1 - x.correct_prob, axis=1)
        
        metrics_path = os.path.join(experiment_dir, f"{epoch}/{split}_metrics.json")
        with open(metrics_path) as f:
            metrics_dict = json.load(f)
            task = next(metrics_dict.keys().__iter__())
            metrics_dict = next(metrics_dict.values().__iter__())
        
        # check that we're computing the same metrics that were recorded at test time
        assert(np.abs(metrics_dict["roc_auc"] - skl.roc_auc_score(y_true=preds_df.target, y_score=preds_df.prob)) < 0.01)
        
        mask = preds_df.exam_id.isin(exam_ids)
        metrics = {
            task: {
              "roc_auc": skl.roc_auc_score(
                  y_true=preds_df[mask].target, 
                  y_score=preds_df[mask].prob
              )
            }
        }
        
        path = os.path.join(experiment_dir, f"{epoch}/{split}_{len(exam_ids)}_metrics.json")
        with open(path, 'w') as f:
            logging.info(f"Writing metrics to path {path}.")
            json.dump(metrics, f)

@process
def compute_subset_metrics_many(process_dir, experiment_dirs: list, subset_path: str):

    exam_ids = set(pd.read_csv(subset_path).exam_id)

    for experiment_dir in experiment_dirs:
        if os.path.isdir(os.path.join(experiment_dir, "candidates")):
            for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                subdir = os.path.join(experiment_dir, "candidates", dirname)
                compute_subset_metrics(subdir, exam_ids=exam_ids)