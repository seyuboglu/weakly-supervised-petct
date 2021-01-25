# Run stats tests for xmodal paper
import pickle
import logging
import os
from typing import List
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel

from pet_ct.analysis.visualization import unpack_experiment_groups


def get_experiment_group_metrics_df(
    experiment_groups: List[dict],
    tasks=List[str],
    metric="roc_auc",
    splits=["test"],
    epoch="best",
):
    experiments = []
    for experiment_group in experiment_groups:
        name = experiment_group["name"]
        for experiment_dir in experiment_group["dirs"]:
            experiments.append(
                {
                    "name": name,
                    "dir": experiment_dir,
                    "splits": experiment_group.get("splits", splits),
                }
            )

    data = []

    def add_metrics(experiment_dir, name, splits):
        for split in splits:
            metrics_path = os.path.join(experiment_dir, f"{epoch}/{split}_metrics.json")
            if not os.path.isfile(metrics_path):
                return

            with open(metrics_path) as f:
                metrics_dict = json.load(f)
            for task, metrics in metrics_dict.items():
                if tasks is not None and task not in tasks:
                    continue
                entry = {
                    f"{metric}": metrics[metric],
                    "name": name,
                    f"exeriment_dir": experiment_dir,
                    "task": task,
                    f"split": split,
                    f"epoch": epoch,
                }
                data.append(entry)

    for experiment in experiments:
        name = experiment["name"]
        experiment_dir = experiment["dir"]
        if os.path.isdir(os.path.join(experiment_dir, "candidates")):
            for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                subdir = os.path.join(experiment_dir, "candidates", dirname)
                add_metrics(subdir, name, splits=experiment["splits"])
        else:
            add_metrics(experiment_dir, name, splits=experiment["splits"])
    data_df = pd.DataFrame(data)
    return data_df
