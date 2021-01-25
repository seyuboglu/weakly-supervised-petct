"""
Source: Jared Dunnmon
"""
# Run stats tests for xmodal paper
import pickle
import logging
import os
from typing import List
from itertools import combinations

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
from mlxtend.evaluate import permutation_test
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


from pet_ct.analysis.visualization import unpack_experiment_groups
from pet_ct.analysis.source_data import get_experiment_group_metrics_df
from pet_ct.util.util import process

# Setting up conversion from numpy to R
rpy2.robjects.numpy2ri.activate()
logging.getLogger().setLevel(logging.INFO)
proc = importr("pROC")


@process
def compute_permutation_test(
    process_dir,
    experiment_groups: List[dict],
    tasks=List[str],
    metric="roc_auc",
    split="test",
    epoch="best",
    method="approximate",
    num_rounds=10000,
    one_sided=False,
    all_tasks=None,
):
    data_df = get_experiment_group_metrics_df(
        experiment_groups,
        tasks=tasks if "all" not in tasks else None,
        metric=metric,
        splits=[split],
        epoch=epoch,
    )
    entries = []
    for task in tasks:
        if task is not "all":
            df = data_df[data_df.task == task]
        else:
            if all_tasks is None:
                df = data_df
            else:
                df = data_df[data_df.task.isin(all_tasks)]

        names = df.name.unique()
        for idx_1, name_1 in enumerate(names):
            for idx_2, name_2 in enumerate(names):
                if name_1 == name_2 or (idx_1 > idx_2 and not one_sided):
                    continue
                values_1, values_2 = (
                    np.array(df[df.name == name_1][metric]),
                    np.array(df[df.name == name_2][metric]),
                )

                pvalue = permutation_test(
                    values_1,
                    values_2,
                    method=method,
                    num_rounds=num_rounds,
                    func="x_mean > y_mean" if one_sided else "x_mean != y_mean",
                )

                bs_result_1 = bs.bootstrap(values_1, stat_func=bs_stats.mean)
                bs_result_2 = bs.bootstrap(values_2, stat_func=bs_stats.mean)

                entries.append(
                    {
                        "pvalue": pvalue,
                        "task": task,
                        "name_1": name_1,
                        "name_2": name_2,
                        "mean_1": bs_result_1.value,
                        "lower_1": bs_result_1.lower_bound,
                        "upper_1": bs_result_1.upper_bound,
                        "mean_2": bs_result_2.value,
                        "lower_2": bs_result_2.lower_bound,
                        "upper_2": bs_result_2.upper_bound,
                        "metric": metric,
                    }
                )
    pvalue_df = pd.DataFrame(entries)
    pvalue_df.to_csv(os.path.join(process_dir, "pvalue.csv"), index=False)
    if "all" in tasks:
        data_df.to_csv(os.path.join(process_dir, "data.csv"))
    else:
        data_df[data_df.task.isin(tasks)].to_csv(os.path.join(process_dir, "data.csv"))

    return pvalue_df


# Setting up function to get delong statistics
def delong_p(scores_1, scores_2, labels_1, labels_2=None):
    if labels_2 is None:
        labels_2 = labels_1
    # uses two-sided delong test
    roc1 = proc.roc(labels_1, scores_1)
    roc2 = proc.roc(labels_2, scores_2)
    test = proc.roc_test(roc1, roc2, method="delong")
    test_dict = dict(zip(test.names, list(test)))
    auc1, auc2 = [float(a) for a in test_dict["estimate"]]
    p = [float(a) for a in (test_dict["p.value"])][0]
    return p, auc1, auc2


def compute_experiment_delong(
    experiments_1, experiments_2, task="primary", split="train", epoch="best"
):
    median_results = []
    aurocs = []
    for experiments in [experiments_1, experiments_2]:
        results = []
        for experiment_dir in experiments:
            if os.path.isdir(os.path.join(experiment_dir, "candidates")):
                for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                    subdir = os.path.join(experiment_dir, "candidates", dirname)
                    preds_path = os.path.join(subdir, f"{epoch}/{split}_preds.csv")
                    if not os.path.exists(preds_path):
                        continue
                    preds_df = pd.read_csv(preds_path, index_col=[0], header=[0, 1],)

                    if task not in preds_df:
                        continue

                    correct_probs = np.array(
                        list(
                            map(
                                lambda x: float(x.strip("[]")),
                                preds_df[task, "correct_prob"].values[1:],
                            )
                        )
                    )
                    targets = np.array(
                        list(
                            map(
                                lambda x: int(x.strip("[]")),
                                preds_df[task, "target"].values[1:],
                            )
                        )
                    )
                    probs = np.abs((1 - targets) - correct_probs)

                    auroc = roc_auc_score(targets, probs)
                    print(targets.sum())

                    results.append({"probs": probs, "targets": targets, "auroc": auroc})
        median_results.append(
            sorted(results, key=lambda x: x["auroc"])[len(results) // 2]
        )
        aurocs.append([result["auroc"] for result in results])
        print([result["auroc"] for result in results])

    print(ttest_rel(aurocs[0], aurocs[1]))
    return delong_p(
        median_results[0]["probs"],
        median_results[1]["probs"],
        median_results[0]["targets"],
        median_results[1]["targets"],
    )

