"""
Process subclass that reads reports and outputs a labels csv.
"""

import os
import copy
from collections import defaultdict, OrderedDict

import pandas as pd
from tqdm import tqdm, tqdm_notebook
import json
import numpy as np
import networkx as nx
import torch
import logging
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, Markdown

from pet_ct.util.util import Process, soft_to_hard
from pet_ct.util.graphs import TriangleGraph
import pet_ct.learn.dataloaders as dataloaders
import pet_ct.learn.datasets as datasets
from pet_ct.data.report_transforms import extract_impression, split_impression_sections, word_tokenize
from metal.multitask.mt_label_model import MTLabelModel
from metal.analysis import lf_summary
from metal.metrics import metric_score
from metal.utils import pred_to_prob
import pet_ct.data.labeler as labeler
import pet_ct.data.task_graphs as task_graphs
from pet_ct.data.term_graphs import TermGraph


METRICS_LIST = [
    "accuracy",
    "coverage",
    "precision",
    "recall",
    "f1",
    "fbeta",
    "roc-auc"
]


class LabelsBuilder(Process):
    """
    """
    def __init__(self, dir, seed=123,
                 valid_targets_path="",
                 dataset_class="ReportDataset",
                 dataset_args={},
                 dataloader_configs=[],
                 labeler_class="Labeler",
                 labeler_args={},
                 task_configs=[],
                 lf_configs=[],
                 task_graph_class='TaskGraph'):
        """
        Initializes the dataloaders and parses the label and task configs.
        """
        super().__init__(dir)
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.validate = False
        if valid_targets_path:
            self.validate = True
            self.valid_targets_path = valid_targets_path

        # load dataloaders (typically train and val split)
        self._build_dataloaders(dataset_class, dataset_args, dataloader_configs)
        # initialize labeler object
        self.labeler = getattr(labeler, labeler_class)(**labeler_args)
        # organize configurations to match tasks to label functions
        self._init_config_dicts(task_configs, lf_configs)
        # set task graph
        self._build_task_graph(task_graph_class)

    def _build_dataloaders(self, dataset_class, dataset_args, dataloader_configs):
        """
        TODO: Pulled from experiment.py. Likely possible to decompose.
        """
        self.datasets = {}
        self.dataloaders = {}
        for dataloader_config in dataloader_configs:
            split = dataloader_config["split"]
            dataloader_class = dataloader_config["dataloader_class"]
            dataloader_args = dataloader_config["dataloader_args"]
            logging.info(f"Loading {split} data")
            self._build_dataloader(split, dataset_class, dataset_args,
                                   dataloader_class, dataloader_args)

    def _build_dataloader(self, split, dataset_class, dataset_args,
                          dataloader_class, dataloader_args):
        """
        """
        # create dataset
        dataset = getattr(datasets, dataset_class)(split=split,
                                                   **dataset_args)
        self.datasets[split] = dataset
        dataloader = (getattr(dataloaders, dataloader_class)(dataset,
                                                             **dataloader_args))
        self.dataloaders[split] = dataloader

    def _init_config_dicts(self, task_configs, lf_configs):
        # NOTE: idx_to_task is the ordering of task_configs
        self.task_configs = task_configs
        self.idx_to_task = {idx: config['task'] for idx, config in enumerate(self.task_configs)}

        self.lf_configs = lf_configs
        lf_fn_to_lf_args = {config['fn']: config['args'] for config in lf_configs}

        self.lf_name_to_lf_config = {}
        self.task_to_lf_names = {}
        for task_config in self.task_configs:
            lf_names = []
            for lf_task_config in task_config['lf_task_configs']:
                lf_name = lf_task_config['lf_name']
                lf_names.append(lf_name)

                lf_fn = lf_task_config['lf_fn']
                lf_args = lf_fn_to_lf_args[lf_fn].copy()
                lf_args.update(lf_task_config.get('lf_task_args', {}))
                self.lf_name_to_lf_config[lf_name] = {
                    "fn": lf_fn,
                    "args": lf_args,
                    "cardinality": task_config['cardinality']
                }
            self.task_to_lf_names[task_config['task']] = lf_names

    def _build_task_graph(self, task_graph_class):
        """
        """
        edges = []
        mutex_tasks = []
        task_to_idx = {task: idx for idx, task in self.idx_to_task.items()}
        for idx, task_config in enumerate(self.task_configs):
            if 'parents' in task_config.keys():
                edges += [(task_to_idx[parent_task], idx)
                          for parent_task in task_config['parents']]
            if task_config.get('mutex_task', False):
                mutex_tasks.append(idx)

        tg_args = {
            'cardinalities': [config['cardinality'] for config in self.task_configs],
            'edges': edges,
        }
        if task_graph_class == 'TaskHierarchyFlex':
            tg_args['mutex_tasks'] = mutex_tasks

        task_graph_dict = {
            'tg_class': task_graph_class,
            'tg_args': tg_args,
            'idx_to_task': self.idx_to_task
        }
        with open(os.path.join(self.dir, 'task_graph.json'), 'w') as f:
            json.dump(task_graph_dict, f, indent=4)

        self.task_graph = getattr(task_graphs,
                                  task_graph_class)(**tg_args)

    def _load_reports(self, split):
        """
        """
        exam_id_to_report = OrderedDict() # maintains dataloader order
        for i, (label, report, info) in tqdm(enumerate(self.dataloaders[split]),
                                             total=len(self.dataloaders[split])):
            exam_id_to_report[info['exam_id'][0]] = {
                'label': label[0],
                'text': report,
                'info': info
            }
        return exam_id_to_report

    def _run(self, overwrite=False, train_split="train", valid_split="val"):
        """
        NOTE: Can only validate on the valid split of the dataset.
        """
        for split in [train_split, valid_split]:
            exam_id_to_report = self._load_reports(split)
            sources_df, label_sources = self._load_label_sources(exam_id_to_report)
            self._save_label_sources(sources_df, label_sources, split)
            if split == train_split:
                probs = self.train(label_sources)
            else:
                probs = self.predict(label_sources)
                if self.validate:
                    logging.info('Computing metrics')
                    target_probs = self._load_label_probs(self.valid_targets_path)
                    metrics = self.score(probs, target_probs)
                    with open(os.path.join(self.dir, f'{split}_metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=4)
            self._save_label_probs(probs, exam_id_to_report.keys(), split)

    def train(self, label_sources):
        """
        Trains the label model.

        Args:
            label_sources (object) user-defined inputs

        Returns:
            t-length list of [n,k_t] matrices where the (i,j) entry of the
                sth matrix represents the estimated P((Y_i)_s | \lambda_j(x_i))
        """
        raise NotImplementedError

    def predict(self, label_sources):
        """
        Outputs label model predictions.

        Args:
            label_sources (object) user-defined inputs

        Returns:
            A t-length list of [n,k_t] matrices where the (i,j) entry of the
                sth matrix represents the estimated P((Y_i)_s | \lambda_j(x_i))
        """
        raise NotImplementedError

    def score(self, probs, target_probs):
        """
        """
        metrics = defaultdict(dict)
        for task_idx, _ in enumerate(probs):
            probs_t = torch.tensor(probs[task_idx]).double()
            preds_t = soft_to_hard(probs_t, break_ties='random')

            target_probs_t = torch.tensor(target_probs[task_idx]).double()
            targets = soft_to_hard(target_probs_t, break_ties='random')

            print(pred_to_prob(targets, k=probs_t.shape[1]))
            for metric in METRICS_LIST:
                metrics[self.idx_to_task[task_idx]][metric] = metric_score(targets + 1, preds_t + 1, metric, probs=probs_t)

        return metrics

    def _load_label_sources(self, exam_id_to_report):
        """
        Extracts label sources from the reports in a given split.

        Args:
            exam_id_to_report:  (dict)  matches exam_ids to their reports
        Returns:
            A tuple of (sources_df, label_sources). We leave it to the user
            to format sources_df and label_sources due to high variability in
            LabelBuilder approaches.
        """
        raise NotImplementedError

    def _save_label_sources(self, sources_df, label_sources, split):
        """
        """
        sources_df.to_csv(os.path.join(self.dir, f'{split}_labels_raw.csv'))

    def _load_label_probs(self, filepath):
        """
        Returns labels from a probs dataset.

        Args:
            filepath    (string)    full filepath to a targets dataset (valid split)
        """
        targets_df = pd.read_csv(filepath, index_col=0, header=[0, 1], skipinitialspace=True)
        targets_exam_ids = list(targets_df.index)
        gold = []
        for idx, task in sorted(self.idx_to_task.items()):
            gold.append(targets_df[task].to_numpy())
        return gold

    def _save_label_probs(self, Y_ps, exam_ids, split):
        """
        Saves model predictions (labels) in a csv.

        NOTE: relies on ordered nature of exam_ids to match with indices of Y_ps.
        In this class we extract the exam_ids from the exam_id_to_report dict.
        Dictionaries are ordered in Python 3.7. This is a language feature, not
        happenstance, and this can be verified at the following links:

        https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
        https://mail.python.org/pipermail/python-dev/2017-December/151283.html

        Args:
            Y_ps: (list)    a t-length list of [n,k_t] matrices where the (i,j)
                entry of the sth matrix represents the estimated P((Y_i)_s | \lambda_j(x_i))
            exam_ids:  (list)  exam_ids.
            split:  (string)    split used to name the final csv.
        """
        num_rows = len(exam_ids)
        num_cols = sum([m.shape[1] for m in Y_ps])
        probs_matrix = np.zeros((num_rows, num_cols))
        multi_index = []
        for i in range(num_rows):
            curr_col = 0
            for task_idx in range(len(Y_ps)):
                cardinality = Y_ps[task_idx].shape[1]
                for class_idx in range(cardinality):
                    probs_matrix[i, curr_col] = Y_ps[task_idx][i][class_idx]
                    if i == 0:
                        multi_index.append((self.idx_to_task[task_idx], class_idx))
                    curr_col += 1
        multi_index = pd.MultiIndex.from_tuples(multi_index, names=["task", "class"])
        probs_df = pd.DataFrame(probs_matrix, columns=multi_index, index=exam_ids)
        probs_df.to_csv(os.path.join(self.dir, f'{split}_labels_probs.csv'))


class MetalLabelsBuilder(LabelsBuilder):
    """
    Uses Metal's MTLabelModel to create probabilistic labels for reports.
    """
    def __init__(self, dir, seed=123,
                 valid_targets_path="",
                 dataset_class="ReportDataset",
                 dataset_args={},
                 dataloader_configs=[],
                 task_configs=[],
                 labeler_class="Labeler",
                 labeler_args={},
                 lf_configs=[],
                 task_graph_class="TaskGraph",
                 metal_model_args={}):
        """
        """
        super().__init__(dir, seed=seed,
                         valid_targets_path=valid_targets_path,
                         dataset_class=dataset_class,
                         dataset_args=dataset_args,
                         dataloader_configs=dataloader_configs,
                         labeler_class=labeler_class,
                         labeler_args=labeler_args,
                         task_configs=task_configs,
                         lf_configs=lf_configs,
                         task_graph_class=task_graph_class)
        """
        """
        self.metal_model_args = metal_model_args

    def _load_label_sources(self, exam_id_to_report):
        """
        """
        # stores all non-abstentions for downstream coo_matrix initialization
        sources_dict = {t: {'row_idxs': [], 'col_idxs': [], 'data': []}
                        for t in range(len(self.task_configs))}
        # stores the results of all label_fns for each report
        sources_df_dict = defaultdict(dict)

        deps = set()
        sources_cardinalities = []

        for t, task_config in enumerate(self.task_configs):
            task = task_config['task']
            print(f"Extracting sources for task `{task}`...")
            for i, (exam_id, report) in tqdm(enumerate(exam_id_to_report.items()),
                                             total=len(exam_id_to_report)):
                text = ' '.join([el.lower() for el in report['text']])
                # track all labels applied
                j = 0
                for lf_name, lf_config in self.lf_name_to_lf_config.items():
                    lf_fn = lf_config['fn']
                    lf_args = lf_config['args']

                    labels, edges = getattr(self.labeler, lf_fn)(text, **lf_args)
                    # add dependencies for new set of labels

                    # only add dependencies once
                    if t == 0 and i == 0:
                        edges = (np.array(edges) + j)
                        edges = [tuple(edge) for edge in edges]
                        assert deps.isdisjoint(set(edges)), (
                            'only one edge between each source pair permitted'
                        )
                        deps.update(edges)
                        sources_cardinalities += list(np.full_like(labels, lf_config['cardinality']))

                    if lf_name not in self.task_to_lf_names[task]:
                        labels = np.zeros_like(labels)

                    for k, label in enumerate(labels):
                        sources_df_dict[exam_id][f"{lf_name}_{k}"] = max(label, sources_df_dict[exam_id].get(f"{lf_name}_{k}", 0))
                        if label != 0:
                            sources_dict[t]['row_idxs'].append(i)
                            sources_dict[t]['col_idxs'].append(j)
                            sources_dict[t]['data'].append(label)
                        sources_cardinalities.append(task_config['cardinality'])
                        j += 1

        exam_ids, label_fn_to_label = list(zip(*sources_df_dict.items()))
        sources_df = pd.DataFrame(list(label_fn_to_label), index=list(exam_ids))

        # add chords
        if deps:
            g = TriangleGraph(deps, cardinalities=sources_cardinalities)
            g_chordal = g.triangulate()
            deps = g.edges

        label_sources = self._format_sources_dict(sources_dict,
                                                  len(exam_id_to_report))
        return sources_df, {"L": label_sources, "deps": deps}

    def _format_sources_dict(self, sources_dict, n):
        """
        """
        t = len(sources_dict)
        m = max([max(info['col_idxs']) if info['col_idxs'] else 0
                 for task, info in sources_dict.items()]) + 1 # get last non-abstain col

        L = []
        for task_idx in range(t):
            row_idxs = sources_dict[task_idx]['row_idxs']
            col_idxs = sources_dict[task_idx]['col_idxs']
            data = sources_dict[task_idx]['data']
            sources = coo_matrix((data, (row_idxs, col_idxs)), shape=(n, m))
            L.append(sources)
        return L

    def train(self, label_sources):
        """
        """
        L = label_sources['L']
        deps = label_sources['deps']
        self.label_model = MTLabelModel(seed=self.seed,
                                        task_graph=self.task_graph,
                                        **self.metal_model_args)
        self.label_model.train_model(L, seed=self.seed,
                                     deps=deps)
        return self.predict(label_sources)

    def predict(self, label_sources):
        """
        """
        L = label_sources['L']
        return self.label_model.predict_proba(L)


class ProgrammaticLabelsBuilder(LabelsBuilder):
    """
    """
    def __init__(self, dir, seed=123,
                 valid_targets_path="",
                 dataset_class="ReportDataset",
                 dataset_args={},
                 dataloader_configs=[],
                 task_configs=[],
                 labeler_class="Labeler",
                 labeler_args={},
                 lf_configs=[],
                 task_graph_class='TaskGraph',
                 strategy="simple"):
        super().__init__(dir, seed=seed,
                         valid_targets_path=valid_targets_path,
                         dataset_class=dataset_class,
                         dataset_args=dataset_args,
                         dataloader_configs=dataloader_configs,
                         labeler_class=labeler_class,
                         labeler_args=labeler_args,
                         task_configs=task_configs,
                         lf_configs=lf_configs,
                        task_graph_class=task_graph_class)

        self.strategy = strategy

    def _load_label_sources(self, exam_id_to_report):
        """
        Extracts label sources from the reports in a given split.

        Args:
            exam_id_to_report:  (dict)  matches exam_ids to their reports
        Returns:
            A tuple of (sources_df, label_sources). We leave it to the user
            to format sources_df and label_sources due to high variability in
            LabelBuilder approaches.
        """
        label_sources = OrderedDict()
        sources_df_dict = defaultdict(dict)
        for i, (exam_id, report) in tqdm(enumerate(exam_id_to_report.items()),
                                            total=len(exam_id_to_report)):
            text = ' '.join([el.lower() for el in report['text']])

            labeler_results = {}
            for lf_name, lf_config in self.lf_name_to_lf_config.items():
                lf_fn = lf_config['fn']
                lf_args = lf_config['args']

                labels, edges = getattr(self.labeler, lf_fn)(text, **lf_args)
                labeler_results[lf_name] = {
                    'labels': labels,
                    'deps': edges
                }
                for k, label in enumerate(labels):
                    sources_df_dict[exam_id][f"{lf_name}_{k}"] = max(label, sources_df_dict[exam_id].get(f"{lf_name}_{k}", 0))

            label_sources[exam_id] = labeler_results

        exam_ids, label_fn_to_label = list(zip(*sources_df_dict.items()))
        sources_df = pd.DataFrame(list(label_fn_to_label), index=list(exam_ids))
        return sources_df, label_sources

    def train(self, label_sources):
        """
        Trains the label model.

        Args:
            label_sources (object) user-defined inputs

        Returns:
            t-length list of [n,k_t] matrices where the (i,j) entry of the
                sth matrix represents the estimated P((Y_i)_s | \lambda_j(x_i))
        """
        return self.predict(label_sources)

    def predict(self, label_sources):
        """
        Outputs label model predictions.

        Args:
            label_sources (OrderedDict(dict)) contains the exam_id/lf_name
                along with a list of labels for the corresponding name.
                OrderedDict guarantees dataloader order.

        Returns:
            A t-length list of [n,k_t] matrices where the (i,j) entry of the
                sth matrix represents the estimated P((Y_i)_s | \lambda_j(x_i))
        """
        Y_ps = []
        for task_config in self.task_configs:
            task = task_config['task']
            lf_names = self.task_to_lf_names[task]

            cardinality = task_config['cardinality']
            y_t = np.zeros((len(label_sources), cardinality))
            for i, (exam_id, labeler_results) in enumerate(label_sources.items()):
                label_idx = self._label_example(task, labeler_results)
                y_t[i][label_idx] = 1.0
            Y_ps.append(y_t)
        return Y_ps

    def _label_example(self, task, labeler_results):
        """
        Returns a binary classification based on a set of labeler_results.

        Relies on the current self.strategy to determine final labeling
        mechanism. Only implements binary classification.

        TODO: Add multi-class compatibility
        """
        if self.strategy == "simple":
            equivocation = sum(labeler_results['contains_evidence']['labels'])
            indicator = labeler_results[f'{task}_indicator']['labels'][0]
            if indicator == 2 and equivocation == 0:
                return 1
            else:
                return 0
        else:
            raise ValueError(f'{self.strategy} strategy not implemented.')
