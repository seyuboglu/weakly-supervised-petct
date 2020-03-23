"""
"""
import logging
import os

import pandas as pd
import torch
from torch import nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm

from pet_ct.data.h5_dataset import H5Dataset
from pet_ct.learn.experiment import Experiment
from pet_ct.analysis.visualization import plot_training_curve
from pet_ct.learn.datasets import BaseDataset
from pet_ct.util.util import place_on_gpu, log_cuda_memory, log_predictions


class ExperimentViewer(Experiment):

    def __init__(self, dir,
                 dataset_class="BinaryDataset", dataset_args={},
                 dataloader_configs=[],
                 train_args={}, evaluate_args={},
                 model_class="BaseModel", model_args={},
                 reload_weights='best', cuda=True, devices=[0],
                 seed=123, remote_model_dir="/data4/data/fdg-pet-ct/models"):
        """
        """
        super().__init__(dir, dataset_class, dataset_args, dataloader_configs,
                         train_args, evaluate_args, model_class, model_args,
                         reload_weights, cuda, devices, seed, remote_model_dir)

    def plot_training_curves(self, plot_configs):
        """
        """
        history = self.get_history()
        for plot_config in plot_configs:
            plot_training_curve(history, **plot_config)

    def get_prediction(self, exam_id: str, task: str="primary",
                       split: str="valid", name: str="best"):
        """
        """
        predictions = pd.read_csv(os.path.join(self.dir, f"{name}/{split}_preds.csv"),
                                  index_col=[1], header=[0, 1])
        target = predictions.loc[exam_id][task, "target"]
        correct_prob = predictions.loc[exam_id][task, "correct_prob"]

        return target, correct_prob

    def get_report(self, exam_id: str, split: str="valid"):
        """
        """
        h5_dataset = H5Dataset(self.datasets[split].dataset_name,
                               self.datasets[split].data_dir,
                               mode="read")
        report = h5_dataset.read_reports(exam_id)

        return report