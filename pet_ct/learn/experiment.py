"""
"""
import logging
import os
from time import time
import json
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from metal import EndModel
import click
from tensorboardX import SummaryWriter

import pet_ct.learn.datasets as datasets
import pet_ct.learn.dataloaders as dataloaders
import pet_ct.model.losses as losses
import pet_ct.model.models as models
from pet_ct.learn.history import TrainHistory
from pet_ct.util.util import (
    Process,
    save_dict_to_json,
    ensure_dir_exists,
    get_latest_file,
)
from pet_ct.analysis.metrics import Metrics


def load_experiment_params():
    if task_configs is not None:
        new_task_configs = []
        for task_config in task_configs:
            new_task_config = default_task_config.copy()
            new_task_config.update(task_config)
            new_task_configs.append(new_task_config)
        task_configs = new_task_configs

        model_args["task_configs"] = task_configs


class Experiment(Process):
    """
    """

    def __init__(
        self,
        dir,
        dataset_class="BinaryDataset",
        dataset_args={},
        dataloader_configs=[],
        train_args={},
        evaluate_args={},
        model_class="BaseModel",
        model_args={},
        task_configs=None,
        default_task_config={},
        primary_metric="roc_auc",
        reload_weights="best",
        cuda=True,
        devices=[0],
        seed=123,
        remote_model_dir="/data4/data/fdg-pet-ct/models",
    ):
        """
        Initializes the Trainer subclass of Process.
        """
        super().__init__(dir)

        # load instance variables
        self.train_args = train_args
        self.evaluate_args = evaluate_args

        # set the model_dir
        self.model_dir = None
        self.remote_model_dir = remote_model_dir

        # set random seed for reproducible experimentsm
        self.cuda = cuda
        self.device = devices[0]
        self.devices = devices
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # distribute shared params to other params
        if task_configs is not None:
            new_task_configs = []
            for task_config in task_configs:
                new_task_config = default_task_config.copy()
                new_task_config.update(task_config)
                new_task_configs.append(new_task_config)
            task_configs = new_task_configs

            model_args["task_configs"] = task_configs
            dataset_args["task_configs"] = task_configs
            self.primary_task = task_configs[0]["task"]
        self.primary_metric = primary_metric

        # load dataloaders
        self._build_dataloaders(dataset_class, dataset_args, dataloader_configs)

        logging.info("Building model")
        self._build_model(model_class, model_args, reload_weights=reload_weights)

        # records epoch data in csv
        self.train_history = TrainHistory(self.dir)

        # creates log dir
        self.log_dir = os.path.join(self.dir, "logs")
        ensure_dir_exists(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # timestamp acts as checkpoint name
        experiment_time = str(time()).replace(".", "_")
        self.experiment_t = f"{uuid4()}-time{experiment_time}"
        logging.info("-" * 30)

    def __del__(self):
        self.writer.close()

    def _link_model(self, filepath, symlink):
        """ Creates symlink from weights_dir to
        """
        if not os.path.islink(symlink):
            os.symlink(filepath, symlink)

    def is_trained(self):
        """ Returns true if the model has been trained for at least one epoch, false
        otherwise.
        """
        return os.path.isdir(os.path.join(self.dir, "last"))

    def _build_dataloaders(self, dataset_class, dataset_args, dataloader_configs):
        """
        """
        self.datasets = {}
        self.dataloaders = {}
        for dataloader_config in dataloader_configs:
            split = dataloader_config["split"]
            dataloader_class = dataloader_config["dataloader_class"]
            dataloader_args = dataloader_config["dataloader_args"]
            logging.info(f"Loading {split} data")
            self._build_dataloader(
                split, dataset_class, dataset_args, dataloader_class, dataloader_args
            )

    def _build_dataloader(
        self, split, dataset_class, dataset_args, dataloader_class, dataloader_args
    ):
        """
        """
        # create dataset
        dataset = getattr(datasets, dataset_class)(split=split, **dataset_args)
        print(len(dataset))
        self.datasets[split] = dataset

        dataloader = getattr(dataloaders, dataloader_class)(dataset, **dataloader_args)
        self.dataloaders[split] = dataloader

    def _build_model(self, model_class, model_args, reload_weights="best"):
        """
        Builds the model. If the model was previously trained, it is loaded from a
        previous model.
        """
        model_class = getattr(models, model_class)
        self.model = model_class(cuda=self.cuda, devices=self.devices, **model_args)
        if self.is_trained() and reload_weights and len(reload_weights):
            print(f"Reloading {reload_weights}...")
            self._load_model_weights(name=reload_weights)

    def _run(self, overwrite=False, mode=None, train_split="train", eval_split="valid"):
        """
        """
        if mode == "train":
            self.train(train_split, eval_split, overwrite)
        elif mode == "eval":
            self.evaluate(eval_split)
        else:
            logging.info("Please specify a mode with --mode train or --mode eval")

    def predict(self, split="predict", task="primary"):
        """
        """
        for inputs, targets, preds, info in self.model.predict_many(
            self.dataloaders[split]
        ):
            example = {}
            targets = targets[task].cpu().numpy()
            preds = preds[task].cpu().numpy()
            print(inputs)
            print(np.where((targets != -1)))
            print(targets[(targets != -1)])
            print(preds[(targets != -1)])
            print(info)

    def evaluate(self, eval_split="valid"):
        """
        """
        metrics = self.model.score(self.dataloaders[eval_split], **self.evaluate_args)
        if self.model_dir:
            self._save_metrics(self.model_dir, metrics, f"{eval_split}")
        else:
            metrics_path = os.path.join(self.dir, "best")
            ensure_dir_exists(metrics_path)
            print(metrics_path)
            self._save_metrics(metrics_path, metrics, f"{eval_split}")

    def train(self, train_split="train", valid_split="valid", overwrite=False):
        """
        """
        assert not self.is_trained() or overwrite, "The model has already been trained."
        # initializes directories
        if self.is_trained():
            metrics_path = os.path.join(self.dir, "best", "valid_metrics.json")
            with open(metrics_path) as f:
                val_metrics = json.load(f)
            best_score = val_metrics[self.primary_task][self.primary_metric]
        else:
            best_score = None

        # get initial validation metrics
        val_metrics = self.model.score(
            self.dataloaders[valid_split], **self.evaluate_args
        )
        self.train_history.record_epoch(
            {"valid": val_metrics.metrics}, self.model.scheduler.get_lr()[0]
        )
        self.train_history.write()

        metrics = None
        for epoch_num, train_metrics in enumerate(
            self.model.train_model(
                dataloader=self.dataloaders[train_split],
                writer=self.writer,
                **self.train_args,
            )
        ):
            val_metrics = self.model.score(
                self.dataloaders[valid_split], **self.evaluate_args
            )

            # update dataloader
            if hasattr(self.dataloaders[train_split], "update_epoch"):
                pass
                # self.dataloaders[train_split].update_epoch(val_metrics)

            self._save_model(name="last")
            self._save_weights(name="last")
            self._save_epoch(epoch_num, train_metrics, val_metrics, name="last")

            metrics = {"train": train_metrics.metrics, "valid": val_metrics.metrics}
            self.train_history.record_epoch(metrics, self.model.scheduler.get_lr()[0])
            self.train_history.write()

            curr_score = val_metrics.get_metric(self.primary_metric, self.primary_task)
            if best_score is None or curr_score > best_score:
                self._save_model(name="best")
                self._save_weights(name="best")
                self._save_epoch(epoch_num, train_metrics, val_metrics, name="best")
                best_score = curr_score

                for epoch_limit in range(epoch_num, 5):
                    self._save_model(name=f"best_{epoch_limit + 1}")
                    self._save_weights(name=f"best_{epoch_limit + 1}")
                    self._save_epoch(
                        epoch_num,
                        train_metrics,
                        val_metrics,
                        name=f"best_{epoch_limit + 1}",
                    )

        return metrics

    def get_history(self):
        """
        """
        return TrainHistory(self.dir)

    def _save_weights(self, name="last"):
        """
        """
        ensure_dir_exists(self.remote_model_dir)
        remote_weights_path = os.path.join(
            self.remote_model_dir, f"{self.experiment_t}_{name}_weights"
        )
        self.model.save_weights(remote_weights_path)

        self.model_dir = os.path.join(self.dir, name)
        ensure_dir_exists(self.model_dir)
        link_weights_path = os.path.join(self.model_dir, "weights.link")
        self._link_model(remote_weights_path, link_weights_path)

    def _save_model(self, name="last"):
        """ Saves the model and symlinks to experiment directory.

        Args:
            - is_best (bool)    True when model exceeds best_score. Saves twice.
        """
        remote_model_path = os.path.join(
            self.remote_model_dir, f"{self.experiment_t}_{name}_model"
        )
        self.model.save(remote_model_path)

        self.model_dir = os.path.join(self.dir, name)
        ensure_dir_exists(self.model_dir)
        link_model_path = os.path.join(self.model_dir, "model.link")
        self._link_model(remote_model_path, link_model_path)

    def _load_model_weights(self, name="best"):
        """
        """
        self.model_dir = os.path.join(self.dir, name)
        model_path = os.path.join(self.model_dir, "weights.pth.tar")
        if not os.path.isfile(model_path):
            model_path = os.path.join(self.model_dir, "weights.link")
        if not os.path.exists(model_path):
            raise (Exception(f"Checkpoint file does not exist {model_path}."))
        self.model.load_weights(model_path, device=self.device)

    def _load_model(self, model_class, name="best"):
        """
        """
        self.model_dir = os.path.join(self.dir, name)
        model_path = os.path.join(self.model_dir, "model.pth.tar")
        if not os.path.isfile(model_path):
            model_path = os.path.join(self.model_dir, "model.link")
        if not os.path.exists(model_path):
            raise (Exception(f"Checkpoint file does not exist {model_path}."))
        self.model = model_class.load(model_path)

    def _save_epoch(self, epoch_num, train_metrics, valid_metrics, name="last"):
        """
        """
        save_dir = os.path.join(self.dir, name)
        ensure_dir_exists(save_dir)
        logging.info("Saving checkpoint...")

        self.writer.export_scalars_to_json(get_latest_file(self.log_dir))
        # records the most recent training epoch
        self._save_metrics(save_dir, train_metrics, "train")
        self._save_metrics(save_dir, valid_metrics, "valid")

    def _save_metrics(self, metrics_dir, metrics, split):
        """
        """
        save_dict_to_json(
            os.path.join(metrics_dir, f"{split}_metrics.json"), metrics.metrics
        )
        preds_df = metrics.get_preds()
        preds_df.to_csv(os.path.join(metrics_dir, f"{split}_preds.csv"))
