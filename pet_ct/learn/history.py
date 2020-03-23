"""
"""
import logging
import os
import time

import torch
import pandas as pd

from pet_ct.util.util import save_dict_to_json, flatten_nested_dicts

class TrainHistory:
    def __init__(self, dir):
        """
        Class wrapping a dataframe, which stores the progression of epochs in a training
        model. Call record_epoch after each epoch.
        """
        self.path = os.path.join(dir, "epochs.csv")
        if os.path.exists(self.path):
            self.load()
        else:
            self.df = None

    def record_epoch(self, metrics, learning_rate):
        """
        """
        metrics = flatten_nested_dicts(metrics)
        if self.df is None:
            columns = ['time'] + sorted(list(metrics.keys()))
            self.df = pd.DataFrame(columns=columns)
        metrics['time'] = time.strftime("%H:%M:%S %Y-%m-%d", time.localtime())
        metrics['learning_rate'] = learning_rate
        self.df = self.df.append(metrics, ignore_index=True)

    def load(self):
        """
        """
        self.df = pd.read_csv(self.path, index_col=0)

    def write(self):
        """
        """
        if self.df is None:
            logging.error("No dataframe loaded.")
        else:
            self.df.to_csv(self.path)