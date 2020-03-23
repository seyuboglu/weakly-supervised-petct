"""
"""
import os

import torch
import numpy as np
import pandas as pd

from pet_ct.util.util import Process, ensure_dir_exists


class EmbeddingsBuilder(Process):

    def __init__(self, dir, seed=123, out_dir="/data/asdf", reload_experiment_dir=""):
        """
        """
        self.seed = seed
        self.reload_experiment_dir = reload_experiment_dir

    def run(self, overwrite=False):
        """
        """
        for i, (x, y, info) in dataloader:
            emb = model.encoder(x)

            patient_dir = path.join(self.out_dir, patient_id)
            ensure_dir_exists()
            np.save(emb, path.join(patient_dir, exam_id))