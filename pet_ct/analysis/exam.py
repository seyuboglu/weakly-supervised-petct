import logging
import json
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
from pet_ct.util.util import place_on_gpu, log_cuda_memory, log_predictions, ensure_dir_exists


def save_exam(exam: dict, experiment_dir: str):
    """
    """
    exam_dir = os.path.join(experiment_dir, "exams", exam["exam_id"])

    ensure_dir_exists(exam_dir)

    for saliency_map in exam.get("saliency_maps", []):
        ani = saliency_map["ani"]
        ani_file_name = f"saliency_ani_{saliency_map['ani_task']}.gif"
        ani.save(os.path.join(exam_dir, ani_file_name), writer='imagemagick')
        saliency_map["ani"] = ani_file_name

        #TODO: save saliency plot
    
    print( [prob for prob in exam["targets"][saliency_map['ani_task']].data.numpy()])
    with open(os.path.join(exam_dir, "exam.json"), 'w') as f:
        exam_summary = {
            "exam_id": exam["exam_id"],
            "patient_id": exam["patient_id"],
            "task": saliency_map['ani_task'],
            "targets": [float(prob) for prob in exam["targets"][saliency_map['ani_task']].data.numpy()],
            "target_prob": exam["target_probs"]
        }
        json.dump(exam_summary, f, indent=4)

    

