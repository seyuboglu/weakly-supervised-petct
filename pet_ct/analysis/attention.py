from datetime import datetime

import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import plotly.offline as ply
import plotly.graph_objs as go
import colorlover as cl
import pandas as pd
import cv2
import torch
from torch.nn import ReLU
from sklearn.metrics import roc_curve
import matplotlib.animation as animation
import os
matplotlib.rcParams['animation.embed_limit'] = 2**128

from pet_ct.util.util import place_on_gpu, place_on_cpu
from pet_ct.learn.history import TrainHistory


def compute_mt_attention(model, inputs, targets, task=None, device=0):
    """
    """
    inputs = place_on_gpu(inputs, device=device)
    model = model.to(device=device)
    model.eval()
    for task_head in model.decoder.task_heads.values():
        task_head.region_aware = True
    model_output = model(inputs, targets)
    #attention_module.region_aware = region_aware
    if task:
        return model_output[task]['attn_scores']
    else:
        return {task: output["attn_scores"] for task, output in model_output.items()}


def compute_attention(model, attention_module, inputs, targets, device=0):
    """
    """
    inputs = place_on_gpu(inputs, device=device)
    model = model.to(device=device)
    model.eval()

    # require gradient on scan so we can backpropagate through the pixels
    scan = inputs["scan"] if isinstance(inputs, dict) else inputs
    scan.requires_grad = True

    attention_module.keep_attention = True

    # forward pass
    model_output = model(inputs, targets)

    attention_module.keep_attention = False
    attention_probs = attention_module.attention_probs[-1]
    attention_module.keep_attention = False
    attention_module.attention_probs = []

    return attention_probs