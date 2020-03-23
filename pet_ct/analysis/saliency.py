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

from pet_ct.util.util import place_on_gpu, place_on_cpu
from pet_ct.learn.history import TrainHistory


def generate_gradient(model, inputs, targets, target_task="primary"):
    """
    Generates
    """
    # require gradient on scan so we can backpropagate through the pixels
    scan = inputs["scan"] if isinstance(inputs, dict) else inputs
    scan.requires_grad  = True

    # forward pass
    output = model(inputs, targets)
    model.zero_grad()

    # backward pass
    targets = targets[target_task] if isinstance(targets, dict) else targets
    target_class = targets[0]

    # if not an index but a softmax (probabilistic case)
    if len(target_class.shape) > 0:
        target_class = target_class.argmax()
    print(output)
    output = output[target_task] if isinstance(output, dict) else output
    print(output)
    if type(output) == dict:
        output = output['out']

    # get first element in batch
    output = output[0]

    output[target_class].backward()

    grad, scan = place_on_cpu([scan.grad, scan])
    grad = torch.tensor(grad)

    scan.requires_grad = False


    return grad, scan

def generate_integrated_gradient(model, inputs, targets,
                                  target_task="primary", steps=50,
                                  baseline=None):
    """
    """
    scan = inputs["scan"] if isinstance(inputs, dict) else inputs

    if baseline is None:
        baseline = torch.zeros_like(scan)
    expanded_scan = torch.cat([baseline + (float(i) / steps) * (scan-baseline)
                     for i in range(0, steps + 1)], dim=0)

    # require gradient on scan so we can backpropagate through the pixels
    expanded_scan.requires_grad  = True

    # forward pass
    output = model(expanded_scan, targets)
    model.zero_grad()

    # backward pass
    targets = targets[target_task] if isinstance(targets, dict) else targets
    target_class = targets[0]

    # if not an index but a softmax (probabilistic case)
    if len(target_class.shape) > 0:
        target_class = target_class.argmax()
    output = output[target_task] if isinstance(output, dict) else output
    if type(output) == dict:
        output = output['out']

    # get first element in batch
    output[:, target_class].backward()
    grad, scan = place_on_cpu([expanded_scan.grad, scan])
    avg_grad = torch.mean(grad[:-1], axis=0)
    integrated_grad = (scan - baseline) * avg_grad

    integrated_grad = torch.tensor(integrated_grad)

    scan.requires_grad = False

    return integrated_grad, scan



class GuidedBackProp(object):
    """
       Produces gradients through scan
       @author: Utku Ozbulak - github.com/utkuozbulak
    """
    def __init__(self, model):
        """
        """
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []

        # put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        """
        """
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, inputs, targets,
                           target_task=None, token_idx=None, device=0):
        """
        Generates gradients through the scan for the first element in the batch
        specified by inputs and targets. If batch size is greater than 1, only gradients
        for the first example will be computed. Supports multi-task output via the
        target_task argument. Supports tasks with multiple outputs (as in masked language
        modeling or natural language generation).

        @inputs (dict or torch.Tensor) the 3D input scan. if dict, should contain key "scan".
        @targets (dict, torch.Tensor) target tensor or dict
        @target_task    (None or str) required if targets is dict
        @token_idx      (None or int) optional: specific
        @grad     (torch.Tensor, torch.Tensor) the gradient through the scan on the cpu
        @scan   (torch.Tensor) the scan itself through the scan on the cpu
        """
        inputs = place_on_gpu(inputs, device=device)
        self.model = self.model.to(device=device)

        # require gradient on scan so we can backpropagate through the pixels
        scan = inputs["scan"] if isinstance(inputs, dict) else inputs
        scan.requires_grad  = True

        # forward pass
        output = self.model(inputs, targets)
        self.model.zero_grad()

        # backward pass
        targets = targets[target_task] if isinstance(targets, dict) else targets
        target_class = targets[0, token_idx] if token_idx is not None else targets[0]

        # if not an index but a softmax (probabilistic case)
        if len(target_class.shape) > 0:
            target_class = target_class.argmax()

        output = output[target_task] if target_task is not None else output
        if type(output) == dict:
            output = output['out']
        output = output[0, token_idx] if token_idx is not None else output[0]



        output[target_class].backward()

        scan.requires_grad = False
        grad, scan = place_on_cpu([scan.grad, scan])

        # empty relu outputs so we don't leak CPU memory
        self.forward_relu_outputs = []

        return grad, scan


