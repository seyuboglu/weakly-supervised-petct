"""
"""
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pet_ct.util.util import place_on_gpu


class InterTaskSubsetStrategy(nn.Module):

    def __init__(self, task, target_task="primary", target_subset=[]):
        """
        Uses the labels of a different task to determine task head activation.
        """
        super().__init__()
        self.target_task = target_task
        self.target_subset = target_subset

    def forward(self, targets):
        """
        """
        target_task_Y = targets[self.target_task].cpu().detach()
        batch_size, k = target_task_Y.shape

        if not self.target_subset:
            return torch.ones(batch_size)

        # NOTE: only works with SoftCELoss (k > 1)
        # TODO: make compatible with index-based labels
        relevant_examples = torch.zeros(batch_size,
                                        dtype=torch.uint8) # bit-ops
        _, argmaxes = target_task_Y.max(dim=1)
        for relevant_target in self.target_subset:
            relevant_examples |= argmaxes == relevant_target
        return relevant_examples


class BinaryBalancingStrategy(nn.Module):

    def __init__(self, task, task_targets, desired_frac_pos=0.5):
        """
        Class balancing for binary tasks.
        """
        super().__init__()
        self.task = task
        self._calculate_lambda_coeffs(task_targets, desired_frac_pos)

    def _calculate_lambda_coeffs(self, task_targets, desired_frac_pos):
        probs = task_targets.values[:, 1]

        p_pos = np.sum(probs >= 0.5) / len(probs)
        p_neg = 1 - p_pos

        self.lambda_p = 1
        self.lambda_n = ((1-desired_frac_pos) * p_pos) / (desired_frac_pos * p_neg)
        # lambda is a sampling probability, thus has a maximum value of 1
        if self.lambda_n > 1:
            self.lambda_p = ((desired_frac_pos) * p_neg) / ((1-desired_frac_pos) * p_pos)
            self.lambda_n = 1

    def forward(self, targets):
        """
        """
        task_targets = targets[self.task].cpu().detach()
        batch_size, k = task_targets.shape

        # TODO: make compatible with index-based labels
        relevant_examples = torch.zeros(batch_size,
                                        dtype=torch.uint8) # bit-ops
        for idx in range(batch_size):
            if (task_targets[idx, 1] < 0.5):
                relevant_examples[idx] = torch.rand(1) < self.lambda_n
            else:
                relevant_examples[idx] = torch.rand(1) < self.lambda_p

        return relevant_examples


class WeightedRandomSamplerBinaryBalancingStrategy(nn.Module):

    def __init__(self, task, targets_df, desired_frac_pos=0.5,
                 weight_task='primary', weight_task_frac_pos=0.5):
        """
        Class balancing for binary tasks when using a sampler.

        Not a subclass of BinaryBalancingStrategy because of weirdness in lambda
        calcuations requiring all targets.
        """
        super().__init__()
        self.task = task
        self.weight_task = weight_task
        self._calculate_lambda_coeffs(targets_df, desired_frac_pos, weight_task_frac_pos)


    def _calculate_lambda_coeffs(self, targets_df, desired_frac_pos, weight_task_frac_pos):
        """
        Finds the lambda coefficients, which represent p(draw_[pos|neg] | take).

        This differs from BinaryBalancingStrategy because we must account for
        the weighted random sampling, which alters the probability of receiving
        particular classes. We must diverge from using true counts in the
        targets_df (see `BinaryBalancingStrategy._calculate_lambda_coeffs`
        for comparison). Solves the following equation:

        p(draw_pos) = p(draw_pos | weight_draw_pos) * p(weight_draw_pos)
        """
        task_targets = targets_df[self.task].values[:, 1]
        task_pos_targets = task_targets >= 0.5
        task_neg_targets = task_targets < 0.5

        weight_task_targets = targets_df[self.weight_task].values[:, 1]
        weight_task_pos_targets = weight_task_targets >= 0.5
        weight_task_num_pos = sum(weight_task_pos_targets)
        weight_task_neg_targets = weight_task_targets < 0.5
        weight_task_num_neg = sum(weight_task_neg_targets)

        # solving p(draw_pos | weight_draw_pos) and p(draw_pos | weight_draw_neg)
        p_pos_given_pos = sum(task_pos_targets & weight_task_pos_targets) / weight_task_num_pos
        p_pos_given_neg = sum(task_pos_targets & weight_task_neg_targets) / weight_task_num_neg

        # solving p(draw_pos, weight_draw_pos) and p(draw_pos, weight_draw_neg)
        p_pos_pos = p_pos_given_pos * weight_task_frac_pos
        p_pos_neg = p_pos_given_neg * (1.0 - weight_task_frac_pos)

        # solving p(draw_pos) and p(draw_neg)
        p_pos = p_pos_pos + p_pos_neg
        p_neg = 1 - (p_pos)

        self.lambda_p = 1
        self.lambda_n = ((1-desired_frac_pos) * p_pos) / (desired_frac_pos * p_neg)
        # lambda is a sampling probability, thus has a maximum value of 1
        if self.lambda_n > 1:
            self.lambda_p = (desired_frac_pos * p_neg) / ((1-desired_frac_pos) * p_pos)
            self.lambda_n = 1

    def forward(self, targets):
        """
        """
        task_targets = targets[self.task].cpu().detach()
        batch_size, k = task_targets.shape

        # TODO: make compatible with index-based labels
        relevant_examples = torch.zeros(batch_size,
                                        dtype=torch.uint8) # bit-ops
        for idx in range(batch_size):
            if (task_targets[idx, 1] < 0.5):
                relevant_examples[idx] = torch.rand(1) < self.lambda_n
            else:
                relevant_examples[idx] = torch.rand(1) < self.lambda_p

        return relevant_examples


class TaskHeadManager(nn.Module):

    def __init__(self, task_configs=[], targets_path=None):
        """
        Determines when to compute loss for each task head given batch of exams.
        """
        super().__init__()
        self.task_to_weight = defaultdict(lambda: 1.0)
        self.task_to_strategy_fn = defaultdict(dict)
        if targets_path:
            self.targets_df = pd.read_csv(targets_path, index_col=0,
                                          header=[0, 1], skipinitialspace=True)
        for task_config in task_configs:
            task = task_config['task']
            self.task_to_weight[task] = task_config.get('loss_weight', 1.0)

            strategy_class = task_config.get('loss_strategy_class', None)
            strategy_args = task_config.get('loss_strategy_args', {})
            if strategy_class == "BinaryBalancingStrategy":
                task_targets = self.targets_df[task]
                self.task_to_strategy_fn[task] = globals()[strategy_class](task, task_targets=task_targets,
                                                                           **strategy_args)
            elif strategy_class == 'WeightedRandomSamplerBinaryBalancingStrategy':
                self.task_to_strategy_fn[task] = globals()[strategy_class](task, targets_df=self.targets_df,
                                                                           **strategy_args)
            elif strategy_class:
                self.task_to_strategy_fn[task] = globals()[strategy_class](task, **strategy_args)
            else:
                self.task_to_strategy_fn[task] = None

    def forward(self, task, targets):
        """
        """
        if not self.task_to_strategy_fn[task]:
            batch_size, _ = targets[task].shape
            relevance = torch.ones(batch_size,
                                   dtype=torch.uint8,
                                   device=targets[task].device) # bit-ops
        else:
            relevance = self.task_to_strategy_fn[task](targets)
        relevance = relevance.to(targets[task].dtype) * self.task_to_weight[task]
        if targets[task].is_cuda:
            relevance = place_on_gpu(relevance, targets[task].device)
        return relevance
