"""
Handles loss balancing and gradient updates for multitask learning.
"""
from collections import defaultdict

import torch.nn as nn
import torch
import numpy as np


class Balancer(nn.Module):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, losses):
        """
        """
        loss = self._get_loss(losses)
        self._update()
        return loss

    def _get_loss(self, losses):
        """
        """
        raise NotImplementedError

    def _update(self):
        """
        """
        raise NotImplementedError


class RandomBalancer(Balancer):
    """
    """
    def __init__(self, task_probs={}):
        """
        """
        super().__init__()

        self.task_probs = None
        if task_probs:
            tasks = []
            probs = []
            for task, p in task_probs.items():
                tasks.append(task)
                probs.append(p)
            probs_normalized = np.array(probs) / sum(probs)

            self.task_probs = {task: probs_normalized[i] for i, task in enumerate(tasks)}

    def _get_loss(self, losses):
        """
        """
        tasks = list(losses.keys())

        if self.task_probs:
            probs = [self.task_probs[task] for task in tasks]
            task = np.random.choice(tasks, p=probs)
        else:
            task = np.random.choice(tasks)

        loss = losses[task]
        return loss

    def _update(self):
        """
        """
        pass


class OrderedBalancer(Balancer):
    """
    """
    def __init__(self, task_periods=[]):
        """
        """
        super().__init__()

        self.task_periods = task_periods

        self.curr_period = 0
        self.curr_task = self.task_periods[self.curr_period]['task']
        self.curr_iter = 0

    def _get_loss(self, losses):
        """
        """
        loss = losses[self.curr_task]
        return loss

    def _update(self):
        """
        """
        self.curr_iter += 1
        if self.curr_iter % self.task_periods[self.curr_period]['iters'] == 0:
            self.curr_period = (self.curr_period + 1) % len(self.task_periods)
            self.curr_task = self.task_periods[self.curr_period]['task']
            self.curr_iter = 0


class LBTWBalancer(Balancer):
    """
    Algorithm as proposed in Liu et al.'s 'Loss-Balanced Task Weighting to
    Reduce Negative Transfer in Multi-Task Learning.'

    link: https://www.aaai.org/Papers/AAAI/2019/SA-LiuS.371.pdf
    """
    def __init__(self, alpha=None, batches_per_epoch=100):
        """
        """
        super().__init__()

        if type(alpha) != dict:
            self.alpha = defaultdict(lambda: alpha)
        else:
            self.alpha = alpha

        self.batches_per_epoch = batches_per_epoch
        self.batch_0_loss = {}
        self.batch_num = 0

    def _get_loss(self, losses):
        """
        """
        weights = {}
        print('--')
        print('LBTWBalancer')
        for task, loss in losses.items():
            if self.batch_num == 0 or self.batch_0_loss[task] == 0:
                self.batch_0_loss[task] = loss.cpu().detach()
                weights[task] = 1.0
            else:
                # QUESTION: we should detach the loss here shouldn't we?
                weights[task] = (loss / self.batch_0_loss[task]) ** \
                                 self.alpha[task]
            print(f'task: {task}')
            print(f'- weight:       \t{weights[task]}')
            print(f'- weighted loss:\t{weights[task] * loss}')
        loss = sum([weights[task] * losses[task] for task in losses.keys() if losses[task] != 0.0])
        return loss

    def _update(self):
        """
        """
        self.batch_num = (self.batch_num + 1) % self.batches_per_epoch