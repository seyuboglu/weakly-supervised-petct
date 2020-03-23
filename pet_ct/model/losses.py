"""
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss
from metal.end_model.loss import SoftCrossEntropyLoss

import pet_ct.model.loss_balancers as balancers
from pet_ct.model.task_head_managers import TaskHeadManager
from pet_ct.util.util import place_on_gpu


class MTWeightedCrossEntropyLoss(SoftCrossEntropyLoss):

    def __init__(self, class_reduction='mean'):
        """
        """
        # separated for exam-level task head activation conditions
        self.class_reduction = class_reduction
        super().__init__(reduction='none')

    def forward(self, inputs, targets, task_to_weights):
        """
        """
        losses = self._compute_loss(inputs, targets)
        loss = self._balance(losses, targets, task_to_weights)
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        losses = {}
        for task, Y in targets.items():
            Y_hat = inputs[task]
            losses[task] = super().forward(Y_hat, Y)
        return losses

    def _balance(self, losses, targets, task_to_weights):
        """
        """
        for task, Y in targets.items():
            # broadcast multiply task-specific coefficients (uses all targets)
            losses[task] = losses[task] * task_to_weights[task]

            if self.class_reduction == "mean":
                losses[task] = losses[task].mean()
            if self.class_reduction == "sum":
                losses[task] = losses[task].sum()


        loss = sum(losses.values())
        return loss


class MTCrossEntropyLoss(SoftCrossEntropyLoss):

    def __init__(self, class_weight=None, class_reduction='mean',
                 task_configs=[], manager_args={}, balancer_class=None,
                 balancer_args={}):
        """
        """
        # separated for exam-level task head activation conditions
        self.class_reduction = class_reduction
        super().__init__(weight=class_weight, reduction='none')

        self.task_head_manager = TaskHeadManager(task_configs, **manager_args)
        if balancer_class:
            self.balancer = getattr(balancers, balancer_class)(**balancer_args)
        else:
            self.balancer = None

    def forward(self, inputs, targets):
        """
        """
        losses = self._compute_loss(inputs, targets)
        loss = self._balance(losses, targets)
        # print('--')
        # print(f'final loss:\t{loss}')
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        # print('--')
        # print('MTCrossEntropyLoss')
        losses = {}
        for task, Y in targets.items():
            Y_hat = inputs[task]
            # print(f"task: {task}")
            # print(f'- true:\t{Y[0,1]}')
            # print(f'- pred:\t{nn.functional.softmax(Y_hat, dim=1)[0,1]}')
            losses[task] = super().forward(Y_hat, Y)
            # print(f"- loss:\t{losses[task]}")
        return losses

    def _balance(self, losses, targets):
        """
        """
        # print('--')
        # print('TaskHeadManager')
        for task, Y in targets.items():
            # broadcast multiply task-specific coefficients (uses all targets)
            losses[task] = losses[task] * self.task_head_manager(task, targets)
            if self.class_reduction == "mean":
                losses[task] = losses[task].mean()
            if self.class_reduction == "sum":
                losses[task] = losses[task].sum()
            # print(f'task: {task}')
            # print(f'- loss:\t{losses[task]}')
        if self.balancer is not None:
            loss = self.balancer(losses)
        else:
            loss = sum(losses.values())
        return loss


class MTCrossEntropyWeightedBehaviorLoss(MTCrossEntropyLoss):

    def __init__(self, class_weight=None, class_reduction='mean',
                 task_configs=[], manager_args={}, balancer_class=None,
                 balancer_args={}):
        """
        Weights error modes differently as specified in `task_configs`.

        Similar to assigning class weights, but does not give a boost when
        things are correctly classified...
        """
        super().__init__(class_weight, class_reduction,
                         task_configs, manager_args,
                         balancer_class, balancer_args)

        self.task_to_fp_weight = {}
        self.task_to_fn_weight = {}
        for task_config in task_configs:
            task = task_config['task']
            fp_weight = task_config.get('fp_weight', 1.0)
            fn_weight = task_config.get('fn_weight', 1.0)
            self.task_to_fp_weight[task] = fp_weight
            self.task_to_fn_weight[task] = fn_weight

    def _compute_loss(self, inputs, targets):
        """
        """
        Y_hats = {}
        for task, _ in targets.items():
            if type(inputs[task]) == dict:
                Y_hats[task] = inputs[task]['out']
            else:
                Y_hats[task] = inputs[task]
        classification_losses = super()._compute_loss(Y_hats, targets)
        print('--')
        print('MTCrossEntropyWeightedBehaviorLoss')
        losses = {}
        for task, Y in targets.items():
            Y_hat = inputs[task]
            print(f"task: {task}")
            losses[task] = classification_losses[task]

            Y_hard = Y.max(dim=1)[1]
            Y_hat_hard = Y_hat.max(dim=1)[1]
            for idx, exam_loss in enumerate(losses[task]):
                if Y_hard[idx] != Y_hat_hard[idx]:
                    print(f'- true: {Y_hard[idx]}, {Y[idx]}')
                    print(f'- pred: {Y_hat_hard[idx]}, {Y_hat[idx]}')
                    print(f'- unweighted loss: {losses[task][idx]}')
                    if Y_hard[idx] == 0: # FP
                        print(f'- false positive. weighting {losses[task][idx]} by {self.task_to_fp_weight[task]}')
                        losses[task][idx] *= self.task_to_fp_weight[task]
                    if Y_hard[idx] == 1: # FN
                        print(f'- false negative. weighting {losses[task][idx]} by {self.task_to_fn_weight[task]}')
                        losses[task][idx] *= self.task_to_fn_weight[task]
                    print(f'- behavioral loss: {losses[task][idx]}')

            print(f"- loss:\t{losses[task]}")
        return losses


class RegionPenaltyLoss(nn.Module):

    def __init__(self, scale=1.0):
        """
        Multiplicative penalty.

        Penalizes "forbidden" regions instead of exact distribution matches.
        Optionally used in tandem with MTCrossEntropyRegionAwareLoss.
        `scale` param allows caller to scale the loss in order to match
        magnitude of other loss terms
        """
        super().__init__()
        self.scale = scale

    def forward(self, preds, targets):
        """
        """
        batch_size = preds.shape[0]

        penalty = torch.abs(targets - targets.max())
        penalty /= torch.sum(penalty)

        loss = preds * penalty
        loss = loss.view(batch_size, -1)

        return torch.sum(loss, dim=1) * self.scale


class MTCrossEntropyHeadDiversityLoss(MTCrossEntropyLoss):

    def __init__(self, class_weight=None, class_reduction='mean',
                 task_configs=[], manager_args={}, balancer_class=None, balancer_args={},
                 diversity_lambda=0.0):
        """
        Exposes attention probabilities to the loss function.

        Can be used to enforce orthogonality across attention heads.
        """
        super().__init__(class_weight=class_weight, class_reduction=class_reduction,
                         task_configs=task_configs, manager_args=manager_args,
                         balancer_class=balancer_class, balancer_args=balancer_args)

        assert 0 <= diversity_lambda <= 1, (
            'weight of diversity lambda must be in (0, 1)'
        )
        self.diversity_lambda = diversity_lambda

    def _compute_loss(self, inputs, targets):
        """
        """
        Y_hats = {}
        for task, _ in targets.items():
            if type(inputs[task]) == dict:
                Y_hats[task] = inputs[task]['out']
            else:
                Y_hats[task] = inputs[task]
        classification_losses = super()._compute_loss(Y_hats, targets)

        print('--')
        print('MTCrossEntropyHeadDiversityLoss')
        loss = {}
        for task, _ in targets.items():
            classification_loss = classification_losses[task]
            diversity_loss = self._compute_diversity_loss(inputs[task]['attn_scores'])
            loss[task] = classification_loss * (1 - self.diversity_lambda) + \
                         diversity_loss * self.diversity_lambda
            print(f'task: {task}')
            print(f'- class_loss:\t{classification_loss}')
            print(f'- diversity_loss:\t{diversity_loss}')
            print(f'- final loss:\t{loss[task]}')
        return loss


    def _compute_diversity_loss(self, attn_scores):
        batch_size, num_heads, L, H, W = attn_scores.shape

        A = attn_scores.view(batch_size, num_heads, -1)
        A_transpose = A.transpose(1, 2)
        I = torch.eye(num_heads).repeat(batch_size, 1, 1).to(A.device)

        attn_loss = torch.norm(torch.bmm(A, A_transpose) - I, p='fro')
        return attn_loss


class MTCrossEntropyRegionAwareLoss(MTCrossEntropyLoss):

    def __init__(self, class_weight=None, class_reduction='mean',
                 task_configs=[], manager_args={}, balancer_class=None, balancer_args={},
                 region_aware_only=False, region_lambda=0.0, region_configs=[]):
        """
        """
        super().__init__(class_weight=class_weight, class_reduction=class_reduction,
                         task_configs=task_configs, manager_args=manager_args,
                         balancer_class=balancer_class, balancer_args=balancer_args)

        self.region_aware_only = region_aware_only
        assert 0 <= region_lambda <= 1, (
            'region lambda must be between 0 and 1'
        )
        self.region_lambda = region_lambda
        self.task_to_region_config = {}
        for region_config in region_configs:
            task = region_config['task']
            self.task_to_region_config[task] = region_config

            region_loss_class = region_config.get('loss_class', 'KLDivLoss')
            self.task_to_region_config[task]['loss_class'] = region_loss_class
            region_loss_args = region_config.get('loss_args', {'reduction': 'batchmean'})
            self.task_to_region_config[task]['loss_args'] = region_loss_args
            self.task_to_region_config[task]['loss_fn'] = globals()[region_loss_class](**region_loss_args)

    def _load_region(self, shape, distribution='normal', **kwargs):
        """Generates a volumetric pdf for region-based loss penalties.

        Args:
            shape: (array-like)   a 3-vector with the shape of the current exam.
            distribution:   (string)    type of distribution

            kwargs: distribution-dependent arguments
                distribution == "normal"
                loc:   (np.array)  array positioning the mean of the target
                    distribution. Uses proportions in order to scale with
                    various sized exams, thus must be a vector in [0,1]^3.
                cov:    (np.array)  covariance. Can be scalar, array, or matrix.
                    most commonly an array indicating std in length, height, and
                    width, respectively. Note that these integer values are
                    relative to the total size of the body (i.e. [1,1,1] will
                    become a cov array of [26, 7, 7] if exam is [26, 7, 7].)
        """
        region = np.array(list(np.ndindex(shape))).reshape(shape + (-1,))
        if distribution == 'normal':
            mean = (np.array(shape) - 1) * kwargs['mean']
            cov = (np.array(shape) - 1) * kwargs['cov']
            region = multivariate_normal.pdf(region, mean=mean, cov=cov)
            region = region / np.sum(region)
        else:
            raise ValueError(f'distribution {distribution} has not yet been implemented.')

        return torch.DoubleTensor(region)


    def _compute_loss(self, inputs, targets):
        """
        """
        Y_hats = {}
        for task, _ in targets.items():
            if type(inputs[task]) == dict:
                Y_hats[task] = inputs[task]['out']
            else:
                Y_hats[task] = inputs[task]
        classification_losses = super()._compute_loss(Y_hats, targets)
        print('--')
        print('MTCrossEntropyRegionAwareLoss')
        loss = {}
        for task, _ in targets.items():
            classification_loss = classification_losses[task]
            classification_loss *= 0.0 if self.region_aware_only else 1.0

            if task in self.task_to_region_config:
                region_loss = self._compute_region_loss(task, inputs[task]['attn_scores'])
                loss[task] = classification_loss * (1 - self.region_lambda) + \
                             region_loss * self.region_lambda
            else:
                region_loss = 0.0
                # no need for region_lambda coefficient
                loss[task] = classification_loss

            print(f'task: {task}')
            print(f'- class_loss:\t{classification_loss}')
            print(f'- region_loss:\t{region_loss}')
            print(f'- final loss:\t{loss[task]}')

        return loss

    def _compute_region_loss(self, task, attn_scores):
        """
        """
        # one attention head
        if len(attn_scores.shape) == 4:
            attn_scores = attn_scores.unsqueeze(1)

        batch_size, num_heads, L, H, W = attn_scores.shape
        exam_dims = (L, H, W)
        region_config = self.task_to_region_config[task]
        region_loss_fn = region_config['loss_fn']

        region_target = self._load_region(exam_dims, **region_config['region'])
        region_target = region_target.unsqueeze(0).expand((batch_size,) + exam_dims).double()
        region_target = place_on_gpu(region_target, attn_scores.device)

        region_loss = 0.0
        for head in range(num_heads):
            head_scores = attn_scores[:,head,:,:]
            if region_config['loss_class'] == 'KLDivLoss':
                head_scores_flat = head_scores.view(batch_size, L * H * W)
                log_head_scores_flat = nn.functional.log_softmax(head_scores_flat, dim=-1)
                region_preds = log_head_scores_flat.view(batch_size, L, H, W)
            else:
                region_preds = head_scores
            region_loss += region_loss_fn(region_preds.double(),
                                          region_target)
        return place_on_gpu(region_loss.float(), attn_scores.device)


class WeightedCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self, params):
        """
        """
        self.__dict__.update(params)
        weights = torch.Tensor(self.weights)
        super().__init__(weight=torch.Tensor(self.weights), reduction=self.reduction)


class ReportLoss(nn.Module):

    def __init__(self, vocab, vocab_weights_path=None):
        """
        """
        super().__init__()
        self.vocab = vocab
        self.padding_idx=self.vocab['<pad>']

        self.is_weighting = vocab_weights_path is not None
        if self.is_weighting:
            with open(vocab_weights_path) as f:
                self.word2weight = json.load(f)

            self.weights = torch.ones(len(vocab))
            for word, weight in self.word2weight.items():
                self.weights[vocab[word]] = weight

    def forward(self, inputs, targets):
        """
        """
        probs = nn.functional.log_softmax(inputs, dim=-1)

        # zero out, probabilities for which we have nothing in the target text
        target_masks = (targets != self.padding_idx).float()

        # compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(probs,
                                                  index=targets[:, 1:].unsqueeze(-1),
                                                  dim=-1).squeeze(-1) * target_masks[:, 1:]
        if self.is_weighting:
            self.weights = self.weights.to(targets.device)
            target_weights = self.weights[targets[:, 1:]]
            loss = (target_weights * target_gold_words_log_prob).sum()
        else:
            loss = target_gold_words_log_prob.sum()

        return -loss
