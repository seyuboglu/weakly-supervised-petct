"""
Defines DataLoader classes to use.
"""
import logging
from collections import defaultdict
from itertools import groupby

import torch
import torch.nn as nn
import numpy as np
from torch._six import int_classes as _int_classes
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, Sampler
import torch.nn.functional as F

from pet_ct.util.util import flex_stack, get_batch_size, soft_to_hard

def pad_inputs(inputs, max_length):
    """
    """
    inputs_length = inputs.size()[0]
    pad_length = max_length - inputs_length
    pad_dims = (0, 0, 0, 0, 0, 0, 0, pad_length) # pads first dim evenly
    padded_inputs = F.pad(inputs, pad=pad_dims)
    return padded_inputs

def pad_scans(scan_inputs, value=0):
    """
    """
    max_length = max(scan_inputs, key=lambda x: x.shape[0]).shape[0]
    padded_inputs = []
    for inputs in scan_inputs:
        inputs_length = inputs.size()[0]
        pad_length = max_length - inputs_length
        pad_dims = (0, 0, 0, 0, 0, 0, 0, pad_length) # pads first dim evenly
        padded = F.pad(inputs, pad=pad_dims, value=value)
        padded_inputs.append(padded)

    return padded_inputs


def pad_targets(targets, value=-1):
    max_length = max(targets, key=lambda x: x.shape[-1]).shape[-1]
    padded_targets = []
    for tgt in targets:
        tgt_length = tgt.shape[-1]
        pad_length = max_length - tgt_length
        pad_dims = (0, pad_length)
        padded_targets.append(F.pad(tgt, pad=pad_dims, value=value))
    return padded_targets


def exam_collate(batch_list):
    """ Collate function fro a multi-task dataset.
    args:
        exam_list (list) list of exams
    """
    max_length = max(batch_list, key=lambda x: x[0].size())[0].size()[0]
    all_inputs = []
    all_targets = []
    all_info = []

    for inputs, targets, info in batch_list:
        all_inputs.append(pad_inputs(inputs, max_length))
        all_info.append(info)
        all_targets.append(targets)

    # stack targets and inputs
    all_targets = flex_stack(all_targets, dim=0)
    all_inputs = flex_stack(all_inputs, dim=0)
    return all_inputs, all_targets, all_info


def mt_exam_collate(batch_list):
    """ Collate function for a multi-task dataset.
    args:
        exam_list (list) list of exams
    """
    max_exam = max(batch_list, key=lambda x: x[0].size()[0])
    max_length = max_exam[0].size()[0]

    all_inputs = []
    all_targets = defaultdict(list)
    all_info = []

    for inputs, targets, info in batch_list:
        all_inputs.append(pad_inputs(inputs, max_length))
        all_info.append(info)
        for task, target in targets.items():
            all_targets[task].append(target)

    # stack targets and inputs
    all_targets = {task: flex_stack(targets, dim=0) for task, targets in all_targets.items()}
    all_inputs = flex_stack(all_inputs, dim=0)
    return all_inputs, all_targets, all_info


def mt_mi_exam_collate(batch_list):
    """ Collate function for a multi-task multi-input dataset
    """
    all_inputs = defaultdict(list)
    all_targets = defaultdict(list)
    all_info = []

    for inputs, targets, info in batch_list:
        all_info.append(info)

        for name, inpt in inputs.items():
            all_inputs[name].append(inpt)

        for task, target in targets.items():
            all_targets[task].append(target)

    if "scan" in all_inputs:
        all_inputs["scan"] = pad_scans(all_inputs["scan"], value=-1)

    if "mlm" in all_targets:
        all_targets["mlm"] = pad_targets(all_targets["mlm"], value=-1)
    elif "fdg_abnorm" in all_targets:
        all_targets["fdg_abnorm"] = pad_targets(all_targets["fdg_abnorm"], value=-1)

    # stack targets and inputs
    all_targets = {task: flex_stack(targets, dim=0) for task, targets in all_targets.items()}
    all_inputs = {name: flex_stack(inputs, dim=0) for name, inputs in all_inputs.items()}
    return all_inputs, all_targets, all_info


class ExamBatchSampler(Sampler):

    def __init__(self, batch_size, num_slices, sampler=None,
                 weights=None, num_samples=None, replacement=None, shuffle=None,
                 drop_last=False):
        """
        Creates batches of exams with same number of slices.

        TODO: Implement `drop_last`.
        """
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_slices = num_slices

        self.sampler = sampler
        if self.sampler is not None:
            self.weights = weights if weights is not None else torch.ones(len(num_slices))
        else:
            self.weights = torch.ones(len(num_slices))

        self.replacement = replacement
        self.shuffle = shuffle

        self.drop_last = drop_last

    def __iter__(self):
        """
        """
        if self.sampler is not None:
            samples = torch.multinomial(self.weights, self.num_samples,
                                        replacement=self.replacement)
        else:
            if self.shuffle:
                samples = torch.multinomial(self.weights, self.num_samples,
                                            replacement=False)
            else:
                samples = torch.tensor(range(self.num_samples))
        samples = sorted(samples, key=lambda idx: self.num_slices[idx])

        curr_iter = 0
        batches = []
        while curr_iter < self.num_samples:
            batch = [samples[curr_iter]]
            batch_slices = self.num_slices[samples[curr_iter]]

            offset = self.batch_size
            for i in range(1, self.batch_size):
                if curr_iter + i < self.num_samples and \
                    batch_slices == self.num_slices[samples[curr_iter + i]]:
                    batch.append(samples[curr_iter + i])
                else:
                    offset = i
                    break
            batches.append(batch)
            curr_iter = curr_iter + offset

        batch_idxs = torch.randperm(len(batches)).tolist()
        for batch_idx in batch_idxs:
            yield batches[batch_idx]

    def __len__(self):
        """
        This is approximate because we cannot know number of batches a priori.
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size



class ExamDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 sampler=None,
                 num_samples=1000,
                 replacement=False,
                 class_probs=None,
                 pin_memory=False):
        """
        """
        # get example weights so examples are sampled according to class_probs
        if sampler in {"WeightedRandomSampler", "RandomSampler"}:
            if sampler == "WeightedRandomSampler":
                classes = dataset.get_targets()
                counts = torch.bincount(classes)
                weights = torch.zeros_like(classes, dtype=torch.float)
                for example_idx, class_idx in enumerate(classes):
                    class_prob = class_probs[class_idx] / float(counts[class_idx])
                    weights[example_idx] = class_prob
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                replacement=replacement)
            elif sampler == "RandomSampler":
                weights = None
                sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                        replacement=True)
        elif sampler is not None:
            raise ValueError(f"Sampler {sampler} not supported.")
        else:
            num_samples = len(dataset)
            weights = None

        if batch_size > 1:
            num_slices = dataset.get_num_slices()
            batch_sampler = ExamBatchSampler(batch_size, num_slices,
                                             sampler=sampler,
                                             weights=weights,
                                             num_samples=num_samples,
                                             replacement=replacement,
                                             shuffle=shuffle,
                                             drop_last=False)

            super().__init__(dataset=dataset, num_workers=num_workers,
                             batch_sampler=batch_sampler, pin_memory=pin_memory,
                             collate_fn=mt_exam_collate)
        else:
            super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                             collate_fn=mt_exam_collate)


class MTExamDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 sampler=None,
                 num_samples=1000,
                 replacement=False,
                 weight_task=None,
                 class_probs=None,
                 pin_memory=False):
        """
        """
        if sampler in {"WeightedRandomSampler", "RandomSampler"}:
            # get example weights so examples are sampled according to class_probs
            if sampler == "WeightedRandomSampler":
                classes = []
                for target in dataset.get_targets(tasks=[weight_task], hard=True):
                    target_class = target[weight_task]
                    classes.append(target_class)
                classes = torch.stack(classes)
                if classes.shape[-1] > 1:
                    classes = soft_to_hard(classes, break_ties="random").long()
                classes = torch.LongTensor(classes)
                counts = torch.bincount(classes)
                weights = torch.zeros_like(classes, dtype=torch.float)
                for example_idx, class_idx in enumerate(classes):
                    class_prob = class_probs[class_idx] / float(counts[class_idx])
                    weights[example_idx] = class_prob
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                replacement=replacement)
            elif sampler == "RandomSampler":
                weights = None
                sampler = RandomSampler(data_source=dataset, num_samples=num_samples, 
                                        replacement=True)
        elif sampler is not None:
            raise ValueError(f"Sampler {sampler} not supported.")
        else:
            num_samples = len(dataset)
            weights = None

        if batch_size > 1:
            num_slices = dataset.get_num_slices()
            batch_sampler = ExamBatchSampler(batch_size, num_slices,
                                             sampler=sampler,
                                             weights=weights,
                                             num_samples=num_samples,
                                             replacement=True,
                                             shuffle=shuffle,
                                             drop_last=False)

            super().__init__(dataset=dataset, num_workers=num_workers,
                             batch_sampler=batch_sampler, pin_memory=pin_memory,
                             collate_fn=mt_exam_collate)
        else:
            super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                             collate_fn=mt_exam_collate)

class MTAdaptiveDataloader(DataLoader):

    def __init__(self, dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 num_samples=1000,
                 weight_task=None,
                 pin_memory=False):
        """
        """
        self.num_samples = num_samples
        sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                replacement=True)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, sampler=sampler)

    def update(self, metrics, metric="roc_auc"):
        """
        """
        for task in self.dataset.tasks:
            value = metrics.get_metric(metric=metric, task=task)


        weights = torch.zeros(len(dataset))
        for example_idx, targets in enumerate(self.dataset.get_targets(hard=True)):
            weights[example_idx] = 0

        sampler = WeightedRandomSampler(weights=weights, num_samples=self.num_samples, replacement=True)
        self.batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)


class MTMIExamDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 sampler=None,
                 num_samples=1000,
                 replacement=False,
                 weight_task=None,
                 class_probs=None,
                 pin_memory=False):
        """
        """
        # get example weights so examples are sampled according to class_probs
        if sampler == "WeightedRandomSampler":
            self.num_samples = int(round(num_samples / batch_size))
            classes = torch.LongTensor([target[weight_task]
                                        for target in dataset.get_targets([weight_task])])

            counts = torch.bincount(classes)
            weights = torch.zeros_like(classes, dtype=torch.float)
            for example_idx, class_idx in enumerate(classes):
                class_prob = class_probs[class_idx] / float(counts[class_idx])
                weights[example_idx] = class_prob
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                            replacement=replacement)
        elif sampler == "RandomSampler":
            self.num_samples = int(round(num_samples / batch_size))
            sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                    replacement=True)
        elif sampler is not None:
            raise ValueError(f"Sampler {sampler} not supported.")

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                         collate_fn=mt_mi_exam_collate)

    def __len__(self):
        """
        """
        if hasattr(self, "num_samples"):
            return self.num_samples
        else:
            return super().__len__()


class DynamicRandomSampler(Sampler):

    def __init__(self, weights, num_samples, replacement=False):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        if not replacement:
            self.mask = torch.ones_like(self.weights, dtype=torch.double)

    def __iter__(self):
        for _ in range(self.num_samples):
            weights = self.weights if self.replacement else self.weights * self.mask
            idx = int(torch.multinomial(weights, 1, self.replacement))
            if self.replacement:
                self.mask[idx] = 0.0
            yield idx

    def __len__(self):
        return self.num_samples


class MTDynamicDataLoader(DataLoader):

    def __init__(self, dataset,
                 batch_size=1,
                 num_workers=6,
                 num_samples=1000,
                 update_period=10,
                 replacement=False,
                 priority_metric="roc_auc",
                 priority_scale=100,
                 pin_memory=False):
        # initialize uniform weights
        self.exams = list(dataset.get_targets(hard=True))

        weights = torch.ones(len(self.exams))
        self.sampler = DynamicRandomSampler(weights, num_samples, replacement=replacement)

        self.priority_metric = priority_metric
        self.priority_scale = priority_scale
        self.task_to_priority = defaultdict(lambda: 1)
        self.task_to_samples = defaultdict(list)
        self.sample_idxs = []
        self.tasks = [{"name": task,
                       "cardinality": len(self.exams[0][task])}
                      for task in dataset.tasks]
        self._init_demands()

        # support for periodic updates
        self.update_counter = 0
        self.update_period = update_period
        self.queued_idxs = []

        super().__init__(dataset=dataset, batch_size=batch_size,
                         num_workers=num_workers, sampler=self.sampler, pin_memory=pin_memory,
                         collate_fn=mt_exam_collate)

    def update_epoch(self, metrics):

        # reset
        self.task_to_samples = defaultdict(list)
        self.sample_idxs = []
        self._init_demands()
        self.update_counter = 0
        self.queued_idxs = []

        # update priorities
        task_to_metric = {}
        total = 0
        for task in self.tasks:
            metric = metrics.get_metric(self.priority_metric, task["name"])
            task_to_metric[task["name"]] = metric
            total += metric

        self.task_to_priority = {
            task: self.priority_scale * (1 - metric/total) for task, metric in task_to_metric.items()
        }
        logging.info(task_to_metric)
        logging.info(self.task_to_priority)

    def update_batch(self, idxs):
        """
        """
        self.queued_idxs.extend(idxs)
        self.update_counter += len(idxs)

        if self.update_counter % self.update_period == 0:
            self.update_demands()
            self.update_weights()


    def _init_demands(self):
        """
        """
        self.task_to_demands = {}
        for task in self.tasks:
            self.task_to_demands[task["name"]] = np.full((1, task["cardinality"]), fill_value=1/task["cardinality"])
            self.task_to_samples[task["name"]] = []

    def get_loss_weights(self, targets):
        """
        """
        task_to_weight = {}
        for task, tgt in targets.items():
            tgt = int(torch.argmax(tgt).cpu().numpy())

            task_to_weight[task] = self.task_to_demands[task][-1, tgt]
        return task_to_weight

    def update_demands(self):
        """
        """
        for idx in self.queued_idxs:
            self.sample_idxs.append(idx)
            exam = self.exams[idx]
            for task, tgt in exam.items():
                self.task_to_samples[task].append(int(torch.argmax(tgt)))

            for task in self.tasks:
                samples = np.array(self.task_to_samples[task["name"]])
                total = np.sum(self.task_to_demands[task["name"]][np.arange(samples.size), samples])
                new_demands = []
                for tgt in range(task["cardinality"]):
                    tgt_total = np.sum(self.task_to_demands[task["name"]]
                                       [np.where(samples == tgt), tgt])
                    new_demand = (1 - (tgt_total + 1) / (total + task["cardinality"]))
                    new_demand *= self.task_to_priority[task["name"]]
                    new_demands.append(new_demand)
                self.task_to_demands[task["name"]] = np.append(self.task_to_demands[task["name"]],
                                                               np.array(new_demands)[None, :], axis=0)
        self.queued_idxs = []


    def update_weights(self):
        weights = []
        for exam in self.exams:
            weight = self._get_weight(exam)
            weights.append(weight)
        self.sampler.weights = torch.nn.functional.softmax(torch.tensor(weights), dim=0).double()

    def _get_weight(self, exam):
        """
        """
        weight = 0
        for task in self.tasks:
            tgt = torch.argmax(exam[task["name"]])
            weight += self.task_to_demands[task["name"]][-1, tgt]
        return weight
