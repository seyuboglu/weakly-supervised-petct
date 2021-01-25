"""
"""
import os
import json
from time import time
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize

import torch
from torch.utils.data import Dataset, DataLoader

from pet_ct.data.h5_dataset import H5Dataset
import pet_ct.data.preprocess as preprocess
import pet_ct.data.image_transforms as image_transforms
import pet_ct.data.report_transforms as report_transforms
from pet_ct.data.vocab import WordPieceVocab
from pet_ct.data.term_graphs import TermGraph
from pet_ct.data import task_graphs


class BaseDataset(Dataset):
    """
    """
    def __init__(self, dataset_dir, split=None, data_dir=None,):
        """
        """
        self.dataset_dir = dataset_dir
        self.split = split

        if split is None or split == "all":
            exams_path = os.path.join(dataset_dir, 'exams.csv')
        else:
            exams_path = os.path.join(dataset_dir, 'split', f'{split}_exams.csv')
        self.exams_df = pd.read_csv(exams_path)

        self._get_data_dir(data_dir)

    def _get_data_dir(self, data_dir=None):
        """
        """
        # load dataset name and directory from dataset params file
        with open(os.path.join(self.dataset_dir, "params.json")) as f:
            params = json.load(f)
            self.dataset_name = params["dataset_name"]
            if data_dir is None:
                self.data_dir = params["data_dir"]
            else:
                self.data_dir = data_dir

    def __len__(self):
        """
        """
        return len(self.exams_df)

    def get_exam(self, exam_id):
        idx = self.exams_df.loc[self.exams_df['exam_id']==exam_id].index[0]
        return self[idx]

    def __getitem__(self, idx):
        """
        """
        pass

    def _apply_transforms(self, images):
        pass


class ReportDataset(BaseDataset):
    """
    """

    def __init__(self, dataset_dir, transform_configs=[],
                 split=None):
        """
        """
        BaseDataset.__init__(self, dataset_dir, split)
        self.exams_df = self.exams_df
        self.report_transform_configs = transform_configs

    def __len__(self):
        """
        """
        return len(self.exams_df.index)

    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.iloc[int(idx)]

        label = exam['label']
        exam_id = exam['exam_id']
        patient_id = exam['patient_id']

        report = self._get_report(exam, dataset)

        return label, report, {"exam_id": exam_id,
                               "patient_id": patient_id}

    def _get_report(self, exam, dataset):
        """
        """
        report = dataset.read_reports(exam["exam_id"])
        report = self._apply_report_transforms(report)
        return report

    def _apply_report_transforms(self, report):
        """
        """
        for transform_config in self.report_transform_configs:
            report = self._apply_report_transform(report, **transform_config)
        return report

    def _apply_report_transform(self, report, fn, args={}, splits=None):
        """
        Recursively applies transforms to reports or list of reports. Important for
        transformations that split reports into sections.
        """
        if splits is None or self.split in splits:
            if type(report) is str:
                report = getattr(report_transforms, fn)(report, **args)
            elif type(report) is list:
                report = [self._apply_report_transform(txt, fn, args, splits)
                          for txt in report]
        return report


class ImageDataset(BaseDataset):
    """
    Dataset for loading binary prediction task.
    """

    def __init__(self, dataset_dir, image_types, size, normalize=True,
                 transform_configs=[], sampling_window=None, sampling_rate=1,
                 split=None, data_dir=None, labels_dir=None, task=None):
        """
        """
        # ingest parameters
        BaseDataset.__init__(self, dataset_dir, split, data_dir)

        self.image_transform_configs = transform_configs
        self.normalize = normalize
        self.sampling_window = sampling_window
        self.sampling_rate = sampling_rate
        self.image_types = image_types
        self.size = size

    def get_num_slices(self):
        """
        """
        keys = [key for key in self.exams_df.keys() if key[-5:] == 'dim_0']
        res = list(self.exams_df[keys].min(axis=1))
        return res

    def get_targets(self):
        """
        Returns a tensor equal to the length of the split exam csv where
        labels[idx] gives label of idx.
        """
        return torch.tensor(self.exams_df['label'], dtype=torch.long)

    def __len__(self):
        """
        Gets the number of exams in the dataset.
        """
        return len(self.exams_df.index)

    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.iloc[int(idx)]

        label = exam['label']
        exam_id = exam['exam_id']
        patient_id = exam['patient_id']

        inputs = self._get_images(exam, dataset)
        targets = torch.tensor(label)
        del dataset
        return inputs, targets, {"exam_id": exam_id,
                                 "patient_id": patient_id}

    def _get_images(self, exam, dataset):
        """
        """
        exam_id = exam['exam_id']

        sample_start = None
        sample_end = None
        if self.sampling_window is not None:
            # randomize starting sample
            exam_length = exam[f'{self.image_types[0]}/dim_0']
            sample_start = int(np.ceil(exam_length * self.sample_window[0]))
            sample_end = int(np.ceil(exam_length * self.sample_window[1]))

        # open all images before applying transforms
        image_channels = []
        for image_type in self.image_types:
            images = dataset.read_images(exam_id, image_type,
                                         sample_start, sample_end, self.sampling_rate)
            image_channels.append(images)
        for i, images in enumerate(image_channels):
            images = preprocess.resize(images, tuple(self.size))
            if self.normalize:
                images = preprocess.normalize(images)
            image_channels[i] = images
        res = preprocess.join_channels(image_channels)  # (T, H, W, C) matrix
        res = self._apply_image_transforms(res)

        return torch.Tensor(res)

    def _apply_image_transforms(self, images):
        """
        Applies the transforms in the sequence specified in experiment params.

        Note that self.transform_fns will be a dictionary of dictionaries, where
        the dictionary will consist of the function name as the key and the
        function parameters the value, which itself is a dictionary pairing
        parameter names to values.

        args:
            exam (np.ndarray) (n_frames, height, width, channels)
        """
        if self.split == 'train':
            for transform_config in self.image_transform_configs:
                fn = transform_config["fn"]
                args = transform_config.get("args", {})
                images = getattr(image_transforms, fn)(images, **args)
        return images


class MTClassifierDataset(ImageDataset):
    """
    """
    def __init__(self, dataset_dir, targets_dir, image_types, size, normalize=True,
                 image_transform_configs=[], sampling_window=None, sampling_rate=1,
                 task_configs=[], split=None, data_dir=None):
        """
        """
        ImageDataset.__init__(self, dataset_dir, image_types=image_types, size=size,
                              normalize=normalize, sampling_window=sampling_window,
                              transform_configs=image_transform_configs,
                              sampling_rate=sampling_rate, split=split, data_dir=data_dir)
        self.tasks = [task_config["task"] for task_config in task_configs]
        self.targets_df = pd.read_csv(os.path.join(targets_dir, f'exam_labels.csv'),
                                      index_col=0, header=[0, 1], skipinitialspace=True)

    def get_num_slices(self):
        """
        """
        keys = [key for key in self.exams_df.keys() if key[-5:] == 'dim_0']
        res = list(self.exams_df[keys].min(axis=1))
        return res

    def get_targets(self, tasks=[], hard=False):
        """
        @param tasks (list) the tasks to get targets for
        @param hard (bool) if True, returns hard targets by argmaxing probability
        """
        for idx, exam in self.exams_df.iterrows():
            yield self._get_targets(exam, tasks=tasks, hard=hard)

    def _get_targets(self, exam, tasks=[], hard=False):
        """Gets targets for given exam.

        Uses the probabilistic labels for all but the `primary` labels.
        """
        targets = {}

        if not tasks:
            tasks = self.tasks # use all tasks if unspecified

        target_series = self.targets_df.loc[exam['exam_id']]
        for task in tasks:
            if task == 'primary' or task == 'mortality':
                continue

            soft_target = torch.tensor(target_series[task])
            soft_target[torch.isnan(soft_target)] = 0.5
            if hard:
                hard_target = torch.zeros_like(soft_target)
                hard_target[soft_target.argmax()] = 1.0
                targets[task] = hard_target
            else:
                targets[task] = soft_target

        if 'primary' in tasks:
            target = torch.zeros(2)
            target[exam['label']] = 1.0
            targets['primary'] = target

        return targets

    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.loc[int(idx)]

        label = exam['label']
        exam_id = exam['exam_id']
        patient_id = exam['patient_id']

        images = self._get_images(exam, dataset)
        targets = self._get_targets(exam)
        info = {"exam_id": exam_id,
                "patient_id": patient_id,
                "idx": idx}

        return images, targets, info


class MTReportDataset(ImageDataset, ReportDataset):
    """
    """
    def __init__(self, dataset_dir, image_types, size, normalize=True,
                 image_transform_configs=[], sampling_window=None, sampling_rate=1,
                 report_transform_configs=[],
                 tasks=["abnormality_detection", "report_generation"],
                 split=None):
        """
        """
        ImageDataset.__init__(self, dataset_dir, image_types=image_types, size=size,
                              normalize=normalize, sampling_window=sampling_window,
                              transform_configs=image_transform_configs,
                              sampling_rate=sampling_rate, split=split)
        ReportDataset.__init__(self, dataset_dir, split=split,
                               transform_configs=report_transform_configs)
        self.tasks = tasks

    def get_targets(self, tasks=["abnormality_detection",
                                 "report_generation"]):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        for idx, exam in self.exams_df.iterrows():
            yield self._get_targets(exam, dataset, tasks=tasks)

    def _get_targets(self, exam, dataset=None, tasks=["abnormality_detection",
                                                      "report_generation"]):
        """
        """
        targets = {}
        if "report_generation" in tasks:
            if dataset is None:
                raise ValueError("Need dataset to load reports.")
            report = self._get_report(exam, dataset)
            targets["report_generation"] = report

        if "abnormality_detection" in tasks:
            targets["abnormality_detection"] = torch.tensor(exam['label'])

        return targets


    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.iloc[int(idx)]

        label = exam['label']
        exam_id = exam['exam_id']
        patient_id = exam['patient_id']

        images = self._get_images(exam, dataset)
        targets = self._get_targets(exam, dataset, self.tasks)

        info = {"exam_id": exam_id,
                "patient_id": patient_id}

        return images, targets, info


class BertScanDataset(ImageDataset, ReportDataset):
    """
    """
    def __init__(self, dataset_dir, vocab_args, term_graph_args,
                 image_types=None, size=None, normalize=True,
                 image_transform_configs=[], sampling_window=None, sampling_rate=1,
                 report_transform_configs=[],
                 task_configs=[], max_len=None, skip_scans=False, split=None):
        """
        """
        self.skip_scans = skip_scans
        if not skip_scans:
            ImageDataset.__init__(self, dataset_dir, image_types=image_types, size=size,
                                  normalize=normalize, sampling_window=sampling_window,
                                  transform_configs=image_transform_configs,
                                  sampling_rate=sampling_rate, split=split)
        ReportDataset.__init__(self, dataset_dir, split=split,
                               transform_configs=report_transform_configs)

        self.task_configs = {task_config["fn"]: task_config
                             for task_config in task_configs}

        self.max_len = max_len
        self.vocab = WordPieceVocab(**vocab_args)
        self.term_graph = TermGraph(**term_graph_args)

    def __len__(self):
        """
        """
        return len(self.exams_df.index)

    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.iloc[int(idx)]
        label, exam_id, patient_id = exam['label'], exam['exam_id'], exam['patient_id']

        report = self._get_report(exam, dataset)
        images = self._get_images(exam, dataset) if not self.skip_scans else None
        targets = {}

        # must perform scan_match first
        if "scan_match" in self.task_configs:
            args = self.task_configs["scan_match"]["args"]
            report, labels = self.scan_match(exam_id, report, dataset, **args)
            targets["scan_match"] = labels

        # tokenize, trim if over max length, and wrap sentence
        report = self.vocab.tokenize(report)
        if self.max_len is not None and self.max_len < len(report):
            report = report[:self.max_len]
        report = self.vocab.wrap_sentence(report)

        if "scan_mlm" in self.task_configs:
            args = self.task_configs["scan_mlm"]["args"]
            report, labels = self.scan_mlm(exam_id, report, **args)
            targets["scan_mlm"] = labels

        info = {"exam_id": exam_id,
                "patient_id": patient_id}

        inputs = {"report": report,
                  "scan": images}

        return inputs, targets, info

    def scan_match(self, exam_id, report, dataset, pct_same):
        """
        """
        sample = torch.rand(1).item()
        label = torch.tensor(1)
        if sample >= pct_same:
            self.mismatched = True
            label = torch.tensor(0)
            random_idx = np.random.randint(len(self.exams_df))
            random_exam = self.exams_df.iloc[random_idx]
            report = self._get_report(random_exam, dataset)

        return report, label

    def scan_mlm(self, exam_id, report, rand_default_mask_prob=0.025,
                 term_default_mask_prob=0.85, term_to_mask_prob={}):
        """
        """
        # TODO:
        matches = self.term_graph.get_matches(report)
        #
        masked_report = report.copy()
        labels = -1 * torch.ones(len(report), dtype=torch.long)

        # mask tokens
        self.term_graph.bernoulli_sample(term_to_prob=term_to_mask_prob,
                                         default_prob=term_default_mask_prob,
                                         sample_name="mask_sample")

        for token_idx, token_matches in enumerate(matches):
            token = report[token_idx]

            masked = False
            if len(token_matches) == 0:
                masked = bool(torch.bernoulli(torch.tensor(rand_default_mask_prob)))
            else:
                token = report[token_idx]
                for match in token_matches:
                    if self.term_graph[match]["mask_sample"]:
                        masked = True
                        break

            if masked:
                masked_report[token_idx] = "[MASK]"
                labels[token_idx] = self.vocab.token_to_idx[token]
        return masked_report, labels


class BertPretrainingDataset(ReportDataset):
    """
    """

    def __init__(self, dataset_dir, vocab_args={}, transform_configs=[],
                 pct_tokens=0.15, pct_random=0.10, pct_same=0.10,
                 max_len=None, split=None):
        ReportDataset.__init__(self, dataset_dir, transform_configs, split)
        """
        WARNING: The vocab file used here MUST match the vocab file used in your model.
        """
        self.vocab = WordPieceVocab(**vocab_args)

        # parameters for mlm masking
        self.pct_tokens = pct_tokens
        self.pct_random = pct_random
        self.pct_same = pct_same

        self.max_len = max_len

    def get_targets(self, tasks=[]):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        for idx, exam in self.exams_df.iterrows():
            yield self._get_targets(exam, dataset, tasks=tasks)

    def _get_targets(self, exam, dataset=None, tasks=[]):
        """
        """
        targets = {}
        if "abnorm" in tasks:
            targets["abnorm"] = torch.tensor(exam['label'])

        return targets

    def __len__(self):
        """
        """
        return len(self.exams_df.index)

    def __getitem__(self, idx):
        """
        """
        dataset = H5Dataset(self.dataset_name, self.data_dir , mode="read")
        exam = self.exams_df.iloc[int(idx)]

        label = exam['label']
        exam_id = exam['exam_id']
        patient_id = exam['patient_id']

        report = self._get_report(exam, dataset)

        report = self.vocab.tokenize(report)

        if self.max_len is not None and self.max_len < len(report):
            report = report[:self.max_len]

        report = self.vocab.wrap_sentence(report)

        report, mask_labels = self._mask_inputs(report)

        inputs = {"report": report}
        targets = {"mlm": mask_labels, "abnorm": torch.tensor(label)}
        info = {"exam_id": exam_id,
                "patient_id": patient_id}

        return inputs, targets, info

    def _mask_inputs(self, report):
        """
        """
        # don't sample [CLS] or [SEP] tokens TODO: make this work for middle of sequence [SEP]
        n_tokens = max(0, np.ceil((len(report) - 2) * self.pct_tokens).astype(int))
        token_idxs = np.random.choice(np.arange(1, len(report) - 1),
                                      size=n_tokens, replace=False)

        labels = -1 * torch.ones(len(report), dtype=torch.long)
        masked_report = report.copy()

        for token_idx in token_idxs:
            sample = np.random.rand()
            if sample < self.pct_random:
                # TODO: check if this is correct, do we also change the input word
                labels[token_idx] = int(np.random.choice(list(self.vocab.idx_to_token.keys())))

            elif sample < self.pct_random + self.pct_same:
                labels[token_idx] = self.vocab.token_to_idx[report[token_idx]]

            else:
                masked_report[token_idx] = "[MASK]"
                labels[token_idx] = self.vocab.token_to_idx[report[token_idx]]

        return masked_report, labels


class MatchDataset:

    def __init__(self, dataset_dir, split=None, task_configs={}, vocab_args={}, max_len=200, cls_label=True):
        """
        """
        self.dataset_dir = dataset_dir
        self.split = split

        if split is None or split == "all":
            matches_path = os.path.join(dataset_dir, 'match_labels.csv')
        else:
            matches_path = os.path.join(dataset_dir, 'split', f'{split}.csv')
        self.matches_df = pd.read_csv(matches_path)
        self.matches_df = self.matches_df[self.matches_df["not_applicable"] == False]

        self.vocab = WordPieceVocab(**vocab_args)

        self.max_len = max_len
        self.cls_label = cls_label


    def __len__(self):
        """
        """
        return len(self.matches_df)

    def __getitem__(self, idx):
        """
        """
        match = self.matches_df.iloc[idx]

        text = match["text"]
        tokens = self.vocab.tokenize(text)

        if self.max_len is not None and self.max_len < len(tokens):
            tokens = tokens[:self.max_len]

        tokens = self.vocab.wrap_sentence(tokens)

        # get numeric label
        if match["fdg_abnormality_label"] == "abnormal":
            label = 1
        elif match["fdg_abnormality_label"] == "normal":
            label = 0
        else:
            label = 0

        labels = -1 * torch.ones(len(tokens), dtype=torch.long)
        if self.cls_label:
            labels[0] = label

        # label tokens
        match_start, match_end = match["start"], match["end"]
        find_start = 0
        for idx, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if token.startswith("##"):
                # remove pounds
                token = token[2:]

            token_start = text.find(token, find_start)
            token_end = token_start + len(token)
            find_start = token_end

            if ((token_start >= match_start and token_start < match_end) or
                (token_end >= match_start and token_end < match_end)):
                labels[idx] = label
        inputs = {"report": tokens}
        targets = {"fdg_abnorm": labels}
        info = {"exam_id": match["exam_id"],
                "term_name": match["term_name"]}

        return inputs, targets, info


class MTMortalityDataset(MTClassifierDataset):
    """
    e.g. args
    {
        "dataset_dir": "data/mortality",
        "targets_dir": "data/regional_dataset_data/labels/full_labels_6_altered_valid_test",
        "mortality_targets_dir": "data/mortality/mortality.csv",
        "image_types": [
            "CT Images",
            "PET_BODY_CTAC"
        ],
        "normalize": True,
        "image_transform_configs": [],
        "size": [
            224,
            224
        ],
        "class_boundaries": [180, 2000, 3000]
    }
    """
    def __init__(self, dataset_dir, targets_dir, mortality_targets_dir, image_types, size, normalize=True,
                 image_transform_configs=[], sampling_window=None, sampling_rate=1,
                 task_configs=[], split=None, data_dir=None, class_boundaries=[180]):

        super().__init__(dataset_dir=dataset_dir, targets_dir=targets_dir, image_types=image_types,
                         size=size, normalize=normalize, image_transform_configs=image_transform_configs,
                         sampling_window=sampling_window, sampling_rate=sampling_rate,
                         task_configs=task_configs, split=split, data_dir=data_dir)

        self.mortality_targets_df = pd.read_csv(mortality_targets_dir, index_col=0)
        self.class_boundaries = class_boundaries

    def _get_targets(self, exam, tasks=[], hard=False):
        """
        """
        targets = super()._get_targets(exam, tasks=tasks, hard=hard)

        if "mortality" in self.tasks:
            mortality_val = self.mortality_targets_df.loc[exam['exam_id']]["days_from_death"]
            if math.isnan(mortality_val):
                mortality_val = self.mortality_targets_df.loc[exam['exam_id']]["days_from_followup"]

            mortality_target = torch.zeros(len(self.class_boundaries) + 1)
            for idx, boundary in enumerate(self.class_boundaries):
                if mortality_val < boundary:
                    mortality_target[idx] = 1
                    break
            if mortality_target.sum() == 0:
                mortality_target[-1] = 1

            targets["mortality"] = mortality_target

        return targets

