"""
Defines the DatasetSplitter class.
"""
import logging
import os
import random
from collections import OrderedDict, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from pet_ct.util.util import Process


class BinarySplitter(Process):
    """

    """

    def __init__(self, dir, seed, split_to_frac, confirmation_configs=[]):
        """
        Loads the "exams.csv" file from the splits parent dataset directory. Sets the
        seed for reproducible splits. Converts labels to binary labels. Remove exams
        that are not confirmed by self.confirmation_fn.
        args:
            process_dir (str) the process directory, should be inside a DatasetBuilder
                              process dir.
        """
        super().__init__(dir)

        # set random seed for reproducible splits
        np.random.seed(seed)
        random.seed(seed)

        # load the exams csv
        exams_path = os.path.join(self.dir, "..", "exams.csv")
        if not os.path.exists(exams_path):
            raise(Exception("Parent directory of splitter directory must contain an \
                             an \"exams.csv\" file."))
        self.exams_df = pd.read_csv(exams_path, index_col=0)

        # convert {1,2,4,9} labels to {0, 1}
        self.exams_df['label'] = 1 * (self.exams_df['label'] > 1)

        # confirm exams
        for index, row in self.exams_df.iterrows():
            for confirmation_config in confirmation_configs:
                fn = confirmation_config['fn']
                args = confirmation_config['args']

                if not getattr(self, fn)(row, **args):
                    self.exams_df = self.exams_df.drop(index=index)
                    break

        # load split fractions
        if np.sum(list(split_to_frac.values())) != 1.0:
            raise ValueError(f"Split fractions do not sum to 1.0.")
        self.split_to_frac = split_to_frac

    def _run(self, overwrite=False):
        """
        Runs the splitter and writes the splits to separate CSV files.
        """
        self._split()
        self._write()

    def _split(self):
        """
        Builds  the splits via stratified random sampling. For each split, patients are
        repeatedly sampled without replacement. If the patient's exams would overflow
        the quotas for each label (this class is for binary labels so quotas are
        half of the total. )
        """
        logging.info("Grouping exams by patient...")
        patient_id_to_exam_ids = {patient_id: list(self.exams_df.loc[row_indices]['exam_id'])
                                  for patient_id, row_indices
                                  in tqdm(self.exams_df.groupby('patient_id').groups.items())}
        patient_ids = list(self.exams_df["patient_id"].unique())
        random.shuffle(patient_ids)


        logging.info("Building splits...")
        self.split_to_exam_ids = defaultdict(list)
        num_patients = len(patient_ids)
        running_frac = 0.0
        for split, frac in tqdm(self.split_to_frac.items()):
            for patient_id in patient_ids[int(running_frac * num_patients) :
                                           int((running_frac + frac) * num_patients)]:
                exam_ids = patient_id_to_exam_ids[patient_id]
                self.split_to_exam_ids[split].extend(exam_ids)
            running_frac += frac

        # ensure there is no overlap
        self._check_overlap()

    def _write(self):
        """
        Writes each split to a separate csv in the process directory.
        """
        logging.info("Writing splits to disk...")
        stats = {}
        for name, exams in self.split_to_exam_ids.items():
            path = os.path.join(self.dir, f"{name}_exams.csv")
            logging.info(f"Writing {len(exams)} to split {path}")
            split_df = self.exams_df.loc[exams]
            split_df.to_csv(path)

            # get label distribution in split
            stats[name] = split_df.groupby('label')['exam_id'].count()

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(self.dir, "stats.csv"))
        logging.info("Done.")

    def _check_overlap(self):
        """
        Ensures there is no exam overlap between splits.
        """
        logging.info("Ensuring no overlap between splits...")
        for exams_a, exams_b in combinations(self.split_to_exam_ids.values(), r=2):
            if set(exams_a).intersection(set(exams_b)):
                raise(Exception("Exams overlap between splits."))

    def _confirm_modality_lengths(self, exam_row, threshold=2):
        """
        Confirms that the number of pet images matches the number of CT images in an exam.
        The numbers need only match within threshold.
        return:
            confirmed   (bool) if the difference in the number of images differs by at
                        any more than the threshold, return False, else return True.
        """
        difference = (int(exam_row['PET_BODY_CTAC/dim_0']) -
                      int(exam_row['CT Images/dim_0']))
        return np.abs(difference) <= threshold

    def _confirm_exam_length(self, exam_row, threshold=100):
        """
        """
        is_sufficient = ((int(exam_row['PET_BODY_CTAC/dim_0']) > threshold) &
                         (int(exam_row['CT Images/dim_0']) > threshold))
        return is_sufficient


class MatchesSplitter(Process):
    """
    """

    def __init__(self, dir, split_configs=[]):
        """
        """
        super().__init__(dir)
        self.match_labels_df = pd.read_csv(os.path.join(dir, "..", "match_labels.csv"))
        self.split_configs = split_configs


    def _run(self, overwrite=False):
        """
        Splits the dataset and checks for data coverage and leakage.
        """
        logging.info("Loading exams...")

        exam_ids = list(self.match_labels_df["exam_id"].unique())
        random.shuffle(exam_ids)

        logging.info("Splitting exams...")
        num_exams = len(exam_ids)
        running_frac = 0.0
        split_to_exam_ids = defaultdict(set)
        for split_config in self.split_configs:
            frac = split_config["frac"]
            name = split_config["name"]
            for exam_id in exam_ids[int(running_frac * num_exams) :
                                    int((running_frac + frac) * num_exams)]:
                split_to_exam_ids[name].add(exam_id)

            running_frac += frac

        if running_frac != 1.0:
            raise ValueError(f"Split fractions do not sum to 1.0.")

        logging.info("Checking for data leakage...")
        for exam_set_a, exam_set_b in combinations(split_to_exam_ids.values(), 2):
            if exam_set_a & exam_set_b:
                raise ValueError("Data leakage detected!")

        for split, exam_ids in split_to_exam_ids.items():
            split_df = self.match_labels_df[self.match_labels_df["exam_id"].isin(exam_ids)]

            logging.info(f"{split} length: {len(split_df)}")
            split_df.to_csv(os.path.join(self.dir, f"{split}.csv"))


class StratifiedBinarySplitter(Process):
    """
    The StratifiedBinarySplitter class defines a process that divides a dataset into
    via stratified subsampling. The counts of all but one of the splits are specified
    in the params.json file. These splits will have an even class distribution if possible
    via stratified random sampling. The last split will take all leftover exams. No two
    exams from the same patient will appear in different splits. Each split will be
    written to its own csv file in the process directory.
    The process directory must be inside an existing DatasetBuilder process
    directory with a valid "exams.csv" file created by the DatasetBuilder process.
    The params file for the process should be formatted:
        {
            "process_class": "StratifiedBinarySplitter",

            "seed": 123,

            "confirmation_fn": "_confirm_exam_length",

            "split_totals":{
                "train": -1,
                "val": 200,
                "test": 200
            }
        }
    params:
            process_class (class) the process class "StratifiedBinarySplitter"
            seed (int)  Seed of the experiment for reproducible splits
            confirmation_fn (method) predicate class function used to confirm the
                                     inclusion  of each exam.
            split_totals (dict) dictionary from dict name to count. Exactly one split
                                should be -1. This will be filled with leftovers after
                                meeting quotas of first two splits.
    """

    def __init__(self, process_dir):
        """
        Loads the "exams.csv" file from the splits parent dataset directory. Sets the
        seed for reproducible splits. Converts labels to binary labels. Remove exams
        that are not confirmed by self.confirmation_fn.
        args:
            process_dir (str) the process directory, should be inside a DatasetBuilder
                              process dir.
        """
        super().__init__(process_dir)

        # set random seed for reproducible splits
        np.random.seed(self.seed)

        # load the exams csv
        self.exams_path = os.path.join(self.dir, "..", "exams.csv")
        if not os.path.exists(self.exams_path):
            raise(Exception("Parent directory of splitter directory must contain an \
                             an \"exams.csv\" file."))
        self.exams_df = pd.read_csv(self.exams_path, index_col=0)

        # convert {1,2,4,9} labels to {0, 1}
        self.exams_df['label'] = 1 * (self.exams_df['label'] > 1)

        for index, row in self.exams_df.iterrows():
            for confirmation_config in confirmation_configs:
                fn = confirmation_configs['fn']
                args = confirmation_configs['args']

                if not getattr(self, fn)(row, **args):
                    self.exams_df = self.exams_df.drop(index=index)

    def _run(self, overwrite=False):
        """
        Runs the splitter and writes the splits to separate CSV files.
        """
        self._split()
        self._write()

    def _split(self):
        """
        Builds  the splits via stratified random sampling. For each split, patients are
        repeatedly sampled without replacement. If the patient's exams would overflow
        the quotas for each label (this class is for binary labels so quotas are
        half of the total. )
        """
        logging.info("Grouping exams by patient...")
        patients = {patient_id: {"exam_ids": self.exams_df.loc[row_indices]['exam_id'],
                                 "labels": self.exams_df.loc[row_indices]['label']}
                    for patient_id, row_indices
                    in tqdm(self.exams_df.groupby('patient_id').groups.items())}
        remaining = list(patients.keys())

        logging.info("Building splits with stratified random sampling...")
        self.splits = {}
        for split, total in tqdm(self.split_totals.items()):
            # left
            if total == -1:
                leftover_split = split
                continue

            # randomly sample until label quotas are met
            exams = []
            tot_pos = tot_neg = total / 2
            while tot_pos != 0 or tot_neg != 0:
                index = np.random.randint(0, len(remaining))
                patient = patients[remaining[index]]

                # get number of positive and negative labels for patient
                n_pos = np.sum(patient['labels'])
                n_neg = len(patient['labels']) - n_pos

                # only add if it won't overflow label quotas
                if tot_pos - n_pos >= 0 and tot_neg - n_neg >= 0:
                    tot_pos -= n_pos
                    tot_neg -= n_neg
                    remaining.pop(index)
                    exams.extend(patient['exam_ids'])
            self.splits[split] = exams
        self.splits[leftover_split] = [exam_id for patient in remaining
                                       for exam_id in patients[patient]['exam_ids']]

        # ensure there is no overlap
        self._check_overlap()

    def _write(self):
        """
        Writes each split to a separate csv in the process directory.
        """
        logging.info("Writing splits to disk...")
        stats = {}
        for name, exams in self.splits.items():
            path = os.path.join(self.dir, f"{name}_exams.csv")
            logging.info(f"Writing {len(exams)} to split {path}")
            split_df = self.exams_df.loc[exams]
            split_df.to_csv(path)

            # get label distribution in split
            stats[name] = split_df.groupby('label')['exam_id'].count()

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(self.dir, "stats.csv"))
        logging.info("Done.")

    def _check_overlap(self):
        """
        Ensures there is no exam overlap between splits.
        """
        logging.info("Ensuring no overlap between splits...")
        for exams_a, exams_b in combinations(self.splits.values(), r=2):
            if set(exams_a).intersection(set(exams_b)):
                raise(Exception("Exams overlap between splits."))

    def _confirm_exam_length(self, exam_row, threshold=2):
        """
        Confirms that the number of pet images matches the number of CT images in an exam.
        The numbers need only match within threshold.
        return:
            confirmed   (bool) if the difference in the number of images differs by at
                        any more than the threshold, return False, else return True.
        """
        difference = (int(exam_row['PET_BODY_CTAC/dim_0']) -
                      int(exam_row['CT Images/dim_0']))
        return np.abs(difference) <= threshold