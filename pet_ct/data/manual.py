"""
Process subclass that reads reports and outputs a labels csv.
"""

import os
import copy
from collections import defaultdict, OrderedDict

import pandas as pd
from tqdm import tqdm, tqdm_notebook
import json
import numpy as np
import networkx as nx
import torch
import logging
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, Markdown

from pet_ct.util.util import Process
from pet_ct.util.graphs import TriangleGraph
import pet_ct.learn.dataloaders as dataloaders
import pet_ct.learn.datasets as datasets
from pet_ct.data.report_transforms import extract_impression, split_impression_sections, word_tokenize, sent_tokenize
from metal.multitask.mt_label_model import MTLabelModel
from metal.analysis import lf_summary
import pet_ct.model.models as models
import pet_ct.data.labeler as labeler
import pet_ct.data.task_graphs as task_graphs
from pet_ct.data.term_graphs import TermGraph
from pet_ct.data.vocab import WordPieceVocab


class ExamLabelsPredictor(Process):
    """
    """
    def __init__(self, dir,
                 model_dir,
                 dataset_class="ReportDataset",
                 dataset_args={},
                 term_graph_dir="data/pet_ct_terms/terms.json",
                 terms="all",
                 match_task="fdg_abnorm",
                 split_fn="split_impression_sections",
                 max_len=200,
                 vocab_args={},
                 seed=123,
                 cuda=True,
                 devices=[0]):
        """
        """
        super().__init__(dir)
        self.cuda = cuda
        self.devices = devices
        self.device = devices[0]

        self.split_fn = globals()[split_fn]

        dataset = getattr(datasets, dataset_class)(**dataset_args)
        self.dataloader = DataLoader(dataset, batch_size=1)

        logging.info("Loading TermGraph and Vocab...")
        self.match_task = match_task
        self.terms = terms
        self.term_graph = TermGraph(term_graph_dir)
        if terms == "all":
            self.terms = self.term_graph.term_names
        else:
            self.terms = terms

        self.vocab = WordPieceVocab(**vocab_args)
        self.max_len = max_len

        logging.info("Loading Model...")
        self._load_model(model_dir)


    def _load_model(self, model_dir):
        """
        """
        with open(os.path.join(model_dir, "params.json")) as f:
            args = json.load(f)["process_args"]
            model_class = args["model_class"]
            model_args = args["model_args"]
            if "task_configs" in args:
                new_task_configs = []
                for task_config in args["task_configs"]:
                    new_task_config = args["default_task_config"].copy()
                    new_task_config.update(task_config)
                    new_task_configs.append(new_task_config)
            task_configs = new_task_configs

            model_args["task_configs"] = task_configs

        model_class = getattr(models, model_class)
        self.model = model_class(cuda=self.cuda, devices=self.devices, **model_args)

        model_dir = os.path.join(model_dir, "best")
        model_path = os.path.join(model_dir, "weights.pth.tar")
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, "weights.link")

        self.model.load_weights(model_path, device=self.device)

    def label_exam(self, label, report, info):
        """
        """
        report_sections = self.split_fn(report[0].lower())
        term_to_outputs = defaultdict(list)

        #logging.info(f"exam_id: {info['exam_id']}")
        for report_section in report_sections:
            curr_matches = self.term_graph.match_string(report_section)
            if not curr_matches:
                # skip report sections without matches
                continue

            tokens = self.vocab.tokenize(report_section)

            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]

            tokens = self.vocab.wrap_sentence(tokens)
            inputs = {"report": [tokens]}
            output = self.model.predict(inputs)[self.match_task]
            output = output.cpu().detach().numpy().squeeze()

            #logging.info(f"section:{report_section}")
            for match in curr_matches:
                match_idxs = self.vocab.get_tokens_in_range(tokens,
                                                            report_section,
                                                            match["start"],
                                                            match["end"])

                match["output"] = output[match_idxs, 1]
                term = match["term_name"]
                term_to_outputs[term].append(np.mean(match["output"]))
                #logging.info(f"term: {match['term_name']} - {match['output']}")
            #logging.info("-"*5)

        labels = {}
        for term in self.terms:
            all_outputs = term_to_outputs[term][:]
            for descendant in self.term_graph.get_descendants(term):
                all_outputs.extend(term_to_outputs[descendant])
            all_outputs = np.array(all_outputs)
            prob = 1 - np.prod(1 - all_outputs)

            labels[(term, 0)] = 1 - prob
            labels[(term, 1)] = prob

            #logging.info(f"term: {term}")
            #logging.info(f"all_outputs: {all_outputs}")
            #logging.info(f"prob: {prob}")

        #logging.info("="*30 + "\n")
        return labels


    def _run(self, overwrite=False):
        """
        """
        exam_id_to_labels = {}
        for idx, (label, report, info) in enumerate(tqdm(self.dataloader)):
            labels = self.label_exam(label, report, info)
            exam_id_to_labels[info["exam_id"][0]] = labels

        labels_df = pd.DataFrame.from_dict(exam_id_to_labels, orient="index")
        labels_df.to_csv(os.path.join(self.dir, "exam_labels.csv"))


class ExamLabelsBuilder(Process):
    """
    """

    def __init__(self, dir, exams_path, match_labeler_dir, term_graph_dir, terms, task,
                 propagate_labels=True):
        """
        """
        super().__init__(dir)
        self.task = task
        self.propagate_labels = propagate_labels
        self.terms = terms
        self.exams = pd.read_csv(exams_path, index_col=0).index
        self.match_labels_df = pd.read_csv(os.path.join(match_labeler_dir, "match_labels.csv"), engine='python')
        self.term_graph = TermGraph(term_graph_dir)

    def _run(self, overwrite=False):
        """
        """
        exam_id_to_labels = {}
        for exam_id in self.exams:
            exam_id_to_labels[exam_id] = self.label_exam(exam_id)

        labels_df = pd.DataFrame.from_dict(exam_id_to_labels, orient="index")
        labels_df.to_csv(os.path.join(self.dir, "exam_labels.csv"))

    def label_exam(self, exam_id):
        """
        """
        term_to_label = {term: False for term in self.terms}
        exam_matches = self.match_labels_df[self.match_labels_df["exam_id"] == exam_id]
        for idx, exam_match in exam_matches.iterrows():
            term = exam_match["term_name"]
            label = exam_match[f"{self.task}_label"] == "abnormal"
            if term in term_to_label:
                term_to_label[term] |= label

            if self.propagate_labels:
                for ancestor_term in self.term_graph.get_ancestors(term):
                    if ancestor_term in term_to_label:
                        term_to_label[ancestor_term] |= label
        labels = {}
        for term, abnorm in term_to_label.items():
            labels[(term, 0)] = float(not abnorm)
            labels[(term, 1)] = float(abnorm)

        return labels

class LabelerDatasetBuilder(Process):
    """
    """
    def __init__(self, dir, manual_dirs=[]):
        """
        """
        super().__init__(dir)
        self.manual_dirs = manual_dirs

    def _run(self, overwrite=False):
        """
        """
        dfs = []
        for manual_dir in self.manual_dirs:
            curr_df = pd.read_csv(os.path.join(manual_dir,
                                               "match_labels.csv"))
            dfs.append(curr_df)
        match_labels_df = pd.concat(dfs)
        match_labels_df.to_csv(os.path.join(self.dir, "match_labels.csv"))


class MatchLabeler(Process):
    """
    A manual labeler for labeling term matches in PET/CT impressions./
    """
    def __init__(self, dir,
                 dataset_class="ReportDataset",
                 dataset_args={},
                 term_graph_dir="data/pet_ct_terms/terms.json",
                 task_configs=[],
                 split_fn="split_impression_sections",
                 num_exams=None):
        """
        """
        super().__init__(dir)
        self.split_fn = globals()[split_fn]
        dataset = getattr(datasets, dataset_class)(**dataset_args)
        dataloader = DataLoader(dataset, batch_size=1)

        logging.info("Loading tasks...")
        self.task_configs = task_configs
        self.term_graph = TermGraph(term_graph_dir)

        labels_path = os.path.join(self.dir, "match_labels.csv")
        if os.path.exists(labels_path):
            logging.info("Loading labels...")
            self.labels_df = pd.read_csv(labels_path)
            self.labeled_exams = set(self.labels_df["exam_id"].unique())
        else:
            self.labels_df = None
            self.labeled_exams = set()
        self.skipped_exams = set()

        logging.info("Loading reports...")
        self.exams = {}
        for idx, (label, report, info) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if num_exams is not None and idx == num_exams:
                break

            self.exams[info["exam_id"][0]] = {
                "patient_id": info["patient_id"][0],
                "report": report[0].lower(),
                "curr_label": label[0]
            }

    def build_match_labeler(self, match, text):
        """
        Builds the match labeler GUI. This GUI includes:
        1) The matched term (e.g. lymph_node)
        2) The report text with the matched words in bold
        3) A ToggleButtons interface for each task. (task_buttons)
        4) A not applicable ToggleButton, to be selected when the matched words
           don't actually correspond with the matched term. (na_button)
        """
        match_labeler = {}
        match_labeler["match"] = match
        out = widgets.Output()
        term_name = match['term_name']

        bold_text = text[:]
        bold_text = (bold_text[:match["start"]] + "** `" +
                     text[match["start"] : match["end"]] + "` **" +
                     bold_text[match["end"]:])
        display(Markdown(f"## {term_name}"))
        display(Markdown(f"> {bold_text}"))

        # task buttons
        task_buttons = {}
        for task_config in self.task_configs:
            # abnormality buttons
            task_button = widgets.ToggleButtons(
                value=task_config["default"],
                options=task_config["options"],
                description=task_config["description"])

            def task_change(change):
                """
                Callback function triggered by the task buttons toggles.
                """
                with out:
                    pass
            task_button.observe(task_change)
            task_buttons[task_config["task"]] = task_button

        match_labeler['task_buttons'] = task_buttons

        # not applicable
        na_button = widgets.ToggleButton(value=False, description='Not applicable', icon='')

        def na_change(change):
            """
            Function triggered by not applicable button toggles.
            """
            with out:
                for task_button in task_buttons.values():
                    task_button.value = None
                na_button.icon = 'check' if na_button.icon else ''
        na_button.observe(na_change)
        match_labeler["na_button"] = na_button

        display(widgets.VBox(list(task_buttons.values())))
        display(na_button)
        return match_labeler

    def label_next(self):
        """
        Label next example
        """
        # get next unlabeled exam
        for exam_id in self.exams:
            if (exam_id not in self.labeled_exams and
                exam_id not in self.skipped_exams):
                break
        exam = self.exams[exam_id]
        report = exam['report']

        report_sections = self.split_fn(exam["report"])

        # display report and exam
        display(Markdown(f"### Progress"))
        display(Markdown(f"Exams labeled: {len(self.labeled_exams)}"))
        display(Markdown(f"Exams skipped: {len(self.skipped_exams)}"))
        display(Markdown(f"Exams remaining: {len(self.exams) - len(self.skipped_exams) - len(self.labeled_exams)}"))
        display(Markdown(f"# Exam: {exam_id}"))
        display(Markdown(f"## Full Impression"))
        display(Markdown(f"> {report}"))

        # get all matches
        match_labelers = []
        for idx, report_section in enumerate(report_sections):
            matches = self.term_graph.match_string(report_section)
            if len(matches) == 0:
                continue
            display(Markdown(f"---\n ## Section {idx + 1}"))
            for match in matches:
                labeler = self.build_match_labeler(match, report_section)
                labeler["match"]["section_idx"] = idx
                match_labelers.append(labeler)

        out = widgets.Output()
        save_button =  widgets.Button(
            description='Save',
            disabled=False,
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save',
            icon=''
        )

        def record_labels(b):
            """
            Record the labels currently entered in the match_labelers.
            """
            with out:
                labeled_matches = []
                for match_labeler in match_labelers:
                    match = match_labeler["match"]
                    labeled_match = copy.deepcopy(match)
                    labeled_match["exam_id"] = exam_id
                    for task, task_button in match_labeler["task_buttons"].items():
                        labeled_match[f"{task}_label"] = task_button.value
                    labeled_match["not_applicable"] = match_labeler["na_button"].value
                    labeled_matches.append(labeled_match)

                if self.labels_df is not None:
                    # if exam already labeled, filter out old labels, concat new labels
                    self.labels_df = self.labels_df[self.labels_df["exam_id"] != exam_id]
                    self.labels_df = pd.concat([self.labels_df,
                                                pd.DataFrame(labeled_matches)])
                else:
                    self.labels_df = pd.DataFrame(labeled_matches)
                self.labels_df.to_csv(os.path.join(self.dir, "match_labels.csv"))
                self.labeled_exams.add(exam_id)

                # update save button
                save_button.button_style = 'success'
                save_button.icon = 'check'
                save_button.description = 'Saved.'
                skip_button.disabled = True
        save_button.on_click(record_labels)

        skip_button =  widgets.Button(
            description='Skip',
            disabled=False,
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Skip',
            icon=''
        )
        def skip(b):
            """
            Record the labels currently entered in the match_labelers.
            """
            with out:
                self.skipped_exams.add(exam_id)

                # update save button
                skip_button.button_style = 'danger'
                skip_button.description = 'Skipped.'
                save_button.disabled = True
        skip_button.on_click(skip)

        display(Markdown("---"))
        display(widgets.HBox([save_button, skip_button]))
