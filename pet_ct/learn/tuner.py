"""
An implementation of a Hyperband tuner.

Source of hyperband schedule implementation:
https://github.com/HazyResearch/metal/
"""
import logging
import json
import os
import pickle
import random
from itertools import cycle, product
from time import strftime, time
from collections import deque
import math

import numpy as np
import pandas as pd

from pet_ct.learn.experiment import Experiment
from pet_ct.util.util import Process, ensure_dir_exists, create_notebook


class Tuner(Process):
    def __init__(self, dir, train_schema, eval_args={}):
        """
        """
        super().__init__(dir)
        self.train_schema = train_schema
        self.eval_args = eval_args

    def search(self, train_args={}, max_search=None,
               train_split="train", valid_split="valid", overwrite=False):
        """
        Performs a simple search over all
        """
        self._create_experiments(self.train_schema)
        for idx, experiment_dir in enumerate(self.experiments.keys()):
            if os.path.isfile(os.path.join(experiment_dir, 'best',
                                           f"{valid_split}_metrics.json")):
                logging.info(f'{experiment_dir} is already trained. Continuing...')
                continue
            logging.info(f"Training experiment at {experiment_dir}\n {'-'*30}")
            if max_search is not None and idx >= max_search:
                break

            experiment = Experiment.load_from_dir(process_dir=experiment_dir)
            experiment.train(train_split=train_split, valid_split=valid_split, overwrite=overwrite)

    def evaluate(self, eval_split, reload_epochs=["best"],
                 data_loader_config=None, dataset_dir=None, targets_dir=None, devices=[0]):
        """
        """
        for reload_epoch in reload_epochs:
            for experiment_dir in os.listdir(os.path.join(self.dir, "candidates")):
                experiment_dir = os.path.join(self.dir, "candidates", experiment_dir)
                if os.path.isfile(os.path.join(experiment_dir, reload_epoch,
                                               f"{eval_split}_metrics.json")) or \
                   os.path.isfile(os.path.join(experiment_dir,
                                               f"{eval_split}_metrics.json")):
                    logging.info(f"Already evaluated: {experiment_dir} at {reload_epoch}")
                    continue
                logging.info(f"Evaluating {experiment_dir} at {reload_epoch}.")

                with open(os.path.join(experiment_dir, "params.json")) as f:
                    params = json.load(f)
                params["process_args"]["devices"] = devices
                if data_loader_config is not None:
                    params["process_args"]["dataloader_configs"] = [data_loader_config]
                if dataset_dir is not None:
                    params["process_args"]["dataset_args"]["dataset_dir"] = dataset_dir
                if targets_dir is not None:
                    params["process_args"]["dataset_args"]["targets_dir"] = targets_dir
                params["process_args"]["reload_weights"] = reload_epoch
                try:
                    experiment = Experiment(experiment_dir, **params["process_args"])
                    experiment.evaluate(eval_split)
                except Exception as e:
                    logging.info(f"Failed: {os.path.join(experiment_dir, reload_epoch)}")
                    logging.info(str(e))

    def _train_model(self, experiment_dir):
        experiment = Experiment.load_from_dir(process_dir=experiment_dir)
        last_metrics = experiment.train()
        self.experiments[experiment_dir]["last_metrics"] = last_metrics
        return last_metrics

    def _run(self, overwrite=False,
             mode="eval", train_split="train", valid_split="valid"):
        if mode == "eval":
            logging.info("Tuner evaluating...")
            self.evaluate(**self.eval_args)
        elif mode == "train":
            logging.info("Tuner searching...")
            self.search(train_split=train_split, valid_split=valid_split, overwrite=overwrite)

    def _create_experiments(self, params):
        """
        """
        logging.info("Expanding params")
        params = self._expand_params(params)

        logging.info("Creating experiment directories")
        self.candidates_dir = os.path.join(self.dir, "candidates")
        self.experiments = {}
        for idx, curr_params in enumerate(params):
            experiment_dir = os.path.join(self.candidates_dir, f"exp_{idx}")
            self.experiments[experiment_dir] = {"last_metrics": None,
                                                "best_metrics": None,
                                                "params": curr_params}
            ensure_dir_exists(experiment_dir)
            with open(os.path.join(experiment_dir, 'params.json'), 'w') as f:
                json.dump(curr_params, f, indent=4)
            create_notebook(experiment_dir, notes=f"Created as part of tuner {self.dir}")

    def _expand_params(self, param):
        """
        Recursively expands param and its children.
        """
        if type(param) is list:
            child_expansions = []
            for child_idx, child_value in enumerate(param):
                child_expansion = self._expand_params(child_value)
                child_expansions.append(child_expansion)

            return list(product(*child_expansions))

        elif type(param) is dict and not param.get("tuneable", False):
            child_expansions = []
            child_keys = []
            for child_key, child_value in param.items():
                child_expansion = self._expand_params(child_value)
                child_expansions.append(child_expansion)
                child_keys.append(child_key)

            expansions = []
            for expansion in product(*child_expansions):
                expansions.append({child_keys[idx]: val for idx, val in enumerate(expansion)})
            return expansions

        elif type(param) is dict:
            tune_params = param
            expanded_params = []
            if tune_params["type"] == "discrete":
                if tune_params["search_mode"] == "grid":
                    for opt in tune_params["opts"]:
                        expanded_params.extend(self._expand_params(param=opt))
                else:
                    raise ValueError(f"Search mode {tune_params['search_mode']} " +
                                     f"not recognized")
            elif tune_params["type"] == "continuous":
                pass
            else:
                raise ValueError(f"Hyperparameter type {tune_params['type']} not " +
                                 f"recognized.")
            return expanded_params
        else:
            return [param]


# class HyperbandTuner(Tuner):
#     """
#     """
#     def __init__(self,
#                  epochs_budget=20,
#                  proportion_discard=3):
#         """
#         Args:
#             epochs_budget   (int) the maximum number of epochs for each
#                             heat (R from the manuscript).
#             proportion_discard  (float) the proportion of experiments to discard at each
#                                 iteration (eta from manuscript)
#         """
#         super().__init__()
#         # Hyperband parameters
#         self.hyperband_epochs_budget = epochs_budget
#         self.hyperband_proportion_discard = proportion_discard

#     def get_largest_schedule_within_budget(self, budget, proportion_discard):
#         """
#         Gets the largest hyperband schedule within target_budget.
#         This is required since the original hyperband algorithm uses R,
#         the maximum number of resources per configuration.
#         TODO(maxlam): Possibly binary search it if this becomes a bottleneck.
#         Args:
#             budget: total budget of the schedule.
#             proportion_discard: hyperband parameter that specifies
#                 the proportion of configurations to discard per iteration.
#         """

#         # Exhaustively generate schedules and check if
#         # they're within budget, adding to a list.
#         valid_schedules_and_costs = []
#         for R in range(1, budget):
#             schedule = self.generate_hyperband_schedule(R, proportion_discard)
#             cost = self.compute_schedule_cost(schedule)
#             if cost <= budget:
#                 valid_schedules_and_costs.append((schedule, cost))

#         # Choose a valid schedule that maximizes usage of the budget.
#         valid_schedules_and_costs.sort(key=lambda x: x[1], reverse=True)
#         return valid_schedules_and_costs[0][0]

#     def compute_schedule_cost(self, schedule):
#         # Sum up all n_i * r_i for each band.
#         flattened = [item for sublist in schedule for item in sublist]
#         return sum([x[0] * x[1] for x in flattened])

#     def generate_hyperband_schedule(self, R, eta):
#         """
#         Generate hyperband schedule according to the paper.
#         Args:
#             R: maximum resources per config.
#             eta: proportion of configruations to discard per
#                 iteration of successive halving.
#         Returns: hyperband schedule, which is represented
#             as a list of brackets, where each bracket
#             contains a list of (num configurations,
#             num resources to use per configuration).
#             See the paper for more details.
#         """
#         schedule = []
#         s_max = int(math.floor(math.log(R, eta)))
#         # B = (s_max + 1) * R
#         for s in range(0, s_max + 1):
#             n = math.ceil(int((s_max + 1) / (s + 1)) * eta ** s)
#             r = R * eta ** (-s)
#             bracket = []
#             for i in range(0, s + 1):
#                 n_i = int(math.floor(n * eta ** (-i)))
#                 r_i = int(r * eta ** i)
#                 bracket.append((n_i, r_i))
#             schedule = [bracket] + schedule
#         return schedule

#     def search(self, experiments, valid_data, init_args=[], train_args=[],
#                init_kwargs={}, train_kwargs={}, module_args={}, module_kwargs={},
#                max_search=None, shuffle=True, verbose=True, seed=None, **score_kwargs):
#         """
#         Performs hyperband search according to the generated schedule.
#         At the beginning of each bracket, we generate a
#         list of random configurations and perform
#         successive halving on it; we repeat this process
#         for the number of brackets in the schedule.
#         Args:
#             init_args: (list) positional args for initializing the model
#             train_args: (list) positional args for training the model
#             valid_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
#                 X (data) and Y (labels) for the dev split
#             search_space: see ModelTuner's config_generator() documentation
#             max_search: see ModelTuner's config_generator() documentation
#             shuffle: see ModelTuner's config_generator() documentation
#         Returns:
#             best_model: the highest performing trained model found by Hyperband
#             best_config: (dict) the config corresponding to the best model
#         Note: Initialization is performed by ModelTuner instead of passing a
#         pre-initialized model so that tuning may be performed over all model
#         parameters, including the network architecture (which is defined before
#         the train loop).
#         """
#         self._clear_state(seed)
#         self.search_space = experiments

#         # Loop over each bracket
#         n_models_scored = 0
#         for bracket_index, bracket in enumerate(self.hyperband_schedule):

#             # Sample random configurations to seed SuccessiveHalving
#             n_starting_configurations, _ = bracket[0]
#             configurations = list(
#                 self.config_generator(
#                     experiments,
#                     max_search=n_starting_configurations,
#                     rng=self.rng,
#                     shuffle=True,
#                 )
#             )

#             # Successive Halving
#             for band_index, (n_i, r_i) in enumerate(bracket):

#                 assert len(configurations) <= n_i

#                 # Evaluate each configuration for r_i epochs
#                 scored_configurations = []
#                 for i, configuration in enumerate(configurations):

#                     cur_model_index = n_models_scored

#                     # Set epochs of the configuration
#                     configuration["n_epochs"] = r_i

#                     # Train model and get the score
#                     score, model = self._test_model_config(
#                         f"{band_index}_{i}",
#                         configuration,
#                         valid_data,
#                         init_args=init_args,
#                         train_args=train_args,
#                         init_kwargs=init_kwargs,
#                         train_kwargs=train_kwargs,
#                         module_args=module_args,
#                         module_kwargs=module_kwargs,
#                         verbose=verbose,
#                         **score_kwargs,
#                     )

#                     # Add score and model to list
#                     scored_configurations.append(
#                         (score, cur_model_index, configuration)
#                     )
#                     n_models_scored += 1

#                 # Sort scored configurations by score
#                 scored_configurations.sort(key=lambda x: x[0], reverse=True)

#                 # Successively halve the configurations
#                 if band_index + 1 < len(bracket):
#                     n_to_keep, _ = bracket[band_index + 1]
#                     configurations = [x[2] for x in scored_configurations][:n_to_keep]

#         print("=" * 60)
#         print(f"[SUMMARY]")
#         print(f"Best model: [{self.best_index}]")
#         print(f"Best config: {self.best_config}")
#         print(f"Best score: {self.best_score}")
#         print("=" * 60)

#         # Return best model
#         return self._load_best_model(clean_up=True)

#     def pretty_print_schedule(self, hyperband_schedule, describe_hyperband=True):
#         """
#         Prints scheduler for user to read.
#         """
#         print("=========================================")
#         print("|           Hyperband Schedule          |")
#         print("=========================================")
#         if describe_hyperband:
#             # Print a message indicating what the below schedule means
#             print(
#                 "Table consists of tuples of "
#                 "(num configs, num_resources_per_config) "
#                 "which specify how many configs to run and "
#                 "for how many epochs. "
#             )
#             print(
#                 "Each bracket starts with a list of random "
#                 "configurations which is successively halved "
#                 "according the schedule."
#             )
#             print(
#                 "See the Hyperband paper "
#                 "(https://arxiv.org/pdf/1603.06560.pdf) for more details."
#             )
#             print("-----------------------------------------")
#         for bracket_index, bracket in enumerate(hyperband_schedule):
#             bracket_string = "Bracket %d:" % bracket_index
#             for n_i, r_i in bracket:
#                 bracket_string += " (%d, %d)" % (n_i, r_i)
#             print(bracket_string)
#         print("-----------------------------------------")
