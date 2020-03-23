"""
Module for building an HDF5 file.
"""
import sys
import logging
import os
import json
from time import localtime, strftime
from datetime import datetime
from shutil import copyfile

import click

from pet_ct.data.h5_dataset import DatasetBuilder, DatasetEditor
from pet_ct.data.dicom import DicomParser
from pet_ct.data.splitter import BinarySplitter, MatchesSplitter
from pet_ct.data.labels_builder import LabelsBuilder, MetalLabelsBuilder, ProgrammaticLabelsBuilder
from pet_ct.data.vocab import GenerateWordPieceVocab, MergeVocabs
from pet_ct.data.manual import ExamLabelsPredictor, MatchLabeler, LabelerDatasetBuilder
from pet_ct.learn.experiment import Experiment
from pet_ct.learn.tuner import Tuner
from pet_ct.util.util import (set_logger, log_title, load_params, extract_kwargs,
                              ensure_dir_exists, create_notebook)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument(
    "process_dir",
    type=str,
)
@click.option(
    "--overwrite",
    is_flag=True
)
@click.option(
    "--notify",
    type=bool,
    default=True
)
@click.option(
    "--commit/--no-commit",
    default=False
)
@click.argument(
    "kwargs",
    nargs=-1,
    type=click.UNPROCESSED
)
def run(process_dir, overwrite, notify, commit, kwargs):
    """
    """
    kwargs = extract_kwargs(kwargs)
    params = load_params(process_dir)

    process_class = params["process_class"]
    log_title(process_class)

    process = globals()[process_class](process_dir, **params["process_args"])
    process.run(overwrite=overwrite, notify=notify, commit=commit, **kwargs)


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument(
    "group_dir",
    type=str
)
@click.option(
    "-c",
    "--copy_dir",
    type=str,
    default=None
)
@click.option(
    "-n",
    "--name",
    type=str,
    default="exp"
)
@click.option(
    "-m",
    "--message",
    type=str,
    default=None
)
def create(group_dir, copy_dir, name, message):
    """ Creates a new process directory with a params.json file and notebook.ipynb.
    Every process belongs to a group (a directory containing a set of related processes).
    When we call create on a group (specified by group_dir), by default a new process
    will be created with the same parameters as the most recently created experiment in
    the group.
    args:
        group_dir (str)
    """
    print(copy_dir)
    ensure_dir_exists(group_dir)

    if copy_dir is None:
        copy_dirs = [curr_dir for curr_dir in os.listdir(group_dir)]
        if not copy_dirs:
            copy_dir = "experiments/_default"
        else:
            # take params from most recent experiment, exclude dirs with leading '_'
            copy_dirs = [copy_dir for copy_dir in copy_dirs if copy_dir[0] != '_']
            copy_dir = os.path.join(group_dir, sorted(copy_dirs)[-1])

    name = f"{strftime('%m-%d_%H-%M', localtime())}_{name}"
    process_dir = os.path.join(group_dir, name)
    ensure_dir_exists(process_dir)

    copyfile(src=os.path.join(copy_dir, "params.json"),
             dst=os.path.join(process_dir, "params.json"))

    create_notebook(process_dir, notes=message)

    print(f"Created process at '{process_dir}' with parameters from '{copy_dir}'")
    print(f"Open process notebook with:")
    print(f"jupyter notebook {process_dir}/notebook.ipynb --no-browser --port=8200")
    print(f"Run the process with:")
    print(f"run {process_dir}")

