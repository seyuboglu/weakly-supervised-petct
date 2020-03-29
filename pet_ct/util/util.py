"""General utility functions"""

import json
import os
import sys
import logging
import smtplib
import traceback
import socket
from re import sub
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import collections
from inspect import getcallargs, signature, getargvalues
import glob
from functools import wraps


import numpy as np
import torch
import tqdm


def write_params(process_dir, params):
    """
    """
    ensure_dir_exists(process_dir)
    with open(os.path.join(process_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=4)


def process(fn):
    """
    Decorator to wrap process functions in.
    """
    @wraps(fn)
    def with_logging(*args, **kwargs):
        args_dict = getcallargs(fn, *args, **kwargs)
        if args_dict["process_dir"] is not None:
            ensure_dir_exists(args_dict["process_dir"])
            set_logger(os.path.join(args_dict["process_dir"], 'process.log'),
                       level=logging.INFO,
                       console=True)
            params_dict = {
                "notebook_dir": sys.path[0],
                "process_fn": fn.__name__,
                "process_args": args_dict
            }
            write_params(args_dict["process_dir"], params_dict)
        return fn(*args, **kwargs)

    return with_logging


class Process():
    """
    """
    def __init__(self, dir, name=None):
        self.dir = dir
        set_logger(os.path.join(dir, "process.log"), level=logging.INFO, console=True)

        params = load_params(self.dir)
        logging.info(json.dumps(params, indent=2))

        process_class = params["process_class"]
        log_title(process_class)

    @classmethod
    def load_from_dir(cls, process_dir, name=None):
        params = load_params(process_dir)
        process = cls(process_dir, **params["process_args"])
        return process

    def _run(self, overwrite=False):
        """
        """
        pass

    def run(self, overwrite=False, notify=False, commit=False, **kwargs):
        """
        """
        if os.path.isfile(os.path.join(self.dir, 'stats.csv')) and not overwrite:
            print("Process already run.")
            return False

        if notify:
            try:
                self._run(overwrite, **kwargs)
            except:
                tb = traceback.format_exc()
                self.notify_user(error=tb)
                return False
            else:
                self.notify_user()
                return True
        self._run(overwrite, **kwargs)
        return True

    def notify_user(self, error=None, email_address=None):
        """
        Notify the user by email if there is an exception during the process.
        If the program completes without error, the user will also be notified.
        """
        if email_address is None:
            return
        # read params
        params = load_params(self.dir)
        params_string = str(params)

        if error is None:
            subject = "Process Completed: " + self.dir
            message = ("Yo!\n",
                       "Good news, your process just finished.",
                       "You were running the process on: {}".format(
                           socket.gethostname()),
                       "---------------------------------------------",
                       "See the results here: {}".format(self.dir),
                       "---------------------------------------------",
                       "The parameters you fed to this process were: {}".format(
                           params_string),
                       "---------------------------------------------",
                       "Thanks!")
        else:
            subject = "Process Error: " + self.dir
            message = ("Uh Oh!\n",
                       "Your process encountered an error.",
                       "You were running a process found at: {}".format(self.dir),
                       "You were running the process on: {}".format(
                           socket.gethostname()),
                       "---------------------------------------------",
                       "Check out the error message: \n{}".format(error),
                       "---------------------------------------------",
                       "The parameters you fed to this process were: {}".format(
                           params_string),
                       "---------------------------------------------",
                       "Thanks!")
            logging.error(error)

        message = "\n".join(message)
        send_email(subject, message, to_addr=email_address)


def load_params(process_dir):
    """
    Loads the params file in the process directory specified by process_dir.
    @param process_dir (str)
    @returns params (dict)
    """
    if os.path.exists(os.path.join(process_dir, "params.json")):
        params_path = os.path.join(process_dir, "params.json")
        with open(params_path) as f:
            params = json.load(f)

    else:
        raise Exception(f"No params.json file found at {[process_dir]}.")

    return params


def send_email(subject, message, to_addr):
    username = None # TODO: add your username
    password = None # TODO: add your username
    if username is None or password is None:
        return
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg['From'] = f"{username}@gmail.com"
    msg['To'] = to_addr
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    problems = server.sendmail(f"{username}@gmail.com",
                               to_addr,
                               msg.as_string())
    server.quit()


def set_logger(log_path, level=logging.INFO, console=True):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.basicConfig(format='')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    if console and False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def log_title(title):
    """
    """
    logging.info("{}".format(title))
    logging.info("Sabri Eyuboglu and Geoff Angus")
    logging.info("AIMI – Stanford University – 2018")
    logging.info("---------------------------------")


def dir_has_subdirs(dir, subdirs):
    """
    Checks if a directory has all the subdirectories specified by subdirs.
    args:
        dir     (str) the directory to check
        subdirs (list)  the directory names to look for
    returns:
        has_subdirs (bool)  if the exam has all
    """
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(dir, subdir)):
            return False
    return True


def ensure_dir_exists(dir):
    """
    Ensures that a directory exists. Creates it if it does not.
    args:
        dir     (str)   directory to be created
    """
    if not(os.path.exists(dir)):
        ensure_dir_exists(os.path.dirname(dir))
        os.mkdir(dir)

def get_latest_file(dir):
    list_of_files = glob.glob(f'{dir}/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def save_dict_to_json(json_path, d):
    """
    Saves a python dictionary into a json file
    Args:
        d           (dict) of float-castable values (np.float, int, float, etc.)
        json_path   (string) path to json file
    """
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def flatten_nested_dicts(d, parent_key='', sep='_'):
    """
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_from_nested_dict(nested_dicts, path):
    """
    Retrieves a value from a nested_dict.
    Args:

        path    (str)
    """
    value = nested_dicts
    for key in path.split("/"):
        if type(value) is dict:
            value = value[key]

        elif type(value) is list:
            value = value[int(key)]

    return value


def hard_to_soft(Y_h, k):
    """Converts a 1D tensor of hard labels into a 2D tensor of soft labels
    Source: MeTaL from HazyResearch, https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    Args:
        Y_h: an [n], or [n,1] tensor of hard (int) labels in {1,...,k}
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [n, k] where Y_s[i, j-1] is the soft
            label for item i and label j
    """
    Y_h = Y_h.clone()
    if Y_h.dim() > 1:
        Y_h = Y_h.squeeze()
    assert Y_h.dim() == 1
    assert (Y_h >= 0).all()
    assert (Y_h < k).all()
    n = Y_h.shape[0]
    Y_s = torch.zeros((n, k), dtype=Y_h.dtype, device=Y_h.device)
    for i, j in enumerate(Y_h):
        Y_s[i, int(j)] = 1.0
    return Y_s


def soft_to_hard(Y_s, break_ties="random"):
    """Break ties in each row of a tensor according to the specified policy

    Source: MeTaL from HazyResearch, https://github.com/HazyResearch/metal/
    Modified slightly to accommodate PyTorch tensors.
    Args:
        Y_s: An [n, k] np.ndarray of probabilities
        break_ties: A tie-breaking policy:
            "abstain": return an abstain vote (0)
            "random": randomly choose among the tied options
                NOTE: if break_ties="random", repeated runs may have
                slightly different results due to difference in broken ties
            [int]: ties will be broken by using this label
    """
    n, k = Y_s.shape
    maxes, argmaxes = Y_s.max(dim=1)
    diffs = torch.abs(Y_s - maxes.reshape(-1, 1))

    TOL = 1e-5
    Y_h = torch.zeros(n, dtype=torch.int64)
    for i in range(n):
        max_idxs = torch.where(diffs[i, :] < TOL, Y_s[i], torch.tensor(0.0, dtype=Y_s.dtype))
        max_idxs = torch.nonzero(max_idxs).reshape(-1)
        if len(max_idxs) == 1:
            Y_h[i] = max_idxs[0]
        # Deal with "tie votes" according to the specified policy
        elif break_ties == "random":
            Y_h[i] = torch.as_tensor(np.random.choice(max_idxs))
        elif break_ties == "abstain":
            Y_h[i] = 0
        elif isinstance(break_ties, int):
            Y_h[i] = break_ties
        else:
            ValueError(f"break_ties={break_ties} policy not recognized.")
    return Y_h


def extract_kwargs(kwargs_tuple):
    """  Converts  a tuple of kwarg tokens to a dictionary. Expects in format
    (--key1, value1, --key2, value2)
    Args:
        kwargs_tuple (tuple(str)) tuple of list

    """
    if len(kwargs_tuple) == 0:
        return {}
    assert kwargs_tuple[0][:2] == "--", f"No key for first kwarg {kwargs_tuple[0]}"
    curr_key = None
    kwargs_dict = {}
    for token in kwargs_tuple:
        if token[:2] == "--":
            curr_key = token[2:]
        else:
            kwargs_dict[curr_key] = token

    return kwargs_dict


def expand_to_list(item, n):
    """Expands item into list of n length, unless already a list."""
    if type(item) is list:
        assert len(item) == n
        return item
    else:
        return [item for _ in range(n)]


def flex_concat(items, dim=0):
    """
    Concatenates the items in list items. All elements in items must be of the same type.
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return sum(items, [])

    elif isinstance(items[0], torch.Tensor):
        # zero-dimensional tensors cannot be concatenated
        items = [item.expand(1) if not item.shape else item for item in items]
        return torch.cat(items, dim=0)

    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")

def flex_stack(items, dim=0):
    """
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return items

    elif isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0)

    elif isinstance(items[0], np.ndarray):
        return np.stack(items, axis=0)

    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")

def place_on_gpu(data, device=0):
    """
    Recursively places all 'torch.Tensor's in data on gpu and detaches.
    If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i], device) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_gpu(val, device) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def place_on_cpu(data):
    """
    Recursively places all 'torch.Tensor's in data on cpu and detaches from computation
    graph. If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_cpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_cpu(val) for key,val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data


def get_batch_size(data, dim=0):
    """
    """
    if isinstance(data, list):
        return len(data)

    elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return data.shape[0]

    else:
        raise TypeError(f"Unrecognized type f{type(data)}")

def log_cuda_memory(devices=[0], units="g", ndigits=4):
    """
    """
    conversion = 1
    if units.lower() == "g":
        conversion = 1e-9
    elif units.lower() == "m":
        conversion = 1e-6
    elif units.lower() == "k":
        conversion = 1e-3
    else:
        raise ValueError(f"Units {units} not recognized.")
    units = f"{units.upper()}B"

    for device in devices:
        logging.info(f"Device {device}")
        allocated = round(torch.cuda.memory_allocated(device) * conversion, ndigits)
        max_allocated = round(torch.cuda.max_memory_allocated(device) * conversion, ndigits)
        logging.info(f"Memory allocated: {allocated} {units}  / {max_allocated} {units} ")

        cached = round(torch.cuda.memory_cached(device) * conversion, ndigits)
        max_cached = round(torch.cuda.max_memory_cached(device) * conversion, ndigits)
        logging.info(f"Memory cached: {cached} {units} / {max_cached} {units} ")
        logging.info("---------------")


def log_predictions(targets, predictions, info=None):
    """
    """
    if "report_generation" in predictions:
        for idx, (target, pred) in enumerate(zip(targets["report_generation"],
                                          predictions["report_generation"])):
            if info is not None:
                logging.info(f"exam_id: {info[idx]['exam_id']}")
            logging.info(f"target: {' '.join(target)}")
            logging.info(f"preds : {' '.join(pred)}")


def create_notebook(experiment_dir,
                    notes=None,
                    template_path="notebooks/experiment_template.ipynb"):
    """
    """
    notebook_path = os.path.join(experiment_dir, "notebook.ipynb")
    if not os.path.exists(notebook_path):
        logging.info("Generating notebook")
        with open(template_path) as f:
            notebook_json = json.load(f)

        # set directory and name
        for cell in notebook_json["cells"]:
            if "<<EXPERIMENT SPEC>>" in cell["source"][0]:
                cell["source"] = ["### <<EXPERIMENT SPEC>> ###\n",
                                  f"experiment_dir = \"{experiment_dir}\"\n",
                                 f"name = \"{experiment_dir.split('/')[-2]}\"\n"]
                break

        if notes is not None:
            for cell in notebook_json["cells"]:
                if "## Notes" in cell["source"][0]:
                    cell["source"].insert(2, f"{notes} \n")
                    break

        with open(notebook_path, 'w') as f:
            json.dump(notebook_json, f, indent=2)


def get_exps_by_param(experiment_dirs, param_path, param_values=None):
    """
    """
    param_value_to_exp_dirs = collections.defaultdict(list)
    for exp_dir in experiment_dirs:
        with open(os.path.join(exp_dir, "params.json")) as f:
            params = json.load(f)
        param_value = get_from_nested_dict(params, param_path)
        if param_values is None or param_value in param_values:
            param_value_to_exp_dirs[param_value].append(exp_dir)

    return param_value_to_exp_dirs