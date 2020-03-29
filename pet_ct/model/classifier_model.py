"""
"""
import logging
import gc
import json

import torch
from torch import nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm
from metal.utils import place_on_gpu

from pet_ct.model.base_model import BaseModel
from pet_ct.analysis.metrics import Metrics
import pet_ct.model.modules as modules
import pet_ct.model.losses as losses
from pet_ct.util.util import log_cuda_memory, soft_to_hard


class ClassifierModel(BaseModel):
    """
    """
    def __init__(self,
                 encoder_class=None, encoder_args={},
                 decoder_config={},
                 loss_class=None, loss_args={},
                 optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 pretrained_configs=[],
                 cuda=True, devices=[0]):
        """
        """
        super().__init__(optim_class, optim_args, scheduler_class, scheduler_args,
                         pretrained_configs, cuda, devices)

        self.encoder = getattr(modules, encoder_class)(**encoder_args)
        self.encoder = nn.DataParallel(self.encoder, device_ids=self.devices)

        self.decoder = getattr(modules, decoder_config['class'])(**decoder_config['args'])

        self.loss_fn = getattr(losses, loss_class)(**loss_args)

        self._post_init()

    def forward(self, inputs, targets):
        """
        Args:
            inputs  (torch.Tensor) a (batch_size, ...) shaped input tensor
            targets     (list) a list of task targets from t0, t1...tn. The last element
                        should be a list(list(str)) representing the target report.

        Return:
            outputs (list) of
        """
        encoding = self.encoder(inputs)
        outputs = self.decoder(encoding)
        return outputs

    def _encode(self, inputs):
        """ Accepts a stack of images as input and outputs an encoding of that
        Args:
            inputs  (torch.Tensor)  A torch tensor representing a stack of images.
                    Has shape (batch_size, length, height, width, num_channels).
        Returns
            outputs (torch.Tensor) encoding with shape (batch_size, length, height, width,
                                                        num features)
        """
        encoding = self.encoder(inputs)
        return encoding

    def predict(self, inputs):
        """
        """
        encoding = self._encode(inputs)
        outputs = self.decoder(encoding)
        if type(outputs) != dict:
            probs = nn.functional.softmax(outputs, dim=1)
        else:
            probs = nn.functional.softmax(outputs['out'], dim=1)
        return probs

    def loss(self, inputs, targets):
        """
        """
        return self.loss_fn(inputs, targets)


class MTClassifierModel(BaseModel):
    """
    """
    def __init__(self,
                 encoder_class=None, encoder_args={},
                 decoder_config={},
                 task_configs=[],
                 loss_class=None, loss_args={},
                 optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 pretrained_configs=[],
                 cuda=True, devices=[0],
                 break_ties="random"):
        """
        """
        super().__init__(optim_class, optim_args, scheduler_class, scheduler_args,
                         pretrained_configs, cuda, devices)

        self.encoder = getattr(modules, encoder_class)(**encoder_args)
        self.encoder = nn.DataParallel(self.encoder, device_ids=self.devices)

        self.decoder = getattr(modules,
                               decoder_config['class'])(task_configs=task_configs,
                                                        **decoder_config['args'])

        self.loss_fn = getattr(losses, loss_class)(task_configs=task_configs,
                                                   **loss_args)

        self.break_ties = break_ties

        self._post_init()


    def forward(self, inputs, targets):
        """
        Args:
            inputs  (torch.Tensor) a (batch_size, ...) shaped input tensor
            targets     (list) a list of task targets from t0, t1...tn. The last element
                        should be a list(list(str)) representing the target report.

        Return:
            outputs (dict) a dict that matches keys to their respective tasks.
        """
        encoding = self.encoder(inputs)
        outputs = self.decoder(encoding)
        assert(outputs.keys() == targets.keys())
        return outputs

    def _encode(self, inputs):
        """ Accepts a stack of images as input and outputs an encoding of that
        Args:
            inputs  (torch.Tensor)  A torch tensor representing a stack of images.
                    Has shape (batch_size, length, height, width, num_channels).
        Returns
            outputs (torch.Tensor) encoding with shape (batch_size, length, height, width,
                                                        num features)
        """
        encoding = self.encoder(inputs)
        return encoding

    def _predict_tasks(self, encoding):
        """
        """
        outputs = self.decoder(encoding)
        outputs = {task: out if type(out) != dict else out['out']
                   for task, out in outputs.items()}

        probs = {task: nn.functional.softmax(out, dim=1)
                 for task, out in outputs.items()}
        return probs

    def predict(self, inputs):
        """
        """
        encoding = self._encode(inputs)
        probs = self._predict_tasks(encoding)
        return probs

    def loss(self, inputs, targets):
        """
        """
        return self.loss_fn(inputs, targets)

    def _get_labels(self, targets):
        """Returns the targets as indices

        Only uses the tie breaking feature if training. Else, it defaults to index 0.
        """
        for task in targets.keys():
            targets[task] = soft_to_hard(targets[task].cpu().detach(),
                                         break_ties=0)
        return targets
    
    @staticmethod
    def load_params(path):
        """
        Loads parameters from .json file at path and distributes shared parameters as 
        necessary for the mt_classifier model.
        """
        with open(path) as f:
            params = json.load(f)["process_args"]
        
        # distribute shared params
        new_task_configs = []
        for task_config in params["task_configs"]:
            new_task_config = params["default_task_config"].copy()
            new_task_config.update(task_config)
            new_task_configs.append(new_task_config)
        task_configs = new_task_configs

        params["model_args"]["task_configs"] = task_configs
        params["dataset_args"]["task_configs"] = task_configs
        
        return params