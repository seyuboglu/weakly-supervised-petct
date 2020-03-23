"""
"""
import logging
import gc

import torch
from torch import nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm
from metal.utils import place_on_gpu

from pet_ct.model.base_model import BaseModel
from pet_ct.data.vocab import WordVocab
from pet_ct.analysis.metrics import Metrics
import pet_ct.model.modules as modules
import pet_ct.model.losses as losses
from pet_ct.util.util import log_cuda_memory


class MTReportModel(BaseModel):
    """
    """
    def __init__(self,
                 encoder_class=None, encoder_args={},
                 class_decoder_class=None, class_decoder_args={},
                 report_decoder_class=None, report_decoder_args={},
                 class_loss_class=None, class_loss_args={},
                 report_loss_class=None, report_loss_args={},
                 loss_weights=None, vocab_args={},
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

        self.class_decoder = getattr(modules, class_decoder_class)(**class_decoder_args)

        self.vocab = WordVocab(**vocab_args)
        self.report_decoder = getattr(modules,
                                      report_decoder_class)(self.vocab,
                                                            **report_decoder_args)
        self.class_loss = getattr(losses, class_loss_class)(**class_loss_args)
        self.report_loss = getattr(losses,
                                   report_loss_class)(vocab=self.vocab,
                                                      **report_loss_args)
        self.loss_weights = loss_weights

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
        reports_padded = self.vocab.to_input_tensor(targets["report_generation"],
                                                    device=self.device)
        encoding = self.encoder(inputs)

        report_outputs = self.report_decoder(encoding, reports_padded)

        class_outputs = self.class_decoder(encoding)
        outputs = {"report_generation": report_outputs,
                   "abnormality_detection": class_outputs}

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
        class_outputs = self.class_decoder(encoding)
        #class_probs = {task: nn.functional.softmax(out, dim=1)
        #                    for task, out in class_outputs.items()}
        return {"abnormality_detection": nn.functional.softmax(class_outputs, dim=1)}

    def _predict_reports(self, encoding):
        """
        """
        preds = []
        for curr_encoding in torch.split(encoding, split_size_or_sections=1, dim=0):
            preds.append(self.report_decoder.beam_search(curr_encoding)[0].value)

        return preds

    def predict(self, inputs):
        """
        """
        encoding = self._encode(inputs)
        task_predictions = self._predict_tasks(encoding)
        report_prediction = self._predict_reports(encoding)
        predictions = {"report_generation": report_prediction, **task_predictions}
        return predictions

    def loss(self, outputs, targets):
        """
        """
        report_targets = targets.pop("report_generation")
        report_targets_padded = self.vocab.to_input_tensor(report_targets,
                                                           device=self.device)
        reports_loss = self.report_loss(inputs=outputs.pop("report_generation"),
                                        targets=report_targets_padded)
        class_loss = self.class_loss(inputs=outputs, targets=targets)

        targets["report_generation"] = report_targets

        losses = {"report_generation": reports_loss,
                  "abnormality_detection": class_loss}

        if self.loss_weights is None:
            loss = torch.mean(self.losses.values())
        else:
            loss = 0.0
            for task, weight in self.loss_weights.items():
                loss += weight * losses[task]

        return loss
