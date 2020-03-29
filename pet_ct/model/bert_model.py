"""
"""
import logging
import gc
from collections import namedtuple

import torch
from torch import nn
import numpy as np
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm
from metal.utils import place_on_gpu
from transformers import BertModel, BertForPreTraining, BertConfig
from transformers.modeling_bert import BertOnlyMLMHead
from pet_ct.model.base_model import BaseModel
from pet_ct.data.vocab import WordPieceVocab
from pet_ct.analysis.metrics import Metrics
import pet_ct.model.modules as modules
import pet_ct.model.losses as losses
from pet_ct.util.util import log_cuda_memory


class BertScanModel(BaseModel):
    """
    """
    def __init__(self,
                 scan_encoder_class=None, scan_encoder_args={},
                 bert_class=None, bert_args={},
                 scan_decoder_class=None, scan_decoder_args={},
                 task_configs=[],
                 vocab_args={}, loss_weighting=None,
                 optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 pretrained_configs=[],
                 cuda=True, devices=[0]):
        """
        """
        super().__init__(optim_class, optim_args, scheduler_class, scheduler_args,
                         pretrained_configs, cuda, devices)

        self.encodes_scans = scan_encoder_class is not None
        if self.encodes_scans:
            self.scan_encoder = getattr(modules, scan_encoder_class)(**scan_encoder_args)
            self.scan_encoder = nn.DataParallel(self.scan_encoder,
                                                device_ids=self.devices)

        if bert_class == "BertModelPreTrained":
            self.bert = BertModel.from_pretrained(**bert_args)
        elif bert_class == "BertForPretraining":
            self.bert = BertForPreTraining.from_pretrained(**bert_args)
        elif bert_class == "BertModel":
            bert_args["config"] = BertConfig.from_dict(bert_args["config"])
            self.bert = BertModel(**bert_args)
        else:
            self.bert = getattr(modules, bert_class)(**bert_args)
        self.bert = nn.DataParallel(self.bert, device_ids=self.devices)

        self.decodes_scans = scan_decoder_class is not None
        if self.decodes_scans:
            self.scan_decoder = getattr(modules, scan_decoder_class)(**scan_decoder_args)

        self.task_heads = {}
        self.task_inputs = {}
        for task_head_config in task_configs:
            task = task_head_config["task"]
            head_class = getattr(modules, task_head_config["class"])
            args = task_head_config["args"]
            self.task_inputs[task] = (task_head_config["inputs"]
                                      if "inputs" in task_head_config else "pool")

            if "config" in args:
                # bert task heads take config object for parameters, must convert from dict
                config = args["config"]
                args["config"] = namedtuple("Config", config.keys())(*config.values())

            if head_class is BertOnlyMLMHead:
                embs = self.bert.module.embeddings.word_embeddings.weight
                self.task_heads[task] = head_class(bert_model_embedding_weights=embs,
                                                   **args)
            else:
                self.task_heads[task] = head_class(**args)

        self.task_heads = torch.nn.ModuleDict(self.task_heads)

        self.vocab = WordPieceVocab(**vocab_args)

        self._build_loss(loss_weighting)

        self._post_init()

    def show_attention(self, inputs):
        """
        Must be initialized with BertConfig where config.output_attentions = True
        """
        # turn on output attentions
        report_inputs = inputs["report"]
        report_input_ids, attention_mask = self.vocab.to_input_tensor(report_inputs,
                                                                      device=self.device)

        outputs = self.bert(input_ids=report_input_ids,
                            attention_mask=attention_mask)

        return outputs


    def forward(self, inputs, targets):
        """
        Args:
            inputs  (torch.Tensor) a (batch_size, ...) shaped input tensor
            targets     (list) a list of task targets from t0, t1...tn. The last element
                        should be a list(list(str)) representing the target report.

        Return:
            outputs (list) of
        """
        if self.encodes_scans:
            scan_inputs = inputs["scan"]
            scan_encodings = self.scan_encoder(scan_inputs)
            # TODO: 3d positional encodings
            scan_encodings = scan_encodings.view(scan_encodings.shape[0], scan_encodings.shape[1], -1)
            scan_encodings = scan_encodings.permute(0,2, 1)

        report_inputs = inputs["report"]
        report_input_ids, attention_mask = self.vocab.to_input_tensor(report_inputs,
                                                                      device=self.device)

        bert_seq, pool = self.bert(input_ids=report_input_ids,
                                   attention_mask=attention_mask,
                                   output_all_encoded_layers=False)

        if self.decodes_scans:
            scan_seq, pool = self.scan_decoder(scan_encodings=scan_encodings,
                                          report_encodings=bert_seq,
                                          attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
            # residual
            seq = bert_seq + scan_seq
        else:
            seq = bert_seq

        task_outputs = {}
        for task, task_head in self.task_heads.items():
            if self.task_inputs[task] == "seq":
                task_outputs[task] = task_head(seq)
            elif self.task_inputs[task] == "pool":
                task_outputs[task] = task_head(pool)
            else:
                raise ValueError("Input type not recognized.")

        return task_outputs

    def predict(self, inputs, probabilities=True):
        """
        """
        task_outputs = self.forward(inputs, None)
        task_predictions = {task: nn.functional.softmax(output, dim=-1)
                            for task, output in task_outputs.items()}

        if not probabilities:
            task_predictions = {task: torch.argmax(output, dim=-1, keepdim=True)
                                for task, output in task_predictions.items()}

        return task_predictions

    def loss(self, outputs, targets):
        """
        """
        total_loss = 0
        for task, output in outputs.items():
            curr_loss = self.loss_fn(output.view(-1, output.shape[-1]),
                                     targets[task].view(-1))
            total_loss += curr_loss

        return total_loss

    def _build_loss(self, loss_weighting=None):
        """
        """
        if loss_weighting is None:
            class_weights = None

        elif loss_weighting == "log":
            class_weights = torch.ones(len(self.vocab))
            for token, freq in self.vocab.token_to_freq.items():
                if freq == 0:
                    continue
                class_weights[self.vocab.token_to_idx[token]] = 1 / np.log(freq + 1)
        else:
            raise ValueError("Loss weighting scheme not recognized.")

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def _log_predictions(self, inputs, targets, predictions, info=None):
        """
        """
        predictions = {task: torch.argmax(output, dim=-1)
                       for task, output in predictions.items()}

        if "scan_mlm" in predictions:
            logging.info(f"Exam ID: {info[0]['exam_id']}")
            mlm_preds, mlm_targets = predictions["scan_mlm"], targets["scan_mlm"]
            mlm_preds = mlm_preds[mlm_targets != -1]
            mlm_targets = mlm_targets[mlm_targets != -1]

            # handle batch size of 1
            if len(mlm_targets.shape) < 2:
                mlm_preds = mlm_preds.unsqueeze(0)
                mlm_targets = mlm_targets.unsqueeze(0)

            if mlm_preds.shape[-1] != 0:
                mlm_preds = self.vocab.from_output_tensor(mlm_preds)
                mlm_targets = self.vocab.from_output_tensor(mlm_targets)

                for curr_inputs, target, pred in zip(inputs["report"], mlm_targets, mlm_preds):
                    logging.info(f"Inputs: {curr_inputs}")
                    logging.info(f"Targets: {' --- '.join(target)}")
                    logging.info(f"Predict: {' --- '.join(pred)}")

        if "scan_match" in predictions:
            logging.info(f"Inputs: {inputs['report'][0]}")
            logging.info(f"Matched:   {targets['scan_match']}")
            logging.info(f"Predicted: {predictions['scan_match']}")


