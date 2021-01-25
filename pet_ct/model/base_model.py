"""
"""
import logging
import re

import torch
from torch import nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm

import pet_ct.model.losses as losses
from pet_ct.analysis.metrics import Metrics
from pet_ct.util.util import place_on_gpu, log_cuda_memory, log_predictions


class BaseModel(nn.Module):
    """
    """

    def __init__(
        self,
        optim_class="Adam",
        optim_args={},
        scheduler_class=None,
        scheduler_args={},
        pretrained_configs=[],
        cuda=True,
        devices=[0],
    ):
        """
        args:
            device (int or list(int)) A device or a list of devices
        """
        super().__init__()
        self.cuda = cuda
        self.device = devices[0]
        self.devices = devices
        self.pretrained_configs = pretrained_configs
        self.optim_class = optim_class
        self.optim_args = optim_args
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args

    def _post_init(self):
        """
        WARNING: must be called at the end of subclass init
        """
        self._build_optimizer(
            self.optim_class, self.optim_args, self.scheduler_class, self.scheduler_args
        )

        # load weights after building model, the experiment should handle loading current model
        for pretrained_config in self.pretrained_configs:
            self.load_weights(device=self.device, **pretrained_config)

        self.to(self.device)

    def _build_optimizer(
        self, optim_class="Adam", optim_args={}, scheduler_class=None, scheduler_args={}
    ):
        """
        """
        # load optimizer
        if "params" in optim_args:
            for params_dict in optim_args["params"]:
                params_dict["params"] = self._modules[
                    params_dict["params"]
                ].parameters()
        elif "modules" in optim_args:
            params = []
            for module in optim_args["modules"]:
                params.extend(self._modules[module].parameters())
            logging.info(f"Training params: {params}")
            optim_args["params"] = params
            del optim_args["modules"]
        else:
            optim_args["params"] = self.parameters()

        self.optimizer = getattr(optims, optim_class)(**optim_args)
        # load scheduler
        if scheduler_class is not None:
            self.scheduler = getattr(schedulers, scheduler_class)(
                self.optimizer, **scheduler_args
            )
        else:
            self.scheduler = None

    def predict_many(self, dataloader):
        """
        """
        logging.info("Prediction")
        self.eval()

        # move to cuda
        if self.cuda:
            self._to_gpu()

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, labels, info) in enumerate(dataloader):
                # move to GPU if available
                if self.cuda:
                    inputs = place_on_gpu(inputs, self.device)

                # forward pass
                predictions = self.predict(inputs)

                # compute average loss and update the progress bar
                t.update()

                yield inputs, labels, predictions, info

    def score(self, dataloader, metric_configs=[], log_predictions=False):
        """
        """
        logging.info("Validation")
        self.eval()

        # move to cuda
        if self.cuda:
            self._to_gpu()

        metrics = Metrics(metric_configs)
        avg_loss = 0

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, targets, info) in enumerate(dataloader):
                # move to GPU if available
                if self.cuda:
                    inputs, targets = place_on_gpu([inputs, targets], self.device)

                # forward pass
                predictions = self.predict(inputs)
                if log_predictions:
                    self._log_predictions(
                        inputs=inputs,
                        targets=targets,
                        predictions=predictions,
                        info=info,
                    )

                labels = self._get_labels(targets)
                metrics.add(predictions, labels, info)

                # compute average loss and update the progress bar
                t.update()

        metrics.compute()
        return metrics

    def train_model(
        self,
        dataloader,
        num_epochs=20,
        metric_configs=[],
        summary_period=1,
        writer=None,
    ):
        """
        Main training function.

        Trains the model, then collects metrics on the validation set. Saves
        weights on every epoch, denoting the best iteration by some specified
        metric.
        """
        logging.info(f"Starting training for {num_epochs} epoch(s)")

        # move to cuda
        if self.cuda:
            self._to_gpu()

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1} of {num_epochs}")

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                learning_rate = self.scheduler.get_lr()[0]
                logging.info(f"- Current learning rate: {learning_rate}")

            train_metrics = self._train_epoch(
                dataloader, metric_configs, summary_period, writer
            )
            yield train_metrics

    def _train_epoch(
        self,
        dataloader,
        metric_configs=[],
        summary_period=1,
        writer=None,
        log_predictions=True,
    ):
        """ Train the model for one epoch
        Args:
            train_data  (DataLoader)
        """
        logging.info("Training")

        self.train()

        metrics = Metrics(metric_configs)

        avg_loss = 0

        with tqdm(total=len(dataloader)) as t:
            for i, (inputs, targets, info) in enumerate(dataloader):
                if self.cuda:
                    inputs, targets = place_on_gpu([inputs, targets], self.device)
                # forward pass
                outputs = self.forward(inputs, targets)

                # loss for dynamic dataloader
                if hasattr(dataloader, "get_loss_weights"):
                    loss = self.loss(
                        outputs, targets, dataloader.get_loss_weights(targets)
                    )
                else:
                    loss = self.loss(outputs, targets)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.cpu().detach().numpy()
                # compute metrics periodically:
                if i % summary_period == 0:
                    predictions = self.predict(inputs)

                    if log_predictions:
                        self._log_predictions(inputs, targets, predictions, info)

                    labels = self._get_labels(targets)
                    metrics.add(predictions, labels, info, {"loss": loss})
                    del predictions

                # update dynamic dataloader
                if hasattr(dataloader, "update_batch"):
                    dataloader.update_batch([i["idx"] for i in info])

                # compute average loss and update progress bar
                avg_loss = ((avg_loss * i) + loss) / (i + 1)
                if writer is not None:
                    writer.add_scalar(tag="loss", scalar_value=loss)
                t.set_postfix(loss="{:05.3f}".format(float(avg_loss)))
                t.update()

                del loss, outputs, inputs, targets, labels

        metrics.compute()
        return metrics

    def forward(self, inputs, targets):
        """
        Forward Pass

        Args:
            x   (torch.Tensor) a (batch_size, ...) shaped input tensor

        IF SINGLE_TASK:
        Returns:
            output  (torch.Tensor) a (batch_size, ...)
        ELIF MULTI_TASK:
        Returns:
            outputs  (list(torch.Tensor)) list of outputs for each task
        """
        raise NotImplementedError

    def loss(self, outputs, targets):
        """
        """
        return self.loss_fn(outputs, targets)

    def predict(self, inputs):
        """
        """
        raise NotImplementedError

    def _get_labels(self, targets):
        """Optional target processing (primarily for metrics)."""
        return targets

    def save_weights(self, destination):
        """
        args:
            destination (str)   path where to save weights
        """
        torch.save(self.state_dict(), destination)

    def load_weights(
        self, src_path, inclusion_res=None, substitution_res=None, device=None
    ):
        """
        args:
            src_path (str) path to the weights file
            inclusion_res (list(str) or str) list of regex patterns or one regex pattern.
                    If not None, only loads weights that match at least one of the regex patterns.
            substitution_res (list(tuple(str, str))) list of tuples like
                    (regex_pattern, replacement). re.sub is called on each key in the dict
        """
        src_state_dict = torch.load(src_path, map_location=torch.device(device))

        if type(inclusion_res) is str:
            inclusion_res = [inclusion_res]
        if inclusion_res is not None:
            src_state_dict = {
                key: val
                for key, val in src_state_dict.items()
                if re.match("|".join(inclusion_res), key) is not None
            }

        if substitution_res is not None:
            for pattern, repl in substitution_res:
                src_state_dict = {
                    re.sub(pattern, repl, key): val
                    for key, val in src_state_dict.items()
                }

        self.load_state_dict(src_state_dict, strict=False)
        n_loaded_params = len(
            set(self.state_dict().keys()) & set(src_state_dict.keys())
        )
        n_tot_params = len(src_state_dict.keys())
        if n_loaded_params < n_tot_params:
            logging.info(
                "Could not load these parameters due to name mismatch: "
                + f"{set(src_state_dict.keys()) - set(self.state_dict().keys())}"
            )
        logging.info(
            f"Loaded {n_loaded_params}/{n_tot_params} pretrained parameters"
            + f"from {src_path} matching '{inclusion_res}'."
        )

    def save(self, destination, **kwargs):
        """ Serialize and save a model.
        Args:
            destination (str)   path indicating saving destination
        """
        pass
        # with open(destination, "wb") as f:
        #    torch.save(self, f, **kwargs)

    @staticmethod
    def load(source, **kwargs):
        """  Deserialize and load a model.
        Args:
            destination (str) path on which one has previously called BaseModel.save
        """
        with open(source, "rb") as f:
            model = torch.load(f, **kwargs)
            return model

    def _to_gpu(self):
        """ Moves the model to the gpu. Should be reimplemented by child model for
        data parallel.
        """
        if self.cuda:
            self.to(self.device)

    def _log_predictions(self, inputs, targets, predictions, info=None):
        """
        """
        pass
