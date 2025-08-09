# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.optim as optim
# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
from torch.utils.data import DataLoader
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


class GRU(Model):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
        lr=0.001,
        metric="",
        batch_size=2000,
        # 🧠 ML Signal: Model hyperparameters are set, which can be used to learn common configurations
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        # 🧠 ML Signal: Optimizer choice is a key decision in model training
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # 🧠 ML Signal: Logging detailed model parameters for traceability
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                # ⚠️ SAST Risk (Low): Seed setting for reproducibility, but should be used with caution in secure contexts
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                # 🧠 ML Signal: Model instantiation with specific architecture parameters
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # 🧠 ML Signal: Logging model size can be used to understand resource requirements
            np.random.seed(self.seed)
            # ⚠️ SAST Risk (Low): Use of different optimizers, potential for unsupported optimizers
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Checking for GPU usage is a common pattern in ML models to optimize performance
        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            # ⚠️ SAST Risk (Low): Potential for incorrect device comparison if `self.device` is not properly initialized
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function
            dropout=self.dropout,
        # ⚠️ SAST Risk (Low): Ensure 'weight' is validated to prevent unexpected behavior
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        # ⚠️ SAST Risk (Low): Ensure model is moved to the correct device
        # 🧠 ML Signal: Use of torch.mean for reducing loss
        # ✅ Best Practice: Use descriptive variable names for better readability
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GRU_model)))

        # ✅ Best Practice: Use is None for None checks
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Use torch.full_like for consistency and potential future flexibility
        elif optimizer.lower() == "gd":
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Conditional logic based on self.loss indicates model behavior
        else:
            # 🧠 ML Signal: Use of torch.isfinite indicates handling of numerical stability or invalid values
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # 🧠 ML Signal: Use of mask to handle NaN values in label

        # 🧠 ML Signal: Conditional logic based on metric type can indicate model evaluation or training phase
        self.fitted = False
        # ⚠️ SAST Risk (Low): Potential information disclosure through error message
        self.GRU_model.to(self.device)
    # 🧠 ML Signal: Use of loss function suggests model training or evaluation context

    # 🧠 ML Signal: Iterating over data_loader indicates a training loop
    @property
    # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if metric is unknown
    def use_gpu(self):
        # 🧠 ML Signal: Extracting features and labels from data
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        # 🧠 ML Signal: Model prediction step
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)
    # 🧠 ML Signal: Loss calculation with custom loss function

    def loss_fn(self, pred, label, weight=None):
        # 🧠 ML Signal: Optimizer step preparation
        mask = ~torch.isnan(label)
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization.

        # 🧠 ML Signal: Backpropagation step
        if weight is None:
            weight = torch.ones_like(label)
        # ⚠️ SAST Risk (Low): Gradient clipping to prevent exploding gradients

        if self.loss == "mse":
            # 🧠 ML Signal: Optimizer step to update model parameters
            # ✅ Best Practice: Use descriptive variable names for better readability.
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)
    # ✅ Best Practice: Use torch.no_grad() to prevent tracking history in evaluation mode.

    def metric_fn(self, pred, label):
        # 🧠 ML Signal: Model prediction step, useful for understanding model usage patterns.
        mask = torch.isfinite(label)

        # 🧠 ML Signal: Custom loss function usage, useful for understanding model evaluation.
        if self.metric in ("", "loss"):
            # ✅ Best Practice: Use .item() to convert tensors to Python scalars for logging.
            # 🧠 ML Signal: Custom metric function usage, useful for understanding model evaluation.
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        # ✅ Best Practice: Use .item() to convert tensors to Python scalars for logging.
        self.GRU_model.train()

        # ✅ Best Practice: Use numpy for efficient computation of mean values.
        # ✅ Best Practice: Consider using a more descriptive variable name than 'dl_train' for clarity.
        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # ✅ Best Practice: Consider using a more descriptive variable name than 'dl_valid' for clarity.
            label = data[:, -1, -1].to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            # 🧠 ML Signal: Default weight initialization for training and validation datasets.
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        # 🧠 ML Signal: Custom reweighting logic for datasets.
        self.GRU_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            # ✅ Best Practice: Consider using a more descriptive variable name than 'train_loader' for clarity.
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                # ✅ Best Practice: Consider using a more descriptive variable name than 'valid_loader' for clarity.
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        # ⚠️ SAST Risk (Low): Ensure 'save_path' is validated to prevent path traversal vulnerabilities.
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            # 🧠 ML Signal: Indicates the model has been fitted.
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            # 🧠 ML Signal: Captures the best model parameters during training.
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        # ⚠️ SAST Risk (Low): Method assumes 'self.fitted' is a boolean attribute; ensure it's properly initialized.
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            # 🧠 ML Signal: Usage of 'prepare' method indicates a preprocessing step for ML datasets.
            batch_size=self.batch_size,
            shuffle=False,
            # 🧠 ML Signal: Configuration of data handling, such as filling missing values, is a common ML preprocessing step.
            # ⚠️ SAST Risk (Low): Ensure 'save_path' is validated to prevent path traversal vulnerabilities.
            num_workers=self.n_jobs,
            drop_last=True,
        # 🧠 ML Signal: DataLoader is used for batching data, a common pattern in ML for handling large datasets.
        )
        # ⚠️ SAST Risk (Low): Ensure proper GPU resource management to prevent memory leaks.

        # 🧠 ML Signal: Setting model to evaluation mode is a common practice in ML to disable dropout and batchnorm layers.
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        # 🧠 ML Signal: Data is moved to a specific device (e.g., GPU) for computation, a common ML practice.
        # 🧠 ML Signal: Definition of a custom neural network model class
        best_score = -np.inf
        best_epoch = 0
        # 🧠 ML Signal: Disabling gradient calculation for inference is a common ML practice to save memory.
        # ✅ Best Practice: Use of default values for function parameters improves usability and flexibility.
        evals_result["train"] = []
        # 🧠 ML Signal: Model prediction and conversion to numpy array for further processing.
        # 🧠 ML Signal: Use of GRU indicates a sequence modeling task, common in time-series or NLP.
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            # 🧠 ML Signal: Concatenating predictions and aligning with dataset index is a common pattern in ML for result interpretation.
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            # 🧠 ML Signal: Linear layer following RNN suggests a regression or binary classification task.
            self.train_epoch(train_loader)
            # 🧠 ML Signal: Use of RNN layer indicates sequence processing, common in time-series or NLP tasks
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Accessing the last output of RNN suggests interest in final state, typical in classification tasks
            # ✅ Best Practice: Squeezing the output is a common practice to ensure correct dimensionality for loss functions
            # ✅ Best Practice: Storing input feature size as an instance variable can improve code readability and maintainability.
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()