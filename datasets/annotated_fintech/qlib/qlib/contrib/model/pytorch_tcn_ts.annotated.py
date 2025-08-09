# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
# ✅ Best Practice: Use of relative imports for internal modules
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for internal modules
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for internal modules
import torch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for internal modules
# 🧠 ML Signal: Class definition for a machine learning model, useful for identifying model patterns
import torch.optim as optim
# ✅ Best Practice: Use of relative imports for internal modules
from torch.utils.data import DataLoader

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .tcn import TemporalConvNet


class TCN(Model):
    """TCN Model

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
        n_chans=128,
        kernel_size=5,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        # ✅ Best Practice: Use of a logger for information and debugging
        lr=0.001,
        metric="",
        batch_size=2000,
        # 🧠 ML Signal: Initialization of model hyperparameters
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("TCN")
        self.logger.info("TCN pytorch version...")
        # 🧠 ML Signal: Use of optimizer and loss function

        # set hyper-parameters.
        self.d_feat = d_feat
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        self.n_chans = n_chans
        self.kernel_size = kernel_size
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
            "TCN parameters setting:"
            "\nd_feat : {}"
            "\nn_chans : {}"
            "\nkernel_size : {}"
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
                n_chans,
                kernel_size,
                num_layers,
                dropout,
                n_epochs,
                # 🧠 ML Signal: Setting random seed for reproducibility
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                # 🧠 ML Signal: Model architecture definition
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # 🧠 ML Signal: Logging model size
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # ⚠️ SAST Risk (Low): Use of hardcoded strings for optimizer selection

        self.TCN_model = TCNModel(
            # 🧠 ML Signal: Checking if a GPU is used for computation
            num_input=self.d_feat,
            output_size=1,
            # ⚠️ SAST Risk (Low): Potential for incorrect device comparison if `self.device` is not properly initialized
            # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
            num_channels=[self.n_chans] * self.num_layers,
            # ✅ Best Practice: Use `torch.cuda.is_available()` for a more reliable GPU check
            kernel_size=self.kernel_size,
            # 🧠 ML Signal: Use of mean squared error (MSE) indicates a regression task
            dropout=self.dropout,
        # 🧠 ML Signal: Custom loss function implementation
        # 🧠 ML Signal: Model training state
        )
        # ⚠️ SAST Risk (Low): Ensure 'torch' is imported to avoid runtime errors
        self.logger.info("model:\n{:}".format(self.TCN_model))
        # 🧠 ML Signal: Model deployment to device
        # 🧠 ML Signal: Handling missing values in labels
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.TCN_model)))

        # 🧠 ML Signal: Conditional logic based on loss type
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.TCN_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Use of mean squared error for loss calculation
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.TCN_model.parameters(), lr=self.lr)
        # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
        # 🧠 ML Signal: Conditional logic based on metric type
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # ⚠️ SAST Risk (Low): Potential for negative loss values if not handled elsewhere

        self.fitted = False
        # 🧠 ML Signal: Iterating over data_loader indicates a training loop
        # ⚠️ SAST Risk (Low): Use of string interpolation in exception message
        self.TCN_model.to(self.device)

    # ✅ Best Practice: Transposing data for correct input shape
    @property
    def use_gpu(self):
        # 🧠 ML Signal: Splitting data into features and labels is common in ML training
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        # 🧠 ML Signal: Model prediction step
        loss = (pred - label) ** 2
        return torch.mean(loss)
    # 🧠 ML Signal: Calculating loss for model training

    def loss_fn(self, pred, label):
        # 🧠 ML Signal: Optimizer step preparation
        # 🧠 ML Signal: Method for evaluating model performance on a dataset
        mask = ~torch.isnan(label)

        # 🧠 ML Signal: Backpropagation step
        # ✅ Best Practice: Initialize lists to store batch results
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # ⚠️ SAST Risk (Low): Clipping gradients to prevent exploding gradients

        raise ValueError("unknown loss `%s`" % self.loss)
    # 🧠 ML Signal: Optimizer step to update model weights
    # ⚠️ SAST Risk (Low): Potential for data shape mismatch during transpose

    def metric_fn(self, pred, label):
        # 🧠 ML Signal: Separating features and labels for model input
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        # 🧠 ML Signal: Model inference without gradient computation

        raise ValueError("unknown metric `%s`" % self.metric)
    # 🧠 ML Signal: Calculating loss for model evaluation
    # ✅ Best Practice: Convert loss to a scalar for storage

    def train_epoch(self, data_loader):
        self.TCN_model.train()

        # 🧠 ML Signal: Calculating performance metric for model evaluation
        for data in data_loader:
            data = torch.transpose(data, 1, 2)
            # ✅ Best Practice: Consider using a more explicit data structure for evals_result instead of a mutable default argument
            # ✅ Best Practice: Convert score to a scalar for storage
            feature = data[:, 0:-1, :].to(self.device)
            label = data[:, -1, -1].to(self.device)
            # ✅ Best Practice: Return average loss and score for the epoch

            pred = self.TCN_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.TCN_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.TCN_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            data = torch.transpose(data, 1, 2)
            feature = data[:, 0:-1, :].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.TCN_model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        # process nan brought by dataloader
        # ⚠️ SAST Risk (Low): Ensure save_path is validated to prevent path traversal vulnerabilities
        dl_train.config(fillna_type="ffill+bfill")
        # process nan brought by dataloader
        # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
        dl_valid.config(fillna_type="ffill+bfill")
        # ⚠️ SAST Risk (Low): Consider handling exceptions for torch.cuda.empty_cache() to prevent potential runtime errors

        train_loader = DataLoader(
            # 🧠 ML Signal: Usage of dataset preparation with specific column sets
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        # 🧠 ML Signal: Configuration of data handling with fillna_type
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        # 🧠 ML Signal: Usage of DataLoader with specific batch size and number of workers
        )

        # 🧠 ML Signal: Model evaluation mode set before prediction
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        # 🧠 ML Signal: Data slicing and device transfer for model input
        # 🧠 ML Signal: Custom model class definition for PyTorch
        best_score = -np.inf
        best_epoch = 0
        # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
        # ✅ Best Practice: Use of torch.no_grad() for inference to save memory
        evals_result["train"] = []
        evals_result["valid"] = []
        # 🧠 ML Signal: Storing input parameters as instance variables for later use
        # 🧠 ML Signal: Model prediction and conversion to numpy

        # train
        # 🧠 ML Signal: Initializing a TemporalConvNet, indicating use of temporal convolutional layers
        self.logger.info("training...")
        # 🧠 ML Signal: Concatenation of predictions and use of index from data loader
        # 🧠 ML Signal: Use of a forward method suggests this is a neural network model
        self.fitted = True
        # 🧠 ML Signal: Initializing a linear layer, common in neural network architectures

        # ✅ Best Practice: Squeeze is used to remove dimensions of size 1, which is common in output processing
        # 🧠 ML Signal: Use of a linear layer indicates a common pattern in neural networks
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.TCN_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.TCN_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.TCN_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.TCN_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()