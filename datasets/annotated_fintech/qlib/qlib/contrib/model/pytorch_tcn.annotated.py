# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Use of relative imports for better modularity and maintainability

import numpy as np
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
# ✅ Best Practice: Use of relative imports for better modularity and maintainability

import torch
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import torch.nn as nn
# 🧠 ML Signal: Definition of a class that inherits from Model, indicating a custom model implementation
import torch.optim as optim
# ✅ Best Practice: Use of relative imports for better modularity and maintainability

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from .tcn import TemporalConvNet


class TCN(Model):
    """TCN Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    n_chans: int
        number of channels
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
        num_layers=5,
        dropout=0.5,
        # 🧠 ML Signal: Logging initialization and parameters can be useful for ML model training and debugging
        n_epochs=200,
        lr=0.0001,
        metric="",
        # 🧠 ML Signal: Model hyperparameters are often used as features in ML model training
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("TCN")
        self.logger.info("TCN pytorch version...")
        # ✅ Best Practice: Normalize optimizer input to lowercase for consistency

        # set hyper-parameters.
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # 🧠 ML Signal: Logging model parameters can be useful for ML model training and debugging
        self.d_feat = d_feat
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
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                n_chans,
                kernel_size,
                num_layers,
                dropout,
                # 🧠 ML Signal: Setting random seed for reproducibility is a common practice in ML
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                # 🧠 ML Signal: Model architecture details are important for ML model training
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            # 🧠 ML Signal: Logging model size can be useful for resource management in ML
            torch.manual_seed(self.seed)
        # ✅ Best Practice: Use of lower() ensures case-insensitive comparison

        self.tcn_model = TCNModel(
            # 🧠 ML Signal: Checking if a GPU is used for computation
            num_input=self.d_feat,
            output_size=1,
            # ✅ Best Practice: Using torch.device to handle device types
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in ML models
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            # ✅ Best Practice: Use of descriptive variable names for clarity
            dropout=self.dropout,
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
        )
        # ✅ Best Practice: Use of torch.isnan to handle NaN values in tensors
        # ⚠️ SAST Risk (Low): Assumes pred and label are tensors; no input validation
        self.logger.info("model:\n{:}".format(self.tcn_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.tcn_model)))
        # 🧠 ML Signal: Conditional logic based on loss type
        # 🧠 ML Signal: Moving model to device (CPU/GPU) is a common pattern in ML

        if optimizer.lower() == "adam":
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            # 🧠 ML Signal: Use of mask to filter out NaN values before computation
            self.train_optimizer = optim.Adam(self.tcn_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
            # 🧠 ML Signal: Use of torch.isfinite indicates handling of numerical stability in ML models
            self.train_optimizer = optim.SGD(self.tcn_model.parameters(), lr=self.lr)
        else:
            # 🧠 ML Signal: Conditional logic based on metric type suggests dynamic behavior in ML evaluation
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # 🧠 ML Signal: Use of loss function indicates model evaluation or training process
        self.fitted = False
        self.tcn_model.to(self.device)
    # ⚠️ SAST Risk (Low): Potential for unhandled exception if metric is unknown

    @property
    def use_gpu(self):
        # 🧠 ML Signal: Shuffling data is a common practice in training ML models to ensure randomness.
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly.
        return torch.mean(loss)

    # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly.
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        # ✅ Best Practice: Gradient clipping is used to prevent exploding gradients.
        raise ValueError("unknown loss `%s`" % self.loss)

    # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization layers.
    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            # ✅ Best Practice: Using np.arange for index generation is efficient and clear.
            return -self.loss_fn(pred[mask], label[mask])

        # 🧠 ML Signal: Iterating over data in batches is a common pattern in ML for efficiency.
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are properly validated to prevent unexpected data types.
        y_train_values = np.squeeze(y_train.values)

        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are properly validated to prevent unexpected data types.
        self.tcn_model.train()

        # ✅ Best Practice: Use torch.no_grad() to prevent tracking history in evaluation mode, saving memory.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        # 🧠 ML Signal: Using a loss function to evaluate model predictions is a standard ML practice.

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            # 🧠 ML Signal: Using a metric function to evaluate model predictions is a standard ML practice.

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # ✅ Best Practice: Returning the mean of losses and scores provides a summary metric for the epoch.
            # ✅ Best Practice: Consider using a more explicit data structure for evals_result to avoid shared state issues.
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.tcn_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.tcn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.tcn_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.tcn_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        # ⚠️ SAST Risk (Low): Ensure save_path is validated or sanitized to prevent path traversal vulnerabilities.
        dataset: DatasetH,
        evals_result=dict(),
        # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
        save_path=None,
    # ⚠️ SAST Risk (Low): Ensure that GPU resources are properly managed to prevent memory leaks.
    ):
        df_train, df_valid, df_test = dataset.prepare(
            # 🧠 ML Signal: Usage of dataset preparation for prediction
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        # 🧠 ML Signal: Model evaluation mode set before prediction
        )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # ✅ Best Practice: Use of batch processing for predictions

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        # 🧠 ML Signal: Conversion of data to torch tensor and device allocation
        evals_result["valid"] = []
        # 🧠 ML Signal: Custom model class definition for PyTorch

        # ✅ Best Practice: Use of torch.no_grad() for inference to save memory
        # train
        # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
        self.logger.info("training...")
        # 🧠 ML Signal: Model prediction and conversion back to numpy
        self.fitted = True
        # 🧠 ML Signal: Storing input size as an instance variable, useful for model architecture

        for step in range(self.n_epochs):
            # 🧠 ML Signal: Returning predictions as a pandas Series
            # 🧠 ML Signal: Initializing a temporal convolutional network, indicating sequence processing
            self.logger.info("Epoch%d:", step)
            # 🧠 ML Signal: Reshaping input data for model processing
            self.logger.info("training...")
            # 🧠 ML Signal: Linear layer initialization, common in neural network architectures
            self.train_epoch(x_train, y_train)
            # 🧠 ML Signal: Passing data through a temporal convolutional network (TCN)
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Applying a linear transformation to the TCN output
            # 🧠 ML Signal: Squeezing the output to remove single-dimensional entries
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.tcn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.tcn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.tcn_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.tcn_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.num_input, -1)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()