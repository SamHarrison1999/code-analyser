# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function
import copy
from typing import Text, Union

import numpy as np

# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Best Practice: Using a logger instead of print statements for logging is a best practice.

from qlib.workflow import R

# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
# ✅ Best Practice: Utility functions like get_or_create_path help in managing file paths effectively.
from ...data.dataset import DatasetH

# ✅ Best Practice: Importing specific utility functions can improve code readability and maintainability.
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ...utils import get_or_create_path
from .pytorch_utils import count_parameters


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
        # 🧠 ML Signal: Logging initialization of a model
        lr=0.001,
        metric="",
        # 🧠 ML Signal: Logging model version
        batch_size=2000,
        early_stop=20,
        # 🧠 ML Signal: Storing model configuration parameters
        loss="mse",
        optimizer="adam",
        # 🧠 ML Signal: Storing model configuration parameters
        GPU=0,
        seed=None,
        # 🧠 ML Signal: Storing model configuration parameters
        **kwargs,
    ):
        # 🧠 ML Signal: Storing model configuration parameters
        # Set logger.
        self.logger = get_module_logger("GRU")
        # 🧠 ML Signal: Storing model configuration parameters
        self.logger.info("GRU pytorch version...")

        # 🧠 ML Signal: Storing model configuration parameters
        # set hyper-parameters.
        # 🧠 ML Signal: Storing model configuration parameters
        # ⚠️ SAST Risk (Low): Potential GPU index out of range
        # 🧠 ML Signal: Logging model configuration
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
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
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
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # ⚠️ SAST Risk (Low): Seed setting for reproducibility
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUModel(
            # 🧠 ML Signal: Initializing a GRU model
            # 🧠 ML Signal: Checking if a GPU is used for computation
            d_feat=self.d_feat,
            # ⚠️ SAST Risk (Low): Potential for incorrect device comparison if `self.device` is not properly initialized
            hidden_size=self.hidden_size,
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            num_layers=self.num_layers,
            # ✅ Best Practice: Use `torch.device` for device comparison to ensure consistency
            dropout=self.dropout,
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function, common in regression tasks
        )
        # 🧠 ML Signal: Custom loss function implementation
        self.logger.info("model:\n{:}".format(self.gru_model))
        # ⚠️ SAST Risk (Low): Assumes pred and label are compatible tensors; no input validation
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.gru_model))
        )
        # 🧠 ML Signal: Logging model structure
        # ✅ Best Practice: Use of torch.isnan to handle NaN values in labels

        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Logging model size
            # 🧠 ML Signal: Conditional logic based on loss type
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Use of mean squared error for loss calculation
            # 🧠 ML Signal: Choosing optimizer based on configuration
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid values
        else:
            # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )
        # 🧠 ML Signal: Conditional logic based on metric type

        self.fitted = False
        # 🧠 ML Signal: Use of .values to extract numpy arrays from pandas DataFrames
        # 🧠 ML Signal: Use of mask to filter predictions and labels
        self.gru_model.to(self.device)

    # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers

    # ⚠️ SAST Risk (Low): Potential for unhandled exception if metric is unknown
    # 🧠 ML Signal: Use of np.squeeze to adjust dimensions of the target array
    @property
    # 🧠 ML Signal: Tracking model fitting status
    def use_gpu(self):
        # 🧠 ML Signal: Setting the model to training mode
        return self.device != torch.device("cpu")

    # 🧠 ML Signal: Moving model to the specified device

    # 🧠 ML Signal: Shuffling indices for training data
    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        # 🧠 ML Signal: Conversion of numpy arrays to PyTorch tensors and moving to device

        if self.loss == "mse":
            # 🧠 ML Signal: Conversion of numpy arrays to PyTorch tensors and moving to device
            return self.mse(pred[mask], label[mask])

        # 🧠 ML Signal: Model prediction step
        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Loss calculation
    def metric_fn(self, pred, label):
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization.
        mask = torch.isfinite(label)
        # 🧠 ML Signal: Optimizer gradient reset

        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Backpropagation step
            return -self.loss_fn(pred[mask], label[mask])

        # ⚠️ SAST Risk (Low): Potential for exploding gradients if not properly clipped
        # 🧠 ML Signal: Iterating over data in batches is a common pattern in ML for efficiency.
        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Optimizer step to update model parameters
    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are sanitized to prevent unexpected data types.
        y_train_values = np.squeeze(y_train.values)

        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are sanitized to prevent unexpected data types.
        self.gru_model.train()

        # ✅ Best Practice: Use torch.no_grad() to prevent tracking of gradients during evaluation.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        # 🧠 ML Signal: Using a loss function to evaluate model performance is a common ML pattern.
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            # ✅ Best Practice: Return the mean of losses and scores for a more stable evaluation metric.

            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.gru_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # prepare training and validation data
        dfs = {
            k: dataset.prepare(
                k,
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            for k in ["train", "valid"]
            if k in dataset.segments
        }
        df_train, df_valid = dfs.get("train", pd.DataFrame()), dfs.get(
            "valid", pd.DataFrame()
        )

        # check if training data is empty
        if df_train.empty:
            raise ValueError(
                "Empty training data from dataset, please check your dataset config."
            )

        # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
        df_train = df_train.dropna()
        x_train, y_train = df_train["feature"], df_train["label"]

        # 🧠 ML Signal: Usage of dataset preparation for prediction
        # check if validation data is provided
        if not df_valid.empty:
            df_valid = df_valid.dropna()
            # 🧠 ML Signal: Model evaluation mode set before prediction
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        else:
            x_valid, y_valid = None, None

        # 🧠 ML Signal: Iterating over data in batches
        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        # 🧠 ML Signal: Conversion of numpy array to torch tensor for model input

        # 🧠 ML Signal: Custom model class definition, useful for model architecture analysis
        # train
        # ✅ Best Practice: Using torch.no_grad() to prevent gradient computation
        self.logger.info("training...")
        # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
        self.fitted = True
        # 🧠 ML Signal: Returning predictions as a pandas Series
        # 🧠 ML Signal: Use of GRU indicates a sequence modeling task, common in time-series or NLP
        # 🧠 ML Signal: Model prediction and conversion back to numpy

        best_param = copy.deepcopy(self.gru_model.state_dict())
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)
            # 🧠 ML Signal: Linear layer suggests a regression or binary classification task

            # 🧠 ML Signal: Reshaping input tensor for RNN processing
            # evaluate on validation data if provided
            # ✅ Best Practice: Storing d_feat as an instance variable for potential future use
            if x_valid is not None and y_valid is not None:
                # 🧠 ML Signal: Permuting dimensions to match RNN input requirements
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                # 🧠 ML Signal: Applying fully connected layer to RNN output
                # 🧠 ML Signal: Passing data through RNN layer
                # ✅ Best Practice: Squeezing the output to remove unnecessary dimensions
                self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.gru_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("early stop")
                        break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.gru_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        # Logging
        rec = R.get_recorder()
        for k, v_l in evals_result.items():
            for i, v in enumerate(v_l):
                rec.log_metrics(step=i, **{k: v})

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.gru_model.eval()
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
                pred = self.gru_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


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
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
