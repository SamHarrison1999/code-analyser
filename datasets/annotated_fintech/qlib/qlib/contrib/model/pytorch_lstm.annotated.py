# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

import numpy as np
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

# 🧠 ML Signal: Definition of a class inheriting from Model, indicating a custom ML model implementation
import torch
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class LSTM(Model):
    """LSTM Model

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
        # 🧠 ML Signal: Logging initialization of the model
        n_epochs=200,
        lr=0.001,
        metric="",
        # 🧠 ML Signal: Storing model hyperparameters
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("LSTM")
        # ✅ Best Practice: Normalize optimizer input to lowercase
        self.logger.info("LSTM pytorch version...")

        # ⚠️ SAST Risk (Low): Potential GPU index out of range
        # 🧠 ML Signal: Logging model parameters
        # set hyper-parameters.
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
        self.seed = seed

        self.logger.info(
            "LSTM parameters setting:"
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
                # 🧠 ML Signal: Setting random seed for reproducibility
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                # 🧠 ML Signal: Initializing LSTM model with parameters
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        # ✅ Best Practice: Use of conditional logic for optimizer selection
        if self.seed is not None:
            np.random.seed(self.seed)
            # 🧠 ML Signal: Checks if the computation is set to run on a GPU, indicating hardware usage preference
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in ML models
        # ✅ Best Practice: Directly compares device to torch.device("cpu") for clarity
        self.lstm_model = LSTMModel(
            d_feat=self.d_feat,
            # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
            # 🧠 ML Signal: Calculation of squared error, a key step in MSE
            hidden_size=self.hidden_size,
            # 🧠 ML Signal: Custom loss function implementation
            num_layers=self.num_layers,
            # 🧠 ML Signal: Use of torch.mean, indicating integration with PyTorch for tensor operations
            dropout=self.dropout,
        # 🧠 ML Signal: Moving model to the specified device
        # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if `label` is not a tensor
        )
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Conditional logic based on loss type
            self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Use of mask for handling missing values
            self.train_optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values
        else:
            # ⚠️ SAST Risk (Low): Error message may expose internal state
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # 🧠 ML Signal: Conditional logic based on self.metric value

        self.fitted = False
        # 🧠 ML Signal: Use of mask to filter predictions and labels
        self.lstm_model.to(self.device)
    # ⚠️ SAST Risk (Low): Potential for negative loss values if not handled properly

    # 🧠 ML Signal: Indicates usage of a model training loop
    @property
    # ⚠️ SAST Risk (Low): Use of string interpolation with user-controlled input
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)
    # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly

    def loss_fn(self, pred, label):
        # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly
        mask = ~torch.isnan(label)

        # 🧠 ML Signal: Model prediction step
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # 🧠 ML Signal: Loss calculation step

        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Backpropagation step
    def metric_fn(self, pred, label):
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization layers.
        mask = torch.isfinite(label)
        # ✅ Best Practice: Gradient clipping to prevent exploding gradients

        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Optimizer step to update model parameters
            return -self.loss_fn(pred[mask], label[mask])

        # 🧠 ML Signal: Iterating over data in batches is a common pattern in ML model evaluation.
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are properly validated to prevent unexpected data types.
        y_train_values = np.squeeze(y_train.values)

        # ⚠️ SAST Risk (Low): Ensure that data_x and data_y are properly validated to prevent unexpected data types.
        self.lstm_model.train()

        # 🧠 ML Signal: Using a model to make predictions on a batch of features.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        # 🧠 ML Signal: Calculating loss between predictions and true labels.
        # 🧠 ML Signal: Calculating a metric score for model evaluation.

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # 🧠 ML Signal: Returning the mean loss and score as evaluation metrics.
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.lstm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        # ⚠️ SAST Risk (Low): Potential resource leak if GPU memory is not cleared properly
        evals_result=dict(),
        save_path=None,
    ):
        # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            # 🧠 ML Signal: Usage of dataset preparation for prediction
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            # 🧠 ML Signal: Model evaluation mode set before prediction
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # 🧠 ML Signal: Iterating over data in batches

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        # ⚠️ SAST Risk (Low): Assumes 'self.device' is correctly set for torch device
        evals_result["valid"] = []
        # 🧠 ML Signal: Custom model class definition for PyTorch

        # 🧠 ML Signal: Use of torch.no_grad() for inference
        # ✅ Best Practice: Use of default values for function parameters improves flexibility and usability.
        # train
        self.logger.info("training...")
        # 🧠 ML Signal: Returning predictions as a pandas Series
        # 🧠 ML Signal: Use of LSTM indicates a sequence modeling task, common in time-series or NLP.
        # 🧠 ML Signal: Detaching and moving tensor to CPU for numpy conversion
        # ✅ Best Practice: Proper use of inheritance with super() to initialize the parent class.
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            # 🧠 ML Signal: d_feat as input_size suggests feature dimensionality for the model.
            # 🧠 ML Signal: hidden_size is a hyperparameter that affects model capacity and performance.
            train_loss, train_score = self.test_epoch(x_train, y_train)
            # 🧠 ML Signal: num_layers indicates the depth of the LSTM, affecting learning complexity.
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # 🧠 ML Signal: Reshaping input data for model processing
            # 🧠 ML Signal: batch_first=True is a common setting for batch processing in PyTorch.
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            # 🧠 ML Signal: Permuting tensor dimensions for RNN input
            # 🧠 ML Signal: dropout is used to prevent overfitting, a common practice in training neural networks.

            # 🧠 ML Signal: Linear layer suggests a regression or binary classification task.
            # ✅ Best Practice: Storing d_feat as an instance variable for potential future use.
            # 🧠 ML Signal: Using RNN for sequence processing
            # 🧠 ML Signal: Applying fully connected layer to RNN output
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_model.eval()
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
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
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