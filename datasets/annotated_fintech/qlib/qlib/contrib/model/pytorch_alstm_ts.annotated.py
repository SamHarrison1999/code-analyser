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
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import torch.optim as optim
from torch.utils.data import DataLoader
# ✅ Best Practice: Use of relative imports for better modularity and maintainability

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


class ALSTM(Model):
    """ALSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        # ✅ Best Practice: Use a logger to track and debug the flow of the program.
        lr=0.001,
        metric="",
        batch_size=2000,
        # 🧠 ML Signal: Storing model hyperparameters for later use in training.
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("ALSTM")
        # 🧠 ML Signal: Normalizing optimizer input to lowercase for consistency.
        self.logger.info("ALSTM pytorch version...")

        # set hyper-parameters.
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available.
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
            "ALSTM parameters setting:"
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
                # ⚠️ SAST Risk (Low): Potential undefined attribute 'use_gpu'.
                num_layers,
                dropout,
                n_epochs,
                # 🧠 ML Signal: Setting random seed for reproducibility in ML experiments.
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                # 🧠 ML Signal: Initializing the ALSTM model with specified parameters.
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # 🧠 ML Signal: Logging model size, which can be useful for resource allocation.
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Checks if the computation is set to run on a GPU, indicating hardware usage preference
        # 🧠 ML Signal: Using Adam optimizer for training.
        self.ALSTM_model = ALSTMModel(
            d_feat=self.d_feat,
            # ✅ Best Practice: Using torch.device to compare ensures compatibility with PyTorch's device management
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            hidden_size=self.hidden_size,
            # 🧠 ML Signal: Using SGD optimizer for training.
            num_layers=self.num_layers,
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function
            dropout=self.dropout,
        # ⚠️ SAST Risk (Low): Ensure 'weight' is validated to prevent unexpected behavior
        # 🧠 ML Signal: Custom loss function implementation
        )
        # ⚠️ SAST Risk (Low): Raises an exception for unsupported optimizers.
        self.logger.info("model:\n{:}".format(self.ALSTM_model))
        # 🧠 ML Signal: Use of torch.mean for reducing tensor dimensions
        # 🧠 ML Signal: Handling missing values in labels
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ALSTM_model)))
        # 🧠 ML Signal: Tracking whether the model has been fitted.

        # 🧠 ML Signal: Default weight handling
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Moving model to the specified device (CPU/GPU).
            self.train_optimizer = optim.Adam(self.ALSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Conditional logic based on loss type
            self.train_optimizer = optim.SGD(self.ALSTM_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Check for finite values in label to avoid computation errors
        else:
            # 🧠 ML Signal: Use of mask for valid data points
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # ⚠️ SAST Risk (Low): Potential information disclosure through error message
        # ✅ Best Practice: Use mask to filter out non-finite values before loss computation
        self.fitted = False
        self.ALSTM_model.to(self.device)

    # ✅ Best Practice: Check for NaN values in label to avoid computation errors
    @property
    def use_gpu(self):
        # ✅ Best Practice: Initialize weight tensor with ones for consistent weighting
        return self.device != torch.device("cpu")

    # ✅ Best Practice: Use mask to filter out NaN values before MSE computation
    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        # ⚠️ SAST Risk (Low): Potential for format string vulnerability if `self.metric` is user-controlled
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        # ✅ Best Practice: Clipping gradients to prevent exploding gradients
        if weight is None:
            weight = torch.ones_like(label)

        # 🧠 ML Signal: Method for evaluating model performance on a dataset
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])
        # 🧠 ML Signal: Collecting scores and losses for evaluation

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        # 🧠 ML Signal: Data preprocessing step before model prediction
        mask = torch.isfinite(label)

        # 🧠 ML Signal: Extracting labels for evaluation
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        elif self.metric == "mse":
            # ⚠️ SAST Risk (Low): Ensure model is in evaluation mode to prevent gradient updates
            mask = ~torch.isnan(label)
            weight = torch.ones_like(label)
            # ⚠️ SAST Risk (Low): Ensure loss function is correctly applied with weights
            # 🧠 ML Signal: Tracking loss values for analysis
            return -self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Calculating performance metric
    def train_epoch(self, data_loader):
        self.ALSTM_model.train()

        # 🧠 ML Signal: Tracking score values for analysis
        # 🧠 ML Signal: Preparing training and validation datasets
        for data, weight in data_loader:
            # ✅ Best Practice: Return average loss and score for better interpretability
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # ⚠️ SAST Risk (Low): Potential for unhandled exception if dataset is empty
            pred = self.ALSTM_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            # ✅ Best Practice: Consistent data preprocessing with fillna_type

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)
            # 🧠 ML Signal: Default weighting for training and validation datasets
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.ALSTM_model.eval()
        # 🧠 ML Signal: Custom reweighting of datasets

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            # ⚠️ SAST Risk (Low): Potential for unhandled exception with unsupported reweighter
            # 🧠 ML Signal: DataLoader setup for training
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.ALSTM_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                # 🧠 ML Signal: DataLoader setup for validation
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        # ✅ Best Practice: Ensure save_path is valid or created
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 🧠 ML Signal: Tracking evaluation results
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            # 🧠 ML Signal: Training for each epoch
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            # 🧠 ML Signal: Evaluation of training and validation datasets
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        # ⚠️ SAST Risk (Low): No check for dataset being None or invalid type
        # 🧠 ML Signal: Storing best model parameters
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            # 🧠 ML Signal: Usage of dataset preparation with specific segment and column set
            batch_size=self.batch_size,
            shuffle=False,
            # ✅ Best Practice: Configuring data handling with fillna_type for consistency
            num_workers=self.n_jobs,
            drop_last=True,
        # 🧠 ML Signal: Usage of DataLoader with specific batch size and number of workers
        )
        # 🧠 ML Signal: Loading best model parameters

        # 🧠 ML Signal: Model evaluation mode set before prediction
        save_path = get_or_create_path(save_path)
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if save_path is invalid

        stop_steps = 0
        train_loss = 0
        # 🧠 ML Signal: Custom model class definition for PyTorch
        # ✅ Best Practice: Clear GPU cache after training
        # ⚠️ SAST Risk (Low): Assumes data shape without validation
        best_score = -np.inf
        best_epoch = 0
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
        evals_result["train"] = []
        # 🧠 ML Signal: Model prediction without gradient computation
        evals_result["valid"] = []

        # train
        # ✅ Best Practice: Returning predictions as a pandas Series with index
        self.logger.info("training...")
        self.fitted = True

        # ✅ Best Practice: Encapsulation of model building in a separate method for clarity and reusability
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            # ⚠️ SAST Risk (Low): Catching broad exceptions can mask other issues
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            # ✅ Best Practice: Use of nn.Sequential for model layers improves readability
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # ✅ Best Practice: Naming modules improves model interpretability
            # 🧠 ML Signal: Use of RNN layer indicates sequence processing
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break
        # ✅ Best Practice: Use of nn.Sequential for attention network improves readability

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.ALSTM_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            # 🧠 ML Signal: Use of Dropout layer indicates regularization
            torch.cuda.empty_cache()

    # 🧠 ML Signal: Use of RNN layer indicates sequence processing, common in NLP tasks
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            # 🧠 ML Signal: Attention mechanism usage, common in advanced sequence models
            raise ValueError("model is not fitted yet!")

        # 🧠 ML Signal: Element-wise multiplication for attention application
        # 🧠 ML Signal: Use of Softmax layer indicates classification or attention mechanism
        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        # 🧠 ML Signal: Summing over a dimension, typical in reducing sequence data
        # 🧠 ML Signal: Concatenation of features, a common pattern in deep learning
        # ✅ Best Practice: Explicitly returning a specific slice of the output
        self.ALSTM_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.ALSTM_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out[..., 0]