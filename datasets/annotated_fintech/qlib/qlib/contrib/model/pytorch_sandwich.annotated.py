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
# 🧠 ML Signal: Custom model class definition for PyTorch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from .pytorch_krnn import CNNKRNNEncoder


class SandwichModel(nn.Module):
    def __init__(
        self,
        fea_dim,
        cnn_dim_1,
        cnn_dim_2,
        cnn_kernel_size,
        rnn_dim_1,
        rnn_dim_2,
        rnn_dups,
        rnn_layers,
        dropout,
        device,
        **params,
    ):
        """Build a Sandwich model

        Parameters
        ----------
        fea_dim : int
            The feature dimension
        cnn_dim_1 : int
            The hidden dimension of the first CNN
        cnn_dim_2 : int
            The hidden dimension of the second CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_dim_1 : int
            The hidden dimension of the first KRNN
        rnn_dim_2 : int
            The hidden dimension of the second KRNN
        rnn_dups : int
            The number of parallel duplicates
        rnn_layers: int
            The number of RNN layers
        """
        super().__init__()
        # 🧠 ML Signal: Instantiation of a second custom encoder model, useful for model architecture analysis

        self.first_encoder = CNNKRNNEncoder(
            cnn_input_dim=fea_dim,
            cnn_output_dim=cnn_dim_1,
            cnn_kernel_size=cnn_kernel_size,
            rnn_output_dim=rnn_dim_1,
            rnn_dup_num=rnn_dups,
            rnn_layers=rnn_layers,
            dropout=dropout,
            device=device,
        )

        self.second_encoder = CNNKRNNEncoder(
            # 🧠 ML Signal: Use of a linear layer, common in neural network architectures
            # 🧠 ML Signal: Use of encoder layers suggests a deep learning model, likely for sequence data.
            cnn_input_dim=rnn_dim_1,
            cnn_output_dim=cnn_dim_2,
            # 🧠 ML Signal: Storing device information, relevant for model deployment and training
            # 🧠 ML Signal: Chaining multiple encoders indicates a complex model architecture.
            cnn_kernel_size=cnn_kernel_size,
            rnn_output_dim=rnn_dim_2,
            # ✅ Best Practice: Use of slicing to access the last element in a sequence, common in sequence models.
            # 🧠 ML Signal: Defines a class for a machine learning model, which can be used to train ML models on class structure and design patterns
            rnn_dup_num=rnn_dups,
            # ⚠️ SAST Risk (Low): Potential risk if encode is not guaranteed to have at least one element.
            # ✅ Best Practice: Returning the output tensor directly is a common practice in model forward methods.
            rnn_layers=rnn_layers,
            dropout=dropout,
            device=device,
        )

        self.out_fc = nn.Linear(rnn_dim_2, 1)
        self.device = device

    def forward(self, x):
        # x: [batch_size, node_num, seq_len, input_dim]
        encode = self.first_encoder(x)
        encode = self.second_encoder(encode)
        out = self.out_fc(encode[:, -1, :]).squeeze().to(self.device)

        return out


class Sandwich(Model):
    """Sandwich Model

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
        # ✅ Best Practice: Use of a logger for information and debugging
        fea_dim=6,
        cnn_dim_1=64,
        cnn_dim_2=32,
        cnn_kernel_size=3,
        rnn_dim_1=16,
        rnn_dim_2=8,
        rnn_dups=3,
        rnn_layers=2,
        dropout=0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        # ✅ Best Practice: Normalize optimizer input to lowercase
        seed=None,
        **kwargs,
    ):
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # Set logger.
        self.logger = get_module_logger("Sandwich")
        self.logger.info("Sandwich pytorch version...")

        # set hyper-parameters.
        self.fea_dim = fea_dim
        self.cnn_dim_1 = cnn_dim_1
        self.cnn_dim_2 = cnn_dim_2
        self.cnn_kernel_size = cnn_kernel_size
        self.rnn_dim_1 = rnn_dim_1
        self.rnn_dim_2 = rnn_dim_2
        self.rnn_dups = rnn_dups
        self.rnn_layers = rnn_layers
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
            "Sandwich parameters setting:"
            "\nfea_dim : {}"
            "\ncnn_dim_1 : {}"
            "\ncnn_dim_2 : {}"
            "\ncnn_kernel_size : {}"
            "\nrnn_dim_1 : {}"
            "\nrnn_dim_2 : {}"
            "\nrnn_dups : {}"
            "\nrnn_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size: {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                # ✅ Best Practice: Set random seed for reproducibility
                fea_dim,
                cnn_dim_1,
                cnn_dim_2,
                cnn_kernel_size,
                rnn_dim_1,
                rnn_dim_2,
                rnn_dups,
                rnn_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                # ✅ Best Practice: Use of conditional logic to select optimizer
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )
        # 🧠 ML Signal: Checks if the computation is set to run on a GPU, which is a common pattern in ML for performance optimization

        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
        # ⚠️ SAST Risk (Low): Assumes 'self.device' is a valid torch.device object, which could lead to errors if not properly initialized
        if self.seed is not None:
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in regression tasks
            np.random.seed(self.seed)
            # ✅ Best Practice: Using torch.device to compare ensures compatibility with PyTorch's device management
            torch.manual_seed(self.seed)
        # ✅ Best Practice: Use of descriptive variable names for clarity
        # ✅ Best Practice: Explicitly move model to the specified device

        # 🧠 ML Signal: Custom loss function implementation
        self.sandwich_model = SandwichModel(
            # ⚠️ SAST Risk (Low): Assumes pred and label are tensors; no input validation
            fea_dim=self.fea_dim,
            # 🧠 ML Signal: Handling missing values in labels
            cnn_dim_1=self.cnn_dim_1,
            cnn_dim_2=self.cnn_dim_2,
            cnn_kernel_size=self.cnn_kernel_size,
            # 🧠 ML Signal: Use of mean squared error as a loss function
            rnn_dim_1=self.rnn_dim_1,
            # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values
            rnn_dim_2=self.rnn_dim_2,
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if self.loss is not "mse"
            rnn_dups=self.rnn_dups,
            # 🧠 ML Signal: Conditional logic based on self.metric value
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            # ⚠️ SAST Risk (Low): Potential for negative loss values if self.loss_fn returns positive values
            device=self.device,
        )
        # ⚠️ SAST Risk (Low): Use of string interpolation in exception message could expose internal state
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.sandwich_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.sandwich_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Shuffling data indices for training
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
        self.sandwich_model.to(self.device)

    # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        # ✅ Best Practice: Clipping gradients to prevent exploding gradients
        return torch.mean(loss)

    # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization layers.
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            # ✅ Best Practice: Use np.arange for generating indices, which is efficient and clear.
            return self.mse(pred[mask], label[mask])

        # 🧠 ML Signal: Iterating over data in batches is a common pattern in ML for handling large datasets.
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # ⚠️ SAST Risk (Low): Ensure that x_values and y_values are properly sanitized to prevent data leakage.

        if self.metric in ("", "loss"):
            # ⚠️ SAST Risk (Low): Ensure that x_values and y_values are properly sanitized to prevent data leakage.
            return -self.loss_fn(pred[mask], label[mask])

        # 🧠 ML Signal: Model prediction step, a key operation in ML workflows.
        raise ValueError("unknown metric `%s`" % self.metric)
    # 🧠 ML Signal: Loss calculation is a critical step in evaluating model performance.

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        # 🧠 ML Signal: Metric calculation is important for assessing model accuracy or other performance metrics.
        self.sandwich_model.train()

        # ✅ Best Practice: Return the mean of losses and scores for a summary of the model's performance.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # ⚠️ SAST Risk (Low): Potential directory traversal if save_path is user-controlled

            pred = self.sandwich_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            # 🧠 ML Signal: Tracking training and validation results
            torch.nn.utils.clip_grad_value_(self.sandwich_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # 🧠 ML Signal: Model training state
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.sandwich_model.eval()
        # 🧠 ML Signal: Training epoch

        scores = []
        losses = []
        # 🧠 ML Signal: Evaluation metrics

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            # 🧠 ML Signal: Model checkpointing
            pred = self.sandwich_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        # ⚠️ SAST Risk (Low): No check for dataset validity or integrity
        return np.mean(losses), np.mean(scores)

    # ⚠️ SAST Risk (Low): Ensure save_path is secure to prevent overwriting critical files
    def fit(
        # 🧠 ML Signal: Usage of dataset preparation method
        self,
        dataset: DatasetH,
        # ✅ Best Practice: Free GPU memory after use
        evals_result=dict(),
        # 🧠 ML Signal: Model evaluation mode set
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            # ✅ Best Practice: Use of range with step for batch processing
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        # 🧠 ML Signal: Model prediction without gradient tracking
        # ⚠️ SAST Risk (Low): Potential device compatibility issues
        # ✅ Best Practice: Returning predictions as a pandas Series
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.sandwich_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.sandwich_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.sandwich_model.eval()
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
                pred = self.sandwich_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)