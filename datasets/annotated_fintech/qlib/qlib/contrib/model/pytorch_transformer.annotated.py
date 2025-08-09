# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package structure.
import numpy as np
import pandas as pd

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package structure.
from typing import Text, Union
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package structure.

import torch

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package structure.
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

# qrun examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml ”


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        # ✅ Best Practice: Convert optimizer to lowercase to ensure case-insensitive comparison
        **kwargs,
    ):
        # set hyper-parameters.
        self.d_model = d_model
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available or index is invalid
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        # ✅ Best Practice: Use a logger for better traceability and debugging
        self.reg = reg
        self.metric = metric
        # ✅ Best Practice: Log important configuration details for debugging and traceability
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        # 🧠 ML Signal: Setting random seed for reproducibility
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device(
            "cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        # 🧠 ML Signal: Instantiating a Transformer model with specified parameters
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        # ✅ Best Practice: Use a factory method or configuration to handle different optimizers
        self.logger.info(
            "Naive Transformer:"
            "\nbatch_size : {}"
            "\ndevice : {}".format(self.batch_size, self.device)
        )

        # 🧠 ML Signal: Checking if a GPU is being used for computation
        if self.seed is not None:
            np.random.seed(self.seed)
            # ✅ Best Practice: Using torch.device to handle device types
            torch.manual_seed(self.seed)
        # ✅ Best Practice: Ensure input tensors are of the same shape for element-wise operations

        # ⚠️ SAST Risk (Low): Potential denial of service if an unsupported optimizer is provided
        self.model = Transformer(
            d_feat, d_model, nhead, num_layers, dropout, self.device
        )
        # 🧠 ML Signal: Use of mean squared error (MSE) loss function
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        # ✅ Best Practice: Explicitly move model to the specified device
        # ✅ Best Practice: Use descriptive variable names for better readability
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        # 🧠 ML Signal: Conditional logic based on a class attribute
        else:
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )
        # 🧠 ML Signal: Use of masking to handle missing values

        # 🧠 ML Signal: Use of torch.isfinite indicates handling of numerical stability in ML models
        self.fitted = False
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if self.loss is not "mse"
        self.model.to(self.device)

    # 🧠 ML Signal: Conditional logic based on metric type suggests dynamic behavior in model evaluation

    @property
    # 🧠 ML Signal: Use of loss function indicates model evaluation or training process
    def use_gpu(self):
        return self.device != torch.device("cpu")

    # ⚠️ SAST Risk (Low): Potential for unhandled exception if metric is unknown
    # 🧠 ML Signal: Model training loop with data shuffling and batching

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        # 🧠 ML Signal: Random shuffling of training data indices
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        # ✅ Best Practice: Gradient clipping to prevent exploding gradients
        if self.metric in ("", "loss"):
            # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization layers.
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Use of indices for batching indicates a custom batching strategy.
    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        # ✅ Best Practice: Break early if remaining data is less than batch size.

        self.model.train()

        # ⚠️ SAST Risk (Low): Ensure that x_values and y_values are properly sanitized before conversion.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        # ⚠️ SAST Risk (Low): Ensure that x_values and y_values are properly sanitized before conversion.

        for i in range(len(indices))[:: self.batch_size]:
            # ✅ Best Practice: Use torch.no_grad() to prevent tracking history in evaluation mode.
            if len(indices) - i < self.batch_size:
                break
            # 🧠 ML Signal: Use of a custom loss function indicates a specific training objective.

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

            # 🧠 ML Signal: Use of a custom metric function indicates a specific evaluation criterion.
            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            # ✅ Best Practice: Return the mean of losses and scores for a summary of the epoch's performance.

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

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
                pred = self.model(feature)
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
        # ⚠️ SAST Risk (Low): Potential resource leak if GPU memory is not cleared properly
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            # ⚠️ SAST Risk (Low): No check for dataset validity or integrity
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            # 🧠 ML Signal: Usage of dataset preparation method
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        # 🧠 ML Signal: Model evaluation mode set
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # ⚠️ SAST Risk (Low): Potential device mismatch if self.device is not set correctly
        # train
        self.logger.info("training...")
        # 🧠 ML Signal: Custom neural network module definition
        self.fitted = True
        # 🧠 ML Signal: Model prediction without gradient tracking

        # ✅ Best Practice: Explicitly call the superclass's __init__ method to ensure proper initialization
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # ✅ Best Practice: Returning a pandas Series for better data handling
            # 🧠 ML Signal: Usage of torch.zeros to initialize a tensor
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            # 🧠 ML Signal: Usage of torch.arange to create a sequence of numbers
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            # 🧠 ML Signal: Usage of torch.exp and mathematical operations to create a scaling factor
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # 🧠 ML Signal: Usage of torch.sin to apply sine function to tensor elements
            # ✅ Best Practice: Include a docstring to describe the purpose and parameters of the function
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            # 🧠 ML Signal: Usage of torch.cos to apply cosine function to tensor elements
            # 🧠 ML Signal: Usage of slicing to manipulate tensor dimensions
            # 🧠 ML Signal: Custom neural network module definition

            # ⚠️ SAST Risk (Low): Potential for index out of range if x.size(0) exceeds self.pe dimensions
            if val_score > best_score:
                # ✅ Best Practice: Use of default parameters for flexibility and ease of use
                # 🧠 ML Signal: Usage of tensor operations to reshape and transpose
                best_score = val_score
                stop_steps = 0
                # ✅ Best Practice: Use register_buffer to store tensors that should not be considered model parameters
                # 🧠 ML Signal: Initialization of a linear layer for feature transformation
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            # 🧠 ML Signal: Use of positional encoding in transformer architecture
            else:
                stop_steps += 1
                # 🧠 ML Signal: Initialization of a transformer encoder layer
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    # 🧠 ML Signal: Use of transformer encoder with specified number of layers
                    break
        # 🧠 ML Signal: Reshaping and permuting tensors are common in ML models for data preparation.

        # 🧠 ML Signal: Initialization of a linear layer for decoding
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 🧠 ML Signal: Passing data through a feature layer is typical in neural networks.
        self.model.load_state_dict(best_param)
        # ✅ Best Practice: Storing device information for potential use in computations
        torch.save(best_param, save_path)
        # 🧠 ML Signal: Transposing tensors is a common operation in ML for aligning dimensions.

        # ✅ Best Practice: Storing feature dimension for potential use in computations
        if self.use_gpu:
            torch.cuda.empty_cache()

    # 🧠 ML Signal: Positional encoding is a common technique in transformer models.
    # 🧠 ML Signal: Using a transformer encoder is indicative of a transformer-based model.
    # 🧠 ML Signal: Decoding the output of a transformer is a typical step in sequence models.
    # 🧠 ML Signal: Squeezing the output is a common step to adjust tensor dimensions.

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.model.eval()
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
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(
        self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
