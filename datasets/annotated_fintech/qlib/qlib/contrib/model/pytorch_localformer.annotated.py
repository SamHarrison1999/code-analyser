# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package.
import numpy as np
import pandas as pd

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package.
from typing import Text, Union
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package.

import torch

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package.
# 🧠 ML Signal: Inheritance from a base class, indicating a custom model implementation
import torch.nn as nn

# ✅ Best Practice: Use of relative imports for internal modules ensures maintainability and clarity within a package.
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from torch.nn.modules.container import ModuleList

# qrun examples/benchmarks/Localformer/workflow_config_localformer_Alpha360.yaml ”


class LocalformerModel(Model):
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
        # ✅ Best Practice: Use a factory method to create optimizer instances
        self.logger.info(
            "Naive Transformer:"
            "\nbatch_size : {}"
            "\ndevice : {}".format(self.batch_size, self.device)
        )

        # 🧠 ML Signal: Checks if the computation is set to run on a GPU, which is a common pattern in ML for performance optimization
        if self.seed is not None:
            np.random.seed(self.seed)
            # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
            # ⚠️ SAST Risk (Low): Assumes 'self.device' is a valid torch.device object; if not, this could raise an exception
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Use of mean squared error (MSE) loss function, common in regression tasks.
        # ⚠️ SAST Risk (Low): Potential denial of service if an unsupported optimizer is provided
        self.model = Transformer(
            d_feat, d_model, nhead, num_layers, dropout, self.device
        )
        # ✅ Best Practice: Ensure that both pred and label are tensors to avoid runtime errors.
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        # 🧠 ML Signal: Moving model to the specified device (CPU/GPU)
        # 🧠 ML Signal: Use of torch.mean to compute the average loss, indicating a reduction operation.
        # ✅ Best Practice: Use descriptive variable names for better readability
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        # 🧠 ML Signal: Conditional logic based on loss type indicates model configuration
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )
        # 🧠 ML Signal: Use of mask suggests handling of missing or invalid data
        # ✅ Best Practice: Check for finite values to avoid computation errors with invalid data

        self.fitted = False
        # ⚠️ SAST Risk (Low): Error message may expose internal state if not handled properly
        # 🧠 ML Signal: Conditional logic based on metric type indicates model evaluation behavior
        self.model.to(self.device)

    # 🧠 ML Signal: Use of loss function suggests model training or evaluation context
    @property
    def use_gpu(self):
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if metric is unknown
        return self.device != torch.device("cpu")

    # 🧠 ML Signal: Model training loop

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        # 🧠 ML Signal: Data shuffling for training
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        # ✅ Best Practice: Gradient clipping to prevent exploding gradients
        if self.metric in ("", "loss"):
            # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Use of indices for batching indicates a custom batching strategy
    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # 🧠 ML Signal: Iterating over data in batches is a common pattern in ML training/testing
        y_train_values = np.squeeze(y_train.values)

        # ✅ Best Practice: Break condition to handle the last incomplete batch
        self.model.train()

        indices = np.arange(len(x_train_values))
        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
        np.random.shuffle(indices)

        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                # ✅ Best Practice: Use of torch.no_grad() to prevent gradient computation during evaluation
                break
            # 🧠 ML Signal: Use of a loss function to evaluate model performance

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

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            # 🧠 ML Signal: Use of a metric function to evaluate model performance
            # ✅ Best Practice: Return the mean of losses and scores for overall evaluation

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
            data_key=DataHandlerLP.DK_L,
        )
        # ⚠️ SAST Risk (Low): Potential for exception if 'prepare' method does not handle 'segment' properly
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        # 🧠 ML Signal: Model evaluation mode is set, indicating inference phase
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        # ⚠️ SAST Risk (Low): Potential device mismatch if 'self.device' is not properly set

        # train
        # ✅ Best Practice: Using 'torch.no_grad()' for inference to save memory
        self.logger.info("training...")
        # 🧠 ML Signal: Custom neural network module definition
        self.fitted = True
        # 🧠 ML Signal: Model prediction step

        # ✅ Best Practice: Call to superclass's __init__ method ensures proper initialization of the base class.
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # 🧠 ML Signal: Initialization of positional encoding matrix, common in transformer models.
            # ⚠️ SAST Risk (Low): Assumes 'index' is unique and matches the length of concatenated predictions
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            # 🧠 ML Signal: Use of torch.arange to create a sequence of positions, typical in sequence models.
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            # 🧠 ML Signal: Calculation of div_term for scaling positions, a pattern in positional encoding.
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
            # 🧠 ML Signal: Use of sine and cosine functions for positional encoding, a common pattern in transformers.
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            # 🧠 ML Signal: Usage of slicing and tensor operations, common in ML model implementations
            # ✅ Best Practice: Function name starts with an underscore, indicating it's intended for internal use.

            # 🧠 ML Signal: Reshaping positional encoding for batch processing, typical in deep learning models.
            # ⚠️ SAST Risk (Low): Potential for index out of bounds if x.size(0) is greater than self.pe's first dimension
            if val_score > best_score:
                # 🧠 ML Signal: Custom class definition for a neural network module
                # 🧠 ML Signal: Use of deepcopy suggests the need for independent copies of the module.
                best_score = val_score
                # ✅ Best Practice: Inheriting from nn.Module for custom neural network components
                # ✅ Best Practice: Use of register_buffer to store non-parameter tensors, ensuring they are not updated during training.
                # ✅ Best Practice: List comprehension is used for creating a list of clones, which is concise and efficient.
                stop_steps = 0
                best_epoch = step
                # ✅ Best Practice: Using __constants__ to define immutable class attributes
                # ✅ Best Practice: Call to superclass initializer ensures proper initialization of inherited attributes
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                # 🧠 ML Signal: Use of cloning pattern for creating multiple layers
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Use of convolutional layers in a transformer model
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Storing the number of layers as an attribute

        # 🧠 ML Signal: Iterating over layers in a neural network model
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        # ✅ Best Practice: Use meaningful variable names for readability
        torch.save(best_param, save_path)

        # ⚠️ SAST Risk (Low): Potential performance issue with multiple transpose operations
        if self.use_gpu:
            # ✅ Best Practice: Inheriting from nn.Module is standard for defining custom neural network models in PyTorch.
            torch.cuda.empty_cache()

    # ⚠️ SAST Risk (Low): Potential performance issue with multiple transpose operations

    # 🧠 ML Signal: Use of default parameters in model initialization
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        # ✅ Best Practice: Use of default parameters for flexibility and ease of use
        # 🧠 ML Signal: Use of GRU in model architecture
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

        # 🧠 ML Signal: Use of Linear layer for feature transformation
        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                # 🧠 ML Signal: Use of positional encoding in transformer model
                end = sample_num
            else:
                # 🧠 ML Signal: Use of TransformerEncoderLayer in model architecture
                end = begin + self.batch_size
            # 🧠 ML Signal: Reshaping and permuting tensors is common in ML models for data preparation.

            # 🧠 ML Signal: Custom transformer encoder implementation
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            # 🧠 ML Signal: Passing data through a feature layer is typical in neural networks.

            # 🧠 ML Signal: Use of Linear layer for decoding in model architecture
            with torch.no_grad():
                # 🧠 ML Signal: Transposing tensors is a common operation in sequence models.
                pred = self.model(x_batch).detach().cpu().numpy()
            # ✅ Best Practice: Storing device information for model deployment

            preds.append(pred)
        # 🧠 ML Signal: Positional encoding is a common technique in transformer models.
        # ✅ Best Practice: Storing feature dimension for reference

        # 🧠 ML Signal: Using a transformer encoder is a common pattern in sequence modeling.
        # 🧠 ML Signal: Squeezing the output is a common operation to adjust tensor dimensions.
        # 🧠 ML Signal: Using RNNs for sequence data is a common pattern in ML models.
        # 🧠 ML Signal: Decoding the output of a sequence model is a common pattern.
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


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class LocalformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, d_model):
        super(LocalformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        output = src
        out = src

        for i, mod in enumerate(self.layers):
            # [T, N, F] --> [N, T, F] --> [N, F, T]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)

            output = mod(output + out, src_mask=mask)

        return output + out


class Transformer(nn.Module):
    def __init__(
        self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None
    ):
        super(Transformer, self).__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = LocalformerEncoder(
            self.encoder_layer, num_layers=num_layers, d_model=d_model
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

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
