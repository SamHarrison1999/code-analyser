# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Use of relative imports for internal modules helps maintain package structure.

import numpy as np
# ✅ Best Practice: Use of relative imports for internal modules helps maintain package structure.
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for internal modules helps maintain package structure.
import torch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for internal modules helps maintain package structure.
import torch.optim as optim
# ✅ Best Practice: Use of relative imports for internal modules helps maintain package structure.
# 🧠 ML Signal: Inheritance from a base class, indicating a potential pattern for ML model architecture
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from torch.nn.modules.container import ModuleList


class LocalformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        early_stop=5,
        loss="mse",
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        optimizer="adam",
        reg=1e-3,
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        n_jobs=10,
        GPU=0,
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        seed=None,
        **kwargs,
    # 🧠 ML Signal: Use of hyperparameters for model configuration
    ):
        # set hyper-parameters.
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.d_model = d_model
        self.dropout = dropout
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.n_epochs = n_epochs
        self.lr = lr
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.early_stop = early_stop
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.optimizer = optimizer.lower()
        self.loss = loss
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        # 🧠 ML Signal: Use of hyperparameters for model configuration
        self.logger.info(
            "Improved Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device)
        # ✅ Best Practice: Use of logging for tracking model configuration
        )

        # ✅ Best Practice: Use of logging for tracking model configuration
        if self.seed is not None:
            np.random.seed(self.seed)
            # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating hardware preference
            torch.manual_seed(self.seed)

        # ⚠️ SAST Risk (Low): Seed setting for reproducibility, but may not cover all sources of randomness
        # ✅ Best Practice: Use of torch.device to handle device type
        # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in regression tasks
        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        if optimizer.lower() == "adam":
            # ✅ Best Practice: Convert inputs to float to ensure consistent numerical operations
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Model initialization with specified parameters
            # ✅ Best Practice: Use torch.mean to compute the mean of the tensor, a standard practice for loss functions
            # ✅ Best Practice: Use of torch.isnan to handle NaN values in tensors
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            # 🧠 ML Signal: Conditional logic based on self.loss value
            # ⚠️ SAST Risk (Low): Use of dynamic optimizer selection, potential for unsupported optimizers
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
        # 🧠 ML Signal: Use of mask to filter out NaN values before computation
        self.fitted = False
        self.model.to(self.device)
    # ⚠️ SAST Risk (Low): Potential for unhandled exception if self.loss is not "mse"
    # 🧠 ML Signal: Use of torch.isfinite indicates handling of numerical stability and potential NaN/Inf values in tensors.

    @property
    # ✅ Best Practice: Using a tuple for multiple string comparisons is efficient and readable.
    def use_gpu(self):
        # 🧠 ML Signal: Tracking model training state
        return self.device != torch.device("cpu")
    # ⚠️ SAST Risk (Low): Ensure that pred and label are tensors of compatible shapes to avoid runtime errors.

    # ✅ Best Practice: Explicitly moving model to the specified device
    # 🧠 ML Signal: Iterating over data_loader indicates a training loop
    def mse(self, pred, label):
        # ⚠️ SAST Risk (Low): Raising a ValueError for unknown metrics is good, but consider logging the error for better traceability.
        loss = (pred.float() - label.float()) ** 2
        # 🧠 ML Signal: Slicing data to separate features and labels is common in ML training
        return torch.mean(loss)

    # 🧠 ML Signal: Moving data to a specific device (e.g., GPU) is typical in ML workflows
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        # 🧠 ML Signal: Model prediction step in a training loop

        if self.loss == "mse":
            # 🧠 ML Signal: Calculating loss is a key step in training
            return self.mse(pred[mask], label[mask])

        # 🧠 ML Signal: Zeroing gradients is a standard practice in training loops
        # 🧠 ML Signal: Model evaluation mode is set, indicating a testing phase
        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Backpropagation step in training
    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # ⚠️ SAST Risk (Low): Clipping gradients can prevent exploding gradients but should be used cautiously

        # ✅ Best Practice: Explicitly specifying dimensions for slicing improves code readability
        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Optimizer step to update model parameters
            return -self.loss_fn(pred[mask], label[mask])

        # 🧠 ML Signal: Disabling gradient calculation for inference
        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Model prediction step
    def train_epoch(self, data_loader):
        self.model.train()
        # 🧠 ML Signal: Loss calculation for model evaluation

        # ✅ Best Practice: Using .item() to convert tensors to Python scalars
        # 🧠 ML Signal: Metric calculation for model evaluation
        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)
            # ✅ Best Practice: Using .item() to convert tensors to Python scalars
            # ✅ Best Practice: Use a default argument of None instead of a mutable type like dict

            # ✅ Best Practice: Using numpy for mean calculation ensures compatibility with numerical operations
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:
            # ⚠️ SAST Risk (Low): Potential directory traversal if save_path is user-controlled
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
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
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # 🧠 ML Signal: Use of deepcopy to save model state for best parameters
        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )
        # ⚠️ SAST Risk (Low): Ensure save_path is validated to prevent overwriting critical files

        # ⚠️ SAST Risk (Low): No check for dataset validity or type, which could lead to runtime errors.
        save_path = get_or_create_path(save_path)

        # ⚠️ SAST Risk (Low): Ensure proper GPU resource management to prevent memory leaks
        stop_steps = 0
        # 🧠 ML Signal: Usage of dataset preparation with specific column sets and data keys.
        train_loss = 0
        best_score = -np.inf
        # ✅ Best Practice: Configuring data handling to fill missing values.
        best_epoch = 0
        evals_result["train"] = []
        # 🧠 ML Signal: Usage of DataLoader with specific batch size and number of workers.
        evals_result["valid"] = []

        # ✅ Best Practice: Setting the model to evaluation mode before prediction.
        # train
        self.logger.info("training...")
        self.fitted = True

        # ⚠️ SAST Risk (Low): Assumes data shape and device compatibility without validation.
        # 🧠 ML Signal: Custom neural network module definition
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # ✅ Best Practice: Call to superclass's __init__ method ensures proper initialization of the base class.
            self.logger.info("training...")
            # ✅ Best Practice: Using no_grad for inference to save memory and computations.
            self.train_epoch(train_loader)
            # 🧠 ML Signal: Initialization of positional encoding matrix, common in transformer models.
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            # 🧠 ML Signal: Returning predictions as a pandas Series with a specific index.
            # 🧠 ML Signal: Use of torch.arange to create a sequence of positions, typical in sequence models.
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # 🧠 ML Signal: Calculation of div_term for scaling positions, a pattern in positional encoding.
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            # 🧠 ML Signal: Use of sine function for even indices in positional encoding.
            # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters

            if val_score > best_score:
                # 🧠 ML Signal: Use of tensor slicing, common in ML model implementations
                # 🧠 ML Signal: Use of cosine function for odd indices in positional encoding.
                # ✅ Best Practice: Use of deepcopy ensures that each clone is a distinct copy, preventing shared state issues.
                best_score = val_score
                # ⚠️ SAST Risk (Low): Potential for index out of range if x.size(0) exceeds self.pe dimensions
                # 🧠 ML Signal: Cloning modules is a common pattern in neural network architectures for creating multiple layers or components.
                stop_steps = 0
                # 🧠 ML Signal: Reshaping positional encoding for batch processing.
                # ✅ Best Practice: Class should inherit from nn.Module for PyTorch models
                best_epoch = step
                # ✅ Best Practice: List comprehension is a concise and efficient way to create a list of clones.
                best_param = copy.deepcopy(self.model.state_dict())
            # ⚠️ SAST Risk (Low): register_buffer is used to store tensors not considered model parameters, ensure it's used correctly.
            # ✅ Best Practice: Use of __constants__ to define immutable class attributes
            else:
                # ✅ Best Practice: Call to superclass initializer ensures proper initialization of the base class
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Use of cloning pattern for creating multiple layers
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Use of convolutional layers in a transformer model

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 🧠 ML Signal: Storing the number of layers as an attribute
        self.model.load_state_dict(best_param)
        # 🧠 ML Signal: Iterating over layers in a neural network model
        torch.save(best_param, save_path)

        # ✅ Best Practice: Transposing tensors for correct dimensionality
        if self.use_gpu:
            torch.cuda.empty_cache()
    # 🧠 ML Signal: Applying convolutional layers in a sequence

    # 🧠 ML Signal: Custom neural network module definition
    def predict(self, dataset):
        # 🧠 ML Signal: Using residual connections in a neural network
        if not self.fitted:
            # ✅ Best Practice: Call to super() ensures proper initialization of the parent class.
            raise ValueError("model is not fitted yet!")
        # 🧠 ML Signal: Returning the final output with residual connection
        # 🧠 ML Signal: Use of GRU indicates a sequence modeling task.

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # 🧠 ML Signal: Linear layer used for feature transformation.

            with torch.no_grad():
                # 🧠 ML Signal: Positional encoding is used, indicating a transformer-based model.
                pred = self.model(feature.float()).detach().cpu().numpy()

            # 🧠 ML Signal: Transformer encoder layer used, indicating a transformer-based architecture.
            preds.append(pred)

        # 🧠 ML Signal: Use of feature_layer indicates a preprocessing step common in ML models
        # 🧠 ML Signal: Custom encoder used, suggesting model customization.
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

# ✅ Best Practice: Transposing tensors is common in ML to match expected input dimensions
# 🧠 ML Signal: Linear layer used for decoding, common in regression tasks.

class PositionalEncoding(nn.Module):
    # ✅ Best Practice: Storing device information for potential use in model operations.
    def __init__(self, d_model, max_len=1000):
        # 🧠 ML Signal: Use of positional encoding is typical in transformer models
        super(PositionalEncoding, self).__init__()
        # ✅ Best Practice: Storing feature dimension for potential use in model operations.
        pe = torch.zeros(max_len, d_model)
        # 🧠 ML Signal: Use of RNN indicates sequential data processing
        # ✅ Best Practice: Transposing and slicing tensors for decoder input is a common pattern
        # ✅ Best Practice: Squeezing output is a common practice to remove single-dimensional entries
        # 🧠 ML Signal: Use of transformer_encoder suggests a sequence-to-sequence model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()