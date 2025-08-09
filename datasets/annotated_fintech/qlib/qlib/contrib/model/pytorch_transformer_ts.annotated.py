# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

import numpy as np
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TransformerModel(Model):
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
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # ✅ Best Practice: Convert optimizer to lowercase to ensure consistent comparison
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        self.lr = lr
        self.reg = reg
        self.metric = metric
        # 🧠 ML Signal: Logging initialization details can be useful for debugging and monitoring
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        # 🧠 ML Signal: Setting random seed for reproducibility
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        # ✅ Best Practice: Encapsulate model creation in a separate method for clarity
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))
        # ✅ Best Practice: Use a factory method or pattern for optimizer creation

        if self.seed is not None:
            np.random.seed(self.seed)
            # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating hardware preference
            torch.manual_seed(self.seed)

        # ✅ Best Practice: Use of torch.device to handle device type
        # ✅ Best Practice: Consider adding type hints for better code readability and maintainability
        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function, common in regression tasks
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        # ✅ Best Practice: Ensure inputs are converted to float for consistent numerical operations
        # 🧠 ML Signal: Custom loss function implementation
        elif optimizer.lower() == "gd":
            # ✅ Best Practice: Ensure model is moved to the correct device
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        # 🧠 ML Signal: Use of torch.mean, indicating usage of PyTorch for tensor operations
        # 🧠 ML Signal: Handling missing values in labels
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # 🧠 ML Signal: Conditional logic based on loss type

        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        self.fitted = False
        # 🧠 ML Signal: Use of mean squared error for loss calculation
        self.model.to(self.device)
    # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values

    # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
    @property
    # 🧠 ML Signal: Conditional logic based on metric type
    def use_gpu(self):
        return self.device != torch.device("cpu")
    # 🧠 ML Signal: Use of mask to filter predictions and labels

    # 🧠 ML Signal: Iterating over data_loader indicates a training loop
    def mse(self, pred, label):
        # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
        loss = (pred.float() - label.float()) ** 2
        # 🧠 ML Signal: Data slicing to separate features and labels
        return torch.mean(loss)

    # 🧠 ML Signal: Data slicing to separate features and labels
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        # 🧠 ML Signal: Model prediction step

        if self.loss == "mse":
            # 🧠 ML Signal: Loss calculation step
            return self.mse(pred[mask], label[mask])

        # 🧠 ML Signal: Optimizer gradient reset step
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization
        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Backpropagation step
    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # ⚠️ SAST Risk (Low): Gradient clipping can mask exploding gradients but may hide underlying issues

        # ✅ Best Practice: Use slicing to separate features and labels for clarity
        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Optimizer step to update model parameters
            return -self.loss_fn(pred[mask], label[mask])

        # ✅ Best Practice: Use torch.no_grad() to prevent gradient computation for efficiency
        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Model prediction step, useful for understanding model usage patterns
    def train_epoch(self, data_loader):
        self.model.train()
        # 🧠 ML Signal: Loss computation step, useful for understanding model evaluation

        # ✅ Best Practice: Use .item() to convert single-element tensors to Python scalars
        # 🧠 ML Signal: Metric computation step, useful for understanding model evaluation
        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)
            # ✅ Best Practice: Use .item() to convert single-element tensors to Python scalars
            # ✅ Best Practice: Use of descriptive variable names for clarity

            # ✅ Best Practice: Use numpy to compute mean for better performance and readability
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if dataset is empty
            self.train_optimizer.step()

    # ✅ Best Practice: Configuring data loaders with fillna_type for data consistency
    def test_epoch(self, data_loader):
        self.model.eval()

        # ✅ Best Practice: Use of DataLoader for efficient data handling
        scores = []
        losses = []

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # ✅ Best Practice: Ensuring save_path is valid or created
            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                # ✅ Best Practice: Initializing evals_result for tracking performance
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)
    # ✅ Best Practice: Logging for tracking training progress

    def fit(
        self,
        dataset: DatasetH,
        # ✅ Best Practice: Logging each epoch for better traceability
        evals_result=dict(),
        save_path=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        # ✅ Best Practice: Logging training and validation scores
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        # 🧠 ML Signal: Use of model state_dict for saving best model parameters
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )
        # ⚠️ SAST Risk (Low): Potential for exception if 'self.fitted' is not a boolean

        save_path = get_or_create_path(save_path)
        # ✅ Best Practice: Implementing early stopping for efficiency

        # 🧠 ML Signal: Usage of dataset preparation with specific column sets
        stop_steps = 0
        train_loss = 0
        # ✅ Best Practice: Logging the best score and epoch
        # 🧠 ML Signal: Configuration of data handling with fillna_type
        best_score = -np.inf
        best_epoch = 0
        # 🧠 ML Signal: Usage of DataLoader with specific batch size and number of workers
        evals_result["train"] = []
        # ⚠️ SAST Risk (Low): Potential risk if save_path is not writable
        evals_result["valid"] = []
        # 🧠 ML Signal: Model evaluation mode set before prediction

        # train
        # ✅ Best Practice: Clearing GPU cache to free up memory
        self.logger.info("training...")
        self.fitted = True
        # 🧠 ML Signal: Custom neural network module definition
        # 🧠 ML Signal: Data slicing and device transfer for model input

        for step in range(self.n_epochs):
            # ✅ Best Practice: Call to superclass initializer ensures proper initialization of inherited attributes.
            # 🧠 ML Signal: Use of torch.no_grad for inference
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            # 🧠 ML Signal: Initialization of positional encoding matrix, common in transformer models.
            # 🧠 ML Signal: Model prediction and conversion to numpy
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Use of torch.arange to create a sequence of positions, typical in sequence models.
            train_loss, train_score = self.test_epoch(train_loader)
            # 🧠 ML Signal: Concatenation of predictions and use of index from data handler
            val_loss, val_score = self.test_epoch(valid_loader)
            # 🧠 ML Signal: Calculation of div_term for scaling positions, a pattern in positional encoding.
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            # 🧠 ML Signal: Use of sine function for even indices in positional encoding.
            # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
            evals_result["valid"].append(val_score)

            # 🧠 ML Signal: Use of cosine function for odd indices in positional encoding.
            # 🧠 ML Signal: Usage of tensor slicing, common in ML models for handling sequences
            # ✅ Best Practice: Inheriting from nn.Module is standard for defining custom models in PyTorch.
            if val_score > best_score:
                # ⚠️ SAST Risk (Low): Potential for index out of range if x.size(0) exceeds self.pe dimensions
                best_score = val_score
                # 🧠 ML Signal: Reshaping positional encoding for batch processing.
                # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
                stop_steps = 0
                best_epoch = step
                # ✅ Best Practice: Use of register_buffer to store tensors not considered model parameters.
                # 🧠 ML Signal: Use of nn.Linear indicates a linear transformation layer
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                # 🧠 ML Signal: Use of PositionalEncoding suggests handling of sequence data
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Use of nn.TransformerEncoderLayer indicates a transformer architecture
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Use of nn.TransformerEncoder suggests a stack of transformer layers

        # 🧠 ML Signal: Use of feature_layer indicates a preprocessing step common in ML models
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 🧠 ML Signal: Use of nn.Linear for decoder layer indicates output transformation
        self.model.load_state_dict(best_param)
        # ✅ Best Practice: Transposing tensors is common in ML to match expected input dimensions
        torch.save(best_param, save_path)
        # ✅ Best Practice: Storing device for potential use in tensor operations

        if self.use_gpu:
            # ✅ Best Practice: Storing d_feat for potential use in other methods
            # 🧠 ML Signal: Use of positional encoding is typical in transformer models
            torch.cuda.empty_cache()
    # 🧠 ML Signal: Use of transformer_encoder suggests a transformer-based architecture
    # ✅ Best Practice: Squeezing output is a common practice to remove single-dimensional entries
    # ✅ Best Practice: Transposing and slicing tensors for decoder input is a common pattern

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
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

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()