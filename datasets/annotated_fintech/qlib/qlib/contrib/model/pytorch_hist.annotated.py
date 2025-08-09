# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
import numpy as np
import pandas as pd

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
from typing import Text, Union
import urllib.request
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
import torch
import torch.nn as nn

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
import torch.optim as optim
from .pytorch_utils import count_parameters

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
from ...model.base import Model

# ✅ Best Practice: Use of relative imports for internal modules improves maintainability and clarity.
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class HIST(Model):
    """HIST Model

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
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
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
        base_model="GRU",
        model_path=None,
        stock2concept=None,
        stock_index=None,
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("HIST")
        # ✅ Best Practice: Use of .lower() ensures case-insensitive comparison for optimizer
        self.logger.info("HIST pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # 🧠 ML Signal: Logging parameters can be used to understand model configuration patterns
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.stock2concept = stock2concept
        self.stock_index = stock_index
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.seed = seed

        self.logger.info(
            "HIST parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\nstock2concept : {}"
            "\nstock_index : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                early_stop,
                # ⚠️ SAST Risk (Low): Setting a random seed can lead to reproducibility issues if not handled properly
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                stock2concept,
                stock_index,
                GPU,
                seed,
            )
        )

        if self.seed is not None:
            # 🧠 ML Signal: Logging model structure can be used to understand model architecture patterns
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # 🧠 ML Signal: Logging model size can be used to understand resource usage patterns

        self.HIST_model = HISTModel(
            # ✅ Best Practice: Use of .lower() ensures case-insensitive comparison for optimizer
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            # 🧠 ML Signal: Checking for GPU usage is a common pattern in ML code to optimize performance.
            num_layers=self.num_layers,
            dropout=self.dropout,
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in ML
            # ⚠️ SAST Risk (Low): Potential for incorrect device comparison if `self.device` is not properly initialized.
            base_model=self.base_model,
            # ✅ Best Practice: Consider handling cases where `self.device` might not be set or is None.
        )
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
        # ✅ Best Practice: Use descriptive variable names for clarity
        self.logger.info("model:\n{:}".format(self.HIST_model))
        # 🧠 ML Signal: Custom loss function implementation
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.HIST_model))
        )
        # 🧠 ML Signal: Use of torch.mean indicates integration with PyTorch, a popular ML library
        if optimizer.lower() == "adam":
            # ✅ Best Practice: Handle NaN values in labels to avoid computation errors
            self.train_optimizer = optim.Adam(self.HIST_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Conditional logic based on loss type
            self.train_optimizer = optim.SGD(self.HIST_model.parameters(), lr=self.lr)
        else:
            # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid values in label
            # 🧠 ML Signal: Use of mean squared error for loss calculation
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        # 🧠 ML Signal: Conditional logic based on self.metric value
        # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
        self.fitted = False
        self.HIST_model.to(self.device)

    # 🧠 ML Signal: Masking pred and label tensors

    @property
    def use_gpu(self):
        # 🧠 ML Signal: Calculation of mean-centered values
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        # 🧠 ML Signal: Calculation of a correlation-like metric
        loss = (pred - label) ** 2
        # 🧠 ML Signal: Use of groupby operation on a DataFrame, indicating data aggregation pattern
        return torch.mean(loss)

    # 🧠 ML Signal: Handling of different metric types

    # 🧠 ML Signal: Use of numpy operations for array manipulation
    def loss_fn(self, pred, label):
        # 🧠 ML Signal: Use of a loss function with masked values
        mask = ~torch.isnan(label)

        # ⚠️ SAST Risk (Low): Potential information disclosure through error message
        # 🧠 ML Signal: Conditional logic based on a parameter, indicating a behavioral pattern
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # 🧠 ML Signal: Use of random shuffling, indicating data randomization pattern

        # 🧠 ML Signal: Method for training a model epoch
        raise ValueError("unknown loss `%s`" % self.loss)

    # ⚠️ SAST Risk (Low): Loading external data without validation
    def metric_fn(self, pred, label):
        # ✅ Best Practice: Returning multiple values as a tuple for clarity and convenience
        mask = torch.isfinite(label)

        if self.metric == "ic":
            x = pred[mask]
            # ⚠️ SAST Risk (Low): Replacing NaN values with a constant without context
            y = label[mask]

            # 🧠 ML Signal: Switching model to training mode
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            # 🧠 ML Signal: Shuffling data for training
            return torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
            )

        if self.metric == ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        # 🧠 ML Signal: Converting data to PyTorch tensors and moving to device

        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        # 🧠 ML Signal: Model prediction step
        # organize the train data into daily batches
        # ⚠️ SAST Risk (Low): Loading data from a file path without validation can lead to potential security risks.
        daily_count = df.groupby(level=0, group_keys=False).size().values
        # 🧠 ML Signal: Calculating loss
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        # 🧠 ML Signal: Optimizer step preparation
        if shuffle:
            # shuffle data
            # 🧠 ML Signal: Backpropagation step
            # ⚠️ SAST Risk (Low): Replacing NaN values with a constant without validation may lead to incorrect data handling.
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            # ✅ Best Practice: Gradient clipping to prevent exploding gradients
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    # 🧠 ML Signal: Optimizer step to update model parameters

    # 🧠 ML Signal: Using a method to get daily intervals suggests a time-series or sequential data processing pattern.
    def train_epoch(self, x_train, y_train, stock_index):
        stock2concept_matrix = np.load(self.stock2concept)
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        # ✅ Best Practice: Converting data to torch tensors for GPU processing is efficient for ML tasks.
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.HIST_model.train()

        # organize the train data into daily batches
        # 🧠 ML Signal: Using a model in evaluation mode indicates inference or validation phase.
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        # 🧠 ML Signal: Calculating loss during evaluation suggests model performance tracking.
        # 🧠 ML Signal: Using a metric function to evaluate predictions indicates performance measurement.
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            concept_matrix = (
                torch.from_numpy(stock2concept_matrix[stock_index[batch]])
                .float()
                .to(self.device)
            )
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            pred = self.HIST_model(feature, concept_matrix)
            # ✅ Best Practice: Returning the mean of losses and scores provides a summary metric for evaluation.
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.HIST_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, stock_index):
        # ⚠️ SAST Risk (Medium): Downloading files from a URL without validation can lead to security risks.
        # prepare training data
        stock2concept_matrix = np.load(self.stock2concept)
        x_values = data_x.values
        # ⚠️ SAST Risk (Low): Using `allow_pickle=True` can lead to arbitrary code execution if the file is tampered.
        y_values = np.squeeze(data_y.values)
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.HIST_model.eval()

        scores = []
        losses = []

        # ✅ Best Practice: Ensure the save path is valid and created if it doesn't exist.
        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = (
                torch.from_numpy(stock2concept_matrix[stock_index[batch]])
                .float()
                .to(self.device)
            )
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.HIST_model(feature, concept_matrix)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                # ⚠️ SAST Risk (Medium): Loading a model from a file without validation can lead to security risks.
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        if not os.path.exists(self.stock2concept):
            url = "https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/qlib_csi300_stock2concept.npy"
            urllib.request.urlretrieve(url, self.stock2concept)

        stock_index = np.load(self.stock_index, allow_pickle=True).item()
        df_train["stock_index"] = 733
        df_train["stock_index"] = df_train.index.get_level_values("instrument").map(
            stock_index
        )
        df_valid["stock_index"] = 733
        # 🧠 ML Signal: Deep copying model state for best parameters is a common pattern in model training.
        df_valid["stock_index"] = df_valid.index.get_level_values("instrument").map(
            stock_index
        )

        x_train, y_train, stock_index_train = (
            df_train["feature"],
            df_train["label"],
            df_train["stock_index"],
        )
        x_valid, y_valid, stock_index_valid = (
            df_valid["feature"],
            df_valid["label"],
            df_valid["stock_index"],
        )

        save_path = get_or_create_path(save_path)
        # ⚠️ SAST Risk (Low): No check for dataset being None or invalid type

        stop_steps = 0
        best_score = -np.inf
        # ⚠️ SAST Risk (Low): No validation for self.stock2concept path
        # ⚠️ SAST Risk (Medium): Saving a model to a file without validation can lead to security risks.
        best_epoch = 0
        evals_result["train"] = []
        # ⚠️ SAST Risk (Low): No validation for self.stock_index path
        evals_result["valid"] = []

        # 🧠 ML Signal: Usage of dataset preparation method
        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        # ⚠️ SAST Risk (Low): Potential KeyError if "instrument" level is missing
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        else:
            # ⚠️ SAST Risk (Low): No check for NaN values before assignment
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path))
        # 🧠 ML Signal: Model evaluation mode set

        model_dict = self.HIST_model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_model.state_dict().items()
            if k in model_dict  # pylint: disable=E1135
            # 🧠 ML Signal: Usage of custom method to get daily intervals
        }
        model_dict.update(pretrained_dict)
        # 🧠 ML Signal: Custom model class definition for PyTorch
        self.HIST_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")
        # ⚠️ SAST Risk (Low): No validation for device compatibility

        # train
        self.logger.info("training...")
        # 🧠 ML Signal: Conditional logic to select model architecture
        self.fitted = True
        # 🧠 ML Signal: Model prediction without gradient tracking
        # ✅ Best Practice: Returning a pandas Series for better data handling

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train, stock_index_train)

            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(
                x_train, y_train, stock_index_train
            )
            # 🧠 ML Signal: Conditional logic to select model architecture
            val_loss, val_score = self.test_epoch(x_valid, y_valid, stock_index_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.HIST_model.state_dict())
            # ⚠️ SAST Risk (Low): Potential for exception if base_model is not "GRU" or "LSTM"
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # ✅ Best Practice: Use of Xavier initialization for weights
                    self.logger.info("early stop")
                    break

        # ✅ Best Practice: Use of Xavier initialization for weights
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.HIST_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

    # ✅ Best Practice: Use of Xavier initialization for weights

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            # ✅ Best Practice: Use of Xavier initialization for weights
            raise ValueError("model is not fitted yet!")

        stock2concept_matrix = np.load(self.stock2concept)
        # ✅ Best Practice: Use of Xavier initialization for weights
        stock_index = np.load(self.stock_index, allow_pickle=True).item()
        df_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        df_test["stock_index"] = 733
        # ✅ Best Practice: Use of Xavier initialization for weights
        df_test["stock_index"] = df_test.index.get_level_values("instrument").map(
            stock_index
        )
        stock_index_test = df_test["stock_index"].values
        stock_index_test[np.isnan(stock_index_test)] = 733
        # ✅ Best Practice: Use of Xavier initialization for weights
        stock_index_test = stock_index_test.astype("int")
        df_test = df_test.drop(["stock_index"], axis=1)
        index = df_test.index
        # ✅ Best Practice: Use of Xavier initialization for weights

        self.HIST_model.eval()
        # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
        x_values = df_test.values
        # ✅ Best Practice: Use of Xavier initialization for weights
        preds = []
        # 🧠 ML Signal: Use of matrix multiplication and cosine similarity calculation, common in ML models for similarity measures.

        # organize the data into daily batches
        # ✅ Best Practice: Use of Xavier initialization for weights
        # 🧠 ML Signal: Calculation of vector norms, often used in normalization processes in ML.
        daily_index, daily_count = self.get_daily_inter(df_test, shuffle=False)

        # 🧠 ML Signal: Calculation of vector norms, often used in normalization processes in ML.
        for idx, count in zip(daily_index, daily_count):
            # 🧠 ML Signal: Use of device management for tensors, indicating GPU/CPU usage
            batch = slice(idx, idx + count)
            # ⚠️ SAST Risk (Low): Adding a small constant (1e-6) to prevent division by zero, but consider handling edge cases more explicitly.
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            # ✅ Best Practice: Reshape operation for better data manipulation
            concept_matrix = (
                torch.from_numpy(stock2concept_matrix[stock_index_test[batch]])
                .float()
                .to(self.device)
            )

            # ✅ Best Practice: Permute operation for changing tensor dimensions
            with torch.no_grad():
                pred = self.HIST_model(x_batch, concept_matrix).detach().cpu().numpy()

            # ✅ Best Practice: Selecting the last output of RNN for further processing
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


# ✅ Best Practice: Summing and reshaping for broadcasting


# ✅ Best Practice: Element-wise multiplication for matrix operations
class HISTModel(nn.Module):
    def __init__(
        self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"
    ):
        # ✅ Best Practice: Adding a tensor of ones for numerical stability
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        # ✅ Best Practice: Element-wise division for normalization

        if base_model == "GRU":
            # ✅ Best Practice: Matrix multiplication for feature transformation
            self.rnn = nn.GRU(
                input_size=d_feat,
                # ✅ Best Practice: Filtering out zero-sum rows for cleaner data
                hidden_size=hidden_size,
                num_layers=num_layers,
                # 🧠 ML Signal: Use of cosine similarity for measuring similarity between vectors
                batch_first=True,
                dropout=dropout,
                # 🧠 ML Signal: Use of softmax for probability distribution
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                # ✅ Best Practice: Use of activation function for non-linearity
                batch_first=True,
                dropout=dropout,
            )
        else:
            # 🧠 ML Signal: Use of cosine similarity for measuring similarity between vectors
            raise ValueError("unknown base model name `%s`" % base_model)

        # ✅ Best Practice: Extracting diagonal for special handling
        self.fc_es = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es.weight)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        # ✅ Best Practice: Element-wise operations for matrix manipulation
        torch.nn.init.xavier_uniform_(self.fc_is.weight)

        # ✅ Best Practice: Use of linspace for generating sequences
        self.fc_es_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_middle.weight)
        # ✅ Best Practice: Use of max for finding maximum values and indices
        self.fc_is_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_middle.weight)

        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        # ✅ Best Practice: Adding diagonal elements for matrix stability
        # ✅ Best Practice: Transpose and matrix multiplication for feature transformation
        # ✅ Best Practice: Filtering out zero-sum rows for cleaner data
        # 🧠 ML Signal: Use of softmax for probability distribution
        # ✅ Best Practice: Use of activation function for non-linearity
        # 🧠 ML Signal: Use of cosine similarity for measuring similarity between vectors
        # ✅ Best Practice: Summing outputs for final prediction
        # ✅ Best Practice: Squeeze operation for removing single-dimensional entries
        torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)
        self.fc_indi_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi_fore.weight)

        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)

        self.fc_out_es = nn.Linear(hidden_size, 1)
        self.fc_out_is = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
        return cos_similarity

    def forward(self, x, concept_matrix):
        device = torch.device(torch.get_device(x))

        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Predefined Concept Module

        stock_to_concept = concept_matrix

        stock_to_concept_sum = (
            torch.sum(stock_to_concept, 0)
            .reshape(1, -1)
            .repeat(stock_to_concept.shape[0], 1)
        )
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + (
            torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device)
        )
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        hidden = hidden[hidden.sum(1) != 0]

        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        e_shared_info = concept_to_stock.mm(hidden)
        e_shared_info = self.fc_es(e_shared_info)

        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.fc_es_fore(e_shared_info)
        output_es = self.leaky_relu(output_es)

        # Hidden Concept Module
        i_shared_info = x_hidden - e_shared_back
        hidden = i_shared_info
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, hidden)
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        i_stock_to_concept = i_stock_to_concept * (
            torch.ones(dim, dim) - torch.eye(dim)
        ).to(device)
        row = torch.linspace(0, dim - 1, dim).to(device).long()
        column = i_stock_to_concept.max(1)[1].long()
        value = i_stock_to_concept.max(1)[0]
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed(
            (i_stock_to_concept.sum(0) != 0).float() * diag
        )
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
        hidden = hidden[hidden.sum(1) != 0]

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = i_concept_to_stock.mm(hidden)
        i_shared_info = self.fc_is(i_shared_info)

        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.fc_is_fore(i_shared_info)
        output_is = self.leaky_relu(output_is)

        # Individual Information Module
        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)

        # Stock Trend Prediction
        all_info = output_es + output_is + output_indi
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all
