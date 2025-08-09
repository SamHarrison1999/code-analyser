# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.

import numpy as np
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.

import torch
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
import torch.nn as nn
import torch.optim as optim
# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters

# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class IGMTF(Model):
    """IGMTF Model

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
        # ✅ Best Practice: Use of a logger for information and debugging purposes
        lr=0.001,
        metric="",
        # ✅ Best Practice: Logging the start of a process
        early_stop=20,
        loss="mse",
        # 🧠 ML Signal: Storing model configuration parameters
        base_model="GRU",
        model_path=None,
        # 🧠 ML Signal: Storing model configuration parameters
        optimizer="adam",
        GPU=0,
        # 🧠 ML Signal: Storing model configuration parameters
        seed=None,
        **kwargs,
    # 🧠 ML Signal: Storing model configuration parameters
    ):
        # Set logger.
        # 🧠 ML Signal: Storing model configuration parameters
        self.logger = get_module_logger("IGMTF")
        self.logger.info("IMGTF pytorch version...")
        # 🧠 ML Signal: Storing model configuration parameters

        # set hyper-parameters.
        # 🧠 ML Signal: Storing model configuration parameters
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # ✅ Best Practice: Logging detailed configuration settings
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.logger.info(
            "IGMTF parameters setting:"
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
                early_stop,
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # ⚠️ SAST Risk (Low): Seed setting for reproducibility, but not secure for cryptographic purposes
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.igmtf_model = IGMTFModel(
            # 🧠 ML Signal: Initializing a model with specific parameters
            # 🧠 ML Signal: Checks if the computation is set to run on a GPU, indicating hardware usage preference
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in ML
            # ✅ Best Practice: Using torch.device to handle device type ensures compatibility with PyTorch
            num_layers=self.num_layers,
            dropout=self.dropout,
            # ✅ Best Practice: Use of descriptive variable names for clarity
            base_model=self.base_model,
        # 🧠 ML Signal: Custom loss function implementation
        )
        # ⚠️ SAST Risk (Low): Assumes pred and label are compatible tensors, potential for runtime errors
        self.logger.info("model:\n{:}".format(self.igmtf_model))
        # 🧠 ML Signal: Handling missing values in labels
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.igmtf_model)))
        # ✅ Best Practice: Logging model details

        # 🧠 ML Signal: Conditional logic based on loss type
        if optimizer.lower() == "adam":
            # ✅ Best Practice: Logging model size for resource management
            self.train_optimizer = optim.Adam(self.igmtf_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Use of mean squared error for loss calculation
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid values
        elif optimizer.lower() == "gd":
            # ⚠️ SAST Risk (Low): Use of dynamic optimizer selection without validation
            self.train_optimizer = optim.SGD(self.igmtf_model.parameters(), lr=self.lr)
        # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
        else:
            # 🧠 ML Signal: Calculation of correlation coefficient
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.igmtf_model.to(self.device)
    # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers

    # ⚠️ SAST Risk (Low): Potential division by zero if vx or vy sums to zero
    @property
    # 🧠 ML Signal: Tracking model training state
    def use_gpu(self):
        return self.device != torch.device("cpu")
    # 🧠 ML Signal: Moving model to the specified device
    # 🧠 ML Signal: Use of negative loss for optimization
    # 🧠 ML Signal: Use of groupby operation on a DataFrame, indicating data aggregation pattern

    def mse(self, pred, label):
        # ⚠️ SAST Risk (Low): Use of string interpolation in exception message
        # 🧠 ML Signal: Use of numpy operations for array manipulation
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        # 🧠 ML Signal: Conditional logic to shuffle data, indicating data randomization pattern
        mask = ~torch.isnan(label)

        # ⚠️ SAST Risk (Low): Use of np.random.shuffle can lead to non-deterministic results
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # 🧠 ML Signal: Use of shuffle=True indicates a need for randomization, common in ML training

        # ✅ Best Practice: Explicit return of multiple values improves code readability
        raise ValueError("unknown loss `%s`" % self.loss)
    # ✅ Best Practice: Setting the model to evaluation mode ensures no gradients are computed

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric == "ic":
            x = pred[mask]
            # ⚠️ SAST Risk (Low): Direct conversion from numpy to torch without validation could lead to unexpected errors
            y = label[mask]

            # 🧠 ML Signal: get_hidden=True suggests the model returns intermediate representations, useful for debugging or analysis
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            # ✅ Best Practice: Detaching and moving to CPU is good for memory management and avoiding side effects
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

        # ✅ Best Practice: Using mean and unsqueeze ensures consistent tensor dimensions
        if self.metric == ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        # ⚠️ SAST Risk (Low): Using dtype=object in np.asarray can lead to inconsistent data types
        # 🧠 ML Signal: Model is set to training mode, indicating a training phase

        raise ValueError("unknown metric `%s`" % self.metric)
    # 🧠 ML Signal: Shuffling data for training, which is a common practice in ML

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        # 🧠 ML Signal: Converting data to PyTorch tensors and moving to device
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        # 🧠 ML Signal: Converting labels to PyTorch tensors and moving to device
        if shuffle:
            # shuffle data
            # 🧠 ML Signal: Forward pass through the model
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            # 🧠 ML Signal: Calculating loss between predictions and labels
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count
    # 🧠 ML Signal: Zeroing gradients before backpropagation

    # 🧠 ML Signal: Model evaluation mode is set, indicating a testing phase
    def get_train_hidden(self, x_train):
        # 🧠 ML Signal: Backpropagation step
        x_train_values = x_train.values
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)
        # ⚠️ SAST Risk (Low): Gradient clipping to prevent exploding gradients
        self.igmtf_model.eval()
        # 🧠 ML Signal: Data is being prepared for batch processing
        train_hidden = []
        # 🧠 ML Signal: Optimizer step to update model parameters
        train_hidden_day = []

        for idx, count in zip(daily_index, daily_count):
            # ⚠️ SAST Risk (Low): Potential for large data to be loaded into memory
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            # ⚠️ SAST Risk (Low): Potential for large data to be loaded into memory
            out = self.igmtf_model(feature, get_hidden=True)
            train_hidden.append(out.detach().cpu())
            # 🧠 ML Signal: Model prediction is being made
            train_hidden_day.append(out.detach().cpu().mean(dim=0).unsqueeze(dim=0))

        # 🧠 ML Signal: Loss calculation for model evaluation
        # ✅ Best Practice: Storing loss values for later analysis
        train_hidden = np.asarray(train_hidden, dtype=object)
        train_hidden_day = torch.cat(train_hidden_day)

        return train_hidden, train_hidden_day
    # 🧠 ML Signal: Metric calculation for model evaluation

    def train_epoch(self, x_train, y_train, train_hidden, train_hidden_day):
        # ✅ Best Practice: Storing score values for later analysis
        # ✅ Best Practice: Returning mean values for losses and scores
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.igmtf_model.train()

        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            pred = self.igmtf_model(feature, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.igmtf_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, train_hidden, train_hidden_day):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        # ⚠️ SAST Risk (Low): Loading model state from a file without validation can lead to code execution risks.
        self.igmtf_model.eval()

        scores = []
        losses = []

        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            pred = self.igmtf_model(feature, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        # 🧠 ML Signal: Use of deepcopy to save model parameters for best epoch.
        save_path=None,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # ⚠️ SAST Risk (Low): Saving model state to a file without validation can lead to code execution risks.
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # ⚠️ SAST Risk (Low): No check for dataset validity or integrity

        # ✅ Best Practice: Clearing GPU cache to free up memory after training.
        save_path = get_or_create_path(save_path)
        stop_steps = 0
        # 🧠 ML Signal: Usage of dataset preparation for training data
        train_loss = 0
        best_score = -np.inf
        # 🧠 ML Signal: Extraction of hidden states from training data
        best_epoch = 0
        evals_result["train"] = []
        # 🧠 ML Signal: Usage of dataset preparation for test data
        evals_result["valid"] = []

        # load pretrained base_model
        # 🧠 ML Signal: Model evaluation mode set before prediction
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        # 🧠 ML Signal: Extraction of daily indices and counts for batching
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        # ⚠️ SAST Risk (Low): Potential device compatibility issues with torch tensors

        # 🧠 ML Signal: Model prediction without gradient computation
        model_dict = self.igmtf_model.state_dict()
        # ✅ Best Practice: Class should inherit from object for Python 2/3 compatibility, but in Python 3, it's optional as all classes implicitly inherit from object.
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        # 🧠 ML Signal: Conditional logic to select model architecture
        model_dict.update(pretrained_dict)
        # ✅ Best Practice: Returning predictions as a pandas Series for easy handling
        # 🧠 ML Signal: Use of GRU model with specific parameters
        self.igmtf_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            # 🧠 ML Signal: Conditional logic to select model architecture
            # 🧠 ML Signal: Use of LSTM model with specific parameters
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            train_hidden, train_hidden_day = self.get_train_hidden(x_train)
            self.train_epoch(x_train, y_train, train_hidden, train_hidden_day)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train, train_hidden, train_hidden_day)
            val_loss, val_score = self.test_epoch(x_valid, y_valid, train_hidden, train_hidden_day)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                # ⚠️ SAST Risk (Low): Potential for unhandled exception if base_model is not recognized
                best_score = val_score
                stop_steps = 0
                # 🧠 ML Signal: Use of sequential model with linear and activation layers
                best_epoch = step
                best_param = copy.deepcopy(self.igmtf_model.state_dict())
            else:
                # 🧠 ML Signal: Adding linear layers in a loop
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # ✅ Best Practice: Method name should be descriptive and use snake_case for readability
                    # 🧠 ML Signal: Adding activation layers in a loop
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Use of matrix multiplication to calculate cosine similarity
        # 🧠 ML Signal: Use of linear layer for output transformation

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 🧠 ML Signal: Normalization of vectors, common in ML for cosine similarity
        # 🧠 ML Signal: Use of linear layers for projection
        self.igmtf_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        # 🧠 ML Signal: Normalization of vectors, common in ML for cosine similarity
        # ✅ Best Practice: Method name is descriptive and follows snake_case naming convention

        # 🧠 ML Signal: Use of linear layer for final prediction
        if self.use_gpu:
            # ⚠️ SAST Risk (Low): Potential division by zero, mitigated by adding a small constant
            # 🧠 ML Signal: Accessing indices of a sparse tensor, common in ML for sparse data operations
            torch.cuda.empty_cache()
    # 🧠 ML Signal: Use of activation function

    # 🧠 ML Signal: Accessing values of a sparse tensor, common in ML for sparse data operations
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        # ✅ Best Practice: Storing input feature dimension for potential future use
        if not self.fitted:
            # ⚠️ SAST Risk (Low): Direct indexing into dense tensor without bounds checking
            raise ValueError("model is not fitted yet!")
        x_train = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        # 🧠 ML Signal: Creating a sparse tensor from indices and values, common in ML for efficiency
        train_hidden, train_hidden_day = self.get_train_hidden(x_train)
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.igmtf_model.eval()
        x_values = x_test.values
        preds = []

        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = (
                    self.igmtf_model(x_batch, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
                    .detach()
                    .cpu()
                    .numpy()
                )

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class IGMTFModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.lins = nn.Sequential()
        for i in range(2):
            self.lins.add_module("linear" + str(i), nn.Linear(hidden_size, hidden_size))
            self.lins.add_module("leakyrelu" + str(i), nn.LeakyReLU())
        self.fc_output = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.project1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.project2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out_pred = nn.Linear(hidden_size * 2, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.d_feat = d_feat

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
        return cos_similarity

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, x, get_hidden=False, train_hidden=None, train_hidden_day=None, k_day=10, n_neighbor=10):
        # x: [N, F*T]
        device = x.device
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.lins(out)
        mini_batch_out = out
        if get_hidden is True:
            return mini_batch_out

        mini_batch_out_day = torch.mean(mini_batch_out, dim=0).unsqueeze(0)
        day_similarity = self.cal_cos_similarity(mini_batch_out_day, train_hidden_day.to(device))
        day_index = torch.topk(day_similarity, k_day, dim=1)[1]
        sample_train_hidden = train_hidden[day_index.long().cpu()].squeeze()
        sample_train_hidden = torch.cat(list(sample_train_hidden)).to(device)
        sample_train_hidden = self.lins(sample_train_hidden)
        cos_similarity = self.cal_cos_similarity(self.project1(mini_batch_out), self.project2(sample_train_hidden))

        row = (
            torch.linspace(0, x.shape[0] - 1, x.shape[0])
            .reshape([-1, 1])
            .repeat(1, n_neighbor)
            .reshape(1, -1)
            .to(device)
        )
        column = torch.topk(cos_similarity, n_neighbor, dim=1)[1].reshape(1, -1)
        mask = torch.sparse_coo_tensor(
            torch.cat([row, column]),
            torch.ones([row.shape[1]]).to(device) / n_neighbor,
            (x.shape[0], sample_train_hidden.shape[0]),
        )
        cos_similarity = self.sparse_dense_mul(mask, cos_similarity)

        agg_out = torch.sparse.mm(cos_similarity, self.project2(sample_train_hidden))
        # out = self.fc_out(out).squeeze()
        out = self.fc_out_pred(torch.cat([mini_batch_out, agg_out], axis=1)).squeeze()
        return out