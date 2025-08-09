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

# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from .pytorch_utils import count_parameters
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class GATs(Model):
    """GATs Model

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
        lr=0.001,
        metric="",
        # ✅ Best Practice: Consider using a more descriptive logger name for clarity.
        early_stop=20,
        loss="mse",
        base_model="GRU",
        # 🧠 ML Signal: Storing model hyperparameters for later use.
        model_path=None,
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GATs")
        self.logger.info("GATs pytorch version...")
        # 🧠 ML Signal: Normalizing optimizer input to lowercase for consistency.

        # set hyper-parameters.
        self.d_feat = d_feat
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if not validated.
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
            "GATs parameters setting:"
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
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                # ⚠️ SAST Risk (Low): AttributeError if 'use_gpu' is not defined elsewhere.
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                self.device,
                # 🧠 ML Signal: Initializing the GAT model with specified parameters.
                # 🧠 ML Signal: Setting random seed for reproducibility.
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            # 🧠 ML Signal: Using Adam optimizer for training.
            # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating hardware preference
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # ⚠️ SAST Risk (Low): Assumes 'self.device' is a valid torch.device object
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in regression tasks
            dropout=self.dropout,
            # 🧠 ML Signal: Using SGD optimizer for training.
            # 🧠 ML Signal: Returns a boolean indicating GPU usage, useful for model training context
            base_model=self.base_model,
        # 🧠 ML Signal: Calculation of squared error, a key step in mean squared error computation
        )
        # 🧠 ML Signal: Custom loss function implementation
        self.logger.info("model:\n{:}".format(self.GAT_model))
        # ⚠️ SAST Risk (Low): Potential Denial of Service if input is not validated.
        # 🧠 ML Signal: Use of torch.mean, indicating usage of PyTorch for tensor operations
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GAT_model)))
        # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if `torch.isnan` is not used correctly

        # 🧠 ML Signal: Tracking if the model has been fitted.
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Conditional logic based on loss type
            self.train_optimizer = optim.Adam(self.GAT_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Moving model to the specified device (CPU/GPU).
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Use of mask to handle NaN values in labels
            # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values
            self.train_optimizer = optim.SGD(self.GAT_model.parameters(), lr=self.lr)
        else:
            # ⚠️ SAST Risk (Low): Use of string formatting with `%` operator can lead to issues if not properly handled
            # 🧠 ML Signal: Conditional logic based on self.metric value
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # ⚠️ SAST Risk (Low): Potential for negative loss values if self.loss_fn does not handle inputs correctly
        self.fitted = False
        # 🧠 ML Signal: Use of groupby operation on a DataFrame, indicating data aggregation pattern
        self.GAT_model.to(self.device)
    # ⚠️ SAST Risk (Low): Use of string interpolation in exception message, potential for format string vulnerability

    # 🧠 ML Signal: Use of numpy operations for array manipulation
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")
    # ✅ Best Practice: Conditional logic to handle optional shuffling

    def mse(self, pred, label):
        # 🧠 ML Signal: Use of shuffling, indicating data randomization pattern
        loss = (pred - label) ** 2
        return torch.mean(loss)
    # ⚠️ SAST Risk (Low): Use of np.random.shuffle can lead to non-deterministic behavior

    def loss_fn(self, pred, label):
        # 🧠 ML Signal: Model training loop
        mask = ~torch.isnan(label)

        # 🧠 ML Signal: Data shuffling for training
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)
    # 🧠 ML Signal: Conversion of data to PyTorch tensors

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # 🧠 ML Signal: Model prediction

        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Loss calculation
            return -self.loss_fn(pred[mask], label[mask])

        # 🧠 ML Signal: Optimizer gradient reset
        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Backpropagation
    # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization.
    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        # ⚠️ SAST Risk (Low): Potential for exploding gradients if not clipped properly
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        # 🧠 ML Signal: Optimizer step
        # 🧠 ML Signal: Using a method to get daily intervals for batching data.
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            # ⚠️ SAST Risk (Low): Potential for large memory usage if data_x is large.
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        # ⚠️ SAST Risk (Low): Potential for large memory usage if data_y is large.
        return daily_index, daily_count

    # 🧠 ML Signal: Model prediction step.
    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # 🧠 ML Signal: Loss calculation step.
        y_train_values = np.squeeze(y_train.values)
        self.GAT_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        # 🧠 ML Signal: Aggregating results over batches.
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.GAT_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.GAT_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            pred = self.GAT_model(feature)
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
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        # ⚠️ SAST Risk (Low): Check if 'self.fitted' is properly set elsewhere to avoid false negatives.
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        # ✅ Best Practice: Ensure 'dataset.prepare' handles exceptions or errors internally.
        evals_result["valid"] = []

        # load pretrained base_model
        # ✅ Best Practice: Ensure 'self.GAT_model' is properly initialized before calling 'eval()'.
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        # ✅ Best Practice: Consider handling exceptions in 'self.get_daily_inter' for robustness.
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            # ⚠️ SAST Risk (Low): Ensure 'x_values' is sanitized to prevent data integrity issues.
            self.logger.info("Loading pretrained model...")
            # 🧠 ML Signal: Definition of a custom neural network model class
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # ✅ Best Practice: Ensure 'self.GAT_model' output is validated for expected shape and type.
        model_dict = self.GAT_model.state_dict()
        # 🧠 ML Signal: Conditional logic to select model architecture
        pretrained_dict = {
            # ✅ Best Practice: Validate 'index' and 'preds' lengths match before creating the Series.
            # 🧠 ML Signal: Use of GRU for sequence modeling
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        # 🧠 ML Signal: Conditional logic to select model architecture
        # 🧠 ML Signal: Use of LSTM for sequence modeling
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
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if base_model is invalid

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                # 🧠 ML Signal: Use of linear transformation layer
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            # 🧠 ML Signal: Use of learnable parameter for attention mechanism
            else:
                # 🧠 ML Signal: Use of transformation function on input data
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Use of transformation function on input data
                    # 🧠 ML Signal: Use of fully connected layers for output transformation
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Use of tensor shape to determine sample number

        # 🧠 ML Signal: Use of activation function for non-linearity
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 🧠 ML Signal: Use of tensor shape to determine dimensionality
        self.GAT_model.load_state_dict(best_param)
        # 🧠 ML Signal: Use of softmax for probability distribution
        torch.save(best_param, save_path)
        # 🧠 ML Signal: Use of tensor expansion for attention mechanism

        if self.use_gpu:
            # 🧠 ML Signal: Use of tensor transposition for attention mechanism
            torch.cuda.empty_cache()

    # 🧠 ML Signal: Concatenation of tensors for attention input
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        # 🧠 ML Signal: Reshaping input data for model processing
        if not self.fitted:
            # ⚠️ SAST Risk (Low): Potential misuse of tensor transpose without checking dimensions
            raise ValueError("model is not fitted yet!")
        # 🧠 ML Signal: Permuting tensor dimensions for RNN input

        # 🧠 ML Signal: Matrix multiplication for attention score calculation
        x_test = dataset.prepare(segment, col_set="feature")
        # 🧠 ML Signal: Using RNN to process sequential data
        index = x_test.index
        # 🧠 ML Signal: Use of activation function in attention mechanism
        self.GAT_model.eval()
        # 🧠 ML Signal: Extracting the last hidden state from RNN output
        x_values = x_test.values
        # 🧠 ML Signal: Use of softmax for attention weight calculation
        preds = []
        # 🧠 ML Signal: Return of attention weights
        # 🧠 ML Signal: Calculating attention weights
        # 🧠 ML Signal: Applying attention mechanism to hidden state
        # 🧠 ML Signal: Passing data through a fully connected layer
        # 🧠 ML Signal: Applying activation function
        # 🧠 ML Signal: Final output layer with squeeze operation

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class GATModel(nn.Module):
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

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()