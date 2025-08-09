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
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
# ✅ Best Practice: Docstring provides clear documentation of class parameters and their types
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


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
        # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
        n_epochs=200,
        lr=0.001,
        metric="",
        # 🧠 ML Signal: Storing model hyperparameters for later use
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("ALSTM")
        # ✅ Best Practice: Normalize optimizer input to lowercase for consistency
        self.logger.info("ALSTM pytorch version...")

        # ⚠️ SAST Risk (Low): Potential GPU index out of range if not validated
        # 🧠 ML Signal: Logging detailed model parameters for debugging and analysis
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
                # 🧠 ML Signal: Initializing the ALSTM model with specified parameters
                loss,
                self.device,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            # 🧠 ML Signal: Logging model size for resource management and analysis
            np.random.seed(self.seed)
            # ✅ Best Practice: Use of conditional logic to select optimizer
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Checking if a GPU is used for computation
        self.ALSTM_model = ALSTMModel(
            # ⚠️ SAST Risk (Low): Potential for incorrect device comparison if `self.device` is not properly initialized
            d_feat=self.d_feat,
            # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in ML models
            hidden_size=self.hidden_size,
            # ✅ Best Practice: Use `torch.device` for device comparison to ensure consistency
            num_layers=self.num_layers,
            # 🧠 ML Signal: Calculation of squared error, a key step in MSE
            dropout=self.dropout,
        # 🧠 ML Signal: Custom loss function implementation
        # ⚠️ SAST Risk (Low): Potential denial of service if unsupported optimizer is used
        )
        # 🧠 ML Signal: Use of torch.mean, indicating integration with PyTorch for tensor operations
        self.logger.info("model:\n{:}".format(self.ALSTM_model))
        # ✅ Best Practice: Use of torch.isnan to handle NaN values in labels
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ALSTM_model)))
        # 🧠 ML Signal: Moving model to the specified device (CPU/GPU)

        # 🧠 ML Signal: Conditional logic based on loss type
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.ALSTM_model.parameters(), lr=self.lr)
        # 🧠 ML Signal: Use of mask to filter predictions and labels
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.ALSTM_model.parameters(), lr=self.lr)
        # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
        # ⚠️ SAST Risk (Low): Potential for self.metric to be an unexpected value leading to ValueError
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # 🧠 ML Signal: Use of a mask to filter predictions and labels for loss calculation

        self.fitted = False
        # ⚠️ SAST Risk (Low): Use of string interpolation in exception message, potential for unexpected metric values
        self.ALSTM_model.to(self.device)
    # 🧠 ML Signal: Usage of model training method

    @property
    def use_gpu(self):
        # 🧠 ML Signal: Shuffling data for training
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)
    # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly

    def loss_fn(self, pred, label):
        # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
        mask = ~torch.isnan(label)

        # 🧠 ML Signal: Model prediction step
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        # 🧠 ML Signal: Loss calculation step

        raise ValueError("unknown loss `%s`" % self.loss)
    # 🧠 ML Signal: Optimizer step preparation

    # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization.
    def metric_fn(self, pred, label):
        # 🧠 ML Signal: Backpropagation step
        mask = torch.isfinite(label)

        # ✅ Best Practice: Gradient clipping to prevent exploding gradients
        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Use of indices for batching indicates a pattern for processing data in chunks.
            return -self.loss_fn(pred[mask], label[mask])
        # 🧠 ML Signal: Optimizer step execution

        raise ValueError("unknown metric `%s`" % self.metric)
    # ✅ Best Practice: Early exit if remaining data is less than batch size.

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        # ⚠️ SAST Risk (Low): Ensure that data conversion to tensors is safe and handles exceptions.
        y_train_values = np.squeeze(y_train.values)

        # ⚠️ SAST Risk (Low): Ensure that data conversion to tensors is safe and handles exceptions.
        self.ALSTM_model.train()

        # ✅ Best Practice: Use torch.no_grad() to prevent tracking history in evaluation mode.
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        # 🧠 ML Signal: Model prediction step, useful for understanding model inference patterns.
        # 🧠 ML Signal: Loss calculation step, important for training and evaluation metrics.

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # 🧠 ML Signal: Metric calculation step, useful for evaluating model performance.
            # ✅ Best Practice: Return average loss and score for better interpretability of results.
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.ALSTM_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.ALSTM_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.ALSTM_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        # ⚠️ SAST Risk (Low): Potential resource leak if GPU memory is not cleared properly
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            # 🧠 ML Signal: Usage of dataset preparation for prediction
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        # 🧠 ML Signal: Model evaluation mode set before prediction
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        # ✅ Best Practice: Use of batch processing for predictions
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        # ⚠️ SAST Risk (Low): Potential device compatibility issue with 'to(self.device)'
        evals_result["train"] = []
        # 🧠 ML Signal: Custom model class definition for PyTorch
        evals_result["valid"] = []

        # 🧠 ML Signal: Use of model prediction with no gradient tracking
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        # train
        self.logger.info("training...")
        # 🧠 ML Signal: Initialization of model parameters like hidden_size and input_size
        self.fitted = True
        # 🧠 ML Signal: Conversion of predictions to pandas Series

        # 🧠 ML Signal: Initialization of model parameters like hidden_size and input_size
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # 🧠 ML Signal: Use of dropout as a regularization technique
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            # 🧠 ML Signal: Use of rnn_type to specify the type of RNN (e.g., GRU, LSTM)
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Setting the number of layers in the RNN
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            # ⚠️ SAST Risk (Low): Catching broad exceptions can mask other issues
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # ✅ Best Practice: Encapsulation of model building logic in a separate method
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            # ✅ Best Practice: Use descriptive module names for clarity

            # 🧠 ML Signal: Use of RNN type and parameters can indicate model architecture
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    # ✅ Best Practice: Use descriptive variable names for clarity
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.ALSTM_model.load_state_dict(best_param)
        # ✅ Best Practice: Use descriptive module names for clarity
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        # 🧠 ML Signal: Use of dropout can indicate regularization techniques
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        # 🧠 ML Signal: Use of view to reshape tensors, common in ML models for data manipulation

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        # 🧠 ML Signal: Use of permute to change tensor dimensions, common in ML models for data manipulation
        index = x_test.index
        self.ALSTM_model.eval()
        # 🧠 ML Signal: Use of RNN layer, indicative of sequence processing in ML models
        x_values = x_test.values
        sample_num = x_values.shape[0]
        # 🧠 ML Signal: Use of attention mechanism, common in advanced ML models
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            # 🧠 ML Signal: Element-wise multiplication of tensors, common in attention mechanisms
            # 🧠 ML Signal: Summing over a specific dimension, common in pooling operations in ML models
            # 🧠 ML Signal: Concatenation of tensors, common in ML models for combining features
            # ✅ Best Practice: Explicitly returning a specific slice of the output tensor
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.ALSTM_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


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
        # inputs: [batch_size, input_size*input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out[..., 0]