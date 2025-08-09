# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import numpy as np
import pandas as pd
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.nn as nn
import torch.nn.init as init
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# ✅ Best Practice: Class names should follow the CapWords convention for readability
import torch.optim as optim
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class SFM_Model(nn.Module):
    def __init__(
        self,
        d_feat=6,
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        device="cpu",
    ):
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        super().__init__()

        self.input_dim = d_feat
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.hidden_dim = hidden_size
        self.device = device

        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        # 🧠 ML Signal: Reshaping input data is a common preprocessing step in ML models

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        # 🧠 ML Signal: Permuting dimensions is often used in sequence models
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        # 🧠 ML Signal: Use of Xavier and orthogonal initialization for weights
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        # 🧠 ML Signal: Initializing states is typical in RNNs and LSTMs
        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        # ✅ Best Practice: Use of nn.Linear for defining fully connected layers
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))
        # 🧠 ML Signal: Getting constants is a pattern in recurrent models

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 1)

        self.states = []

    def forward(self, input):
        input = input.reshape(len(input), self.input_dim, -1)  # [N, F, T]
        # 🧠 ML Signal: Matrix multiplication is a core operation in neural networks
        input = input.permute(0, 2, 1)  # [N, T, F]
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                # 🧠 ML Signal: Activation functions are key components in neural networks
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]  # noqa: F841
            h_tm1 = self.states[1]
            # ✅ Best Practice: Reshaping tensors for compatibility in operations
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]
            # 🧠 ML Signal: Use of trigonometric functions in signal processing models

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            # ✅ Best Practice: Reshaping tensors for compatibility in operations
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            # ✅ Best Practice: Reshaping tensors for compatibility in operations
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            # ✅ Best Practice: Reshaping tensors for compatibility in operations
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            # 🧠 ML Signal: Updating states is a common pattern in recurrent models
            time = time_tm1 + 1

            # ✅ Best Practice: Clearing states after processing
            omega = torch.tensor(2 * np.pi) * time * frequency

            # 🧠 ML Signal: Fully connected layers are common in neural network outputs
            # 🧠 ML Signal: Storing initial states in a list for a model, indicating a pattern for stateful computations
            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            # ✅ Best Practice: Using list comprehension for concise and efficient list creation
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)
            # ✅ Best Practice: Using list comprehension for concise and efficient list creation

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))
            # ✅ Best Practice: Using numpy for efficient numerical operations

            # 🧠 ML Signal: Defines a machine learning model class with parameters, useful for model architecture learning
            h = o * a
            # ✅ Best Practice: Converting numpy array to torch tensor for compatibility with PyTorch operations
            # ⚠️ SAST Risk (Low): Potential risk if self.states is not properly initialized or if index 5 is out of bounds
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]
    # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns

    def get_constants(self, x):
        constants = []
        # 🧠 ML Signal: Model configuration parameters are set as instance variables
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants


class SFM(Model):
    """SFM Model

    Parameters
    ----------
    input_dim : int
        input dimension
    output_dim : int
        output dimension
    lr : float
        learning rate
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        output_dim=1,
        freq_dim=10,
        dropout_W=0.0,
        dropout_U=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        eval_steps=5,
        loss="mse",
        optimizer="gd",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("SFM")
        self.logger.info("SFM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        # ⚠️ SAST Risk (Low): Seed setting for reproducibility, but may not cover all sources of randomness
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        # 🧠 ML Signal: Instantiation of the model with configuration parameters
        self.seed = seed

        self.logger.info(
            "SFM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\noutput_size : {}"
            # 🧠 ML Signal: Logging model architecture and size for analysis
            "\nfrequency_dimension : {}"
            "\ndropout_W: {}"
            # ✅ Best Practice: Use of conditional logic to select optimizer
            "\ndropout_U: {}"
            "\nn_epochs : {}"
            # 🧠 ML Signal: Checking if a GPU is being used for computation
            "\nlr : {}"
            "\nmetric : {}"
            # ✅ Best Practice: Using torch.device to handle device types
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\neval_steps : {}"
            # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization
            # ⚠️ SAST Risk (Low): Use of NotImplementedError to handle unsupported optimizers
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            # 🧠 ML Signal: Model is moved to the specified device (CPU/GPU)
            "\nuse_GPU : {}"
            # 🧠 ML Signal: Use of indices for batching indicates a custom batching mechanism
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                output_dim,
                freq_dim,
                # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
                dropout_W,
                dropout_U,
                # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
                n_epochs,
                lr,
                # 🧠 ML Signal: Model prediction step
                metric,
                batch_size,
                # 🧠 ML Signal: Loss calculation step
                early_stop,
                # 🧠 ML Signal: Use of .values to extract numpy arrays from pandas DataFrames
                eval_steps,
                # ✅ Best Practice: Use .item() to convert a single-valued tensor to a Python number
                optimizer.lower(),
                # 🧠 ML Signal: Use of np.squeeze to remove single-dimensional entries from the shape of an array
                loss,
                # 🧠 ML Signal: Metric calculation step
                self.device,
                # 🧠 ML Signal: Setting model to training mode
                self.use_gpu,
                # ✅ Best Practice: Use .item() to convert a single-valued tensor to a Python number
                seed,
            # 🧠 ML Signal: Use of np.arange to create an array of indices
            )
        # ✅ Best Practice: Return the mean of losses and scores for better interpretability
        )
        # 🧠 ML Signal: Shuffling data indices for stochastic gradient descent

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Conversion of numpy arrays to PyTorch tensors and moving to device
        self.sfm_model = SFM_Model(
            d_feat=self.d_feat,
            # 🧠 ML Signal: Conversion of numpy arrays to PyTorch tensors and moving to device
            output_dim=self.output_dim,
            # 🧠 ML Signal: Forward pass through the model
            # 🧠 ML Signal: Calculation of loss using a loss function
            hidden_size=self.hidden_size,
            freq_dim=self.freq_dim,
            dropout_W=self.dropout_W,
            dropout_U=self.dropout_U,
            device=self.device,
        )
        # 🧠 ML Signal: Zeroing gradients before backward pass
        # 🧠 ML Signal: Backward pass for gradient computation
        # ⚠️ SAST Risk (Low): Potential mutable default argument for evals_result
        self.logger.info("model:\n{:}".format(self.sfm_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.sfm_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.sfm_model.parameters(), lr=self.lr)
        # ⚠️ SAST Risk (Low): Potential for gradient explosion without proper clipping
        # 🧠 ML Signal: Optimizer step to update model parameters
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.sfm_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.sfm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.sfm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.sfm_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())
        # ⚠️ SAST Risk (Low): Potential resource management issue if not using 'cpu'

        # 🧠 ML Signal: Function for calculating mean squared error, a common loss function in regression tasks
        return np.mean(losses), np.mean(scores)

    # ✅ Best Practice: Use of descriptive variable names like 'pred' and 'label' for clarity
    def train_epoch(self, x_train, y_train):
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        x_train_values = x_train.values
        # ⚠️ SAST Risk (Low): Assumes 'pred' and 'label' are tensors; lacks input validation
        y_train_values = np.squeeze(y_train.values)
        # 🧠 ML Signal: Usage of torch.isnan to handle missing values in labels

        # 🧠 ML Signal: Use of torch.mean, indicating integration with PyTorch for tensor operations
        self.sfm_model.train()
        # 🧠 ML Signal: Conditional logic based on self.loss attribute

        # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
        indices = np.arange(len(x_train_values))
        # 🧠 ML Signal: Usage of mask to filter predictions and labels
        np.random.shuffle(indices)
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid (finite) values in the label tensor.

        # ⚠️ SAST Risk (Low): Potential for unhandled exception if self.loss is not "mse"
        for i in range(len(indices))[:: self.batch_size]:
            # 🧠 ML Signal: Conditional logic based on the value of self.metric, indicating dynamic behavior based on configuration.
            if len(indices) - i < self.batch_size:
                break
            # ⚠️ SAST Risk (Low): Potential risk if self.loss_fn is not properly validated or sanitized, leading to unexpected behavior.
            # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # ⚠️ SAST Risk (Low): Use of string formatting with user-controlled input in exception message, though risk is minimal here.
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            # 🧠 ML Signal: Usage of dataset preparation for prediction

            pred = self.sfm_model(feature)
            loss = self.loss_fn(pred, label)
            # 🧠 ML Signal: Model evaluation mode set before prediction

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.sfm_model.parameters(), 3.0)
            # ✅ Best Practice: Use of range with step for batch processing
            self.train_optimizer.step()

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    # ⚠️ SAST Risk (Low): Potential device mismatch if 'self.device' is not set correctly
    ):
        df_train, df_valid = dataset.prepare(
            # ✅ Best Practice: Use of 'torch.no_grad()' for inference to save memory
            ["train", "valid"],
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose
            # ✅ Best Practice: Use of a constructor method to initialize an object
            col_set=["feature", "label"],
            # 🧠 ML Signal: Model prediction and conversion to numpy
            data_key=DataHandlerLP.DK_L,
        # ✅ Best Practice: Encapsulating initialization logic in a separate method
        )
        # ✅ Best Practice: Initializing or resetting instance variables to default values
        if df_train.empty or df_valid.empty:
            # 🧠 ML Signal: Returning predictions as a pandas Series
            raise ValueError("Empty data from dataset, please check your dataset config.")
        # ✅ Best Practice: Initializing or resetting instance variables to default values
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # ✅ Best Practice: Initializing or resetting instance variables to default values

        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        save_path = get_or_create_path(save_path)
        # ✅ Best Practice: Initializing or resetting instance variables to default values
        stop_steps = 0
        # 🧠 ML Signal: Tracking cumulative sum and count for average calculation
        train_loss = 0
        # 🧠 ML Signal: Incremental update pattern for count
        # ⚠️ SAST Risk (Low): Potential division by zero if self.count is zero
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
                best_param = copy.deepcopy(self.sfm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.sfm_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.sfm_model.eval()
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
                pred = self.sfm_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count