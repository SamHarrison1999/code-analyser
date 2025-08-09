# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

import numpy as np
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

# 🧠 ML Signal: Definition of a class likely used for machine learning model architecture
import torch
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

########################################################################
########################################################################
########################################################################

# ✅ Best Practice: Call the superclass's __init__ method to ensure proper initialization

class CNNEncoderBase(nn.Module):
    # 🧠 ML Signal: Storing input dimension as an instance variable
    def __init__(self, input_dim, output_dim, kernel_size, device):
        """Build a basic CNN encoder

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        kernel_size : int
            The size of convolutional kernels
        """
        super().__init__()

        self.input_dim = input_dim
        # 🧠 ML Signal: Reshaping and permuting tensors is common in neural network layers
        self.output_dim = output_dim
        # ⚠️ SAST Risk (Low): Ensure x.shape[0] and self.input_dim are valid to prevent runtime errors
        self.kernel_size = kernel_size
        self.device = device
        # 🧠 ML Signal: Applying convolution operation on tensors

        # 🧠 ML Signal: Custom neural network module definition
        # set padding to ensure the same length
        # 🧠 ML Signal: Permuting tensor dimensions is a common operation in neural networks
        # it is correct only when kernel_size is odd, dilation is 1, stride is 1
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            Updated representations
        """

        # input shape: [batch_size, seq_len*input_dim]
        # output shape: [batch_size, seq_len, input_dim]
        x = x.view(x.shape[0], -1, self.input_dim).permute(0, 2, 1).to(self.device)
        y = self.conv(x)  # [batch_size, output_dim, conved_seq_len]
        # ✅ Best Practice: Use of nn.ModuleList to store a list of modules is a good practice for managing multiple RNNs.
        y = y.permute(0, 2, 1)  # [batch_size, conved_seq_len, output_dim]

        return y
# 🧠 ML Signal: Pattern of creating multiple RNNs with the same architecture.

# ✅ Best Practice: Appending RNN modules to a ModuleList allows for easy management and iteration.

class KRNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, dup_num, rnn_layers, dropout, device):
        """Build K parallel RNNs

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        dup_num : int
            The number of parallel RNNs
        rnn_layers: int
            The number of RNN layers
        """
        # 🧠 ML Signal: Iterating over RNN modules indicates a sequence processing pattern.
        super().__init__()

        # 🧠 ML Signal: Use of RNNs suggests a temporal or sequential data processing task.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dup_num = dup_num
        # 🧠 ML Signal: Stacking hidden states indicates aggregation of multiple RNN outputs.
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        # ✅ Best Practice: Reshaping tensors for further processing is a common pattern in deep learning.
        # 🧠 ML Signal: Custom neural network module definition
        self.device = device
        # 🧠 ML Signal: Taking the mean across a dimension is a common pattern for reducing or aggregating features.

        self.rnn_modules = nn.ModuleList()
        for _ in range(dup_num):
            # ✅ Best Practice: Permuting dimensions back to the original order for consistency in output.
            # ✅ Best Practice: Docstring provides clear documentation of parameters and purpose
            self.rnn_modules.append(nn.GRU(input_dim, output_dim, num_layers=self.rnn_layers, dropout=dropout))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        n_id : torch.Tensor
            Node indices

        Returns
        -------
        torch.Tensor
            Updated representations
        """

        # ✅ Best Practice: Calling super().__init__() ensures proper initialization of the base class
        # input shape: [batch_size, seq_len, input_dim]
        # output shape: [batch_size, seq_len, output_dim]
        # 🧠 ML Signal: Usage of CNN and RNN components indicates a deep learning model
        # [seq_len, batch_size, input_dim]
        # 🧠 ML Signal: Usage of CNN and RNN components indicates a deep learning model
        batch_size, seq_len, input_dim = x.shape
        x = x.permute(1, 0, 2).to(self.device)

        hids = []
        for rnn in self.rnn_modules:
            h, _ = rnn(x)  # [seq_len, batch_size, output_dim]
            hids.append(h)
        # [seq_len, batch_size, output_dim, num_dups]
        hids = torch.stack(hids, dim=-1)
        hids = hids.view(seq_len, batch_size, self.output_dim, self.dup_num)
        hids = hids.mean(dim=3)
        hids = hids.permute(1, 0, 2)
        # ✅ Best Practice: Use of clear and descriptive variable names improves code readability.

        return hids
# 🧠 ML Signal: Sequential processing of data through multiple layers is common in neural networks.

# 🧠 ML Signal: Custom neural network model class definition

# 🧠 ML Signal: Returning the output of a neural network layer is a common pattern in model definitions.
class CNNKRNNEncoder(nn.Module):
    # ✅ Best Practice: Docstring provides clear documentation of parameters and purpose
    def __init__(
        self, cnn_input_dim, cnn_output_dim, cnn_kernel_size, rnn_output_dim, rnn_dup_num, rnn_layers, dropout, device
    ):
        """Build an encoder composed of CNN and KRNN

        Parameters
        ----------
        cnn_input_dim : int
            The input dimension of CNN
        cnn_output_dim : int
            The output dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_output_dim : int
            The output dimension of KRNN
        rnn_dup_num : int
            The number of parallel duplicates for KRNN
        rnn_layers : int
            The number of RNN layers
        """
        super().__init__()

        self.cnn_encoder = CNNEncoderBase(cnn_input_dim, cnn_output_dim, cnn_kernel_size, device)
        self.krnn_encoder = KRNNEncoderBase(cnn_output_dim, rnn_output_dim, rnn_dup_num, rnn_layers, dropout, device)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        n_id : torch.Tensor
            Node indices

        Returns
        -------
        torch.Tensor
            Updated representations
        """
        cnn_out = self.cnn_encoder(x)
        krnn_out = self.krnn_encoder(cnn_out)

        return krnn_out


class KRNNModel(nn.Module):
    def __init__(self, fea_dim, cnn_dim, cnn_kernel_size, rnn_dim, rnn_dups, rnn_layers, dropout, device, **params):
        """Build a KRNN model

        Parameters
        ----------
        fea_dim : int
            The feature dimension
        cnn_dim : int
            The hidden dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_dim : int
            The hidden dimension of KRNN
        rnn_dups : int
            The number of parallel duplicates
        rnn_layers: int
            The number of RNN layers
        """
        super().__init__()

        # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
        self.encoder = CNNKRNNEncoder(
            cnn_input_dim=fea_dim,
            # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
            cnn_output_dim=cnn_dim,
            cnn_kernel_size=cnn_kernel_size,
            # ✅ Best Practice: Store parameters as instance variables for easy access and modification
            rnn_output_dim=rnn_dim,
            rnn_dup_num=rnn_dups,
            # ✅ Best Practice: Store parameters as instance variables for easy access and modification
            rnn_layers=rnn_layers,
            dropout=dropout,
            # ✅ Best Practice: Store parameters as instance variables for easy access and modification
            device=device,
        )
        # ✅ Best Practice: Store parameters as instance variables for easy access and modification

        self.out_fc = nn.Linear(rnn_dim, 1)
        # ✅ Best Practice: Store parameters as instance variables for easy access and modification
        self.device = device

    # ✅ Best Practice: Store parameters as instance variables for easy access and modification
    def forward(self, x):
        # x: [batch_size, node_num, seq_len, input_dim]
        # ✅ Best Practice: Store parameters as instance variables for easy access and modification
        encode = self.encoder(x)
        out = self.out_fc(encode[:, -1, :]).squeeze().to(self.device)
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # 🧠 ML Signal: Logging initialization and parameters can be used to understand model configuration patterns
        # ✅ Best Practice: Store parameters as instance variables for easy access and modification
        # ✅ Best Practice: Normalize optimizer name to lowercase for consistency

        return out


class KRNN(Model):
    """KRNN Model

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
        fea_dim=6,
        cnn_dim=64,
        cnn_kernel_size=3,
        rnn_dim=64,
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
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("KRNN")
        self.logger.info("KRNN pytorch version...")

        # set hyper-parameters.
        self.fea_dim = fea_dim
        self.cnn_dim = cnn_dim
        self.cnn_kernel_size = cnn_kernel_size
        self.rnn_dim = rnn_dim
        self.rnn_dups = rnn_dups
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        # ✅ Best Practice: Set random seed for reproducibility
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating hardware preference

        # ✅ Best Practice: Initialize model with parameters for modularity and flexibility
        self.logger.info(
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            # ⚠️ SAST Risk (Low): Assumes 'self.device' is a valid torch.device object
            "KRNN parameters setting:"
            "\nfea_dim : {}"
            # 🧠 ML Signal: Use of mean squared error (MSE) for loss calculation
            "\ncnn_dim : {}"
            # 🧠 ML Signal: Custom loss function implementation
            "\ncnn_kernel_size : {}"
            # ⚠️ SAST Risk (Low): Ensure 'torch' is imported and available in the scope
            "\nrnn_dim : {}"
            # ✅ Best Practice: Use of mask to handle NaN values in labels
            "\nrnn_dups : {}"
            "\nrnn_layers : {}"
            "\ndropout : {}"
            # 🧠 ML Signal: Use of mean squared error for loss calculation
            "\nn_epochs : {}"
            # ✅ Best Practice: Check for finite values to avoid computation errors with invalid data
            "\nlr : {}"
            # ✅ Best Practice: Use conditional logic to select optimizer for flexibility
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if self.loss is not "mse"
            "\nmetric : {}"
            # 🧠 ML Signal: Conditional logic based on metric type indicates model evaluation behavior
            "\nbatch_size: {}"
            "\nearly_stop : {}"
            # 🧠 ML Signal: Use of loss function suggests model training or evaluation context
            # ✅ Best Practice: Consider adding type hints for function parameters and return types for better readability and maintainability.
            "\noptimizer : {}"
            "\nloss_type : {}"
            # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
            # 🧠 ML Signal: Usage of groupby operation on a DataFrame, which is common in data preprocessing tasks.
            "\nvisible_GPU : {}"
            # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
            "\nuse_GPU : {}"
            # 🧠 ML Signal: Use of numpy operations for efficient numerical computations.
            "\nseed : {}".format(
                # ✅ Best Practice: Initialize state variables for tracking model status
                fea_dim,
                cnn_dim,
                # ✅ Best Practice: Move model to the appropriate device for computation
                # ✅ Best Practice: Use of a conditional to control the shuffling behavior, enhancing function flexibility.
                cnn_kernel_size,
                rnn_dim,
                # 🧠 ML Signal: Shuffling data, which is a common practice in preparing datasets for machine learning.
                rnn_dups,
                rnn_layers,
                # ⚠️ SAST Risk (Low): Use of np.random.shuffle can lead to non-deterministic results, which might affect reproducibility.
                dropout,
                # 🧠 ML Signal: Model training loop
                n_epochs,
                lr,
                # ✅ Best Practice: Returning multiple values as a tuple, which is a clear and concise way to return related data.
                metric,
                # 🧠 ML Signal: Data shuffling for training
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
                self.use_gpu,
                seed,
            # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
            )
        )
        # 🧠 ML Signal: Model prediction

        if self.seed is not None:
            # 🧠 ML Signal: Loss computation
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 🧠 ML Signal: Backpropagation
        # 🧠 ML Signal: Model evaluation mode is set, indicating a testing phase
        self.krnn_model = KRNNModel(
            fea_dim=self.fea_dim,
            # ✅ Best Practice: Gradient clipping to prevent exploding gradients
            cnn_dim=self.cnn_dim,
            cnn_kernel_size=self.cnn_kernel_size,
            # 🧠 ML Signal: Optimizer step
            # 🧠 ML Signal: Use of indices for batching, common in ML data processing
            rnn_dim=self.rnn_dim,
            rnn_dups=self.rnn_dups,
            # 🧠 ML Signal: Iterating over data in batches, typical in ML model testing
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            # ✅ Best Practice: Early exit for loop if remaining data is less than batch size
            device=self.device,
        )
        if optimizer.lower() == "adam":
            # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly
            self.train_optimizer = optim.Adam(self.krnn_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # ⚠️ SAST Risk (Low): Potential for device mismatch if `self.device` is not set correctly
            self.train_optimizer = optim.SGD(self.krnn_model.parameters(), lr=self.lr)
        else:
            # 🧠 ML Signal: Model prediction step, key part of testing phase
            # 🧠 ML Signal: Loss calculation, essential for evaluating model performance
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.krnn_model.to(self.device)
    # 🧠 ML Signal: Collecting loss values for analysis

    @property
    # 🧠 ML Signal: Metric calculation, important for model evaluation
    # 🧠 ML Signal: Collecting score values for analysis
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        # 🧠 ML Signal: Returning average loss and score, common in model evaluation
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

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.krnn_model.train()

        indices = np.arange(len(x_train_values))
        # ⚠️ SAST Risk (Low): Potential resource leak if GPU memory is not cleared properly
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
            if len(indices) - i < self.batch_size:
                break

            # 🧠 ML Signal: Usage of dataset preparation for prediction
            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            # 🧠 ML Signal: Model evaluation mode set before prediction
            pred = self.krnn_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            # ✅ Best Practice: Use of batch processing for predictions
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.krnn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        # ⚠️ SAST Risk (Low): Potential device mismatch if 'self.device' is not correctly set
        # 🧠 ML Signal: Use of model prediction with no gradient tracking
        # 🧠 ML Signal: Conversion of predictions to pandas Series
        y_values = np.squeeze(data_y.values)

        self.krnn_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.krnn_model(feature)
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
                best_param = copy.deepcopy(self.krnn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.krnn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.krnn_model.eval()
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
                pred = self.krnn_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)