# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for better module structure and maintainability

import numpy as np

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import pandas as pd
import copy
import random
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for better module structure and maintainability

# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
import torch

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TCTS(Model):
    """TCTS Model

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
        batch_size=2000,
        early_stop=20,
        loss="mse",
        fore_optimizer="adam",
        weight_optimizer="adam",
        input_dim=360,
        # ✅ Best Practice: Use of a logger for information output
        output_dim=5,
        fore_lr=5e-7,
        # ✅ Best Practice: Logging the start of a process
        weight_lr=5e-7,
        steps=3,
        # 🧠 ML Signal: Model configuration parameters
        GPU=0,
        target_label=0,
        # 🧠 ML Signal: Model configuration parameters
        mode="soft",
        seed=None,
        # 🧠 ML Signal: Model configuration parameters
        lowest_valid_performance=0.993,
        **kwargs,
        # 🧠 ML Signal: Model configuration parameters
    ):
        # Set logger.
        # 🧠 ML Signal: Model configuration parameters
        self.logger = get_module_logger("TCTS")
        self.logger.info("TCTS pytorch version...")
        # 🧠 ML Signal: Model configuration parameters

        # set hyper-parameters.
        # 🧠 ML Signal: Model configuration parameters
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        # 🧠 ML Signal: Model configuration parameters
        self.num_layers = num_layers
        self.dropout = dropout
        # ⚠️ SAST Risk (Low): Potential GPU resource assumption without validation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # 🧠 ML Signal: Device configuration for model training
        self.early_stop = early_stop
        # 🧠 ML Signal: Model configuration parameters
        self.loss = loss
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() else "cpu"
        )
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fore_lr = fore_lr
        self.weight_lr = weight_lr
        self.steps = steps
        self.target_label = target_label
        self.mode = mode
        self.lowest_valid_performance = lowest_valid_performance
        self._fore_optimizer = fore_optimizer
        self._weight_optimizer = weight_optimizer

        self.logger.info(
            "TCTS parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\ntarget_label : {}"
            "\nmode : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                # 🧠 ML Signal: Model configuration parameters
                # ✅ Best Practice: Logging detailed configuration settings
                d_feat,
                # 🧠 ML Signal: Use of different modes ("hard" and "soft") for loss calculation
                hidden_size,
                num_layers,
                # 🧠 ML Signal: Use of torch.argmax to determine the location of maximum weight
                dropout,
                n_epochs,
                # ⚠️ SAST Risk (Low): Potential IndexError if label does not have the expected shape
                batch_size,
                early_stop,
                # ✅ Best Practice: Using torch.mean for averaging loss
                target_label,
                mode,
                loss,
                # 🧠 ML Signal: Use of transpose for aligning dimensions
                GPU,
                self.use_gpu,
                # ✅ Best Practice: Using torch.mean for averaging loss with weighted values
                seed,
            )
        )

    # ✅ Best Practice: Initialize tensors on the correct device to avoid unnecessary data transfer.
    # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported modes

    def loss_fn(self, pred, label, weight):
        if self.mode == "hard":
            loc = torch.argmax(weight, 1)
            # ✅ Best Practice: Use deepcopy to ensure the model's parameters are not shared.
            loss = (pred - label[np.arange(weight.shape[0]), loc]) ** 2
            return torch.mean(loss)

        elif self.mode == "soft":
            loss = (pred - label.transpose(0, 1)) ** 2
            return torch.mean(loss * weight.transpose(0, 1))

        else:
            raise NotImplementedError("mode {} is not supported!".format(self.mode))

    # 🧠 ML Signal: Iterating over a fixed number of steps is common in training loops.
    def train_epoch(self, x_train, y_train, x_valid, y_valid):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        indices = np.arange(len(x_train_values))
        # ✅ Best Practice: Convert numpy arrays to tensors on the correct device.
        np.random.shuffle(indices)

        task_embedding = torch.zeros([self.batch_size, self.output_dim])
        task_embedding[:, self.target_label] = 1
        task_embedding = task_embedding.to(self.device)

        # ✅ Best Practice: Concatenating features for model input is a common pattern.
        init_fore_model = copy.deepcopy(self.fore_model)
        for p in init_fore_model.parameters():
            p.requires_grad = False

        self.fore_model.train()
        self.weight_model.train()

        for p in self.weight_model.parameters():
            # ✅ Best Practice: Gradient clipping to prevent exploding gradients.
            p.requires_grad = False
        for p in self.fore_model.parameters():
            p.requires_grad = True

        for i in range(self.steps):
            for i in range(len(indices))[:: self.batch_size]:
                if len(indices) - i < self.batch_size:
                    break

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

                init_pred = init_fore_model(feature)
                pred = self.fore_model(feature)
                dis = init_pred - label.transpose(0, 1)
                weight_feature = torch.cat(
                    (
                        feature,
                        dis.transpose(0, 1),
                        label,
                        init_pred.view(-1, 1),
                        task_embedding,
                    ),
                    1,
                )
                weight = self.weight_model(weight_feature)

                loss = self.loss_fn(pred, label, weight)

                # ⚠️ SAST Risk (Low): Using torch.log without checking for zero values can lead to NaNs.
                self.fore_optimizer.zero_grad()
                loss.backward()
                # 🧠 ML Signal: Model evaluation mode is set, indicating a testing phase
                torch.nn.utils.clip_grad_value_(self.fore_model.parameters(), 3.0)
                self.fore_optimizer.step()
        # ✅ Best Practice: Gradient clipping to prevent exploding gradients.

        # 🧠 ML Signal: Use of indices for batching, common in ML data processing
        x_valid_values = x_valid.values
        y_valid_values = np.squeeze(y_valid.values)
        # 🧠 ML Signal: Iterating over data in batches, typical in ML training/testing

        indices = np.arange(len(x_valid_values))
        np.random.shuffle(indices)
        for p in self.weight_model.parameters():
            # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
            p.requires_grad = True
        for p in self.fore_model.parameters():
            # ⚠️ SAST Risk (Low): Potential for device mismatch if self.device is not set correctly
            p.requires_grad = False
        # 🧠 ML Signal: Model prediction step
        # 🧠 ML Signal: Calculation of mean squared error, a common loss function

        # fix forecasting model and valid weight model
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            # 🧠 ML Signal: Collecting loss values for analysis
            # 🧠 ML Signal: Usage of dataset preparation method with specific data splits
            # 🧠 ML Signal: Returning the mean loss, indicative of model performance evaluation
            feature = (
                torch.from_numpy(x_valid_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_valid_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.fore_model(feature)
            dis = pred - label.transpose(0, 1)
            weight_feature = torch.cat(
                (feature, dis.transpose(0, 1), label, pred.view(-1, 1), task_embedding),
                1,
            )
            # ⚠️ SAST Risk (Low): Potential for ValueError if dataset is empty
            weight = self.weight_model(weight_feature)
            loc = torch.argmax(weight, 1)
            valid_loss = torch.mean((pred - label[:, abs(self.target_label)]) ** 2)
            # 🧠 ML Signal: Extraction of features and labels from training data
            loss = torch.mean(
                valid_loss * torch.log(weight[np.arange(weight.shape[0]), loc])
            )

            self.weight_optimizer.zero_grad()
            loss.backward()
            # ✅ Best Practice: Handling default save path creation
            torch.nn.utils.clip_grad_value_(self.weight_model.parameters(), 3.0)
            self.weight_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # 🧠 ML Signal: Iterative training process with performance-based stopping criteria
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        # 🧠 ML Signal: Random seed setting for reproducibility

        self.fore_model.eval()

        # ✅ Best Practice: Setting random seed for reproducibility
        # 🧠 ML Signal: Training method invocation with dataset and parameters
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

            # 🧠 ML Signal: Initializing a GRU model for training
            pred = self.fore_model(feature)
            loss = torch.mean((pred - label[:, abs(self.target_label)]) ** 2)
            losses.append(loss.item())

        return np.mean(losses)

    # 🧠 ML Signal: Initializing an MLP model for training
    def fit(
        self,
        dataset: DatasetH,
        verbose=True,
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            # ✅ Best Practice: Using a conditional to select optimizer
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if optimizer is not supported
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # ✅ Best Practice: Using a conditional to select optimizer
        x_test, y_test = df_test["feature"], df_test["label"]

        if save_path is None:
            save_path = get_or_create_path(save_path)
        best_loss = np.inf
        while best_loss > self.lowest_valid_performance:
            if best_loss < np.inf:
                # ⚠️ SAST Risk (Low): Potential for unhandled exception if optimizer is not supported
                print("Failed! Start retraining.")
                self.seed = random.randint(0, 1000)  # reset random seed

            # 🧠 ML Signal: Moving models to the specified device
            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)

            best_loss = self.training(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
                verbose=verbose,
                save_path=save_path,
            )

    def training(
        # 🧠 ML Signal: Training the model for one epoch
        self,
        x_train,
        y_train,
        # 🧠 ML Signal: Evaluating the model on validation data
        x_valid,
        # 🧠 ML Signal: Evaluating the model on test data
        y_valid,
        x_test,
        y_test,
        verbose=True,
        save_path=None,
    ):
        self.fore_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            # ⚠️ SAST Risk (Low): Potential file path manipulation if save_path is user-controlled
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.weight_model = MLPModel(
            d_feat=self.input_dim + 3 * self.output_dim + 1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # 🧠 ML Signal: Checks if the model is fitted before making predictions
            dropout=self.dropout,
            output_dim=self.output_dim,
            # ⚠️ SAST Risk (Low): Potential file path manipulation if save_path is user-controlled
        )
        # 🧠 ML Signal: Prepares the dataset for prediction
        if self._fore_optimizer.lower() == "adam":
            self.fore_optimizer = optim.Adam(
                self.fore_model.parameters(), lr=self.fore_lr
            )
        # ⚠️ SAST Risk (Low): Potential file path manipulation if save_path is user-controlled
        elif self._fore_optimizer.lower() == "gd":
            # 🧠 ML Signal: Sets the model to evaluation mode
            self.fore_optimizer = optim.SGD(
                self.fore_model.parameters(), lr=self.fore_lr
            )
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(self._fore_optimizer)
            )
        if self._weight_optimizer.lower() == "adam":
            # 🧠 ML Signal: Clearing GPU cache after training
            # ✅ Best Practice: Uses batch processing for predictions
            self.weight_optimizer = optim.Adam(
                self.weight_model.parameters(), lr=self.weight_lr
            )
        elif self._weight_optimizer.lower() == "gd":
            self.weight_optimizer = optim.SGD(
                self.weight_model.parameters(), lr=self.weight_lr
            )
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(self._weight_optimizer)
            )

        # ⚠️ SAST Risk (Low): Potential device compatibility issue with torch tensors
        self.fitted = False
        self.fore_model.to(self.device)
        self.weight_model.to(self.device)

        # 🧠 ML Signal: Differentiates prediction logic based on GPU usage
        best_loss = np.inf
        # 🧠 ML Signal: Custom neural network model class definition
        best_epoch = 0
        stop_round = 0
        # ✅ Best Practice: Use of super() to initialize the parent class

        for epoch in range(self.n_epochs):
            # 🧠 ML Signal: Use of nn.Sequential to build a neural network model
            print("Epoch:", epoch)
            # 🧠 ML Signal: Returns predictions as a pandas Series

            # 🧠 ML Signal: Use of nn.Softmax indicates a classification task
            print("training...")
            self.train_epoch(x_train, y_train, x_valid, y_valid)
            print("evaluating...")
            val_loss = self.test_epoch(x_valid, y_valid)
            # 🧠 ML Signal: Use of nn.Dropout for regularization
            test_loss = self.test_epoch(x_test, y_test)

            # 🧠 ML Signal: Use of a forward method suggests this is part of a neural network model
            # 🧠 ML Signal: Use of nn.Linear to define fully connected layers
            if verbose:
                # 🧠 ML Signal: The use of self.mlp(x) indicates a multi-layer perceptron is being used
                print("valid %.6f, test %.6f" % (val_loss, test_loss))
            # 🧠 ML Signal: Use of nn.ReLU as an activation function

            # 🧠 ML Signal: Applying squeeze suggests handling of tensor dimensions, common in ML models
            # 🧠 ML Signal: Custom model class definition for a GRU-based neural network
            if val_loss < best_loss:
                # 🧠 ML Signal: Final output layer with nn.Linear
                best_loss = val_loss
                # 🧠 ML Signal: Use of softmax indicates this is likely a classification task
                # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
                stop_round = 0
                # 🧠 ML Signal: Use of GRU indicates a sequence modeling task, common in time-series or NLP
                # ✅ Best Practice: Returning the output directly is clear and concise
                best_epoch = epoch
                torch.save(
                    copy.deepcopy(self.fore_model.state_dict()),
                    save_path + "_fore_model.bin",
                )
                torch.save(
                    copy.deepcopy(self.weight_model.state_dict()),
                    save_path + "_weight_model.bin",
                )

            else:
                stop_round += 1
                if stop_round >= self.early_stop:
                    print("early stop")
                    break
        # 🧠 ML Signal: Linear layer suggests a regression or binary classification task

        # 🧠 ML Signal: Reshaping input data for model processing
        print("best loss:", best_loss, "@", best_epoch)
        # ✅ Best Practice: Storing d_feat as an instance variable for potential future use
        best_param = torch.load(save_path + "_fore_model.bin", map_location=self.device)
        # 🧠 ML Signal: Permuting tensor dimensions for RNN input
        self.fore_model.load_state_dict(best_param)
        # 🧠 ML Signal: Using RNN to process sequential data
        # 🧠 ML Signal: Applying fully connected layer to RNN output
        best_param = torch.load(
            save_path + "_weight_model.bin", map_location=self.device
        )
        self.weight_model.load_state_dict(best_param)
        self.fitted = True

        if self.use_gpu:
            torch.cuda.empty_cache()

        return best_loss

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index
        self.fore_model.eval()
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
                if self.use_gpu:
                    pred = self.fore_model(x_batch).detach().cpu().numpy()
                else:
                    pred = self.fore_model(x_batch).detach().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class MLPModel(nn.Module):
    def __init__(
        self, d_feat, hidden_size=256, num_layers=3, dropout=0.0, output_dim=1
    ):
        super().__init__()

        self.mlp = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module("drop_%d" % i, nn.Dropout(dropout))
            self.mlp.add_module(
                "fc_%d" % i, nn.Linear(d_feat if i == 0 else hidden_size, hidden_size)
            )
            self.mlp.add_module("relu_%d" % i, nn.ReLU())

        self.mlp.add_module("fc_out", nn.Linear(hidden_size, output_dim))

    def forward(self, x):
        # feature
        # [N, F]
        out = self.mlp(x).squeeze()
        out = self.softmax(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
