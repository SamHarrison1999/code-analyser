# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import torch
import torch.nn as nn
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
import torch.optim as optim
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


class LSTM(Model):
    """LSTM Model

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
        # ✅ Best Practice: Use a logger for logging instead of print statements
        lr=0.001,
        metric="",
        batch_size=2000,
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        early_stop=20,
        loss="mse",
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        optimizer="adam",
        n_jobs=10,
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        GPU=0,
        seed=None,
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        **kwargs,
    ):
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        # Set logger.
        self.logger = get_module_logger("LSTM")
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        self.logger.info("LSTM pytorch version...")

        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        # set hyper-parameters.
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
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
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "LSTM parameters setting:"
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
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                # ⚠️ SAST Risk (Low): AttributeError if self.use_gpu is not defined
                self.use_gpu,
                seed,
            )
        )

        # 🧠 ML Signal: Setting random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
            # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating a preference for hardware acceleration
            torch.manual_seed(self.seed)
        # 🧠 ML Signal: Setting random seed for reproducibility

        # ✅ Best Practice: Direct comparison with torch.device ensures clarity and correctness
        self.LSTM_model = LSTMModel(
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            # 🧠 ML Signal: Initializing the LSTM model with specified parameters
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            # 🧠 ML Signal: Usage of mean squared error (MSE) loss function, common in regression tasks.
            # 🧠 ML Signal: Custom loss function implementation
            num_layers=self.num_layers,
            dropout=self.dropout,
        # ⚠️ SAST Risk (Low): Ensure that 'torch' is imported and available in the scope to avoid runtime errors.
        # 🧠 ML Signal: Handling missing values in labels
        ).to(self.device)
        if optimizer.lower() == "adam":
            # ✅ Best Practice: Default weight handling
            self.train_optimizer = optim.Adam(self.LSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # 🧠 ML Signal: Using Adam optimizer for training
            self.train_optimizer = optim.SGD(self.LSTM_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        # 🧠 ML Signal: Conditional logic for different loss functions
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # 🧠 ML Signal: Using SGD optimizer for training
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid label values

        # ⚠️ SAST Risk (Low): Potential for unhandled loss types
        self.fitted = False
        # 🧠 ML Signal: Conditional logic based on metric type
        self.LSTM_model.to(self.device)
    # ⚠️ SAST Risk (Low): Potential denial of service if unsupported optimizer is used

    # 🧠 ML Signal: Use of mask to filter predictions and labels
    @property
    # 🧠 ML Signal: Tracking whether the model has been fitted
    # ⚠️ SAST Risk (Low): Potential for runtime error if pred and label shapes do not match
    def use_gpu(self):
        return self.device != torch.device("cpu")
    # ✅ Best Practice: Ensure model is on the correct device
    # ⚠️ SAST Risk (Low): Use of string interpolation in exception message

    def mse(self, pred, label, weight):
        # 🧠 ML Signal: Use of a custom loss function with weights
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight):
        # ⚠️ SAST Risk (Low): Potential for exploding gradients without clipping
        mask = ~torch.isnan(label)

        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization.
        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])
        # 🧠 ML Signal: Extracting features and labels from data for model prediction.

        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Model prediction using LSTM model.
    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # 🧠 ML Signal: Calculating loss using a custom loss function.

        # ⚠️ SAST Risk (Low): Potential for device mismatch if `weight` is not on the same device.
        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Collecting loss values for analysis.
            # 🧠 ML Signal: Calculating metric score for model evaluation.
            return -self.loss_fn(pred[mask], label[mask], weight=None)

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        # 🧠 ML Signal: Collecting score values for analysis.
        self.LSTM_model.train()

        # 🧠 ML Signal: Returning average loss and score for the epoch.
        # 🧠 ML Signal: Preparing training and validation datasets
        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.LSTM_model(feature.float())
            # ✅ Best Practice: Consistent data preprocessing with fillna_type
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            # 🧠 ML Signal: Default weights for training and validation datasets
            torch.nn.utils.clip_grad_value_(self.LSTM_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        # 🧠 ML Signal: Custom reweighting of datasets
        self.LSTM_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            # 🧠 ML Signal: DataLoader configuration for training
            label = data[:, -1, -1].to(self.device)

            pred = self.LSTM_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            # 🧠 ML Signal: DataLoader configuration for validation
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        # ✅ Best Practice: Ensure save_path is valid or created
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        # 🧠 ML Signal: Training for each epoch
        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        # 🧠 ML Signal: Evaluation of training and validation datasets
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            # 🧠 ML Signal: Storing best model parameters
            num_workers=self.n_jobs,
            drop_last=True,
        )
        # ⚠️ SAST Risk (Low): No check for dataset validity or type, could lead to runtime errors
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            # ⚠️ SAST Risk (Low): Error message could expose internal state if not handled properly
            batch_size=self.batch_size,
            shuffle=False,
            # 🧠 ML Signal: Usage of dataset preparation with specific column sets
            num_workers=self.n_jobs,
            # 🧠 ML Signal: Loading best model parameters
            drop_last=True,
        # 🧠 ML Signal: Configuration of data handling with fillna_type
        )
        # ⚠️ SAST Risk (Low): Ensure save_path is secure and validated

        # 🧠 ML Signal: Usage of DataLoader with specific batch size and number of workers
        save_path = get_or_create_path(save_path)

        # 🧠 ML Signal: Model evaluation mode set before prediction
        # ✅ Best Practice: Clear GPU cache after training
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        # 🧠 ML Signal: Custom neural network model class definition
        best_epoch = 0
        # ⚠️ SAST Risk (Low): Assumes data shape and device compatibility without checks
        evals_result["train"] = []
        # ✅ Best Practice: Use of default values for function parameters improves usability and flexibility.
        evals_result["valid"] = []
        # 🧠 ML Signal: Use of torch.no_grad() for inference
        # 🧠 ML Signal: Use of LSTM indicates a sequence modeling task, common in time-series or NLP.

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # ✅ Best Practice: Using pd.Series for structured output with index
            self.logger.info("training...")
            self.train_epoch(train_loader)
            # 🧠 ML Signal: Use of a linear layer after LSTM suggests a regression or binary classification task.
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Use of RNN layer indicates sequence processing, common in time-series or NLP tasks
            train_loss, train_score = self.test_epoch(train_loader)
            # ✅ Best Practice: Storing input feature size as an instance variable can improve code readability and maintainability.
            # 🧠 ML Signal: Accessing the last output of RNN suggests interest in final state, typical in classification tasks
            # ✅ Best Practice: Squeezing the output is a common practice to ensure correct dimensionality for loss functions
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.LSTM_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.LSTM_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.LSTM_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.LSTM_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()