# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function


import copy
import math
from typing import Text, Union

# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
import numpy as np
import pandas as pd
import torch
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qlib.contrib.model.pytorch_gru import GRUModel
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_utils import count_parameters
# 🧠 ML Signal: Defines a machine learning model class, which is a common pattern in ML codebases
from qlib.data.dataset import DatasetH
# ✅ Best Practice: Importing specific classes or functions from a module can improve code readability and reduce memory usage.
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from torch.autograd import Function


class ADD(Model):
    """ADD Model

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
        dec_dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="mse",
        batch_size=5000,
        early_stop=20,
        # ✅ Best Practice: Use of a logger for information and debugging
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        # 🧠 ML Signal: Initialization of model hyperparameters
        gamma=0.1,
        gamma_clip=0.4,
        mu=0.05,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("ADD")
        self.logger.info("ADD pytorch version...")

        # 🧠 ML Signal: Use of optimizer parameter
        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        self.dropout = dropout
        self.dec_dropout = dec_dropout
        # ✅ Best Practice: Logging parameter settings for traceability
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.gamma = gamma
        self.gamma_clip = gamma_clip
        self.mu = mu

        self.logger.info(
            "ADD parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\ndec_dropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\ngamma : {}"
            "\ngamma_clip : {}"
            "\nmu : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                dec_dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                # 🧠 ML Signal: Setting random seed for reproducibility
                early_stop,
                optimizer.lower(),
                base_model,
                model_path,
                gamma,
                gamma_clip,
                mu,
                self.device,
                self.use_gpu,
                seed,
            # 🧠 ML Signal: Model instantiation with parameters
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.ADD_model = ADDModel(
            # ✅ Best Practice: Logging model size for resource management
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # 🧠 ML Signal: Use of Adam optimizer
            # 🧠 ML Signal: Checking if a GPU is being used for computation
            dropout=self.dropout,
            dec_dropout=self.dec_dropout,
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            # ✅ Best Practice: Using torch.device to handle device types
            base_model=self.base_model,
            # 🧠 ML Signal: Use of SGD optimizer
            gamma=self.gamma,
            # 🧠 ML Signal: Usage of torch.isnan to create a mask for valid data points
            gamma_clip=self.gamma_clip,
        )
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
        # 🧠 ML Signal: Usage of F.mse_loss to calculate mean squared error loss
        self.logger.info("model:\n{:}".format(self.ADD_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ADD_model)))
        # 🧠 ML Signal: Function for calculating loss, common in ML model training
        # ✅ Best Practice: Check if 'record' is not None before attempting to modify it

        # 🧠 ML Signal: Model moved to specified device (CPU/GPU)
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Use of cross-entropy loss, typical in classification tasks
            # 🧠 ML Signal: Storing loss value in a dictionary for later analysis or logging
            self.train_optimizer = optim.Adam(self.ADD_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # ✅ Best Practice: Checking if 'record' is not None before using it
            self.train_optimizer = optim.SGD(self.ADD_model.parameters(), lr=self.lr)
        # ✅ Best Practice: Function parameters are descriptive and indicate their purpose
        else:
            # ✅ Best Practice: Storing loss value in a dictionary for logging or analysis
            # 🧠 ML Signal: Combines multiple loss functions, indicating a composite loss calculation
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        # 🧠 ML Signal: Returning loss value, common in training loops
        self.ADD_model.to(self.device)

    # ✅ Best Practice: Checks if 'record' is not None before using it
    @property
    # 🧠 ML Signal: Function for calculating adversarial excess loss, useful for ML model training
    def use_gpu(self):
        # 🧠 ML Signal: Storing loss value in a record, useful for tracking and analysis
        return self.device != torch.device("cpu")
    # 🧠 ML Signal: Handling NaN values in label_excess, common in data preprocessing

    # ✅ Best Practice: Returns the computed loss value
    def loss_pre_excess(self, pred_excess, label_excess, record=None):
        # 🧠 ML Signal: Use of mean squared error loss, a common loss function in regression tasks
        mask = ~torch.isnan(label_excess)
        pre_excess_loss = F.mse_loss(pred_excess[mask], label_excess[mask])
        # ✅ Best Practice: Check if record is not None before updating it
        # 🧠 ML Signal: Function definition for calculating adversarial market loss
        if record is not None:
            record["pre_excess_loss"] = pre_excess_loss.item()
        # 🧠 ML Signal: Recording loss value, useful for logging and monitoring during training
        # 🧠 ML Signal: Use of cross-entropy loss function, common in classification tasks
        return pre_excess_loss

    # ✅ Best Practice: Check if 'record' is not None before attempting to use it
    def loss_pre_market(self, pred_market, label_market, record=None):
        pre_market_loss = F.cross_entropy(pred_market, label_market)
        # 🧠 ML Signal: Method for calculating adversarial loss, useful for ML model training
        # ✅ Best Practice: Store the loss value in a dictionary for logging or debugging
        if record is not None:
            record["pre_market_loss"] = pre_market_loss.item()
        return pre_market_loss
    # 🧠 ML Signal: Returning the calculated loss, typical in loss function implementations

    def loss_pre(self, pred_excess, label_excess, pred_market, label_market, record=None):
        # ✅ Best Practice: Check if 'record' is not None before using it
        pre_loss = self.loss_pre_excess(pred_excess, label_excess, record) + self.loss_pre_market(
            # 🧠 ML Signal: Custom loss function combining multiple loss components
            pred_market, label_market, record
        # 🧠 ML Signal: Use of multiple loss functions for different prediction components
        # ✅ Best Practice: Parentheses used for multi-line expression for readability
        )
        if record is not None:
            record["pre_loss"] = pre_loss.item()
        return pre_loss

    # 🧠 ML Signal: Custom loss function for excess and market predictions
    def loss_adv_excess(self, adv_excess, label_excess, record=None):
        # 🧠 ML Signal: Adversarial loss component for robustness
        mask = ~torch.isnan(label_excess)
        adv_excess_loss = F.mse_loss(adv_excess.squeeze()[mask], label_excess[mask])
        # 🧠 ML Signal: Regularization term for reconstruction
        if record is not None:
            # 🧠 ML Signal: Reshaping input data, common in preprocessing for ML models
            record["adv_excess_loss"] = adv_excess_loss.item()
        return adv_excess_loss
    # ✅ Best Practice: Conditional check for optional parameter 'record'
    # 🧠 ML Signal: Permuting tensor dimensions, often used in ML for aligning data

    def loss_adv_market(self, adv_market, label_market, record=None):
        # ⚠️ SAST Risk (Low): Potential for 'record' to be a mutable default argument if not handled properly
        # 🧠 ML Signal: Using mean squared error loss, a common loss function in regression tasks
        adv_market_loss = F.cross_entropy(adv_market, label_market)
        if record is not None:
            # ✅ Best Practice: Checking if 'record' is not None before using it
            record["adv_market_loss"] = adv_market_loss.item()
        # 🧠 ML Signal: Use of DataFrame groupby operation, common in data processing tasks
        return adv_market_loss
    # 🧠 ML Signal: Storing loss value, useful for tracking model performance

    # 🧠 ML Signal: Use of numpy operations for array manipulation
    def loss_adv(self, adv_excess, label_excess, adv_market, label_market, record=None):
        adv_loss = self.loss_adv_excess(adv_excess, label_excess, record) + self.loss_adv_market(
            adv_market, label_market, record
        # 🧠 ML Signal: Conditional logic based on a parameter, indicating a configurable behavior
        )
        if record is not None:
            # 🧠 ML Signal: Use of random shuffling, indicating a need for randomized data order
            record["adv_loss"] = adv_loss.item()
        # ⚠️ SAST Risk (Low): Use of np.random.shuffle can lead to non-deterministic behavior
        return adv_loss

    # ⚠️ SAST Risk (Low): Using negative of MSE loss might be confusing; ensure this is intentional
    def loss_fn(self, x, preds, label_excess, label_market, record=None):
        loss = (
            # ✅ Best Practice: Returning multiple values as a tuple for clarity and simplicity
            # ✅ Best Practice: Consider renaming "loss" to something more descriptive if it always mirrors "mse"
            self.loss_pre(preds["excess"], label_excess, preds["market"], label_market, record)
            + self.loss_adv(preds["adv_excess"], label_excess, preds["adv_market"], label_market, record)
            # 🧠 ML Signal: Converting tensors to pandas Series for correlation calculation
            + self.mu * self.loss_rec(x, preds["reconstructed_feature"], record)
        )
        if record is not None:
            # 🧠 ML Signal: Calculating Pearson correlation
            record["loss"] = loss.item()
        return loss
    # 🧠 ML Signal: Calculating Spearman correlation

    def loss_rec(self, x, rec_x, record=None):
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        rec_loss = F.mse_loss(x, rec_x)
        # 🧠 ML Signal: Usage of a custom method to get daily indices and counts
        if record is not None:
            record["rec_loss"] = rec_loss.item()
        return rec_loss

    # ✅ Best Practice: Convert numpy arrays to torch tensors for model input
    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        # 🧠 ML Signal: Model prediction step
        if shuffle:
            # shuffle data
            # 🧠 ML Signal: Custom loss function usage
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            # 🧠 ML Signal: Custom metric calculation
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def cal_ic_metrics(self, pred, label):
        metrics = {}
        # 🧠 ML Signal: Shuffling data indices before training is a common practice in ML to ensure randomness.
        metrics["mse"] = -F.mse_loss(pred, label).item()
        metrics["loss"] = metrics["mse"]
        # ✅ Best Practice: Calculate average metrics over all batches
        pred = pd.Series(pred.cpu().detach().numpy())
        label = pd.Series(label.cpu().detach().numpy())
        # 🧠 ML Signal: Iterating over data in batches is a common pattern in training ML models.
        metrics["ic"] = pred.corr(label)
        metrics["ric"] = pred.corr(label, method="spearman")
        return metrics

    # 🧠 ML Signal: Creating batches of data for training.
    def test_epoch(self, data_x, data_y, data_m):
        x_values = data_x.values
        # 🧠 ML Signal: Converting numpy arrays to torch tensors for model input.
        y_values = np.squeeze(data_y.values)
        m_values = np.squeeze(data_m.values.astype(int))
        self.ADD_model.eval()

        # 🧠 ML Signal: Forward pass through the model to get predictions.
        metrics_list = []

        # 🧠 ML Signal: Calculating loss is a key step in training ML models.
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)
        # 🧠 ML Signal: Method for logging metrics, useful for tracking model performance

        # 🧠 ML Signal: Zeroing gradients is a standard step before backpropagation.
        for idx, count in zip(daily_index, daily_count):
            # 🧠 ML Signal: Iterating over metrics to format them, common in logging and monitoring
            batch = slice(idx, idx + count)
            # 🧠 ML Signal: Backward pass to compute gradients.
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            # ✅ Best Practice: Joining list of strings for efficient string concatenation
            label_excess = torch.from_numpy(y_values[batch]).float().to(self.device)
            # ⚠️ SAST Risk (Low): Clipping gradients can prevent exploding gradients but should be used cautiously.
            label_market = torch.from_numpy(m_values[batch]).long().to(self.device)
            # ⚠️ SAST Risk (Low): Potential information exposure if sensitive metrics are logged

            # 🧠 ML Signal: Updating model parameters using the optimizer.
            metrics = {}
            preds = self.ADD_model(feature)
            self.loss_fn(feature, preds, label_excess, label_market, metrics)
            metrics.update(self.cal_ic_metrics(preds["excess"], label_excess))
            metrics_list.append(metrics)
        metrics = {}
        # ✅ Best Practice: Consider adding type hints for the function parameters for better readability and maintainability.
        keys = metrics_list[0].keys()
        for k in keys:
            vs = [m[k] for m in metrics_list]
            metrics[k] = sum(vs) / len(vs)
        # 🧠 ML Signal: Iterating over epochs is a common pattern in training machine learning models.

        return metrics

    def train_epoch(self, x_train_values, y_train_values, m_train_values):
        self.ADD_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        cur_step = 1

        # ⚠️ SAST Risk (Low): Potential for a ValueError if `self.metric` is not in `valid_metrics`.
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            batch = indices[i : i + self.batch_size]
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label_excess = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            # ⚠️ SAST Risk (Low): Deep copying large model states can be memory intensive.
            label_market = torch.from_numpy(m_train_values[batch]).long().to(self.device)

            preds = self.ADD_model(feature)

            loss = self.loss_fn(feature, preds, label_excess, label_market)

            self.train_optimizer.zero_grad()
            loss.backward()
            # 🧠 ML Signal: Use of groupby and mean to aggregate data, common in data preprocessing for ML.
            torch.nn.utils.clip_grad_value_(self.ADD_model.parameters(), 3.0)
            self.train_optimizer.step()
            # 🧠 ML Signal: Use of np.inf and pd.cut for binning, a common technique in feature engineering.
            cur_step += 1

    # 🧠 ML Signal: Binning continuous data into discrete intervals, useful for classification tasks.
    def log_metrics(self, mode, metrics):
        metrics = ["{}/{}: {:.6f}".format(k, mode, v) for k, v in metrics.items()]
        # ✅ Best Practice: Explicitly setting the name of the series for clarity and consistency.
        # 🧠 ML Signal: Method for fitting thresholds based on training labels
        metrics = ", ".join(metrics)
        self.logger.info(metrics)
    # ⚠️ SAST Risk (Low): Potential risk if df and market_label have mismatched indices.
    # 🧠 ML Signal: Grouping data by datetime to calculate mean market label

    # 🧠 ML Signal: Calculating quantiles to determine threshold values
    def bootstrap_fit(self, x_train, y_train, m_train, x_valid, y_valid, m_valid):
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0

        # train
        # ✅ Best Practice: Consider using a default value of None for mutable arguments like dictionaries to avoid shared state issues.
        self.logger.info("training...")
        self.fitted = True
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        m_train_values = np.squeeze(m_train.values.astype(int))

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train_values, y_train_values, m_train_values)
            self.logger.info("evaluating...")
            train_metrics = self.test_epoch(x_train, y_train, m_train)
            valid_metrics = self.test_epoch(x_valid, y_valid, m_valid)
            self.log_metrics("train", train_metrics)
            self.log_metrics("valid", valid_metrics)

            if self.metric in valid_metrics:
                val_score = valid_metrics[self.metric]
            else:
                raise ValueError("unknown metric name `%s`" % self.metric)
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                # ⚠️ SAST Risk (Low): Loading models from a path can introduce security risks if the path is not trusted.
                best_param = copy.deepcopy(self.ADD_model.state_dict())
            else:
                stop_steps += 1
                # ⚠️ SAST Risk (Medium): Loading a model state dict from a file can be risky if the file is not trusted.
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break
            self.ADD_model.before_adv_excess.step_alpha()
            self.ADD_model.before_adv_market.step_alpha()
        self.logger.info("bootstrap_fit best score: {:.6f} @ {}".format(best_score, best_epoch))
        self.ADD_model.load_state_dict(best_param)
        return best_score

    def gen_market_label(self, df, raw_label):
        market_label = raw_label.groupby("datetime", group_keys=False).mean().squeeze()
        bins = [-np.inf, self.lo, self.hi, np.inf]
        market_label = pd.cut(market_label, bins, labels=False)
        # ⚠️ SAST Risk (Low): Saving models to a path can overwrite existing files if not handled carefully.
        market_label.name = ("market_return", "market_return")
        df = df.join(market_label)
        # 🧠 ML Signal: Usage of dataset and segment parameters indicates a pattern for model prediction
        return df
    # 🧠 ML Signal: Indicates the use of GPU resources, which can be a feature for ML model training.

    def fit_thresh(self, train_label):
        # 🧠 ML Signal: Model evaluation mode is set, indicating a prediction phase
        market_label = train_label.groupby("datetime", group_keys=False).mean().squeeze()
        self.lo, self.hi = market_label.quantile([1 / 3, 2 / 3])

    def fit(
        # 🧠 ML Signal: get_daily_inter method usage suggests a pattern for handling time-series data
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    # ⚠️ SAST Risk (Low): Direct conversion of numpy array to torch tensor without validation
    ):
        label_train, label_valid = dataset.prepare(
            # ✅ Best Practice: Use of torch.no_grad() for inference to save memory
            ["train", "valid"],
            col_set=["label"],
            # 🧠 ML Signal: Model prediction step
            # 🧠 ML Signal: Custom model class definition for PyTorch
            data_key=DataHandlerLP.DK_R,
        # ⚠️ SAST Risk (Low): Potential risk if "excess" key is not present in pred
        # 🧠 ML Signal: Concatenation of predictions into a pandas Series
        )
        self.fit_thresh(label_train)
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_train = self.gen_market_label(df_train, label_train)
        df_valid = self.gen_market_label(df_valid, label_valid)

        x_train, y_train, m_train = df_train["feature"], df_train["label"], df_train["market_return"]
        x_valid, y_valid, m_valid = df_valid["feature"], df_valid["label"], df_valid["market_return"]

        evals_result["train"] = []
        # 🧠 ML Signal: Conditional logic based on model type (GRU or LSTM) indicates model architecture customization
        evals_result["valid"] = []
        # 🧠 ML Signal: Use of GRU layers suggests a recurrent neural network architecture
        # load pretrained base_model

        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            # 🧠 ML Signal: Conditional logic based on model type (GRU or LSTM) indicates model architecture customization
            # 🧠 ML Signal: Use of LSTM layers suggests a recurrent neural network architecture
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            model_dict = self.ADD_model.enc_excess.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.rnn.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ADD_model.enc_excess.load_state_dict(model_dict)
            model_dict = self.ADD_model.enc_market.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.rnn.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ADD_model.enc_market.load_state_dict(model_dict)
            self.logger.info("Loading pretrained model Done...")

        self.bootstrap_fit(x_train, y_train, m_train, x_valid, y_valid, m_valid)

        # ⚠️ SAST Risk (Low): Use of ValueError for handling unknown model types; consider logging the error
        best_param = copy.deepcopy(self.ADD_model.state_dict())
        save_path = get_or_create_path(save_path)
        torch.save(best_param, save_path)
        if self.use_gpu:
            # 🧠 ML Signal: Use of nn.Sequential for defining neural network layers
            # 🧠 ML Signal: Use of a custom Decoder class indicates a specific decoding process in the model
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.ADD_model.eval()
        # 🧠 ML Signal: Reshaping input data for model processing
        x_values = x_test.values
        # 🧠 ML Signal: Use of nn.Sequential for defining neural network layers
        preds = []

        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)
        # 🧠 ML Signal: Permuting tensor dimensions for model compatibility

        for idx, count in zip(daily_index, daily_count):
            # 🧠 ML Signal: Encoding input data with enc_excess model
            # 🧠 ML Signal: Use of RevGrad indicates adversarial training or domain adaptation
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            # 🧠 ML Signal: Encoding input data with enc_market model

            with torch.no_grad():
                pred = self.ADD_model(x_batch)
                # 🧠 ML Signal: Processing LSTM hidden states
                pred = pred["excess"].detach().cpu().numpy()

            preds.append(pred)

        # 🧠 ML Signal: Processing non-LSTM hidden states
        r = pd.Series(np.concatenate(preds), index=index)
        return r

# 🧠 ML Signal: Predicting excess features

class ADDModel(nn.Module):
    def __init__(
        # 🧠 ML Signal: Predicting market features
        self,
        d_feat=6,
        # 🧠 ML Signal: Adversarial prediction for market features
        hidden_size=64,
        num_layers=1,
        # 🧠 ML Signal: Adversarial prediction for excess features
        dropout=0.0,
        dec_dropout=0.5,
        base_model="GRU",
        # 🧠 ML Signal: Concatenating LSTM hidden states
        gamma=0.1,
        # 🧠 ML Signal: Custom neural network module definition
        gamma_clip=0.4,
    ):
        # 🧠 ML Signal: Concatenating non-LSTM hidden states
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        super().__init__()
        self.d_feat = d_feat
        # ✅ Best Practice: Initializing tensor with zeros for reconstruction
        # 🧠 ML Signal: Use of a parameter to select between different RNN models
        self.base_model = base_model
        # 🧠 ML Signal: Conditional logic to select model architecture
        # 🧠 ML Signal: Decoding step in sequence processing
        # 🧠 ML Signal: Use of GRU model with specified parameters
        if base_model == "GRU":
            self.enc_excess, self.enc_market = [
                nn.GRU(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    # 🧠 ML Signal: Stacking reconstructed features
                    dropout=dropout,
                # 🧠 ML Signal: Conditional logic to select model architecture
                # 🧠 ML Signal: Adding reconstructed features to predictions
                )
                for _ in range(2)
            ]
        elif base_model == "LSTM":
            self.enc_excess, self.enc_market = [
                nn.LSTM(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    # 🧠 ML Signal: Use of LSTM model with specified parameters
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                # 🧠 ML Signal: Use of unsqueeze to add a dimension, common in data preprocessing for ML models
                )
                for _ in range(2)
            # 🧠 ML Signal: Use of RNN layer, indicative of sequence modeling tasks
            ]
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if base_model is not recognized
        else:
            # 🧠 ML Signal: Use of squeeze to remove a dimension, common in data postprocessing for ML models
            raise ValueError("unknown base model name `%s`" % base_model)
        # 🧠 ML Signal: Use of a fully connected layer after RNN
        # 🧠 ML Signal: Use of fully connected layer for prediction, common in neural network architectures
        # ✅ Best Practice: Use of @staticmethod decorator for methods that do not access instance or class data
        self.dec = Decoder(d_feat, 2 * hidden_size, num_layers, dec_dropout, base_model)

        ctx_size = hidden_size * num_layers
        # ✅ Best Practice: Returning both prediction and hidden state, useful for RNNs in sequence tasks
        # ✅ Best Practice: Store input in context for backward computation
        # ✅ Best Practice: Save tensors for backward pass to ensure gradients can be computed
        self.pred_excess, self.adv_excess = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 1))
            # 🧠 ML Signal: Directly returning input as output, indicating identity operation
            for _ in range(2)
        ]
        self.adv_market, self.pred_market = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 3))
            # ✅ Best Practice: Retrieve saved tensors for backward computation
            for _ in range(2)
        # ✅ Best Practice: Check if input gradient is needed before computing it
        ]
        self.before_adv_market, self.before_adv_excess = [RevGrad(gamma, gamma_clip) for _ in range(2)]
    # 🧠 ML Signal: Custom backward function for gradient reversal
    # 🧠 ML Signal: Pattern for implementing custom backward pass in autograd

    # ✅ Best Practice: Inheriting from nn.Module is standard for defining custom layers in PyTorch
    def forward(self, x):
        # ✅ Best Practice: Return a consistent number of elements as expected by the forward method
        x = x.reshape(len(x), self.d_feat, -1)
        # ✅ Best Practice: Provide a docstring to describe the purpose and behavior of the class.
        N = x.shape[0]
        T = x.shape[-1]
        x = x.permute(0, 2, 1)

        out, hidden_excess = self.enc_excess(x)
        out, hidden_market = self.enc_market(x)
        # ✅ Best Practice: Call the superclass's __init__ method to ensure proper initialization.
        if self.base_model == "LSTM":
            # 🧠 ML Signal: Use of custom autograd function in forward pass
            feature_excess = hidden_excess[0].permute(1, 0, 2).reshape(N, -1)
            # 🧠 ML Signal: Usage of hyperparameters (gamma, gamma_clip) for model behavior.
            feature_market = hidden_market[0].permute(1, 0, 2).reshape(N, -1)
        else:
            # ⚠️ SAST Risk (Low): Ensure torch is imported to avoid runtime errors.
            feature_excess = hidden_excess.permute(1, 0, 2).reshape(N, -1)
            # 🧠 ML Signal: Use of torch tensors, indicating deep learning context.
            # 🧠 ML Signal: Method that updates internal state, useful for tracking object behavior over time
            feature_market = hidden_market.permute(1, 0, 2).reshape(N, -1)
        # 🧠 ML Signal: Use of torch tensors, indicating deep learning context.
        # ✅ Best Practice: Use of min function to ensure _alpha does not exceed gamma_clip
        predicts = {}
        predicts["excess"] = self.pred_excess(feature_excess).squeeze(1)
        # ⚠️ SAST Risk (Low): Potential precision issues with floating-point arithmetic
        predicts["market"] = self.pred_market(feature_market)
        # 🧠 ML Signal: Tracking internal state with self._p, possibly for learning rate or iteration count.
        # ✅ Best Practice: Method name 'forward' is commonly used in ML models for the forward pass
        predicts["adv_market"] = self.adv_market(self.before_adv_market(feature_excess))
        # 🧠 ML Signal: Usage of a custom function 'RevGradFunc.apply' indicates a potential custom gradient operation
        predicts["adv_excess"] = self.adv_excess(self.before_adv_excess(feature_market).squeeze(1))
        if self.base_model == "LSTM":
            hidden = [torch.cat([hidden_excess[i], hidden_market[i]], -1) for i in range(2)]
        else:
            hidden = torch.cat([hidden_excess, hidden_market], -1)
        x = torch.zeros_like(x[:, 1, :])
        reconstructed_feature = []
        for i in range(T):
            x, hidden = self.dec(x, hidden)
            reconstructed_feature.append(x)
        reconstructed_feature = torch.stack(reconstructed_feature, 1)
        predicts["reconstructed_feature"] = reconstructed_feature
        return predicts


class Decoder(nn.Module):
    def __init__(self, d_feat=6, hidden_size=128, num_layers=1, dropout=0.5, base_model="GRU"):
        super().__init__()
        self.base_model = base_model
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

        self.fc = nn.Linear(hidden_size, d_feat)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = output.squeeze(1)
        pred = self.fc(output)
        return pred, hidden


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, gamma=0.1, gamma_clip=0.4, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.gamma_clip = torch.tensor(float(gamma_clip), requires_grad=False)
        self._alpha = torch.tensor(0, requires_grad=False)
        self._p = 0

    def step_alpha(self):
        self._p += 1
        self._alpha = min(
            self.gamma_clip, torch.tensor(2 / (1 + math.exp(-self.gamma * self._p)) - 1, requires_grad=False)
        )

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)