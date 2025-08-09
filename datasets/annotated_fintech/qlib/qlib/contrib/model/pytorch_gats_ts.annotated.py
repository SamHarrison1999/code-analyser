# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from __future__ import print_function

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import torch.optim as optim
from torch.utils.data import DataLoader

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from torch.utils.data import Sampler

# ✅ Best Practice: Inheriting from a base class (Sampler) promotes code reuse and consistency.

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from .pytorch_utils import count_parameters

# 🧠 ML Signal: Initialization of class with data source, common pattern in data processing classes
from ...model.base import Model

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
# ✅ Best Practice: Use of pandas for data manipulation, a standard library for such tasks
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel

# ✅ Best Practice: Use of relative imports for better modularity and maintainability


# ✅ Best Practice: Use of numpy for efficient numerical operations
class DailyBatchSampler(Sampler):
    # 🧠 ML Signal: Use of zip to iterate over two lists in parallel
    def __init__(self, data_source):
        # ✅ Best Practice: Explicitly setting the first element of an array, improving code clarity
        self.data_source = data_source
        # 🧠 ML Signal: Method overriding to customize behavior of built-in functions
        # 🧠 ML Signal: Use of np.arange to generate a range of numbers
        # calculate number of samples in each batch
        self.daily_count = (
            # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
            # ✅ Best Practice: Using len() for objects that support it
            pd.Series(index=self.data_source.get_index())
            .groupby("datetime", group_keys=False)
            .size()
            .values
        )
        self.daily_index = np.roll(
            np.cumsum(self.daily_count), 1
        )  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


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
        d_feat=20,
        hidden_size=64,
        # ✅ Best Practice: Use of a logger for information and debugging purposes
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        # 🧠 ML Signal: Initialization of model hyperparameters
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        GPU=0,
        n_jobs=10,
        # 🧠 ML Signal: Use of optimizer choice in model training
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GATs")
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available
        self.logger.info("GATs pytorch version...")

        # set hyper-parameters.
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
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.n_jobs = n_jobs
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
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                # 🧠 ML Signal: Setting random seed for reproducibility
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                early_stop,
                # 🧠 ML Signal: Initialization of a GAT model
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                GPU,
                self.use_gpu,
                seed,
                # 🧠 ML Signal: Logging model size for resource management
            )
        )
        # ⚠️ SAST Risk (Low): Use of hardcoded strings for optimizer names

        if self.seed is not None:
            # 🧠 ML Signal: Checks if the computation is set to run on a GPU
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # ⚠️ SAST Risk (Low): Assumes 'self.device' is a valid torch.device object
        # ✅ Best Practice: Consider adding type hints for function parameters and return type

        self.GAT_model = GATModel(
            # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported optimizers
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function, common in regression tasks
            d_feat=self.d_feat,
            # 🧠 ML Signal: Custom loss function implementation
            hidden_size=self.hidden_size,
            # ⚠️ SAST Risk (Low): Ensure 'torch' is imported and 'pred' and 'label' are tensors to avoid runtime errors
            num_layers=self.num_layers,
            # 🧠 ML Signal: Moving model to the specified device (CPU/GPU)
            # 🧠 ML Signal: Handling missing values in labels
            dropout=self.dropout,
            base_model=self.base_model,
            # 🧠 ML Signal: Conditional logic based on loss type
        )
        # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
        self.logger.info("model:\n{:}".format(self.GAT_model))
        # 🧠 ML Signal: Use of mean squared error for loss calculation
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.GAT_model))
        )
        # 🧠 ML Signal: Use of torch.isfinite to create a mask for valid (finite) values in the label tensor.

        # ⚠️ SAST Risk (Low): Potential for unhandled loss types leading to exceptions
        if optimizer.lower() == "adam":
            # 🧠 ML Signal: Conditional logic to handle different metric types, indicating a pattern of metric evaluation.
            self.train_optimizer = optim.Adam(self.GAT_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # ⚠️ SAST Risk (Low): Potential risk if loss_fn is not properly handling masked tensors.
            # 🧠 ML Signal: Use of DataFrame groupby operation, common in data processing tasks
            self.train_optimizer = optim.SGD(self.GAT_model.parameters(), lr=self.lr)
        else:
            # 🧠 ML Signal: Use of numpy operations for array manipulation
            # ⚠️ SAST Risk (Low): Use of string interpolation with user-controlled input in exception message.
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        self.fitted = False
        # ✅ Best Practice: Conditional logic to handle optional shuffling
        self.GAT_model.to(self.device)

    # 🧠 ML Signal: Use of shuffling, indicating potential data randomization
    @property
    def use_gpu(self):
        # ⚠️ SAST Risk (Low): Use of np.random.shuffle, which may affect reproducibility if not controlled
        return self.device != torch.device("cpu")

    # 🧠 ML Signal: Iterating over data_loader indicates a training loop pattern

    def mse(self, pred, label):
        # ✅ Best Practice: Squeeze is used to remove dimensions of size 1, ensuring data consistency
        loss = (pred - label) ** 2
        return torch.mean(loss)

    # ✅ Best Practice: Explicitly selecting features and labels improves code readability

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        # 🧠 ML Signal: Model prediction step in a training loop

        if self.loss == "mse":
            # 🧠 ML Signal: Loss calculation is a key step in training
            return self.mse(pred[mask], label[mask])

        # 🧠 ML Signal: Optimizer zero_grad is a common pattern in training loops
        # ✅ Best Practice: Set the model to evaluation mode to disable dropout and batch normalization layers.
        raise ValueError("unknown loss `%s`" % self.loss)

    # 🧠 ML Signal: Backward pass for gradient calculation
    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        # ⚠️ SAST Risk (Low): Clipping gradients to prevent exploding gradients, ensure it's set appropriately

        # ✅ Best Practice: Squeeze the data to remove any singleton dimensions.
        if self.metric in ("", "loss"):
            # 🧠 ML Signal: Optimizer step is a key part of the training loop
            return -self.loss_fn(pred[mask], label[mask])
        # 🧠 ML Signal: Extracting features and labels from data is a common pattern in ML model evaluation.

        raise ValueError("unknown metric `%s`" % self.metric)

    # 🧠 ML Signal: Model prediction step, typical in ML workflows.
    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        # 🧠 ML Signal: Calculating loss, a key step in model evaluation.
        daily_count = df.groupby(level=0, group_keys=False).size().values
        # 🧠 ML Signal: Calculating a metric score, common in model evaluation.
        # 🧠 ML Signal: Collecting loss values for later aggregation.
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            # 🧠 ML Signal: Collecting metric scores for later aggregation.
            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            daily_index, daily_count = zip(*daily_shuffle)
        # 🧠 ML Signal: Returning mean loss and score, typical in model evaluation.
        return daily_index, daily_count

    def train_epoch(self, data_loader):
        self.GAT_model.train()
        # ✅ Best Practice: Configuring data with fillna_type ensures data consistency.

        for data in data_loader:
            data = data.squeeze()
            # ✅ Best Practice: Use of custom sampler for data loading.
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # ✅ Best Practice: Use of DataLoader for efficient data handling.
            pred = self.GAT_model(feature.float())
            loss = self.loss_fn(pred, label)

            # ✅ Best Practice: Ensures save_path is valid or creates a new one.
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        # 🧠 ML Signal: Tracking evaluation results for training and validation.
        self.GAT_model.eval()

        scores = []
        losses = []
        # 🧠 ML Signal: Conditional model initialization based on base_model type.

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            pred = self.GAT_model(feature.float())
            # ⚠️ SAST Risk (Medium): Loading model state from a file can be risky if the file is untrusted.
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            # ✅ Best Practice: Use of dictionary comprehension for filtering state_dict.
            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
    ):
        dl_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        # 🧠 ML Signal: Iterative training process over epochs.
        dl_valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if dl_train.empty or dl_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)

        train_loader = DataLoader(
            dl_train, sampler=sampler_train, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, sampler=sampler_valid, num_workers=self.n_jobs, drop_last=True
        )
        # 🧠 ML Signal: Use of deepcopy to save the best model parameters.

        save_path = get_or_create_path(save_path)
        # ⚠️ SAST Risk (Low): No check for dataset validity or type, which could lead to runtime errors.

        stop_steps = 0
        train_loss = 0
        # 🧠 ML Signal: Usage of dataset preparation with specific column sets and data keys.
        best_score = -np.inf
        best_epoch = 0
        # 🧠 ML Signal: Configuration of data handling with specific fillna strategy.
        evals_result["train"] = []
        evals_result["valid"] = []
        # ⚠️ SAST Risk (Medium): Saving model state to a file can be risky if the file path is untrusted.
        # 🧠 ML Signal: Usage of a custom sampler for data loading.

        # load pretrained base_model
        # ⚠️ SAST Risk (Low): Potential for resource exhaustion if n_jobs is set too high.
        if self.base_model == "LSTM":
            # ✅ Best Practice: Clearing GPU cache to free up memory.
            pretrained_model = LSTMModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        # 🧠 ML Signal: Model evaluation mode set before prediction.
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)
        # 🧠 ML Signal: Custom neural network model class definition
        # ⚠️ SAST Risk (Low): Assumes data shape without validation, which could lead to errors.

        if self.model_path is not None:
            # ⚠️ SAST Risk (Low): Assumes specific data structure for feature extraction.
            self.logger.info("Loading pretrained model...")
            # 🧠 ML Signal: Conditional logic to select model architecture
            pretrained_model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        # 🧠 ML Signal: Use of GRU for sequence modeling
        # 🧠 ML Signal: Use of model prediction with no_grad for inference.

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_model.state_dict().items()
            if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)
        # ✅ Best Practice: Using pd.Series for structured output with index.
        self.logger.info("Loading pretrained model Done...")
        # 🧠 ML Signal: Conditional logic to select model architecture
        # 🧠 ML Signal: Use of LSTM for sequence modeling

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if base_model is invalid
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            # 🧠 ML Signal: Use of linear transformation layer
            if val_score > best_score:
                best_score = val_score
                # 🧠 ML Signal: Use of learnable parameter for attention mechanism
                stop_steps = 0
                # 🧠 ML Signal: Use of transformation function on input data
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            # 🧠 ML Signal: Use of transformation function on input data
            # 🧠 ML Signal: Use of fully connected layers for output transformation
            else:
                stop_steps += 1
                # 🧠 ML Signal: Use of tensor shape to determine sample number
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Use of activation function for non-linearity
                    self.logger.info("early stop")
                    # 🧠 ML Signal: Use of tensor shape to determine dimensionality
                    break
        # 🧠 ML Signal: Use of softmax for probability distribution

        # 🧠 ML Signal: Expanding tensor for attention mechanism
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)
        # 🧠 ML Signal: Transposing tensor for attention mechanism
        torch.save(best_param, save_path)

        # 🧠 ML Signal: Concatenating tensors for attention input
        if self.use_gpu:
            torch.cuda.empty_cache()

    # ⚠️ SAST Risk (Low): Potential misuse of transpose if self.a is not a 2D tensor

    # 🧠 ML Signal: Use of attention mechanism in neural network
    def predict(self, dataset):
        # ⚠️ SAST Risk (Low): Potential misuse of matrix multiplication if dimensions do not align
        if not self.fitted:
            # ⚠️ SAST Risk (Low): Potential for incorrect matrix multiplication dimensions
            raise ValueError("model is not fitted yet!")
        # 🧠 ML Signal: Use of activation function in attention mechanism

        # 🧠 ML Signal: Use of softmax for attention weights
        # 🧠 ML Signal: Use of activation function in neural network
        # ✅ Best Practice: Explicit return of computed attention weights
        # 🧠 ML Signal: Use of fully connected layer in neural network
        dl_test = dataset.prepare(
            "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GAT_model.eval()
        preds = []

        for data in test_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GATModel(nn.Module):
    def __init__(
        self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"
    ):
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
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()
