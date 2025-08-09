# Copyright (c) Microsoft Corporation.
import os
from torch.utils.data import Dataset, DataLoader

import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# ✅ Best Practice: Import only necessary functions or classes to reduce memory usage and improve readability.
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
# 🧠 ML Signal: Custom model class definition for machine learning
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path


class ADARNN(Model):
    """ADARNN Model

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
        pre_epoch=40,
        dw=0.5,
        loss_type="cosine",
        len_seq=60,
        len_win=0,
        lr=0.001,
        metric="mse",
        batch_size=2000,
        early_stop=20,
        # 🧠 ML Signal: Logging initialization parameters can be useful for debugging and model training analysis
        loss="mse",
        optimizer="adam",
        # 🧠 ML Signal: Logging initialization parameters can be useful for debugging and model training analysis
        n_splits=2,
        GPU=0,
        # ⚠️ SAST Risk (Low): Directly setting environment variables can lead to unexpected behavior in multi-threaded applications
        seed=None,
        **_,
    ):
        # Set logger.
        self.logger = get_module_logger("ADARNN")
        self.logger.info("ADARNN pytorch version...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.pre_epoch = pre_epoch
        self.dw = dw
        # ✅ Best Practice: Use consistent casing for string operations to avoid potential bugs
        self.loss_type = loss_type
        self.len_seq = len_seq
        # ⚠️ SAST Risk (Low): Potentially unsafe device selection without validation of GPU index
        # 🧠 ML Signal: Logging initialization parameters can be useful for debugging and model training analysis
        self.len_win = len_win
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_splits = n_splits
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.logger.info(
            "ADARNN parameters setting:"
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
                batch_size,
                early_stop,
                # 🧠 ML Signal: Setting seeds is important for reproducibility in ML experiments
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        n_hiddens = [hidden_size for _ in range(num_layers)]
        self.model = AdaRNN(
            use_bottleneck=False,
            bottleneck_width=64,
            n_input=d_feat,
            n_hiddens=n_hiddens,
            n_output=1,
            # 🧠 ML Signal: Logging model architecture can be useful for debugging and model training analysis
            dropout=dropout,
            # 🧠 ML Signal: Logging model size can be useful for resource management and optimization
            model_type="AdaRNN",
            len_seq=len_seq,
            # 🧠 ML Signal: Checks if a GPU is being used, which is common in ML for performance.
            trans_loss=loss_type,
        # ✅ Best Practice: Use consistent casing for string operations to avoid potential bugs
        # ✅ Best Practice: Using torch.device for device management is a good practice.
        )
        self.logger.info("model:\n{:}".format(self.model))
        # ✅ Best Practice: Explicitly checking against "cpu" improves code readability.
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.model)))
        # ✅ Best Practice: Use consistent casing for string operations to avoid potential bugs

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            # ⚠️ SAST Risk (Low): Raising a generic exception can make error handling difficult
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        # ⚠️ SAST Risk (Low): Moving models to devices without checking device availability can lead to runtime errors

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def train_AdaRNN(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
        self.model.train()
        criterion = nn.MSELoss()
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        out_weight_list = None
        for data_all in zip(*train_loader_list):
            #  for data_all in zip(*train_loader_list):
            self.train_optimizer.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                # feature :[36, 24, 6]
                feature, label_reg = data[0].to(self.device).float(), data[1].to(self.device).float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = get_index(len(data_all) - 1)
            for temp_index in index:
                s1 = temp_index[0]
                s2 = temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).to(self.device)
            # ⚠️ SAST Risk (Low): Clipping gradients can prevent exploding gradients but may hide underlying issues
            for i, n in enumerate(index):
                feature_s = list_feat[n[0]]
                feature_t = list_feat[n[1]]
                label_reg_s = list_label[n[0]]
                label_reg_t = list_label[n[1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                if epoch < self.pre_epoch:
                    pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
                        feature_all, len_win=self.len_win
                    )
                else:
                    pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(feature_all, weight_mat)
                    # 🧠 ML Signal: Use of correlation metrics to evaluate predictions
                    dist_mat = dist_mat + dist
                # 🧠 ML Signal: Use of Spearman correlation for ranking predictions
                pred_s = pred_all[0 : feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0) :]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)
                # 🧠 ML Signal: Calculation of mean correlation as a performance metric

                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer
            # ⚠️ SAST Risk (Low): Potential division by zero if ic.std() is zero
            self.train_optimizer.zero_grad()
            total_loss.backward()
            # 🧠 ML Signal: Calculation of mean rank correlation as a performance metric
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        # ⚠️ SAST Risk (Low): Potential division by zero if rank_ic.std() is zero
        # 🧠 ML Signal: Method for evaluating model performance on a dataset
        if epoch >= self.pre_epoch:
            if epoch > self.pre_epoch:
                # 🧠 ML Signal: Inference step using model predictions
                # 🧠 ML Signal: Use of mean squared error as a performance metric
                weight_mat = self.model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
            return weight_mat, dist_mat
        # ✅ Best Practice: Consistent naming for loss metric
        # ✅ Best Practice: Ensure labels are in the correct shape for comparison
        else:
            weight_mat = self.transform_type(out_weight_list)
            # ✅ Best Practice: Use of DataFrame to organize predictions and labels
            # 🧠 ML Signal: Method for logging metrics, useful for tracking model performance
            return weight_mat, None

    # 🧠 ML Signal: Calculation of performance metrics
    # ✅ Best Practice: Use of list comprehension for concise and readable code
    @staticmethod
    def calc_all_metrics(pred):
        # ✅ Best Practice: Return metrics for further analysis or logging
        # ✅ Best Practice: Joining list elements into a single string for logging
        # ⚠️ SAST Risk (Low): Potential information exposure if sensitive data is logged
        """pred is a pandas dataframe that has two attributes: score (pred) and label (real)"""
        res = {}
        ic = pred.groupby(level="datetime", group_keys=False).apply(lambda x: x.label.corr(x.score))
        rank_ic = pred.groupby(level="datetime", group_keys=False).apply(
            lambda x: x.label.corr(x.score, method="spearman")
        )
        # ✅ Best Practice: Consider using a more descriptive variable name for df_train and df_valid for clarity.
        res["ic"] = ic.mean()
        res["icir"] = ic.mean() / ic.std()
        res["ric"] = rank_ic.mean()
        res["ricir"] = rank_ic.mean() / rank_ic.std()
        res["mse"] = -(pred["label"] - pred["score"]).mean()
        res["loss"] = res["mse"]
        # 🧠 ML Signal: Usage of unique days to create training splits indicates time-series data handling.
        return res

    # 🧠 ML Signal: Splitting data into multiple parts for cross-validation or time-series validation.
    def test_epoch(self, df):
        self.model.eval()
        # ✅ Best Practice: Consider adding error handling for index out of range in slicing.
        preds = self.infer(df["feature"])
        label = df["label"].squeeze()
        # 🧠 ML Signal: Use of batch processing for training data.
        preds = pd.DataFrame({"label": label, "score": preds}, index=df.index)
        metrics = self.calc_all_metrics(preds)
        # ✅ Best Practice: Ensure save_path is a valid directory or handle exceptions.
        return metrics

    def log_metrics(self, mode, metrics):
        metrics = ["{}/{}: {:.6f}".format(k, mode, v) for k, v in metrics.items()]
        metrics = ", ".join(metrics)
        self.logger.info(metrics)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # 🧠 ML Signal: Custom training function indicating a specialized model training process.
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            # 🧠 ML Signal: Evaluation of model performance on training data.
            data_key=DataHandlerLP.DK_L,
        )
        # 🧠 ML Signal: Evaluation of model performance on validation data.
        #  splits = ['2011-06-30']
        days = df_train.index.get_level_values(level=0).unique()
        train_splits = np.array_split(days, self.n_splits)
        train_splits = [df_train[s[0] : s[-1]] for s in train_splits]
        train_loader_list = [get_stock_loader(df, self.batch_size) for df in train_splits]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        # ⚠️ SAST Risk (Low): Deep copy of model state could be memory intensive.
        self.logger.info("training...")
        self.fitted = True
        best_score = -np.inf
        best_epoch = 0
        # ⚠️ SAST Risk (Low): No check on the type or validity of 'dataset', which could lead to runtime errors.
        # 🧠 ML Signal: Implementation of early stopping to prevent overfitting.
        weight_mat, dist_mat = None, None

        # ⚠️ SAST Risk (Low): Raises a generic ValueError which might not be specific enough for error handling.
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # ✅ Best Practice: Using descriptive variable names like 'x_test' improves code readability.
            self.logger.info("training...")
            # ⚠️ SAST Risk (Low): Loading model state without validation could lead to corrupted state.
            weight_mat, dist_mat = self.train_AdaRNN(train_loader_list, step, dist_mat, weight_mat)
            # 🧠 ML Signal: Model evaluation mode is set, indicating inference phase
            # 🧠 ML Signal: The use of 'prepare' method on 'dataset' indicates a preprocessing step common in ML workflows.
            self.logger.info("evaluating...")
            # ⚠️ SAST Risk (Low): Saving model state without validation could lead to corrupted files.
            train_metrics = self.test_epoch(df_train)
            # 🧠 ML Signal: The 'infer' method suggests a prediction or inference step, typical in ML models.
            valid_metrics = self.test_epoch(df_valid)
            self.log_metrics("train: ", train_metrics)
            # ⚠️ SAST Risk (Low): Clearing GPU cache without checking could affect other processes.
            # ✅ Best Practice: Reshape and transpose operations are clearly separated for readability
            self.log_metrics("valid: ", valid_metrics)

            valid_score = valid_metrics[self.metric]
            # ✅ Best Practice: Using range with step size for batch processing
            train_score = train_metrics[self.metric]
            evals_result["train"].append(train_score)
            evals_result["valid"].append(valid_score)
            if valid_score > best_score:
                best_score = valid_score
                stop_steps = 0
                # ⚠️ SAST Risk (Low): Ensure x_values is sanitized before converting to tensor
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
            # 🧠 ML Signal: No gradient computation during inference
            else:
                stop_steps += 1
                # 🧠 ML Signal: Usage of torch.ones indicates a pattern of initializing tensors, which is common in ML model setups.
                # ⚠️ SAST Risk (Low): Ensure model.predict is safe and handles inputs correctly
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    # ✅ Best Practice: Consider using more descriptive variable names for i and j to improve code readability.
                    break
        # ✅ Best Practice: Using pd.Series for returning predictions with index

        # ✅ Best Practice: Class names should follow the CapWords convention for readability and consistency.
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # ⚠️ SAST Risk (Low): Accessing elements without bounds checking can lead to IndexError if init_weight dimensions are incorrect.
        self.model.load_state_dict(best_param)
        # 🧠 ML Signal: Use of DataFrame column selection, common in data preprocessing
        torch.save(best_param, save_path)

        # 🧠 ML Signal: Use of DataFrame column selection, common in data preprocessing
        if self.use_gpu:
            # 🧠 ML Signal: Use of DataFrame index, common in data preprocessing
            torch.cuda.empty_cache()
        return best_score

    # ✅ Best Practice: Explicitly specify dtype for tensor conversion for clarity and precision
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            # 🧠 ML Signal: Reshaping and transposing data, common in data preprocessing for ML models
            # 🧠 ML Signal: Accessing elements by index, common in data handling and preprocessing
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        # ✅ Best Practice: Returning a tuple for consistent output structure
        # 🧠 ML Signal: Custom implementation of __len__ method indicates class is likely a container or collection
        return self.infer(x_test)
    # ✅ Best Practice: Explicitly specify dtype for tensor conversion for clarity and precision

    # ✅ Best Practice: Using len() on an attribute suggests df_feature is a list-like or DataFrame object
    # 🧠 ML Signal: Function to create a data loader, common in ML data preprocessing
    def infer(self, x_test):
        index = x_test.index
        # ✅ Best Practice: Use of DataLoader for efficient data handling in batches
        self.model.eval()
        # ✅ Best Practice: Provide a docstring to describe the function's purpose and parameters
        x_values = x_test.values
        # 🧠 ML Signal: Returning a data loader, indicating usage in a training pipeline
        sample_num = x_values.shape[0]
        # ✅ Best Practice: Initialize variables before use
        x_values = x_values.reshape(sample_num, self.d_feat, -1).transpose(0, 2, 1)
        preds = []
        # 🧠 ML Signal: Iterating over a range, common pattern in loops

        for begin in range(sample_num)[:: self.batch_size]:
            # 🧠 ML Signal: Nested loop pattern, often used in combinatorial problems
            # ✅ Best Practice: Include a docstring to describe the class and its parameters
            if sample_num - begin < self.batch_size:
                # 🧠 ML Signal: Appending to a list, common pattern for building collections
                end = sample_num
            else:
                end = begin + self.batch_size
            # 🧠 ML Signal: Returning a list, common pattern for functions that generate collections

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model.predict(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

    def transform_type(self, init_weight):
        weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight


class data_loader(Dataset):
    def __init__(self, df):
        self.df_feature = df["feature"]
        self.df_label_reg = df["label"]
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available or index is invalid
        self.df_index = df.index
        self.df_feature = torch.tensor(
            self.df_feature.values.reshape(-1, 6, 60).transpose(0, 2, 1), dtype=torch.float32
        )
        self.df_label_reg = torch.tensor(self.df_label_reg.values.reshape(-1), dtype=torch.float32)
    # ✅ Best Practice: Use of nn.GRU for RNN layer, which is a standard practice for sequence models

    def __getitem__(self, index):
        sample, label_reg = self.df_feature[index], self.df_label_reg[index]
        return sample, label_reg
    # ✅ Best Practice: Use of nn.Sequential for chaining layers

    def __len__(self):
        return len(self.df_feature)


def get_stock_loader(df, batch_size, shuffle=True):
    train_loader = DataLoader(data_loader(df), batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_index(num_domain=2):
    # ✅ Best Practice: Initializing weights and biases for better training convergence
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index

# ✅ Best Practice: Use of Xavier initialization for weights

class AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(
        # ✅ Best Practice: Use of nn.Linear for gate weights
        self,
        use_bottleneck=False,
        bottleneck_width=256,
        n_input=128,
        # 🧠 ML Signal: Iterating over hidden layers to initialize weights and biases
        n_hiddens=[64, 64],
        n_output=6,
        # ✅ Best Practice: Use of nn.BatchNorm1d for normalization
        # 🧠 ML Signal: Initializing weights with a normal distribution
        dropout=0.0,
        len_seq=9,
        # 🧠 ML Signal: Initializing biases to zero
        model_type="AdaRNN",
        # ✅ Best Practice: Use of Softmax for output normalization
        trans_loss="mmd",
        GPU=0,
    # 🧠 ML Signal: Custom initialization method for layers
    ):
        super(AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        # ✅ Best Practice: Use of enumerate for index and value retrieval
        self.model_type = model_type
        self.trans_loss = trans_loss
        # 🧠 ML Signal: Use of custom loss function TransferLoss
        self.len_seq = len_seq
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        in_size = self.n_input
        # ✅ Best Practice: Use of range with step for loop control

        features = nn.ModuleList()
        # ✅ Best Practice: Use of range with step for loop control
        for hidden in n_hiddens:
            rnn = nn.GRU(input_size=in_size, num_layers=1, hidden_size=hidden, batch_first=True, dropout=dropout)
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck is True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                # ⚠️ SAST Risk (Low): Potential floating-point division by zero if len_win is 0
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            # ✅ Best Practice: Use of conditional expression for concise initialization
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            # 🧠 ML Signal: Use of GRU layer to extract features from input data
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            # 🧠 ML Signal: Collecting outputs from each layer for further processing
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            # ✅ Best Practice: Explicit comparison with False for clarity
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        # 🧠 ML Signal: Processing gate weights for adaptive RNN models
        if self.model_type == "AdaRNN":
            gate = nn.ModuleList()
            # ✅ Best Practice: Use descriptive variable names for better readability
            for i in range(len(n_hiddens)):
                # ✅ Best Practice: Returning multiple outputs for flexibility in usage
                gate_weight = nn.Linear(len_seq * self.hiddens[i] * 2, len_seq)
                gate.append(gate_weight)
            # 🧠 ML Signal: Use of sigmoid function indicates a binary classification or gating mechanism
            self.gate = gate

            # 🧠 ML Signal: Use of mean function to aggregate weights
            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                # 🧠 ML Signal: Use of softmax function indicates a multi-class classification
                bnlst.append(nn.BatchNorm1d(len_seq))
            # ✅ Best Practice: Initialize lists before the loop to collect results
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()
    # 🧠 ML Signal: Splitting features into source and target, indicating a common pattern in domain adaptation tasks

    def init_layers(self):
        for i in range(len(self.hiddens)):
            # ✅ Best Practice: Return multiple values as a tuple for clarity
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        out = self.gru_features(x)
        fea = out[0]  # [2N,L,H]
        if self.use_bottleneck is True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()  # [N,]
        # ✅ Best Practice: Use of default values for weight matrix when not provided

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            # 🧠 ML Signal: Use of custom loss function TransferLoss
            h_start = 0
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                # 🧠 ML Signal: Iterative computation of transfer loss
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    # ⚠️ SAST Risk (Low): Potential for numerical instability with loss accumulation
                    weight = (
                        # ✅ Best Practice: Use of a small epsilon value to avoid floating-point precision issues
                        out_weight_list[i][j]
                        if self.model_type == "AdaRNN"
                        # ✅ Best Practice: Detaching tensors to prevent gradient computation
                        else 1 / (self.len_seq - h_start) * (2 * len_win + 1)
                    )
                    # ✅ Best Practice: Detaching tensors to prevent gradient computation
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        n[:, j, :], out_list_t[i][:, k, :]
                    # 🧠 ML Signal: Identifying indices where the new distribution is greater than the old distribution
                    )
        return fc_out, loss_transfer, out_weight_list
    # 🧠 ML Signal: Updating weights based on a condition, common in boosting algorithms

    # 🧠 ML Signal: Method name 'predict' suggests this function is used for making predictions in a machine learning model
    def gru_features(self, x, predict=False):
        # 🧠 ML Signal: Normalizing weights, a common practice in machine learning models
        x_input = x
        # 🧠 ML Signal: Use of GRU (Gated Recurrent Unit) indicates a sequence processing model, common in time-series or NLP tasks
        out = None
        # ⚠️ SAST Risk (Low): Potential division by zero if weight_norm contains zeros
        out_lis = []
        # ✅ Best Practice: Returning the updated weight matrix
        # ✅ Best Practice: Explicit comparison to True is unnecessary; use 'if self.use_bottleneck:'
        out_weight_list = [] if (self.model_type == "AdaRNN") else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == "AdaRNN" and predict is False:
                # ✅ Best Practice: Use of default parameter values for flexibility
                # 🧠 ML Signal: Default parameter values indicate common usage patterns
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0 : int(out.shape[0] // 2)]
        # 🧠 ML Signal: Storing configuration parameters as instance variables
        x_t = out[out.shape[0] // 2 : out.shape[0]]
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if not validated
        # ✅ Best Practice: Use of torch.device for device management
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    @staticmethod
    # ✅ Best Practice: Use of a dictionary or mapping could improve readability and maintainability for loss functions.
    def get_features(output_list):
        fea_list_src, fea_list_tar = [], []
        # 🧠 ML Signal: Use of MMD loss with linear kernel indicates a specific adaptation strategy.
        for fea in output_list:
            fea_list_src.append(fea[0 : fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2 :])
        return fea_list_src, fea_list_tar
    # 🧠 ML Signal: Use of CORAL loss indicates a specific domain adaptation strategy.

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        # 🧠 ML Signal: Use of cosine similarity for loss indicates a specific adaptation strategy.
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            # 🧠 ML Signal: Use of KL divergence for loss indicates a specific adaptation strategy.
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            # 🧠 ML Signal: Use of JS divergence for loss indicates a specific adaptation strategy.
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        # 🧠 ML Signal: Use of MINE estimator indicates a specific adaptation strategy.
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        if weight_mat is None:
            weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq)).to(self.device)
        # 🧠 ML Signal: Use of adversarial loss indicates a specific adaptation strategy.
        else:
            # ✅ Best Practice: Consider renaming the function to reflect its purpose more clearly, such as `cosine_similarity_loss`.
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        # 🧠 ML Signal: Use of MMD loss with RBF kernel indicates a specific adaptation strategy.
        # ✅ Best Practice: Ensure input tensors are not empty to avoid runtime errors.
        for i, n in enumerate(out_list_s):
            criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            # ✅ Best Practice: Import nn from torch at the top of the file for clarity and maintainability.
            for j in range(self.len_seq):
                # 🧠 ML Signal: Use of pairwise distance indicates a specific adaptation strategy.
                # ⚠️ SAST Risk (Low): Ensure that source and target are tensors of the same shape to avoid unexpected behavior.
                # ✅ Best Practice: Use of @staticmethod decorator for methods that do not access instance or class data
                loss_trans = criterion_transder.compute(n[:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        # ⚠️ SAST Risk (Low): Calling mean on a scalar value may be unnecessary; ensure loss is a tensor.
        # 🧠 ML Signal: Function signature indicates a pattern for custom autograd functions in PyTorch
        return fc_out, loss_transfer, dist_mat, weight
    # 🧠 ML Signal: Storing variables in ctx is a common pattern for backward computation in PyTorch

    # For Boosting-based
    # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        # ✅ Best Practice: Using view_as for reshaping maintains the same shape as the input tensor
        epsilon = 1e-5
        # ✅ Best Practice: Use descriptive variable names for better readability
        # ✅ Best Practice: Clear and concise comment explaining the purpose of the backward method
        dist_old = dist_old.detach()
        # 🧠 ML Signal: Custom neural network module definition
        # This method reverses the gradient by multiplying with a negative alpha
        # ✅ Best Practice: Using @staticmethod for methods that do not access the instance is a good practice
        dist_new = dist_new.detach()
        # 🧠 ML Signal: The function returns a tuple, which is common in ML frameworks for gradients and additional data
        ind = dist_new > dist_old + epsilon
        # ✅ Best Practice: Use of default values for function parameters improves usability and flexibility.
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        # ✅ Best Practice: Storing parameters as instance variables enhances code readability and maintainability.
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        # 🧠 ML Signal: Custom gradient reversal function, useful for domain adaptation tasks
        return weight_mat
    # ✅ Best Practice: Storing parameters as instance variables enhances code readability and maintainability.

    def predict(self, x):
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.
        # 🧠 ML Signal: Use of ReLU activation function, common in neural networks
        out = self.gru_features(x, predict=True)
        fea = out[0]
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.
        # 🧠 ML Signal: Sequential layer processing, typical in neural network forward passes
        if self.use_bottleneck is True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            # 🧠 ML Signal: Use of sigmoid activation function, often used for binary classification
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            # ✅ Best Practice: Explicit return of the final output
            # ⚠️ SAST Risk (Low): Ensure that nn.BCELoss() is used with logits if the discriminator outputs logits, to prevent potential numerical instability.
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out
# 🧠 ML Signal: Usage of a discriminator network suggests adversarial training, common in domain adaptation tasks.


# 🧠 ML Signal: Creating domain labels for source and target, indicating a domain adaptation task.
class TransferLoss:
    def __init__(self, loss_type="cosine", input_dim=512, GPU=0):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        # 🧠 ML Signal: Use of ReverseLayerF indicates gradient reversal, a technique used in domain adversarial training.
        self.loss_type = loss_type
        self.input_dim = input_dim
        # 🧠 ML Signal: Function named 'CORAL' suggests a specific algorithm or method used in ML
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
    # 🧠 ML Signal: Passing reversed features through the discriminator, typical in adversarial domain adaptation.

    # 🧠 ML Signal: Usage of tensor size indicates handling of data dimensions, common in ML
    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        # 🧠 ML Signal: Matrix operations on tensors are common in ML for data transformation
        # 🧠 ML Signal: Default parameter values indicate common usage patterns
        loss = None
        # ✅ Best Practice: Use of default parameter values for flexibility
        if self.loss_type in ("mmd_lin", "mmd"):
            # 🧠 ML Signal: Calculation of loss is a common pattern in ML for optimization
            mmdloss = MMD_loss(kernel_type="linear")
            loss = mmdloss(X, Y)
        # 🧠 ML Signal: Normalization of loss is a common pattern in ML
        elif self.loss_type == "coral":
            loss = CORAL(X, Y, self.device)
        # ✅ Best Practice: Explicit return of the loss value improves readability
        elif self.loss_type in ("cosine", "cos"):
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            loss = 1 - cosine(X, Y)
        elif self.loss_type == "kl":
            loss = kl_div(X, Y)
        elif self.loss_type == "js":
            loss = js(X, Y)
        elif self.loss_type == "mine":
            mine_model = Mine_estimator(input_dim=self.input_dim, hidden_dim=60).to(self.device)
            loss = mine_model(X, Y)
        elif self.loss_type == "adv":
            loss = adv(X, Y, self.device, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == "mmd_rbf":
            mmdloss = MMD_loss(kernel_type="rbf")
            loss = mmdloss(X, Y)
        elif self.loss_type == "pairwise":
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            pair_mat = pairwise_dist(X, Y)
            loss = torch.norm(pair_mat)
        # ✅ Best Practice: Add input validation to ensure X and Y are numpy arrays

        return loss
# 🧠 ML Signal: Use of mean and dot product suggests statistical or ML computation


# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
def cosine(source, target):
    source, target = source.mean(), target.mean()
    cos = nn.CosineSimilarity(dim=0)
    # 🧠 ML Signal: Use of Gaussian kernel for MMD calculation
    loss = cos(source, target)
    return loss.mean()


class ReverseLayerF(Function):
    # ⚠️ SAST Risk (Low): Potential for numerical instability in mean calculations
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    # ✅ Best Practice: Class names should follow the CapWords convention for readability
    # 🧠 ML Signal: Calculation of MMD loss using kernel matrices
    @staticmethod
    def backward(ctx, grad_output):
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
        output = grad_output.neg() * ctx.alpha
        return output, None
# ✅ Best Practice: Initializing instance variables in the constructor

# 🧠 ML Signal: Use of torch.randperm for shuffling, common in data augmentation or permutation tests

# 🧠 ML Signal: Instantiation of a model with specific input and hidden dimensions
class Discriminator(nn.Module):
    # 🧠 ML Signal: Custom loss calculation using a model, indicative of advanced ML techniques
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        # 🧠 ML Signal: Use of shuffled data for marginal loss, a pattern in contrastive learning
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 🧠 ML Signal: Use of torch.mean and torch.log for custom loss computation
        # ✅ Best Practice: Inheriting from nn.Module is standard for defining custom neural network models in PyTorch.
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)
    # 🧠 ML Signal: Negating the result for loss minimization, common in optimization
    # ✅ Best Practice: Use of default values for function parameters improves flexibility and usability.

    def forward(self, x):
        # ✅ Best Practice: Explicit return of the loss value for clarity
        # ✅ Best Practice: Explicitly calling the superclass constructor ensures proper initialization.
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.
        # 🧠 ML Signal: Use of leaky_relu activation function indicates a pattern in neural network design
        x = torch.sigmoid(x)
        # ✅ Best Practice: Use of activation functions like leaky_relu is common in neural networks to introduce non-linearity
        return x
# 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.

# 🧠 ML Signal: Sequential layer processing is a common pattern in neural network forward methods

# 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
def adv(source, target, device, input_dim=256, hidden_dim=512):
    # 🧠 ML Signal: Returning the final layer output is a standard practice in model forward methods
    domain_loss = nn.BCELoss()
    # !!! Pay attention to .cuda !!!
    adv_net = Discriminator(input_dim, hidden_dim).to(device)
    # ⚠️ SAST Risk (Low): Using assert for input validation can be bypassed if Python is run with optimizations (e.g., python -O).
    domain_src = torch.ones(len(source)).to(device)
    domain_tar = torch.zeros(len(target)).to(device)
    # 🧠 ML Signal: Use of PyTorch tensor operations indicates potential ML model or data processing.
    domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
    reverse_src = ReverseLayerF.apply(source, 1)
    # 🧠 ML Signal: Use of PyTorch tensor operations indicates potential ML model or data processing.
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    # 🧠 ML Signal: Use of PyTorch tensor operations indicates potential ML model or data processing.
    pred_tar = adv_net(reverse_tar)
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode
    loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s + loss_t
    # 🧠 ML Signal: Use of np.expand_dims indicates manipulation of array dimensions
    return loss

# 🧠 ML Signal: Use of np.expand_dims indicates manipulation of array dimensions

# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
def CORAL(source, target, device):
    # 🧠 ML Signal: Use of np.tile indicates repetition of array elements
    d = source.size(1)
    # ⚠️ SAST Risk (Low): Ensure that X and Y are numpy arrays to avoid unexpected behavior with np.dot and np.sum.
    ns, nt = source.size(0), target.size(0)
    # 🧠 ML Signal: Use of np.tile indicates repetition of array elements

    # 🧠 ML Signal: Use of np.dot suggests matrix multiplication, common in ML algorithms.
    # source covariance
    # 🧠 ML Signal: Use of np.power and np.sum indicates computation of pairwise distances
    tmp_s = torch.ones((1, ns)).to(device) @ source
    # 🧠 ML Signal: np.sum and np.square are often used in ML for vectorized operations.
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    # ✅ Best Practice: Function definition should have a docstring explaining its purpose and parameters

    # ✅ Best Practice: Transposing a 1D array by wrapping it in brackets is a common pattern for reshaping.
    # target covariance
    # ✅ Best Practice: Check for length mismatch and adjust to avoid errors
    tmp_t = torch.ones((1, nt)).to(device) @ target
    # 🧠 ML Signal: Calculation of squared sums is typical in distance computations.
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # 🧠 ML Signal: This formula is used to compute the squared Euclidean distance.
    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    # ⚠️ SAST Risk (Low): Assumes nn is imported and KLDivLoss is used correctly
    loss = loss / (4 * d * d)

    # ⚠️ SAST Risk (Low): Assumes source has a log method; potential AttributeError if not
    # ✅ Best Practice: Check and adjust lengths of source and target to ensure they are equal
    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type="linear", kernel_mul=2.0, kernel_num=5):
        # ✅ Best Practice: Calculate the midpoint for Jensen-Shannon divergence
        super(MMD_loss, self).__init__()
        # ⚠️ SAST Risk (Low): Ensure kl_div function handles division by zero or log of zero
        # ✅ Best Practice: Return the average of the two loss values for final JS divergence
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    @staticmethod
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def linear_mmd(X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == "linear":
            return self.linear_mmd(source, target)
        elif self.kernel_type == "rbf":
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss


class Mine_estimator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine_estimator, self).__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X, Y):
        Y_shffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shffle)
        ret = torch.mean(loss_joint) - torch.log(torch.mean(torch.exp(loss_marginal)))
        loss = -ret
        return loss


class Mine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


def pairwise_dist(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = X.unsqueeze(1).expand(n, m, d)
    b = Y.unsqueeze(0).expand(n, m, d)
    return torch.pow(a - b, 2).sum(2)


def pairwise_dist_np(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = np.expand_dims(X, 1)
    b = np.expand_dims(Y, 0)
    a = np.tile(a, (1, m, 1))
    b = np.tile(b, (n, 1, 1))
    return np.power(a - b, 2).sum(2)


def pa(X, Y):
    XY = np.dot(X, Y.T)
    XX = np.sum(np.square(X), axis=1)
    XX = np.transpose([XX])
    YY = np.sum(np.square(Y), axis=1)
    dist = XX + YY - 2 * XY

    return dist


def kl_div(source, target):
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    criterion = nn.KLDivLoss(reduction="batchmean")
    loss = criterion(source.log(), target)
    return loss


def js(source, target):
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    M = 0.5 * (source + target)
    loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
    return 0.5 * (loss_1 + loss_2)