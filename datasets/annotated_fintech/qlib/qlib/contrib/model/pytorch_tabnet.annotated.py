# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

import numpy as np

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import pandas as pd
from typing import Text, Union

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import torch.nn.functional as F
from torch.autograd import Function

# ✅ Best Practice: Use of relative imports for better modularity and maintainability

# ✅ Best Practice: Use of relative imports for better modularity and maintainability
from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TabnetModel(Model):
    def __init__(
        self,
        d_feat=158,
        out_dim=64,
        final_out_dim=1,
        batch_size=4096,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        n_epochs=100,
        pretrain_n_epochs=50,
        relax=1.3,
        vbs=2048,
        seed=993,
        optimizer="adam",
        loss="mse",
        metric="",
        early_stop=20,
        GPU=0,
        pretrain_loss="custom",
        ps=0.3,
        lr=0.01,
        pretrain=True,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
        pretrain_file=None,
    ):
        """
        TabNet model for Qlib

        Args:
        ps: probability to generate the bernoulli mask
        """
        # set hyper-parameters.
        self.d_feat = d_feat
        self.out_dim = out_dim
        self.final_out_dim = final_out_dim
        # 🧠 ML Signal: Logging is used, which can be a signal for monitoring and debugging.
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        # ⚠️ SAST Risk (Low): Potential GPU index out of range if GPU is not available or index is invalid.
        self.pretrain_loss = pretrain_loss
        self.seed = seed
        self.ps = ps
        # ⚠️ SAST Risk (Low): Potential issue if pretrain_file is not a valid path or is None.
        self.n_epochs = n_epochs
        self.logger = get_module_logger("TabNet")
        self.pretrain_n_epochs = pretrain_n_epochs
        self.device = (
            "cuda:%s" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.loss = loss
        self.metric = metric
        self.early_stop = early_stop
        # 🧠 ML Signal: Logging model configuration details, useful for reproducibility and debugging.
        self.pretrain = pretrain
        self.pretrain_file = get_or_create_path(pretrain_file)
        self.logger.info(
            "TabNet:"
            "\nbatch_size : {}"
            "\nvirtual bs : {}"
            "\ndevice : {}"
            # ⚠️ SAST Risk (Low): Setting a random seed for reproducibility, but might not cover all sources of randomness.
            "\npretrain: {}".format(self.batch_size, vbs, self.device, self.pretrain)
        )
        self.fitted = False
        np.random.seed(self.seed)
        # ✅ Best Practice: Use of device-agnostic code to support both CPU and GPU.
        torch.manual_seed(self.seed)

        # 🧠 ML Signal: Logging model architecture details, useful for debugging and understanding model structure.
        self.tabnet_model = TabNet(
            inp_dim=self.d_feat, out_dim=self.out_dim, vbs=vbs, relax=relax
        ).to(self.device)
        self.tabnet_decoder = TabNet_Decoder(
            self.out_dim, self.d_feat, n_shared, n_ind, vbs, n_steps
        ).to(self.device)
        self.logger.info(
            "model:\n{:}\n{:}".format(self.tabnet_model, self.tabnet_decoder)
        )
        # ✅ Best Practice: Use of conditional logic to handle different optimizers.
        # 🧠 ML Signal: Logging model size, which can be important for deployment and resource allocation.
        self.logger.info(
            "model size: {:.4f} MB".format(
                count_parameters([self.tabnet_model, self.tabnet_decoder])
            )
        )

        if optimizer.lower() == "adam":
            self.pretrain_optimizer = optim.Adam(
                # 🧠 ML Signal: Checks if the computation is set to use GPU, indicating hardware preference
                list(self.tabnet_model.parameters())
                + list(self.tabnet_decoder.parameters()),
                lr=self.lr,
            )
            # ✅ Best Practice: Direct comparison with torch.device for clarity
            self.train_optimizer = optim.Adam(
                self.tabnet_model.parameters(), lr=self.lr
            )
        # ✅ Best Practice: Ensure the directory for the pretrain_file exists or is created

        # 🧠 ML Signal: Preparing dataset for pretraining
        elif optimizer.lower() == "gd":
            self.pretrain_optimizer = optim.SGD(
                list(self.tabnet_model.parameters())
                + list(self.tabnet_decoder.parameters()),
                lr=self.lr,
            )
            self.train_optimizer = optim.SGD(self.tabnet_model.parameters(), lr=self.lr)
        # ⚠️ SAST Risk (Low): Use of NotImplementedError to handle unsupported optimizers.
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

    # ✅ Best Practice: Handle missing values in the training dataset

    @property
    # ✅ Best Practice: Handle missing values in the validation dataset
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def pretrain_fn(self, dataset=DatasetH, pretrain_file="./pretrain/best.model"):
        get_or_create_path(pretrain_file)

        [df_train, df_valid] = dataset.prepare(
            ["pretrain", "pretrain_validation"],
            # 🧠 ML Signal: Logging the current epoch index
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
            # 🧠 ML Signal: Logging the start of pre-training
        )

        df_train.fillna(df_train.mean(), inplace=True)
        # 🧠 ML Signal: Logging the start of evaluation
        df_valid.fillna(df_valid.mean(), inplace=True)

        x_train = df_train["feature"]
        x_valid = df_valid["feature"]
        # 🧠 ML Signal: Logging the training and validation loss

        # Early stop setup
        stop_steps = 0
        # 🧠 ML Signal: Logging model saving event
        # ⚠️ SAST Risk (Low): Ensure the model is saved securely and the path is validated
        train_loss = 0
        best_loss = np.inf

        for epoch_idx in range(self.pretrain_n_epochs):
            self.logger.info("epoch: %s" % (epoch_idx))
            self.logger.info("pre-training...")
            self.pretrain_epoch(x_train)
            self.logger.info("evaluating...")
            # 🧠 ML Signal: Logging early stopping event
            train_loss = self.pretrain_test_epoch(x_train)
            valid_loss = self.pretrain_test_epoch(x_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_loss, valid_loss))

            if valid_loss < best_loss:
                self.logger.info("Save Model...")
                torch.save(self.tabnet_model.state_dict(), pretrain_file)
                best_loss = valid_loss
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    # ⚠️ SAST Risk (Low): Potential data leakage by filling NaN with mean of the entire training set
                    break

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        if self.pretrain:
            # there is a  pretrained model, load the model
            self.logger.info("Pretrain...")
            self.pretrain_fn(dataset, self.pretrain_file)
            self.logger.info("Load Pretrain model")
            self.tabnet_model.load_state_dict(
                torch.load(self.pretrain_file, map_location=self.device)
            )

        # adding one more linear layer to fit the final output dimension
        self.tabnet_model = FinetuneModel(
            self.out_dim, self.final_out_dim, self.tabnet_model
        ).to(self.device)
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )
        df_train.fillna(df_train.mean(), inplace=True)
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        # ⚠️ SAST Risk (Low): Saving model parameters without encryption or access control

        self.logger.info("training...")
        # ⚠️ SAST Risk (Low): Potential exception if 'self.fitted' is not a boolean
        self.fitted = True

        for epoch_idx in range(self.n_epochs):
            # 🧠 ML Signal: Usage of dataset preparation method
            self.logger.info("epoch: %s" % (epoch_idx))
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            # 🧠 ML Signal: Model evaluation mode set
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            # 🧠 ML Signal: Conversion of data to torch tensor
            valid_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            # ⚠️ SAST Risk (Low): Handling of NaN values by setting them to zero
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                # ✅ Best Practice: Use of batch processing for predictions
                best_score = val_score
                stop_steps = 0
                best_epoch = epoch_idx
                best_param = copy.deepcopy(self.tabnet_model.state_dict())
            else:
                stop_steps += 1
                # 🧠 ML Signal: Data moved to specified device for processing
                if stop_steps >= self.early_stop:
                    # 🧠 ML Signal: Conversion of data to torch tensors indicates usage of PyTorch for ML tasks
                    self.logger.info("early stop")
                    break
        # 🧠 ML Signal: Use of no_grad for inference
        # 🧠 ML Signal: Conversion of data to torch tensors indicates usage of PyTorch for ML tasks

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # ⚠️ SAST Risk (Low): Replacing NaNs with 0 might lead to misleading results if NaNs are significant
        # 🧠 ML Signal: Model prediction and conversion to numpy
        self.tabnet_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        # ⚠️ SAST Risk (Low): Replacing NaNs with 0 might lead to misleading results if NaNs are significant

        # 🧠 ML Signal: Returning predictions as a pandas Series
        if self.use_gpu:
            # 🧠 ML Signal: Setting model to evaluation mode is a common practice in ML model evaluation
            torch.cuda.empty_cache()

    # ✅ Best Practice: Initializing lists to store scores and losses for later aggregation
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        # 🧠 ML Signal: Use of numpy to handle indices suggests integration of numpy with PyTorch

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        # ✅ Best Practice: Iterating in batches improves performance and memory usage
        index = x_test.index
        self.tabnet_model.eval()
        # ✅ Best Practice: Breaking loop if remaining data is less than batch size
        x_values = torch.from_numpy(x_test.values)
        x_values[torch.isnan(x_values)] = 0
        sample_num = x_values.shape[0]
        # 🧠 ML Signal: Conversion to float and moving to device indicates preparation for model input
        preds = []

        # 🧠 ML Signal: Conversion to float and moving to device indicates preparation for model input
        # 🧠 ML Signal: Conversion of data to torch tensors indicates usage of PyTorch for ML model training
        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                # 🧠 ML Signal: Use of priors suggests a specific model architecture or requirement
                # 🧠 ML Signal: Conversion of data to torch tensors indicates usage of PyTorch for ML model training
                end = sample_num
            else:
                # ✅ Best Practice: Using torch.no_grad() to prevent gradient computation during evaluation
                # ⚠️ SAST Risk (Low): Replacing NaNs with 0 might lead to incorrect model training if NaNs have significance
                end = begin + self.batch_size

            # 🧠 ML Signal: Model prediction step in evaluation
            # ⚠️ SAST Risk (Low): Replacing NaNs with 0 might lead to incorrect model training if NaNs have significance
            x_batch = x_values[begin:end].float().to(self.device)
            priors = torch.ones(end - begin, self.d_feat).to(self.device)
            # 🧠 ML Signal: Setting model to training mode is a common pattern in ML model training
            # 🧠 ML Signal: Calculation of loss indicates supervised learning

            with torch.no_grad():
                # ✅ Best Practice: Storing loss values for later aggregation
                # 🧠 ML Signal: Shuffling data is a common practice in ML to ensure model generalization
                pred = self.tabnet_model(x_batch, priors).detach().cpu().numpy()

            # 🧠 ML Signal: Calculation of metric score for model evaluation
            preds.append(pred)

        # ✅ Best Practice: Storing score values for later aggregation
        return pd.Series(np.concatenate(preds), index=index)

    # ✅ Best Practice: Returning mean of losses and scores for overall evaluation
    # 🧠 ML Signal: Usage of batch processing is a common pattern in ML model training
    def test_epoch(self, data_x, data_y):
        # prepare training data
        # 🧠 ML Signal: Usage of batch processing is a common pattern in ML model training
        x_values = torch.from_numpy(data_x.values)
        y_values = torch.from_numpy(np.squeeze(data_y.values))
        # 🧠 ML Signal: Use of priors in model prediction indicates a specific model architecture or approach
        # 🧠 ML Signal: Conversion of data to torch tensor for model training
        x_values[torch.isnan(x_values)] = 0
        y_values[torch.isnan(y_values)] = 0
        # 🧠 ML Signal: Model prediction step in training loop
        # ⚠️ SAST Risk (Low): Handling NaN values by replacing them with 0, which might not be appropriate for all datasets
        self.tabnet_model.eval()

        # 🧠 ML Signal: Calculation of loss is a key step in ML model training
        # 🧠 ML Signal: Shuffling data indices for training
        scores = []
        losses = []
        # 🧠 ML Signal: Zeroing gradients is a standard step in the optimization process

        # 🧠 ML Signal: Setting models to training mode
        indices = np.arange(len(x_values))
        # 🧠 ML Signal: Backpropagation step in training loop

        for i in range(len(indices))[:: self.batch_size]:
            # ⚠️ SAST Risk (Low): Clipping gradients to prevent exploding gradients, but might hide underlying issues
            if len(indices) - i < self.batch_size:
                break
            # 🧠 ML Signal: Optimizer step to update model parameters
            feature = x_values[indices[i : i + self.batch_size]].float().to(self.device)
            # 🧠 ML Signal: Randomly generating a mask for feature selection
            label = y_values[indices[i : i + self.batch_size]].float().to(self.device)
            priors = torch.ones(self.batch_size, self.d_feat).to(self.device)
            # 🧠 ML Signal: Applying mask to training data
            with torch.no_grad():
                pred = self.tabnet_model(feature, priors)
                loss = self.loss_fn(pred, label)
                # ✅ Best Practice: Ensure tensors are moved to the correct device
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        # 🧠 ML Signal: Conversion of data to torch tensor for model input
        # 🧠 ML Signal: Forward pass through the model
        return np.mean(losses), np.mean(scores)

    # ⚠️ SAST Risk (Low): Replacing NaNs with 0 might lead to data integrity issues
    def train_epoch(self, x_train, y_train):
        # 🧠 ML Signal: Calculation of loss for backpropagation
        x_train_values = torch.from_numpy(x_train.values)
        # 🧠 ML Signal: Use of indices for batch processing
        y_train_values = torch.from_numpy(np.squeeze(y_train.values))
        # ✅ Best Practice: Zeroing gradients before backpropagation
        x_train_values[torch.isnan(x_train_values)] = 0
        # 🧠 ML Signal: Model evaluation mode set for inference
        y_train_values[torch.isnan(y_train_values)] = 0
        # 🧠 ML Signal: Backpropagation step
        self.tabnet_model.train()
        # 🧠 ML Signal: Decoder evaluation mode set for inference

        # 🧠 ML Signal: Optimizer step to update model parameters
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        # 🧠 ML Signal: Iterating over data in batches

        for i in range(len(indices))[:: self.batch_size]:
            # ✅ Best Practice: Early exit for incomplete batch
            if len(indices) - i < self.batch_size:
                break

            # 🧠 ML Signal: Random mask generation for feature selection
            feature = (
                x_train_values[indices[i : i + self.batch_size]].float().to(self.device)
            )
            label = (
                y_train_values[indices[i : i + self.batch_size]].float().to(self.device)
            )
            # 🧠 ML Signal: Masking input features for training
            priors = torch.ones(self.batch_size, self.d_feat).to(self.device)
            pred = self.tabnet_model(feature, priors)
            # 🧠 ML Signal: Masking target features for training
            loss = self.loss_fn(pred, label)

            # 🧠 ML Signal: Data preparation for model input
            self.train_optimizer.zero_grad()
            # 🧠 ML Signal: Data preparation for model input
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.tabnet_model.parameters(), 3.0)
            self.train_optimizer.step()

    # 🧠 ML Signal: Mask preparation for model input
    # ✅ Best Practice: Use descriptive variable names for better readability

    def pretrain_epoch(self, x_train):
        # 🧠 ML Signal: Priors calculation for model input
        # ✅ Best Practice: Use descriptive variable names for better readability
        train_set = torch.from_numpy(x_train.values)
        train_set[torch.isnan(train_set)] = 0
        # 🧠 ML Signal: No gradient calculation for inference
        # ✅ Best Practice: Use descriptive variable names for better readability
        # ✅ Best Practice: Consider adding type hints for function parameters and return type
        indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        # 🧠 ML Signal: Model inference
        # 🧠 ML Signal: Custom loss function implementation
        # ✅ Best Practice: Use descriptive variable names for better readability

        self.tabnet_model.train()
        # 🧠 ML Signal: Conditional logic based on a class attribute
        # 🧠 ML Signal: Decoder inference
        self.tabnet_decoder.train()
        # ✅ Best Practice: Use of descriptive function name for clarity

        # 🧠 ML Signal: Loss calculation for training
        # 🧠 ML Signal: Use of masking to handle missing values
        for i in range(len(indices))[:: self.batch_size]:
            # ✅ Best Practice: Use of torch.isfinite to handle NaN or infinite values
            if len(indices) - i < self.batch_size:
                # 🧠 ML Signal: Collecting loss values for analysis
                # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
                break
            # 🧠 ML Signal: Conditional logic based on metric type

            # 🧠 ML Signal: Aggregating loss values for epoch result
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            S_mask = torch.bernoulli(
                torch.empty(self.batch_size, self.d_feat).fill_(self.ps)
            )
            # 🧠 ML Signal: Use of mask to filter predictions and labels
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1 - S_mask)
            # 🧠 ML Signal: Use of mean squared error (MSE) loss function, common in regression tasks
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)
            # ⚠️ SAST Risk (Low): Potential for unhandled exception if metric is unknown

            # ⚠️ SAST Risk (Low): Assumes pred and label are compatible tensors; no input validation
            # ✅ Best Practice: Include a docstring to describe the purpose and functionality of the class
            S_mask = S_mask.to(self.device)
            feature = x_train_values.float().to(self.device)
            label = y_train_values.float().to(self.device)
            priors = 1 - S_mask
            # ✅ Best Practice: Call to super() ensures proper initialization of the base class
            (vec, sparse_loss) = self.tabnet_model(feature, priors)
            f = self.tabnet_decoder(vec)
            # 🧠 ML Signal: Storing a trained model as an instance variable
            loss = self.pretrain_loss_fn(label, f, S_mask)
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters

            # 🧠 ML Signal: Creating a linear layer with specified input and output dimensions
            self.pretrain_optimizer.zero_grad()
            # 🧠 ML Signal: Usage of a neural network model's forward pass
            # 🧠 ML Signal: Class definition for a neural network module, common in ML models
            loss.backward()
            # 🧠 ML Signal: Use of priors suggests probabilistic modeling or Bayesian methods
            self.pretrain_optimizer.step()

    # ✅ Best Practice: Ensure that the model and fc attributes are initialized in the class constructor
    # ✅ Best Practice: Use of descriptive variable names improves code readability.
    def pretrain_test_epoch(self, x_train):
        # ⚠️ SAST Risk (Low): Potential for attribute access errors if model or fc are not properly initialized
        train_set = torch.from_numpy(x_train.values)
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models.
        # 🧠 ML Signal: Method definition in a class, common in ML models for forward pass
        train_set[torch.isnan(train_set)] = 0
        indices = np.arange(len(train_set))
        # 🧠 ML Signal: Feature transformation step, typical in ML model layers

        # 🧠 ML Signal: Custom neural network module definition
        self.tabnet_model.eval()
        # 🧠 ML Signal: Returning the result of a fully connected layer, common in ML models
        self.tabnet_decoder.eval()

        losses = []

        # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            # ✅ Best Practice: Using nn.ModuleList for shared layers allows for proper parameter registration
            S_mask = torch.bernoulli(
                torch.empty(self.batch_size, self.d_feat).fill_(self.ps)
            )
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1 - S_mask)
            # 🧠 ML Signal: Use of nn.Linear indicates a fully connected layer, common in neural networks
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)

            feature = x_train_values.float().to(self.device)
            # 🧠 ML Signal: Iterative addition of layers suggests a configurable network depth
            label = y_train_values.float().to(self.device)
            S_mask = S_mask.to(self.device)
            priors = 1 - S_mask
            with torch.no_grad():
                (vec, sparse_loss) = self.tabnet_model(feature, priors)
                # ✅ Best Practice: Initialize tensors on the same device as input to avoid device mismatch errors
                # ✅ Best Practice: Using nn.ModuleList for steps allows for proper parameter registration
                f = self.tabnet_decoder(vec)

                # 🧠 ML Signal: Iterating over a sequence of operations (steps) is a common pattern in neural network layers
                loss = self.pretrain_loss_fn(label, f, S_mask)
            # 🧠 ML Signal: Use of custom DecoderStep class indicates a modular design pattern
            losses.append(loss.item())
        # 🧠 ML Signal: Accumulating results in a loop is a common pattern in neural network forward passes
        # ✅ Best Practice: Inheriting from nn.Module is standard for PyTorch models

        return np.mean(losses)

    # 🧠 ML Signal: Constructor with default hyperparameters for a neural network model

    def pretrain_loss_fn(self, f_hat, f, S):
        """
        Pretrain loss function defined in the original paper, read "Tabular self-supervised learning" in https://arxiv.org/pdf/1908.07442.pdf
        """
        down_mean = torch.mean(f, dim=0)
        down = torch.sqrt(torch.sum(torch.square(f - down_mean), dim=0))
        up = (f_hat - f) * S
        return torch.sum(torch.square(up / down))

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    # ✅ Best Practice: Use of nn.ModuleList to store layers for better integration with PyTorch

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    # ✅ Best Practice: Initializing the first step with a FeatureTransformer

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)


# ✅ Best Practice: Appending DecisionStep instances to a ModuleList for sequential processing


# ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode, potentially hiding issues.
# ✅ Best Practice: Use of nn.Linear for the final fully connected layer
class FinetuneModel(nn.Module):
    """
    FinuetuneModel for adding a layer by the end
    """

    # 🧠 ML Signal: Storing model hyperparameters as instance variables
    # ✅ Best Practice: Use descriptive variable names for clarity and maintainability.

    def __init__(self, input_dim, output_dim, trained_model):
        # ✅ Best Practice: Initialize lists outside loops to avoid repeated allocations.
        super().__init__()
        self.model = trained_model
        # ✅ Best Practice: Use device-agnostic code to ensure compatibility with different hardware.
        self.fc = nn.Linear(input_dim, output_dim)

    # 🧠 ML Signal: Iterative processing of steps indicates a sequential model or layer application.
    def forward(self, x, priors):
        # ✅ Best Practice: Include a docstring to describe the class and its arguments
        return self.fc(self.model(x, priors)[0]).squeeze()  # take the vec out


# 🧠 ML Signal: Use of custom step function suggests a modular or flexible model architecture.
# ✅ Best Practice: Use in-place operations where possible to save memory.


class DecoderStep(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs)
        # ✅ Best Practice: Update variables consistently to avoid unintended side effects.
        # ✅ Best Practice: Accumulate losses in a list for later aggregation.
        self.fc = nn.Linear(out_dim, out_dim)

    # ✅ Best Practice: Call to super() ensures proper initialization of the parent class

    # ✅ Best Practice: Return a tuple for multiple outputs to maintain consistency and clarity.
    def forward(self, x):
        # 🧠 ML Signal: Use of BatchNorm1d indicates a pattern for normalizing inputs in neural networks
        x = self.fea_tran(x)
        return self.fc(x)


# 🧠 ML Signal: Use of a default value for vbs suggests a common pattern or heuristic in model configuration
# 🧠 ML Signal: Conditional logic based on input size, indicating dynamic behavior

# 🧠 ML Signal: Direct return of batch normalization on input


class TabNet_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim, n_shared, n_ind, vbs, n_steps):
        """
        TabNet decoder that is used in pre-training
        # ✅ Best Practice: Include a docstring to describe the purpose and arguments of the class
        """
        # 🧠 ML Signal: Applying batch normalization to each chunk
        # 🧠 ML Signal: Concatenating results after processing
        super().__init__()
        self.out_dim = out_dim
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * out_dim))
            for x in range(n_shared - 1):
                self.shared.append(
                    nn.Linear(out_dim, 2 * out_dim)
                )  # preset the linear function we will use
        # ✅ Best Practice: Use of conditional assignment to handle optional parameters
        else:
            self.shared = None
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network layer, common in ML models
        for x in range(n_steps):
            self.steps.append(DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs))

    # 🧠 ML Signal: Use of batch normalization (GBN) is a common pattern in ML models

    # 🧠 ML Signal: Use of batch normalization and fully connected layers indicates a common pattern in neural network design.
    def forward(self, x):
        # 🧠 ML Signal: Storing output dimension, often used in ML models for layer configuration
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        # 🧠 ML Signal: Use of element-wise multiplication and sigmoid activation is a common pattern in neural network layers.
        # ✅ Best Practice: Class docstring provides a clear explanation of the class and its parameters
        for step in self.steps:
            out += step(x)
        return out


class TabNet(nn.Module):
    def __init__(
        self,
        inp_dim=6,
        out_dim=6,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        relax=1.2,
        vbs=1024,
    ):
        """
        TabNet AKA the original encoder

        Args:
            n_d: dimension of the features used to calculate the final results
            n_a: dimension of the features input to the attention transformer of the next step
            n_shared: numbr of shared steps in feature transformer(optional)
            n_ind: number of independent steps in feature transformer
            n_steps: number of steps of pass through tabbet
            relax coefficient:
            virtual batch size:
        """
        # ✅ Best Practice: Consider returning both 'mask' and modified 'priors' if 'priors' is intended to be used outside this function.
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            # 🧠 ML Signal: Conditional logic based on the presence of shared layers
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                # 🧠 ML Signal: Iterating over shared layers to append to the module list
                self.shared.append(
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a))
                )  # preset the linear function we will use
        else:
            self.shared = None

        self.first_step = FeatureTransformer(
            inp_dim, n_d + n_a, self.shared, n_ind, vbs
        )
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            # 🧠 ML Signal: Handling the case where no shared layers are present
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        # 🧠 ML Signal: Use of a forward method suggests this is part of a neural network model
        # 🧠 ML Signal: Iterating over independent layers to append to the module list
        self.n_d = n_d

    # ⚠️ SAST Risk (Low): Direct use of numpy function without input validation
    # 🧠 ML Signal: Iterating over layers indicates a sequential model structure
    def forward(self, x, priors):
        assert not torch.isnan(x).any()
        x = self.bn(x)
        # ⚠️ SAST Risk (Low): Potential for unexpected behavior if glu is not a callable
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss = []
        # 🧠 ML Signal: Use of element-wise operations on tensors
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, loss = step(x, x_a, priors)
            # ✅ Best Practice: Include a docstring to describe the purpose of the class
            # ⚠️ SAST Risk (Low): Potential for unexpected behavior if glu is not a callable
            out += F.relu(
                x_te[:, : self.n_d]
            )  # split the feature from feat_transformer
            x_a = x_te[:, self.n_d :]
            sparse_loss.append(loss)
        # 🧠 ML Signal: Use of element-wise operations on tensors
        return self.fc(out), sum(sparse_loss)


# ✅ Best Practice: Call to super() ensures proper initialization of the base class


# 🧠 ML Signal: Use of AttentionTransformer indicates a pattern for attention mechanisms in ML models
class GBN(nn.Module):
    """
    Ghost Batch Normalization
    an efficient way of doing batch normalization

    Args:
        vbs: virtual batch size
    """

    # ⚠️ SAST Risk (Low): Multiplying `x` by `mask` could lead to unintended zeroing of elements
    # 🧠 ML Signal: Accessing tensor size, common operation in tensor manipulation

    def __init__(self, inp, vbs=1024, momentum=0.01):
        # ⚠️ SAST Risk (Low): Assumes input is a tensor, which may not always be the case
        # 🧠 ML Signal: Returning a tuple with transformed data and a loss value is common in training loops
        super().__init__()
        # 🧠 ML Signal: Creating a range tensor, useful for learning tensor operations
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    # 🧠 ML Signal: Creating a view list for reshaping, common in tensor operations
    # 🧠 ML Signal: Modifying view for reshaping, common pattern in tensor manipulation
    # ✅ Best Practice: Use of @staticmethod decorator for methods that do not access instance or class data

    def forward(self, x):
        if x.size(0) <= self.vbs:  # can not be chunked
            return self.bn(x)
        # 🧠 ML Signal: Reshaping and transposing tensor, useful for learning tensor transformations
        else:
            # 🧠 ML Signal: Use of forward method indicates a custom autograd function for neural networks
            chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
            # ✅ Best Practice: Use of max with keepdim=True to maintain dimensions for broadcasting
            res = [self.bn(y) for y in chunk]
            # 🧠 ML Signal: Saving tensors for backward pass is a common pattern in custom autograd functions
            return torch.cat(res, 0)


# ✅ Best Practice: In-place operation to save memory

# 🧠 ML Signal: Sorting input tensor is a common operation in neural network layers


# 🧠 ML Signal: Custom threshold and support function for sparsemax, indicating advanced ML operation
class GLU(nn.Module):
    """
    GLU block that extracts only the most essential information

    Args:
        vbs: virtual batch size
    """

    # ✅ Best Practice: Use of in-place operations to potentially save memory
    # 🧠 ML Signal: Summing over dimensions is a common pattern in tensor manipulation

    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        # 🧠 ML Signal: Division and subtraction in tensor operations are common in normalization
        # ✅ Best Practice: Use of sum with dim argument for clarity and efficiency
        super().__init__()
        if fc:
            # 🧠 ML Signal: Use of torch.max for element-wise maximum is common in activation functions
            # ✅ Best Practice: Use of unsqueeze for maintaining dimensionality
            self.fc = fc
        else:
            # ✅ Best Practice: Use of torch.where for conditional operations
            # ✅ Best Practice: Function definition should have a docstring explaining its purpose and parameters.
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        # 🧠 ML Signal: Return of gradient input for backpropagation
        # 🧠 ML Signal: Sorting input tensor, common in ML for ranking or thresholding operations.
        self.od = out_dim

    # 🧠 ML Signal: Cumulative sum calculation, often used in ML for normalization or thresholding.
    def forward(self, x):
        x = self.bn(self.fc(x))
        # 🧠 ML Signal: Use of backward method indicates custom gradient computation
        # 🧠 ML Signal: Creating an index-like tensor, useful for operations that require positional information.
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))


# 🧠 ML Signal: Accessing saved tensors for gradient computation is a common pattern
# 🧠 ML Signal: Creating masks based on tensor values is common in gradient calculations
# 🧠 ML Signal: Counting non-zero elements is a common operation in custom gradients
# 🧠 ML Signal: Element-wise multiplication and summation are typical in gradient calculations
# 🧠 ML Signal: Element-wise operations and broadcasting are common in gradient adjustments
# 🧠 ML Signal: Boolean mask creation, common in ML for filtering or selecting elements.
# 🧠 ML Signal: Summing over a dimension, often used in ML for aggregation or counting.
# 🧠 ML Signal: Gathering specific elements, common in ML for selecting top-k or thresholded values.
# 🧠 ML Signal: Division by support size, typical in ML for averaging or normalization.
# ✅ Best Practice: Returning multiple values as a tuple, clear and idiomatic in Python.


class AttentionTransformer(nn.Module):
    """
    Args:
        relax: relax coefficient. The greater it is, we can
        use the same features more. When it is set to 1
        we can use every feature only once
    """

    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    # a:feature from previous decision step
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)  # updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = float(np.sqrt(0.5))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    """
    One step for the TabNet
    """

    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs):
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss


def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    SparseMax function for replacing reLU
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size
