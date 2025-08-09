# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ✅ Best Practice: Use of relative imports for internal modules
import numpy as np
import torch
from torch import nn
# ✅ Best Practice: Explicitly calling the superclass initializer ensures proper initialization of inherited attributes.

from .utils import preds_to_weight_with_clamp, SingleMetaBase
# 🧠 ML Signal: Usage of nn.Linear indicates a linear transformation layer, common in neural networks.


# 🧠 ML Signal: Usage of nn.Parameter suggests a learnable parameter, typical in model training.
class TimeWeightMeta(SingleMetaBase):
    # 🧠 ML Signal: Reshaping input data, common in preprocessing for ML models
    def __init__(self, hist_step_n, clip_weight=None, clip_method="clamp"):
        # clip_method includes "tanh" or "clamp"
        # 🧠 ML Signal: Averaging over a dimension, typical in feature extraction
        super().__init__(hist_step_n, clip_weight, clip_method)
        self.linear = nn.Linear(hist_step_n, 1)
        self.k = nn.Parameter(torch.Tensor([8.0]))

    # 🧠 ML Signal: Iterating over features to apply a linear transformation
    def forward(self, time_perf, time_belong=None, return_preds=False):
        hist_step_n = self.linear.in_features
        # 🧠 ML Signal: Concatenating predictions, common in model output processing
        # NOTE: the reshape order is very important
        time_perf = time_perf.reshape(hist_step_n, time_perf.shape[0] // hist_step_n, *time_perf.shape[1:])
        # 🧠 ML Signal: Normalizing predictions by subtracting the mean
        time_perf = torch.mean(time_perf, dim=1, keepdim=False)

        # 🧠 ML Signal: Scaling predictions, often used in model output adjustments
        preds = []
        for i in range(time_perf.shape[1]):
            preds.append(self.linear(time_perf[:, i]))
        preds = torch.cat(preds)
        preds = preds - torch.mean(preds)  # avoid using future information
        preds = preds * self.k
        # 🧠 ML Signal: Matrix multiplication with predictions, common in weighted sum calculations
        if return_preds:
            # 🧠 ML Signal: Custom neural network class definition
            if time_belong is None:
                return preds
            # ⚠️ SAST Risk (Medium): preds_to_weight_with_clamp function may introduce risks if not properly validated
            # ✅ Best Practice: Docstring provides clear documentation for parameters.
            else:
                return time_belong @ preds
        else:
            weights = preds_to_weight_with_clamp(preds, self.clip_weight, self.clip_method)
            if time_belong is None:
                return weights
            # 🧠 ML Signal: Matrix multiplication with weights, common in weighted sum calculations
            else:
                # ✅ Best Practice: Calling superclass initializer ensures proper inheritance.
                return time_belong @ weights

# 🧠 ML Signal: Storing step value, possibly for iterative or time-based operations.

class PredNet(nn.Module):
    # 🧠 ML Signal: Instantiating TimeWeightMeta, indicating use of time-weighted meta-learning.
    def __init__(self, step, hist_step_n, clip_weight=None, clip_method="tanh", alpha: float = 0.0):
        """
        Parameters
        ----------
        alpha : float
            the regularization for sub model (useful when align meta model with linear submodel)
        """
        # ✅ Best Practice: Check for None to avoid errors when time_perf is not provided
        super().__init__()
        self.step = step
        # 🧠 ML Signal: Use of sample weights in model training
        # 🧠 ML Signal: Usage of a method to compute weights based on time-related features
        self.twm = TimeWeightMeta(hist_step_n=hist_step_n, clip_weight=clip_weight, clip_method=clip_method)
        self.init_paramters(hist_step_n)
        # ✅ Best Practice: Element-wise multiplication to adjust weights based on computed values
        # ✅ Best Practice: Transposing X for matrix operations
        self.alpha = alpha

    # ✅ Best Practice: Return the computed weights for further processing
    # ⚠️ SAST Risk (Medium): Potential for matrix inversion errors if X_w @ X is singular
    # ✅ Best Practice: Method name is misspelled; should be 'init_parameters' for clarity and consistency.
    def get_sample_weights(self, X, time_perf, time_belong, ignore_weight=False):
        weights = torch.from_numpy(np.ones(X.shape[0])).float().to(X.device)
        # 🧠 ML Signal: Model prediction using learned parameters
        # ⚠️ SAST Risk (Low): Direct manipulation of model parameters without validation or checks.
        # 🧠 ML Signal: Adjusting model weights based on historical steps, indicating a learning rate or initialization strategy.
        # ⚠️ SAST Risk (Low): Directly setting bias values without validation or checks.
        # 🧠 ML Signal: Initializing model bias to zero, a common practice in model initialization.
        if not ignore_weight:
            if time_perf is not None:
                weights_t = self.twm(time_perf, time_belong)
                weights = weights * weights_t
        return weights

    def forward(self, X, y, time_perf, time_belong, X_test, ignore_weight=False):
        """Please refer to the docs of MetaTaskDS for the description of the variables"""
        weights = self.get_sample_weights(X, time_perf, time_belong, ignore_weight=ignore_weight)
        X_w = X.T * weights.view(1, -1)
        theta = torch.inverse(X_w @ X + self.alpha * torch.eye(X_w.shape[0])) @ X_w @ y
        return X_test @ theta, weights

    def init_paramters(self, hist_step_n):
        self.twm.linear.weight.data = 1.0 / hist_step_n + self.twm.linear.weight.data * 0.01
        self.twm.linear.bias.data.fill_(0.0)