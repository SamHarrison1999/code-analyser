# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import lightgbm as lgb
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
import numpy as np
import pandas as pd
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from typing import Text, Union
from ...model.base import Model
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
from ...model.interpret.base import FeatureInt
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# ✅ Best Practice: Class docstring provides a brief description of the class
from ...log import get_module_logger


class DEnsembleModel(Model, FeatureInt):
    """Double Ensemble Model"""

    def __init__(
        self,
        base_model="gbm",
        loss="mse",
        num_models=6,
        enable_sr=True,
        enable_fs=True,
        alpha1=1.0,
        alpha2=1.0,
        bins_sr=10,
        bins_fs=5,
        decay=None,
        # 🧠 ML Signal: Storing model configuration parameters
        sample_ratios=None,
        sub_weights=None,
        # 🧠 ML Signal: Storing model configuration parameters
        epochs=100,
        early_stopping_rounds=None,
        # 🧠 ML Signal: Storing model configuration parameters
        **kwargs,
    ):
        # 🧠 ML Signal: Storing model configuration parameters
        self.base_model = base_model  # "gbm" or "mlp", specifically, we use lgbm for "gbm"
        self.num_models = num_models  # the number of sub-models
        # 🧠 ML Signal: Storing model configuration parameters
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        # 🧠 ML Signal: Storing model configuration parameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        # 🧠 ML Signal: Storing model configuration parameters
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        # 🧠 ML Signal: Storing model configuration parameters
        self.decay = decay
        if sample_ratios is None:  # the default values for sample_ratios
            # 🧠 ML Signal: Storing model configuration parameters
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:  # the default values for sub_weights
            # ✅ Best Practice: Use default values for mutable arguments to avoid shared state
            sub_weights = [1] * self.num_models
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        # ✅ Best Practice: Use default values for mutable arguments to avoid shared state
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")
        # ⚠️ SAST Risk (Low): Potential IndexError if sample_ratios length is not equal to bins_fs
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.logger = get_module_logger("DEnsembleModel")
        # 🧠 ML Signal: Storing model configuration parameters
        self.logger.info("Double Ensemble Model...")
        self.ensemble = []  # the current ensemble model, a list contains all the sub-models
        # 🧠 ML Signal: Usage of dataset preparation method
        # ⚠️ SAST Risk (Low): Potential IndexError if sub_weights length is not equal to num_models
        self.sub_features = []  # the features for each sub model in the form of pandas.Index
        self.params = {"objective": loss}
        self.params.update(kwargs)
        # 🧠 ML Signal: Storing model configuration parameters
        self.loss = loss
        # ⚠️ SAST Risk (Low): Potential for ValueError if dataset is empty
        self.early_stopping_rounds = early_stopping_rounds
    # 🧠 ML Signal: Storing model configuration parameters

    def fit(self, dataset: DatasetH):
        # 🧠 ML Signal: Separation of features and labels
        # ✅ Best Practice: Use a logger for better traceability and debugging
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        # 🧠 ML Signal: Extraction of shape for further processing
        # ✅ Best Practice: Log important events for better traceability
        )
        if df_train.empty or df_valid.empty:
            # 🧠 ML Signal: Initializing ensemble-related attributes
            # ✅ Best Practice: Initialization of weights with ones
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        # 🧠 ML Signal: Initializing ensemble-related attributes
        # 🧠 ML Signal: Accessing column names for features
        # initialize the sample weights
        N, F = x_train.shape
        # 🧠 ML Signal: Storing model configuration parameters
        # ✅ Best Practice: Initialization of prediction DataFrame
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        # 🧠 ML Signal: Allowing additional parameters to be set dynamically
        features = x_train.columns
        # 🧠 ML Signal: Appending features for each sub-model
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        # 🧠 ML Signal: Storing model configuration parameters
        # train sub-models
        # 🧠 ML Signal: Logging training progress
        for k in range(self.num_models):
            # 🧠 ML Signal: Training of sub-model
            # 🧠 ML Signal: Storing model configuration parameters
            self.sub_features.append(features)
            self.logger.info("Training sub-model: ({}/{})".format(k + 1, self.num_models))
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            # 🧠 ML Signal: Appending trained model to ensemble
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            # ✅ Best Practice: Early exit from loop if condition is met
            if k + 1 == self.num_models:
                break

            # 🧠 ML Signal: Logging retrieval of loss curve
            self.logger.info("Retrieving loss curve and loss values...")
            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            # 🧠 ML Signal: Retrieval of loss curve for model evaluation
            pred_k = self.predict_sub(model_k, df_train, features)
            # ✅ Best Practice: Encapsulation of data preparation in a separate method improves readability and maintainability.
            pred_sub.iloc[:, k] = pred_k
            # 🧠 ML Signal: Prediction using sub-model
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            # 🧠 ML Signal: Storing predictions in DataFrame
            # ✅ Best Practice: Use of callbacks for logging and evaluation recording improves modularity and reusability.
            )
            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))
            # 🧠 ML Signal: Calculation of ensemble predictions

            # 🧠 ML Signal: Calculation of loss values
            # ✅ Best Practice: Conditional early stopping improves model training efficiency.
            # 🧠 ML Signal: Logging information about training process can be used for monitoring and debugging.
            if self.enable_sr:
                self.logger.info("Sample re-weighting...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                self.logger.info("Feature selection...")
                features = self.feature_selection(df_train, loss_values)

    # 🧠 ML Signal: Logging sample re-weighting process
    # 🧠 ML Signal: Conditional logic for sample re-weighting
    def train_submodel(self, df_train, df_valid, weights, features):
        # 🧠 ML Signal: Sample re-weighting based on loss curve and values
        dtrain, dvalid = self._prepare_data_gbm(df_train, df_valid, weights, features)
        evals_result = dict()
        # 🧠 ML Signal: Conditional logic for feature selection

        # ✅ Best Practice: Extracting evaluation results for both train and valid sets for further analysis.
        # 🧠 ML Signal: Extracting features and labels from DataFrame for model training
        callbacks = [lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
        # 🧠 ML Signal: Logging feature selection process
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            # 🧠 ML Signal: Feature selection based on loss values
            # ✅ Best Practice: Checking dimensionality of labels to ensure compatibility with LightGBM
            self.logger.info("Training with early_stopping...")

        # ✅ Best Practice: Using np.squeeze to handle single-dimensional entries
        model = lgb.train(
            self.params,
            dtrain,
            # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific context
            num_boost_round=self.epochs,
            valid_sets=[dtrain, dvalid],
            # 🧠 ML Signal: Creating LightGBM datasets for training and validation
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
        return model

    def _prepare_data_gbm(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]

        # 🧠 ML Signal: Normalizing loss curve to rank-based percentile
        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            # 🧠 ML Signal: Normalizing loss values to rank-based percentile
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            # ✅ Best Practice: Unpacking shape for readability
            raise ValueError("LightGBM doesn't support multi-label training")

        # ✅ Best Practice: Using np.maximum to ensure a minimum value
        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        # ✅ Best Practice: Calculating mean of the first part of the loss curve
        return dtrain, dvalid

    # ✅ Best Practice: Calculating mean of the last part of the loss curve
    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        the SR module of Double Ensemble
        :param loss_curve: the shape is NxT
        the loss curve for the previous sub-model, where the element (i, t) if the error on the i-th sample
        after the t-th iteration in the training of the previous sub-model.
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :param k_th: the index of the current sub-model, starting from 1
        :return: weights
        the weights for all the samples.
        """
        # normalize loss_curve and loss_values with ranking
        # ✅ Best Practice: Initializing weights with zeros
        # 🧠 ML Signal: Calculating average h_value per bin
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        # ⚠️ SAST Risk (Low): Potential KeyError if 'feature' or 'label' columns are missing in df_train
        loss_values_norm = (-loss_values).rank(pct=True)
        # ⚠️ SAST Risk (Low): Potential division by zero if h_avg[b] is zero

        # calculate l_start and l_end from loss_curve
        N, T = loss_curve.shape
        # ✅ Best Practice: Use descriptive variable names for readability
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # ⚠️ SAST Risk (Low): Modifying DataFrame in place can lead to unintended side effects
        # calculate h-value for each sample
        # 🧠 ML Signal: Usage of ensemble model pattern
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # calculate weights
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights
    # ⚠️ SAST Risk (Low): Division by zero risk if np.std(loss_feat - loss_values) is zero

    def feature_selection(self, df_train, loss_values):
        """
        the FS module of Double Ensemble
        :param df_train: the shape is NxF
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :return: res_feat: in the form of pandas.Index

        # ⚠️ SAST Risk (Low): np.random.choice with replace=False can raise an error if num_feat > len(b_feat)
        # ✅ Best Practice: Direct calculation of MSE for simplicity and performance
        """
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        # ⚠️ SAST Risk (Low): Generic exception message may expose internal logic
        # 🧠 ML Signal: Checking the type of base model to determine the processing logic
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)
        # 🧠 ML Signal: Using model-specific method to get the number of trees

        # shuffle specific columns and calculate g-value for each feature
        # ✅ Best Practice: Explicitly selecting columns for training features and labels
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            # ✅ Best Practice: Handling potential multi-dimensional label arrays
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    # ⚠️ SAST Risk (Low): Raising a generic exception without specific handling
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                    # 🧠 ML Signal: Using the number of training samples for further processing
                    )
                    / M
                # ✅ Best Practice: Initializing a DataFrame to store loss values
                )
            loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
            # ✅ Best Practice: Initializing prediction array for cumulative predictions
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            # ⚠️ SAST Risk (Low): No input validation for 'dataset' and 'segment', could lead to unexpected errors
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # 🧠 ML Signal: Iteratively predicting using each tree in the model
        # one column in train features is all-nan # if g['g_value'].isna().any()
        # ✅ Best Practice: Use descriptive variable names for clarity
        g["g_value"].replace(np.nan, 0, inplace=True)
        # 🧠 ML Signal: Calculating loss for each tree's predictions

        # 🧠 ML Signal: Initializing prediction series with zeros, common in ensemble methods
        # divide features into bins_fs bins
        # ⚠️ SAST Risk (Low): Raising a generic exception without specific handling
        # 🧠 ML Signal: Iterating over ensemble models, typical in ensemble learning
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # randomly sample features from bins to construct the new features
        res_feat = []
        # 🧠 ML Signal: Using submodel predictions and weights, common in weighted ensemble methods
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            # 🧠 ML Signal: Method for making predictions using a submodel
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            # ✅ Best Practice: Use of descriptive variable names for clarity
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        # 🧠 ML Signal: Normalizing predictions by sum of weights, typical in ensemble methods
        return pd.Index(set(res_feat))
    # 🧠 ML Signal: Pattern of using a model's predict method

    # ⚠️ SAST Risk (Low): Assumes submodel has a predict method, potential for AttributeError
    # ✅ Best Practice: Returning a pandas Series for consistency with input index
    def get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        else:
            raise ValueError("not implemented yet")

    def retrieve_loss_curve(self, model, df_train, features):
        if self.base_model == "gbm":
            # 🧠 ML Signal: Iterating over models to compute feature importance indicates ensemble learning
            num_trees = model.num_trees()
            # ⚠️ SAST Risk (Low): Potential risk if _model.feature_importance is not validated or sanitized
            # ✅ Best Practice: Using pd.concat and sum to aggregate results is efficient and clear
            x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
            # Lightgbm need 1D array as its label
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train = np.squeeze(y_train.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            N = x_train.shape[0]
            loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
            pred_tree = np.zeros(N, dtype=float)
            for i_tree in range(num_trees):
                pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
                loss_curve.iloc[:, i_tree] = self.get_loss(y_train, pred_tree)
        else:
            raise ValueError("not implemented yet")
        return loss_curve

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.ensemble is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )
        pred = pred / np.sum(self.sub_weights)
        return pred

    def predict_sub(self, submodel, df_data, features):
        x_data = df_data["feature"].loc[:, features]
        pred_sub = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            res.append(pd.Series(_model.feature_importance(*args, **kwargs), index=_model.feature_name()) * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)