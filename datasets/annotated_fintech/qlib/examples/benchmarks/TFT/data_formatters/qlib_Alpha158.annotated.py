# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# ✅ Best Practice: Import specific classes or functions instead of entire modules to improve readability and avoid namespace pollution.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 🧠 ML Signal: The class is designed to format data for a specific dataset, which is a common pattern in ML pipelines.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Alpha158 dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class Alpha158Formatter(GenericDataFormatter):
    """Defines and formats data for the Alpha158 dataset.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ("instrument", DataTypes.CATEGORICAL, InputTypes.ID),
        ("LABEL0", DataTypes.REAL_VALUED, InputTypes.TARGET),
        ("date", DataTypes.DATE, InputTypes.TIME),
        ("month", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ("day_of_week", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # Selected features
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        ("RESI5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("WVMA5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # 🧠 ML Signal: CATEGORICAL and STATIC_INPUT types are used for features that do not change over time.
        ("KLEN", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("ROC60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RESI10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("VSTD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("RSQR60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("WVMA60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("STD5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # ✅ Best Practice: Logging or printing messages can help in debugging and understanding the flow of execution.
        ("RSQR20", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORD60", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # 🧠 ML Signal: Using a year-based boundary to split data is a common pattern in time series analysis.
        ("CORD10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("CORR20", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # 🧠 ML Signal: Splitting data into train, validation, and test sets is a common practice in machine learning.
        ("KLOW", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("const", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    # 🧠 ML Signal: Setting scalers on the training data is a common preprocessing step in ML pipelines.
    # 🧠 ML Signal: Transforming inputs is a typical step in preparing data for machine learning models.
    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        self._real_scalers = None
        self._cat_scalers = None
        # 🧠 ML Signal: Usage of a method to get column definitions indicates a pattern for dynamic data handling
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    # 🧠 ML Signal: Dynamic retrieval of ID column based on input type

    # 🧠 ML Signal: Dynamic retrieval of target column based on input type
    def split_data(self, df, valid_boundary=2016, test_boundary=2018):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data

        Returns:
          Tuple of transformed (train, valid, test) data.
        # 🧠 ML Signal: Fitting a standard scaler to target data
        """

        print("Formatting train-valid-test splits.")

        # 🧠 ML Signal: Extracting categorical columns for encoding
        index = df["year"]
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]
        # 🧠 ML Signal: Converting categorical data to string for consistent encoding

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """
        # 🧠 ML Signal: Use of column definitions for feature transformation
        # 🧠 ML Signal: Storing number of classes per categorical input
        print("Setting scalers with training data...")

        column_definitions = self.get_column_definition()
        # 🧠 ML Signal: Extraction of real-valued columns for transformation
        id_column = utils.get_single_col_by_input_type(
            InputTypes.ID, column_definitions
        )
        target_column = utils.get_single_col_by_input_type(
            InputTypes.TARGET, column_definitions
        )

        # 🧠 ML Signal: Extraction of categorical columns for transformation
        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        # ⚠️ SAST Risk (Low): Assumes _real_scalers is properly initialized and used
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME},
            # ✅ Best Practice: Convert categorical columns to string for consistent transformation
            # ⚠️ SAST Risk (Low): Assumes _cat_scalers[col] is properly initialized and used
        )

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values
            # ✅ Best Practice: Use of .copy() to avoid modifying the original dataframe
        )  # used for predictions

        # 🧠 ML Signal: Iterating over dataframe columns to apply transformations
        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            # 🧠 ML Signal: Conditional logic to exclude certain columns from transformation
            DataTypes.CATEGORICAL,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME},
        )

        # ⚠️ SAST Risk (Low): Potential for incorrect inverse transformation if _target_scaler is not properly configured
        categorical_scalers = {}
        # ✅ Best Practice: Use of a dictionary to store related configuration parameters
        # 🧠 ML Signal: Use of time steps in model parameters
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values
            )
            num_classes.append(srs.nunique())

        # 🧠 ML Signal: Use of encoder steps in model parameters
        # Set categorical scaler outputs
        # 🧠 ML Signal: Use of early stopping in model parameters
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    # 🧠 ML Signal: Function returns a dictionary of model hyperparameters, useful for ML model configuration
    # 🧠 ML Signal: Use of multiprocessing workers in model parameters
    # ✅ Best Practice: Returning a dictionary for easy access to configuration parameters

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {"forecast_time", "identifier"}:
                # Using [col] is for aligning with the format when fitting
                output[col] = self._target_scaler.inverse_transform(predictions[[col]])

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            "total_time_steps": 6 + 6,
            "num_encoder_steps": 6,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "multiprocessing_workers": 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            "dropout_rate": 0.4,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "minibatch_size": 128,
            "max_gradient_norm": 0.0135,
            "num_heads": 1,
            "stack_size": 1,
        }

        return model_params
