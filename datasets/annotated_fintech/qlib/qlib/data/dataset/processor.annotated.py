# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc

# ✅ Best Practice: Importing specific functions or classes can improve code readability and maintainability.
from typing import Union, Text, Optional
import numpy as np
import pandas as pd

# ✅ Best Practice: Grouping related imports together improves code organization.

from qlib.utils.data import robust_zscore, zscore
from ...constant import EPS
from .utils import fetch_df_by_index

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
from ...utils.serial import Serializable
from ...utils.paral import datetime_groupby_apply
from qlib.data.inst_processor import InstProcessor
from qlib.data import D


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    # ✅ Best Practice: Use of type hinting for the parameter improves code readability and maintainability.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters

        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.

        """

    # ✅ Best Practice: Method docstring provides a clear explanation of the method's purpose and return value

    # ⚠️ SAST Risk (Medium): In-place modification of input data can lead to unintended side effects
    # 🧠 ML Signal: In-place data modification pattern
    # ✅ Best Practice: Docstring provides detailed information about the method's functionality
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data

        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.

        Returns
        -------
        bool:
            if it is usable for infenrece.
        # ✅ Best Practice: Remove processed attributes from kwargs to avoid duplication
        """
        return True

    # ✅ Best Practice: Call to superclass method ensures proper inheritance behavior
    # ✅ Best Practice: Use of default parameter value for flexibility
    def readonly(self) -> bool:
        """
        Does the processor treat the input data readonly (i.e. does not write the input data) when processing

        Knowning the readonly information is helpful to the Handler to avoid uncessary copy
        # ✅ Best Practice: Consider adding type hints for the return value
        """
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
        return False

    # ✅ Best Practice: Use of default parameter values improves function usability and flexibility.

    # ✅ Best Practice: Method docstring provides a clear explanation of the method's purpose
    def config(self, **kwargs):
        # ✅ Best Practice: Explicitly passing parameters to the superclass constructor enhances code readability and maintainability.
        attr_list = {"fit_start_time", "fit_end_time"}
        # ✅ Best Practice: Type hinting for return value improves code readability and maintainability
        for k, v in kwargs.items():
            if k in attr_list and hasattr(self, k):
                # 🧠 ML Signal: Method indicates whether the object is intended for inference, useful for ML model behavior
                # ⚠️ SAST Risk (Low): Using a mutable default argument like a list can lead to unexpected behavior if the list is modified.
                setattr(self, k, v)

        # ✅ Best Practice: Use 'self' to store instance-specific data.
        for attr in attr_list:
            # ✅ Best Practice: Check if df.columns is a MultiIndex to handle different DataFrame structures
            if attr in kwargs:
                # 🧠 ML Signal: Usage of get_level_values to handle MultiIndex columns
                kwargs.pop(attr)
        super().config(**kwargs)


# 🧠 ML Signal: Usage of isin to filter columns based on a list
# ✅ Best Practice: Use of a method to encapsulate behavior, improving code readability and maintainability
class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        # ✅ Best Practice: Use of loc for DataFrame slicing ensures both label-based and boolean indexing
        # 🧠 ML Signal: Method returning a constant value, indicating a potential flag or status check
        self.fields_group = fields_group

    # ⚠️ SAST Risk (Low): Using a mutable default argument (list) can lead to unexpected behavior if modified.
    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))

    # 🧠 ML Signal: Method is designed to be used as a callable object
    def readonly(self):
        return True


# ✅ Best Practice: Use of numpy set operations for efficient column difference calculation


class DropnaLabel(DropnaProcessor):
    # ✅ Best Practice: Use of numpy union operation for efficient list merging
    def __init__(self, fields_group="label"):
        # ✅ Best Practice: Method should have a docstring explaining its purpose
        super().__init__(fields_group=fields_group)

    # ✅ Best Practice: Use of pandas isin for boolean indexing

    # ✅ Best Practice: Consider adding type hints for the return value
    def is_for_infer(self) -> bool:
        # 🧠 ML Signal: Returns a filtered DataFrame based on dynamic column selection
        """The samples are dropped according to label. So it is not usable for inference"""
        # ✅ Best Practice: Class docstring provides a brief description of the class's purpose
        # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a callable, which is a specific design pattern.
        return False


# 🧠 ML Signal: Use of column filtering based on string matching


class DropCol(Processor):
    # ⚠️ SAST Risk (Low): Potential for KeyError if 'df' is not defined in the current scope
    def __init__(self, col_list=[]):
        self.col_list = col_list

    # ⚠️ SAST Risk (Low): Potential for KeyError if 'df' is not defined in the current scope

    def __call__(self, df):
        # ⚠️ SAST Risk (Low): Potential for KeyError if 'df' is not defined in the current scope
        if isinstance(df.columns, pd.MultiIndex):
            mask = df.columns.get_level_values(-1).isin(self.col_list)
        # ✅ Best Practice: Explicitly return the modified data
        # ✅ Best Practice: Class docstring provides a brief description of the class purpose
        # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a function
        else:
            mask = df.columns.isin(self.col_list)
        # ⚠️ SAST Risk (Low): Potential for NameError if 'df' is not defined in the current scope
        return df.loc[:, ~mask]

    # 🧠 ML Signal: Iterating over DataFrame columns to process each one individually

    def readonly(self):
        # ⚠️ SAST Risk (Low): Potential division by zero if all values are infinite
        return True


# 🧠 ML Signal: Applying a custom function to a DataFrame using groupby
class FilterCol(Processor):
    def __init__(self, fields_group="feature", col_list=[]):
        # ✅ Best Practice: Sorting DataFrame by index for consistent ordering
        self.fields_group = fields_group
        self.col_list = col_list

    # ✅ Best Practice: Class docstring provides a brief description of the class purpose

    # ⚠️ SAST Risk (Low): Undefined function 'replace_inf' used, potential NameError
    # ✅ Best Practice: Use of default parameter values for optional arguments
    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        all_cols = df.columns
        # 🧠 ML Signal: Method overloading with __call__ indicates a pattern for callable objects
        diff_cols = np.setdiff1d(
            all_cols.get_level_values(-1), cols.get_level_values(-1)
        )
        # ⚠️ SAST Risk (Low): In-place modification of input data can lead to unintended side effects
        self.col_list = np.union1d(diff_cols, self.col_list)
        mask = df.columns.get_level_values(-1).isin(self.col_list)
        return df.loc[:, mask]

    # ⚠️ SAST Risk (Low): In-place modification of input data can lead to unintended side effects
    def readonly(self):
        return True


# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability


# 🧠 ML Signal: Usage of time-related variables may indicate time series data processing
class TanhProcess(Processor):
    """Use tanh to process noise data"""

    # 🧠 ML Signal: Optional parameters like fields_group can indicate flexible data processing
    # ✅ Best Practice: Use of default parameter value for df allows for flexibility in function calls.

    def __call__(self, df):
        # 🧠 ML Signal: Fetching data by a specific time range indicates time-series data processing.
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            # 🧠 ML Signal: Grouping columns based on fields suggests feature engineering or preprocessing.
            col = df.columns[~mask]
            data[col] = data[col] - 1
            # 🧠 ML Signal: Calculation of min values across columns is a common normalization step.
            data[col] = np.tanh(data[col])

            # 🧠 ML Signal: Calculation of max values across columns is a common normalization step.
            return data

        # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a function
        # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
        return tanh_denoise(df)


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.


# ⚠️ SAST Risk (Low): Potential division by zero if max_val equals min_val.
class ProcessInf(Processor):
    """Process infinity"""

    # ✅ Best Practice: Setting default min and max values for ignored columns ensures consistent data scaling.
    # 🧠 ML Signal: Usage of DataFrame and column selection indicates data preprocessing, common in ML pipelines.

    def __call__(self, df):
        # ✅ Best Practice: Storing column names for later use improves code maintainability and readability.
        # ✅ Best Practice: Class docstring provides a brief description of the class purpose
        # 🧠 ML Signal: Returning a DataFrame after transformation is a common pattern in data processing for ML.
        def replace_inf(data):
            # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
            def process_inf(df):
                for col in df.columns:
                    # 🧠 ML Signal: Tracking initialization of time-related variables could be useful for time-series analysis models.
                    # FIXME: Such behavior is very weird
                    # 🧠 ML Signal: Method for fitting a model to a DataFrame, common in ML workflows
                    df[col] = df[col].replace(
                        [np.inf, -np.inf], df[col][~np.isinf(df[col])].mean()
                    )
                # 🧠 ML Signal: Optional parameters like fields_group can indicate feature selection or grouping in ML models.
                return df

            # ✅ Best Practice: Use of slicing to filter DataFrame by time range

            data = datetime_groupby_apply(data, process_inf)
            # 🧠 ML Signal: Extracting group columns, a common preprocessing step
            data.sort_index(inplace=True)
            return data

        # 🧠 ML Signal: Calculation of mean, a common feature scaling step

        return replace_inf(df)


# 🧠 ML Signal: Calculation of standard deviation, a common feature scaling step


# 🧠 ML Signal: Identifying columns with zero standard deviation
class Fillna(Processor):
    # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a callable, which is a specific design pattern.
    """Process NaN"""
    # ✅ Best Practice: Iterating with enumerate for index and value
    # ✅ Best Practice: Define default values for function parameters to improve function usability and flexibility

    def __init__(self, fields_group=None, fill_value=0):
        # ✅ Best Practice: Handling zero standard deviation to avoid division by zero
        # 🧠 ML Signal: Normalization is a common preprocessing step in ML pipelines
        self.fields_group = fields_group
        self.fill_value = fill_value

    # 🧠 ML Signal: Use of DataFrame and .loc indicates data manipulation, common in data preprocessing for ML

    # 🧠 ML Signal: Storing column names for later use
    # ✅ Best Practice: Returning the DataFrame allows for method chaining and improves function usability
    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            # this implementation is extremely slow
            # df.fillna({col: self.fill_value for col in cols}, inplace=True)
            df[self.fields_group] = df[self.fields_group].fillna(self.fill_value)
        # ✅ Best Practice: Class docstring provides a clear explanation of the class functionality and reference.
        return df


# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # ✅ Best Practice: Default parameter value is mutable; consider using None and setting inside the function
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        # 🧠 ML Signal: Fetching a specific slice of data for training
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    # 🧠 ML Signal: Dynamic selection of columns based on a group

    def fit(self, df: pd.DataFrame = None):
        # 🧠 ML Signal: Extracting values from a DataFrame for model training
        df = fetch_df_by_index(
            df, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )
        cols = get_group_columns(df, self.fields_group)
        # 🧠 ML Signal: Calculation of median for normalization
        # 🧠 ML Signal: Method for data preprocessing, common in ML pipelines
        self.min_val = np.nanmin(df[cols].values, axis=0)
        self.max_val = np.nanmax(df[cols].values, axis=0)
        # 🧠 ML Signal: Calculation of median absolute deviation for normalization
        # 🧠 ML Signal: Subtracting mean for normalization, common in ML feature scaling
        self.ignore = self.min_val == self.max_val
        # To improve the speed, we set the value of `min_val` to `0` for the columns that do not need to be processed,
        # 🧠 ML Signal: Dividing by standard deviation for normalization, common in ML feature scaling
        # ⚠️ SAST Risk (Low): Potential risk if EPS is not defined or is zero
        # and the value of `max_val` to `1`, when using `(x - min_val) / (max_val - min_val)` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / (1 - 0)`,
        # 🧠 ML Signal: Scaling factor for robust standard deviation
        # 🧠 ML Signal: Clipping outliers, common in ML data preprocessing
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            # ⚠️ SAST Risk (Low): Potential data loss if outliers are significant
            if _con:
                self.min_val[_i] = 0
                # 🧠 ML Signal: Use of default parameter values
                # 🧠 ML Signal: Assigning processed data back to DataFrame, common in ML data transformations
                self.max_val[_i] = 1
        self.cols = cols

    # ✅ Best Practice: Use of if-elif-else for method selection

    def __call__(self, df):
        def normalize(x, min_val=self.min_val, max_val=self.max_val):
            return (x - min_val) / (max_val - min_val)

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        # ⚠️ SAST Risk (Low): Potential for unhandled exception if method is not recognized
        # 🧠 ML Signal: Checks and modifies the type of self.fields_group, indicating dynamic type handling
        return df


# ✅ Best Practice: Use of context manager to temporarily set pandas options
class ZScoreNorm(Processor):
    """ZScore Normalization"""

    # 🧠 ML Signal: Use of a helper function to retrieve columns based on a group
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # ⚠️ SAST Risk (Low): Potential for SettingWithCopyWarning if df[cols] is a view
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(
            df, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        # To improve the speed, we set the value of `std_train` to `1` for the columns that do not need to be processed,
        # and the value of `mean_train` to `0`, when using `(x - mean_train) / std_train` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / 1`,
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            # ✅ Best Practice: Use of default parameter value for flexibility
            if _con:
                self.std_train[_i] = 1
                # 🧠 ML Signal: Method is designed to be used as a callable object, indicating a pattern for flexible object usage.
                self.mean_train[_i] = 0
        self.cols = cols

    # ⚠️ SAST Risk (Low): Potential risk if get_group_columns does not handle unexpected input properly.

    # 🧠 ML Signal: Dynamic column selection based on group fields, indicating a pattern for flexible data processing.
    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            # 🧠 ML Signal: Use of groupby and rank operations, indicating a pattern for data transformation.
            return (x - mean_train) / std_train

        # ✅ Best Practice: Class docstring provides a brief description of the class functionality

        # 🧠 ML Signal: Data normalization pattern by centering around zero.
        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        # ✅ Best Practice: Method docstring provides a brief description of the method functionality
        # ✅ Best Practice: Use of default mutable arguments (None) to avoid shared state issues
        return df


# 🧠 ML Signal: Scaling data, indicating a pattern for data transformation.

# 🧠 ML Signal: Storing a parameter as an instance attribute
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.


# ⚠️ SAST Risk (Low): Directly modifying the input DataFrame, which could lead to unintended side effects.
class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(
        self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True
    ):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        # ✅ Best Practice: Docstring provides clear parameter descriptions
        df = fetch_df_by_index(
            df, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )
        self.cols = get_group_columns(df, self.fields_group)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += EPS
        self.std_train *= 1.4826

    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        # 🧠 ML Signal: Usage of a calendar function with start_time, end_time, and freq
        if self.clip_outlier:
            X = np.clip(X, -3, 3)
        # ✅ Best Practice: Use of conditional expressions for concise assignment
        df[self.cols] = X
        # ✅ Best Practice: Use of conditional expressions for concise assignment
        # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a function
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    # ⚠️ SAST Risk (Low): Potential for NoneType comparison if df.index.min() returns None

    # ⚠️ SAST Risk (Low): Potential for NoneType comparison if df.index.max() returns None
    # ✅ Best Practice: Returning df.head(0) ensures an empty DataFrame with the same structure
    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError("This type of input is not supported")

    def __call__(self, df):
        # try not modify original dataframe
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]
        # depress warning by references:
        # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#getting-and-setting-options
        with pd.option_context("mode.chained_assignment", None):
            for g in self.fields_group:
                cols = get_group_columns(df, g)
                df[cols] = (
                    df[cols]
                    .groupby("datetime", group_keys=False)
                    .apply(self.zscore_func)
                )
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.
    "Cross Sectional" is often used to describe data operations.
    The operations across different stocks are often called Cross Sectional Operation.

    For example, CSRankNorm is an operation that grouping the data by each day and rank `across` all the stocks in each day.

    Explanation about 3.46 & 0.5

    .. code-block:: python

        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # for any variable
        x_rank = pd.Series(x).rank(pct=True)  # if it is converted to rank, it will be a uniform distributed
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # Normally, we will normalize it to make it like normal distribution

        x_rank.mean()   # accounts for 0.5
        1 / x_rank.std()  # accounts for 3.46

    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        t = df[cols].groupby("datetime", group_keys=False).rank(pct=True)
        t -= 0.5
        t *= 3.46  # NOTE: towards unit std
        df[cols] = t
        return df


class CSZFillna(Processor):
    """Cross Sectional Fill Nan"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = (
            df[cols]
            .groupby("datetime", group_keys=False)
            .apply(lambda x: x.fillna(x.mean()))
        )
        return df


class HashStockFormat(Processor):
    """Process the storage of from df into hasing stock format"""

    def __call__(self, df: pd.DataFrame):
        from .storage import HashingStockStorage  # pylint: disable=C0415

        return HashingStockStorage.from_df(df)


class TimeRangeFlt(InstProcessor):
    """
    This is a filter to filter stock.
    Only keep the data that exist from start_time to end_time (the existence in the middle is not checked.)
    WARNING:  It may induce leakage!!!
    """

    def __init__(
        self,
        start_time: Optional[Union[pd.Timestamp, str]] = None,
        end_time: Optional[Union[pd.Timestamp, str]] = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        start_time : Optional[Union[pd.Timestamp, str]]
            The data must start earlier (or equal) than `start_time`
            None indicates data will not be filtered based on `start_time`
        end_time : Optional[Union[pd.Timestamp, str]]
            similar to start_time
        freq : str
            The frequency of the calendar
        """
        # Align to calendar before filtering
        cal = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
        self.start_time = None if start_time is None else cal[0]
        self.end_time = None if end_time is None else cal[-1]

    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        if (
            df.empty
            or (self.start_time is None or df.index.min() <= self.start_time)
            and (self.end_time is None or df.index.max() >= self.end_time)
        ):
            return df
        return df.head(0)
