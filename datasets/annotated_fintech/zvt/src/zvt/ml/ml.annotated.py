# -*- coding: utf-8 -*-
import logging
from typing import Union, Type, List

import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zvt.api.kdata import default_adjust_type, get_kdata
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract import TradableEntity
from zvt.contract.drawer import Drawer
from zvt.domain import Stock
from zvt.factors.transformers import MaTransformer
from zvt.ml.lables import RelativePerformance, BehaviorCategory

# 🧠 ML Signal: Function calculates percentage change, useful for time series analysis
# ✅ Best Practice: Using a logger instead of print statements for logging is a best practice.
from zvt.utils.pd_utils import (
    group_by_entity_id,
    normalize_group_compute_result,
    pd_is_not_null,
)

# ✅ Best Practice: Function name should be more descriptive, e.g., calculate_percentage_change
from zvt.utils.time_utils import to_pd_timestamp

# ✅ Best Practice: Type hint for predict_range would improve readability and maintainability
# 🧠 ML Signal: Function uses percentage change to classify behavior, useful for trend prediction models

# ⚠️ SAST Risk (Low): No input validation for predict_range, could lead to unexpected behavior
# ✅ Best Practice: Function name should be more descriptive to indicate its purpose
logger = logging.getLogger(__name__)

# ⚠️ SAST Risk (Low): No input validation for s, could lead to unexpected behavior if not a pd.Series
# 🧠 ML Signal: Uses percentage change over a specified range, indicating a pattern for time series analysis


# 🧠 ML Signal: Function for time series prediction by shifting data
def cal_change(s: pd.Series, predict_range):
    # ✅ Best Practice: Function name should be more descriptive, e.g., `calculate_prediction`
    # 🧠 ML Signal: Lambda function categorizes data, useful for classification tasks
    return s.pct_change(periods=-predict_range)


# ⚠️ SAST Risk (Low): Function lacks error handling for cases where 's' is not comparable to 'RelativePerformance' values

# ⚠️ SAST Risk (Low): No input validation for `s` and `predict_range`
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability


# ✅ Best Practice: Add type hints for function parameters and return type
def cal_behavior_cls(s: pd.Series, predict_range):
    # ⚠️ SAST Risk (Low): Potential AttributeError if 'RelativePerformance' or its attributes are not defined
    return s.pct_change(periods=-predict_range).apply(
        lambda x: BehaviorCategory.up.value if x > 0 else BehaviorCategory.down.value
    )


# ⚠️ SAST Risk (Low): Potential AttributeError if 'RelativePerformance' or its attributes are not defined

# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.


def cal_predict(s: pd.Series, predict_range):
    # ⚠️ SAST Risk (Low): Potential AttributeError if 'RelativePerformance' or its attributes are not defined
    # ✅ Best Practice: Class attribute with type annotation for better clarity and type checking.
    return s.shift(periods=-predict_range)


def cal_relative_performance(s: pd.Series):
    if s >= RelativePerformance.best.value:
        return RelativePerformance.best
    if s >= RelativePerformance.ordinary.value:
        return RelativePerformance.ordinary
    if s >= RelativePerformance.poor.value:
        return RelativePerformance.poor


class MLMachine(object):
    entity_schema: Type[TradableEntity] = None

    def __init__(
        self,
        entity_ids: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = "2015-01-01",
        end_timestamp: Union[str, pd.Timestamp] = "2021-12-01",
        predict_start_timestamp: Union[str, pd.Timestamp] = "2021-06-01",
        predict_steps: int = 20,
        level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY,
        adjust_type: Union[AdjustType, str] = None,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
        data_provider: str = None,
        label_method: str = "raw",
        # ✅ Best Practice: Convert timestamps to a consistent format for internal processing.
    ) -> None:
        """

        :param entity_ids:
        :param start_timestamp:
        :param end_timestamp:
        :param predict_start_timestamp:
        :param predict_steps:
        :param level:
        :param adjust_type:
        :param data_provider:
        :param label_method: raw, change, or behavior_cls
        """
        super().__init__()
        # 🧠 ML Signal: Building a dataset (kdata) is a common pattern in ML workflows.
        self.entity_ids = entity_ids
        self.start_timestamp = to_pd_timestamp(start_timestamp)
        self.end_timestamp = to_pd_timestamp(end_timestamp)
        # ⚠️ SAST Risk (Low): Logging error messages can expose sensitive information.
        self.predict_start_timestamp = to_pd_timestamp(predict_start_timestamp)
        assert self.start_timestamp < self.predict_start_timestamp < self.end_timestamp
        # ⚠️ SAST Risk (Low): Assertion can be disabled in production, consider using exception handling.
        self.predict_steps = predict_steps

        # 🧠 ML Signal: Feature engineering step, common in ML pipelines.
        self.level = level
        if not adjust_type:
            # ✅ Best Practice: Handle missing data by dropping or imputing.
            adjust_type = default_adjust_type(entity_type=self.entity_schema.__name__)
        self.adjust_type = adjust_type
        # ✅ Best Practice: Use set operations to efficiently determine feature names.

        # 🧠 ML Signal: Splitting data into training and testing sets is a common pattern in ML workflows
        self.data_provider = data_provider
        self.label_method = label_method
        # 🧠 ML Signal: Using timestamps to split data is a common practice in time series analysis
        # 🧠 ML Signal: Label construction is a key step in supervised learning.

        self.kdata_df = self.build_kdata()
        # 🧠 ML Signal: Aligning features and labels for training data
        if not pd_is_not_null(self.kdata_df):
            logger.error("not kdta")
            # 🧠 ML Signal: Using timestamps to split data is a common practice in time series analysis
            # 🧠 ML Signal: Data splitting is a common step in preparing data for ML models.
            # ✅ Best Practice: Define a function to encapsulate logic for building kdata, improving code organization and reusability.
            assert False

        # 🧠 ML Signal: Aligning features and labels for testing data
        # ⚠️ SAST Risk (Low): Logging data can expose sensitive information.
        # ✅ Best Practice: Initialize model-related variables to None before use.
        # ✅ Best Practice: Returning multiple values in a consistent order improves code readability
        # ✅ Best Practice: Use a list to define column names, making it easy to modify or extend.
        # 🧠 ML Signal: Usage of a function to retrieve data based on parameters, indicating a pattern for data retrieval.
        # 🧠 ML Signal: Use of self.entity_ids suggests an object-oriented approach, indicating a pattern for instance variable usage.
        self.feature_df = self.build_feature(
            self.entity_ids, self.start_timestamp, self.end_timestamp
        )
        # drop na in feature
        self.feature_df = self.feature_df.dropna()
        self.feature_names = list(
            set(self.feature_df.columns) - {"entity_id", "timestamp"}
        )
        self.feature_df = self.feature_df.loc[:, self.feature_names]

        # 🧠 ML Signal: Use of self.start_timestamp and self.end_timestamp indicates a pattern for time-based data retrieval.
        self.label_ser = self.build_label()
        # keep same index with feature df
        self.label_ser = self.label_ser.loc[self.feature_df.index]
        # 🧠 ML Signal: Passing a list of columns to a function, indicating a pattern for data selection.
        self.label_name = self.label_ser.name

        # 🧠 ML Signal: Use of self.level suggests a pattern for hierarchical or leveled data retrieval.
        self.training_X, self.training_y, self.testing_X, self.testing_y = (
            self.split_data()
        )
        # 🧠 ML Signal: Dynamic label creation based on prediction steps and method

        # 🧠 ML Signal: Use of self.adjust_type indicates a pattern for data adjustment or transformation.
        logger.info(self.training_X)
        # 🧠 ML Signal: Use of self.data_provider suggests a pattern for specifying data sources.
        # 🧠 ML Signal: Use of raw data for label creation
        logger.info(self.training_y)

        self.model = None
        self.pred_y = None

    # 🧠 ML Signal: Specifying index columns indicates a pattern for data indexing.
    # 🧠 ML Signal: Use of drop_index_col=True suggests a pattern for data preprocessing or cleaning.
    # 🧠 ML Signal: Grouping data by entity for feature calculation
    # 🧠 ML Signal: Applying a prediction calculation function
    def split_data(self):
        # 🧠 ML Signal: Renaming series for label identification
        training_x = self.feature_df[
            self.feature_df.index.get_level_values("timestamp")
            < self.predict_start_timestamp
        ]
        training_y = self.label_ser[
            self.label_ser.index.get_level_values("timestamp")
            < self.predict_start_timestamp
        ]

        testing_x = self.feature_df[
            self.feature_df.index.get_level_values("timestamp")
            >= self.predict_start_timestamp
        ]
        testing_y = self.label_ser[
            self.label_ser.index.get_level_values("timestamp")
            >= self.predict_start_timestamp
        ]
        # 🧠 ML Signal: Use of change data for label creation
        return training_x, training_y, testing_x, testing_y

    # 🧠 ML Signal: Grouping data by entity for feature calculation
    # 🧠 ML Signal: Applying a change calculation function

    def build_kdata(self):
        columns = ["entity_id", "timestamp", "close"]
        return get_kdata(
            entity_ids=self.entity_ids,
            start_timestamp=self.start_timestamp,
            # 🧠 ML Signal: Renaming series for label identification
            end_timestamp=self.end_timestamp,
            columns=columns,
            level=self.level,
            # 🧠 ML Signal: Method for training a machine learning model
            # 🧠 ML Signal: Use of behavior classification for label creation
            adjust_type=self.adjust_type,
            # ✅ Best Practice: Allowing model and parameters to be passed in increases flexibility
            provider=self.data_provider,
            # 🧠 ML Signal: Grouping data by entity for feature calculation
            index=["entity_id", "timestamp"],
            # ⚠️ SAST Risk (Low): Potential for misuse if `params` contains unsafe data
            drop_index_col=True,
            # 🧠 ML Signal: Use of conditional logic to determine different processing paths
            # 🧠 ML Signal: Applying a behavior classification function
        )

    # 🧠 ML Signal: Returns the trained model

    # 🧠 ML Signal: Renaming series for label identification
    # 🧠 ML Signal: DataFrame operations to manipulate and prepare data
    def build_label(self):
        label_name = f"y_{self.predict_steps}"
        # ⚠️ SAST Risk (Low): Use of assert for control flow, which can be disabled
        # 🧠 ML Signal: Conversion of prediction results to DataFrame
        # 🧠 ML Signal: Shifting prediction results for alignment
        if self.label_method == "raw":
            y = (
                group_by_entity_id(self.kdata_df["close"]).apply(
                    lambda x: cal_predict(x, self.predict_steps)
                )
                # 🧠 ML Signal: Normalizing computed results for consistency
                # 🧠 ML Signal: Use of a custom Drawer class for visualization
                .rename(label_name)
            )
        elif self.label_method == "change":
            y = (
                group_by_entity_id(self.kdata_df["close"])
                # 🧠 ML Signal: Visualization method call
                .apply(lambda x: cal_change(x, self.predict_steps)).rename(label_name)
                # 🧠 ML Signal: Method for making predictions using a model
            )
        # 🧠 ML Signal: Conversion of prediction results to DataFrame
        elif self.label_method == "behavior_cls":
            # 🧠 ML Signal: Joining real and predicted results for comparison
            # 🧠 ML Signal: Storing predictions in a pandas Series
            # ✅ Best Practice: Use of pandas Series for aligning predictions with indices
            y = (
                group_by_entity_id(self.kdata_df["close"]).apply(
                    lambda x: cal_behavior_cls(x, self.predict_steps)
                )
                # 🧠 ML Signal: Use of a custom Drawer class for visualization
                # ✅ Best Practice: Docstring provides a clear description of the method's purpose and expected input/output.
                # 🧠 ML Signal: Visualization method call
                .rename(label_name)
            )
        else:
            assert False
        y = normalize_group_compute_result(y)

        return y

    def train(self, model=LinearRegression(), **params):
        self.model = model.fit(self.training_X, self.training_y, **params)
        return self.model

    # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called.
    # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
    def draw_result(self, entity_id):
        if self.label_method == "raw":
            # ✅ Best Practice: Class definition should follow the naming convention of using CamelCase.
            # 🧠 ML Signal: Usage of entity_schema suggests a pattern for defining schemas in ML models.
            df = self.kdata_df.loc[[entity_id], ["close"]].copy()

            pred_df = self.pred_y.to_frame(name="pred_close")
            pred_df = pred_df.loc[[entity_id], :].shift(self.predict_steps)
            # ✅ Best Practice: Add detailed docstring for parameters and return value

            drawer = Drawer(
                main_df=df,
                factor_df_list=[pred_df],
            )
            drawer.draw_line(show=True)
        else:
            pred_df = self.pred_y.to_frame(name="pred_result").loc[[entity_id], :]
            # 🧠 ML Signal: Use of moving average windows for feature transformation
            df = (
                self.testing_y.to_frame(name="real_result")
                .loc[[entity_id], :]
                .join(pred_df, how="outer")
            )

            drawer = Drawer(main_df=df)
            drawer.draw_table()

    # 🧠 ML Signal: Instantiation and usage of a machine learning pipeline

    def predict(self):
        # 🧠 ML Signal: Use of a machine learning pipeline with data preprocessing and model training
        predictions = self.model.predict(self.testing_X)
        # 🧠 ML Signal: Training a machine learning model
        # 🧠 ML Signal: Making predictions with a trained machine learning model
        # 🧠 ML Signal: Visualization of machine learning results
        # ✅ Best Practice: Use of __all__ to define public API of the module
        self.pred_y = pd.Series(data=predictions, index=self.testing_y.index)
        # explained_variance_score(self.testing_y, self.pred_y)
        # mean_squared_error(self.testing_y, self.pred_y)

    def build_feature(
        self,
        entity_ids: List[str],
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        result df format
                                  col1    col2    col3    ...
        entity_id    timestamp
                                  1.2     0.5     0.3     ...
                                  1.0     0.7     0.2     ...

        :param entity_ids: entity id list
        :param start_timestamp:
        :param end_timestamp:
        :rtype: pd.DataFrame
        """
        raise NotImplementedError


class StockMLMachine(MLMachine):
    entity_schema = Stock


class MaStockMLMachine(StockMLMachine):
    def build_feature(
        self,
        entity_ids: List[str],
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """

        :param entity_ids:
        :param start_timestamp:
        :param end_timestamp:
        :return:
        """
        t = MaTransformer(windows=[5, 10, 120, 250])
        df = t.transform(self.kdata_df)
        return df


if __name__ == "__main__":
    machine = MaStockMLMachine(entity_ids=["stock_sz_000001"])
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    machine.train(model=reg)
    machine.predict()
    machine.draw_result(entity_id="stock_sz_000001")

# the __all__ is generated
__all__ = [
    "cal_change",
    "cal_behavior_cls",
    "cal_predict",
    "cal_relative_performance",
    "MLMachine",
    "StockMLMachine",
    "MaStockMLMachine",
]
