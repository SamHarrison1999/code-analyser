"""
Time Series Operators
"""

from typing import cast
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.

from scipy import stats     # type: ignore
import polars as pl
# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
import numpy as np

# 🧠 ML Signal: Use of time series data manipulation, which is common in ML feature engineering
# ✅ Best Practice: Use of descriptive variable names for clarity
from .utility import DataProxy


def ts_delay(feature: DataProxy, window: int) -> DataProxy:
    """Get the value from a fixed time in the past"""
    # ✅ Best Practice: Explicitly selecting columns for clarity and to avoid unintentional data leakage
    df: pl.DataFrame = feature.df.select(
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        pl.col("datetime"),
        # 🧠 ML Signal: Use of window functions, which are often used in time series analysis
        pl.col("vt_symbol"),
        # 🧠 ML Signal: Use of rolling window operations, common in time series analysis
        # ✅ Best Practice: Use of descriptive variable names for clarity
        # ⚠️ SAST Risk (Low): Ensure that the 'window' parameter is validated to prevent unexpected behavior
        pl.col("data").shift(window).over("vt_symbol")
    )
    return DataProxy(df)


# ✅ Best Practice: Returning a DataProxy object, maintaining consistency with input type
# ✅ Best Practice: Explicitly selecting columns improves readability and prevents unintended data manipulation
def ts_min(feature: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
    """Calculate the minimum value over a rolling window"""
    # 🧠 ML Signal: Use of rolling_min function indicates time series data processing
    df: pl.DataFrame = feature.df.select(
        # 🧠 ML Signal: Use of rolling window operations, common in time series analysis
        # ⚠️ SAST Risk (Low): Ensure that the 'window' parameter is validated to prevent misuse or errors
        # ✅ Best Practice: Use of descriptive variable names for clarity
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_min(window, min_samples=1).over("vt_symbol")
    )
    return DataProxy(df)
# ✅ Best Practice: Returning a DataProxy object maintains encapsulation and abstraction
# ✅ Best Practice: Explicitly selecting columns for clarity and to avoid unintended data manipulation

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.

# 🧠 ML Signal: Use of rolling_max function, indicating a focus on maximum value calculations over time
def ts_max(feature: DataProxy, window: int) -> DataProxy:
    # ⚠️ SAST Risk (Low): Ensure that the 'window' parameter is validated to prevent potential misuse or errors
    # ✅ Best Practice: Use explicit type annotations for variables to improve code clarity.
    """Calculate the maximum value over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_max(window, min_samples=1).over("vt_symbol")
    # ✅ Best Practice: Returning a DataProxy object, maintaining consistency with input type
    # ⚠️ SAST Risk (Low): Ensure that the lambda function used in rolling_map does not introduce any side effects or security issues.
    )
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    # 🧠 ML Signal: Usage of rolling_map with a lambda function indicates a pattern for applying operations over a window, useful for time-series analysis.
    return DataProxy(df)

# 🧠 ML Signal: Use of rolling window operations, which are common in time series analysis.
# ✅ Best Practice: Use of type hint for variable 'df' improves code readability.
# ✅ Best Practice: Return a new instance of DataProxy to encapsulate the DataFrame, promoting immutability and separation of concerns.

def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the maximum value over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        # ✅ Best Practice: Explicitly selecting columns improves code readability and maintainability.
        pl.col("vt_symbol"),
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        pl.col("data").rolling_map(lambda s: cast(int, s.arg_max()) + 1, window).over("vt_symbol")
    # ⚠️ SAST Risk (Low): Use of lambda functions can sometimes lead to less readable code if overused or complex.
    )
    # ✅ Best Practice: Returning a DataProxy object maintains consistency with the input type.
    # 🧠 ML Signal: Use of rolling window operations, common in time series analysis
    # ✅ Best Practice: Use of descriptive variable names like 'df' for DataFrame
    return DataProxy(df)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the minimum value over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        # ✅ Best Practice: Include type hints for function parameters for better readability and maintainability
        # ⚠️ SAST Risk (Low): Use of lambda function within rolling_map, ensure input is sanitized
        pl.col("datetime"),
        # 🧠 ML Signal: Use of statistical function 'percentileofscore', indicating statistical analysis
        pl.col("vt_symbol"),
        # ✅ Best Practice: Use descriptive variable names for clarity
        # ✅ Best Practice: Return a new instance of DataProxy, ensuring immutability of input data
        pl.col("data").rolling_map(lambda s: cast(int, s.arg_min()) + 1, window).over("vt_symbol")
    )
    return DataProxy(df)


# ✅ Best Practice: Explicitly specify columns for selection to avoid unintentional data exposure
def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Ensure that the 'data' column exists and is of numeric type to prevent runtime errors
    """Calculate the percentile rank of the current value within the window"""
    df: pl.DataFrame = feature.df.select(
        # 🧠 ML Signal: Returning a DataProxy object indicates a pattern of wrapping data operations
        # 🧠 ML Signal: Use of rolling window mean calculation, common in time series analysis.
        # ✅ Best Practice: Use of method chaining with select and rolling_map for concise data manipulation.
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: stats.percentileofscore(s, s[-1]) / 100, window).over("vt_symbol")
    )
    return DataProxy(df)

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
# ⚠️ SAST Risk (Low): Casting data to Float32 could lead to precision loss if not intended.

def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Returning a DataProxy object maintains encapsulation and abstraction.
    # ✅ Best Practice: Use explicit type annotation for variables to improve code readability
    """Calculate the sum over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_sum(window).over("vt_symbol")
    # ⚠️ SAST Risk (Low): Using lambda functions can sometimes lead to security risks if not properly handled
    )
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
    # 🧠 ML Signal: Use of rolling_map with a lambda function indicates a pattern of applying custom operations over a window
    return DataProxy(df)

# ✅ Best Practice: Use descriptive variable names for clarity
# 🧠 ML Signal: Returning a DataProxy object suggests a pattern of wrapping dataframes for additional functionality

def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the mean over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        # ⚠️ SAST Risk (Low): Using lambda functions can sometimes lead to less readable code and debugging difficulties
        pl.col("vt_symbol"),
        # 🧠 ML Signal: Use of rolling window operations, common in time series analysis
        # 🧠 ML Signal: Function definition with specific parameters can indicate usage patterns for ML models
        pl.col("data").cast(pl.Float32).rolling_map(lambda s: np.nanmean(s), window, min_samples=1).over("vt_symbol")
    )
    # ✅ Best Practice: Return a well-defined object, ensuring the function's purpose is clear
    # ✅ Best Practice: Type hinting for variables improves code readability and maintainability
    return DataProxy(df)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the standard deviation over a rolling window"""
    # ✅ Best Practice: Explicitly selecting columns improves code readability and maintainability
    df: pl.DataFrame = feature.df.select(
        # 🧠 ML Signal: Use of rolling_map with quantile calculation can indicate time-series analysis patterns
        pl.col("datetime"),
        # ⚠️ SAST Risk (Low): Ensure that the lambda function does not introduce side effects or security issues
        pl.col("vt_symbol"),
        # ✅ Best Practice: Docstring provides a clear description of the function's purpose
        pl.col("data").rolling_map(lambda s: np.nanstd(s, ddof=0), window, min_samples=1).over("vt_symbol")
    )
    # ✅ Best Practice: Check if standard deviation is zero to avoid division by zero errors
    # 🧠 ML Signal: Returning a DataProxy object can indicate a pattern of data encapsulation
    return DataProxy(df)
# 🧠 ML Signal: Use of linear regression to calculate R-squared, common in predictive modeling


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Type hint for DataFrame improves code readability and maintainability
    """Calculate the slope of linear regression over a rolling window"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        # ✅ Best Practice: Explicitly selecting columns improves code readability
        pl.col("data").rolling_map(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], window).over("vt_symbol")
    )
    return DataProxy(df)
# ✅ Best Practice: Docstring provides a clear description of the function's purpose
# 🧠 ML Signal: Use of rolling_map for time series analysis, common in financial data processing


# ⚠️ SAST Risk (Low): Ensure DataProxy is a safe wrapper and does not introduce security risks
# ✅ Best Practice: Use of type annotations for variables improves code readability and maintainability.
def ts_quantile(feature: DataProxy, window: int, quantile: float) -> DataProxy:
    """Calculate the quantile value over a rolling window"""
    # ✅ Best Practice: Use of type annotations for variables improves code readability and maintainability.
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        # ✅ Best Practice: Use of type annotations for variables improves code readability and maintainability.
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: s.quantile(quantile=quantile, interpolation="linear"), window).over("vt_symbol")
    # ✅ Best Practice: Use of type annotations for variables improves code readability and maintainability.
    )
    return DataProxy(df)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    # 🧠 ML Signal: Use of rolling_map function indicates a pattern of applying a function over a rolling window, common in time series analysis.
    """Calculate the R-squared value of linear regression over a rolling window"""
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    def rsquare(s: pl.Series) -> float:
        """Calculate R-squared value for a series"""
        if s.std():
            # 🧠 ML Signal: Use of lambda functions for inline operations is a common pattern in data processing.
            # ⚠️ SAST Risk (Low): Use of lambda can sometimes lead to less readable code if overused or complex.
            # 🧠 ML Signal: Usage of DataFrame join operation, common in data preprocessing for ML tasks.
            # 🧠 ML Signal: Usage of rolling correlation, a common feature engineering technique in time series analysis.
            return float(stats.linregress(np.arange(len(s)), s).rvalue ** 2)
        else:
            return float("nan")

    df: pl.DataFrame = feature.df.select(
        # 🧠 ML Signal: Returning a DataProxy object suggests a pattern of wrapping or abstracting data operations.
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: rsquare(s), window).over("vt_symbol"))
    # ⚠️ SAST Risk (Low): Handling of infinite values, which could lead to incorrect data processing if not managed.
    return DataProxy(df)
# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.


def ts_resi(feature: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Use isinstance to check the type of feature2 for better code clarity and error handling.
    # ✅ Best Practice: Returning a DataProxy object, maintaining encapsulation and abstraction.
    """Calculate the residual of linear regression over a rolling window"""
    def resi(s: pl.Series) -> float:
        # ⚠️ SAST Risk (Low): Ensure that the join operation does not expose sensitive data by verifying the columns being joined.
        """Calculate residual for a series"""
        # ✅ Best Practice: Use with_columns to add a new column to the DataFrame, which is clear and concise.
        x: np.ndarray = np.arange(len(s))
        y: np.ndarray = s.to_numpy()
        coefficients: np.ndarray = np.polyfit(x, y, 1)
        predictions: np.ndarray = coefficients[0] * x + coefficients[1]
        resi: np.ndarray = y - predictions
        return float(resi[-1])
    # ✅ Best Practice: Add type hints for function parameters and return type for better readability and maintainability

    # ✅ Best Practice: Use min_horizontal for clear and efficient computation of the minimum value across specified columns.
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        # 🧠 ML Signal: Returning a DataProxy object could indicate a pattern of data transformation or feature engineering.
        # 🧠 ML Signal: Use of isinstance to check type, indicating dynamic type handling
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: resi(s), window).over("vt_symbol")
    # ⚠️ SAST Risk (Low): Potential for key errors if "datetime" or "vt_symbol" columns are missing
    )
    # ✅ Best Practice: Use of with_columns to add a new column, improving code readability
    return DataProxy(df)


def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """Calculate the correlation between two features over a rolling window"""
    # ✅ Best Practice: Use of select to specify columns, enhancing code clarity
    df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability

    df: pl.DataFrame = df_merged.select(
        # ⚠️ SAST Risk (Low): Assumes "data" and "data_right" columns exist, potential for runtime errors
        # 🧠 ML Signal: Use of natural logarithm transformation on data, common in feature engineering
        # ✅ Best Practice: Use of method chaining for concise and readable data manipulation
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.rolling_corr("data", "data_right", window_size=window, min_samples=1).over("vt_symbol").alias("data")
    )

    # 🧠 ML Signal: Returning a DataProxy object, indicating a pattern of wrapping dataframes
    # ✅ Best Practice: Explicitly selecting columns improves code readability and prevents unintended data manipulation
    df = df.with_columns(
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        pl.when(pl.col("data").is_infinite()).then(None).otherwise(pl.col("data")).alias("data")
    # ⚠️ SAST Risk (Low): Ensure that the "data" column does not contain non-positive values to avoid math domain errors
    )
    # ✅ Best Practice: Use descriptive variable names for better readability
    # 🧠 ML Signal: Returning a DataProxy object, indicating a pattern of wrapping dataframes for additional functionality

    return DataProxy(df)


def ts_less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    # ✅ Best Practice: Explicitly specify columns to select for clarity and to avoid unintended data exposure
    # ⚠️ SAST Risk (Low): Ensure that the "data" column exists and contains numeric values to avoid runtime errors
    # 🧠 ML Signal: Returns a DataProxy object, indicating a pattern of data transformation
    """Return the minimum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.min_horizontal("data", "data_right").over("vt_symbol").alias("data")
    )

    return DataProxy(df)


def ts_greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """Return the maximum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])

    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.max_horizontal("data", "data_right").over("vt_symbol").alias("data")
    )

    return DataProxy(df)


def ts_log(feature: DataProxy) -> DataProxy:
    """Calculate the natural logarithm of the feature"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").log().over("vt_symbol")
    )
    return DataProxy(df)


def ts_abs(feature: DataProxy) -> DataProxy:
    """Calculate the absolute value of the feature"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").abs().over("vt_symbol")
    )
    return DataProxy(df)