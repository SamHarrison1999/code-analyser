"""
Technical Analysis Operators
"""

# ✅ Best Practice: Use relative imports for internal modules to maintain package structure

import talib
import polars as pl
import pandas as pd

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability

from .utility import DataProxy

# 🧠 ML Signal: Usage of method chaining to transform data


# ✅ Best Practice: Add type hint for the return value for better readability and maintainability
# ⚠️ SAST Risk (Low): Potential risk if 'feature.df.to_pandas()' returns unexpected data types or structures
def to_pd_series(feature: DataProxy) -> pd.Series:
    """Convert to pandas.Series data structure"""
    # ✅ Best Practice: Return the result directly for simplicity
    series: pd.Series = feature.df.to_pandas().set_index(["datetime", "vt_symbol"])[
        "data"
    ]
    # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability
    # 🧠 ML Signal: Conversion from pandas to polars could indicate performance optimization
    return series


# ⚠️ SAST Risk (Low): Ensure that the input series does not contain sensitive data before conversion


# ✅ Best Practice: Descriptive variable naming improves code readability
def to_pl_dataframe(series: pd.Series) -> pl.DataFrame:
    """Convert to polars.DataFrame data structure"""
    # 🧠 ML Signal: Usage of talib.RSI indicates a pattern for calculating the RSI indicator
    return pl.from_pandas(series.reset_index().rename(columns={0: "data"}))


# ✅ Best Practice: Type hints for function parameters and return type improve code readability and maintainability.

# ✅ Best Practice: Descriptive variable naming improves code readability


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    # ✅ Best Practice: Type hints for variables improve code readability and maintainability.
    # ✅ Best Practice: Returning a DataProxy object maintains consistency with input type
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)
    # ✅ Best Practice: Type hints for variables improve code readability and maintainability.

    result: pd.Series = talib.RSI(close_, timeperiod=window)  # type: ignore
    # ✅ Best Practice: Type hints for variables improve code readability and maintainability.
    # 🧠 ML Signal: Usage of talib.ATR indicates a pattern of using technical analysis indicators.
    # 🧠 ML Signal: Returning a DataProxy object suggests a pattern of using a specific data abstraction.

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)  # type: ignore

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)
