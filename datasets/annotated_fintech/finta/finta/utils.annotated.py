import pandas as pd
# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
# 🧠 ML Signal: Importing pandas indicates data manipulation or analysis tasks


def to_dataframe(ticks: list) -> pd.DataFrame:
    # 🧠 ML Signal: Conversion of list to DataFrame is a common data preprocessing step.
    """Convert list to Series compatible with the library."""

    # 🧠 ML Signal: Converting timestamps to datetime is a common operation in time series data processing.
    df = pd.DataFrame(ticks)
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # ✅ Best Practice: Setting the index to a datetime column is a common practice for time series data.
    df.set_index("time", inplace=True)

    # 🧠 ML Signal: Usage of resample and aggregation functions on DataFrame
    return df
# 🧠 ML Signal: Function definition with specific parameters and return type

# 🧠 ML Signal: Common pattern for financial data resampling

def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample DataFrame by <interval>."""

    d = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    return df.resample(interval).agg(d)
# ✅ Best Practice: Use of a dictionary to map column names to aggregation functions

# ✅ Best Practice: Include necessary import statements for used libraries (e.g., import pandas as pd).

# 🧠 ML Signal: Use of resample and aggregation functions on a DataFrame
# ⚠️ SAST Risk (Low): Assumes 'df' has a DateTimeIndex; may raise an error if not
def resample_calendar(df: pd.DataFrame, offset: str) -> pd.DataFrame:
    """Resample the DataFrame by calendar offset.
    See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets for compatible offsets.
    :param df: data
    :param offset: calendar offset
    :return: result DataFrame
    # 🧠 ML Signal: Usage of time series analysis to determine trends.
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    """
    # ✅ Best Practice: Use f-string for better readability and performance.
    # ⚠️ SAST Risk (Low): Assumes 'df' is a pandas Series, which may not be validated.

    d = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    return df.resample(offset).agg(d)

# 🧠 ML Signal: Use of time series data and trend analysis, which are common in financial and forecasting models.
# ⚠️ SAST Risk (Low): Assumes 'df' is a pd.Series and does not check for None or empty input, which could lead to runtime errors.

def trending_up(df: pd.Series, period: int) -> pd.Series:
    """returns boolean Series if the inputs Series is trending up over last n periods.
    :param df: data
    :param period: range
    :return: result Series
    """

    return pd.Series(df.diff(period) > 0, name="trending_up {}".format(period))


def trending_down(df: pd.Series, period: int) -> pd.Series:
    """returns boolean Series if the input Series is trending up over last n periods.
    :param df: data
    :param period: range
    :return: result Series
    """

    return pd.Series(df.diff(period) < 0, name="trending_down {}".format(period))