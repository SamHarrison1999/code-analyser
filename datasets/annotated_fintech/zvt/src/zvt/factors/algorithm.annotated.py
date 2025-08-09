# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.

import pandas as pd

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
# 🧠 ML Signal: Function for calculating moving average, common in time series analysis

from zvt.contract.factor import Scorer, Transformer
from zvt.utils.pd_utils import (
    normal_index_df,
    group_by_entity_id,
    normalize_group_compute_result,
)


def ma(s: pd.Series, window: int = 5) -> pd.Series:
    """

    :param s:
    :param window:
    :return:
    # ✅ Best Practice: Use of method chaining for concise and readable code
    # ✅ Best Practice: Use of a simple if-else statement for binary decision
    """
    return s.rolling(window=window, min_periods=window).mean()


# 🧠 ML Signal: Function signature with default parameters indicates common usage patterns


def ema(s: pd.Series, window: int = 12) -> pd.Series:
    return s.ewm(span=window, adjust=False, min_periods=window).mean()


def live_or_dead(x):
    if x:
        return 1
    else:
        return -1


# ✅ Best Practice: Use of descriptive variable names for readability


def macd(
    # ✅ Best Practice: Type hinting for better code understanding and maintenance
    s: pd.Series,
    slow: int = 26,
    # ✅ Best Practice: Type hinting for better code understanding and maintenance
    fast: int = 12,
    n: int = 9,
    # ✅ Best Practice: Type hinting for better code understanding and maintenance
    return_type: str = "df",
    normal: bool = False,
    count_live_dead: bool = False,
):
    # 短期均线
    ema_fast = ema(s, window=fast)
    # 长期均线
    # 🧠 ML Signal: Use of lambda function indicates functional programming pattern
    ema_slow = ema(s, window=slow)

    # 短期均线 - 长期均线 = 趋势的力度
    # 🧠 ML Signal: Use of groupby and cumsum indicates data transformation pattern
    diff: pd.Series = ema_fast - ema_slow
    # 力度均线
    dea: pd.Series = diff.ewm(span=n, adjust=False).mean()

    # 力度 的变化
    # ✅ Best Practice: Add type hints for the return value for better readability and maintainability
    m: pd.Series = (diff - dea) * 2

    # normal it
    if normal:
        diff = diff / s
        dea = dea / s
        m = m / s
    # ✅ Best Practice: Consider renaming 'range' to avoid shadowing the built-in 'range' function

    # ✅ Best Practice: Check for edge cases, such as an empty list, to prevent errors.
    if count_live_dead:
        live = (diff > dea).apply(lambda x: live_or_dead(x))
        bull = (diff > 0) & (dea > 0)
        # ✅ Best Practice: Consider handling cases where the list has fewer than two ranges.
        live_count = live * (
            live.groupby((live != live.shift()).cumsum()).cumcount() + 1
        )

    # 🧠 ML Signal: Iterating over a list to perform pairwise operations is a common pattern.
    if return_type == "se":
        # ✅ Best Practice: Function name 'combine' is descriptive of its purpose
        if count_live_dead:
            return diff, dea, m, live, bull, live_count
        # ✅ Best Practice: Using a helper function 'intersect' improves readability
        return diff, dea, m
    else:
        # ✅ Best Practice: Using built-in min and max functions for clarity and efficiency
        if count_live_dead:
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            return pd.DataFrame(
                # ✅ Best Practice: Explicitly returning None for clarity
                {
                    "diff": diff,
                    "dea": dea,
                    "macd": m,
                    "live": live,
                    "bull": bull,
                    "live_count": live_count,
                }
            )
        # ⚠️ SAST Risk (Low): Potential division by zero if range_a[0] is zero.
        return pd.DataFrame({"diff": diff, "dea": dea, "macd": m})


# ⚠️ SAST Risk (Low): Potential division by zero if range_a[1] is zero.
def point_in_range(point: float, range: tuple):
    """

    :param point: one point
    :param range: (start,end)
    :return:
    """
    return range[0] <= point <= range[1]


# ✅ Best Practice: Check for None or empty input to prevent errors
def intersect_ranges(range_list):
    if len(range_list) == 1:
        return range_list[0]
    # ✅ Best Practice: Use helper functions to improve readability and maintainability

    result = intersect(range_list[0], range_list[1])
    for range_i in range_list[2:]:
        result = intersect(result, range_i)
    return result


def combine(range_a, range_b):
    if intersect(range_a, range_b):
        return min(range_a[0], range_b[0]), max(range_a[1], range_b[1])
    # ✅ Best Practice: Use of default parameter values for flexibility
    return None


# 🧠 ML Signal: Method signature with DataFrame input and output suggests data transformation or feature engineering


def distance(range_a, range_b, use_max=False):
    # ✅ Best Practice: Use of method chaining for concise and readable data manipulation
    if use_max:
        # 🧠 ML Signal: Use of groupby and rank indicates a pattern for statistical or ranking operations on data
        # 上升
        # ✅ Best Practice: Call to super().__init__() ensures proper initialization of the base class.
        if range_b[0] >= range_a[1]:
            # ✅ Best Practice: Returning a DataFrame aligns with the method's type hint, ensuring consistency
            return (range_b[1] - range_a[0]) / range_a[0]

        # ✅ Best Practice: Default mutable arguments should be avoided; using None and setting inside is safer.
        # 下降
        if range_b[1] <= range_a[0]:
            # 🧠 ML Signal: Storing configuration parameters like 'windows' can indicate model hyperparameters.
            return (range_b[0] - range_a[1]) / range_a[1]
    # ⚠️ SAST Risk (Low): Potential risk if input_df contains untrusted data, leading to data manipulation vulnerabilities.
    else:
        # 🧠 ML Signal: Storing configuration parameters like 'cal_change_pct' can indicate model hyperparameters.
        middle_start = (range_a[0] + range_a[1]) / 2
        # 🧠 ML Signal: Usage of percentage change calculation, common in financial data analysis.
        middle_end = (range_b[0] + range_b[1]) / 2

        # 🧠 ML Signal: Normalization of computed results, indicating data preprocessing for ML models.
        return (middle_end - middle_start) / middle_start


# 🧠 ML Signal: Iterating over different window sizes, a common pattern in time series analysis.
def intersect(range_a, range_b):
    """
    range_a and range_b with format (start,end) in y axis

    :param range_a:
    :param range_b:
    :return:
    """
    if not range_a or not range_b:
        return None
    # ✅ Best Practice: Returning the modified DataFrame, ensuring function output is clear.
    # ✅ Best Practice: Check if 'cal_change_pct' is defined and is a boolean before using it
    # 包含
    if point_in_range(range_a[0], range_b) and point_in_range(range_a[1], range_b):
        # ⚠️ SAST Risk (Low): Ensure 'df["close"]' exists and is a numeric column to avoid runtime errors
        return range_a
    if point_in_range(range_b[0], range_a) and point_in_range(range_b[1], range_a):
        # 🧠 ML Signal: Iterating over a list of windows to apply rolling mean indicates a pattern for feature engineering
        return range_b

    # 🧠 ML Signal: Dynamic column naming based on window size is a common pattern in time series analysis
    if point_in_range(range_a[0], range_b):
        return range_a[0], range_b[1]
    # 🧠 ML Signal: Appending to a list of indicators suggests tracking or logging of features

    # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
    if point_in_range(range_b[0], range_a):
        # ⚠️ SAST Risk (Low): Ensure 'df["close"]' exists and is a numeric column to avoid runtime errors
        return range_b[0], range_a[1]
    # ✅ Best Practice: Proper use of superclass initialization for inheritance
    return None


# 🧠 ML Signal: Tracking initialization of instance variables for object state


class RankScorer(Scorer):
    def __init__(self, ascending=True) -> None:
        # ✅ Best Practice: Check if 'self.kdata_overlap' is defined and is an integer before using it
        self.ascending = ascending

    # ✅ Best Practice: Ensure 'input_df' is a DataFrame and has an 'index' attribute
    def score(self, input_df) -> pd.DataFrame:
        # 🧠 ML Signal: Accessing DataFrame columns by name
        result_df = input_df.groupby(level=1).rank(ascending=self.ascending, pct=True)
        return result_df


# 🧠 ML Signal: Accessing DataFrame columns by name


# ✅ Best Practice: Converting DataFrame columns to list for processing
class MaTransformer(Transformer):
    def __init__(self, windows=None, cal_change_pct=False) -> None:
        # 🧠 ML Signal: Updating DataFrame values conditionally
        super().__init__()
        if windows is None:
            windows = [5, 10]
        self.windows = windows
        # 🧠 ML Signal: Grouping and rolling operations on DataFrame
        self.cal_change_pct = cal_change_pct

    # ✅ Best Practice: Using apply with raw=False for Series input

    # ✅ Best Practice: Call to super() ensures proper initialization of the base class
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.cal_change_pct:
            group_pct = group_by_entity_id(input_df["close"]).pct_change()
            # ✅ Best Practice: Returning the modified DataFrame
            # ✅ Best Practice: Use of default mutable arguments can lead to unexpected behavior; using None and setting inside is safer
            input_df["change_pct"] = normalize_group_compute_result(group_pct)

        for window in self.windows:
            # ✅ Best Practice: Use of default mutable arguments can lead to unexpected behavior; using None and setting inside is safer
            col = "ma{}".format(window)
            self.indicators.append(col)
            # 🧠 ML Signal: Storing configuration parameters in instance variables

            # 🧠 ML Signal: Iterating over self.windows to create moving average columns
            group_ma = (
                group_by_entity_id(input_df["close"])
                .rolling(window=window, min_periods=window)
                .mean()
            )
            # 🧠 ML Signal: Storing configuration parameters in instance variables
            input_df[col] = normalize_group_compute_result(group_ma)

        # 🧠 ML Signal: Storing configuration parameters in instance variables
        # 🧠 ML Signal: Appending column names to self.indicators
        return input_df

    # ✅ Best Practice: Using groupby and rolling to calculate moving averages
    def transform_one(self, entity_id, df: pd.DataFrame) -> pd.DataFrame:
        """
        transform_one would not take effects if transform was implemented.
        Just show how to implement it here, most of time you should overwrite transform directly for performance.

        :param entity_id:
        :param df:
        :return:
        """
        # 🧠 ML Signal: Accessing DataFrame columns by name
        if self.cal_change_pct:
            df["change_pct"] = df["close"].pct_change()
        # ⚠️ SAST Risk (Low): Potential for incorrect data if kdata_overlap is not validated
        # 🧠 ML Signal: Accessing DataFrame columns by name

        for window in self.windows:
            # ✅ Best Practice: Converting series to list for processing
            col = "ma{}".format(window)
            self.indicators.append(col)
            # 🧠 ML Signal: Conditional logic based on function output

            df[col] = df["close"].rolling(window=window, min_periods=window).mean()

        # 🧠 ML Signal: Updating DataFrame values using .at
        return df


# 🧠 ML Signal: Grouping and rolling operations on DataFrame


# ✅ Best Practice: Using groupby and rolling for time-series data
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use
class IntersectTransformer(Transformer):
    def __init__(self, kdata_overlap=0) -> None:
        # ✅ Best Practice: Initializing instance variables for clarity and maintainability
        super().__init__()
        self.kdata_overlap = kdata_overlap

    def transform(self, input_df) -> pd.DataFrame:
        """

        :param input_df:
        :return:
        # 🧠 ML Signal: Use of lambda function for custom transformation
        # 🧠 ML Signal: Grouping by level=0 suggests hierarchical index usage, relevant for time series data
        """
        if self.kdata_overlap > 0:
            # 没有重叠，区间就是(0,0)
            input_df["overlap"] = [(0, 0)] * len(input_df.index)

            def cal_overlap(s):
                high = input_df.loc[s.index, "high"]
                low = input_df.loc[s.index, "low"]
                intersection = intersect_ranges(
                    list(zip(low.to_list(), high.to_list()))
                )
                if intersection:
                    # 设置column overlap为intersection,即重叠区间
                    input_df.at[s.index[-1], "overlap"] = intersection
                return 0

            # ✅ Best Practice: Resetting index to maintain DataFrame consistency after groupby operation
            # ✅ Best Practice: Consider using logging instead of print for better control over log levels and outputs
            input_df[["high", "low"]].groupby(level=0).rolling(
                window=self.kdata_overlap,
                min_periods=self.kdata_overlap,
                # 🧠 ML Signal: Logging or printing function inputs can be useful for debugging and understanding usage patterns
                # ⚠️ SAST Risk (Low): verify_integrity=True can raise exceptions if indexes overlap, ensure proper handling
                # ✅ Best Practice: Concatenating DataFrames to combine original and transformed data
                # 🧠 ML Signal: Function calls with specific parameters can indicate usage patterns and common configurations
            ).apply(cal_overlap, raw=False)

        return input_df


class MaAndVolumeTransformer(Transformer):
    def __init__(self, windows=None, vol_windows=None, kdata_overlap=0) -> None:
        super().__init__()
        if vol_windows is None:
            vol_windows = [30]
        # ✅ Best Practice: Default mutable arguments can lead to unexpected behavior; consider using None and initializing inside the method
        # ✅ Best Practice: Class docstring is missing, consider adding one to describe the class purpose and usage.
        if windows is None:
            windows = [5, 10]
        # 🧠 ML Signal: Initialization of class attributes with default values
        self.windows = windows
        # ✅ Best Practice: Sorting in place is efficient and avoids creating a new list
        self.vol_windows = vol_windows
        self.kdata_overlap = kdata_overlap

    # 🧠 ML Signal: Grouping and quantile calculation on data, common in data preprocessing

    def transform(self, input_df) -> pd.DataFrame:
        # ✅ Best Practice: Setting index names improves DataFrame readability
        for window in self.windows:
            col = f"ma{window}"
            # 🧠 ML Signal: Logging information about data processing steps
            self.indicators.append(col)

            # ✅ Best Practice: Copying DataFrame to avoid modifying the original data
            ma_df = (
                input_df["close"]
                .groupby(level=0)
                .rolling(window=window, min_periods=window)
                .mean()
            )
            ma_df = ma_df.reset_index(level=0, drop=True)
            # ✅ Best Practice: Resetting index for specific level to facilitate operations
            input_df[col] = ma_df

        # ✅ Best Practice: Initializing new column with None for clarity
        for vol_window in self.vol_windows:
            col = "vol_ma{}".format(vol_window)

            # 🧠 ML Signal: Iterating over timestamps, common in time series data processing
            vol_ma_df = (
                input_df["volume"]
                .groupby(level=0)
                .rolling(window=vol_window, min_periods=vol_window)
                .mean()
            )
            vol_ma_df = vol_ma_df.reset_index(level=0, drop=True)
            # ⚠️ SAST Risk (Low): Potential performance issue with repeated DataFrame access
            input_df[col] = vol_ma_df

        # 🧠 ML Signal: Logging final DataFrame state after processing
        if self.kdata_overlap > 0:
            # 🧠 ML Signal: Iterating over DataFrame columns to apply a function
            input_df["overlap"] = [(0, 0)] * len(input_df.index)

            # 🧠 ML Signal: Using DataFrame apply with a lambda function
            def cal_overlap(s):
                high = input_df.loc[s.index, "high"]
                # ✅ Best Practice: Resetting index after DataFrame operations
                low = input_df.loc[s.index, "low"]
                intersection = intersect_ranges(
                    list(zip(low.to_list(), high.to_list()))
                )
                # ✅ Best Practice: Using a function to normalize DataFrame index
                if intersection:
                    input_df.at[s.index[-1], "overlap"] = intersection
                # ✅ Best Practice: Selecting specific columns from DataFrame
                # ⚠️ SAST Risk (Low): Potential data loss if index has duplicates
                # 🧠 ML Signal: Logging information with dynamic content
                # ✅ Best Practice: Explicitly defining module exports
                return 0

            input_df[["high", "low"]].groupby(level=0).rolling(
                window=self.kdata_overlap, min_periods=self.kdata_overlap
            ).apply(cal_overlap, raw=False)

        return input_df


class MacdTransformer(Transformer):
    def __init__(
        self, slow=26, fast=12, n=9, normal=False, count_live_dead=False
    ) -> None:
        super().__init__()
        self.slow = slow
        self.fast = fast
        self.n = n
        self.normal = normal
        self.count_live_dead = count_live_dead

        self.indicators.append("diff")
        self.indicators.append("dea")
        self.indicators.append("macd")

    def transform(self, input_df) -> pd.DataFrame:
        macd_df = input_df.groupby(level=0)["close"].apply(
            lambda x: macd(
                x,
                slow=self.slow,
                fast=self.fast,
                n=self.n,
                return_type="df",
                normal=self.normal,
                count_live_dead=self.count_live_dead,
            )
        )
        macd_df = macd_df.reset_index(level=0, drop=True)
        input_df = pd.concat(
            [input_df, macd_df], axis=1, sort=False, verify_integrity=True
        )
        return input_df

    def transform_one(self, entity_id, df: pd.DataFrame) -> pd.DataFrame:
        print(f"transform_one {entity_id} {df}")
        return macd(
            df["close"],
            slow=self.slow,
            fast=self.fast,
            n=self.n,
            return_type="df",
            normal=self.normal,
            count_live_dead=self.count_live_dead,
        )


class QuantileScorer(Scorer):
    def __init__(self, score_levels=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) -> None:
        self.score_levels = score_levels

    def score(self, input_df):
        self.score_levels.sort(reverse=True)

        quantile_df = input_df.groupby(level=1).quantile(self.score_levels)
        quantile_df.index.names = [self.time_field, "score_result"]

        self.logger.info(
            "factor:{},quantile:\n{}".format(self.factor_name, quantile_df)
        )

        result_df = input_df.copy()
        result_df.reset_index(inplace=True, level="entity_id")
        result_df["quantile"] = None
        for timestamp in quantile_df.index.levels[0]:
            length = len(result_df.loc[result_df.index == timestamp, "quantile"])
            result_df.loc[result_df.index == timestamp, "quantile"] = [
                quantile_df.loc[timestamp].to_dict()
            ] * length

        self.logger.info(
            "factor:{},df with quantile:\n{}".format(self.factor_name, result_df)
        )

        # result_df = result_df.set_index(['entity_id'], append=True)
        # result_df = result_df.sort_index(level=[0, 1])
        #
        # self.logger.info(result_df)
        #
        def calculate_score(df, factor_name, quantile):
            original_value = df[factor_name]
            score_map = quantile.get(factor_name)
            min_score = self.score_levels[-1]

            if original_value < score_map.get(min_score):
                return 0

            for score in self.score_levels[:-1]:
                if original_value >= score_map.get(score):
                    return score

        for factor in input_df.columns.to_list():
            result_df[factor] = result_df.apply(
                lambda x: calculate_score(x, factor, x["quantile"]), axis=1
            )

        result_df = result_df.reset_index()
        result_df = normal_index_df(result_df)
        result_df = result_df.loc[:, self.factors]

        result_df = result_df.loc[~result_df.index.duplicated(keep="first")]

        self.logger.info("factor:{},df:\n{}".format(self.factor_name, result_df))

        return result_df


# the __all__ is generated
__all__ = [
    "ma",
    "ema",
    "live_or_dead",
    "macd",
    "point_in_range",
    "intersect_ranges",
    "combine",
    "distance",
    "intersect",
    "RankScorer",
    "MaTransformer",
    "IntersectTransformer",
    "MaAndVolumeTransformer",
    "MacdTransformer",
    "QuantileScorer",
]
