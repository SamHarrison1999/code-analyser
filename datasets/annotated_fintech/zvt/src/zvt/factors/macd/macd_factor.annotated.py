# -*- coding: utf-8 -*-
from typing import List, Optional

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
import numpy as np
import pandas as pd

# 🧠 ML Signal: Inheritance from TechnicalFactor suggests a pattern for feature engineering in financial data

from zvt.factors.algorithm import MacdTransformer

# 🧠 ML Signal: Use of a transformer object indicates a pattern of data transformation
# ✅ Best Practice: Type hinting improves code readability and maintainability
from zvt.factors.technical_factor import TechnicalFactor

# ✅ Best Practice: Initializing class variables directly can improve readability and maintainability

# ✅ Best Practice: Explicitly returning None improves code clarity
# ✅ Best Practice: Specify the return type for better readability and maintainability


class MacdFactor(TechnicalFactor):
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
    transformer = MacdTransformer(count_live_dead=True)
    # ✅ Best Practice: Use list comprehension for concise and efficient list creation

    # 🧠 ML Signal: Returns a dictionary with specific keys and values, indicating a pattern in data representation
    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        # ✅ Best Practice: Add a docstring to describe the purpose and usage of the class
        return None

    # 🧠 ML Signal: Method overriding in class inheritance

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        # 🧠 ML Signal: DataFrame column selection and transformation
        # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
        return [self.factor_df[["diff", "dea", "macd"]]]

    # ✅ Best Practice: Use to_frame() for creating a DataFrame from a Series

    # ✅ Best Practice: Class variables should have comments explaining their purpose
    def drawer_sub_col_chart(self) -> Optional[dict]:
        # ✅ Best Practice: Call to superclass method ensures base functionality is executed.
        return {"diff": "line", "dea": "line", "macd": "bar"}


# 🧠 ML Signal: Use of DataFrame and groupby operation indicates data processing pattern.


class BullFactor(MacdFactor):
    def compute_result(self):
        super().compute_result()
        self.result_df = self.factor_df["bull"].to_frame(name="filter_result")


# 🧠 ML Signal: Use of rolling window operation is common in time series analysis.

# 🧠 ML Signal: Use of lambda function for custom aggregation.


# 🧠 ML Signal: Inheritance from MacdFactor suggests a pattern for financial analysis
class KeepBullFactor(BullFactor):
    # 🧠 ML Signal: The pattern attribute could be used to identify specific market conditions
    keep_window = 10
    # ✅ Best Practice: Resetting index improves DataFrame consistency after groupby operations.

    # ✅ Best Practice: Call to superclass method ensures base class functionality is preserved.
    def compute_result(self):
        # 🧠 ML Signal: Assignment back to DataFrame column indicates data transformation.
        super().compute_result()
        # ✅ Best Practice: Using shift() to access previous row values is a common and efficient pattern in data manipulation.
        df = (
            self.result_df["filter_result"]
            # 🧠 ML Signal: Pattern matching on DataFrame columns can indicate feature engineering for ML models.
            .groupby(level=0).rolling(
                window=self.keep_window, min_periods=self.keep_window
            )
            # ✅ Best Practice: Converting a Series to a DataFrame with a specific column name improves code readability.
            .apply(lambda x: np.logical_and.reduce(x))
            # ✅ Best Practice: Use of descriptive variable names improves code readability.
        )
        df = df.reset_index(level=0, drop=True)
        # ✅ Best Practice: Converting a Series to a DataFrame with a specific column name enhances clarity.
        self.result_df["filter_result"] = df


# 🧠 ML Signal: Entry point for script execution, common pattern for standalone scripts.
# 🧠 ML Signal: Instantiation of a class with specific parameters, useful for understanding usage patterns.
# 🧠 ML Signal: Method chaining pattern, often used in fluent interfaces.
# ✅ Best Practice: Defining __all__ for module exports improves code maintainability and clarity.
# 金叉 死叉 持续时间 切换点
class LiveOrDeadFactor(MacdFactor):
    pattern = [-5, 1]

    def compute_result(self):
        super().compute_result()
        self.factor_df["pre"] = self.factor_df["live_count"].shift()
        s = (self.factor_df["pre"] <= self.pattern[0]) & (
            self.factor_df["live_count"] >= self.pattern[1]
        )
        self.result_df = s.to_frame(name="filter_result")


class GoldCrossFactor(MacdFactor):
    def compute_result(self):
        super().compute_result()
        s = self.factor_df["live"] == 1
        self.result_df = s.to_frame(name="filter_result")


if __name__ == "__main__":
    f = GoldCrossFactor(
        provider="em", entity_provider="em", entity_ids=["stock_sz_000338"]
    )
    f.drawer().draw(show=True)


# the __all__ is generated
__all__ = [
    "MacdFactor",
    "BullFactor",
    "KeepBullFactor",
    "LiveOrDeadFactor",
    "GoldCrossFactor",
]
