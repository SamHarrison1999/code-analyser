from datetime import datetime
# ✅ Best Practice: Grouping standard library imports together improves readability.
from enum import Enum
from typing import Union
# ✅ Best Practice: Grouping standard library imports together improves readability.

import polars as pl
# ✅ Best Practice: Grouping standard library imports together improves readability.


# ✅ Best Practice: Grouping third-party library imports together improves readability.
class DataProxy:
    # 🧠 ML Signal: Usage of Polars DataFrame as a parameter
    """Feature data proxy"""
    # ✅ Best Practice: Type hinting for constructor parameters and return type

    # ✅ Best Practice: Type hinting for the method return type improves code readability and maintainability
    def __init__(self, df: pl.DataFrame) -> None:
        # 🧠 ML Signal: Accessing the last column of a DataFrame
        """Constructor"""
        # ✅ Best Practice: Type hinting for instance variables
        self.name: str = df.columns[-1]
        # ✅ Best Practice: Type hinting for the variable improves code readability and maintainability
        self.df: pl.DataFrame = df.rename({self.name: "data"})
    # 🧠 ML Signal: Renaming a DataFrame column

    # 🧠 ML Signal: Usage of method chaining with 'with_columns' indicates a pattern of data transformation
    # ✅ Best Practice: Type hinting improves code readability and maintainability
        # Note that for numerical expressions, variables should be placed before numbers. e.g. a * 2

    # 🧠 ML Signal: Returning a custom object 'DataProxy' could indicate a pattern of wrapping or encapsulating data
    def result(self, s: pl.Series) -> "DataProxy":
        # 🧠 ML Signal: Use of isinstance to check type can indicate dynamic type handling
        """Convert series data to feature object"""
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        # 🧠 ML Signal: Use of Polars library (pl.Series) for data manipulation
        result = result.with_columns(other=s)

        return DataProxy(result)
    # ✅ Best Practice: Include a docstring to describe the method's purpose

    # 🧠 ML Signal: Use of a method to process and return results
    def __add__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        # 🧠 ML Signal: Use of isinstance to check type of 'other'
        """Addition operation"""
        if isinstance(other, DataProxy):
            # 🧠 ML Signal: Accessing a DataFrame column for arithmetic operations
            s: pl.Series = self.df["data"] + other.df["data"]
        else:
            s = self.df["data"] + other
        # 🧠 ML Signal: Handling different data types for arithmetic operations
        return self.result(s)
    # 🧠 ML Signal: Use of isinstance to check type, common pattern in dynamic typing

    # 🧠 ML Signal: Use of a method to process and return the result
    # ⚠️ SAST Risk (Low): Potential for AttributeError if 'df' or 'data' attributes do not exist
    def __sub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Subtraction operation"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] - other.df["data"]
        # ⚠️ SAST Risk (Low): Potential for AttributeError if 'df' or 'data' attributes do not exist
        else:
            s = self.df["data"] - other
        # ✅ Best Practice: Returning the result of an operation, maintains method chaining
        # ✅ Best Practice: Check for type of 'other' to ensure correct operation
        return self.result(s)
    # 🧠 ML Signal: Usage of multiplication operation with custom class

    def __mul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Multiplication operation"""
        if isinstance(other, DataProxy):
            # 🧠 ML Signal: Usage of multiplication operation with primitive types
            s: pl.Series = self.df["data"] * other.df["data"]
        else:
            # ✅ Best Practice: Encapsulate result in a method for consistency and potential future changes
            # ✅ Best Practice: Check if 'other' is an instance of 'DataProxy' to handle different division logic.
            s = self.df["data"] * other
        # 🧠 ML Signal: Usage of division operation between two 'DataProxy' objects.
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Right multiplication operation"""
        # ✅ Best Practice: Type hinting for return value improves code readability and maintainability
        # 🧠 ML Signal: Usage of division operation with a scalar value.
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] * other.df["data"]
        # ✅ Best Practice: Encapsulate the result in a 'DataProxy' object for consistent return type.
        else:
            # 🧠 ML Signal: Accessing a specific column from a DataFrame
            s = self.df["data"] * other
        return self.result(s)
    # 🧠 ML Signal: Returning a processed result from a method

    # ✅ Best Practice: Check if 'other' is an instance of 'DataProxy' to handle different types appropriately
    def __truediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        # 🧠 ML Signal: Usage of Polars library for data manipulation
        """Division operation"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] / other.df["data"]
        else:
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            # 🧠 ML Signal: Handling comparison with primitive data types
            s = self.df["data"] / other
        return self.result(s)
    # ✅ Best Practice: Return a consistent type ('DataProxy') for method chaining

    # 🧠 ML Signal: Use of isinstance to check type
    def __abs__(self) -> "DataProxy":
        """Get absolute value"""
        # 🧠 ML Signal: Use of Polars library for data manipulation
        s: pl.Series = self.df["data"].abs()
        return self.result(s)
    # ✅ Best Practice: Type hinting improves code readability and maintainability

    # 🧠 ML Signal: Handling different data types in comparison
    def __gt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Greater than comparison"""
        # ✅ Best Practice: Returning a result from a method
        # 🧠 ML Signal: Use of isinstance to check type
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] > other.df["data"]
        # 🧠 ML Signal: Use of pandas-like operations for data manipulation
        else:
            s = self.df["data"] > other
        return self.result(s)
    # 🧠 ML Signal: Handling different data types in operations

    # ✅ Best Practice: Check if 'other' is an instance of 'DataProxy' to handle different types appropriately.
    def __ge__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        # 🧠 ML Signal: Returning a result from a method
        # 🧠 ML Signal: Usage of pandas-like operations for data comparison.
        """Greater than or equal comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] >= other.df["data"]
        else:
            # 🧠 ML Signal: Handling scalar comparison with data.
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            s = self.df["data"] >= other
        return self.result(s)
    # ✅ Best Practice: Return the result of the comparison wrapped in a 'DataProxy' object.

    # 🧠 ML Signal: Use of isinstance to check type
    def __lt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than comparison"""
        # 🧠 ML Signal: Accessing attributes of an object
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] < other.df["data"]
        else:
            # 🧠 ML Signal: Handling different data types in comparison
            s = self.df["data"] < other
        # ✅ Best Practice: Importing specific functions instead of entire modules for clarity and to avoid namespace pollution.
        # 🧠 ML Signal: Returning the result of a comparison operation
        return self.result(s)

    def __le__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than or equal comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] <= other.df["data"]
        else:
            s = self.df["data"] <= other
        return self.result(s)

    def __eq__(self, other: Union["DataProxy", int, float]) -> "DataProxy":    # type: ignore
        """Equal comparison"""
        if isinstance(other, DataProxy):
            s = self.df["data"] == other.df["data"]
        else:
            s = self.df["data"] == other
        return self.result(s)


def calculate_by_expression(df: pl.DataFrame, expression: str) -> pl.DataFrame:
    """Execute calculation based on expression"""
    # Import operators locally to avoid polluting global namespace
    # ✅ Best Practice: Using type hints for better code readability and maintainability.
    from .ts_function import (              # noqa
        ts_delay,
        ts_min, ts_max,
        ts_argmax, ts_argmin,
        ts_rank, ts_sum,
        # ✅ Best Practice: Using descriptive variable names for better readability.
        ts_mean, ts_std,
        ts_slope, ts_quantile,
        # 🧠 ML Signal: Dynamic creation of variables based on DataFrame columns.
        # ✅ Best Practice: Add type hints for function parameters and return type for better readability and maintainability
        ts_rsquare, ts_resi,
        ts_corr,
        # 🧠 ML Signal: Use of Polars library for data manipulation
        # ⚠️ SAST Risk (High): Use of eval() can lead to code injection vulnerabilities if the input is not properly sanitized.
        # ✅ Best Practice: Returning the DataFrame directly for clarity.
        # ✅ Best Practice: Use of select method for efficient column selection in Polars
        ts_less, ts_greater,
        ts_log, ts_abs
    )
    from .cs_function import (              # noqa
        cs_rank,
        # ✅ Best Practice: Include type hinting for function parameters and return type for better readability and maintainability.
        cs_mean,
        # ✅ Best Practice: Use of alias to rename the result of an expression for clarity
        cs_std
    )
    # 🧠 ML Signal: Checking the type of a variable to determine processing logic.
    from .ta_function import (              # noqa
        # 🧠 ML Signal: Conditional logic based on string content.
        ta_rsi,
        ta_atr
    )

    # Extract feature objects to local space
    d: dict = locals()
    # ✅ Best Practice: Use of Enum for segment values improves code readability and maintainability
    # ⚠️ SAST Risk (Low): Potential risk if the input string is not a valid date format.

    for column in df.columns:
        # Filter index columns
        # ✅ Best Practice: Enum members are named in uppercase to follow Python naming conventions
        if column in {"datetime", "vt_symbol"}:
            continue

        # Cache feature df
        column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df)

    # Use eval to execute calculation
    other: DataProxy = eval(expression, {}, d)

    # Return result DataFrame
    return other.df


def calculate_by_polars(df: pl.DataFrame, expression: pl.expr.expr.Expr) -> pl.DataFrame:
    """Execute calculation based on Polars expression"""
    return df.select([
        "datetime",
        "vt_symbol",
        expression.alias("data")
    ])


def to_datetime(arg: datetime | str) -> datetime:
    """Convert time data type"""
    if isinstance(arg, str):
        if "-" in arg:
            fmt: str = "%Y-%m-%d"
        else:
            fmt = "%Y%m%d"

        return datetime.strptime(arg, fmt)
    else:
        return arg


class Segment(Enum):
    """Data segment enumeration values"""

    TRAIN = 1
    VALID = 2
    TEST = 3