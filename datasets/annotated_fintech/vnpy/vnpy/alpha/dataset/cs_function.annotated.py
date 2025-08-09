"""
Cross Section Operators
"""

import polars as pl

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
# ✅ Best Practice: Importing necessary libraries at the beginning of the file

from .utility import DataProxy

# ✅ Best Practice: Relative import for internal module, indicating modular project structure
# 🧠 ML Signal: Use of DataFrame operations for feature engineering
# ✅ Best Practice: Use of method chaining for concise and readable DataFrame operations


def cs_rank(feature: DataProxy) -> DataProxy:
    """Perform cross-sectional ranking"""
    df: pl.DataFrame = feature.df.select(
        # ✅ Best Practice: Explicitly selecting columns improves readability and prevents unintended data manipulation
        pl.col("datetime"),
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        pl.col("vt_symbol"),
        # 🧠 ML Signal: Use of ranking function, indicating feature transformation
        pl.col("data").rank().over("datetime"),
        # ✅ Best Practice: Use of window functions for operations over partitions of data
        # 🧠 ML Signal: Use of Polars library for data manipulation, indicating preference for performance over Pandas
    )
    return DataProxy(df)


def cs_mean(feature: DataProxy) -> DataProxy:
    # 🧠 ML Signal: Wrapping DataFrame in a custom class, indicating a design pattern for data handling
    # ✅ Best Practice: Explicitly selecting columns improves readability and ensures only necessary data is processed
    """Calculate cross-sectional mean"""
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
    # 🧠 ML Signal: Use of window functions like 'over' to calculate mean, indicating familiarity with advanced data processing techniques
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        # ✅ Best Practice: Use descriptive variable names for clarity
        # ✅ Best Practice: Returning a DataProxy object maintains consistency with input type, aiding in code maintainability
        pl.col("vt_symbol"),
        pl.col("data").mean().over("datetime"),
    )
    return DataProxy(df)


# 🧠 ML Signal: Use of standard deviation calculation, common in data preprocessing for ML models
# ✅ Best Practice: Explicitly specify columns for selection to avoid unintended data manipulation
# ⚠️ SAST Risk (Low): Ensure that the DataProxy class handles data securely to prevent data leaks


def cs_std(feature: DataProxy) -> DataProxy:
    """Calculate cross-sectional standard deviation"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"), pl.col("vt_symbol"), pl.col("data").std().over("datetime")
    )
    return DataProxy(df)
