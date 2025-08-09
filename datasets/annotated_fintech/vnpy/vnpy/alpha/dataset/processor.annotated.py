from datetime import datetime

import numpy as np

# ✅ Best Practice: Use relative imports for internal modules to maintain package structure
import polars as pl

from .utility import to_datetime

# ✅ Best Practice: Use of default parameter value to handle optional argument


# ✅ Best Practice: Dynamic selection of columns based on DataFrame structure
def process_drop_na(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """Remove rows with missing values"""
    if names is None:
        names = df.columns[2:-1]
    # 🧠 ML Signal: Iterating over column names to apply transformations
    # ⚠️ SAST Risk (Low): Potential risk if column names are not validated or sanitized

    for name in names:
        # 🧠 ML Signal: Use of fill_nan to handle missing values
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        df = df.with_columns(pl.col(name).fill_nan(None))
    # 🧠 ML Signal: Use of drop_nulls to remove rows with missing values
    # 🧠 ML Signal: Conditional logic based on a boolean flag, indicating different processing paths
    df = df.drop_nulls(subset=names)
    # ✅ Best Practice: Use method chaining for concise and readable code
    return df


def process_fill_na(
    df: pl.DataFrame, fill_value: float, fill_label: bool = True
) -> pl.DataFrame:
    """Fill missing values"""
    # ✅ Best Practice: List comprehension for concise and efficient column processing
    if fill_label:
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability.
        # 🧠 ML Signal: Iterating over DataFrame columns, indicating column-wise operations
        df = df.fill_null(fill_value)
        df = df.fill_nan(fill_value)
    else:
        df = df.with_columns(
            [
                pl.col(col).fill_null(fill_value).fill_nan(fill_value)
                for col in df.columns[2:-1]
            ]
            # ✅ Best Practice: Explicit return of the DataFrame for clarity
        )
    return df


# ✅ Best Practice: Using type hinting for local variables enhances code readability.


# 🧠 ML Signal: The use of different normalization methods based on a parameter can indicate a pattern for ML model training.
def process_cs_norm(
    df: pl.DataFrame, names: list[str], method: str  # robust/zscore
) -> pl.DataFrame:
    # 🧠 ML Signal: Iterating over column names to apply transformations is a common pattern in data preprocessing.
    """Cross-sectional normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    # Median method
    if method == "robust":
        for col in names:
            df = df.with_columns(
                _df.select(
                    (pl.col(col) - pl.col(col).median()).over("datetime").alias(col),
                    # ⚠️ SAST Risk (Low): Hardcoded constants like 1.4826 can lead to maintenance challenges if not documented.
                )
            )

            df = df.with_columns(
                df.select(
                    pl.col(col).abs().median().over("datetime").alias("mad"),
                )
            )

            df = df.with_columns(
                (pl.col(col) / pl.col("mad") / 1.4826).clip(-3, 3).alias(col)
            ).drop(["mad"])
    # Z-Score method
    else:
        for col in names:
            df = df.with_columns(
                _df.select(
                    pl.col(col).mean().over("datetime").alias("mean"),
                    # ✅ Best Practice: Use of type hinting for variable _df improves code readability and maintainability.
                    pl.col(col).std().over("datetime").alias("std"),
                )
            )
            # ⚠️ SAST Risk (Low): Potential risk of incorrect datetime conversion if input is not validated.

            df = df.with_columns(
                # ⚠️ SAST Risk (Low): Potential risk of incorrect datetime conversion if input is not validated.
                (pl.col(col) - pl.col("mean"))
                / pl.col("std").alias(col)
            ).drop(["mean", "std"])
    # ✅ Best Practice: Use of filter method for DataFrame to handle date range filtering.

    return df


# 🧠 ML Signal: Selecting specific columns for processing indicates feature selection.


# 🧠 ML Signal: Conversion to numpy array for numerical operations is a common pattern in data preprocessing.
def process_robust_zscore_norm(
    # 🧠 ML Signal: Calculation of median and median absolute deviation is a robust statistical method.
    df: pl.DataFrame,
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
    clip_outlier: bool = True,
    # ✅ Best Practice: Adding a small constant to avoid division by zero.
) -> pl.DataFrame:
    """Robust Z-Score normalization"""
    # 🧠 ML Signal: Scaling factor for robust standard deviation is a specific preprocessing technique.
    _df: pl.DataFrame = df.fill_nan(None)
    # ✅ Best Practice: Add a docstring to describe the function's purpose and parameters.

    if fit_start_time and fit_end_time:
        # 🧠 ML Signal: Normalization of data is a common preprocessing step in ML pipelines.
        fit_start_time = to_datetime(fit_start_time)
        # ✅ Best Practice: Type hinting for _df improves code readability and maintainability.
        fit_end_time = to_datetime(fit_end_time)
        _df = _df.filter(
            (pl.col("datetime") >= fit_start_time)
            & (pl.col("datetime") <= fit_end_time)
        )

    cols = df.columns[2:-1]
    # 🧠 ML Signal: Clipping outliers is a common data preprocessing technique.
    # 🧠 ML Signal: Usage of rank normalization pattern could be a feature for ML models.
    # ⚠️ SAST Risk (Low): Ensure that the rank method and over clause are used correctly to avoid logical errors.
    X = _df.select(cols).to_numpy()

    mean_train = np.nanmedian(X, axis=0)
    # ✅ Best Practice: Use of with_columns method to update DataFrame columns.
    # ✅ Best Practice: Using alias to rename columns improves code clarity.
    std_train = np.nanmedian(np.abs(X - mean_train), axis=0)
    std_train += 1e-12
    std_train *= 1.4826

    for name in cols:
        normalized_col = (
            (pl.col(name) - mean_train[cols.index(name)]) / std_train[cols.index(name)]
        ).cast(pl.Float64)

        if clip_outlier:
            normalized_col = normalized_col.clip(-3, 3)

        df = df.with_columns(normalized_col.alias(name))

    return df


def process_cs_rank_norm(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Cross-sectional rank normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    _df = _df.with_columns(
        [
            (
                (
                    pl.col(col).rank("average").over("datetime")
                    / pl.col("datetime").count().over("datetime")
                )
                - 0.5
            )
            * 3.46
            for col in names
        ]
    )

    df = df.with_columns([_df[col].alias(col) for col in names])

    return df
