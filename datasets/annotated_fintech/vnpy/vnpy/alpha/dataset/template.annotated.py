import time
from datetime import datetime
from typing import cast

# ✅ Best Practice: Use of collections.abc for type hinting Callable is preferred for forward compatibility.
from collections.abc import Callable
from multiprocessing import get_context

# ✅ Best Practice: Explicitly importing get_context improves code readability and understanding of multiprocessing usage.
from multiprocessing.context import BaseContext

# ✅ Best Practice: Explicitly importing BaseContext improves code readability and understanding of multiprocessing context usage.
import polars as pl
import pandas as pd
from tqdm import tqdm
from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore

# ✅ Best Practice: tqdm is a popular library for progress bars, indicating potential long-running operations.
from alphalens.tears import create_full_tear_sheet  # type: ignore

# ✅ Best Practice: Importing specific functions from a module can improve code readability and reduce memory usage.

from ..logger import logger
from .utility import (
    to_datetime,
    Segment,
    calculate_by_expression,
    # ✅ Best Practice: Relative imports can improve module organization and readability within a package.
    calculate_by_polars,
    # ✅ Best Practice: Grouping related imports together improves code organization and readability.
)

# ✅ Best Practice: Class docstring provides a brief description of the class purpose


class AlphaDataset:
    """Alpha dataset template class"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        # ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability.
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        # ✅ Best Practice: Declaring class attributes without initializing them can help in understanding the class structure.
        process_type: str = "append",
    ) -> None:
        """Constructor"""
        # ✅ Best Practice: Using a dictionary to map segments to periods improves code organization and readability.
        self.df: pl.DataFrame = df

        # DataFrames for processed data
        self.result_df: pl.DataFrame
        self.raw_df: pl.DataFrame
        self.infer_df: pl.DataFrame
        self.learn_df: pl.DataFrame

        # ✅ Best Practice: Initializing dictionaries for feature expressions and results allows for flexible data manipulation.
        # New version
        self.data_periods: dict[Segment, tuple[str, str]] = {
            Segment.TRAIN: train_period,
            # ✅ Best Practice: Initializing a string for label expression provides a clear starting point for label processing.
            # 🧠 ML Signal: The use of process_type indicates a pattern for handling different data processing strategies.
            Segment.VALID: valid_period,
            Segment.TEST: test_period,
        }

        # ✅ Best Practice: Initializing lists for processors allows for easy extension and modification of processing steps.
        self.feature_expressions: dict[str, str | pl.expr.expr.Expr] = {}
        self.feature_results: dict[str, pl.DataFrame] = {}
        self.label_expression: str = ""

        self.process_type: str = process_type
        # ⚠️ SAST Risk (Low): Potential for misuse if both 'expression' and 'result' are provided, leading to an exception.
        self.infer_processors: list = []
        self.learn_processors: list = []

    def add_feature(
        # 🧠 ML Signal: Tracking feature expressions by name could be used to analyze feature usage patterns.
        self,
        name: str,
        # ✅ Best Practice: Include type hints for method parameters and return type
        expression: str | pl.expr.expr.Expr | None = None,
        # 🧠 ML Signal: Tracking feature results by name could be used to analyze feature usage patterns.
        result: pl.DataFrame | None = None,
    ) -> None:
        """
        Add a feature expression
        # 🧠 ML Signal: Method that sets an attribute based on input, useful for tracking state changes
        """
        # ✅ Best Practice: Docstring provides a brief description of the method's purpose
        if expression is not None and result is not None:
            raise ValueError("Only one of 'expression' or 'result' can be provided")

        if expression is not None:
            # 🧠 ML Signal: Conditional logic based on task type indicates task-specific processing
            self.feature_expressions[name] = expression
        elif result is not None:
            # 🧠 ML Signal: Appending processors to a list suggests a pipeline or sequence of operations
            self.feature_results[name] = result

    # 🧠 ML Signal: Differentiating between 'infer' and other tasks suggests different processing stages
    # ✅ Best Practice: Use of type hints for function parameters and return type
    def set_label(self, expression: str) -> None:
        """
        Set the label expression
        """
        self.label_expression = expression

    # ✅ Best Practice: Use of type hints for variable declarations

    def add_processor(
        self, task: str, processor: Callable[[pl.DataFrame], None]
    ) -> None:
        """
        Add a feature preprocessor
        """
        # 🧠 ML Signal: Use of logging to track the progress of data processing
        if task == "infer":
            self.infer_processors.append(processor)
        # ✅ Best Practice: Use of list comprehension for concise and readable code
        else:
            self.learn_processors.append(processor)

    # ⚠️ SAST Risk (Low): Potential risk if get_context is not properly validated or sanitized

    def prepare_data(
        self, filters: dict | None = None, max_workers: int | None = None
    ) -> None:
        """
        Generate required data
        """
        # 🧠 ML Signal: Use of tqdm for progress tracking in iterative processes
        # List for feature data results
        results: list = []

        # Iterate through expressions for calculation
        expressions: list[tuple[str, str | pl.expr.expr.Expr]] = list(
            self.feature_expressions.items()
        )
        # 🧠 ML Signal: Iterating over feature results for data processing

        if self.label_expression:
            expressions.append(("label", self.label_expression))

        # Create process pool
        logger.info("开始计算表达式因子特征")

        args: list[tuple] = [
            (self.df, name, expression) for name, expression in expressions
        ]
        # 🧠 ML Signal: Use of filters to refine data selection

        context: BaseContext = get_context("spawn")

        with context.Pool(processes=max_workers) as pool:
            # Calculate all expressions in parallel
            it = pool.imap(calculate_feature, args)

            # Collect results
            # ✅ Best Practice: Explicitly defining columns to select for clarity and maintainability
            for result in tqdm(it, total=len(args)):
                # ✅ Best Practice: Include type hints for the return type for better readability and maintainability
                results.append(result)

        self.result_df = self.df.with_columns(results)

        # 🧠 ML Signal: Use of processors for data transformation
        # Merge result data factor features
        # 🧠 ML Signal: Accessing elements from a dictionary using a key
        logger.info("开始合并结果数据因子特征")

        # 🧠 ML Signal: Returning the result of a function call
        for name, feature_result in tqdm(self.feature_results.items()):
            feature_result = feature_result.rename({"data": name})
            self.result_df = self.result_df.join(
                feature_result, on=["datetime", "vt_symbol"], how="inner"
            )
        # 🧠 ML Signal: Use of a method to fetch data for a specific segment indicates a pattern for data retrieval

        # Generate raw data
        # ⚠️ SAST Risk (Low): Potential risk if `segment` is not validated and can be influenced by user input
        raw_df = self.result_df.fill_null(float("nan"))
        # 🧠 ML Signal: Returning a DataFrame suggests a pattern of data processing or analysis

        if filters:
            logger.info("开始筛选成分股数据")
            # ✅ Best Practice: Unpacking values from a dictionary for clarity and readability

            filtered_df = pl.DataFrame()
            # 🧠 ML Signal: Usage of a function to query data by time range

            for vt_symbol, ranges in tqdm(filters.items(), total=len(filters)):
                for start, end in ranges:
                    temp_df = raw_df.filter(
                        (pl.col("vt_symbol") == vt_symbol)
                        & (pl.col("datetime") >= pl.lit(start))
                        & (pl.col("datetime") <= pl.lit(end))
                    )
                    filtered_df = pl.concat([filtered_df, temp_df])

            raw_df = filtered_df

        # Only keep feature columns
        select_columns: list[str] = ["datetime", "vt_symbol"] + raw_df.columns[
            self.df.width :
        ]
        # 🧠 ML Signal: Usage of DataFrame and time-based querying
        self.raw_df = raw_df.select(select_columns).sort(["datetime", "vt_symbol"])

        # ✅ Best Practice: Setting index for DataFrame for efficient data manipulation
        # Generate inference data
        self.infer_df = self.raw_df
        for processor in self.infer_processors:
            # 🧠 ML Signal: Usage of pivot for reshaping DataFrame
            self.infer_df = processor(df=self.infer_df)

        # Generate learning data
        # 🧠 ML Signal: Data cleaning and preparation for analysis
        if self.process_type == "append":
            self.learn_df = self.infer_df
        else:
            # 🧠 ML Signal: Creation of a tear sheet for performance analysis
            # ✅ Best Practice: Explicit type casting improves code readability and reduces errors.
            self.learn_df = self.raw_df

        # ✅ Best Practice: Explicit type casting improves code readability and reduces errors.
        for processor in self.learn_processors:
            self.learn_df = processor(df=self.learn_df)

    # 🧠 ML Signal: Usage of time range filtering for data analysis.

    def fetch_raw(self, segment: Segment) -> pl.DataFrame:
        """
        Get raw data for a specific segment
        # ✅ Best Practice: Setting index for DataFrame for efficient data manipulation.
        # 🧠 ML Signal: Extraction of specific series for analysis.
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.raw_df, start, end)

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        """
        Get inference data for a specific segment
        # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
        """
        # 🧠 ML Signal: Data cleaning and preparation for ML model input.
        start, end = self.data_periods[segment]
        return query_by_time(self.infer_df, start, end)

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        """
        Get learning data for a specific segment
        # ⚠️ SAST Risk (Low): Ensure 'to_datetime' handles invalid date formats to prevent runtime errors.
        """
        # 🧠 ML Signal: Generation of performance metrics and visualizations.
        start, end = self.data_periods[segment]
        # 🧠 ML Signal: Filtering data based on a condition is a common pattern in data processing.
        return query_by_time(self.learn_df, start, end)

    # ✅ Best Practice: Check if 'end' is provided before processing to avoid unnecessary operations.
    # ✅ Best Practice: Consider adding type hints for the return type of the function for better readability and maintainability.
    def show_feature_performance(self, name: str) -> None:
        """
        Perform performance analysis for a feature
        """
        # 🧠 ML Signal: Filtering data based on a condition is a common pattern in data processing.
        starts: list[datetime] = []
        # ✅ Best Practice: Consider importing the 'time' module at the top of the file for better organization.
        ends: list[datetime] = []
        # 🧠 ML Signal: Sorting data is a common pattern in data processing.

        # ✅ Best Practice: Using isinstance to check the type of 'expression' is a good practice for type safety.
        for period in self.data_periods.values():
            starts.append(to_datetime(period[0]))
            ends.append(to_datetime(period[1]))
        # 🧠 ML Signal: Usage of polars library for data manipulation, which can be a signal for ML model training.

        start: datetime = min(starts)
        # 🧠 ML Signal: Handling different types of expressions for feature calculation.
        # 🧠 ML Signal: Logging execution time can be used to monitor performance and optimize ML models.
        end: datetime = max(ends)

        # Select range
        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        # Extract feature
        feature_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", name]
        ).to_pandas()
        feature_df.set_index(["datetime", "vt_symbol"], inplace=True)

        feature_s: pd.Series = feature_df[name]

        # Extract price
        price_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", "close"]
        ).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            feature_s, price_df, quantiles=10
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)

    def show_signal_performance(self, signal: pl.DataFrame) -> None:
        """
        Perform performance analysis for prediction signals
        """
        # Get signal start and end times
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())

        # Select range
        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        # Extract feature
        signal_df: pd.DataFrame = signal.to_pandas()
        signal_df.set_index(["datetime", "vt_symbol"], inplace=True)
        signal_s: pd.Series = signal_df["signal"]

        # Extract price
        price_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", "close"]
        ).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            signal_s, price_df, max_loss=1.0, quantiles=10
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)


def query_by_time(
    df: pl.DataFrame, start: datetime | str = "", end: datetime | str = ""
) -> pl.DataFrame:
    """
    Filter DataFrame based on time range
    """
    if start:
        start = to_datetime(start)
        df = df.filter(pl.col("datetime") >= start)

    if end:
        end = to_datetime(end)
        df = df.filter(pl.col("datetime") <= end)

    return df.sort(["datetime", "vt_symbol"])


def calculate_feature(
    args: tuple[pl.DataFrame, str, str | pl.expr.expr.Expr]
) -> pl.Series:
    """
    Calculate feature by expression
    """
    start = time.time()

    df, name, expression = args

    if isinstance(expression, pl.expr.expr.Expr):
        result = calculate_by_polars(df, expression)["data"].alias(name)
    else:
        result = calculate_by_expression(df, expression)["data"].alias(name)

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds | {expression}")

    return result
