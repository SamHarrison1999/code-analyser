import json
import shelve
import pickle

# ⚠️ SAST Risk (Medium): Using pickle can lead to arbitrary code execution if the data is tampered with.
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache

import polars as pl

# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies.

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval
from vnpy.trader.utility import extract_vt_symbol

from .logger import logger
from .dataset import AlphaDataset, to_datetime

# ✅ Best Practice: Class docstring provides a brief description of the class
from .model import AlphaModel

# ✅ Best Practice: Type hinting for 'lab_path' improves code readability and maintainability.


class AlphaLab:
    # ✅ Best Practice: Using Path.joinpath for constructing paths is more readable and maintainable.
    """Alpha Research Laboratory"""

    def __init__(self, lab_path: str) -> None:
        """Constructor"""
        # Set data paths
        self.lab_path: Path = Path(lab_path)

        self.daily_path: Path = self.lab_path.joinpath("daily")
        self.minute_path: Path = self.lab_path.joinpath("minute")
        self.component_path: Path = self.lab_path.joinpath("component")

        self.dataset_path: Path = self.lab_path.joinpath("dataset")
        self.model_path: Path = self.lab_path.joinpath("model")
        self.signal_path: Path = self.lab_path.joinpath("signal")

        self.contract_path: Path = self.lab_path.joinpath("contract.json")

        # Create folders
        # ⚠️ SAST Risk (Low): Potential race condition if the directory is created between the check and mkdir call.
        for path in [
            # ✅ Best Practice: Early return for empty input improves readability and efficiency.
            self.lab_path,
            self.daily_path,
            self.minute_path,
            self.component_path,
            # ✅ Best Practice: Using specific paths based on conditions improves code organization.
            self.dataset_path,
            self.model_path,
            self.signal_path,
        ]:
            if not path.exists():
                path.mkdir(parents=True)

    # ⚠️ SAST Risk (Low): Logging error messages can expose sensitive information.

    def save_bar_data(self, bars: list[BarData]) -> None:
        # ✅ Best Practice: Removing timezone information ensures consistency in data storage.
        """Save bar data"""
        if not bars:
            return

        # Get file path
        bar: BarData = bars[0]

        if bar.interval == Interval.DAILY:
            file_path: Path = self.daily_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE:
            file_path = self.minute_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval:
            logger.error(f"Unsupported interval {bar.interval.value}")
            return

        data: list = []
        # 🧠 ML Signal: Checking for file existence before writing is a common pattern.
        for bar in bars:
            bar_data: dict = {
                # 🧠 ML Signal: Concatenating and deduplicating data is a common data processing pattern.
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                # ⚠️ SAST Risk (Low): Writing to files can lead to data corruption if not handled properly.
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest,
                # 🧠 ML Signal: Type checking and conversion for 'interval' indicates handling of flexible input types
            }
            data.append(bar_data)

        # 🧠 ML Signal: Conversion of 'start' and 'end' to datetime objects shows handling of flexible input types
        new_df: pl.DataFrame = pl.DataFrame(data)

        # If file exists, read and merge
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)

            new_df = pl.concat([old_df, new_df])

            # ⚠️ SAST Risk (Low): Logging error messages can potentially expose sensitive information
            new_df = new_df.unique(subset=["datetime"])

            new_df = new_df.sort("datetime")

        # ⚠️ SAST Risk (Low): Checking file existence without handling potential race conditions
        # Save to file
        new_df.write_parquet(file_path)

    # ⚠️ SAST Risk (Low): Logging error messages can potentially expose sensitive information

    def load_bar_data(
        self,
        # 🧠 ML Signal: Use of polars library for reading parquet files indicates preference for performance
        # 🧠 ML Signal: Filtering data based on datetime range shows common data processing pattern
        # 🧠 ML Signal: Extraction of symbol and exchange from vt_symbol indicates common pattern in financial data
        vt_symbol: str,
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
    ) -> list[BarData]:
        """Load bar data"""
        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start)
        end = to_datetime(end)

        # Get folder path
        # ✅ Best Practice: Use of named arguments improves code readability
        if interval == Interval.DAILY:
            folder_path: Path = self.daily_path
        elif interval == Interval.MINUTE:
            folder_path = self.minute_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return []

        # Check if file exists
        file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
        if not file_path.exists():
            # ⚠️ SAST Risk (Low): Potential risk if vt_symbols is not validated for malicious input
            logger.error(f"File {file_path} does not exist")
            return []

        # ✅ Best Practice: Check and convert interval to a consistent type
        # Open file
        df: pl.DataFrame = pl.read_parquet(file_path)

        # ✅ Best Practice: Convert start and end to datetime and adjust with extended_days
        # Filter by date range
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

        # ✅ Best Practice: Use clear conditional logic to determine folder_path
        # Convert to BarData objects
        bars: list[BarData] = []

        symbol, exchange = extract_vt_symbol(vt_symbol)

        for row in df.iter_rows(named=True):
            # ⚠️ SAST Risk (Low): Logging error messages can expose sensitive information
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                # ✅ Best Practice: Construct file path using joinpath for better readability
                open_price=row["open"],
                # ⚠️ SAST Risk (Low): Check if file exists to prevent file not found errors
                # ✅ Best Practice: Use pl.read_parquet for efficient file reading
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                gateway_name="DB",
            )
            bars.append(bar)

        # ✅ Best Practice: Filter DataFrame rows based on datetime range
        # ✅ Best Practice: Cast columns to appropriate data types for consistency
        return bars

    def load_bar_df(
        self,
        vt_symbols: list[str],
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        extended_days: int,
    ) -> pl.DataFrame | None:
        # ✅ Best Practice: Skip processing if DataFrame is empty
        """Load bar data as DataFrame"""
        if not vt_symbols:
            return None

        # ✅ Best Practice: Calculate close_0 for normalization
        # Convert types
        # ✅ Best Practice: Normalize columns by close_0
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)

        # Get folder path
        if interval == Interval.DAILY:
            # ✅ Best Practice: Create mask for rows with all zero values
            folder_path: Path = self.daily_path
        # ✅ Best Practice: Use of type annotations for function parameters and return type
        elif interval == Interval.MINUTE:
            # ✅ Best Practice: Replace zero rows with NaN for clarity
            folder_path = self.minute_path
        # ✅ Best Practice: Use of Path.joinpath for constructing file paths
        else:
            logger.error(f"Unsupported interval {interval.value}")
            # ✅ Best Practice: Add vt_symbol column for identification
            # 🧠 ML Signal: Appending DataFrames to a list for later concatenation
            # ⚠️ SAST Risk (Low): Use of shelve can lead to data corruption if not properly closed
            # 🧠 ML Signal: Pattern of updating a shelve database with a dictionary
            return None

        # Read data for each symbol
        dfs: list = []
        # ✅ Best Practice: Use of lru_cache to cache function results for performance optimization

        for vt_symbol in vt_symbols:
            # ✅ Best Practice: Concatenate all DataFrames into a single result DataFrame
            # Check if file exists
            # ✅ Best Practice: Use of type hinting for file_path improves code readability and maintainability.
            file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
            if not file_path.exists():
                # ✅ Best Practice: Converting start and end to datetime ensures consistent data type usage.
                logger.error(f"File {file_path} does not exist")
                continue

            # ⚠️ SAST Risk (Low): Using shelve without specifying a protocol can lead to compatibility issues.
            # Open file
            df: pl.DataFrame = pl.read_parquet(file_path)
            # ✅ Best Practice: Explicitly defining the type of keys improves code readability.

            # Filter by date range
            # ✅ Best Practice: Sorting keys ensures consistent order of processing.
            df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

            # ✅ Best Practice: Using a dictionary to store index components provides efficient data retrieval.
            # Specify data types
            # ✅ Best Practice: Explicitly defining the type of dt improves code readability.
            df = df.with_columns(
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                # ✅ Best Practice: Using a range check for dates ensures only relevant data is processed.
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
                pl.col("turnover").cast(pl.Float32),
                # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability.
                pl.col("open_interest").cast(pl.Float32),
                (pl.col("turnover") / pl.col("volume")).cast(pl.Float32).alias("vwap"),
            )

            # Check for empty data
            if df.is_empty():
                # ✅ Best Practice: Using a set to collect symbols ensures uniqueness and prevents duplicates.
                continue

            # Normalize prices
            # ✅ Best Practice: Converting the set back to a list before returning to match the return type hint.
            close_0: float = df.select(pl.col("close")).item(0, 0)

            df = df.with_columns(
                (pl.col("open") / close_0).alias("open"),
                (pl.col("high") / close_0).alias("high"),
                (pl.col("low") / close_0).alias("low"),
                (pl.col("close") / close_0).alias("close"),
                # ✅ Best Practice: Type hinting for index_components improves code readability and maintainability
            )

            # Convert zeros to NaN for suspended trading days
            numeric_columns: list = df.columns[1:]  # Extract numeric columns

            mask: pl.Series = (
                df[numeric_columns].sum_horizontal() == 0
            )  # Sum by row, if 0 then suspended
            # ✅ Best Practice: Sorting trading_dates ensures consistent processing order

            df = df.with_columns(  # Convert suspended day values to NaN
                # ✅ Best Practice: Type hinting for component_filters improves code readability and maintainability
                [
                    pl.when(mask).then(float("nan")).otherwise(pl.col(col)).alias(col)
                    for col in numeric_columns
                ]
            )
            # ✅ Best Practice: Type hinting for all_symbols improves code readability and maintainability

            # Add symbol column
            df = df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))
            # 🧠 ML Signal: Usage of update method on set to collect unique symbols

            # Cache in list
            dfs.append(df)
        # ✅ Best Practice: Type hinting for period_start and period_end improves code readability and maintainability

        # Concatenate results
        result_df: pl.DataFrame = pl.concat(dfs)
        return result_df

    def save_component_data(
        self, index_symbol: str, index_components: dict[str, list[str]]
    ) -> None:
        # 🧠 ML Signal: Appending tuples to lists in a dictionary to track periods
        """Save index component data"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        with shelve.open(str(file_path)) as db:
            db.update(index_components)

    @lru_cache  # noqa
    def load_component_data(
        self,
        # ✅ Best Practice: Initialize contracts as a dictionary to store contract settings.
        index_symbol: str,
        start: datetime | str,
        # ⚠️ SAST Risk (Low): Potential file existence check race condition.
        end: datetime | str,
    ) -> dict[datetime, list[str]]:
        # ⚠️ SAST Risk (Low): File is opened without exception handling.
        # ⚠️ SAST Risk (Low): json.load can raise exceptions if the file content is not valid JSON.
        """Load index component data as DataFrame"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        start = to_datetime(start)
        end = to_datetime(end)

        # 🧠 ML Signal: Pattern of updating a dictionary with new data.
        with shelve.open(str(file_path)) as db:
            keys: list[str] = list(db.keys())
            keys.sort()

            index_components: dict[datetime, list[str]] = {}
            for key in keys:
                dt: datetime = datetime.strptime(key, "%Y-%m-%d")
                # ✅ Best Practice: Method name contains a typo, should be 'load_contract_settings'
                # ⚠️ SAST Risk (Low): File is opened without exception handling.
                # ⚠️ SAST Risk (Low): json.dump can raise exceptions if the data is not serializable.
                if start <= dt <= end:
                    index_components[dt] = db[key]

            return index_components

    # ✅ Best Practice: Check if the path exists before opening the file

    def load_component_symbols(
        # ⚠️ SAST Risk (Low): No error handling for file operations
        self,
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability
        index_symbol: str,
        # ⚠️ SAST Risk (Low): No error handling for JSON parsing
        start: datetime | str,
        end: datetime | str,
        # ✅ Best Practice: Using Path.joinpath for file path operations improves code readability and cross-platform compatibility
    ) -> list[str]:
        """Collect index component symbols"""
        # ⚠️ SAST Risk (Low): Ensure that the file path is validated or sanitized to prevent path traversal vulnerabilities
        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            # ✅ Best Practice: Use of type hinting for function parameters and return type improves code readability and maintainability.
            # ⚠️ SAST Risk (Low): Ensure that the dataset object is trusted or sanitized to prevent pickle injection vulnerabilities
            start,
            end,
            # ⚠️ SAST Risk (Low): Potential path traversal if 'name' is not properly validated or sanitized.
        )

        # 🧠 ML Signal: Logging error messages can be used to train models to recognize error patterns.
        component_symbols: set[str] = set()

        for vt_symbols in index_components.values():
            component_symbols.update(vt_symbols)
        # ⚠️ SAST Risk (Medium): Unpickling data from a file can lead to arbitrary code execution if the file is tampered with.

        # ✅ Best Practice: Use of type hint for file_path improves code readability and maintainability
        return list(component_symbols)

    # ⚠️ SAST Risk (Low): Potential for path traversal if 'name' is not properly validated
    def load_component_filters(
        self,
        # 🧠 ML Signal: Logging error messages can be used to identify common failure points
        index_symbol: str,
        start: datetime | str,
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        end: datetime | str,
        # ⚠️ SAST Risk (Low): Directly deleting files without backup or confirmation
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Collect index component duration filters"""
        # 🧠 ML Signal: Use of list comprehension to process file paths
        index_components: dict[datetime, list[str]] = self.load_component_data(
            # 🧠 ML Signal: Use of pathlib for file operations
            index_symbol,
            # ✅ Best Practice: Use of type annotations for function parameters and return type improves code readability and maintainability.
            start,
            end,
            # ✅ Best Practice: Using Path.joinpath is preferred over string concatenation for file paths.
        )

        # ⚠️ SAST Risk (Medium): Pickle is not secure against erroneous or maliciously constructed data. Ensure the source of the model is trusted.
        # Get all trading dates and sort
        # ✅ Best Practice: Use of type hinting for the variable 'file_path' improves code readability and maintainability.
        trading_dates: list[datetime] = sorted(index_components.keys())

        # Initialize component duration dictionary
        # 🧠 ML Signal: Logging an error when a model file does not exist can be used to track model loading issues.
        component_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(
            list
        )

        # Get all component symbols
        # ⚠️ SAST Risk (Medium): Using pickle for loading data can lead to arbitrary code execution if the file is tampered with.
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)
        # 🧠 ML Signal: Function to remove a model file, indicating model lifecycle management

        # Iterate through each component to identify its duration in the index
        # ⚠️ SAST Risk (Low): Potential for path traversal if `name` is not validated
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            # 🧠 ML Signal: Logging an error when a model file does not exist
            period_end: datetime | None = None

            # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability
            # Iterate through each trading day to identify continuous holding periods
            # ⚠️ SAST Risk (Low): Deleting a file without backup or confirmation
            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    # 🧠 ML Signal: Usage of file extensions like .pkl can indicate model serialization/deserialization
                    # ✅ Best Practice: Include a docstring to describe the method's purpose
                    if period_start is None:
                        # ⚠️ SAST Risk (Low): Reliance on file extensions for model identification can be error-prone if files are misnamed
                        period_start = trading_date

                    # ✅ Best Practice: Use type annotations for function parameters and return type
                    period_end = trading_date
                # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability
                else:
                    # 🧠 ML Signal: Usage of file paths and file operations
                    if period_start and period_end:
                        # ⚠️ SAST Risk (Low): Potential risk of path traversal if 'name' is not validated
                        component_filters[vt_symbol].append((period_start, period_end))
                        # ✅ Best Practice: Using Path.joinpath for file path construction is more readable and less error-prone than string concatenation
                        period_start = None
                        # 🧠 ML Signal: Usage of specific file format (parquet) for data storage
                        period_end = None
            # ⚠️ SAST Risk (Low): Logging file paths can expose sensitive information in logs

            # Handle the last holding period
            if period_start and period_end:
                component_filters[vt_symbol].append((period_start, period_end))
        # 🧠 ML Signal: Usage of pl.read_parquet indicates interaction with parquet files, which can be a feature for ML models
        # 🧠 ML Signal: Use of file path operations to manage resources

        return component_filters

    # ⚠️ SAST Risk (Low): Potential for path traversal if 'name' is not validated

    def add_contract_setting(
        # 🧠 ML Signal: Logging error messages for non-existent files
        self,
        vt_symbol: str,
        # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
        long_rate: float,
        # ⚠️ SAST Risk (Low): Deleting files without backup or confirmation
        short_rate: float,
        # 🧠 ML Signal: Usage of file extensions to filter files can indicate data processing patterns
        size: float,
        pricetick: float,
    ) -> None:
        """Add contract information"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick,
        }

        with open(self.contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)

    def load_contract_setttings(self) -> dict:
        """Load contract settings"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        return contracts

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """Save dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(dataset, f)

    def load_dataset(self, name: str) -> AlphaDataset | None:
        """Load dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            dataset: AlphaDataset = pickle.load(f)
            return dataset

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_datasets(self) -> list[str]:
        """List all datasets"""
        return [file.stem for file in self.dataset_path.glob("*.pkl")]

    def save_model(self, name: str, model: AlphaModel) -> None:
        """Save model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str) -> AlphaModel | None:
        """Load model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            model: AlphaModel = pickle.load(f)
            return model

    def remove_model(self, name: str) -> bool:
        """Remove model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all models"""
        return [file.stem for file in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame) -> None:
        """Save signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")

        signal.write_parquet(file_path)

    def load_signal(self, name: str) -> pl.DataFrame | None:
        """Load signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None

        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Remove signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signals"""
        return [file.stem for file in self.model_path.glob("*.parquet")]
