# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import shutil
import traceback
from pathlib import Path
from typing import Iterable, List, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import fire
# 🧠 ML Signal: Importing utility functions for encoding and decoding filenames
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from qlib.utils import fname_to_code, code_to_fname


class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    # 🧠 ML Signal: Constants and configuration patterns can be used to identify application settings and defaults
    # ✅ Best Practice: Use of class-level constants for configuration improves maintainability and readability
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        # ✅ Best Practice: Use tuple and filter to ensure fields are stripped and non-empty
        csv_path = Path(csv_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            # ⚠️ SAST Risk (Low): Potentially large number of files could be loaded into memory
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        # ✅ Best Practice: Limit the number of CSV files if limit_nums is specified
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            # ⚠️ SAST Risk (Low): Backup operation could overwrite existing data
            self.csv_files = self.csv_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            # 🧠 ML Signal: Conditional logic based on frequency could indicate different processing paths
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        # ⚠️ SAST Risk (Medium): Using shutil.copytree without exception handling can lead to unhandled exceptions if the source or target directories are invalid or inaccessible.
        self.freq = freq
        # ✅ Best Practice: Consider adding exception handling to manage potential errors during the copy process.
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT
        # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability

        # 🧠 ML Signal: Usage of shutil.copytree indicates a pattern of directory duplication, which can be a feature for ML models to learn about file operations.
        self.works = max_workers
        # ✅ Best Practice: Converting input to a consistent type at the start of the function
        self.date_field_name = date_field_name
        # 🧠 ML Signal: Usage of strftime for date formatting

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        # ✅ Best Practice: Check if input is a DataFrame to handle different input types
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)
        # 🧠 ML Signal: Usage of a helper function to retrieve data

        self._calendars_list = []

        self._mode = self.ALL_MODE
        # ✅ Best Practice: Check for empty DataFrame or missing column to prevent errors
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        # ✅ Best Practice: Return different data structures based on flags for flexibility
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    # ✅ Best Practice: Use of type hint for the return type improves code readability and maintainability
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            # ⚠️ SAST Risk (Low): Using `low_memory=False` can lead to high memory usage with large files
            df = self._get_source_data(file_or_df)
        else:
            # 🧠 ML Signal: Conversion of date fields to datetime format is a common data preprocessing step
            # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            # 🧠 ML Signal: Usage of string manipulation methods like strip() and lower()
            _calendars = pd.Series(dtype=np.float32)
        # ✅ Best Practice: Using Path object for file paths is more robust than using strings
        # ✅ Best Practice: Use of ternary conditional operator for concise conditional logic
        # 🧠 ML Signal: Usage of slicing to manipulate file names
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            # ✅ Best Practice: Use of type hinting for function return type improves code readability and maintainability
            return _calendars.min(), _calendars.max()
        # ✅ Best Practice: Use of sorted() to ensure the list of timestamps is ordered
        # 🧠 ML Signal: Use of map() function to apply pd.Timestamp to each element in the list
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        # ⚠️ SAST Risk (Low): Assumes the CSV file is well-formed and does not handle potential exceptions
        # ✅ Best Practice: Use of type hinting for function return type improves code readability and maintainability
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        # 🧠 ML Signal: Use of pandas to read CSV files is a common pattern in data processing tasks
        df[self.date_field_name] = df[self.date_field_name].astype(str).astype("datetime64[ns]")
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            # ✅ Best Practice: Returning the DataFrame directly is clear and concise
            # ✅ Best Practice: Ensure the directory exists before saving files
            else set(df_columns) - set(self._exclude_fields) if self._exclude_fields else df_columns
        )
    # 🧠 ML Signal: Usage pattern of constructing file paths

    @staticmethod
    # 🧠 ML Signal: List comprehension for data transformation
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        # ✅ Best Practice: Ensure the directory exists before saving files to avoid errors.
        return sorted(
            # ⚠️ SAST Risk (Low): Ensure that the data being saved is properly sanitized to prevent injection attacks
            map(
                # 🧠 ML Signal: Usage of numpy to save text files
                # ✅ Best Practice: Use resolve() to get the absolute path, which helps in debugging and file management.
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            # 🧠 ML Signal: Checking the type of data can indicate different processing paths, useful for ML models.
            )
        # 🧠 ML Signal: Selecting specific fields from a DataFrame can indicate feature selection.
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        # 🧠 ML Signal: Applying transformations to data fields can be a signal for data preprocessing.
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                # ✅ Best Practice: Use descriptive variable names for clarity
                self.symbol_field_name,
                # ✅ Best Practice: Use to_csv with explicit parameters for clarity and to avoid default behavior.
                self.INSTRUMENTS_START_FIELD,
                # ⚠️ SAST Risk (Low): Ensure that instruments_data is sanitized to prevent injection attacks.
                # ✅ Best Practice: Explicitly specify data types for consistency and clarity
                # ✅ Best Practice: Use parentheses for clarity in complex expressions
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        # ✅ Best Practice: Use inplace=True to modify the DataFrame in place and save memory
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        # 🧠 ML Signal: Function definition with specific input types and return type
        # ✅ Best Practice: Use inplace=True to modify the DataFrame in place and save memory
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")
    # 🧠 ML Signal: Reindexing DataFrame based on another DataFrame's index
    # ⚠️ SAST Risk (Low): Potential ValueError if df.index.min() is not in calendar_list

    # 🧠 ML Signal: Use of DataFrame index and list operations
    # ⚠️ SAST Risk (Low): Logging potentially sensitive information (features_dir.name)
    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            # ⚠️ SAST Risk (Low): Logging potentially sensitive information (calendar_list)
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        # ⚠️ SAST Risk (Low): Logging potentially sensitive information (features_dir.name)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # 🧠 ML Signal: Iterating over fields to process data
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        # ✅ Best Practice: Use Path.joinpath for better readability and compatibility
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype("datetime64[ns]")
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        # 🧠 ML Signal: Checking file existence and mode for conditional processing
        ]
        # ⚠️ SAST Risk (Low): Logging warning without additional context may not be sufficient for debugging
        # align index
        # ⚠️ SAST Risk (Low): Opening files in append mode without validation
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        # 🧠 ML Signal: Converting data to binary format
        r_df = df.reindex(cal_df.index)
        # ✅ Best Practice: Use isinstance for type checking
        return r_df

    # 🧠 ML Signal: Converting and saving data to binary format
    # ⚠️ SAST Risk (Low): Returning without logging or error may lead to silent failures
    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())
    # 🧠 ML Signal: Extracting code from DataFrame for further processing

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            # 🧠 ML Signal: Extracting code from file path for further processing
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # ⚠️ SAST Risk (Low): Raising generic ValueError without specific error handling
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        # ⚠️ SAST Risk (Low): Logging warning without additional context may not be sufficient for debugging
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            # ✅ Best Practice: Method raises NotImplementedError to indicate it should be overridden in subclasses
            return
        # used when creating a bin file
        # ✅ Best Practice: Provides a clear error message for unimplemented method
        # ✅ Best Practice: Dropping duplicates to ensure data integrity
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the __call__ method.
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            # 🧠 ML Signal: Creating directory structure based on code
            # 🧠 ML Signal: Method invocation pattern for callable objects.
            # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            # ⚠️ SAST Risk (Low): Ensure that the dump method does not expose sensitive information.
            if field not in _df.columns:
                # ✅ Best Practice: Using mkdir with exist_ok=True to avoid exceptions if directory exists
                continue
            # 🧠 ML Signal: Logging the start of a process can be used to identify function entry points.
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # 🧠 ML Signal: Converting data to binary format for storage
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            # ✅ Best Practice: Using tqdm for progress indication improves user experience in long-running tasks.
            else:
                # ✅ Best Practice: Using ProcessPoolExecutor for parallel processing can improve performance.
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        # ✅ Best Practice: Use logging to track the start of a process for better debugging and monitoring.
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        # 🧠 ML Signal: Logging the end of a process can be used to identify function exit points.
        # 🧠 ML Signal: Usage of sorted and map functions indicates data transformation patterns.
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            # 🧠 ML Signal: Method call pattern for saving data, useful for understanding data persistence behavior.
            # ✅ Best Practice: Method name suggests it's a private method; consider using a single underscore for convention.
            return

        # 🧠 ML Signal: Logging usage pattern can be used to identify logging practices.
        # ✅ Best Practice: Use logging to track the end of a process for better debugging and monitoring.
        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)
        # 🧠 ML Signal: Method call with self indicates instance method usage pattern.

        # ✅ Best Practice: Use of logging for tracking the execution flow and debugging
        # features save dir
        # 🧠 ML Signal: Logging usage pattern can be used to identify logging practices.
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        # 🧠 ML Signal: Use of partial function to pre-fill arguments for another function
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)
    # ✅ Best Practice: Use of tqdm for progress tracking in loops

    @abc.abstractmethod
    # ✅ Best Practice: Use of ProcessPoolExecutor for parallel processing
    def dump(self):
        raise NotImplementedError("dump not implemented!")
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the method.
    # 🧠 ML Signal: Use of executor.map for parallel execution of a function over a list

    def __call__(self, *args, **kwargs):
        # ✅ Best Practice: Updating progress bar inside the loop
        # ✅ Best Practice: Ensure that the method name '_get_all_date' accurately reflects its functionality.
        self.dump()

# ✅ Best Practice: Use of logging for tracking the execution flow and debugging
# ✅ Best Practice: Ensure that the method name '_dump_calendars' accurately reflects its functionality.

# ✅ Best Practice: Class names should follow the CapWords convention for readability.
class DumpDataAll(DumpDataBase):
    # ✅ Best Practice: Ensure that the method name '_dump_instruments' accurately reflects its functionality.
    def _get_all_date(self):
        # 🧠 ML Signal: Logging at the start of a function indicates a common pattern for tracking execution flow.
        logger.info("start get all date......")
        # ✅ Best Practice: Using partial functions can improve code readability and reusability.
        # ✅ Best Practice: Ensure that the method name '_dump_features' accurately reflects its functionality.
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.csv_files, executor.map(_fun, self.csv_files)
                ):
                    all_datetime = all_datetime | _set_calendars
                    # ✅ Best Practice: Using sorted and filter together is a common pattern for processing collections.
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        _begin_time = self._format_datetime(_begin_time)
                        # ✅ Best Practice: Using tqdm for progress indication is a good practice for long-running operations.
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        # ⚠️ SAST Risk (Low): Ensure that the number of workers is controlled to prevent resource exhaustion.
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        # ✅ Best Practice: Type checking ensures that the variables are of expected types.
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        # 🧠 ML Signal: Method name 'dump' suggests data serialization or export operation
        logger.info("start dump calendars......")
        # ✅ Best Practice: Using from_dict with orient="index" is a common pattern for DataFrame creation.
        # ✅ Best Practice: Use of joinpath for file path construction improves readability and OS compatibility
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        # 🧠 ML Signal: Saving data to a persistent storage is a common pattern for data processing tasks.
        # ✅ Best Practice: Chaining methods for concise and readable data transformation
        logger.info("start dump instruments......")
        # 🧠 ML Signal: Logging at the end of a function indicates a common pattern for tracking execution flow.
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")
    # ✅ Best Practice: Class should have a docstring explaining its purpose and usage

    def _dump_features(self):
        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        # ✅ Best Practice: Docstring provides clear parameter descriptions and default values.
        self._dump_instruments()
        self._dump_features()


class DumpDataFix(DumpDataAll):
    def _dump_instruments(self):
        logger.info("start dump instruments......")
        _fun = partial(self._get_date, is_begin_end=True)
        new_stock_files = sorted(
            filter(
                lambda x: fname_to_code(x.name[: -len(self.file_suffix)].strip().lower()).upper()
                not in self._old_instruments,
                self.csv_files,
            )
        )
        with tqdm(total=len(new_stock_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as execute:
                for file_path, (_begin_time, _end_time) in zip(new_stock_files, execute.map(_fun, new_stock_files)):
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        symbol = fname_to_code(self.get_symbol_from_file(file_path).lower()).upper()
                        _dt_map = self._old_instruments.setdefault(symbol, dict())
                        _dt_map[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_begin_time)
                        _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end_time)
                    p_bar.update()
        _inst_df = pd.DataFrame.from_dict(self._old_instruments, orient="index")
        _inst_df.index.names = [self.symbol_field_name]
        # 🧠 ML Signal: Use of super() indicates inheritance, which is a common pattern in OOP.
        self.save_instruments(_inst_df.reset_index())
        logger.info("end of instruments dump.\n")

    def dump(self):
        self._calendars_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # noinspection PyAttributeOutsideInit
        self._old_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict
        self._dump_instruments()
        self._dump_features()

# 🧠 ML Signal: Use of self indicates instance variable assignment, common in class methods.
# ✅ Best Practice: Use of joinpath for path operations improves readability and cross-platform compatibility.

class DumpDataUpdate(DumpDataBase):
    def __init__(
        self,
        csv_path: str,
        # ✅ Best Practice: Chaining methods like set_index and to_dict improves readability.
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        # 🧠 ML Signal: Logging usage pattern
        # 🧠 ML Signal: Loading all source data at once could indicate a pattern for batch processing.
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        # ✅ Best Practice: Initialize variables before use
        # ⚠️ SAST Risk (Low): Potential risk of large memory usage if _all_data is very large.
        exclude_fields: str = "",
        # ⚠️ SAST Risk (Low): Potential for file path traversal if file_path is not validated
        include_fields: str = "",
        # ✅ Best Practice: Use of lambda for inline filtering improves code conciseness.
        limit_nums: int = None,
    # ⚠️ SAST Risk (Low): Assumes self.date_field_name is a valid column name, potential KeyError
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        super().__init__(
            csv_path,
            qlib_dir,
            backup_dir,
            # 🧠 ML Signal: Filtering and sorting data
            freq,
            max_workers,
            date_field_name,
            file_suffix,
            symbol_field_name,
            exclude_fields,
            # 🧠 ML Signal: Updating data structures with new information
            include_fields,
        )
        # ✅ Best Practice: Submitting tasks to executor for parallel execution
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # NOTE: all.txt only exists once for each stock
        # NOTE: if a stock corresponds to multiple different time ranges, user need to modify self._update_instruments
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            # ✅ Best Practice: Submitting tasks to executor for parallel execution
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        # ✅ Best Practice: Use of tqdm for progress tracking
        )  # type: dict

        # 🧠 ML Signal: Method that involves saving or dumping data, indicating a data persistence pattern
        # load all csv files
        self._all_data = self._load_all_source_data()  # type: pd.DataFrame
        # 🧠 ML Signal: Method call that suggests feature processing or transformation
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(lambda x: x > self._old_calendar_list[-1], self._all_data[self.date_field_name].unique())
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
        # 🧠 ML Signal: DataFrame creation from a dictionary, indicating data manipulation
        )

    # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if input is not controlled
    # 🧠 ML Signal: Setting index names, indicating data labeling or organization
    # 🧠 ML Signal: Saving a DataFrame, indicating data persistence
    # 🧠 ML Signal: Command-line interface pattern using fire.Fire
    def _load_all_source_data(self):
        # NOTE: Need more memory
        logger.info("start load all source data....")
        all_df = []

        def _read_csv(file_path: Path):
            _df = pd.read_csv(file_path, parse_dates=[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        with tqdm(total=len(self.csv_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for df in executor.map(_read_csv, self.csv_files):
                    if not df.empty:
                        all_df.append(df)
                    p_bar.update()

        logger.info("end of load all data.\n")
        return pd.concat(all_df, sort=False)

    def _dump_calendars(self):
        pass

    def _dump_instruments(self):
        pass

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(self.symbol_field_name, group_keys=False):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue
                if _code in self._update_instruments:
                    # exists stock, will append data
                    _update_calendars = (
                        _df[_df[self.date_field_name] > self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]][
                            self.date_field_name
                        ]
                        .sort_values()
                        .to_list()
                    )
                    if _update_calendars:
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                        futures[executor.submit(self._dump_bin, _df, _update_calendars)] = _code
                else:
                    # new stock
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self._new_calendar_list)] = _code

            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"dump bin errors: {error_code}")

        logger.info("end of features dump.\n")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())


if __name__ == "__main__":
    fire.Fire({"dump_all": DumpDataAll, "dump_fix": DumpDataFix, "dump_update": DumpDataUpdate})