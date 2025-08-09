# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TODO:
- A more well-designed PIT database is required.
    - separated insert, delete, update, query operations are required.
"""

import shutil
import struct
from pathlib import Path
# ✅ Best Practice: Use of fire library for creating CLIs easily
from typing import Iterable
from functools import partial
# ✅ Best Practice: Use of pandas for data manipulation
from concurrent.futures import ProcessPoolExecutor

# ✅ Best Practice: Use of tqdm for progress bar in loops
import fire
import pandas as pd
# ✅ Best Practice: Use of loguru for logging
# 🧠 ML Signal: Class definition for data handling, indicating a pattern for data processing
from tqdm import tqdm
from loguru import logger
# 🧠 ML Signal: Importing specific functions from a module
# 🧠 ML Signal: Constants for directory and file naming conventions
from qlib.utils import fname_to_code, get_period_offset
from qlib.config import C
# 🧠 ML Signal: Importing specific configuration from a module
# 🧠 ML Signal: Constants for CSV file handling


# 🧠 ML Signal: Constants for file suffixes
class DumpPitData:
    PIT_DIR_NAME = "financial"
    PIT_CSV_SEP = ","
    # 🧠 ML Signal: Constants for data intervals
    # 🧠 ML Signal: Usage of external configuration for data types
    DATA_FILE_SUFFIX = ".data"
    INDEX_FILE_SUFFIX = ".index"

    INTERVAL_quarterly = "quarterly"
    INTERVAL_annual = "annual"

    PERIOD_DTYPE = C.pit_record_type["period"]
    INDEX_DTYPE = C.pit_record_type["index"]
    # 🧠 ML Signal: Concatenation of data types for structured data
    DATA_DTYPE = "".join(
        [
            C.pit_record_type["date"],
            C.pit_record_type["period"],
            C.pit_record_type["value"],
            C.pit_record_type["index"],
        # 🧠 ML Signal: Handling of missing data with a specific constant
        # 🧠 ML Signal: Calculation of data type sizes for structured data
        # 🧠 ML Signal: Constants for operational modes
        ]
    )

    NA_INDEX = C.pit_record_nan["index"]

    INDEX_DTYPE_SIZE = struct.calcsize(INDEX_DTYPE)
    PERIOD_DTYPE_SIZE = struct.calcsize(PERIOD_DTYPE)
    DATA_DTYPE_SIZE = struct.calcsize(DATA_DTYPE)

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "quarterly",
        max_workers: int = 16,
        date_column_name: str = "date",
        period_column_name: str = "period",
        value_column_name: str = "value",
        field_column_name: str = "field",
        file_suffix: str = ".csv",
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
        freq: str, default "quarterly"
            data frequency
        max_workers: int, default None
            number of threads
        date_column_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        # 🧠 ML Signal: Usage of sorted and glob to list files in a directory
        """
        csv_path = Path(csv_path).expanduser()
        # ✅ Best Practice: Check if limit_nums is not None before slicing
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        # ✅ Best Practice: Use Path.expanduser for consistent path handling
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        # ✅ Best Practice: Check if backup_dir is not None before calling backup method
        self.file_suffix = file_suffix
        # ⚠️ SAST Risk (Medium): Using shutil.copytree without exception handling can lead to unhandled exceptions if the source or target directories are invalid or inaccessible.
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        # ✅ Best Practice: Consider adding exception handling to manage potential errors during the copy process.
        if limit_nums is not None:
            # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
            self.csv_files = self.csv_files[: int(limit_nums)]
        # 🧠 ML Signal: Usage of shutil.copytree indicates a pattern of directory duplication, which can be a feature for ML models to learn about file operations.
        self.qlib_dir = Path(qlib_dir).expanduser()
        # 🧠 ML Signal: Usage of pandas to read CSV files is a common pattern in data processing tasks
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        # ⚠️ SAST Risk (Low): Potential for large memory usage if the CSV file is very large
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())
        # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
        # 🧠 ML Signal: Converting data types to optimize memory usage is a common pattern

        self.works = max_workers
        # ⚠️ SAST Risk (Low): Assumes date_column_name exists and is in the expected format
        # 🧠 ML Signal: Usage of string slicing and manipulation to extract information from a file name
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        self.date_column_name = date_column_name
        # 🧠 ML Signal: Use of a method to convert file names to a specific code format
        # ✅ Best Practice: Using set operations for efficient field inclusion/exclusion
        # 🧠 ML Signal: Conditional logic based on instance attributes
        self.period_column_name = period_column_name
        self.value_column_name = value_column_name
        self.field_column_name = field_column_name

        self._mode = self.ALL_MODE

    def _backup_qlib_dir(self, target_dir: Path):
        # 🧠 ML Signal: Use of set operations to filter data
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    # 🧠 ML Signal: Conditional logic based on instance attributes
    def get_source_data(self, file_path: Path) -> pd.DataFrame:
        # 🧠 ML Signal: Function signature and parameter names can be used to infer the purpose and usage of the function.
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        df[self.value_column_name] = df[self.value_column_name].astype("float32")
        # ✅ Best Practice: Using pathlib for path operations improves code readability and maintainability.
        # ⚠️ SAST Risk (Low): Directory creation with user-controlled input could lead to directory traversal if not properly validated.
        df[self.date_column_name] = df[self.date_column_name].str.replace("-", "").astype("int32")
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    # 🧠 ML Signal: The pattern of constructing file paths can be used to learn about file naming conventions.
    # ✅ Best Practice: Using f-strings for string formatting is more readable and efficient.
    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df: Iterable[str]) -> Iterable[str]:
        return (
            set(self._include_fields)
            if self._include_fields
            else (
                set(df[self.field_column_name]) - set(self._exclude_fields)
                if self._exclude_fields
                else set(df[self.field_column_name])
            )
        )

    def get_filenames(self, symbol, field, interval):
        dir_name = self.qlib_dir.joinpath(self.PIT_DIR_NAME, symbol)
        dir_name.mkdir(parents=True, exist_ok=True)
        return (
            dir_name.joinpath(f"{field}_{interval[0]}{self.DATA_FILE_SUFFIX}".lower()),
            dir_name.joinpath(f"{field}_{interval[0]}{self.INDEX_FILE_SUFFIX}".lower()),
        )

    def _dump_pit(
        self,
        file_path: str,
        interval: str = "quarterly",
        overwrite: bool = False,
    # ✅ Best Practice: Use of descriptive variable names improves code readability.
    ):
        """
        dump data as the following format:
            `/path/to/<field>.data`
                [date, period, value, _next]
                [date, period, value, _next]
                [...]
            `/path/to/<field>.index`
                [first_year, index, index, ...]

        `<field.data>` contains the data as the point-in-time (PIT) order: `value` of `period`
        is published at `date`, and its successive revised value can be found at `_next` (linked list).

        `<field>.index` contains the index of value for each period (quarter or year). To save
        disk space, we only store the `first_year` as its followings periods can be easily infered.

        Parameters
        ----------
        symbol: str
            stock symbol
        interval: str
            data interval
        overwrite: bool
            whether overwrite existing data or update only
        """
        symbol = self.get_symbol_from_file(file_path)
        df = self.get_source_data(file_path)
        # ✅ Best Practice: Use of descriptive variable names improves code readability.
        if df.empty:
            logger.warning(f"{symbol} file is empty")
            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            return
        for field in self.get_dump_fields(df):
            df_sub = df.query(f'{self.field_column_name}=="{field}"').sort_values(self.date_column_name)
            if df_sub.empty:
                logger.warning(f"field {field} of {symbol} is empty")
                continue
            data_file, index_file = self.get_filenames(symbol, field, interval)

            ## calculate first & last period
            start_year = df_sub[self.period_column_name].min()
            # ✅ Best Practice: Logging provides insight into the program's execution flow.
            end_year = df_sub[self.period_column_name].max()
            if interval == self.INTERVAL_quarterly:
                start_year //= 100
                end_year //= 100

            # adjust `first_year` if existing data found
            if not overwrite and index_file.exists():
                with open(index_file, "rb") as fi:
                    (first_year,) = struct.unpack(self.PERIOD_DTYPE, fi.read(self.PERIOD_DTYPE_SIZE))
                    n_years = len(fi.read()) // self.INDEX_DTYPE_SIZE
                    if interval == self.INTERVAL_quarterly:
                        n_years //= 4
                    # ✅ Best Practice: Use of descriptive variable names improves code readability.
                    start_year = first_year + n_years
            else:
                with open(index_file, "wb") as f:
                    f.write(struct.pack(self.PERIOD_DTYPE, start_year))
                first_year = start_year

            # if data already exists, continue to the next field
            if start_year > end_year:
                # ✅ Best Practice: Use of descriptive variable names improves code readability.
                logger.warning(f"{symbol}-{field} data already exists, continue to the next field")
                continue

            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            # dump index filled with NA
            with open(index_file, "ab") as fi:
                # 🧠 ML Signal: Method parameters with default values indicate common usage patterns.
                for year in range(start_year, end_year + 1):
                    # ✅ Best Practice: Logging the start of a process helps in debugging and tracking execution flow.
                    if interval == self.INTERVAL_quarterly:
                        fi.write(struct.pack(self.INDEX_DTYPE * 4, *[self.NA_INDEX] * 4))
                    # 🧠 ML Signal: Use of partial functions indicates a pattern of function customization.
                    else:
                        # ✅ Best Practice: Use of descriptive variable names improves code readability.
                        fi.write(struct.pack(self.INDEX_DTYPE, self.NA_INDEX))
            # ✅ Best Practice: Using tqdm for progress indication improves user experience in long-running processes.

            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            # if data already exists, remove overlapped data
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of this method.
            # 🧠 ML Signal: Use of ProcessPoolExecutor indicates parallel processing patterns.
            if not overwrite and data_file.exists():
                # ⚠️ SAST Risk (Low): Ensure that the function _dump_pit is thread-safe when used with ProcessPoolExecutor.
                with open(data_file, "rb") as fd:
                    # 🧠 ML Signal: Usage of fire.Fire indicates a command-line interface pattern.
                    # 🧠 ML Signal: Iterating over executor.map shows a pattern of processing items in parallel.
                    # ✅ Best Practice: Updating progress bar inside the loop provides real-time feedback.
                    # 🧠 ML Signal: Method call pattern within a special method like __call__ can indicate dynamic behavior or usage patterns.
                    # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if input is not properly sanitized.
                    fd.seek(-self.DATA_DTYPE_SIZE, 2)
                    last_date, _, _, _ = struct.unpack(self.DATA_DTYPE, fd.read())
                df_sub = df_sub.query(f"{self.date_column_name}>{last_date}")
            # otherwise,
            # 1) truncate existing file or create a new file with `wb+` if overwrite,
            # 2) or append existing file or create a new file with `ab+` if not overwrite
            else:
                with open(data_file, "wb+" if overwrite else "ab+"):
                    pass

            with open(data_file, "rb+") as fd, open(index_file, "rb+") as fi:
                # update index if needed
                for i, row in df_sub.iterrows():
                    # get index
                    offset = get_period_offset(first_year, row.period, interval == self.INTERVAL_quarterly)

                    fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                    (cur_index,) = struct.unpack(self.INDEX_DTYPE, fi.read(self.INDEX_DTYPE_SIZE))

                    # Case I: new data => update `_next` with current index
                    if cur_index == self.NA_INDEX:
                        fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                        fi.write(struct.pack(self.INDEX_DTYPE, fd.tell()))
                    # Case II: previous data exists => find and update the last `_next`
                    else:
                        _cur_fd = fd.tell()
                        prev_index = self.NA_INDEX
                        while cur_index != self.NA_INDEX:  # NOTE: first iter always != NA_INDEX
                            fd.seek(cur_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                            prev_index = cur_index
                            (cur_index,) = struct.unpack(self.INDEX_DTYPE, fd.read(self.INDEX_DTYPE_SIZE))
                        fd.seek(prev_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                        fd.write(struct.pack(self.INDEX_DTYPE, _cur_fd))  # NOTE: add _next pointer
                        fd.seek(_cur_fd)

                    # dump data
                    fd.write(struct.pack(self.DATA_DTYPE, row.date, row.period, row.value, self.NA_INDEX))

    def dump(self, interval="quarterly", overwrite=False):
        logger.info("start dump pit data......")
        _dump_func = partial(self._dump_pit, interval=interval, overwrite=overwrite)

        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

    def __call__(self, *args, **kwargs):
        self.dump()


if __name__ == "__main__":
    fire.Fire(DumpPitData)