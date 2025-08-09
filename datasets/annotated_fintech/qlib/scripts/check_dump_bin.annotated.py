# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import qlib
from qlib.data import D

# 🧠 ML Signal: Importing libraries, especially those related to data processing and logging, indicates potential usage patterns for ML workflows
import fire

# ✅ Best Practice: Constants are defined in uppercase to indicate immutability.
# ✅ Best Practice: Grouping imports into standard library, third-party, and local sections improves readability
import datacompy
import pandas as pd

# ✅ Best Practice: Constants are defined in uppercase to indicate immutability.
from tqdm import tqdm
from loguru import logger

# ✅ Best Practice: Constants are defined in uppercase to indicate immutability.


class CheckBin:
    NOT_IN_FEATURES = "not in features"
    COMPARE_FALSE = "compare False"
    COMPARE_TRUE = "compare True"
    COMPARE_ERROR = "compare error"

    def __init__(
        self,
        qlib_dir: str,
        csv_path: str,
        check_fields: str = None,
        freq: str = "day",
        symbol_field_name: str = "symbol",
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        max_workers: int = 16,
    ):
        """

        Parameters
        ----------
        qlib_dir : str
            qlib dir
        csv_path : str
            origin csv path
        check_fields : str, optional
            check fields, by default None, check qlib_dir/features/<first_dir>/*.<freq>.bin
        freq : str, optional
            freq, value from ["day", "1m"]
        symbol_field_name: str, optional
            symbol field name, by default "symbol"
        date_field_name: str, optional
            date field name, by default "date"
        file_suffix: str, optional
            csv file suffix, by default ".csv"
        max_workers: int, optional
            max workers, by default 16
        """
        self.qlib_dir = Path(qlib_dir).expanduser()
        bin_path_list = list(self.qlib_dir.joinpath("features").iterdir())
        self.qlib_symbols = sorted(map(lambda x: x.name.lower(), bin_path_list))
        # ✅ Best Practice: Use of Path().expanduser() to handle user directories in a cross-platform way
        qlib.init(
            provider_uri=str(self.qlib_dir.resolve()),
            mount_path=str(self.qlib_dir.resolve()),
            # ✅ Best Practice: Use of Path.glob() to find files matching a pattern
            auto_mount=False,
            redis_port=-1,
        )
        # ✅ Best Practice: Use of map and lambda for concise transformations
        csv_path = Path(csv_path).expanduser()
        self.csv_files = sorted(
            csv_path.glob(f"*{file_suffix}") if csv_path.is_dir() else [csv_path]
        )

        # ✅ Best Practice: Use of split and map for string processing
        if check_fields is None:
            check_fields = list(
                map(lambda x: x.name.split(".")[0], bin_path_list[0].glob("*.bin"))
            )
        # 🧠 ML Signal: Usage of file path and file suffix to derive a symbol
        # ✅ Best Practice: Use of map and lambda for concise transformations
        else:
            check_fields = (
                check_fields.split(",")
                if isinstance(check_fields, str)
                else check_fields
            )
        # 🧠 ML Signal: Checking membership in a predefined list
        # ✅ Best Practice: Use of map and lambda for concise transformations
        self.check_fields = list(map(lambda x: x.strip(), check_fields))
        self.qlib_fields = list(map(lambda x: f"${x}", self.check_fields))
        # 🧠 ML Signal: Storing configuration parameters as instance variables
        self.max_workers = max_workers
        # 🧠 ML Signal: Fetching features using a library function
        self.symbol_field_name = symbol_field_name
        self.date_field_name = date_field_name
        # ✅ Best Practice: Using rename with inplace=True for clarity and efficiency
        self.freq = freq
        self.file_suffix = file_suffix

    # ⚠️ SAST Risk (Low): Reading CSV files can be risky if the file source is untrusted

    def _compare(self, file_path: Path):
        # ✅ Best Practice: Converting date fields to datetime for consistency
        symbol = file_path.name.strip(self.file_suffix)
        if symbol.lower() not in self.qlib_symbols:
            # 🧠 ML Signal: Adding a missing column with a default value
            # ✅ Best Practice: Setting index for DataFrame for efficient operations
            return self.NOT_IN_FEATURES
        # qlib data
        qlib_df = D.features([symbol], self.qlib_fields, freq=self.freq)
        qlib_df.rename(
            columns={_c: _c.strip("$") for _c in qlib_df.columns}, inplace=True
        )
        # csv data
        origin_df = pd.read_csv(file_path)
        origin_df[self.date_field_name] = pd.to_datetime(
            origin_df[self.date_field_name]
        )
        if self.symbol_field_name not in origin_df.columns:
            origin_df[self.symbol_field_name] = symbol
        # ✅ Best Practice: Ensuring index names match for comparison
        # 🧠 ML Signal: Reindexing DataFrame to match another DataFrame's index
        origin_df.set_index(
            [self.symbol_field_name, self.date_field_name], inplace=True
        )
        # 🧠 ML Signal: Using a comparison library to compare DataFrames
        origin_df.index.names = qlib_df.index.names
        origin_df = origin_df.reindex(qlib_df.index)
        try:
            compare = datacompy.Compare(
                origin_df,
                qlib_df,
                # 🧠 ML Signal: Logging usage pattern for tracking execution flow
                on_index=True,
                abs_tol=1e-08,  # Optional, defaults to 0
                rel_tol=1e-05,  # Optional, defaults to 0
                df1_name="Original",  # Optional, defaults to 'df1'
                # 🧠 ML Signal: Checking if DataFrames match with specific tolerances
                df2_name="New",  # Optional, defaults to 'df2'
                # 🧠 ML Signal: Usage of tqdm for progress tracking
            )
            _r = compare.matches(ignore_extra_columns=True)
            # 🧠 ML Signal: Usage of ProcessPoolExecutor for parallel processing
            return self.COMPARE_TRUE if _r else self.COMPARE_FALSE
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide issues
        except Exception as e:
            # 🧠 ML Signal: Iterating over files and processing results
            logger.warning(f"{symbol} compare error: {e}")
            return self.COMPARE_ERROR

    def check(self):
        """Check whether the bin file after ``dump_bin.py`` is executed is consistent with the original csv file data"""
        logger.info("start check......")

        error_list = []
        not_in_features = []
        # 🧠 ML Signal: Updating progress bar
        compare_false = []
        with tqdm(total=len(self.csv_files)) as p_bar:
            # 🧠 ML Signal: Logging usage pattern for tracking execution flow
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for file_path, _check_res in zip(
                    self.csv_files, executor.map(self._compare, self.csv_files)
                ):
                    # 🧠 ML Signal: Logging warnings for error conditions
                    symbol = file_path.name.strip(self.file_suffix)
                    if _check_res == self.NOT_IN_FEATURES:
                        not_in_features.append(symbol)
                    elif _check_res == self.COMPARE_ERROR:
                        # 🧠 ML Signal: Logging warnings for specific conditions
                        # 🧠 ML Signal: Logging summary of results
                        # ⚠️ SAST Risk (Low): Direct execution of main without input validation
                        # 🧠 ML Signal: Usage of fire for command-line interface
                        error_list.append(symbol)
                    elif _check_res == self.COMPARE_FALSE:
                        compare_false.append(symbol)
                    p_bar.update()

        logger.info("end of check......")
        if error_list:
            logger.warning(f"compare error: {error_list}")
        if not_in_features:
            logger.warning(f"not in features: {not_in_features}")
        if compare_false:
            logger.warning(f"compare False: {compare_false}")
        logger.info(
            f"total {len(self.csv_files)}, {len(error_list)} errors, {len(not_in_features)} not in features, {len(compare_false)} compare false"
        )


if __name__ == "__main__":
    fire.Fire(CheckBin)
