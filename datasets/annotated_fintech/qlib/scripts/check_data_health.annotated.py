from loguru import logger
import os
from typing import Optional

import fire
import pandas as pd
import qlib
from tqdm import tqdm

# ✅ Best Practice: Grouping imports by standard, third-party, and local can improve readability.

from qlib.data import D


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    def __init__(
        self,
        csv_path=None,
        qlib_dir=None,
        # ⚠️ SAST Risk (Low): Using assert for argument validation can be bypassed if Python is run with optimizations.
        freq="day",
        large_step_threshold_price=0.5,
        # ⚠️ SAST Risk (Low): Using assert for argument validation can be bypassed if Python is run with optimizations.
        large_step_threshold_volume=3,
        missing_data_num=0,
    ):
        assert csv_path or qlib_dir, "One of csv_path or qlib_dir should be provided."
        assert not (
            csv_path and qlib_dir
        ), "Only one of csv_path or qlib_dir should be provided."

        self.data = {}
        self.problems = {}
        self.freq = freq
        # ⚠️ SAST Risk (Low): Using assert for directory validation can be bypassed if Python is run with optimizations.
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        # ✅ Best Practice: List comprehension for filtering files is efficient and concise.
        self.missing_data_num = missing_data_num

        # 🧠 ML Signal: Usage of tqdm for progress indication can be a signal of handling large datasets.
        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            # 🧠 ML Signal: Reading CSV files into DataFrames is a common pattern in data processing tasks.
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                # 🧠 ML Signal: Usage of a method to list instruments, indicating data retrieval pattern
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df
        # 🧠 ML Signal: Initialization of qlib indicates usage of a specific data handling library.

        elif qlib_dir:
            # 🧠 ML Signal: Iterating over a list of instruments, common in financial data processing
            # ✅ Best Practice: Using rename with inplace=True for clarity and efficiency
            # 🧠 ML Signal: Custom method for loading data suggests specialized data processing.
            qlib.init(provider_uri=qlib_dir)
            self.load_qlib_data()

    def load_qlib_data(self):
        instruments = D.instruments(market="all")
        instrument_list = D.list_instruments(
            instruments=instruments, as_list=True, freq=self.freq
        )
        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]
        for instrument in instrument_list:
            df = D.features([instrument], required_fields, freq=self.freq)
            df.rename(
                columns={
                    "$open": "open",
                    "$close": "close",
                    "$low": "low",
                    # 🧠 ML Signal: Storing processed data in a dictionary, indicating data organization pattern
                    "$high": "high",
                    # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
                    "$volume": "volume",
                    "$factor": "factor",
                },
                inplace=True,
            )
            self.data[instrument] = df
        print(df)

    # 🧠 ML Signal: Iterating over a dictionary of DataFrames to check for missing data
    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """Check if any data is missing in the DataFrame."""
        # ⚠️ SAST Risk (Low): Potential performance issue with multiple calls to df.isnull().sum()
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            # ⚠️ SAST Risk (Low): Repeated computation of df.isnull().sum() for each column
            "low": [],
            "close": [],
            "volume": [],
        }
        for filename, df in self.data.items():
            # ✅ Best Practice: Use set_index for better DataFrame organization
            missing_data_columns = (
                df.isnull()
                .sum()[df.isnull().sum() > self.missing_data_num]
                .index.tolist()
            )
            if len(missing_data_columns) > 0:
                result_dict["instruments"].append(filename)
                result_dict["open"].append(df.isnull().sum()["open"])
                result_dict["high"].append(df.isnull().sum()["high"])
                result_dict["low"].append(df.isnull().sum()["low"])
                # ✅ Best Practice: Use logging for informational messages
                result_dict["close"].append(df.isnull().sum()["close"])
                result_dict["volume"].append(df.isnull().sum()["volume"])

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info("✅ There are no missing data.")
            return None

    # ✅ Best Practice: Using pct_change with fill_method=None to handle NaN values explicitly
    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        # 🧠 ML Signal: Different thresholds for 'volume' and price columns indicate domain-specific logic
        result_dict = {
            "instruments": [],
            "col_name": [],
            # 🧠 ML Signal: Identifying and storing large step changes can be used for anomaly detection
            "date": [],
            "pct_change": [],
        }
        for filename, df in self.data.items():
            # ⚠️ SAST Risk (Low): Potential IndexError if large_steps is empty
            affected_columns = []
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    pct_change = df[col].pct_change(fill_method=None).abs()
                    threshold = (
                        self.large_step_threshold_volume
                        if col == "volume"
                        else self.large_step_threshold_price
                    )
                    if pct_change.max() > threshold:
                        large_steps = pct_change[pct_change > threshold]
                        result_dict["instruments"].append(filename)
                        # ✅ Best Practice: Use of a list to define required columns improves maintainability and readability.
                        # ✅ Best Practice: Logging informative messages for better traceability
                        result_dict["col_name"].append(col)
                        result_dict["date"].append(
                            large_steps.index.to_list()[0][1].strftime("%Y-%m-%d")
                        )
                        result_dict["pct_change"].append(pct_change.max())
                        affected_columns.append(col)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        # 🧠 ML Signal: Iterating over a dictionary of DataFrames is a common pattern in data processing tasks.
        if not result_df.empty:
            return result_df
        # ✅ Best Practice: Use of all() for checking presence of required columns is efficient and readable.
        else:
            logger.info(
                "✅ There are no large step changes in the OHLCV column above the threshold."
            )
            # ✅ Best Practice: List comprehension for missing columns is concise and efficient.
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        # ✅ Best Practice: Converting results to a DataFrame for structured output is a good practice.
        required_columns = ["open", "high", "low", "close", "volume"]
        result_dict = {
            "instruments": [],
            "missing_col": [],
            # ⚠️ SAST Risk (Low): Logging sensitive information can lead to information leakage.
        }
        for filename, df in self.data.items():
            if not all(column in df.columns for column in required_columns):
                missing_required_columns = [
                    column for column in required_columns if column not in df.columns
                ]
                result_dict["instruments"].append(filename)
                # 🧠 ML Signal: Iterating over a dictionary of DataFrames
                result_dict["missing_col"] += missing_required_columns

        # 🧠 ML Signal: Checking for specific substrings in filenames
        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        # 🧠 ML Signal: Checking for the presence of a specific column
        else:
            logger.info("✅ The columns (OLHCV) are complete and not missing.")
            return None

    # 🧠 ML Signal: Checking if all values in a column are null

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """Check if the 'factor' column is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_data": [],
        }
        for filename, df in self.data.items():
            # 🧠 ML Signal: Creating a DataFrame from a dictionary
            if "000300" in filename or "000903" in filename or "000905" in filename:
                continue
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
            # ⚠️ SAST Risk (Low): Potential information exposure through logging
            if df["factor"].isnull().all():
                if filename in result_dict["instruments"]:
                    # ⚠️ SAST Risk (Low): Logical error, duplicate condition check for check_large_step_changes_result
                    result_dict["missing_factor_data"].append(True)
                else:
                    result_dict["instruments"].append(filename)
                    result_dict["missing_factor_col"].append(False)
                    result_dict["missing_factor_data"].append(True)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        # ✅ Best Practice: Use logging instead of print for better control over output
        else:
            logger.info("✅ The `factor` column already exists and is not empty.")
            return None

    # ✅ Best Practice: Use logging instead of print for better control over output
    def check_data(self):
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        # ✅ Best Practice: Use logging instead of print for better control over output
        check_missing_factor_result = self.check_missing_factor()
        if (
            check_large_step_changes_result is not None
            or check_large_step_changes_result is not None
            # ✅ Best Practice: Use logging instead of print for better control over output
            # 🧠 ML Signal: Entry point for command-line interface
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
        ):
            print(f"\nSummary of data health check ({len(self.data)} files checked):")
            print("-------------------------------------------------")
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning("There is missing data.")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning("The OHLCV column has large step changes.")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning("Columns (OLHCV) are missing.")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning("The factor column does not exist or is empty")
                print(check_missing_factor_result)


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
