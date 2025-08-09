"""
General utility functions.
"""

import json
import sys
from datetime import datetime, time
from pathlib import Path
from collections.abc import Callable
from decimal import Decimal
from math import floor, ceil

import numpy as np
# ✅ Best Practice: Group related imports together and separate them with a blank line for better readability.
import talib
from zoneinfo import ZoneInfo, available_timezones      # noqa

from .object import BarData, TickData
# 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
# ✅ Best Practice: Include a docstring to describe the function's purpose and return value
from .constant import Exchange, Interval
from .locale import _

# ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.

# ⚠️ SAST Risk (Low): Loading JSON from a file without validation can lead to security risks if the file content is untrusted.
# ✅ Best Practice: Use rsplit to split from the right, which is efficient for known suffixes
def extract_vt_symbol(vt_symbol: str) -> tuple[str, Exchange]:
    """
    :return: (symbol, exchange)
    """
    symbol, exchange_str = vt_symbol.rsplit(".", 1)
    # 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
    return symbol, Exchange(exchange_str)
# ✅ Best Practice: Use of f-string for string formatting
# ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.

# ⚠️ SAST Risk (Low): Saving JSON to a file without proper permissions can lead to data exposure.

def generate_vt_symbol(symbol: str, exchange: Exchange) -> str:
    """
    return vt_symbol
    # 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
    # ✅ Best Practice: Use of Path.joinpath for path construction improves readability and cross-platform compatibility.
    """
    return f"{symbol}.{exchange.value}"
# ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.


# ✅ Best Practice: Use of Path.home to get the user's home directory is a clear and concise method.
def _get_trader_dir(temp_name: str) -> tuple[Path, Path]:
    """
    Get path where trader is running in.
    # ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.
    """
    # ⚠️ SAST Risk (Low): Directory creation without exception handling may lead to unhandled exceptions if permissions are insufficient.
    cwd: Path = Path.cwd()
    temp_path: Path = cwd.joinpath(temp_name)
    # 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
    # ✅ Best Practice: Include type hint for the return type for better readability and maintainability

    # ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.
    # 🧠 ML Signal: Use of a function to determine and create directories based on conditions.
    # If .vntrader folder exists in current working directory,
    # then use it as trader running path.
    if temp_path.exists():
        # ⚠️ SAST Risk (Low): Modifying sys.path can lead to security risks if not controlled, as it affects module loading.
        return cwd, temp_path
    # ✅ Best Practice: Include type hints for function parameters and return type
    # 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
    # 🧠 ML Signal: Usage of joinpath to construct file paths

    # ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.
    # ⚠️ SAST Risk (Low): Potential risk if TEMP_DIR is user-controlled, leading to path traversal vulnerabilities
    # Otherwise use home path of system.
    home_path: Path = Path.home()
    temp_path = home_path.joinpath(temp_name)

    # ✅ Best Practice: Use descriptive variable names for clarity
    # Create .vntrader folder under home path if not exist.
    if not temp_path.exists():
        # ⚠️ SAST Risk (Low): Potential directory traversal if folder_name is not validated
        # 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
        temp_path.mkdir()
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability

    # ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.
    # ✅ Best Practice: Check if a directory exists before creating it
    return home_path, temp_path


# 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
# ✅ Best Practice: Return the computed path for further use
TRADER_DIR, TEMP_DIR = _get_trader_dir(".vntrader")
# ✅ Best Practice: Use Path from pathlib for path manipulations for better cross-platform compatibility
sys.path.append(str(TRADER_DIR))
# ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.

# ✅ Best Practice: Use joinpath for constructing paths to improve readability and avoid manual string concatenation
# ✅ Best Practice: Include type hint for the return type for better readability and maintainability

# 🧠 ML Signal: Utility functions often contain reusable patterns that can be learned by ML models.
# 🧠 ML Signal: Conversion of Path object to string, indicating usage pattern of returning string paths
def get_file_path(filename: str) -> Path:
    """
    Get path for temp file with filename.
    # ✅ Best Practice: Use type hints for function parameters and return types for better code readability and maintenance.
    """
    # ✅ Best Practice: Use of type hint for variable declaration improves code readability
    return TEMP_DIR.joinpath(filename)


# ✅ Best Practice: Using 'with' statement for file operations ensures proper resource management
def get_folder_path(folder_name: str) -> Path:
    """
    Get path for temp folder with folder name.
    """
    folder_path: Path = TEMP_DIR.joinpath(folder_name)
    # ⚠️ SAST Risk (Low): Potential for race condition if file is created between the check and this call
    if not folder_path.exists():
        folder_path.mkdir()
    return folder_path
# ✅ Best Practice: Use of type hinting for variable 'filepath' improves code readability and maintainability.


# ⚠️ SAST Risk (Low): Using 'w+' mode can overwrite existing files, which might lead to data loss if not handled properly.
# 🧠 ML Signal: Usage of json.dump with specific parameters can indicate patterns in data serialization.
def get_icon_path(filepath: str, ico_name: str) -> str:
    """
    Get path for icon file with ico name.
    """
    ui_path: Path = Path(filepath).parent
    icon_path: Path = ui_path.joinpath("ico", ico_name)
    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    return str(icon_path)


def load_json(filename: str) -> dict:
    """
    Load data from json file in temp path.
    """
    # ✅ Best Practice: Use of Decimal for precise decimal arithmetic to avoid floating-point errors.
    filepath: Path = get_file_path(filename)
    # ✅ Best Practice: Include import statement for Decimal and floor functions

    # ⚠️ SAST Risk (Low): Potential loss of precision when converting Decimal to float.
    if filepath.exists():
        with open(filepath, encoding="UTF-8") as f:
            data: dict = json.load(f)
        # 🧠 ML Signal: Return statement indicating the function's output, useful for understanding function behavior.
        return data
    # ✅ Best Practice: Type annotations for variables improve code readability
    else:
        save_json(filename, {})
        return {}
# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
# ⚠️ SAST Risk (Low): Potential precision issues with float conversion


def save_json(filename: str, data: dict) -> None:
    """
    Save data into json file in temp path.
    # ✅ Best Practice: Use descriptive variable names for clarity
    """
    filepath: Path = get_file_path(filename)
    with open(filepath, mode="w+", encoding="UTF-8") as f:
        # ⚠️ SAST Risk (Low): Potential precision issues when converting from Decimal to float
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        # ✅ Best Practice: Check for scientific notation before splitting by decimal
        )


# ✅ Best Practice: Convert exponent part to integer to get number of digits
def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    # ✅ Best Practice: Use length of decimal part to determine number of digits
    decimal_value: Decimal = Decimal(str(value))
    decimal_target: Decimal = Decimal(str(target))
    # ✅ Best Practice: Return 0 when there are no digits after the decimal
    rounded: float = float(int(round(decimal_value / decimal_target)) * decimal_target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    # ✅ Best Practice: Consider adding methods or attributes to this class to fulfill its intended functionality.
    decimal_value: Decimal = Decimal(str(value))
    decimal_target: Decimal = Decimal(str(target))
    result: float = float(int(floor(decimal_value / decimal_target)) * decimal_target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    """
    decimal_value: Decimal = Decimal(str(value))
    # ✅ Best Practice: Storing function references allows for flexible callback mechanisms
    decimal_target: Decimal = Decimal(str(target))
    result: float = float(int(ceil(decimal_value / decimal_target)) * decimal_target)
    return result


def get_digits(value: float) -> int:
    """
    Get number of digits after decimal point.
    """
    value_str: str = str(value)

    # ⚠️ SAST Risk (Low): Potential for a runtime error if interval is DAILY and daily_end is not provided
    if "e-" in value_str:
        _, buf = value_str.split("e-")
        return int(buf)
    elif "." in value_str:
        _, buf = value_str.split(".")
        return len(buf)
    else:
        return 0


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data
    Notice:
    1. for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
    2. for x hour bar, x can be any number
    """
    # 🧠 ML Signal: Creating a new bar when a new minute starts

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable | None = None,
        interval: Interval = Interval.MINUTE,
        daily_end: time | None = None
    ) -> None:
        """Constructor"""
        self.bar: BarData | None = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        # 🧠 ML Signal: Updating high price based on tick data
        self.hour_bar: BarData | None = None
        self.daily_bar: BarData | None = None

        self.window: int = window
        # 🧠 ML Signal: Updating low price based on tick data
        self.window_bar: BarData | None = None
        self.on_window_bar: Callable | None = on_window_bar

        self.last_tick: TickData | None = None
        # 🧠 ML Signal: Updating close price with the latest tick price

        self.daily_end: time | None = daily_end
        # 🧠 ML Signal: Updating open interest with the latest tick data
        if self.interval == Interval.DAILY and not self.daily_end:
            raise RuntimeError(_("合成日K线必须传入每日收盘时间"))
    # 🧠 ML Signal: Updating bar datetime with the latest tick datetime

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        # 🧠 ML Signal: Calculating volume change from the last tick
        # 🧠 ML Signal: Method uses conditional logic based on the 'interval' attribute
        """
        new_minute: bool = False
        # 🧠 ML Signal: Calculating turnover change from the last tick
        # ✅ Best Practice: Clear method naming indicates specific functionality

        # 🧠 ML Signal: Method uses conditional logic based on the 'interval' attribute
        # Filter tick data with 0 last price
        if not tick.last_price:
            # 🧠 ML Signal: Storing the last tick for future comparisons
            return
        # ✅ Best Practice: Clear method naming indicates specific functionality

        if not self.bar:
            # ✅ Best Practice: Check if window_bar is initialized before using it
            new_minute = True
        # ✅ Best Practice: Clear method naming indicates specific functionality
        elif (
            # ✅ Best Practice: Use of named arguments for clarity
            (self.bar.datetime.minute != tick.datetime.minute)
            or (self.bar.datetime.hour != tick.datetime.hour)
        ):
            self.bar.datetime = self.bar.datetime.replace(
                second=0, microsecond=0
            )
            self.on_bar(self.bar)

            new_minute = True

        # ✅ Best Practice: Use of max function for readability
        if new_minute:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                # ✅ Best Practice: Use of min function for readability
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            # ✅ Best Practice: Direct assignment for clarity
            )
        elif self.bar:
            # ✅ Best Practice: Incremental update for volume
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            if self.last_tick and tick.high_price > self.last_tick.high_price:
                # ✅ Best Practice: Incremental update for turnover
                self.bar.high_price = max(self.bar.high_price, tick.high_price)

            # ✅ Best Practice: Direct assignment for clarity
            # ✅ Best Practice: Check if hour_bar is None to initialize it properly
            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            if self.last_tick and tick.low_price < self.last_tick.low_price:
                # ✅ Best Practice: Use of modulo for periodic checks
                # ✅ Best Practice: Check if callback is set before calling
                # 🧠 ML Signal: Callback pattern for event-driven programming
                # ✅ Best Practice: Reset window_bar after processing
                # ✅ Best Practice: Use of named parameters for clarity
                self.bar.low_price = min(self.bar.low_price, tick.low_price)

            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

        if self.last_tick and self.bar:
            volume_change: float = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

            turnover_change: float = tick.turnover - self.last_tick.turnover
            self.bar.turnover += max(turnover_change, 0)

        self.last_tick = tick

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.interval == Interval.MINUTE:
            # ✅ Best Practice: Use max to update high_price
            self.update_bar_minute_window(bar)
        elif self.interval == Interval.HOUR:
            self.update_bar_hour_window(bar)
        else:
            # ✅ Best Practice: Use min to update low_price
            self.update_bar_daily_window(bar)

    def update_bar_minute_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            self.window_bar = BarData(
                symbol=bar.symbol,
                # ✅ Best Practice: Check for hour change to finalize the bar
                # ✅ Best Practice: Use of named parameters for clarity
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price,
                bar.high_price
            )
            self.window_bar.low_price = min(
                self.window_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        # ✅ Best Practice: Use max to update high_price
        self.window_bar.open_interest = bar.open_interest

        # Check if window bar completed
        if not (bar.datetime.minute + 1) % self.window:
            # ✅ Best Practice: Use min to update low_price
            if self.on_window_bar:
                self.on_window_bar(self.window_bar)

            self.window_bar = None
    # 🧠 ML Signal: Checks for a specific condition (self.window == 1) to determine behavior

    def update_bar_hour_window(self, bar: BarData) -> None:
        # 🧠 ML Signal: Conditional execution of a callback function
        """"""
        # If not inited, create window bar object
        if not self.hour_bar:
            # ✅ Best Practice: Check if finished_bar is not None before processing
            # 🧠 ML Signal: Lazy initialization pattern for self.window_bar
            dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            # ✅ Best Practice: Use of max and min functions for readability and correctness
            return

        finished_bar: BarData | None = None

        # If minute is 59, update minute bar into window bar and push
        if bar.datetime.minute == 59:
            self.hour_bar.high_price = max(
                self.hour_bar.high_price,
                # 🧠 ML Signal: Updates to object attributes based on input data
                bar.high_price
            )
            self.hour_bar.low_price = min(
                self.hour_bar.low_price,
                bar.low_price
            )
            # 🧠 ML Signal: Modulo operation to determine periodic behavior

            self.hour_bar.close_price = bar.close_price
            # 🧠 ML Signal: Checks if daily_bar is initialized, indicating a pattern of conditional initialization
            self.hour_bar.volume += bar.volume
            # 🧠 ML Signal: Conditional execution of a callback function
            # ✅ Best Practice: Resetting state after use
            # ✅ Best Practice: Initializes daily_bar with attributes from bar, ensuring consistency
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

            finished_bar = self.hour_bar
            self.hour_bar = None

        # If minute bar of new hour, then push existing window bar
        elif bar.datetime.hour != self.hour_bar.datetime.hour:
            finished_bar = self.hour_bar

            # ✅ Best Practice: Updates high_price with the maximum value, ensuring correct data aggregation
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                # ✅ Best Practice: Updates low_price with the minimum value, ensuring correct data aggregation
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                # ✅ Best Practice: Updates close_price to the latest bar's close_price, maintaining data accuracy
                open_interest=bar.open_interest
            # ✅ Best Practice: Accumulates volume, ensuring correct data aggregation
            )
        # Otherwise only update minute bar
        else:
            self.hour_bar.high_price = max(
                self.hour_bar.high_price,
                bar.high_price
            # ✅ Best Practice: Updates open_interest to the latest bar's open_interest, maintaining data accuracy
            )
            # 🧠 ML Signal: Checks if the bar's time matches daily_end, indicating a pattern of time-based operations
            self.hour_bar.low_price = min(
                self.hour_bar.low_price,
                # ✅ Best Practice: Resets datetime to the start of the day, ensuring consistency in daily_bar
                bar.low_price
            )

            self.hour_bar.close_price = bar.close_price
            # ✅ Best Practice: Type hinting improves code readability and maintainability.
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest
        # ✅ Best Practice: Using replace to modify datetime ensures immutability of original datetime object.
        # 🧠 ML Signal: Checks if on_window_bar is callable, indicating a pattern of event-driven programming

        # Push finished window bar
        # 🧠 ML Signal: Callback pattern usage can indicate event-driven architecture.
        # 🧠 ML Signal: Calls a callback function, indicating a pattern of using hooks or callbacks
        if finished_bar:
            self.on_hour_bar(finished_bar)
    # 🧠 ML Signal: Resetting state after use is a common pattern in stateful objects.
    # 🧠 ML Signal: Returning None is a common pattern for functions that may not always produce a result.
    # ✅ Best Practice: Resets daily_bar to None, preparing for the next day's data

    def on_hour_bar(self, bar: BarData) -> None:
        """"""
        if self.window == 1:
            if self.on_window_bar:
                # ✅ Best Practice: Consider adding type hints for better code readability and maintainability
                self.on_window_bar(bar)
        else:
            # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
            if not self.window_bar:
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                    datetime=bar.datetime,
                    gateway_name=bar.gateway_name,
                    # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                    low_price=bar.low_price
                )
            # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
            else:
                # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                self.window_bar.high_price = max(
                    self.window_bar.high_price,
                    bar.high_price
                # 🧠 ML Signal: Incrementing a counter to track the number of updates
                # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                )
                self.window_bar.low_price = min(
                    # 🧠 ML Signal: Conditional logic based on initialization state and count
                    # 🧠 ML Signal: Usage of numpy arrays for data storage, indicating numerical data processing
                    self.window_bar.low_price,
                    bar.low_price
                )
            # ✅ Best Practice: Efficiently updating arrays by shifting elements

            self.window_bar.close_price = bar.close_price
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                # 🧠 ML Signal: Updating the latest values in arrays with new data
                self.interval_count = 0

                if self.on_window_bar:
                    self.on_window_bar(self.window_bar)

                self.window_bar = None
    # ✅ Best Practice: Include type hints for better code readability and maintainability.

    def update_bar_daily_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create daily bar object
        if not self.daily_bar:
            # 🧠 ML Signal: Accessing class attributes directly can indicate usage patterns for ML models.
            self.daily_bar = BarData(
                # ✅ Best Practice: Use @property decorator for getter methods to provide a Pythonic interface.
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=bar.datetime,
                # ✅ Best Practice: Use of a docstring to describe the function's purpose
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                # ✅ Best Practice: Include type hints for better code readability and maintainability.
                low_price=bar.low_price
            )
        # Otherwise, update high/low price into daily bar
        else:
            self.daily_bar.high_price = max(
                self.daily_bar.high_price,
                # ✅ Best Practice: Include type hints for better code readability and maintainability.
                bar.high_price
            )
            self.daily_bar.low_price = min(
                self.daily_bar.low_price,
                bar.low_price
            )
        # ✅ Best Practice: Use @property decorator to provide a getter method for accessing attributes.

        # Update close price/volume/turnover into daily bar
        self.daily_bar.close_price = bar.close_price
        # 🧠 ML Signal: Method returning a time series, useful for time series analysis models
        self.daily_bar.volume += bar.volume
        self.daily_bar.turnover += bar.turnover
        self.daily_bar.open_interest = bar.open_interest
        # ✅ Best Practice: Use of @property decorator for a method that acts like a getter

        # Check if daily bar completed
        if bar.datetime.time() == self.daily_end:
            # 🧠 ML Signal: Method returning a numpy array, indicating usage of numpy for data handling
            self.daily_bar.datetime = bar.datetime.replace(
                hour=0,
                minute=0,
                # ✅ Best Practice: Include a docstring to describe the method's purpose and return value.
                second=0,
                microsecond=0
            )

            # 🧠 ML Signal: Method returns an attribute, indicating a pattern of accessing class data.
            if self.on_window_bar:
                self.on_window_bar(self.daily_bar)

            self.daily_bar = None
    # 🧠 ML Signal: Use of talib.SMA indicates a pattern of using technical analysis for financial data.

    def generate(self) -> BarData | None:
        """
        Generate the bar data and call callback immediately.
        # ✅ Best Practice: Explicitly defining the type of result_value improves code readability and maintainability.
        """
        bar: BarData | None = self.bar
        # ✅ Best Practice: Docstring provides a brief description of the function's purpose.

        if bar:
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(bar)
        # 🧠 ML Signal: Use of talib.EMA indicates a pattern of using technical analysis libraries.

        self.bar = None
        return bar
# ✅ Best Practice: Returning early for a specific condition improves readability.


# ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.
class ArrayManager:
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    # ✅ Best Practice: Return early to reduce nesting and improve readability.
    def __init__(self, size: int = 100) -> None:
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False
        # ✅ Best Practice: Type hint for result_array improves code readability and maintainability

        self.open_array: np.ndarray = np.zeros(size)
        # 🧠 ML Signal: Conditional return based on a boolean flag indicates a pattern for dual output types
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        # ✅ Best Practice: Type hint for result_value improves code readability and maintainability
        self.volume_array: np.ndarray = np.zeros(size)
        self.turnover_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            # ✅ Best Practice: Use of type hinting for result_array improves code readability and maintainability
            self.inited = True

        # 🧠 ML Signal: Conditional return based on a boolean flag (array) indicates a pattern of dual output types
        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        # ✅ Best Practice: Use of type hinting for result_value improves code readability and maintainability
        self.close_array[:-1] = self.close_array[1:]
        # ✅ Best Practice: Include a more descriptive docstring explaining the function's purpose and parameters.
        self.volume_array[:-1] = self.volume_array[1:]
        self.turnover_array[:-1] = self.turnover_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        # 🧠 ML Signal: Usage of talib.CMO indicates a pattern of using technical analysis libraries.
        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        # 🧠 ML Signal: Conditional return based on a boolean flag is a common pattern.
        self.close_array[-1] = bar.close_price
        # ✅ Best Practice: Include a docstring that describes the function's purpose and parameters.
        self.volume_array[-1] = bar.volume
        # 🧠 ML Signal: Accessing the last element of an array is a common pattern.
        self.turnover_array[-1] = bar.turnover
        self.open_interest_array[-1] = bar.open_interest

    @property
    # 🧠 ML Signal: Use of talib library indicates financial data analysis.
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        # ✅ Best Practice: Type hint for result_value improves code readability and maintainability.
        # ✅ Best Practice: Returning early for simple conditions improves readability.
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        # ✅ Best Practice: Include a docstring to describe the function's purpose and behavior
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        # 🧠 ML Signal: Conditional return based on a boolean flag
        return self.low_array

    # ✅ Best Practice: Include a docstring to describe the function's purpose.
    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        # 🧠 ML Signal: Usage of talib.ROC indicates financial data processing.
        """
        return self.close_array

    @property
    # ✅ Best Practice: Use descriptive variable names for clarity.
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        # 🧠 ML Signal: Use of talib library for financial calculations
        return self.volume_array

    # 🧠 ML Signal: Conditional return based on a boolean flag
    @property
    def turnover(self) -> np.ndarray:
        """
        Get trading turnover time series.
        """
        return self.turnover_array

    # 🧠 ML Signal: Use of talib library indicates financial data analysis
    @property
    def open_interest(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        # ✅ Best Practice: Clear separation of logic for returning different types
        return self.open_interest_array
    # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.

    def sma(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Simple moving average.
        # 🧠 ML Signal: Use of talib library indicates financial data analysis.
        """
        result_array: np.ndarray = talib.SMA(self.close, n)
        if array:
            return result_array
        # 🧠 ML Signal: Returning the last element of an array is a common pattern in time series analysis.
        # ✅ Best Practice: Type hinting improves code readability and maintainability

        result_value: float = result_array[-1]
        return result_value

    def ema(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Exponential moving average.
        # 🧠 ML Signal: Conditional return based on a boolean flag
        """
        result_array: np.ndarray = talib.EMA(self.close, n)
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        if array:
            # 🧠 ML Signal: Accessing the last element of an array for a single value
            return result_array

        result_value: float = result_array[-1]
        return result_value
    # 🧠 ML Signal: Use of talib library indicates financial data analysis

    def kama(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        KAMA.
        """
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.
        result_array: np.ndarray = talib.KAMA(self.close, n)
        if array:
            return result_array

        # 🧠 ML Signal: Use of talib library for financial analysis.
        result_value: float = result_array[-1]
        return result_value

    # 🧠 ML Signal: Conditional return based on a boolean flag.
    def wma(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        WMA.
        """
        result_array: np.ndarray = talib.WMA(self.close, n)
        # 🧠 ML Signal: Use of talib.CCI indicates financial data processing
        if array:
            return result_array
        # ✅ Best Practice: Explicitly checking the 'array' flag for clarity

        result_value: float = result_array[-1]
        return result_value
    # ✅ Best Practice: Extracting the last element for single value return

    # ✅ Best Practice: Docstring provides a brief description of the function's purpose.
    def apo(
        self,
        fast_period: int,
        slow_period: int,
        # 🧠 ML Signal: Usage of talib.ATR indicates a pattern of using financial technical analysis libraries.
        matype: int = 0,
        array: bool = False
    ) -> float | np.ndarray:
        """
        APO.
        """
        result_array: np.ndarray = talib.APO(self.close, fast_period, slow_period, matype)      # type: ignore
        if array:
            # ✅ Best Practice: Type hint for result_array improves code readability and maintainability
            return result_array

        result_value: float = result_array[-1]
        # 🧠 ML Signal: Conditional return based on a boolean flag
        return result_value

    # ✅ Best Practice: Type hint for result_value improves code readability and maintainability
    def cmo(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        CMO.
        """
        # ✅ Best Practice: Type hint for result_array improves code readability and maintainability
        result_array: np.ndarray = talib.CMO(self.close, n)
        if array:
            # 🧠 ML Signal: Conditional return based on a boolean flag indicates a pattern of flexible output
            return result_array

        result_value: float = result_array[-1]
        # ✅ Best Practice: Type hint for result_value improves code readability and maintainability
        return result_value

    def mom(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MOM.
        """
        result_array: np.ndarray = talib.MOM(self.close, n)
        # ✅ Best Practice: Include a docstring to describe the function's purpose.
        if array:
            return result_array

        # 🧠 ML Signal: Usage of talib.MACD indicates financial data analysis.
        result_value: float = result_array[-1]
        return result_value

    def ppo(
        self,
        # ✅ Best Practice: Clear conditional logic for returning different data types.
        fast_period: int,
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability.
        slow_period: int,
        # ✅ Best Practice: Return the last element of arrays for non-array mode.
        matype: int = 0,
        array: bool = False
    ) -> float | np.ndarray:
        """
        PPO.
        """
        # ✅ Best Practice: Explicit check for 'array' improves code readability.
        result_array: np.ndarray = talib.PPO(self.close, fast_period, slow_period, matype)      # type: ignore
        if array:
            return result_array
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.
        # ✅ Best Practice: Storing the last element in a variable before returning improves readability.

        result_value: float = result_array[-1]
        return result_value

    # 🧠 ML Signal: Usage of talib library for financial technical analysis.
    def roc(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROC.
        # 🧠 ML Signal: Conditional return based on a boolean flag.
        """
        result_array: np.ndarray = talib.ROC(self.close, n)
        if array:
            return result_array

        # 🧠 ML Signal: Use of talib.DX indicates a pattern of using technical analysis functions
        result_value: float = result_array[-1]
        return result_value
    # ✅ Best Practice: Explicitly checking the 'array' flag improves code readability

    def rocr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCR.
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.
        """
        result_array: np.ndarray = talib.ROCR(self.close, n)
        if array:
            return result_array
        # 🧠 ML Signal: Use of talib library indicates financial data analysis.

        result_value: float = result_array[-1]
        return result_value
    # 🧠 ML Signal: Conditional return based on a boolean flag.

    def rocp(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCP.
        """
        result_array: np.ndarray = talib.ROCP(self.close, n)
        # 🧠 ML Signal: Use of talib library indicates financial or stock market analysis.
        if array:
            return result_array

        # 🧠 ML Signal: Conditional return based on a boolean flag.
        result_value: float = result_array[-1]
        return result_value
    # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters.

    def rocr_100(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCR100.
        # 🧠 ML Signal: Use of talib library indicates financial data analysis.
        """
        result_array: np.ndarray = talib.ROCR100(self.close, n)
        if array:
            return result_array
        # ✅ Best Practice: Use descriptive variable names for clarity.

        result_value: float = result_array[-1]
        return result_value

    def trix(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        TRIX.
        # ✅ Best Practice: Include a docstring to describe the function's purpose and behavior.
        """
        result_array: np.ndarray = talib.TRIX(self.close, n)
        if array:
            return result_array
        # 🧠 ML Signal: Use of talib library indicates financial data analysis.

        result_value: float = result_array[-1]
        return result_value
    # ✅ Best Practice: Explicitly return the result when the array flag is True.

    def std(self, n: int, nbdev: int = 1, array: bool = False) -> float | np.ndarray:
        """
        Standard deviation.
        """
        result_array: np.ndarray = talib.STDDEV(self.close, n, nbdev)
        # 🧠 ML Signal: Use of talib.TRANGE indicates financial data processing.
        if array:
            return result_array

        result_value: float = result_array[-1]
        # ✅ Best Practice: Use descriptive variable names for clarity.
        return result_value

    def obv(self, array: bool = False) -> float | np.ndarray:
        """
        OBV.
        """
        result_array: np.ndarray = talib.OBV(self.close, self.volume)
        if array:
            return result_array
        # 🧠 ML Signal: Use of talib.SMA indicates a pattern of using technical indicators for financial data analysis

        result_value: float = result_array[-1]
        # 🧠 ML Signal: Use of talib.STDDEV indicates a pattern of using standard deviation for financial data analysis
        return result_value

    def cci(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Commodity Channel Index (CCI).
        """
        result_array: np.ndarray = talib.CCI(self.high, self.low, self.close, n)
        if array:
            # ✅ Best Practice: Use of type annotations for float variables improves code readability and maintainability
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def atr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Average True Range (ATR).
        """
        result_array: np.ndarray = talib.ATR(self.high, self.low, self.close, n)
        if array:
            # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
            return result_array

        # 🧠 ML Signal: Use of talib.SMA indicates a pattern of using technical indicators for financial data analysis.
        result_value: float = result_array[-1]
        return result_value
    # 🧠 ML Signal: Use of talib.ATR indicates a pattern of using technical indicators for financial data analysis.

    def natr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        NATR.
        """
        result_array: np.ndarray = talib.NATR(self.high, self.low, self.close, n)
        if array:
            # ⚠️ SAST Risk (Low): Accessing the last element of an array without checking if the array is empty could lead to an IndexError.
            return result_array
        # ⚠️ SAST Risk (Low): Accessing the last element of an array without checking if the array is empty could lead to an IndexError.

        result_value: float = result_array[-1]
        return result_value

    def rsi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Relative Strenght Index (RSI).
        """
        result_array: np.ndarray = talib.RSI(self.close, n)
        # 🧠 ML Signal: Use of conditional logic to return different data types based on a flag.
        if array:
            return result_array
        # 🧠 ML Signal: Returning the last element of an array, a common pattern in time series analysis.

        result_value: float = result_array[-1]
        return result_value

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        # 🧠 ML Signal: Use of talib.AROON indicates financial data analysis
        signal_period: int,
        # ⚠️ SAST Risk (Low): Potential for incorrect handling of financial data
        array: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        """
        MACD.
        # ✅ Best Practice: Returning the last element for non-array mode
        """
        macd, signal, hist = talib.MACD(
            self.close, fast_period, slow_period, signal_period
        # 🧠 ML Signal: Use of talib.AROONOSC indicates a pattern of using TA-Lib for technical analysis
        )
        if array:
            # ✅ Best Practice: Explicitly checking the 'array' flag improves code readability and maintainability
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    # ✅ Best Practice: Storing the last element in a variable before returning improves readability
    def adx(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ADX.
        """
        result_array: np.ndarray = talib.ADX(self.high, self.low, self.close, n)
        # 🧠 ML Signal: Use of talib library indicates financial or time series analysis
        if array:
            return result_array

        result_value: float = result_array[-1]
        # ✅ Best Practice: Explicitly typing variables improves code readability and maintainability
        return result_value
    # ✅ Best Practice: Docstring is present but could be more descriptive about the function's purpose and parameters.

    def adxr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ADXR.
        # 🧠 ML Signal: Use of talib library indicates financial or time series analysis.
        """
        result_array: np.ndarray = talib.ADXR(self.high, self.low, self.close, n)
        if array:
            # 🧠 ML Signal: Conditional return based on a boolean flag, indicating a pattern of flexible output types.
            return result_array

        # 🧠 ML Signal: Accessing the last element of an array, common in time series analysis to get the most recent value.
        result_value: float = result_array[-1]
        return result_value

    # 🧠 ML Signal: Use of talib library for financial calculations
    def dx(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        DX.
        """
        result_array: np.ndarray = talib.DX(self.high, self.low, self.close, n)
        # 🧠 ML Signal: Accessing the last element of an array
        if array:
            # ✅ Best Practice: Include a docstring to describe the function's purpose.
            return result_array

        result_value: float = result_array[-1]
        return result_value
    # 🧠 ML Signal: Use of talib.AD function indicates financial data processing.

    def minus_di(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MINUS_DI.
        # 🧠 ML Signal: Returning the last element of an array, common in time series analysis.
        """
        result_array: np.ndarray = talib.MINUS_DI(self.high, self.low, self.close, n)
        if array:
            return result_array

        result_value: float = result_array[-1]
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
        return result_value

    def plus_di(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        PLUS_DI.
        """
        result_array: np.ndarray = talib.PLUS_DI(self.high, self.low, self.close, n)
        # 🧠 ML Signal: Conditional return based on a boolean flag
        if array:
            return result_array
        # ✅ Best Practice: Include a docstring to describe the function's purpose.

        result_value: float = result_array[-1]
        return result_value

    # 🧠 ML Signal: Use of talib library for financial analysis.
    def willr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        WILLR.
        # 🧠 ML Signal: Conditional return based on a boolean flag.
        """
        result_array: np.ndarray = talib.WILLR(self.high, self.low, self.close, n)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def ultosc(
        self,
        # ✅ Best Practice: Docstring provides a brief description of the function's purpose.
        time_period1: int = 7,
        time_period2: int = 14,
        time_period3: int = 28,
        # 🧠 ML Signal: Usage of talib.STOCH indicates a pattern of using technical analysis indicators.
        array: bool = False
    ) -> float | np.ndarray:
        """
        Ultimate Oscillator.
        """
        result_array: np.ndarray = talib.ULTOSC(self.high, self.low, self.close, time_period1, time_period2, time_period3)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    # ✅ Best Practice: Conditional return based on the 'array' flag improves function flexibility.
    def trange(self, array: bool = False) -> float | np.ndarray:
        """
        TRANGE.
        """
        result_array: np.ndarray = talib.TRANGE(self.high, self.low, self.close)
        if array:
            # 🧠 ML Signal: Use of talib.SAR indicates financial time series analysis.
            return result_array

        result_value: float = result_array[-1]
        # 🧠 ML Signal: Conditional return based on a boolean flag.
        return result_value
    # ✅ Best Practice: Add type hint for the parameter 'func' to improve code readability and maintainability

    # 🧠 ML Signal: Returning the last element of an array, common in time series analysis.
    def boll(
        self,
        n: int,
        dev: float,
        array: bool = False
    # ✅ Best Practice: Return the function itself to allow for decorator chaining and maintain the original function's signature
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Bollinger Channel.
        """
        mid_array: np.ndarray = talib.SMA(self.close, n)
        std_array: np.ndarray = talib.STDDEV(self.close, n, 1)

        if array:
            up_array: np.ndarray = mid_array + std_array * dev
            down_array: np.ndarray = mid_array - std_array * dev
            return up_array, down_array
        else:
            mid: float = mid_array[-1]
            std: float = std_array[-1]
            up: float = mid + std * dev
            down: float = mid - std * dev
            return up, down

    def keltner(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Keltner Channel.
        """
        mid_array: np.ndarray = talib.SMA(self.close, n)
        atr_array: np.ndarray = talib.ATR(self.high, self.low, self.close, n)

        if array:
            up_array: np.ndarray = mid_array + atr_array * dev
            down_array: np.ndarray = mid_array - atr_array * dev
            return up_array, down_array
        else:
            mid: float = mid_array[-1]
            atr: float = atr_array[-1]
            up: float = mid + atr * dev
            down: float = mid - atr * dev
            return up, down

    def donchian(
        self, n: int, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Donchian Channel.
        """
        up: np.ndarray = talib.MAX(self.high, n)
        down: np.ndarray = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(
        self,
        n: int,
        array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Aroon indicator.
        """
        aroon_down, aroon_up = talib.AROON(self.high, self.low, n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]

    def aroonosc(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Aroon Oscillator.
        """
        result_array: np.ndarray = talib.AROONOSC(self.high, self.low, n)

        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def minus_dm(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MINUS_DM.
        """
        result_array: np.ndarray = talib.MINUS_DM(self.high, self.low, n)

        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def plus_dm(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        PLUS_DM.
        """
        result_array: np.ndarray = talib.PLUS_DM(self.high, self.low, n)

        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def mfi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Money Flow Index.
        """
        result_array: np.ndarray = talib.MFI(self.high, self.low, self.close, self.volume, n)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def ad(self, array: bool = False) -> float | np.ndarray:
        """
        AD.
        """
        result_array: np.ndarray = talib.AD(self.high, self.low, self.close, self.volume)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def adosc(
        self,
        fast_period: int,
        slow_period: int,
        array: bool = False
    ) -> float | np.ndarray:
        """
        ADOSC.
        """
        result_array: np.ndarray = talib.ADOSC(self.high, self.low, self.close, self.volume, fast_period, slow_period)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def bop(self, array: bool = False) -> float | np.ndarray:
        """
        BOP.
        """
        result_array: np.ndarray = talib.BOP(self.open, self.high, self.low, self.close)

        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value

    def stoch(
        self,
        fastk_period: int,
        slowk_period: int,
        slowk_matype: int,
        slowd_period: int,
        slowd_matype: int,
        array: bool = False
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Indicator
        """
        k, d = talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period,
            slowk_period,
            slowk_matype,    # type: ignore
            slowd_period,
            slowd_matype     # type: ignore
        )
        if array:
            return k, d
        return k[-1], d[-1]

    def sar(self, acceleration: float, maximum: float, array: bool = False) -> float | np.ndarray:
        """
        SAR.
        """
        result_array: np.ndarray = talib.SAR(self.high, self.low, acceleration, maximum)
        if array:
            return result_array

        result_value: float = result_array[-1]
        return result_value


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func