from abc import ABC, abstractmethod
from datetime import datetime
from types import ModuleType
from dataclasses import dataclass
from importlib import import_module

# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.

from .constant import Interval, Exchange
from .object import BarData, TickData
from .setting import SETTINGS
from .utility import ZoneInfo
from .locale import _

# ⚠️ SAST Risk (Low): Using settings from an external source can introduce security risks if not validated.


DB_TZ = ZoneInfo(SETTINGS["database.timezone"])

# ⚠️ SAST Risk (Medium): Potential timezone conversion issues if DB_TZ is not defined or incorrect


# ⚠️ SAST Risk (Low): Removing timezone info can lead to ambiguity in datetime representation
def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    # ✅ Best Practice: Use of @dataclass for automatic generation of special methods
    """
    dt = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)


# ✅ Best Practice: Type annotations improve code readability and maintainability.


@dataclass
# ✅ Best Practice: Type annotations improve code readability and maintainability.
class BarOverview:
    """
    Overview of bar data stored in database.
    """

    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    symbol: str = ""
    exchange: Exchange | None = None
    interval: Interval | None = None
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    count: int = 0
    start: datetime | None = None
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    # ⚠️ SAST Risk (Low): Missing import for @dataclass, which could lead to runtime errors if not imported elsewhere.
    end: datetime | None = None


# ✅ Best Practice: Type annotations improve code readability and maintainability.


@dataclass
# ✅ Best Practice: Type annotations improve code readability and maintainability.
class TickOverview:
    """
    Overview of tick data stored in database.
    """

    # ✅ Best Practice: Use of abstractmethod enforces implementation in subclasses

    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    symbol: str = ""
    exchange: Exchange | None = None
    count: int = 0
    start: datetime | None = None
    end: datetime | None = None


# ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses, promoting a clear contract for class design.
# ✅ Best Practice: Include a docstring to describe the purpose and usage of the function


class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """

    # ✅ Best Practice: Use @abstractmethod to enforce implementation in subclasses

    @abstractmethod
    def save_bar_data(self, bars: list[BarData], stream: bool = False) -> bool:
        """
        Save bar data into database.
        """
        pass

    # 🧠 ML Signal: Function signature with type annotations indicates expected input and output types
    @abstractmethod
    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
        """
        Save tick data into database.
        """
        # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses
        pass

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        # ✅ Best Practice: Docstring provided for method description
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """
        Load bar data from database.
        """
        pass

    @abstractmethod
    def load_tick_data(
        # ✅ Best Practice: Docstring provides a clear description of the method's purpose.
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
        # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses.
    ) -> list[TickData]:
        """
        Load tick data from database.
        """
        pass

    @abstractmethod
    def delete_bar_data(
        # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses, promoting a clear contract for class design.
        self,
        symbol: str,
        exchange: Exchange,
        # ✅ Best Practice: Include a docstring to describe the method's purpose and behavior
        interval: Interval,
    ) -> int:
        """
        Delete all bar data with given symbol + exchange + interval.
        """
        pass

    # ✅ Best Practice: Include a docstring to describe the function's purpose and behavior
    # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses

    @abstractmethod
    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        # ✅ Best Practice: Use type hinting for variable declarations to improve code readability and maintainability
    ) -> int:
        """
        Delete all tick data with given symbol + exchange.
        """
        pass

    # 🧠 ML Signal: Dynamic module import based on configuration, indicating a pattern of plugin or driver loading.
    @abstractmethod
    def get_bar_overview(self) -> list[BarOverview]:
        """
        Return bar data avaible in database.
        """
        pass

    # ⚠️ SAST Risk (Low): Use of print statements for error handling can expose sensitive information in production environments.
    # 🧠 ML Signal: Instantiation of a database object, indicating a pattern of database connection or initialization.
    # ✅ Best Practice: Fallback to a default module ensures robustness if the specified module is not found.

    @abstractmethod
    def get_tick_overview(self) -> list[TickOverview]:
        """
        Return tick data avaible in database.
        """
        pass


database: BaseDatabase | None = None


def get_database() -> BaseDatabase:
    """"""
    # Return database object if already inited
    global database
    if database:
        return database

    # Read database related global setting
    database_name: str = SETTINGS["database.name"]
    module_name: str = f"vnpy_{database_name}"

    # Try to import database module
    try:
        module: ModuleType = import_module(module_name)
    except ModuleNotFoundError:
        print(_("找不到数据库驱动{}，使用默认的SQLite数据库").format(module_name))
        module = import_module("vnpy_sqlite")

    # Create database object from module
    database = module.Database()
    return database
