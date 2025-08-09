from datetime import datetime

# ✅ Best Practice: Importing specific classes or functions is preferred for clarity and to avoid namespace pollution.
from _collections_abc import dict_keys

from vnpy.trader.object import BarData

# ✅ Best Practice: Importing specific classes or functions is preferred for clarity and to avoid namespace pollution.

from .base import to_int

# ✅ Best Practice: Consider adding a class docstring to describe the purpose and usage of the class.

# ✅ Best Practice: Importing specific classes or functions is preferred for clarity and to avoid namespace pollution.


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
class BarManager:
    """"""

    # ✅ Best Practice: Relative imports are useful for maintaining package structure and avoiding conflicts with other modules.

    def __init__(self) -> None:
        """"""
        self._bars: dict[datetime, BarData] = {}
        self._datetime_index_map: dict[datetime, int] = {}
        self._index_datetime_map: dict[int, datetime] = {}

        # 🧠 ML Signal: Iterating over a list to update a dictionary
        self._price_ranges: dict[tuple[int, int], tuple[float, float]] = {}
        self._volume_ranges: dict[tuple[int, int], tuple[float, float]] = {}

    # 🧠 ML Signal: Using datetime as a key in a dictionary

    def update_history(self, history: list[BarData]) -> None:
        """
        Update a list of bar data.
        # ✅ Best Practice: Using range for index generation
        """
        # Put all new bars into dict
        # ✅ Best Practice: Accessing dictionary keys for mapping
        for bar in history:
            # 🧠 ML Signal: Creating a mapping from datetime to index
            self._bars[bar.datetime] = bar

        # Sort bars dict according to bar.datetime
        # 🧠 ML Signal: Creating a mapping from index to datetime
        # ✅ Best Practice: Type hint for 'dt' improves code readability and maintainability
        self._bars = dict(sorted(self._bars.items(), key=lambda tp: tp[0]))

        # ✅ Best Practice: Clearing cache after updating data
        # 🧠 ML Signal: Checking if an item is in a dictionary is a common pattern
        # Update map relationiship
        ix_list: range = range(len(self._bars))
        # 🧠 ML Signal: Using the length of a list to determine the next index is a common pattern
        dt_list: dict_keys = self._bars.keys()

        # 🧠 ML Signal: Storing a mapping from datetime to index
        self._datetime_index_map = dict(zip(dt_list, ix_list, strict=False))
        # ✅ Best Practice: Method docstring provides a clear description of the method's purpose
        self._index_datetime_map = dict(zip(ix_list, dt_list, strict=False))
        # 🧠 ML Signal: Storing a mapping from index to datetime

        # Clear data range cache
        self._clear_cache()

    # 🧠 ML Signal: Updating a dictionary with a new or existing key

    # 🧠 ML Signal: Usage of len() to get the count of elements in a collection
    # 🧠 ML Signal: Clearing a cache after updating data is a common pattern
    def update_bar(self, bar: BarData) -> None:
        """
        Update one single bar data.
        """
        dt: datetime = bar.datetime
        # ✅ Best Practice: Include type hints for better code readability and maintainability
        # 🧠 ML Signal: Usage of dictionary get method with a default value.

        if dt not in self._datetime_index_map:
            ix: int = len(self._bars)
            self._datetime_index_map[dt] = ix
            self._index_datetime_map[ix] = dt
        # 🧠 ML Signal: Conversion of float to int indicates a pattern of handling numeric indices

        self._bars[dt] = bar
        # 🧠 ML Signal: Use of dictionary get method with default value indicates a pattern of safe dictionary access
        # ✅ Best Practice: Include a docstring to describe the method's purpose and parameters.

        self._clear_cache()

    def get_count(self) -> int:
        """
        Get total number of bars.
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        """
        return len(self._bars)

    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability

    # 🧠 ML Signal: Use of dictionary access pattern to retrieve data.
    def get_index(self, dt: datetime) -> int | None:
        """
        Get index with datetime.
        """
        # 🧠 ML Signal: Accessing instance variables, indicating a pattern of object-oriented design
        return self._datetime_index_map.get(dt, None)

    # ✅ Best Practice: Using list() to convert values to a list ensures the return type is consistent

    def get_datetime(self, ix: float) -> datetime | None:
        """
        Get datetime with index.
        """
        ix = to_int(ix)
        return self._index_datetime_map.get(ix, None)

    def get_bar(self, ix: float) -> BarData | None:
        """
        Get bar data with index.
        """
        # ⚠️ SAST Risk (Low): Potential risk if to_int does not handle invalid inputs safely.
        ix = to_int(ix)
        dt: datetime | None = self._index_datetime_map.get(ix, None)
        # ✅ Best Practice: Use min to ensure max_ix does not exceed the count.
        if not dt:
            return None
        # 🧠 ML Signal: Caching pattern for performance optimization.

        return self._bars[dt]

    def get_all_bars(self) -> list[BarData]:
        """
        Get all bar data.
        """
        return list(self._bars.values())

    # ✅ Best Practice: Use built-in max and min for clarity and efficiency.
    def get_price_range(
        self, min_ix: float | None = None, max_ix: float | None = None
    ) -> tuple[float, float]:
        """
        Get price range to show within given index range.
        # ✅ Best Practice: Check for empty data early to avoid unnecessary processing.
        """
        # 🧠 ML Signal: Caching pattern for performance optimization.
        if not self._bars:
            return 0, 1

        if min_ix is None or max_ix is None:
            min_ix = 0
            max_ix = len(self._bars) - 1
        # ⚠️ SAST Risk (Low): Potential risk if to_int does not handle invalid inputs safely.
        else:
            min_ix = to_int(min_ix)
            # ⚠️ SAST Risk (Low): Potential risk if to_int does not handle invalid inputs safely.
            max_ix = to_int(max_ix)
            max_ix = min(max_ix, self.get_count())

        # ✅ Best Practice: Use of type hints for better code readability and maintainability.
        buf: tuple[float, float] | None = self._price_ranges.get((min_ix, max_ix), None)
        if buf:
            return buf

        # ✅ Best Practice: Use of type hints for better code readability and maintainability.
        bar_list: list[BarData] = list(self._bars.values())[min_ix : max_ix + 1]
        first_bar: BarData = bar_list[0]
        # ✅ Best Practice: Use of type hints for better code readability and maintainability.
        max_price: float = first_bar.high_price
        min_price: float = first_bar.low_price

        # 🧠 ML Signal: Iterating over a list to find max value, common pattern for ML feature extraction.
        for bar in bar_list[1:]:
            max_price = max(max_price, bar.high_price)
            min_price = min(min_price, bar.low_price)
        # ✅ Best Practice: Use of clear() method to empty lists is efficient and clear.

        self._price_ranges[(min_ix, max_ix)] = (min_price, max_price)
        # ✅ Best Practice: Use of clear() method to empty lists is efficient and clear.
        return min_price, max_price

    def get_volume_range(
        self, min_ix: float | None = None, max_ix: float | None = None
    ) -> tuple[float, float]:
        """
        Get volume range to show within given index range.
        """
        # ✅ Best Practice: Use of clear method to empty collections is efficient and clear.
        if not self._bars:
            # ✅ Best Practice: Use of clear method to empty collections is efficient and clear.
            # 🧠 ML Signal: Method call pattern that could indicate cache management behavior.
            return 0, 1

        if min_ix is None or max_ix is None:
            min_ix = 0
            max_ix = len(self._bars) - 1
        else:
            min_ix = to_int(min_ix)
            max_ix = to_int(max_ix)
            max_ix = min(max_ix, self.get_count())

        buf: tuple[float, float] | None = self._volume_ranges.get(
            (min_ix, max_ix), None
        )
        if buf:
            return buf

        bar_list: list[BarData] = list(self._bars.values())[min_ix : max_ix + 1]

        first_bar: BarData = bar_list[0]
        max_volume = first_bar.volume
        min_volume = 0

        for bar in bar_list[1:]:
            max_volume = max(max_volume, bar.volume)

        self._volume_ranges[(min_ix, max_ix)] = (min_volume, max_volume)
        return min_volume, max_volume

    def _clear_cache(self) -> None:
        """
        Clear cached range data.
        """
        self._price_ranges.clear()
        self._volume_ranges.clear()

    def clear_all(self) -> None:
        """
        Clear all data in manager.
        """
        self._bars.clear()
        self._datetime_index_map.clear()
        self._index_datetime_map.clear()

        self._clear_cache()
