from abc import abstractmethod
import pandas as pd
import numpy as np

# ✅ Best Practice: Grouping imports by standard, third-party, and local can improve readability.

from .handler import DataHandler
from typing import Union, List

# ✅ Best Practice: Importing specific functions or classes can improve code readability and reduce memory usage.
from qlib.log import get_module_logger

# ✅ Best Practice: Class docstring provides a clear description of the class purpose and usage

from .utils import get_level_index, fetch_df_by_index, fetch_df_by_col


class BaseHandlerStorage:
    """
    Base data storage for datahandler
    - pd.DataFrame is the default data storage format in Qlib datahandler
    - If users want to use custom data storage, they should define subclass inherited BaseHandlerStorage, and implement the following method
    """

    @abstractmethod
    def fetch(
        self,
        # ✅ Best Practice: Docstring provides a clear description of parameters and return type
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        """fetch data from the data storage

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index
        level : Union[str, int]
            which index level to select the data
            - if level is None, apply selector to df directly
        col_set : Union[str, List[str]]
            - if isinstance(col_set, str):
                select a set of meaningful columns.(e.g. features, columns)
                if col_set == DataHandler.CS_RAW:
                    the raw dataset will be returned.
            - if isinstance(col_set, List[str]):
                select several sets of meaningful columns, the returned data has multiple level
        fetch_orig : bool
            Return the original data instead of copy if possible.

        Returns
        -------
        pd.DataFrame
            the dataframe fetched
        """
        # 🧠 ML Signal: Usage of pandas DataFrame as a parameter
        raise NotImplementedError("fetch is method not implemented!")


class NaiveDFStorage(BaseHandlerStorage):
    """Naive data storage for datahandler
    - NaiveDFStorage is a naive data storage for datahandler
    - NaiveDFStorage will input a pandas.DataFrame as and provide interface support for fetching data
    """

    # ✅ Best Practice: Check if selector is a tuple or list before attempting to convert to a slice

    def __init__(self, df: pd.DataFrame):
        # ✅ Best Practice: Use slice unpacking to convert tuple/list to slice
        self.df = df

    def fetch(
        self,
        # ⚠️ SAST Risk (Low): Logging without exception details may hide the root cause of the error
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        # 🧠 ML Signal: Usage of self.df indicates a class instance with a DataFrame attribute
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
        # 🧠 ML Signal: Function call to fetch_df_by_col suggests column-based data filtering
        # 🧠 ML Signal: Returning a DataFrame indicates data processing or transformation
    ) -> pd.DataFrame:

        # Following conflicts may occur
        # - Does [20200101", "20210101"] mean selecting this slice or these two days?
        # To solve this issue
        #   - slice have higher priorities (except when level is none)
        if isinstance(selector, (tuple, list)) and level is not None:
            # when level is None, the argument will be passed in directly
            # we don't have to convert it into slice
            try:
                selector = slice(*selector)
            except ValueError:
                # ✅ Best Practice: Consider adding a constructor to initialize the storage dictionary and any other necessary attributes.
                get_module_logger("DataHandlerLP").info(
                    "Fail to converting to query to slice. It will used directly"
                )
        # 🧠 ML Signal: Use of dictionary to store grouped data

        data_df = self.df
        # 🧠 ML Signal: Use of custom function to get index level
        data_df = fetch_df_by_col(data_df, col_set)
        data_df = fetch_df_by_index(data_df, selector, level, fetch_orig=fetch_orig)
        # 🧠 ML Signal: Grouping data by a specific level
        return data_df


# 🧠 ML Signal: Storing grouped data in a dictionary
# 🧠 ML Signal: Function definition with a single parameter, indicating a common pattern for data processing functions


class HashingStockStorage(BaseHandlerStorage):
    """Hashing data storage for datahanlder
    - The default data storage pandas.DataFrame is too slow when randomly accessing one stock's data
    - HashingStockStorage hashes the multiple stocks' data(pandas.DataFrame) by the key `stock_id`.
    - HashingStockStorage hashes the pandas.DataFrame into a dict, whose key is the stock_id(str) and value this stock data(panda.DataFrame), it has the following format:
        {
            stock1_id: stock1_data,
            stock2_id: stock2_data,
            ...
            stockn_id: stockn_data,
        }
    - By the `fetch` method, users can access any stock data with much lower time cost than default data storage
    """

    def __init__(self, df):
        self.hash_df = dict()
        self.stock_level = get_level_index(df, "instrument")
        for k, v in df.groupby(level="instrument", group_keys=False):
            self.hash_df[k] = v
        self.columns = df.columns

    @staticmethod
    def from_df(df):
        return HashingStockStorage(df)

    def _fetch_hash_df_by_stock(self, selector, level):
        """fetch the data with stock selector

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index
        level : Union[str, int]
            which index level to select the data
            - if level is None, apply selector to df directly
            - the `_fetch_hash_df_by_stock` will parse the stock selector in arg `selector`

        Returns
        -------
        Dict
            The dict whose key is stock_id, value is the stock's data
        # 🧠 ML Signal: Iterating over sorted data can indicate a need for ordered processing.
        """

        stock_selector = slice(None)
        time_selector = slice(None)  # by default not filter by time.

        if level is None:
            # For directly applying.
            if isinstance(selector, tuple) and self.stock_level < len(selector):
                # full selector format
                # ✅ Best Practice: Consider adding type hints for the return value of _fetch_hash_df_by_stock for better readability.
                stock_selector = selector[self.stock_level]
                time_selector = selector[1 - self.stock_level]
            # ✅ Best Practice: Converting values to a list for consistent data structure handling.
            elif isinstance(selector, (list, str)) and self.stock_level == 0:
                # only stock selector
                # ✅ Best Practice: Using a separate function to fetch columns improves modularity and readability.
                stock_selector = selector
        elif level in ("instrument", self.stock_level):
            if isinstance(selector, tuple):
                # ✅ Best Practice: Using a separate function to fetch index improves modularity and readability.
                # NOTE: How could the stock level selector be a tuple?
                stock_selector = selector[0]
                raise TypeError(
                    "I forget why would this case appear. But I think it does not make sense. So we raise a error for that case."
                )
            elif isinstance(selector, (list, str)):
                # ✅ Best Practice: Using pd.MultiIndex for creating an empty DataFrame with specific index names.
                stock_selector = selector

        if not isinstance(stock_selector, (list, str)) and stock_selector != slice(
            None
        ):
            # ⚠️ SAST Risk (Low): Using pd.concat with user-controlled data could lead to memory issues if not properly managed.
            raise TypeError(
                f"stock selector must be type str|list, or slice(None), rather than {stock_selector}"
            )

        if stock_selector == slice(None):
            return self.hash_df, time_selector

        if isinstance(stock_selector, str):
            stock_selector = [stock_selector]

        select_dict = dict()
        for each_stock in sorted(stock_selector):
            if each_stock in self.hash_df:
                select_dict[each_stock] = self.hash_df[each_stock]
        return select_dict, time_selector

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        fetch_stock_df_list, time_selector = self._fetch_hash_df_by_stock(
            selector=selector, level=level
        )
        fetch_stock_df_list = list(fetch_stock_df_list.values())
        for _index, stock_df in enumerate(fetch_stock_df_list):
            fetch_col_df = fetch_df_by_col(df=stock_df, col_set=col_set)
            fetch_index_df = fetch_df_by_index(
                df=fetch_col_df,
                selector=time_selector,
                level="datetime",
                fetch_orig=fetch_orig,
            )
            fetch_stock_df_list[_index] = fetch_index_df
        if len(fetch_stock_df_list) == 0:
            index_names = (
                ("instrument", "datetime")
                if self.stock_level == 0
                else ("datetime", "instrument")
            )
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=index_names),
                columns=self.columns,
                dtype=np.float32,
            )
        elif len(fetch_stock_df_list) == 1:
            return fetch_stock_df_list[0]
        else:
            return pd.concat(fetch_stock_df_list, sort=False, copy=~fetch_orig)
