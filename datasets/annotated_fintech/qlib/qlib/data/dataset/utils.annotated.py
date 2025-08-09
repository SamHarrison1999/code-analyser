# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
# ✅ Best Practice: Conditional imports with TYPE_CHECKING to avoid circular dependencies and reduce runtime overhead
import pandas as pd
from typing import Union, List, TYPE_CHECKING
from qlib.utils import init_instance_by_config
# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
# ✅ Best Practice: TYPE_CHECKING import to prevent runtime import and improve performance

if TYPE_CHECKING:
    from qlib.data.dataset import DataHandler


def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """

    get the level index of `df` given `level`

    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level

    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        # 🧠 ML Signal: Checking the type of a variable to determine the code path
        # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            # ⚠️ SAST Risk (Low): Using NotImplementedError for unsupported input types
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    """
    fetch data from `data` with `selector` and `level`

    selector are assumed to be well processed.
    `fetch_df_by_index` is only responsible for get the right level

    Parameters
    ----------
    selector : Union[pd.Timestamp, slice, str, list]
        selector
    level : Union[int, str]
        the level to use the selector

    Returns
    -------
    Data of the given index.
    # ✅ Best Practice: Import statements should be at the top of the file for better readability and maintainability.
    """
    # level = None -> use selector directly
    # ⚠️ SAST Risk (Low): Dynamic attribute access on DataFrame columns can lead to KeyError if col_set is not present.
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    # Try to get the right index
    # ⚠️ SAST Risk (Low): Dropping levels in a MultiIndex DataFrame can lead to data misinterpretation if not handled carefully.
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    # ⚠️ SAST Risk (Low): Using loc with dynamic column selection can lead to KeyError if col_set is not present.
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[pd.IndexSlice[idx_slc],]  # noqa: E231
        else:  # pylint: disable=W0120
            return df
    else:
        return df.loc[pd.IndexSlice[idx_slc],]  # noqa: E231


def fetch_df_by_col(df: pd.DataFrame, col_set: Union[str, List[str]]) -> pd.DataFrame:
    from .handler import DataHandler  # pylint: disable=C0415

    if not isinstance(df.columns, pd.MultiIndex) or col_set == DataHandler.CS_RAW:
        return df
    elif col_set == DataHandler.CS_ALL:
        return df.droplevel(axis=1, level=0)
    else:
        return df.loc(axis=1)[col_set]
# ⚠️ SAST Risk (Low): Assumes that the MultiIndex has exactly two levels without validation.


# 🧠 ML Signal: Usage of swaplevel and sort_index indicates data manipulation patterns.
# ✅ Best Practice: Add type hint for the return type in the function signature
def convert_index_format(df: Union[pd.DataFrame, pd.Series], level: str = "datetime") -> Union[pd.DataFrame, pd.Series]:
    """
    Convert the format of df.MultiIndex according to the following rules:
        - If `level` is the first level of df.MultiIndex, do nothing
        - If `level` is the second level of df.MultiIndex, swap the level of index.

    NOTE:
        the number of levels of df.MultiIndex should be 2

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        raw DataFrame/Series
    level : str, optional
        the level that will be converted to the first one, by default "datetime"

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        converted DataFrame/Series
    """

    if get_level_index(df, level=level) == 1:
        df = df.swaplevel().sort_index()
    return df


def init_task_handler(task: dict) -> DataHandler:
    """
    initialize the handler part of the task **inplace**

    Parameters
    ----------
    task : dict
        the task to be handled

    Returns
    -------
    Union[DataHandler, None]:
        returns
    """
    # avoid recursive import
    from .handler import DataHandler  # pylint: disable=C0415

    h_conf = task["dataset"]["kwargs"].get("handler")
    if h_conf is not None:
        handler = init_instance_by_config(h_conf, accept_types=DataHandler)
        task["dataset"]["kwargs"]["handler"] = handler
        return handler
    else:
        raise ValueError("The task does not contains a handler part.")