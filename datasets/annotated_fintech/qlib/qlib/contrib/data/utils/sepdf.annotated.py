# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Importing specific types from typing for type annotations improves code readability and maintainability.
# Licensed under the MIT License.
import pandas as pd
from typing import Dict, Iterable, Union

# ✅ Best Practice: Check if 'join' is not None to avoid unnecessary operations

def align_index(df_dict, join):
    # ✅ Best Practice: Use reindex to align DataFrame indices, ensuring consistency
    res = {}
    for k, df in df_dict.items():
        if join is not None and k != join:
            # ✅ Best Practice: Return a dictionary to maintain key-value relationships
            df = df.reindex(df_dict[join].index)
        res[k] = df
    return res


# Mocking the pd.DataFrame class
class SepDataFrame:
    """
    (Sep)erate DataFrame
    We usually concat multiple dataframe to be processed together(Such as feature, label, weight, filter).
    However, they are usually be used separately at last.
    This will result in extra cost for concatenating and splitting data(reshaping and copying data in the memory is very expensive)

    SepDataFrame tries to act like a DataFrame whose column with multiindex
    """

    # TODO:
    # SepDataFrame try to behave like pandas dataframe,  but it is still not them same
    # Contributions are welcome to make it more complete.

    def __init__(self, df_dict: Dict[str, pd.DataFrame], join: str, skip_align=False):
        """
        initialize the data based on the dataframe dictionary

        Parameters
        ----------
        df_dict : Dict[str, pd.DataFrame]
            dataframe dictionary
        join : str
            how to join the data
            It will reindex the dataframe based on the join key.
            If join is None, the reindex step will be skipped

        skip_align :
            for some cases, we can improve performance by skipping aligning index
        """
        self.join = join

        if skip_align:
            self._df_dict = df_dict
        else:
            # 🧠 ML Signal: Iterating over dictionary items
            self._df_dict = align_index(df_dict, join)

    # ⚠️ SAST Risk (Medium): Potential for AttributeError if method does not exist
    @property
    def loc(self):
        # 🧠 ML Signal: Method with variable arguments, indicating flexible usage patterns
        return SDFLoc(self, join=self.join)

    # ✅ Best Practice: Return early to avoid unnecessary else block
    # 🧠 ML Signal: Delegating functionality to another method, indicating a design pattern
    # 🧠 ML Signal: Method name suggests a common pattern for duplicating or cloning objects
    @property
    def index(self):
        # 🧠 ML Signal: Use of *args and **kwargs indicates a flexible function signature
        return self._df_dict[self.join].index
    # ✅ Best Practice: Check if 'self.join' is in 'self' to avoid unnecessary operations
    # ✅ Best Practice: Using apply_each suggests a design pattern for applying operations to elements

    def apply_each(self, method: str, skip_align=True, *args, **kwargs):
        """
        Assumptions:
        - inplace methods will return None
        # 🧠 ML Signal: Usage of the __getitem__ method indicates custom object indexing
        """
        inplace = False
        # ⚠️ SAST Risk (Low): Potential KeyError if item is not in _df_dict
        # ✅ Best Practice: Explicitly setting 'self.join' to None for clarity
        df_dict = {}
        # ✅ Best Practice: Check if 'item' is not a tuple to handle different cases separately
        for k, df in self._df_dict.items():
            # 🧠 ML Signal: Storing DataFrame or Series in a dictionary with a string key
            df_dict[k] = getattr(df, method)(*args, **kwargs)
            if df_dict[k] is None:
                inplace = True
        if not inplace:
            # ✅ Best Practice: Unpack tuple to separate key and column names
            return SepDataFrame(df_dict=df_dict, join=self.join, skip_align=skip_align)

    def sort_index(self, *args, **kwargs):
        # ✅ Best Practice: Check if key exists in dictionary before accessing it
        return self.apply_each("sort_index", True, *args, **kwargs)

    # ✅ Best Practice: Simplify single-element tuples for easier access
    def copy(self, *args, **kwargs):
        return self.apply_each("copy", True, *args, **kwargs)

    # 🧠 ML Signal: Assigning a DataFrame or Series to a specific column in an existing DataFrame
    def _update_join(self):
        if self.join not in self:
            if len(self._df_dict) > 0:
                # ✅ Best Practice: Handle case where key does not exist in dictionary
                self.join = next(iter(self._df_dict.keys()))
            else:
                # ✅ Best Practice: Simplify single-element tuples for easier access
                # NOTE: this will change the behavior of previous reindex when all the keys are empty
                # ⚠️ SAST Risk (Low): Directly deleting items from a dictionary without checking if the key exists can raise a KeyError.
                self.join = None

    # ✅ Best Practice: Use of dunder method __contains__ for implementing 'in' keyword functionality
    # 🧠 ML Signal: Converting Series to DataFrame with a specific column name
    # ✅ Best Practice: Ensure that dependent methods are called after modifying internal state to maintain consistency.
    def __getitem__(self, item):
        # TODO: behave more like pandas when multiindex
        # 🧠 ML Signal: Checks membership in a dictionary, a common pattern for data structure operations
        # ✅ Best Practice: Implementing __len__ allows the object to be used with len(), improving usability.
        return self._df_dict[item]
    # ✅ Best Practice: Use a copy of the DataFrame to avoid modifying the original

    # 🧠 ML Signal: Accessing a dictionary with a key suggests a pattern of dictionary usage.
    def __setitem__(self, item: str, df: Union[pd.DataFrame, pd.Series]):
        # 🧠 ML Signal: Creating a MultiIndex for DataFrame columns
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which is a placeholder and may cause runtime errors if not implemented.
        # TODO: consider the join behavior
        if not isinstance(item, tuple):
            # 🧠 ML Signal: Storing a modified DataFrame in a dictionary
            self._df_dict[item] = df
        # ✅ Best Practice: Use of @property decorator for defining a property, which is a Pythonic way to use getters.
        else:
            # NOTE: corner case of MultiIndex
            # 🧠 ML Signal: Iterating over dictionary items
            _df_dict_key, *col_name = item
            col_name = tuple(col_name)
            # ✅ Best Practice: Using pd.MultiIndex for hierarchical indexing
            if _df_dict_key in self._df_dict:
                if len(col_name) == 1:
                    col_name = col_name[0]
                # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability
                # ✅ Best Practice: Using pd.concat to combine DataFrames
                self._df_dict[_df_dict_key][col_name] = df
            else:
                # 🧠 ML Signal: Usage of dictionary to store and access multiple DataFrames
                if isinstance(df, pd.Series):
                    if len(col_name) == 1:
                        col_name = col_name[0]
                    self._df_dict[_df_dict_key] = df.to_frame(col_name)
                # ⚠️ SAST Risk (Low): Potential for KeyError if 'join' key is not present in df_dict
                else:
                    df_copy = df.copy()  # avoid changing df
                    # 🧠 ML Signal: Constructor method for initializing class instances
                    df_copy.columns = pd.MultiIndex.from_tuples([(*col_name, *idx) for idx in df.columns.to_list()])
                    self._df_dict[_df_dict_key] = df_copy
    # 🧠 ML Signal: Storing an object as an instance variable

    def __delitem__(self, item: str):
        # ✅ Best Practice: Initialize all instance variables in the constructor
        del self._df_dict[item]
        # ✅ Best Practice: Consider adding type hints for the 'axis' parameter for better readability and maintainability.
        self._update_join()
    # 🧠 ML Signal: Storing a parameter as an instance variable

    # 🧠 ML Signal: Storing a parameter as an instance variable, indicating stateful behavior.
    def __contains__(self, item):
        # 🧠 ML Signal: Method overloading based on argument type
        return item in self._df_dict
    # 🧠 ML Signal: Returning self from a method, indicating a fluent interface pattern.

    # 🧠 ML Signal: Handling string type for indexing
    def __len__(self):
        return len(self._df_dict[self.join])

    # 🧠 ML Signal: Handling tuple or list type for indexing
    def droplevel(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `droplevel` method")
    # ✅ Best Practice: Dictionary comprehension for concise and readable code

    @property
    # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported input types
    def columns(self):
        dfs = []
        for k, df in self._df_dict.items():
            df = df.head(0)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            # ✅ Best Practice: Dictionary comprehension for concise and readable code
            dfs.append(df)
        return pd.concat(dfs, axis=1).columns

    # Useless methods
    @staticmethod
    def merge(df_dict: Dict[str, pd.DataFrame], join: str):
        # 🧠 ML Signal: Handling tuple type for multi-axis indexing
        all_df = df_dict[join]
        for k, df in df_dict.items():
            if k != join:
                all_df = all_df.join(df)
        return all_df


# ✅ Best Practice: Check if instance is of a specific type before further processing
class SDFLoc:
    """Mock Class"""
    # ✅ Best Practice: Check if cls is an Iterable before iterating over it

    def __init__(self, sdf: SepDataFrame, join):
        self._sdf = sdf
        # ✅ Best Practice: Use 'is' for comparing with singleton objects like classes
        self.axis = None
        self.join = join

    # ✅ Best Practice: Use 'is' for comparing with singleton objects like classes
    def __call__(self, axis):
        self.axis = axis
        return self
    # ✅ Best Practice: Fallback to original isinstance function for other cases

    def __getitem__(self, args):
        # ⚠️ SAST Risk (Medium): Overwriting built-in functions can lead to unexpected behavior
        # 🧠 ML Signal: Example of creating a custom DataFrame object
        # 🧠 ML Signal: Example of using isinstance with a tuple of types
        if self.axis == 1:
            if isinstance(args, str):
                return self._sdf[args]
            elif isinstance(args, (tuple, list)):
                new_df_dict = {k: self._sdf[k] for k in args}
                return SepDataFrame(new_df_dict, join=self.join if self.join in args else args[0], skip_align=True)
            else:
                raise NotImplementedError(f"This type of input is not supported")
        elif self.axis == 0:
            return SepDataFrame(
                {k: df.loc(axis=0)[args] for k, df in self._sdf._df_dict.items()}, join=self.join, skip_align=True
            )
        else:
            df = self._sdf
            if isinstance(args, tuple):
                ax0, *ax1 = args
                if len(ax1) == 0:
                    ax1 = None
                if ax1 is not None:
                    df = df.loc(axis=1)[ax1]
                if ax0 is not None:
                    df = df.loc(axis=0)[ax0]
                return df
            else:
                return df.loc(axis=0)[args]


# Patch pandas DataFrame
# Tricking isinstance to accept SepDataFrame as its subclass
import builtins


def _isinstance(instance, cls):
    if isinstance_orig(instance, SepDataFrame):  # pylint: disable=E0602  # noqa: F821
        if isinstance(cls, Iterable):
            for c in cls:
                if c is pd.DataFrame:
                    return True
        elif cls is pd.DataFrame:
            return True
    return isinstance_orig(instance, cls)  # pylint: disable=E0602  # noqa: F821


builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance

if __name__ == "__main__":
    sdf = SepDataFrame({}, join=None)
    print(isinstance(sdf, (pd.DataFrame,)))
    print(isinstance(sdf, pd.DataFrame))