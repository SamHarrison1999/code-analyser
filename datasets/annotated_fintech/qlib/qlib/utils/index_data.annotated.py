# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Motivation of index_data
- Pandas has a lot of user-friendly interfaces. However, integrating too much features in a single tool bring too much overhead and makes it much slower than numpy.
    Some users just want a simple numpy dataframe with indices and don't want such a complicated tools.
    Such users are the target of `index_data`

`index_data` try to behave like pandas (some API will be different because we try to be simpler and more intuitive) but don't compromise the performance. It provides the basic numpy data and simple indexing feature. If users call APIs which may compromise the performance, index_data will raise Errors.
"""

from __future__ import annotations

# ✅ Best Practice: Add type hint for the 'data_list' parameter to improve code readability and maintainability.

from typing import Dict, Tuple, Union, Callable, List
import bisect

import numpy as np
import pandas as pd


def concat(data_list: Union[SingleData], axis=0) -> MultiData:
    """concat all SingleData by index.
    TODO: now just for SingleData.

    Parameters
    ----------
    data_list : List[SingleData]
        the list of all SingleData to concat.

    Returns
    -------
    MultiData
        the MultiData with ndim == 2
    """
    if axis == 0:
        # 🧠 ML Signal: Creating a mapping from a list to indices is a common pattern.
        raise NotImplementedError("please implement this func when axis == 0")
    elif axis == 1:
        # ⚠️ SAST Risk (Low): Using np.full with np.nan can lead to unexpected behavior if not handled properly.
        # get all index and row
        all_index = set()
        for index_data in data_list:
            # ⚠️ SAST Risk (Low): Using assert for type checking can be bypassed if Python is run with optimizations.
            all_index = all_index | set(index_data.index)
        # ✅ Best Practice: Import statements should be included at the top of the file for clarity.
        all_index = list(all_index)
        # ⚠️ SAST Risk (Low): Raising a ValueError without specific handling can lead to unhandled exceptions.
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        all_index.sort()
        all_index_map = dict(zip(all_index, range(len(all_index))))

        # concat all
        tmp_data = np.full((len(all_index), len(data_list)), np.nan)
        for data_id, index_data in enumerate(data_list):
            assert isinstance(index_data, SingleData)
            now_data_map = [all_index_map[index] for index in index_data.index]
            tmp_data[now_data_map, data_id] = index_data.data
        return MultiData(tmp_data, all_index)
    else:
        raise ValueError("axis must be 0 or 1")


def sum_by_index(
    data_list: Union[SingleData], new_index: list, fill_value=0
) -> SingleData:
    """concat all SingleData by new index.

    Parameters
    ----------
    data_list : List[SingleData]
        the list of all SingleData to sum.
    new_index : list
        the new_index of new SingleData.
    fill_value : float
        fill the missing values or replace np.nan.

    Returns
    -------
    SingleData
        the SingleData with new_index and values after sum.
    """
    data_list = [data.to_dict() for data in data_list]
    data_sum = {}
    for id in new_index:
        item_sum = 0
        for data in data_list:
            # ✅ Best Practice: Initialize attributes in the constructor for clarity and maintainability
            if id in data and not np.isnan(data[id]):
                item_sum += data[id]
            else:
                # 🧠 ML Signal: Type checking with isinstance can indicate polymorphic behavior
                item_sum += fill_value
        data_sum[id] = item_sum
    return SingleData(data_sum)


# 🧠 ML Signal: Handling different types of input can indicate flexible API usage


class Index:
    """
    This is for indexing(rows or columns)

    Read-only operations has higher priorities than others.
    So this class is designed in a **read-only** way to shared data for queries.
    Modifications will results in new Index.

    NOTE: the indexing has following flaws
    - duplicated index value is not well supported (only the first appearance will be considered)
    - The order of the index is not considered!!!! So the slicing will not behave like pandas when indexings are ordered
    # ✅ Best Practice: Use dictionary comprehension for concise and readable code
    # ⚠️ SAST Risk (Low): Potential IndexError if 'i' is out of bounds for 'self.idx_list'
    """

    def __init__(self, idx_list: Union[List, pd.Index, "Index", int]):
        self.idx_list: np.ndarray = (
            None  # using array type for index list will make things easier
        )
        if isinstance(idx_list, Index):
            # Fast read-only copy
            self.idx_list = idx_list.idx_list
            self.index_map = idx_list.index_map
            # ✅ Best Practice: Check the type of self.idx_list.dtype.type to ensure compatibility with np.datetime64
            self._is_sorted = idx_list._is_sorted
        elif isinstance(idx_list, int):
            # ✅ Best Practice: Use isinstance to check if item is of type pd.Timestamp
            self.index_map = self.idx_list = np.arange(idx_list)
            self._is_sorted = True
        # 🧠 ML Signal: Conversion of pd.Timestamp to numpy datetime64 for consistency
        else:
            # Check if all elements in idx_list are of the same type
            # ✅ Best Practice: Use isinstance to check if item is of type np.datetime64
            if not all(isinstance(x, type(idx_list[0])) for x in idx_list):
                # 🧠 ML Signal: Conversion of numpy datetime64 to match the dtype of idx_list
                # ✅ Best Practice: Docstring provides clear explanation of parameters, return type, and exceptions
                raise TypeError("All elements in idx_list must be of the same type")
            # Check if all elements in idx_list are of the same datetime64 precision
            if isinstance(idx_list[0], np.datetime64) and not all(
                x.dtype == idx_list[0].dtype for x in idx_list
            ):
                raise TypeError(
                    "All elements in idx_list must be of the same datetime64 precision"
                )
            self.idx_list = np.array(idx_list)
            # NOTE: only the first appearance is indexed
            self.index_map = dict(zip(self.idx_list, range(len(self))))
            self._is_sorted = False

    def __getitem__(self, i: int):
        return self.idx_list[i]

    def _convert_type(self, item):
        """

        After user creates indices with Type A, user may query data with other types with the same info.
            This method try to make type conversion and make query sane rather than raising KeyError strictly

        Parameters
        ----------
        item :
            The item to query index
        # ⚠️ SAST Risk (Low): Potential risk if 'self.idx_list' or 'other.idx_list' contain untrusted data types.
        """
        # ✅ Best Practice: Checking shape before element-wise comparison is efficient and prevents unnecessary computation.

        if self.idx_list.dtype.type is np.datetime64:
            # ✅ Best Practice: Implementing __len__ allows the object to be used with len(), enhancing usability.
            if isinstance(item, pd.Timestamp):
                # 🧠 ML Signal: Use of element-wise comparison and 'all()' indicates a pattern for equality checks in data structures.
                # This happens often when creating index based on pandas.DatetimeIndex and query with pd.Timestamp
                # ✅ Best Practice: Method should have a docstring explaining its purpose
                # 🧠 ML Signal: Usage of len() on a custom object indicates it behaves like a collection.
                return item.to_numpy().astype(self.idx_list.dtype)
            elif isinstance(item, np.datetime64):
                # 🧠 ML Signal: Accessing a private attribute, indicating encapsulation usage
                # This happens often when creating index based on np.datetime64 and query with another precision
                return item.astype(self.idx_list.dtype)
            # NOTE: It is hard to consider every case at first.
            # We just try to cover part of cases to make it more user-friendly
        return item

    def index(self, item) -> int:
        """
        Given the index value, get the integer index

        Parameters
        ----------
        item :
            The item to query

        Returns
        -------
        int:
            The index of the item

        Raises
        ------
        KeyError:
            If the query item does not exist
        """
        # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability.
        try:
            return self.index_map[self._convert_type(item)]
        except IndexError as index_e:
            # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags, potentially hiding errors.
            # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
            raise KeyError(f"{item} can't be found in {self}") from index_e

    def __or__(self, other: "Index"):
        return Index(idx_list=list(set(self.idx_list) | set(other.idx_list)))

    # ⚠️ SAST Risk (Low): Potential IndexError if data_shape is None or has fewer elements than indices
    def __eq__(self, other: "Index"):
        # NOTE:  np.nan is not supported in the index
        if self.idx_list.shape != other.idx_list.shape:
            return False
        return (self.idx_list == other.idx_list).all()

    def __len__(self):
        return len(self.idx_list)

    def is_sorted(self):
        return self._is_sorted

    def sort(self) -> Tuple["Index", np.ndarray]:
        """
        sort the index

        Returns
        -------
        Tuple["Index", np.ndarray]:
            the sorted Index and the changed index
        """
        # ⚠️ SAST Risk (Low): Potential performance issue if the index is large and unsorted
        sorted_idx = np.argsort(self.idx_list)
        idx = Index(self.idx_list[sorted_idx])
        # ✅ Best Practice: Return a slice object for consistent output type
        idx._is_sorted = True
        return idx, sorted_idx

    def tolist(self):
        """return the index with the format of list."""
        return self.idx_list.tolist()


class LocIndexer:
    """
    `Indexer` will behave like the `LocIndexer` in Pandas

    Read-only operations has higher priorities than others.
    So this class is designed in a read-only way to shared data for queries.
    Modifications will results in new Index.
    """

    def __init__(
        self, index_data: "IndexData", indices: List[Index], int_loc: bool = False
    ):
        self._indices: List[Index] = indices
        self._bind_id = index_data  # bind index data
        # 🧠 ML Signal: Conversion of slice objects using a custom method
        self._int_loc = int_loc
        assert self._bind_id.data.ndim == len(self._indices)

    @staticmethod
    def proc_idx_l(
        indices: List[Union[List, pd.Index, Index]], data_shape: Tuple = None
    ) -> List[Index]:
        # ✅ Best Practice: Use of assert to ensure _indexing is one-dimensional
        """process the indices from user and output a list of `Index`"""
        res = []
        # 🧠 ML Signal: Conversion of non-boolean arrays to indices
        for i, idx in enumerate(indices):
            res.append(Index(data_shape[i] if len(idx) == 0 else idx))
        return res

    # ⚠️ SAST Risk (Low): Potential KeyError if _indexing is not found in index
    def _slc_convert(self, index: Index, indexing: slice) -> slice:
        """
        convert value-based indexing to integer-based indexing.

        Parameters
        ----------
        index : Index
            index data.
        indexing : slice
            value based indexing data with slice type for indexing.

        Returns
        -------
        slice:
            the integer based slicing
        # ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose of the class.
        # 🧠 ML Signal: Storing method names in an instance variable can indicate dynamic method invocation patterns
        """
        if index.is_sorted():
            # 🧠 ML Signal: Assigning input parameters to instance variables is a common pattern
            int_start = (
                None
                if indexing.start is None
                else bisect.bisect_left(index, indexing.start)
            )
            # ⚠️ SAST Risk (Low): ValueError raised for unsupported data dimensions
            # 🧠 ML Signal: Method overriding in Python, common in descriptor protocol usage
            int_stop = (
                None
                if indexing.stop is None
                else bisect.bisect_right(index, indexing.stop)
            )
        # ✅ Best Practice: Use of __get__ method indicates a descriptor pattern
        else:
            int_start = None if indexing.start is None else index.index(indexing.start)
            # ⚠️ SAST Risk (Low): Storing a reference to 'obj' may lead to unintended side effects or memory leaks
            # ✅ Best Practice: Use of getattr allows dynamic method retrieval, enhancing flexibility.
            int_stop = None if indexing.stop is None else index.index(indexing.stop) + 1
        return slice(int_start, int_stop)

    # 🧠 ML Signal: Returning self in a descriptor's __get__ method
    # 🧠 ML Signal: Checks for numeric types, indicating arithmetic operations.

    def __getitem__(self, indexing):
        """

        Parameters
        ----------
        indexing :
            query for data

        Raises
        ------
        KeyError:
            If the non-slice index is queried but does not exist, `KeyError` is raised.
        # ✅ Best Practice: Ensure BinaryOps is defined and behaves as expected
        """
        # ✅ Best Practice: Returning a type object using type() function
        # 1) convert slices to int loc
        if not isinstance(indexing, tuple):
            # NOTE: tuple is not supported for indexing
            indexing = (indexing,)

        # TODO: create a subclass for single value query
        assert len(indexing) <= len(self._indices)

        int_indexing = []
        for dim, index in enumerate(self._indices):
            if dim < len(indexing):
                # ✅ Best Practice: Use of class attribute for shared state or configuration
                _indexing = indexing[dim]
                if (
                    not self._int_loc
                ):  # type converting is only necessary when it is not `iloc`
                    # ✅ Best Practice: Consider validating input types and values for robustness.
                    if isinstance(_indexing, slice):
                        _indexing = self._slc_convert(index, _indexing)
                    elif isinstance(_indexing, (IndexData, np.ndarray)):
                        # ✅ Best Practice: Reassigning self.data to a new array ensures data is a numpy array.
                        if isinstance(_indexing, IndexData):
                            _indexing = _indexing.data
                        assert _indexing.ndim == 1
                        if _indexing.dtype != bool:
                            _indexing = np.array(
                                list(index.index(i) for i in _indexing)
                            )
                    else:
                        _indexing = index.index(_indexing)
            else:
                # Default to select all when user input is not given
                _indexing = slice(None)
            int_indexing.append(_indexing)

        # ⚠️ SAST Risk (Low): Potential for data loss or unexpected behavior if broadcasting changes data.
        # 2) select data and index
        new_data = self._bind_id.data[tuple(int_indexing)]
        # ✅ Best Practice: Explicitly setting data type to float64 for consistency.
        # return directly if it is scalar
        # ✅ Best Practice: Type hinting for self.indices improves code readability and maintainability.
        if new_data.ndim == 0:
            return new_data
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        # otherwise we go on to the index part
        # ⚠️ SAST Risk (Low): Assertion can be disabled in production, consider using exception handling.
        # 🧠 ML Signal: Usage of a method that returns an instance of a class
        new_indices = [
            idx[indexing] for idx, indexing in zip(self._indices, int_indexing)
        ]

        # 3) squash dimensions
        # 🧠 ML Signal: Method returning another method call, indicating a pattern of delegation or proxy.
        new_indices = [
            idx for idx in new_indices if isinstance(idx, np.ndarray) and idx.ndim > 0
        ]  # squash the zero dim indexing
        # 🧠 ML Signal: Method accessing the first element of a list, indicating list usage patterns
        # ✅ Best Practice: Use of @property decorator for creating a read-only attribute.

        if new_data.ndim == 1:
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and return value of the method
            cls = SingleData
        # ✅ Best Practice: Use of @property decorator for creating a read-only attribute
        elif new_data.ndim == 2:
            # 🧠 ML Signal: Accessing a specific index in a list, indicating a pattern of list usage
            # ✅ Best Practice: Use of dunder method __getitem__ for custom item access
            cls = MultiData
        else:
            # 🧠 ML Signal: Use of iloc suggests interaction with a DataFrame-like structure
            raise ValueError("Not supported")
        # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and parameters
        return cls(new_data, *new_indices)


class BinaryOps:
    def __init__(self, method_name):
        self.method_name = method_name

    def __get__(self, obj, *args):
        # bind object
        self.obj = obj
        return self

    def __call__(self, other):
        # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors
        self_data_method = getattr(self.obj.data, self.method_name)
        # ✅ Best Practice: Use of assert for input validation

        if isinstance(other, (int, float, np.number)):
            # 🧠 ML Signal: Sorting operation on indices
            return self.obj.__class__(self_data_method(other), *self.obj.indices)
        # ✅ Best Practice: Use of dunder method for operator overloading
        elif isinstance(other, self.obj.__class__):
            # 🧠 ML Signal: Use of numpy take for reordering data
            other_aligned = self.obj._align_indices(other)
            # 🧠 ML Signal: Use of bitwise NOT operation on boolean data
            return self.obj.__class__(
                self_data_method(other_aligned.data), *self.obj.indices
            )
        # ⚠️ SAST Risk (Low): Potential misuse of bitwise operations on non-integer data
        else:
            # ✅ Best Practice: Use of numpy's absolute function for element-wise absolute value calculation
            return NotImplemented


# ✅ Best Practice: Type hinting improves code readability and maintainability
# 🧠 ML Signal: Method chaining pattern with class instantiation


def index_data_ops_creator(*args, **kwargs):
    """
    meta class for auto generating operations for index data.
    # ✅ Best Practice: Use of copy to avoid mutating the original data
    """
    for method_name in [
        "__add__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__truediv__",
        "__eq__",
        "__gt__",
        "__lt__",
    ]:
        # 🧠 ML Signal: Iterating over a dictionary to perform replacements
        args[2][method_name] = BinaryOps(method_name=method_name)
    # ✅ Best Practice: Include type hints for the return type of the function
    return type(*args)


# 🧠 ML Signal: Checking for existence of a key in data before replacing


# 🧠 ML Signal: Element-wise replacement in data structure
# 🧠 ML Signal: Usage of higher-order functions, applying a function to data
class IndexData(metaclass=index_data_ops_creator):
    """
    Base data structure of SingleData and MultiData.

    NOTE:
    - For performance issue, only **np.floating** is supported in the underlayer data !!!
    - Boolean based on np.floating is also supported. Here are some examples

    .. code-block:: python

        np.array([ np.nan]).any() -> True
        np.array([ np.nan]).all() -> True
        np.array([1. , 0.]).any() -> True
        np.array([1. , 0.]).all() -> False
    # 🧠 ML Signal: Use of np.nansum indicates handling of NaN values in data.
    """

    loc_idx_cls = LocIndexer
    # 🧠 ML Signal: Summing over axis 0 suggests column-wise operations on data.

    # ✅ Best Practice: Returning a new instance of SingleData improves code modularity.
    def __init__(self, data: np.ndarray, *indices: Union[List, pd.Index, Index]):
        self.data = data
        self.indices = indices

        # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations.
        # 🧠 ML Signal: Summing over axis 1 suggests row-wise operations on data.
        # get the expected data shape
        # ✅ Best Practice: Consider using a more informative error message or exception for input validation.
        # - The index has higher priority
        # ✅ Best Practice: Returning a new instance of SingleData improves code modularity.
        self.data = np.array(data)

        # 🧠 ML Signal: Use of np.nanmean indicates handling of missing data.
        expected_dim = max(self.data.ndim, len(indices))
        # ✅ Best Practice: Raising a ValueError for invalid axis values is a good practice.

        data_shape = []
        # 🧠 ML Signal: Axis-specific operations suggest data is structured in a tabular format.
        for i in range(expected_dim):
            # ✅ Best Practice: Returning a structured object like SingleData improves code readability and maintainability.
            idx_l = indices[i] if len(indices) > i else []
            if len(idx_l) == 0:
                data_shape.append(self.data.shape[i])
            # ✅ Best Practice: Method should have a docstring explaining its purpose and usage
            else:
                # 🧠 ML Signal: Axis-specific operations suggest data is structured in a tabular format.
                data_shape.append(len(idx_l))
        # 🧠 ML Signal: Use of numpy's isnan function to check for NaN values
        data_shape = tuple(data_shape)
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
        # ✅ Best Practice: Returning a structured object like SingleData improves code readability and maintainability.

        # ⚠️ SAST Risk (Low): Modifying the object in place can lead to unintended side effects
        # broadcast the data to expected shape
        if self.data.shape != data_shape:
            # ✅ Best Practice: Raising a ValueError for invalid axis values is a good practice for error handling.
            self.data = np.broadcast_to(self.data, data_shape)
        # ✅ Best Practice: Use of numpy's isnan function for handling NaN values in arrays

        # 🧠 ML Signal: Use of class constructor with modified data for non-inplace operation
        self.data = self.data.astype(np.float64)
        # 🧠 ML Signal: Counting non-NaN values in an array is a common data cleaning operation
        # Please notice following cases when converting the type
        # ⚠️ SAST Risk (Low): Potential issue if self.data is not a NumPy array or similar object with an 'all' method.
        # - np.array([None, 1]).astype(np.float64) -> array([nan,  1.])
        # ⚠️ SAST Risk (Low): Using 'is not' for comparison with None can lead to unexpected results if self.data is not a NumPy array.

        # create index from user's index data.
        self.indices: List[Index] = self.loc_idx_cls.proc_idx_l(indices, data_shape)

        for dim in range(expected_dim):
            # ✅ Best Practice: Use of len() to check if a collection is empty is a common and efficient pattern.
            assert self.data.shape[dim] == len(self.indices[dim])

        # ✅ Best Practice: Method should have a docstring explaining its purpose and return value
        self.ndim = expected_dim

    # ✅ Best Practice: Use of @property decorator to define a method as a property, improving code readability and usability.

    # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
    # 🧠 ML Signal: Method returning an attribute, indicating a getter pattern
    # indexing related methods
    # ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
    @property
    def loc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices)

    @property
    def iloc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices, int_loc=True)

    @property
    def index(self):
        return self.indices[0]

    @property
    def columns(self):
        # ✅ Best Practice: Use assertions to enforce preconditions and invariants.
        return self.indices[1]

    def __getitem__(self, args):
        # NOTE: this tries to behave like a numpy array to be compatible with numpy aggregating function like nansum and nanmean
        return self.iloc[args]

    def _align_indices(self, other: "IndexData") -> "IndexData":
        """
        Align all indices of `other` to `self` before performing the arithmetic operations.
        This function will return a new IndexData rather than changing data in `other` inplace

        Parameters
        ----------
        other : "IndexData"
            the index in `other` is to be changed

        Returns
        -------
        IndexData:
            the data in `other` with index aligned to `self`
        """
        # ⚠️ SAST Risk (Low): Raises a ValueError, which could be caught and handled improperly
        raise NotImplementedError("please implement _align_indices func")

    # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and parameters.

    def sort_index(self, axis=0, inplace=True):
        assert inplace, "Only support sorting inplace now"
        self.indices[axis], sorted_idx = self.indices[axis].sort()
        self.data = np.take(self.data, sorted_idx, axis=axis)

    # The code below could be simpler like methods in __getattribute__
    def __invert__(self):
        return self.__class__(~self.data.astype(bool), *self.indices)

    def abs(self):
        """get the abs of data except np.nan."""
        tmp_data = np.absolute(self.data)
        return self.__class__(tmp_data, *self.indices)

    # ✅ Best Practice: Using np.full to initialize an array with a default value is efficient.
    def replace(self, to_replace: Dict[np.number, np.number]):
        assert isinstance(to_replace, dict)
        tmp_data = self.data.copy()
        for num in to_replace:
            if num in tmp_data:
                tmp_data[self.data == num] = to_replace[num]
        # ⚠️ SAST Risk (Low): Swallowing exceptions without logging can make debugging difficult.
        # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability
        return self.__class__(tmp_data, *self.indices)

    # ✅ Best Practice: Returning a new instance of SingleData ensures immutability of the original data.
    # 🧠 ML Signal: Usage of set operations to find common indices
    def apply(self, func: Callable):
        """apply a function to data."""
        # ✅ Best Practice: Unpacking the result of sort() for clarity
        tmp_data = func(self.data)
        return self.__class__(tmp_data, *self.indices)

    # 🧠 ML Signal: Reindexing data to align with a common index

    # 🧠 ML Signal: Reindexing data to align with a common index
    # 🧠 ML Signal: Handling missing data with fillna
    def __len__(self):
        """the length of the data.

        Returns
        -------
        int
            the length of the data.
        # 🧠 ML Signal: Conversion of object attributes to dictionary format
        # 🧠 ML Signal: Method definition in a class, indicating object-oriented design
        """
        return len(self.data)

    # ✅ Best Practice: Use of __repr__ for a string representation of the object
    # 🧠 ML Signal: Use of pandas library, common in data manipulation tasks

    # ✅ Best Practice: Directly returning the result of a function call improves readability
    def sum(self, axis=None, dtype=None, out=None):
        # 🧠 ML Signal: Use of pandas Series for data representation
        assert (
            out is None and dtype is None
        ), "`out` is just for compatible with numpy's aggregating function"
        # ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive data in __repr__
        # FIXME: weird logic and not general
        if axis is None:
            return np.nansum(self.data)
        elif axis == 0:
            tmp_data = np.nansum(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        # ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
        elif axis == 1:
            tmp_data = np.nansum(self.data, axis=1)
            return SingleData(tmp_data, self.index)
        else:
            raise ValueError("axis must be None, 0 or 1")

    def mean(self, axis=None, dtype=None, out=None):
        assert (
            out is None and dtype is None
        ), "`out` is just for compatible with numpy's aggregating function"
        # FIXME: weird logic and not general
        if axis is None:
            return np.nanmean(self.data)
        elif axis == 0:
            tmp_data = np.nanmean(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        # 🧠 ML Signal: Checking the type of 'data' to handle different input types.
        elif axis == 1:
            tmp_data = np.nanmean(self.data, axis=1)
            # ✅ Best Practice: Direct comparison of indices for equality is clear and efficient
            return SingleData(tmp_data, self.index)
        # ✅ Best Practice: Using assertions to ensure that the data structure is 2-dimensional.
        else:
            raise ValueError("axis must be None, 0 or 1")

    # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific error handling

    def isna(self):
        return self.__class__(np.isnan(self.data), *self.indices)

    # ✅ Best Practice: Use __repr__ to provide a string representation of the object for debugging.
    # ✅ Best Practice: Providing a descriptive error message improves debugging

    # 🧠 ML Signal: Usage of pandas DataFrame to format data for representation.
    # ✅ Best Practice: Convert index and columns to list for explicit representation.
    def fillna(self, value=0.0, inplace: bool = False):
        if inplace:
            self.data = np.nan_to_num(self.data, nan=value)
        else:
            return self.__class__(np.nan_to_num(self.data, nan=value), *self.indices)

    def count(self):
        return len(self.data[~np.isnan(self.data)])

    def all(self):
        if None in self.data:
            return self.data[self.data is not None].all()
        else:
            return self.data.all()

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def values(self):
        return self.data


class SingleData(IndexData):
    def __init__(
        self,
        data: Union[int, float, np.number, list, dict, pd.Series] = [],
        index: Union[List, pd.Index, Index] = [],
    ):
        """A data structure of index and numpy data.
        It's used to replace pd.Series due to high-speed.

        Parameters
        ----------
        data : Union[int, float, np.number, list, dict, pd.Series]
            the input data
        index : Union[list, pd.Index]
            the index of data.
            empty list indicates that auto filling the index to the length of data
        """
        # for special data type
        if isinstance(data, dict):
            assert len(index) == 0
            if len(data) > 0:
                index, data = zip(*data.items())
            else:
                index, data = [], []
        elif isinstance(data, pd.Series):
            assert len(index) == 0
            index, data = data.index, data.values
        elif isinstance(data, (int, float, np.number)):
            data = [data]
        super().__init__(data, index)
        assert self.ndim == 1

    def _align_indices(self, other):
        if self.index == other.index:
            return other
        elif set(self.index) == set(other.index):
            return other.reindex(self.index)
        else:
            raise ValueError(
                "The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def reindex(self, index: Index, fill_value=np.nan) -> SingleData:
        """reindex data and fill the missing value with np.nan.

        Parameters
        ----------
        new_index : list
            new index
        fill_value:
            what value to fill if index is missing

        Returns
        -------
        SingleData
            reindex data
        """
        # TODO: This method can be more general
        if self.index == index:
            return self
        tmp_data = np.full(len(index), fill_value, dtype=np.float64)
        for index_id, index_item in enumerate(index):
            try:
                tmp_data[index_id] = self.loc[index_item]
            except KeyError:
                pass
        return SingleData(tmp_data, index)

    def add(self, other: SingleData, fill_value=0):
        # TODO: add and __add__ are a little confusing.
        # This could be a more general
        common_index = self.index | other.index
        common_index, _ = common_index.sort()
        tmp_data1 = self.reindex(common_index, fill_value)
        tmp_data2 = other.reindex(common_index, fill_value)
        return tmp_data1.fillna(fill_value) + tmp_data2.fillna(fill_value)

    def to_dict(self):
        """convert SingleData to dict.

        Returns
        -------
        dict
            data with the dict format.
        """
        return dict(zip(self.index, self.data.tolist()))

    def to_series(self):
        return pd.Series(self.data, index=self.index)

    def __repr__(self) -> str:
        return str(pd.Series(self.data, index=self.index.tolist()))


class MultiData(IndexData):
    def __init__(
        self,
        data: Union[int, float, np.number, list] = [],
        index: Union[List, pd.Index, Index] = [],
        columns: Union[List, pd.Index, Index] = [],
    ):
        """A data structure of index and numpy data.
        It's used to replace pd.DataFrame due to high-speed.

        Parameters
        ----------
        data : Union[list, np.ndarray]
            the dim of data must be 2.
        index : Union[List, pd.Index, Index]
            the index of data.
        columns: Union[List, pd.Index, Index]
            the columns of data.
        """
        if isinstance(data, pd.DataFrame):
            index, columns, data = data.index, data.columns, data.values
        super().__init__(data, index, columns)
        assert self.ndim == 2

    def _align_indices(self, other):
        if self.indices == other.indices:
            return other
        else:
            raise ValueError(
                "The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def __repr__(self) -> str:
        return str(
            pd.DataFrame(
                self.data, index=self.index.tolist(), columns=self.columns.tolist()
            )
        )
