# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List

import numpy as np
import pandas as pd

# 🧠 ML Signal: Logging usage pattern can be used to train models to identify logging practices
from qlib.utils.time import Freq
# ✅ Best Practice: Using a logger instead of print statements for logging is a best practice
from qlib.utils.resam import resam_calendar
from qlib.config import C
# 🧠 ML Signal: Importing specific classes and functions can indicate usage patterns for ML models
# ✅ Best Practice: Use of @property decorator for getter method
# ✅ Best Practice: Use of f-string for string formatting
from qlib.data.cache import H
from qlib.log import get_module_logger
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT

# ✅ Best Practice: Use of a property method to encapsulate access to the provider_uri attribute
logger = get_module_logger("file_storage")
# ⚠️ SAST Risk (Medium): Potential file path manipulation vulnerability
# ✅ Best Practice: Use of a conditional expression for concise logic


class FileStorageMixin:
    """FileStorageMixin, applicable to FileXXXStorage
    Subclasses need to have provider_uri, freq, storage_name, file_name attributes

    """
    # ⚠️ SAST Risk (Medium): Potential file path manipulation vulnerability

    # NOTE: provider_uri priority:
    # ✅ Best Practice: Use of 'with' statement for file operations
    #   1. self._provider_uri : if provider_uri is provided.
    # ✅ Best Practice: Use of a descriptive variable name for caching attribute
    #   2. provider_uri in qlib.config.C

    # ✅ Best Practice: Check if an attribute exists before accessing it
    @property
    def provider_uri(self):
        # ✅ Best Practice: Use of getattr to access an attribute dynamically
        # ⚠️ SAST Risk (Medium): Potential file path manipulation vulnerability
        # ✅ Best Practice: Use of constants for configuration values
        return C["provider_uri"] if getattr(self, "_provider_uri", None) is None else self._provider_uri

    @property
    def dpm(self):
        return (
            # ✅ Best Practice: Use of filter and map for functional programming
            # ✅ Best Practice: Informative logging for non-existent file
            C.dpm
            # ✅ Best Practice: Use of lambda for inline function definition
            if getattr(self, "_provider_uri", None) is None
            else C.DataPathManager(self._provider_uri, C.mount_path)
        # ✅ Best Practice: Use of map for transforming data
        )

    @property
    # 🧠 ML Signal: Method returns a Path object, indicating file or directory path handling
    def support_freq(self) -> List[str]:
        # ✅ Best Practice: Directly using keys from a dictionary
        _v = "_support_freq"
        # ⚠️ SAST Risk (Low): Potential for ValueError to be raised, ensure proper handling where this method is called
        if hasattr(self, _v):
            # 🧠 ML Signal: Conversion of items to a specific class (Freq) indicates a pattern of data transformation
            return getattr(self, _v)
        # ✅ Best Practice: Caching the result in an instance attribute
        # ✅ Best Practice: Use of joinpath for constructing file paths is preferred for readability and OS compatibility
        if len(self.provider_uri) == 1 and C.DEFAULT_FREQ in self.provider_uri:
            freq_l = filter(
                lambda _freq: not _freq.endswith("_future"),
                map(lambda x: x.stem, self.dpm.get_data_uri(C.DEFAULT_FREQ).joinpath("calendars").glob("*.txt")),
            )
        # ⚠️ SAST Risk (Low): Potential information disclosure in error message
        else:
            freq_l = self.provider_uri.keys()
        # ⚠️ SAST Risk (Low): Potential information disclosure in error message
        freq_l = [Freq(freq) for freq in freq_l]
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
        setattr(self, _v, freq_l)
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        return freq_l

    # 🧠 ML Signal: Storing a boolean value that might affect behavior or decision-making.
    @property
    def uri(self) -> Path:
        # 🧠 ML Signal: Conditional assignment based on input, indicating a potential configuration pattern.
        if self.freq not in self.support_freq:
            raise ValueError(f"{self.storage_name}: {self.provider_uri} does not contain data for {self.freq}")
        # 🧠 ML Signal: Boolean flag that might be used to toggle functionality.
        # ✅ Best Practice: Use of f-string for string formatting
        return self.dpm.get_data_uri(self.freq).joinpath(f"{self.storage_name}s", self.file_name)
    # ⚠️ SAST Risk (Low): Accessing a global configuration object, which might be modified elsewhere.
    # 🧠 ML Signal: Conditional logic in return statement

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        # 🧠 ML Signal: Checking membership in a list or set, indicating validation of input against allowed values.
        if not self.uri.exists():
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")
# 🧠 ML Signal: Use of a method to get a recent frequency, indicating a fallback mechanism.


# ⚠️ SAST Risk (Low): Raising a ValueError with potentially sensitive information about internal state.
# ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
class FileCalendarStorage(FileStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        # ⚠️ SAST Risk (Low): Potential issue if self.uri is user-controlled, leading to path traversal or file access vulnerabilities
        super(FileCalendarStorage, self).__init__(freq, future, **kwargs)
        # ✅ Best Practice: Caching the result to avoid redundant calculations and improve performance.
        self.future = future
        # 🧠 ML Signal: Checking for file existence before attempting to read or write
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.enable_read_cache = True  # TODO: make it configurable
        # ⚠️ SAST Risk (Low): Opening files without specifying encoding can lead to issues on different systems
        self.region = C["region"]

    @property
    # 🧠 ML Signal: Reading lines from a file and processing them
    def file_name(self) -> str:
        # ✅ Best Practice: Type hint for 'values' parameter improves code readability and maintainability
        return f"{self._freq_file}_future.txt" if self.future else f"{self._freq_file}.txt".lower()
    # ✅ Best Practice: Default parameter 'mode' allows flexibility in file operation
    # 🧠 ML Signal: Stripping whitespace from lines before processing

    # ⚠️ SAST Risk (Low): Ensure 'self.uri' is a trusted source to prevent path traversal vulnerabilities
    @property
    def _freq_file(self) -> str:
        # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability
        # 🧠 ML Signal: Filtering out empty lines from the result
        """the freq to read from file"""
        # ⚠️ SAST Risk (Low): Ensure 'values' is sanitized to prevent injection attacks in file writing
        # ✅ Best Practice: Specifying 'fmt' and 'encoding' ensures consistent data formatting and encoding
        # 🧠 ML Signal: Usage of method chaining pattern
        if not hasattr(self, "_freq_file_cache"):
            freq = Freq(self.freq)
            # ⚠️ SAST Risk (Low): Potential risk if `get_data_uri` returns an unexpected type
            if freq not in self.support_freq:
                # ✅ Best Practice: Ensure that the method name is descriptive and follows naming conventions.
                # NOTE: uri
                #   1. If `uri` does not exist
                # 🧠 ML Signal: Usage of caching mechanism with a conditional check.
                #       - Get the `min_uri` of the closest `freq` under the same "directory" as the `uri`
                #       - Read data from `min_uri` and resample to `freq`
                # 🧠 ML Signal: Concatenation of strings to form a cache key.

                # ⚠️ SAST Risk (Low): Potential risk of key collision in the cache dictionary.
                freq = Freq.get_recent_freq(freq, self.support_freq)
                if freq is None:
                    raise ValueError(f"can't find a freq from {self.support_freq} that can resample to {self.freq}!")
            # 🧠 ML Signal: Caching the result of a function call.
            self._freq_file_cache = freq
        return self._freq_file_cache

    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # 🧠 ML Signal: Conditional logic based on frequency comparison.
        # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
        # 🧠 ML Signal: Use of resampling function with multiple parameters.
        # 🧠 ML Signal: Use of sorted and set indicates a pattern of deduplication and ordering.
        # Currently, it is not supported for the txt-based calendar
        # 🧠 ML Signal: Use of lambda function for inline operations.

        # ⚠️ SAST Risk (Low): Potential for large memory usage if the directory contains many files.
        # 🧠 ML Signal: Usage of a private method indicates encapsulation and abstraction patterns
        if not self.uri.exists():
            # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
            # ✅ Best Practice: Using a private method suggests that _write_calendar is intended for internal use only
            self._write_calendar(values=[])

        # 🧠 ML Signal: Method call with specific argument pattern (empty list) could indicate a reset or clear operation
        # ✅ Best Practice: Ensure the method is checking preconditions before proceeding with main logic
        with self.uri.open("r") as fp:
            res = []
            # ✅ Best Practice: Reading from a calendar source, consider caching if this is a frequent operation
            for line in fp.readlines():
                # ✅ Best Practice: Type hinting for parameters improves code readability and maintainability
                line = line.strip()
                # ⚠️ SAST Risk (Low): Assumes that np.argwhere will always find the value, which may lead to IndexError
                if len(line) > 0:
                    # 🧠 ML Signal: Reading from a data source before modification is a common pattern
                    # 🧠 ML Signal: Use of numpy for array operations, indicating numerical or data processing tasks
                    res.append(line)
            return res
    # 🧠 ML Signal: Use of numpy for array manipulation is a common pattern

    # 🧠 ML Signal: Method for removing an item from a collection
    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        # 🧠 ML Signal: Writing back to a data source after modification is a common pattern
        with self.uri.open(mode=mode) as fp:
            # 🧠 ML Signal: Pattern of checking preconditions before performing operations
            np.savetxt(fp, values, fmt="%s", encoding="utf-8")

    # 🧠 ML Signal: Usage of index finding in a collection
    @property
    # ✅ Best Practice: Type hints are used for function parameters and return type
    def uri(self) -> Path:
        # 🧠 ML Signal: Reading data from a source before modification
        return self.dpm.get_data_uri(self._freq_file).joinpath(f"{self.storage_name}s", self.file_name)
    # 🧠 ML Signal: Method name suggests this is a special method for item assignment

    # ⚠️ SAST Risk (Low): Potential for IndexError if index is out of bounds
    # 🧠 ML Signal: Use of Union type hint indicates handling of multiple input types
    @property
    # 🧠 ML Signal: Usage of numpy for array manipulation
    def data(self) -> List[CalVT]:
        # ✅ Best Practice: Ensure the object is in a valid state before performing operations
        # 🧠 ML Signal: Direct assignment to a data structure, indicating in-place modification
        self.check()
        # 🧠 ML Signal: Writing data back to a source after modification
        # If cache is enabled, then return cache directly
        # 🧠 ML Signal: Method call pattern for persisting changes
        # 🧠 ML Signal: Reading data from a source before modification
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            # 🧠 ML Signal: Using numpy to delete elements from an array
            if key not in H["c"]:
                # ✅ Best Practice: Ensure the method is type hinted for better readability and maintainability.
                H["c"][key] = self._read_calendar()
            # 🧠 ML Signal: Writing data back to a source after modification
            _calendar = H["c"][key]
        # ✅ Best Practice: Use of __len__ method to define object length
        # 🧠 ML Signal: Method call pattern before accessing an internal method.
        else:
            _calendar = self._read_calendar()
        # 🧠 ML Signal: Accessing an attribute of the object
        # ✅ Best Practice: Class definition with clear inheritance for code organization and reuse
        # 🧠 ML Signal: Access pattern using indexing or slicing on the result of a method call.
        if Freq(self._freq_file) != Freq(self.freq):
            _calendar = resam_calendar(
                # ✅ Best Practice: Constants are defined for easy configuration and readability
                np.array(list(map(pd.Timestamp, _calendar))), self._freq_file, self.freq, self.region
            )
        return _calendar

    # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
    def _get_storage_freq(self) -> List[str]:
        # ✅ Best Practice: Default mutable arguments should be avoided; using None as a default value is a safer pattern.
        return sorted(set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt"))))

    # ✅ Best Practice: Use of conditional expression for concise assignment.
    def extend(self, values: Iterable[CalVT]) -> None:
        # ⚠️ SAST Risk (Low): Potential issue if self.uri is user-controlled, leading to path traversal or file access vulnerabilities
        self._write_calendar(values, mode="ab")
    # 🧠 ML Signal: Use of string formatting to create file names based on input parameters.

    def clear(self) -> None:
        # 🧠 ML Signal: Usage of pandas to read CSV files, indicating data processing patterns
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        # 🧠 ML Signal: Iterating over DataFrame rows, common pattern in data processing
        self._write_calendar(values=calendar)

    # ✅ Best Practice: Check for empty data before proceeding with operations
    # ✅ Best Practice: Using setdefault to handle dictionary entries efficiently
    def remove(self, value: CalVT) -> None:
        self.check()
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    # 🧠 ML Signal: Usage of pandas DataFrame for data manipulation
    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        calendar = self._read_calendar()
        # 🧠 ML Signal: Assigning a constant value to a DataFrame column
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        # 🧠 ML Signal: Concatenating multiple DataFrames
        self.check()
        # ⚠️ SAST Risk (Low): Potential data overwrite if self.uri is not handled properly
        calendar = self._read_calendar()
        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)
    # 🧠 ML Signal: Method call with empty dictionary as argument, indicating a reset or clear operation

    # ⚠️ SAST Risk (Low): Potential data overwrite if self.uri is not handled properly
    # 🧠 ML Signal: Method signature with type hints indicating expected input and output types
    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        # ✅ Best Practice: Use of @property decorator for creating a read-only attribute
        self.check()
        # ✅ Best Practice: Ensure that preconditions are met before proceeding with the main logic
        # ✅ Best Practice: Type hints for parameters and return value improve code readability and maintainability
        return self._read_calendar()[i]

    # 🧠 ML Signal: Returning a method call result, indicating a pattern of delegation or encapsulation
    # 🧠 ML Signal: Reading from a private method, indicating encapsulation and internal state management
    def __len__(self) -> int:
        return len(self.data)
# 🧠 ML Signal: Dictionary-like item assignment pattern

# ✅ Best Practice: Ensure the object is in a valid state before performing operations

# 🧠 ML Signal: Writing to a private method, indicating encapsulation and internal state management
class FileInstrumentStorage(FileStorageMixin, InstrumentStorage):
    # 🧠 ML Signal: Reading an instrument before modifying it
    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"
    # 🧠 ML Signal: Deleting an item from a dictionary-like structure
    INSTRUMENT_END_FIELD = "end_datetime"
    # ✅ Best Practice: Type hints for parameters and return values improve code readability and maintainability.
    SYMBOL_FIELD_NAME = "instrument"
    # 🧠 ML Signal: Writing back the modified instrument

    # 🧠 ML Signal: Method call before accessing a dictionary, indicating a pattern of validation or pre-processing.
    def __init__(self, market: str, freq: str, provider_uri: dict = None, **kwargs):
        # ✅ Best Practice: Check for the number of arguments to prevent unexpected behavior.
        super(FileInstrumentStorage, self).__init__(market, freq, **kwargs)
        # ⚠️ SAST Risk (Low): Potential KeyError if 'k' is not present in the dictionary returned by _read_instrument().
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        # ⚠️ SAST Risk (Low): Error message may expose internal logic details.
        self.file_name = f"{market.lower()}.txt"

    # 🧠 ML Signal: Reading from an instrument, indicating interaction with external systems.
    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        if not self.uri.exists():
            self._write_instrument()

        # ✅ Best Practice: Check if 'other' is a Mapping for safe key-value access.
        _instruments = dict()
        df = pd.read_csv(
            self.uri,
            sep="\t",
            # ✅ Best Practice: Check for 'keys' attribute to handle dictionary-like objects.
            usecols=[0, 1, 2],
            names=[self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
            dtype={self.SYMBOL_FIELD_NAME: str},
            parse_dates=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
        )
        # ✅ Best Practice: Type hinting for return value improves code readability and maintainability
        # ✅ Best Practice: Fallback to iterable unpacking for key-value pairs.
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))
        # 🧠 ML Signal: Usage of __len__ method indicates implementation of a container-like class
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
        return _instruments
    # ✅ Best Practice: Use kwargs to update dictionary, allowing flexible key-value updates.

    # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
    def _write_instrument(self, data: Dict[InstKT, InstVT] = None) -> None:
        if not data:
            # 🧠 ML Signal: Writing to an instrument, indicating interaction with external systems.
            # ✅ Best Practice: Using a conditional expression to handle None values improves code readability.
            with self.uri.open("w") as _:
                pass
            # 🧠 ML Signal: Consistent file naming pattern based on input parameters.
            # ⚠️ SAST Risk (Low): Opening a file in write-binary mode without writing anything may lead to data loss.
            return

        res = []
        # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
        for inst, v_list in data.items():
            _df = pd.DataFrame(v_list, columns=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD])
            # 🧠 ML Signal: Method returning a slice of the object, indicating potential use of custom data structures.
            _df[self.SYMBOL_FIELD_NAME] = inst
            # ✅ Best Practice: Check for empty data_array to avoid unnecessary operations
            res.append(_df)

        df = pd.concat(res, sort=False)
        df.loc[:, [self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD]].to_csv(
            self.uri, header=False, sep=self.INSTRUMENT_SEP, index=False
        )
        df.to_csv(self.uri, sep="\t", encoding="utf-8", header=False, index=False)
    # ⚠️ SAST Risk (Low): Assumes self.uri is a valid path object with an exists method

    def clear(self) -> None:
        # ✅ Best Practice: Initialize index if not provided
        self._write_instrument(data={})

    @property
    # 🧠 ML Signal: Use of numpy for data manipulation
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        # ✅ Best Practice: Handle case where index is None or greater than end_index
        return self._read_instrument()

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        # 🧠 ML Signal: Use of numpy for data manipulation
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        self.check()
        # 🧠 ML Signal: Use of numpy for data manipulation
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)
    # 🧠 ML Signal: Use of pandas for data manipulation

    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        # ⚠️ SAST Risk (Low): Potential file handling issue if self.uri is not a valid file path or if file access is restricted.
        return self._read_instrument()[k]

    # 🧠 ML Signal: Use of pandas for data manipulation
    def update(self, *args, **kwargs) -> None:
        # ⚠️ SAST Risk (Low): Opening a file without exception handling can lead to unhandled exceptions if the file cannot be opened.
        if len(args) > 1:
            # ⚠️ SAST Risk (Low): Assumes the file contains at least 4 bytes; otherwise, this could raise an error.
            # 🧠 ML Signal: Use of pandas for data manipulation
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        # ✅ Best Practice: Specify the return type using Union for clarity and type checking
        # ⚠️ SAST Risk (Low): Assumes the file content can be interpreted as a float, which might not always be the case.
        if args:
            # 🧠 ML Signal: Use of pandas for data manipulation
            other = args[0]  # type: dict
            # ⚠️ SAST Risk (Low): Potential for NoneType attribute access if self.uri is None
            if isinstance(other, Mapping):
                # ✅ Best Practice: Use @property decorator to define a method as a property, improving code readability and usability.
                for key in other:
                    # ✅ Best Practice: Type hinting improves code readability and helps with static analysis.
                    inst[key] = other[key]
            # 🧠 ML Signal: Pattern of calculating an end index from a start index and length
            elif hasattr(other, "keys"):
                # ⚠️ SAST Risk (Low): Potential file existence check race condition.
                for key in other.keys():
                    inst[key] = other[key]
            else:
                for key, value in other:
                    inst[key] = value
        # ✅ Best Practice: Returning an empty Series with specified dtype is clear and explicit.
        for key, value in kwargs.items():
            inst[key] = value

        # ⚠️ SAST Risk (Low): Error message may expose internal type information.
        self._write_instrument(inst)

    def __len__(self) -> int:
        return len(self.data)
# ⚠️ SAST Risk (Medium): File is opened without exception handling, which may lead to resource leaks.


class FileFeatureStorage(FileStorageMixin, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, provider_uri: dict = None, **kwargs):
        # ⚠️ SAST Risk (Low): Error message may expose internal index information.
        super(FileFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        # ✅ Best Practice: Calculating seek position based on index is efficient for file access.
        self.file_name = f"{instrument.lower()}/{field.lower()}.{freq.lower()}.bin"

    # ⚠️ SAST Risk (Medium): Unpacking without validation may lead to unexpected exceptions.
    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.Series:
        return self[:]
    # ✅ Best Practice: Ensure the method is type hinted for better readability and maintainability.

    # 🧠 ML Signal: Use of numpy for efficient data handling and processing.
    # ✅ Best Practice: Using pd.Series with index improves data handling and access.
    # ⚠️ SAST Risk (Low): Error message may expose internal type information.
    # 🧠 ML Signal: Method call pattern can be used to understand object behavior.
    # ⚠️ SAST Risk (Low): Potential integer division and subtraction could lead to unexpected results if not handled properly.
    # 🧠 ML Signal: Usage of file size and arithmetic operations can indicate data processing patterns.
    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        if len(data_array) == 0:
            logger.info(
                "len(data_array) == 0, write"
                "if you need to clear the FeatureStorage, please execute: FeatureStorage.clear"
            )
            return
        if not self.uri.exists():
            # write
            index = 0 if index is None else index
            with self.uri.open("wb") as fp:
                np.hstack([index, data_array]).astype("<f").tofile(fp)
        else:
            if index is None or index > self.end_index:
                # append
                index = 0 if index is None else index
                with self.uri.open("ab+") as fp:
                    np.hstack([[np.nan] * (index - self.end_index - 1), data_array]).astype("<f").tofile(fp)
            else:
                # rewrite
                with self.uri.open("rb+") as fp:
                    _old_data = np.fromfile(fp, dtype="<f")
                    _old_index = _old_data[0]
                    _old_df = pd.DataFrame(
                        _old_data[1:], index=range(_old_index, _old_index + len(_old_data) - 1), columns=["old"]
                    )
                    fp.seek(0)
                    _new_df = pd.DataFrame(data_array, index=range(index, index + len(data_array)), columns=["new"])
                    _df = pd.concat([_old_df, _new_df], sort=False, axis=1)
                    _df = _df.reindex(range(_df.index.min(), _df.index.max() + 1))
                    _df["new"].fillna(_df["old"]).values.astype("<f").tofile(fp)

    @property
    def start_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        with self.uri.open("rb") as fp:
            index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
        return index

    @property
    def end_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        # The next  data appending index point will be  `end_index + 1`
        return self.start_index + len(self) - 1

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        if not self.uri.exists():
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series(dtype=np.float32)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        storage_start_index = self.start_index
        storage_end_index = self.end_index
        with self.uri.open("rb") as fp:
            if isinstance(i, int):
                if storage_start_index > i:
                    raise IndexError(f"{i}: start index is {storage_start_index}")
                fp.seek(4 * (i - storage_start_index) + 4)
                return i, struct.unpack("f", fp.read(4))[0]
            elif isinstance(i, slice):
                start_index = storage_start_index if i.start is None else i.start
                end_index = storage_end_index if i.stop is None else i.stop - 1
                si = max(start_index, storage_start_index)
                if si > end_index:
                    return pd.Series(dtype=np.float32)
                fp.seek(4 * (si - storage_start_index) + 4)
                # read n bytes
                count = end_index - si + 1
                data = np.frombuffer(fp.read(4 * count), dtype="<f")
                return pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        self.check()
        return self.uri.stat().st_size // 4 - 1