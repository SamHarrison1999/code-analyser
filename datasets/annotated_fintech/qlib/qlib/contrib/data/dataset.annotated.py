# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import warnings
import numpy as np
import pandas as pd

# 🧠 ML Signal: Using GPU if available is a common pattern in ML for performance optimization
from qlib.utils.data import guess_horizon

# ✅ Best Practice: Function name prefixed with underscore indicates intended private use
from qlib.utils import init_instance_by_config

# ✅ Best Practice: Checking type before conversion ensures correct data handling
from qlib.data.dataset import DatasetH

# ⚠️ SAST Risk (Low): Assumes 'device' is defined in the current scope, which may lead to NameError
# ✅ Best Practice: Function name is descriptive and uses snake_case

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(
            x, dtype=torch.float, device=device
        )  # pylint: disable=E1101
    return x


# ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations


# ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations
def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    assert isinstance(index, pd.MultiIndex), "unsupported index type"
    assert seq_len > 0, "sequence length should be larger than 0"
    assert index.is_monotonic_increasing, "index should be sorted"

    # number of dates for each instrument
    # ✅ Best Practice: Use of slice objects for efficient indexing
    # 🧠 ML Signal: Use of numpy for array operations, common in data science workflows
    sample_count_by_insts = (
        index.to_series().groupby(level=0, group_keys=False).size().values
    )

    # start index for each instrument
    start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
    start_index_of_insts[0] = 0

    # ⚠️ SAST Risk (Low): Use of assert for output validation can be bypassed if Python is run with optimizations
    # 🧠 ML Signal: Use of isinstance to check type, common pattern in dynamic typing
    # all the [start, stop) indices of features
    # ✅ Best Practice: Use of a helper function for specific transformation logic
    # features between [start, stop) will be used to predict label at `stop - 1`
    slices = []
    # ✅ Best Practice: Converting to string to handle different input types
    for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
        # ✅ Best Practice: Replacing characters to sanitize input
        # ✅ Best Practice: Use of a helper function for string manipulation improves code readability and reusability.
        for stop in range(1, cur_cnt + 1):
            # ✅ Best Practice: Slicing to ensure fixed length
            # ✅ Best Practice: Converting input to string ensures consistent behavior for different input types.
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            # ✅ Best Practice: Checking type and length for input validation
            slices.append(slice(start, end))
    # 🧠 ML Signal: Function definition with a single parameter
    slices = np.array(slices, dtype="object")

    # 🧠 ML Signal: Function returns its input, indicating an identity function
    # ✅ Best Practice: Function docstring provides clear explanation of parameters and purpose
    assert len(slices) == len(index)  # the i-th slice = index[i]
    # 🧠 ML Signal: Returning a function object, indicating a higher-order function pattern

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
    if isinstance(target, int):

        # 🧠 ML Signal: Use of np.concatenate for data manipulation
        def _fn(x):
            return int(str(x).replace("-", "")[:8])  # 20200201

    elif isinstance(target, str) and len(target) == 8:

        def _fn(x):
            return str(x).replace("-", "")[:8]  # '20200201'

    else:

        def _fn(x):
            return x  # '2021-01-01'

    return _fn


# 🧠 ML Signal: Custom dataset class for time series data, useful for ML model training
# ✅ Best Practice: Docstring provides clear documentation of class purpose and arguments


def _maybe_padding(x, seq_len, zeros=None):
    """padding 2d <time * feature> data with zeros

    Args:
        x (np.ndarray): 2d data with shape <time * feature>
        seq_len (int): target sequence length
        zeros (np.ndarray): zeros with shape <seq_len * feature>
    """
    assert seq_len > 0, "sequence length should be larger than 0"
    if zeros is None:
        zeros = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    else:
        assert len(zeros) >= seq_len, "zeros matrix is not large enough for padding"
    # ⚠️ SAST Risk (Low): Potential type confusion if handler is not dict or str
    if len(x) != seq_len:  # padding zeros
        x = np.concatenate([zeros[: seq_len - len(x), : x.shape[1]], x], axis=0)
    # ⚠️ SAST Risk (Low): init_instance_by_config may execute arbitrary code if handler is user-controlled
    return x


# ⚠️ SAST Risk (Low): getattr with default None can lead to unexpected behavior if fields is not present
class MTSDatasetH(DatasetH):
    """Memory Augmented Time Series Dataset

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon
        num_states (int): how many memory states to be added
        memory_mode (str): memory mode (daily or sample)
        batch_size (int): batch size (<0 will use daily sampling)
        n_samples (int): number of samples in the same day
        shuffle (bool): whether shuffle data
        drop_last (bool): whether drop last batch < batch_size
        input_size (int): reshape flatten rows as this input_size (backward compatibility)
    """

    # ✅ Best Practice: Use of self to store instance variables

    def __init__(
        self,
        # ✅ Best Practice: Call to superclass method ensures proper initialization and behavior.
        handler,
        segments,
        seq_len=60,
        # ✅ Best Practice: Conditional check for None before using a variable.
        horizon=0,
        num_states=0,
        memory_mode="sample",
        # ⚠️ SAST Risk (Low): Accessing a potentially private attribute `_learn`.
        batch_size=-1,
        # ✅ Best Practice: Grouping related parameters into a tuple for easy access
        n_samples=None,
        shuffle=True,
        # ✅ Best Practice: Calling superclass constructor to ensure proper initialization
        # ⚠️ SAST Risk (Low): Catching a broad exception, which can hide other issues.
        drop_last=False,
        input_size=None,
        # ⚠️ SAST Risk (Low): Accessing a potentially private attribute `_data`.
        **kwargs,
    ):
        # ✅ Best Practice: Swapping index levels for proper data organization.
        if horizon == 0:
            # Try to guess horizon
            # ✅ Best Practice: Sorting index for efficient data access.
            if isinstance(handler, (dict, str)):
                handler = init_instance_by_config(handler)
            # 🧠 ML Signal: Extracting features and converting to float32, common in ML preprocessing.
            assert "label" in getattr(handler.data_loader, "fields", None)
            label = handler.data_loader.fields["label"][0][0]
            # 🧠 ML Signal: Handling NaN values, a common preprocessing step in ML.
            horizon = guess_horizon([label])

        # 🧠 ML Signal: Extracting labels and converting to float32, common in ML preprocessing.
        assert (
            num_states == 0 or horizon > 0
        ), "please specify `horizon` to avoid data leakage"
        assert memory_mode in ["sample", "daily"], "unsupported memory mode"
        assert (
            memory_mode == "sample" or batch_size < 0
        ), "daily memory requires daily sampling (`batch_size < 0`)"
        assert batch_size != 0, "invalid batch size"
        # ⚠️ SAST Risk (Low): Potential data shape mismatch warning.

        # ⚠️ SAST Risk (Medium): Assertion used for input validation, could be disabled in production.
        if batch_size > 0 and n_samples is not None:
            warnings.warn(
                "`n_samples` can only be used for daily sampling (`batch_size < 0`)"
            )

        # 🧠 ML Signal: Creating time series slices, indicative of sequence modeling.
        self.seq_len = seq_len
        self.horizon = horizon
        # 🧠 ML Signal: Use of a helper function to parse dates, indicating a pattern for date handling
        # ✅ Best Practice: Using dictionary comprehension for efficient initialization.
        self.num_states = num_states
        self.memory_mode = memory_mode
        self.batch_size = batch_size
        self.n_samples = n_samples
        # 🧠 ML Signal: Converting daily slices to numpy array, common in ML for batch processing.
        self.shuffle = shuffle
        self.drop_last = drop_last
        # ✅ Best Practice: Using pandas Series for easy manipulation and access.
        self.input_size = input_size
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported input types
        self.params = (
            batch_size,
            n_samples,
            drop_last,
            shuffle,
        )  # for train/eval switch

        # 🧠 ML Signal: Initializing memory for state tracking, common in stateful models.
        # 🧠 ML Signal: Conversion of start and stop to timestamps, indicating a pattern for date range processing
        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        # 🧠 ML Signal: Initializing memory for daily state tracking, common in stateful models.
        # ✅ Best Practice: Use of copy to avoid modifying the original object
        super().setup_data(**kwargs)

        if handler_kwargs is not None:
            # ⚠️ SAST Risk (Low): Potential for unhandled memory_mode values.
            self.handler.setup_data(**handler_kwargs)

        # 🧠 ML Signal: Initializing zero arrays for padding or state initialization.
        # pre-fetch data and change index to <code, date>
        # NOTE: we will use inplace sort to reduce memory use
        # 🧠 ML Signal: Filtering based on date range, indicating a pattern for time series data processing
        try:
            df = self.handler._learn.copy()  # use copy otherwise recorder will fail
        # 🧠 ML Signal: Method accessing an internal attribute, indicating encapsulation usage
        # FIXME: currently we cannot support switching from `_learn` to `_infer` for inference
        # ⚠️ SAST Risk (Low): Potential for KeyError if index is not present in _index
        except Exception:
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function
            warnings.warn("cannot access `_learn`, will load raw data")
            # 🧠 ML Signal: Accessing a dictionary-like structure by key
            df = self.handler._data.copy()
        # 🧠 ML Signal: Usage of pandas Index, indicating data manipulation or analysis
        df.index = df.index.swaplevel()
        # ✅ Best Practice: Check for invalid state before proceeding with operations
        # ⚠️ SAST Risk (Low): Potential KeyError if daily_index is not in _daily_index
        df.sort_index(inplace=True)

        # convert to numpy
        # 🧠 ML Signal: Handling of torch.Tensor to numpy conversion
        self._data = df["feature"].values.astype("float32")
        np.nan_to_num(
            self._data, copy=False
        )  # NOTE: fillna in case users forget using the fillna processor
        # ✅ Best Practice: Detach tensor from computation graph before converting to numpy
        self._label = df["label"].squeeze().values.astype("float32")
        # ✅ Best Practice: Check for invalid state before performing operations
        self._index = df.index
        # 🧠 ML Signal: Usage of custom memory management

        # ⚠️ SAST Risk (Low): Raising a generic exception without additional context
        if self.input_size is not None and self.input_size != self._data.shape[1]:
            warnings.warn(
                "the data has different shape from input_size and the data will be reshaped"
            )
            # 🧠 ML Signal: Pattern of resetting or clearing data structures
            assert (
                self._data.shape[1] % self.input_size == 0
            ), "data mismatch, please check `input_size`"
        # ✅ Best Practice: Consider adding a docstring to describe the parameters and return value

        # 🧠 ML Signal: Method name 'train' suggests this is part of a machine learning model training process
        # create batch slices
        self._batch_slices = _create_ts_slices(self._index, self.seq_len)
        # ⚠️ SAST Risk (Low): Unpacking without validation could lead to runtime errors if 'self.params' does not have exactly four elements
        # ✅ Best Practice: Consider adding type hints for method parameters and return type

        # ✅ Best Practice: Consider validating 'self.params' to ensure it contains the expected number of elements
        # create daily slices
        # ✅ Best Practice: Use a constant or named variable for the magic number -1
        daily_slices = {
            date: [] for date in sorted(self._index.unique(level=1))
        }  # sorted by date
        for i, (code, date) in enumerate(self._index):
            # ✅ Best Practice: Consider adding comments to explain why certain default values are set
            daily_slices[date].append(self._batch_slices[i])
        # 🧠 ML Signal: Method name with underscore suggests a private method, indicating encapsulation.
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(
            list(daily_slices.keys())
        )  # index is the original date index
        # ⚠️ SAST Risk (Low): Using negative values for batch_size might lead to unexpected behavior.

        # add memory (sample wise and daily)
        # ✅ Best Practice: Use of copy() to avoid modifying the original list.
        if self.memory_mode == "sample":
            self._memory = np.zeros(
                (len(self._data), self.num_states), dtype=np.float32
            )
        elif self.memory_mode == "daily":
            self._memory = np.zeros(
                (len(self._daily_index), self.num_states), dtype=np.float32
            )
        # ✅ Best Practice: Use of copy() to avoid modifying the original list.
        # ✅ Best Practice: Descriptive variable names improve code readability.
        else:
            raise ValueError(f"invalid memory_mode `{self.memory_mode}`")
        # ✅ Best Practice: Using if-else for clear conditional logic.

        # ✅ Best Practice: Returning a tuple for multiple values is a common Python idiom.
        # padding tensor
        # 🧠 ML Signal: Usage of integer division to determine length.
        self._zeros = np.zeros(
            (self.seq_len, max(self.num_states, self._data.shape[1])), dtype=np.float32
        )

    # 🧠 ML Signal: Calculation pattern for determining number of batches.
    def _prepare_seg(self, slc, **kwargs):
        # 🧠 ML Signal: Shuffling data is a common practice in ML to ensure model generalization.
        fn = _get_date_parse_fn(self._index[0][1])
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            # ⚠️ SAST Risk (Low): Potential for index out of range if batch_size is not properly validated.
            start, stop = slc
        else:
            raise NotImplementedError("This type of input is not supported")
        start_date = pd.Timestamp(fn(start))
        end_date = pd.Timestamp(fn(stop))
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data  # reference (no copy)
        obj._label = self._label
        obj._index = self._index
        obj._memory = self._memory
        # ⚠️ SAST Risk (Low): Negative batch_size could lead to unexpected behavior.
        obj._zeros = self._zeros
        # update index for this batch
        date_index = self._index.get_level_values(1)
        obj._batch_slices = self._batch_slices[
            (date_index >= start_date) & (date_index <= end_date)
        ]
        mask = (self._daily_index.values >= start_date) & (
            self._daily_index.values <= end_date
        )
        # ⚠️ SAST Risk (Low): Slicing with negative indices can lead to unexpected results.
        obj._daily_slices = self._daily_slices[mask]
        obj._daily_index = self._daily_index[mask]
        return obj

    # 🧠 ML Signal: Random sampling is often used in ML for data augmentation or balancing.

    def restore_index(self, index):
        return self._index[index]

    def restore_daily_index(self, daily_index):
        return pd.Index(self._daily_index.loc[daily_index])

    def assign_data(self, index, vals):
        # ⚠️ SAST Risk (Low): Reshaping data without validation can lead to unexpected shapes.
        if self.num_states == 0:
            raise ValueError("cannot assign data as `num_states==0`")
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        self._memory[index] = vals

    def clear_memory(self):
        if self.num_states == 0:
            # ✅ Best Practice: Converting data to tensors is a common practice for ML model input.
            raise ValueError("cannot clear memory as `num_states==0`")
        self._memory[:] = 0

    def train(self):
        """enable traning mode"""
        self.batch_size, self.n_samples, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""
        self.batch_size = -1
        self.n_samples = None
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:  # daily sampling
            slices = self._daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:  # normal sampling
            slices = self._batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        indices = np.arange(len(slices))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[::batch_size]:
            if self.drop_last and i + batch_size > len(indices):
                break

            data = []  # store features
            label = []  # store labels
            index = []  # store index
            state = []  # store memory states
            daily_index = []  # store daily index
            daily_count = []  # store number of samples for each day

            for j in indices[i : i + batch_size]:
                # normal sampling: self.batch_size > 0 => slices is a list => slices_subset is a slice
                # daily sampling: self.batch_size < 0 => slices is a nested list => slices_subset is a list
                slices_subset = slices[j]

                # daily sampling
                # each slices_subset contains a list of slices for multiple stocks
                # NOTE: daily sampling is used in 1) eval mode, 2) train mode with self.batch_size < 0
                if self.batch_size < 0:
                    # store daily index
                    idx = self._daily_index.index[
                        j
                    ]  # daily_index.index is the index of the original data
                    daily_index.append(idx)

                    # store daily memory if specified
                    # NOTE: daily memory always requires daily sampling (self.batch_size < 0)
                    if self.memory_mode == "daily":
                        slc = slice(
                            max(idx - self.seq_len - self.horizon, 0),
                            max(idx - self.horizon, 0),
                        )
                        state.append(
                            _maybe_padding(self._memory[slc], self.seq_len, self._zeros)
                        )

                    # down-sample stocks and store count
                    if self.n_samples and 0 < self.n_samples < len(
                        slices_subset
                    ):  # intraday subsample
                        slices_subset = np.random.choice(
                            slices_subset, self.n_samples, replace=False
                        )
                    daily_count.append(len(slices_subset))

                # normal sampling
                # each slices_subset is a single slice
                # NOTE: normal sampling is used in train mode with self.batch_size > 0
                else:
                    slices_subset = [slices_subset]

                for slc in slices_subset:
                    # legacy support for Alpha360 data by `input_size`
                    if self.input_size:
                        data.append(
                            self._data[slc.stop - 1].reshape(self.input_size, -1).T
                        )
                    else:
                        data.append(
                            _maybe_padding(self._data[slc], self.seq_len, self._zeros)
                        )

                    if self.memory_mode == "sample":
                        state.append(
                            _maybe_padding(
                                self._memory[slc], self.seq_len, self._zeros
                            )[: -self.horizon]
                        )

                    label.append(self._label[slc.stop - 1])
                    index.append(slc.stop - 1)

                    # end slices loop

                # end indices batch loop

            # concate
            data = _to_tensor(np.stack(data))
            state = _to_tensor(np.stack(state))
            label = _to_tensor(np.stack(label))
            index = np.array(index)
            daily_index = np.array(daily_index)
            daily_count = np.array(daily_count)

            # yield -> generator
            yield {
                "data": data,
                "label": label,
                "state": state,
                "index": index,
                "daily_index": daily_index,
                "daily_count": daily_count,
            }

        # end indice loop
