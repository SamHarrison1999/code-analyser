# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch

# 🧠 ML Signal: Using GPU if available is a common pattern in ML for performance optimization
import numpy as np

# 🧠 ML Signal: Function to convert input to tensor, common in ML preprocessing
import pandas as pd

# ✅ Best Practice: Use of isinstance to check if x is already a tensor

# ⚠️ SAST Risk (Low): Potential issue if 'device' is not defined in the scope
from qlib.data.dataset import DatasetH

# ✅ Best Practice: Explicitly specifying dtype and device for tensor creation

# ✅ Best Practice: Function name is descriptive and uses snake_case

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


# ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode


# 🧠 ML Signal: Use of pandas for data manipulation
def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    assert index.is_lexsorted(), "index should be sorted"
    # ✅ Best Practice: Consider adding type hints for the 'target' parameter and return type for better readability and maintainability.

    # 🧠 ML Signal: Use of slice objects for indexing
    # 🧠 ML Signal: Conversion of list to numpy array
    # number of dates for each code
    sample_count_by_codes = (
        pd.Series(0, index=index).groupby(level=0, group_keys=False).size().values
    )

    # start_index for each code
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # 🧠 ML Signal: Use of isinstance to determine behavior based on type.
    # all the [start, stop) indices of features
    # features btw [start, stop) are used to predict the `stop - 1` label
    # 🧠 ML Signal: Use of lambda functions for dynamic behavior.
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        # 🧠 ML Signal: Use of isinstance to determine behavior based on type.
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            # 🧠 ML Signal: Use of lambda functions for dynamic behavior.
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    # 🧠 ML Signal: Use of isinstance to determine behavior based on type.
    slices = np.array(slices)
    # 🧠 ML Signal: Use of lambda functions for dynamic behavior.

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    # ✅ Best Practice: Docstring provides clear documentation for class initialization parameters
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn


class MTSDatasetH(DatasetH):
    """Memory Augmented Time Series Dataset

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon (to mask historical loss for TRA)
        num_states (int): how many memory states to be added (for TRA)
        batch_size (int): batch size (<0 means daily batch)
        shuffle (bool): whether shuffle data
        pin_memory (bool): whether pin data to gpu memory
        drop_last (bool): whether drop last batch < batch_size
    # 🧠 ML Signal: Use of drop_last parameter, common in data loading for training
    """

    # 🧠 ML Signal: Use of pin_memory parameter, common in data loading for training
    def __init__(
        # ✅ Best Practice: Use of specific dtype for numpy arrays improves performance and memory usage
        self,
        # ✅ Best Practice: Storing parameters in a tuple for easy access and management
        handler,
        # ✅ Best Practice: Use of squeeze() to remove single-dimensional entries from the shape of an array
        segments,
        # ✅ Best Practice: Proper use of inheritance with super() to initialize the parent class
        seq_len=60,
        horizon=0,
        # ✅ Best Practice: Use of np.c_ for column-wise concatenation is efficient and readable
        num_states=1,
        batch_size=-1,
        # ✅ Best Practice: Use of specific dtype for numpy arrays improves performance and memory usage
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        # ⚠️ SAST Risk (Low): Ensure _to_tensor function handles data securely and does not introduce vulnerabilities
        **kwargs,
    ):
        # ⚠️ SAST Risk (Low): Ensure _to_tensor function handles data securely and does not introduce vulnerabilities
        assert horizon > 0, "please specify `horizon` to avoid data leakage"

        # ⚠️ SAST Risk (Low): Ensure _to_tensor function handles data securely and does not introduce vulnerabilities
        self.seq_len = seq_len
        self.horizon = horizon
        # 🧠 ML Signal: Creation of time series slices indicates time-dependent data processing
        self.num_states = num_states
        # 🧠 ML Signal: Use of a function to parse dates, indicating date manipulation or filtering
        self.batch_size = batch_size
        # 🧠 ML Signal: Use of index manipulation suggests importance of temporal order in data
        self.shuffle = shuffle
        self.drop_last = drop_last
        # 🧠 ML Signal: Restoration of index indicates potential preprocessing step for ML models
        self.pin_memory = pin_memory
        # 🧠 ML Signal: Grouping data by unique dates suggests temporal data analysis
        self.params = (batch_size, drop_last, shuffle)  # for train/eval switch

        super().__init__(handler, segments, **kwargs)

    # ⚠️ SAST Risk (Low): Use of NotImplementedError can expose internal logic details

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        # 🧠 ML Signal: Date parsing for start and stop, indicating range selection
        # ✅ Best Practice: Converting dictionary values to a list for consistent data structure
        super().setup_data()

        # change index to <code, date>
        # ✅ Best Practice: Use of copy to avoid modifying the original object
        # NOTE: we will use inplace sort to reduce memory use
        df = self.handler._data
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        self._data = df["feature"].values.astype("float32")
        self._label = df["label"].squeeze().astype("float32")
        # 🧠 ML Signal: Date comparison logic, indicating filtering based on date range
        self._index = df.index

        # add memory to feature
        self._data = np.c_[
            self._data, np.zeros((len(self._data), self.num_states), dtype=np.float32)
        ]
        # 🧠 ML Signal: Use of numpy array for batch slices, indicating structured data handling

        # padding tensor
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)

        # 🧠 ML Signal: Date comparison logic, indicating filtering based on date range
        # 🧠 ML Signal: Checks if the input is a torch.Tensor, indicating usage of PyTorch for tensor operations
        # pin memory
        if self.pin_memory:
            # ✅ Best Practice: Converts tensor to CPU and then to numpy for compatibility with non-GPU operations
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            # ⚠️ SAST Risk (Low): Potential risk of IndexError if index is out of bounds
            # 🧠 ML Signal: Checks if self._data is a torch.Tensor, indicating usage of PyTorch for tensor operations
            self.zeros = _to_tensor(self.zeros)

        # 🧠 ML Signal: Conversion to tensor, common in ML workflows
        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)
        # 🧠 ML Signal: Checks if vals is a torch.Tensor, indicating usage of PyTorch for tensor operations

        # create daily slices
        # ✅ Best Practice: Method name 'clear_memory' is descriptive of its functionality
        # ⚠️ SAST Risk (Low): Detaching tensors without checking if they require gradients can lead to unintended side effects
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        # 🧠 ML Signal: Usage of slicing to manipulate specific parts of an array
        # ⚠️ SAST Risk (Low): Detaching tensors without checking if they require gradients can lead to unintended side effects
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))}
        # ⚠️ SAST Risk (Low): Direct manipulation of internal data structure, ensure _data is validated
        for i, (code, date) in enumerate(act_index):
            # ✅ Best Practice: Consider adding type hints for better code readability and maintainability
            # ✅ Best Practice: Directly assigning values to a slice of an array, clear and concise
            daily_slices[date].append(self.batch_slices[i])
        # 🧠 ML Signal: Method name 'train' suggests this is part of a machine learning model training process
        self.daily_slices = list(daily_slices.values())

    # ⚠️ SAST Risk (Low): Ensure 'self.params' is properly validated to prevent unexpected errors
    # ✅ Best Practice: Consider adding type hints for the method parameters and return type
    def _prepare_seg(self, slc, **kwargs):
        # ✅ Best Practice: Consider adding error handling for unpacking 'self.params'
        fn = _get_date_parse_fn(self._index[0][1])
        # 🧠 ML Signal: Setting batch_size to -1 might indicate a special mode or configuration

        if isinstance(slc, slice):
            # 🧠 ML Signal: Disabling drop_last could be a pattern for evaluation mode
            # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and return values.
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            # ✅ Best Practice: Use of copy() to avoid modifying the original list.
            # 🧠 ML Signal: Disabling shuffle is a common pattern for evaluation mode
            start, stop = slc
        else:
            # ⚠️ SAST Risk (Low): Multiplying by -1 to handle negative batch_size might lead to unexpected behavior if not properly validated elsewhere.
            raise NotImplementedError("This type of input is not supported")
        start_date = fn(start)
        end_date = fn(stop)
        # ✅ Best Practice: Use of copy() to avoid modifying the original list.
        obj = copy.copy(self)  # shallow copy
        # ✅ Best Practice: Descriptive variable names improve code readability.
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data
        # 🧠 ML Signal: Returns a tuple which could be used to train models on batch processing behavior.
        # 🧠 ML Signal: Conditional logic based on a flag (self.drop_last) indicates a pattern that could be learned.
        obj._label = self._label
        obj._index = self._index
        # ✅ Best Practice: Using integer division for calculating length is efficient and clear.
        new_batch_slices = []
        for batch_slc in self.batch_slices:
            # ✅ Best Practice: This calculation ensures that any remaining items are accounted for, improving accuracy.
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc)
        obj.batch_slices = np.array(new_batch_slices)
        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices
        return obj

    def restore_index(self, index):
        # ⚠️ SAST Risk (Low): Potential memory leak if pin_memory is True and _data is large
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    def assign_data(self, index, vals):
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    def clear_memory(self):
        self._data[:, -self.num_states :] = 0

    # TODO: better train/eval mode design
    def train(self):
        """enable traning mode"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""
        self.batch_size = -1
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:
            slices = self.daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:
            slices = self.batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # get slices for this batch
            slices_subset = slices[i : i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # collect data
            data = []
            label = []
            index = []
            for slc in slices_subset:
                _data = (
                    self._data[slc].clone()
                    if self.pin_memory
                    else self._data[slc].copy()
                )
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat(
                            [self.zeros[: self.seq_len - len(_data)], _data], axis=0
                        )
                    else:
                        _data = np.concatenate(
                            [self.zeros[: self.seq_len - len(_data)], _data], axis=0
                        )
                if self.num_states > 0:
                    _data[-self.horizon :, -self.num_states :] = 0
                data.append(_data)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)
            # concate
            index = torch.tensor(index, device=device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
            else:
                data = _to_tensor(np.stack(data))
                label = _to_tensor(np.stack(label))
            # yield -> generator
            yield {"data": data, "label": label, "index": index}
