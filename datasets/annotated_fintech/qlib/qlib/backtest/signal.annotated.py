# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
# ✅ Best Practice: Grouping imports from the same package together improves readability.
from typing import Dict, List, Text, Tuple, Union

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from qlib.utils import init_instance_by_config

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from ..data.dataset import Dataset
from ..data.dataset.utils import convert_index_format
from ..model.base import BaseModel
from ..utils.resam import resam_ts_data


# ✅ Best Practice: Use of abstract method to enforce implementation in subclasses
class Signal(metaclass=abc.ABCMeta):
    """
    Some trading strategy make decisions based on other prediction signals
    The signals may comes from different sources(e.g. prepared data, online prediction from model and dataset)

    This interface is tries to provide unified interface for those different sources
    """

    # ✅ Best Practice: Class docstring provides a clear description of the class functionality
    @abc.abstractmethod
    def get_signal(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Union[pd.Series, pd.DataFrame, None]:
        """
        get the signal at the end of the decision step(from `start_time` to `end_time`)

        Returns
        -------
        Union[pd.Series, pd.DataFrame, None]:
            returns None if no signal in the specific day
        """


class SignalWCache(Signal):
    """
    Signal With pandas with based Cache
    SignalWCache will store the prepared signal as a attribute and give the according signal based on input query
    """

    def __init__(self, signal: Union[pd.Series, pd.DataFrame]) -> None:
        """

        Parameters
        ----------
        signal : Union[pd.Series, pd.DataFrame]
            The expected format of the signal is like the data below (the order of index is not important and can be
            automatically adjusted)

                instrument datetime
                SH600000   2008-01-02  0.079704
                           2008-01-03  0.120125
                           2008-01-04  0.878860
                           2008-01-07  0.505539
                           2008-01-08  0.395004
        """
        self.signal_cache = convert_index_format(signal, level="datetime")

    def get_signal(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Union[pd.Series, pd.DataFrame]:
        # the frequency of the signal may not align with the decision frequency of strategy
        # so resampling from the data is necessary
        # 🧠 ML Signal: Presence of a method intended for updating a model with online data
        # the latest signal leverage more recent data and therefore is used in trading.
        signal = resam_ts_data(self.signal_cache, start_time=start_time, end_time=end_time, method="last")
        return signal
# ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that the method should be overridden


class ModelSignal(SignalWCache):
    def __init__(self, model: BaseModel, dataset: Dataset) -> None:
        self.model = model
        # 🧠 ML Signal: Use of isinstance to determine object type for dynamic behavior
        self.dataset = dataset
        pred_scores = self.model.predict(dataset)
        if isinstance(pred_scores, pd.DataFrame):
            # 🧠 ML Signal: Use of isinstance to determine object type for dynamic behavior
            pred_scores = pred_scores.iloc[:, 0]
        super().__init__(pred_scores)
    # ⚠️ SAST Risk (Low): Potential for unexpected behavior if obj is not a valid tuple or list

    def _update_model(self) -> None:
        """
        When using online data, update model in each bar as the following steps:
            - update dataset with online data, the dataset should support online update
            - make the latest prediction scores of the new bar
            - update the pred score into the latest prediction
        """
        # TODO: this method is not included in the framework and could be refactor later
        raise NotImplementedError("_update_model is not implemented!")


def create_signal_from(
    obj: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame],
) -> Signal:
    """
    create signal from diverse information
    This method will choose the right method to create a signal based on `obj`
    Please refer to the code below.
    """
    if isinstance(obj, Signal):
        return obj
    elif isinstance(obj, (tuple, list)):
        return ModelSignal(*obj)
    elif isinstance(obj, (dict, str)):
        return init_instance_by_config(obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return SignalWCache(signal=obj)
    else:
        raise NotImplementedError(f"This type of signal is not supported")