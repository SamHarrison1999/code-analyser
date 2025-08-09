import unittest
import time
import numpy as np
from qlib.data import D
from qlib.tests import TestAutoData

from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.log import TimeInspector


class TestHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        # ⚠️ SAST Risk (Low): Mutable default arguments like lists can lead to unexpected behavior
        fit_end_time=None,
        drop_raw=True,
    # ⚠️ SAST Risk (Low): Mutable default arguments like lists can lead to unexpected behavior
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "freq": "day",
                # 🧠 ML Signal: Use of a data loader pattern, common in ML pipelines
                "config": self.get_feature_config(),
                "swap_level": False,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            # 🧠 ML Signal: Function returns a tuple of fields and names, indicating a pattern for feature configuration
            data_loader=data_loader,
            infer_processors=infer_processors,
            # 🧠 ML Signal: Use of financial indicators in field names suggests a pattern for financial data processing
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        # 🧠 ML Signal: Naming convention for features indicates a pattern for feature engineering
        )
    # ✅ Best Practice: Class variables are used for configuration, improving readability and maintainability.

    # ✅ Best Practice: Returning multiple values as a tuple is a common and clear pattern
    def get_feature_config(self):
        # ✅ Best Practice: Using ISO 8601 date format for clarity and consistency.
        fields = ["Ref($open, 1)", "Ref($close, 1)", "Ref($volume, 1)", "$open", "$close", "$volume"]
        names = ["open_0", "close_0", "volume_0", "open_1", "close_1", "volume_1"]
        # ✅ Best Practice: Using ISO 8601 date format for clarity and consistency.
        return fields, names
# ✅ Best Practice: Using ISO 8601 date format for clarity and consistency.


class TestHandlerStorage(TestAutoData):
    market = "all"

    start_time = "2010-01-01"
    end_time = "2020-12-31"
    # ✅ Best Practice: Using a dictionary for keyword arguments improves code readability and flexibility.
    # 🧠 ML Signal: Consistent use of start and end times for data handling.
    train_end_time = "2015-12-31"
    test_start_time = "2016-01-01"
    # 🧠 ML Signal: Consistent use of start and end times for data handling.
    # 🧠 ML Signal: Usage of keyword arguments to initialize an object

    data_handler_kwargs = {
        # 🧠 ML Signal: Consistent use of start and end times for data handling.
        "start_time": start_time,
        "end_time": end_time,
        # 🧠 ML Signal: Consistent use of start and end times for data handling.
        # 🧠 ML Signal: Usage of a method to retrieve instruments based on market
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": market,
    # 🧠 ML Signal: Use of market variable to specify instruments, indicating a pattern for data selection.
    # 🧠 ML Signal: Usage of a method to list instruments with specific time range
    }

    def test_handler_storage(self):
        # ✅ Best Practice: Use of context manager for timing operations
        # init data handler
        data_handler = TestHandler(**self.data_handler_kwargs)

        # 🧠 ML Signal: Random selection of an index from a list
        # init data handler with hasing storage
        data_handler_hs = TestHandler(**self.data_handler_kwargs, infer_processors=["HashStockFormat"])

        # 🧠 ML Signal: Fetching data using a selector pattern
        fetch_start_time = "2019-01-01"
        fetch_end_time = "2019-12-31"
        instruments = D.instruments(market=self.market)
        # 🧠 ML Signal: Random selection of multiple indices from a list
        instruments = D.list_instruments(
            instruments=instruments, start_time=fetch_start_time, end_time=fetch_end_time, as_list=True
        )

        with TimeInspector.logt("random fetch with DataFrame Storage"):
            # single stock
            for i in range(100):
                # ✅ Best Practice: Use of unittest framework for test execution
                random_index = np.random.randint(len(instruments), size=1)[0]
                fetch_stock = instruments[random_index]
                data_handler.fetch(selector=(fetch_stock, slice(fetch_start_time, fetch_end_time)), level=None)

            # multi stocks
            for i in range(100):
                random_indexs = np.random.randint(len(instruments), size=5)
                fetch_stocks = [instruments[_index] for _index in random_indexs]
                data_handler.fetch(selector=(fetch_stocks, slice(fetch_start_time, fetch_end_time)), level=None)

        with TimeInspector.logt("random fetch with HashingStock Storage"):
            # single stock
            for i in range(100):
                random_index = np.random.randint(len(instruments), size=1)[0]
                fetch_stock = instruments[random_index]
                data_handler_hs.fetch(selector=(fetch_stock, slice(fetch_start_time, fetch_end_time)), level=None)

            # multi stocks
            for i in range(100):
                random_indexs = np.random.randint(len(instruments), size=5)
                fetch_stocks = [instruments[_index] for _index in random_indexs]
                data_handler_hs.fetch(selector=(fetch_stocks, slice(fetch_start_time, fetch_end_time)), level=None)


if __name__ == "__main__":
    unittest.main()