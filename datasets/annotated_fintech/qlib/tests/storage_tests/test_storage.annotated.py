# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from pathlib import Path
from collections.abc import Iterable

import numpy as np
from qlib.tests import TestAutoData

# 🧠 ML Signal: Usage of Path to manipulate file paths, which is common in data processing tasks
from qlib.data.storage.file_storage import (
    FileCalendarStorage as CalendarStorage,
    # 🧠 ML Signal: Usage of Path to create directories, which is common in data storage tasks
    FileInstrumentStorage as InstrumentStorage,
    FileFeatureStorage as FeatureStorage,
# ⚠️ SAST Risk (Low): Potential directory traversal if __file__ is manipulated
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is followed here.
)
# ✅ Best Practice: Use of mkdir with exist_ok=True to avoid exceptions if the directory already exists
# 🧠 ML Signal: Testing object creation with specific parameters

_file_name = Path(__file__).name.split(".")[0]
# 🧠 ML Signal: Checking if the object supports slicing and is iterable
DATA_DIR = Path(__file__).parent.joinpath(f"{_file_name}_data")
QLIB_DIR = DATA_DIR.joinpath("qlib")
# 🧠 ML Signal: Checking if the data attribute is iterable
QLIB_DIR.mkdir(exist_ok=True, parents=True)

# ✅ Best Practice: Using print statements for debugging and output verification

class TestStorage(TestAutoData):
    # ✅ Best Practice: Using print statements for debugging and output verification
    def test_calendar_storage(self):
        calendar = CalendarStorage(freq="day", future=False, provider_uri=self.provider_uri)
        # ✅ Best Practice: Using print statements for debugging and output verification
        assert isinstance(calendar[:], Iterable), f"{calendar.__class__.__name__}.__getitem__(s: slice) is not Iterable"
        assert isinstance(calendar.data, Iterable), f"{calendar.__class__.__name__}.data is not Iterable"
        # 🧠 ML Signal: Testing object creation with different parameters

        # 🧠 ML Signal: Testing function for instrument storage behavior
        print(f"calendar[1: 5]: {calendar[1:5]}")
        # 🧠 ML Signal: Iterating over instrument data to validate structure
        # 🧠 ML Signal: Accessing specific instrument data for validation
        # 🧠 ML Signal: Testing exception handling for invalid data access
        # ✅ Best Practice: Docstring provides detailed explanation of the test case
        # ⚠️ SAST Risk (Low): Type checking with assert, could be disabled in optimized mode
        print(f"calendar[0]: {calendar[0]}")
        print(f"calendar[-1]: {calendar[-1]}")

        calendar = CalendarStorage(freq="1min", future=False, provider_uri="not_found")
        with self.assertRaises(ValueError):
            print(calendar.data)

        with self.assertRaises(ValueError):
            print(calendar[:])

        with self.assertRaises(ValueError):
            print(calendar[0])

    def test_instrument_storage(self):
        """
        The meaning of instrument, such as CSI500:

            CSI500 composition changes:

                date            add         remove
                2005-01-01      SH600000
                2005-01-01      SH600001
                2005-01-01      SH600002
                2005-02-01      SH600003    SH600000
                2005-02-15      SH600000    SH600002

            Calendar:
                pd.date_range(start="2020-01-01", stop="2020-03-01", freq="1D")

            Instrument:
                symbol      start_time      end_time
                SH600000    2005-01-01      2005-01-31 (2005-02-01 Last trading day)
                SH600000    2005-02-15      2005-03-01
                SH600001    2005-01-01      2005-03-01
                SH600002    2005-01-01      2005-02-14 (2005-02-15 Last trading day)
                SH600003    2005-02-01      2005-03-01

            InstrumentStorage:
                {
                    "SH600000": [(2005-01-01, 2005-01-31), (2005-02-15, 2005-03-01)],
                    "SH600001": [(2005-01-01, 2005-03-01)],
                    "SH600002": [(2005-01-01, 2005-02-14)],
                    "SH600003": [(2005-02-01, 2005-03-01)],
                }

        """

        instrument = InstrumentStorage(market="csi300", provider_uri=self.provider_uri, freq="day")

        for inst, spans in instrument.data.items():
            assert isinstance(inst, str) and isinstance(
                spans, Iterable
            ), f"{instrument.__class__.__name__} value is not Iterable"
            for s_e in spans:
                assert (
                    isinstance(s_e, tuple) and len(s_e) == 2
                ), f"{instrument.__class__.__name__}.__getitem__(k) TypeError"

        print(f"instrument['SH600000']: {instrument['SH600000']}")

        instrument = InstrumentStorage(market="csi300", provider_uri="not_found", freq="day")
        with self.assertRaises(ValueError):
            print(instrument.data)

        with self.assertRaises(ValueError):
            print(instrument["sSH600000"])

    def test_feature_storage(self):
        """
        Calendar:
            pd.date_range(start="2005-01-01", stop="2005-03-01", freq="1D")

        Instrument:
            {
                "SH600000": [(2005-01-01, 2005-01-31), (2005-02-15, 2005-03-01)],
                "SH600001": [(2005-01-01, 2005-03-01)],
                "SH600002": [(2005-01-01, 2005-02-14)],
                "SH600003": [(2005-02-01, 2005-03-01)],
            }

        Feature:
            Stock data(close):
                            2005-01-01  ...   2005-02-01   ...   2005-02-14  2005-02-15  ...  2005-03-01
                SH600000     1          ...      3         ...      4           5               6
                SH600001     1          ...      4         ...      5           6               7
                SH600002     1          ...      5         ...      6           nan             nan
                SH600003     nan        ...      1         ...      2           3               4

            FeatureStorage(SH600000, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 6)
                ]

                ====> [(0, 1), ..., (59, 6)]


            FeatureStorage(SH600002, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-02-14"), 6)
                ]

                ===> [(0, 1), ..., (44, 6)]

            FeatureStorage(SH600003, close):

                [
                    (calendar.index("2005-02-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 4)
                ]

                ===> [(31, 1), ..., (59, 4)]

        """

        feature = FeatureStorage(instrument="SZ300677", field="close", freq="day", provider_uri=self.provider_uri)

        with self.assertRaises(IndexError):
            print(feature[0])
        assert isinstance(
            feature[3049][1], (float, np.float32)
        ), f"{feature.__class__.__name__}.__getitem__(i: int) error"
        assert len(feature[3049:3052]) == 3, f"{feature.__class__.__name__}.__getitem__(s: slice) error"
        print(f"feature[3049: 3052]: \n{feature[3049: 3052]}")

        print(f"feature[:].tail(): \n{feature[:].tail()}")

        feature = FeatureStorage(instrument="SH600004", field="close", freq="day", provider_uri="not_fount")

        with self.assertRaises(ValueError):
            print(feature[0])
        with self.assertRaises(ValueError):
            print(feature[:].empty)
        with self.assertRaises(ValueError):
            print(feature.data.empty)