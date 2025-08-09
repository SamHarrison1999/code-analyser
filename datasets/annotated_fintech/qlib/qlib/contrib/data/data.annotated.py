# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ⚠️ SAST Risk (Medium): Importing from qlib.data.data could introduce security risks if the library is not properly maintained or if it executes untrusted code.
# We remove arctic from core framework of Qlib to contrib due to
# - Arctic has very strict limitation on pandas and numpy version
#    - https://github.com/man-group/arctic/pull/908
# - pip fail to computing the right version number!!!!
#    - Maybe we can solve this problem by poetry
# ⚠️ SAST Risk (Low): Using a mutable default value for market_transaction_time_list can lead to unexpected behavior if modified.

# FIXME: So if you want to use arctic-based provider, please install arctic manually
# 🧠 ML Signal: Storing the uri parameter, which could be used to identify network patterns or configurations.
# `pip install arctic` may not be enough.
from arctic import Arctic
# 🧠 ML Signal: Storing the retry_time parameter, which could be used to identify retry logic or patterns.
import pandas as pd
# 🧠 ML Signal: Conversion of field to string might indicate dynamic field handling
import pymongo
# 🧠 ML Signal: Storing the market_transaction_time_list parameter, which could be used to identify time-based patterns or configurations.

# ⚠️ SAST Risk (Low): Potential exposure to NoSQL injection if `self.uri` is not properly sanitized
from qlib.data.data import FeatureProvider

# 🧠 ML Signal: Use of Arctic library for time-series data management

class ArcticFeatureProvider(FeatureProvider):
    # ✅ Best Practice: Check if the frequency library exists before proceeding
    def __init__(
        self, uri="127.0.0.1", retry_time=0, market_transaction_time_list=[("09:15", "11:30"), ("13:00", "15:00")]
    ):
        # ✅ Best Practice: Check if the instrument exists in the specified library
        super().__init__()
        self.uri = uri
        # 🧠 ML Signal: Reading a specific range of data from Arctic
        # TODO:
        # retry connecting if error occurs
        # does it real matters?
        self.retry_time = retry_time
        # NOTE: this is especially important for TResample operator
        self.market_transaction_time_list = market_transaction_time_list
    # ✅ Best Practice: Check if the series is empty before processing
    # 🧠 ML Signal: Filtering data based on market transaction times

    def feature(self, instrument, field, start_index, end_index, freq):
        field = str(field)[1:]
        with pymongo.MongoClient(self.uri) as client:
            # TODO: this will result in frequently connecting the server and performance issue
            arctic = Arctic(client)

            if freq not in arctic.list_libraries():
                raise ValueError("lib {} not in arctic".format(freq))

            if instrument not in arctic[freq].list_symbols():
                # instruments does not exist
                return pd.Series()
            else:
                df = arctic[freq].read(instrument, columns=[field], chunk_range=(start_index, end_index))
                s = df[field]

                if not s.empty:
                    s = pd.concat(
                        [
                            s.between_time(time_tuple[0], time_tuple[1])
                            for time_tuple in self.market_transaction_time_list
                        ]
                    )
                return s