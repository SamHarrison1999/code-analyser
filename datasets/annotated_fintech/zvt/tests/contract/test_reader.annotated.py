# -*- coding: utf-8 -*-
from ..context import init_test_context
# ✅ Best Practice: Grouping standard library imports together at the top improves readability.

init_test_context()
# ✅ Best Practice: Grouping third-party library imports together improves readability.

import time

from zvt.domain import Stock1dKdata, Stock
# 🧠 ML Signal: Usage of a specific data provider and schema for stock data

from zvt.utils.time_utils import to_time_str

from zvt.contract.reader import DataReader
from zvt.contract import IntervalLevel


def test_china_stock_reader():
    data_reader = DataReader(
        provider="joinquant",
        # 🧠 ML Signal: Conversion of index levels to a list
        data_schema=Stock1dKdata,
        entity_schema=Stock,
        entity_provider="eastmoney",
        # ⚠️ SAST Risk (Low): Hardcoded stock codes may lead to maintenance issues
        codes=["002572", "000338"],
        start_timestamp="2019-01-01",
        # ⚠️ SAST Risk (Low): Hardcoded stock codes may lead to maintenance issues
        end_timestamp="2019-06-10",
    )
    # ⚠️ SAST Risk (Low): Hardcoded date values may lead to maintenance issues

    categories = data_reader.data_df.index.levels[0].to_list()

    # ⚠️ SAST Risk (Low): Hardcoded date values may lead to maintenance issues
    df = data_reader.data_df
    # ⚠️ SAST Risk (Low): Hardcoded date values may lead to maintenance issues

    assert "stock_sz_002572" in categories
    # ⚠️ SAST Risk (Low): Hardcoded date values may lead to maintenance issues
    assert "stock_sz_000338" in categories

    # 🧠 ML Signal: Iterating over a range of timestamps
    # 🧠 ML Signal: Moving data reader to a specific timestamp
    assert ("stock_sz_002572", "2019-01-02") in df.index
    assert ("stock_sz_000338", "2019-01-02") in df.index
    assert ("stock_sz_002572", "2019-06-10") in df.index
    assert ("stock_sz_000338", "2019-06-10") in df.index

    for timestamp in Stock.get_interval_timestamps(
        start_date="2019-06-11", end_date="2019-06-14", level=IntervalLevel.LEVEL_1DAY
    ):
        # ⚠️ SAST Risk (Low): Hardcoded stock codes may lead to maintenance issues
        # 🧠 ML Signal: Usage of move_on method with specific timestamp
        data_reader.move_on(to_timestamp=timestamp)
        # ⚠️ SAST Risk (Low): Hardcoded stock codes may lead to maintenance issues

        # 🧠 ML Signal: Assertion pattern for checking data presence in index
        df = data_reader.data_df

        # 🧠 ML Signal: Assertion pattern for checking data presence in index
        assert ("stock_sz_002572", timestamp) in df.index
        # ✅ Best Practice: Measuring execution time for performance testing
        # 🧠 ML Signal: Usage of move_on method with timeout parameter
        # ⚠️ SAST Risk (Low): Potential time-based assertion that could be unreliable
        assert ("stock_sz_000338", to_time_str(timestamp)) in df.index


def test_reader_move_on():
    data_reader = DataReader(
        data_schema=Stock1dKdata,
        entity_schema=Stock,
        entity_provider="eastmoney",
        codes=["002572", "000338"],
        start_timestamp="2019-06-13",
        end_timestamp="2019-06-14",
    )

    data_reader.move_on(to_timestamp="2019-06-15")
    assert ("stock_sz_002572", "2019-06-15") not in data_reader.data_df.index
    assert ("stock_sz_000338", "2019-06-15") not in data_reader.data_df.index

    start_time = time.time()
    data_reader.move_on(to_timestamp="2019-06-20", timeout=5)
    assert time.time() - start_time < 5