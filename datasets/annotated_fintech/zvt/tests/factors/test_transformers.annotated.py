# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific classes or functions indicates usage patterns and dependencies
from zvt.api.kdata import get_kdata

# 🧠 ML Signal: Function definition for testing a specific transformer
from zvt.factors.algorithm import MaTransformer, MacdTransformer

# 🧠 ML Signal: Usage of a data retrieval function with specific parameters
# 🧠 ML Signal: Importing specific classes or functions indicates usage patterns and dependencies


def test_ma_transformer():
    df = get_kdata(
        entity_id="stock_sz_000338",
        start_timestamp="2019-01-01",
        provider="joinquant",
        index=["entity_id", "timestamp"],
        # 🧠 ML Signal: Instantiation of a transformer with specific parameters
    )
    # 🧠 ML Signal: Function definition for testing a transformer, useful for identifying test patterns

    # 🧠 ML Signal: Transformation of data using a transformer object
    # ✅ Best Practice: Use of print statement for outputting results in a test function
    # 🧠 ML Signal: Usage of a data retrieval function with specific parameters
    t = MaTransformer(windows=[5, 10])

    result_df = t.transform(df)

    print(result_df)


def test_MacdTransformer():
    # 🧠 ML Signal: Instantiation of a transformer object, indicating usage of transformation patterns
    # 🧠 ML Signal: Transformation operation on data, useful for understanding data processing workflows
    # ✅ Best Practice: Using print statements for debugging or simple output in test functions
    df = get_kdata(
        entity_id="stock_sz_000338",
        start_timestamp="2019-01-01",
        provider="joinquant",
        index=["entity_id", "timestamp"],
    )

    t = MacdTransformer()

    result_df = t.transform(df)

    print(result_df)
