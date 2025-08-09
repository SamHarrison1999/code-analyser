# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies
from zvt.api.intent import compare
from zvt.domain import Indexus1dKdata, Index, Indexus, Index1dKdata, Currency1dKdata
# 🧠 ML Signal: Importing specific classes from a module indicates which data models are being used
# 🧠 ML Signal: Function definition with a specific name indicating a comparison between China and US stock indices
from zvt.domain import TreasuryYield

# 🧠 ML Signal: Importing specific classes from a module indicates which data models are being used
# 🧠 ML Signal: Method call to record data, indicating a pattern of data collection

def china_vs_us_stock():
    # 🧠 ML Signal: Method call to record data, indicating a pattern of data collection
    # 上证，道琼斯指数
    Index.record_data()
    # 🧠 ML Signal: Method call with specific entity_id, indicating a pattern of targeted data collection
    Indexus.record_data()
    # 🧠 ML Signal: Hardcoded entity IDs can indicate specific data sources or entities of interest.
    Index1dKdata.record_data(entity_id="index_sh_000001")
    # 🧠 ML Signal: Method call with specific entity_id, indicating a pattern of targeted data collection
    # 🧠 ML Signal: Function call with specific parameters for comparison, indicating a pattern of data analysis
    # 🧠 ML Signal: The use of a fixed start timestamp can indicate a consistent analysis period.
    # ✅ Best Practice: Use of named parameters improves code readability and maintainability.
    Indexus1dKdata.record_data(entity_id="indexus_us_SPX")
    compare(entity_ids=["index_sh_000001", "indexus_us_SPX"], start_timestamp="2000-01-01", scale_value=100)


def us_yield_and_stock():
    # 美债收益率，道琼斯指数
    # 🧠 ML Signal: Function definition with a specific purpose, useful for understanding code structure
    entity_ids = ["country_galaxy_US", "indexus_us_SPX"]
    # 🧠 ML Signal: Mapping specific columns to schemas can indicate the structure of the data being analyzed.
    compare(
        # 🧠 ML Signal: Use of specific entity IDs indicating domain-specific data handling
        # 🧠 ML Signal: Function call with parameters, useful for understanding function usage patterns
        entity_ids=entity_ids,
        start_timestamp="1990-01-01",
        scale_value=None,
        # ✅ Best Practice: Use of named parameters improves code readability
        schema_map_columns={TreasuryYield: ["yield_2", "yield_5"], Indexus1dKdata: ["close"]},
    )
# 🧠 ML Signal: Function definition with a specific name pattern


# 🧠 ML Signal: List of entity IDs indicating specific domain knowledge
# 🧠 ML Signal: Function call with specific parameters
def commodity_and_stock():
    # 江西铜业，沪铜
    entity_ids = ["stock_sh_600362", "future_shfe_CU"]
    # ✅ Best Practice: Use of named parameters for clarity
    compare(
        entity_ids=entity_ids,
        # 🧠 ML Signal: Function definition with a specific purpose, useful for understanding code intent
        start_timestamp="2005-01-01",
        scale_value=100,
    # 🧠 ML Signal: Function call with multiple parameters, indicating a pattern of usage
    # 🧠 ML Signal: Use of a list to store multiple entity IDs
    # ✅ Best Practice: Use of named parameters for clarity
    )


def compare_metal():
    # 沪铜,沪铝,螺纹钢
    entity_ids = ["future_shfe_CU", "future_shfe_AL", "future_shfe_RB"]
    # 🧠 ML Signal: Function definition with a specific purpose, useful for understanding code intent
    compare(
        entity_ids=entity_ids,
        # 🧠 ML Signal: Use of a dictionary to map schema to columns
        # 🧠 ML Signal: Method call with specific parameters, indicating data recording behavior
        start_timestamp="2009-04-01",
        # ⚠️ SAST Risk (Low): Hardcoded entity_id could lead to inflexibility or misuse if not validated
        # 🧠 ML Signal: List of entity IDs, indicating the entities involved in the comparison
        scale_value=100,
    )


# 🧠 ML Signal: Function call with multiple parameters, indicating a comparison operation
def compare_udi_and_stock():
    # ⚠️ SAST Risk (Low): Hardcoded start_timestamp could lead to inflexibility or misuse if not validated
    # 美股指数
    # ✅ Best Practice: Use of descriptive parameter names improves readability
    # Indexus.record_data()
    # 🧠 ML Signal: Mapping schema to columns, indicating data structure usage
    # 🧠 ML Signal: Entry point for script execution, indicating main functionality
    # 🧠 ML Signal: Function call within main guard, indicating primary action
    entity_ids = ["indexus_us_NDX", "indexus_us_SPX", "indexus_us_UDI"]
    # Indexus1dKdata.record_data(entity_ids=entity_ids, sleeping_time=0)
    compare(
        entity_ids=entity_ids,
        start_timestamp="2015-01-01",
        scale_value=100,
        schema_map_columns={Indexus1dKdata: ["close"]},
    )


def compare_cny_and_stock():
    Currency1dKdata.record_data(entity_id="currency_forex_USDCNYC")
    entity_ids = ["index_sh_000001", "currency_forex_USDCNYC"]
    compare(
        entity_ids=entity_ids,
        start_timestamp="2005-01-01",
        scale_value=100,
        schema_map_columns={Currency1dKdata: ["close"], Index1dKdata: ["close"]},
    )


if __name__ == "__main__":
    # compare_kline()
    # us_yield_and_stock()
    # commodity_and_stock()
    # compare_metal()
    # compare_udi_and_stock()
    compare_cny_and_stock()