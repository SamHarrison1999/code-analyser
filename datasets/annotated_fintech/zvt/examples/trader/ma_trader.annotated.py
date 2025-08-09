# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
from zvt.factors.ma.ma_factor import CrossMaFactor
from zvt.factors.macd.macd_factor import BullFactor
# 🧠 ML Signal: Importing specific classes from a module indicates which functionalities are being utilized.

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# 🧠 ML Signal: Custom class inheriting from a base class, indicating usage of OOP patterns
from zvt.trader.trader import StockTrader


# 🧠 ML Signal: Usage of a list to store and return objects, indicating a pattern of handling multiple factors.
# 🧠 ML Signal: Instantiation of CrossMaFactor with specific parameters, indicating a pattern of factor initialization.
class MyMaTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        return [
            CrossMaFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                exchanges=exchanges,
                codes=codes,
                start_timestamp=start_timestamp,
                # 🧠 ML Signal: Use of specific window sizes, which could be a pattern for model training.
                end_timestamp=end_timestamp,
                windows=[5, 10],
                # 🧠 ML Signal: Setting need_persist to False, indicating a pattern of non-persistent factor usage.
                # ✅ Best Practice: Class definition should include a docstring explaining its purpose
                # 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
                need_persist=False,
            )
        ]
# 🧠 ML Signal: Use of a list to store and return initialized factors
# 🧠 ML Signal: Instantiation of BullFactor with specific parameters


class MyBullTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        return [
            BullFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                exchanges=exchanges,
                codes=codes,
                # 🧠 ML Signal: Instantiation of MyBullTrader with specific parameters
                # ✅ Best Practice: Use of __name__ == "__main__" to allow or prevent parts of code from being run when the modules are imported
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                adjust_type="hfq",
            )
        ]


if __name__ == "__main__":
    # single stock with cross ma factor
    MyBullTrader(
        codes=["000338"],
        level=IntervalLevel.LEVEL_1DAY,
        start_timestamp="2019-01-01",
        end_timestamp="2019-06-30",
        trader_name="000338_ma_trader",
    ).run()

    # single stock with bull factor
    # MyBullTrader(codes=['000338'], level=IntervalLevel.LEVEL_1DAY, start_timestamp='2018-01-01',
    #              end_timestamp='2019-06-30', trader_name='000338_bull_trader').run()

    #  multiple stocks with cross ma factor
    # MyMaTrader(codes=SAMPLE_STOCK_CODES, level=IntervalLevel.LEVEL_1DAY, start_timestamp='2018-01-01',
    #            end_timestamp='2019-06-30', trader_name='sample_stocks_ma_trader').run()

    # multiple stocks with bull factor
    # MyBullTrader(codes=SAMPLE_STOCK_CODES, level=IntervalLevel.LEVEL_1DAY, start_timestamp='2018-01-01',
    #              end_timestamp='2019-06-30', trader_name='sample_stocks_bull_trader').run()