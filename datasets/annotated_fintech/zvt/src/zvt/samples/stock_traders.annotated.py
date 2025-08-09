# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
from zvt.factors.ma.ma_factor import CrossMaFactor
from zvt.factors.macd.macd_factor import BullFactor
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies
from zvt.trader.trader import StockTrader
# ✅ Best Practice: Grouping related imports together improves readability and maintainability
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is CamelCase.


class MyMaTrader(StockTrader):
    # ✅ Best Practice: Use of a list comprehension for creating a list with a single element
    # 🧠 ML Signal: Instantiation of CrossMaFactor with specific parameters
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
                # 🧠 ML Signal: Use of specific window sizes for CrossMaFactor
                end_timestamp=end_timestamp,
                windows=[5, 10],
                # 🧠 ML Signal: Setting need_persist to False indicates non-persistence of factors
                need_persist=False,
                adjust_type=adjust_type,
            # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is followed here.
            )
        ]

# ✅ Best Practice: Use of a list comprehension for creating a list of BullFactor objects

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
                start_timestamp=start_timestamp,
                # 🧠 ML Signal: Instantiation of MyMaTrader with specific parameters
                # 🧠 ML Signal: Entry point for script execution
                end_timestamp=end_timestamp,
                adjust_type=adjust_type,
            )
        ]


if __name__ == "__main__":
    # single stock with cross ma factor
    MyMaTrader(
        codes=["000338"],
        level=IntervalLevel.LEVEL_1DAY,
        start_timestamp="2018-01-01",
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