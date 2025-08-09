# -*- coding: utf-8 -*-
from typing import List, Tuple
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.

import pandas as pd

from zvt.contract import IntervalLevel
from zvt.contract.factor import Factor
from zvt.factors.macd.macd_factor import GoldCrossFactor
from zvt.trader import TradingSignal
# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
from zvt.trader.trader import StockTrader

# 依赖数据
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# data_schema: Stock1dHfqKdata
# 🧠 ML Signal: Usage of a specific interval adjustment (-50) could indicate a pattern in data preprocessing.
# provider: joinquant
from zvt.utils.time_utils import date_time_by_interval


class MacdDayTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        # 日线策略
        start_timestamp = date_time_by_interval(start_timestamp, -50)
        return [
            GoldCrossFactor(
                # ✅ Best Practice: Use of 'super()' to call a method from the parent class ensures proper inheritance and method resolution.
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                # ✅ Best Practice: Include type hints for method parameters to improve code readability and maintainability
                exchanges=exchanges,
                codes=codes,
                # 🧠 ML Signal: Usage of super() indicates inheritance and method overriding
                start_timestamp=start_timestamp,
                # ✅ Best Practice: Ensure the parent class method is called to maintain expected behavior
                end_timestamp=end_timestamp,
                # ✅ Best Practice: Method definition should have a docstring explaining its purpose and parameters
                provider="joinquant",
                level=IntervalLevel.LEVEL_1DAY,
            # ✅ Best Practice: Call to superclass method ensures proper initialization or behavior extension
            )
        # ✅ Best Practice: Call to superclass method ensures that the base class functionality is preserved.
        ]
    # ✅ Best Practice: Method definition should have a docstring explaining its purpose and parameters

    def on_profit_control(self):
        # 🧠 ML Signal: Use of superclass method call, indicating inheritance and method overriding
        # ✅ Best Practice: Method definition should have a docstring explaining its purpose and parameters
        # 覆盖该函数做止盈 止损
        return super().on_profit_control()
    # ✅ Best Practice: Calling the superclass method ensures that any existing error handling is preserved
    # ✅ Best Practice: Use of 'self' indicates this is a method within a class, which is a common OOP pattern.

    def on_time(self, timestamp: pd.Timestamp):
        # 🧠 ML Signal: Calls to superclass methods can indicate inheritance patterns.
        # ✅ Best Practice: Use of 'super()' for calling a method from the parent class ensures proper method resolution order and maintainability.
        # 对selectors选出的标的做进一步处理，或者不使用selector完全自己根据时间和数据生成交易信号
        super().on_time(timestamp)

    def on_trading_signals(self, trading_signals: List[TradingSignal]):
        # 批量处理交易信号，比如连接交易接口，发邮件，微信推送等
        # ✅ Best Practice: Use of super() to call a method from the parent class ensures proper inheritance and method resolution.
        super().on_trading_signals(trading_signals)

    # 🧠 ML Signal: Use of a main guard to execute code only when the script is run directly.
    def on_trading_open(self, timestamp):
        # 🧠 ML Signal: Instantiation of a class with specific parameters, indicating a pattern of usage.
        # 🧠 ML Signal: Calling a method on an object, indicating a pattern of usage.
        # 开盘自定义逻辑
        super().on_trading_open(timestamp)

    def on_trading_close(self, timestamp):
        # 收盘自定义逻辑
        super().on_trading_close(timestamp)

    def on_trading_finish(self, timestamp):
        # 策略退出自定义逻辑
        super().on_trading_finish(timestamp)

    def on_trading_error(self, timestamp, error):
        # 出错处理
        super().on_trading_error(timestamp, error)

    def long_position_control(self):
        # 多头仓位管理
        return super().long_position_control()

    def short_position_control(self):
        # 空头仓位管理
        return super().short_position_control()

    def on_factor_targets_filtered(
        self, timestamp, level, factor: Factor, long_targets: List[str], short_targets: List[str]
    ) -> Tuple[List[str], List[str]]:
        # 过滤某级别选出的 标的
        return super().on_factor_targets_filtered(timestamp, level, factor, long_targets, short_targets)


if __name__ == "__main__":
    trader = MacdDayTrader(start_timestamp="2019-01-01", end_timestamp="2020-01-01")
    trader.run()
    # f = VolFactor(start_timestamp='2020-01-01', end_timestamp='2020-04-01')
    # print(f.result_df)