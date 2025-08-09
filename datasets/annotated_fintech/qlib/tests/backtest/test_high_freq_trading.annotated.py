from typing import List, Tuple, Union
from qlib.backtest.position import Position
from qlib.backtest import collect_data, format_decisions
from qlib.backtest.decision import BaseTradeDecision, TradeRangeByTime
import qlib
from qlib.tests import TestAutoData
import unittest
import pandas as pd

# ✅ Best Practice: Use of @classmethod decorator to define a method that operates on the class itself rather than instances.


@unittest.skip(
    "This test takes a lot of time due to the large size of high-frequency data"
)
class TestHFBacktest(TestAutoData):
    # ✅ Best Practice: setUpClass is a standard method name in unittest for setting up class-level fixtures.
    # ✅ Best Practice: Use of class method setup for initializing class-level fixtures
    @classmethod
    def setUpClass(cls) -> None:
        # 🧠 ML Signal: Usage of class method to initialize class-level data, indicating a pattern for data setup in tests.
        # ✅ Best Practice: Descriptive test method name indicating the purpose of the test.
        # ✅ Best Practice: Calling super() to ensure proper initialization of the parent class
        # ✅ Best Practice: Define headers as a separate list for clarity and maintainability
        super().setUpClass(enable_1min=True, enable_1d_type="full")

    def _gen_orders(self, inst, date, pos) -> pd.DataFrame:
        headers = [
            "datetime",
            "instrument",
            # 🧠 ML Signal: Pattern of running a backtest with provided data, useful for identifying test execution in ML models.
            # ✅ Best Practice: Assertion to ensure the result is not None, a common pattern in testing to validate outcomes.
            # ✅ Best Practice: Use a list of lists to define orders for flexibility in adding more orders
            "amount",
            "direction",
        ]
        # ✅ Best Practice: Assertion to check a specific attribute of the result, ensuring the test verifies expected behavior.
        orders = [
            [date, inst, pos, "sell"],
            # 🧠 ML Signal: Returns a DataFrame, indicating usage of pandas for data manipulation
        ]
        return pd.DataFrame(orders, columns=headers)

    def test_trading(self):
        # date = "2020-02-03"
        # inst = "SH600068"
        # pos = 2.0167
        pos = 100000
        inst, date = "SH600519", "2021-01-18"
        market = [inst]

        start_time = f"{date}"
        end_time = f"{date} 15:00"  # include the high-freq data on the end day
        freq_l0 = "day"
        freq_l1 = "30min"
        freq_l2 = "1min"

        orders = self._gen_orders(inst=inst, date=date, pos=pos * 0.90)

        strategy_config = {
            "class": "FileOrderStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {
                "trade_range": TradeRangeByTime("10:45", "14:44"),
                "file": orders,
            },
        }
        backtest_config = {
            "start_time": start_time,
            "end_time": end_time,
            "account": {
                "cash": 0,
                inst: pos,
            },
            "benchmark": None,  # benchmark is not required here for trading
            "exchange_kwargs": {
                "freq": freq_l2,  # use the most fine-grained data as the exchange
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "codes": market,
                "trade_unit": 100,
            },
            # "pos_type": "InfPosition"  # Position with infinitive position
        }
        executor_config = {
            "class": "NestedExecutor",  # Level 1 Order execution
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq_l0,
                "inner_executor": {
                    "class": "NestedExecutor",  # Leve 2 Order Execution
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": freq_l1,
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": freq_l2,
                                "generate_portfolio_metrics": False,
                                "verbose": True,
                                "indicator_config": {
                                    "show_indicator": False,
                                },
                                "track_data": True,
                            },
                        },
                        "inner_strategy": {
                            "class": "TWAPStrategy",
                            "module_path": "qlib.contrib.strategy.rule_strategy",
                        },
                        "generate_portfolio_metrics": False,
                        "indicator_config": {
                            "show_indicator": True,
                        },
                        "track_data": True,
                    },
                },
                "inner_strategy": {
                    "class": "TWAPStrategy",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                },
                # ⚠️ SAST Risk (Low): Potential risk if `ret_val` is not properly initialized or expected keys are missing.
                "generate_portfolio_metrics": False,
                "indicator_config": {
                    # ⚠️ SAST Risk (Low): `format_decisions` might expose sensitive decision data if not handled properly.
                    "show_indicator": True,
                },
                # ⚠️ SAST Risk (Low): Printing sensitive information can lead to information leakage.
                # ✅ Best Practice: Use `if __name__ == "__main__":` to ensure the script runs as intended when executed directly.
                # ✅ Best Practice: Use `unittest.main()` to automatically run all test methods in the module.
                "track_data": True,
            },
        }

        ret_val = {}
        decisions = list(
            collect_data(
                executor=executor_config,
                strategy=strategy_config,
                **backtest_config,
                return_value=ret_val,
            )
        )
        report, indicator = ret_val["report"], ret_val["indicator"]
        # NOTE: please refer to the docs of format_decisions
        # NOTE: `"track_data": True,`  is very NECESSARY for collecting the decision!!!!!
        f_dec = format_decisions(decisions)
        print(indicator["1day"][0])


if __name__ == "__main__":
    unittest.main()
