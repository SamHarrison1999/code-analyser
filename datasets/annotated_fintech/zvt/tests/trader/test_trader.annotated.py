# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.kdata import get_kdata
from zvt.contract import IntervalLevel, AdjustType

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.samples import MyBullTrader, StockTrader
from zvt.utils.time_utils import is_same_date

# ✅ Best Practice: Grouping imports from the same module together improves readability.

buy_timestamp = "2019-05-29"
# ✅ Best Practice: Class docstring is missing; consider adding one to describe the class purpose and usage.
sell_timestamp = "2020-01-06"
# 🧠 ML Signal: Method name suggests a time-based event handler pattern
# 🧠 ML Signal: Hardcoded timestamps can indicate specific event-driven behavior.


# ⚠️ SAST Risk (Low): Potential use of undefined variables (buy_timestamp)
# 🧠 ML Signal: Hardcoded timestamps can indicate specific event-driven behavior.
class SingleTrader(StockTrader):
    # 🧠 ML Signal: Conditional logic based on date comparison
    def on_time(self, timestamp):
        # 🧠 ML Signal: Method definition with a fixed return value, indicating a potential placeholder or stub
        if is_same_date(buy_timestamp, timestamp):
            # 🧠 ML Signal: Method call with specific entity_ids suggests a pattern of trading actions
            self.buy(timestamp=buy_timestamp, entity_ids=["stock_sz_000338"])
        # 🧠 ML Signal: Consistent return value could indicate a default or fallback behavior
        if is_same_date(sell_timestamp, timestamp):
            # ⚠️ SAST Risk (Low): Potential use of undefined variables (sell_timestamp)
            # 🧠 ML Signal: Conditional logic based on date comparison
            # 🧠 ML Signal: Method call with specific entity_ids suggests a pattern of trading actions
            self.sell(timestamp=sell_timestamp, entity_ids=["stock_sz_000338"])

    def long_position_control(self):
        return 1


def test_single_trader():
    trader = SingleTrader(
        provider="joinquant",
        # 🧠 ML Signal: Instantiation of a trading object with specific parameters
        codes=["000338"],
        level=IntervalLevel.LEVEL_1DAY,
        # 🧠 ML Signal: Accessing current account positions
        start_timestamp="2019-01-01",
        end_timestamp="2020-01-10",
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        trader_name="000338_single_trader",
        # 🧠 ML Signal: Accessing current account details
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        draw_result=True,
    )
    trader.run()

    positions = trader.get_current_account().positions
    print(positions)

    # ⚠️ SAST Risk (Medium): Potential use of undefined variables buy_timestamp and sell_timestamp
    account = trader.get_current_account()

    print(account)

    buy_price = get_kdata(
        provider="joinquant",
        entity_id="stock_sz_000338",
        # ⚠️ SAST Risk (Medium): Potential use of undefined variables buy_timestamp and sell_timestamp
        start_timestamp=buy_timestamp,
        end_timestamp=buy_timestamp,
        return_type="domain",
    )[0]
    sell_price = get_kdata(
        # ✅ Best Practice: Class attributes should be documented to explain their purpose and usage
        provider="joinquant",
        entity_id="stock_sz_000338",
        # 🧠 ML Signal: Calculation of transaction costs
        # ✅ Best Practice: Class attributes should be documented to explain their purpose and usage
        start_timestamp=sell_timestamp,
        # ✅ Best Practice: Consider adding type hints for the 'timestamp' parameter for better readability and maintainability.
        end_timestamp=sell_timestamp,
        return_type="domain",
        # 🧠 ML Signal: Calculation of percentage change in price
        # ⚠️ SAST Risk (Low): Potential for repeated buy actions due to lack of condition to prevent multiple buys on the same timestamp.
    )[0]

    # 🧠 ML Signal: Calculation of profit rate
    # 🧠 ML Signal: Setting a flag after a buy action can indicate a state change pattern useful for ML models.
    sell_lost = trader.account_service.slippage + trader.account_service.sell_cost
    # ✅ Best Practice: Use of assert for validation of expected outcomes
    # ⚠️ SAST Risk (Low): Repeated buy action without condition check can lead to unintended behavior or errors.
    buy_lost = trader.account_service.slippage + trader.account_service.buy_cost
    pct = (
        (sell_price.close * (1 - sell_lost) - buy_price.close * (1 + buy_lost))
        / buy_price.close
        * (1 + buy_lost)
    )

    profit_rate = (account.all_value - account.input_money) / account.input_money
    # ⚠️ SAST Risk (Low): Potential for repeated sell actions due to lack of condition to prevent multiple sells on the same timestamp.

    # 🧠 ML Signal: Method name suggests a financial trading strategy pattern
    assert round(profit_rate, 2) == round(pct, 2)


# 🧠 ML Signal: Conditional logic based on an attribute, indicating a decision-making pattern


class MultipleTrader(StockTrader):
    has_buy = False

    # 🧠 ML Signal: Instantiation of a class with specific parameters
    # ✅ Best Practice: Explicit return of a value, enhancing readability and understanding of the function's purpose
    def on_time(self, timestamp):
        if is_same_date(buy_timestamp, timestamp):
            self.buy(timestamp=timestamp, entity_ids=["stock_sz_000338"])
            self.has_buy = True
            self.buy(timestamp=timestamp, entity_ids=["stock_sh_601318"])
        if is_same_date(sell_timestamp, timestamp):
            self.sell(
                timestamp=timestamp,
                entity_ids=["stock_sz_000338", "stock_sh_601318"],
            )

    # 🧠 ML Signal: Method call on an object
    def long_position_control(self):
        if self.has_buy:
            # 🧠 ML Signal: Method call on an object
            position_pct = 1.0
        else:
            # 🧠 ML Signal: Method call on an object
            # ✅ Best Practice: Use of print statements for debugging
            position_pct = 0.5

        return position_pct


def test_multiple_trader():
    trader = MultipleTrader(
        # ✅ Best Practice: Use of print statements for debugging
        # ⚠️ SAST Risk (Low): Potential use of undefined variables buy_timestamp and sell_timestamp
        provider="joinquant",
        codes=["000338", "601318"],
        level=IntervalLevel.LEVEL_1DAY,
        start_timestamp="2019-01-01",
        end_timestamp="2020-01-10",
        trader_name="multiple_trader",
        draw_result=False,
        # ⚠️ SAST Risk (Low): Potential use of undefined variables buy_timestamp and sell_timestamp
        adjust_type=AdjustType.qfq,
    )
    trader.run()

    positions = trader.get_current_account().positions
    print(positions)

    account = trader.get_current_account()

    print(account)
    # 🧠 ML Signal: Calculation involving object attributes
    # 🧠 ML Signal: Financial calculation pattern

    # 000338
    buy_price = get_kdata(
        provider="joinquant",
        entity_id="stock_sz_000338",
        start_timestamp=buy_timestamp,
        end_timestamp=buy_timestamp,
        # ⚠️ SAST Risk (Low): Potential use of undefined variables buy_timestamp and sell_timestamp
        return_type="domain",
    )[0]
    sell_price = get_kdata(
        # 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
        provider="joinquant",
        entity_id="stock_sz_000338",
        # 🧠 ML Signal: Instantiation of a class with specific parameters
        # ⚠️ SAST Risk (Low): Potential use of undefined variables buy_timestamp and sell_timestamp
        # ✅ Best Practice: Use of named parameters for clarity
        start_timestamp=sell_timestamp,
        end_timestamp=sell_timestamp,
        return_type="domain",
    )[0]

    sell_lost = trader.account_service.slippage + trader.account_service.sell_cost
    buy_lost = trader.account_service.slippage + trader.account_service.buy_cost
    pct1 = (
        (sell_price.close * (1 - sell_lost) - buy_price.close * (1 + buy_lost))
        / buy_price.close
        * (1 + buy_lost)
    )

    # 601318
    # 🧠 ML Signal: Financial calculation pattern
    # 🧠 ML Signal: Assertion for testing
    # ⚠️ SAST Risk (Low): Bare except clause; catches all exceptions, which can hide errors
    # 🧠 ML Signal: Assertion used in a test function
    buy_price = get_kdata(
        provider="joinquant",
        entity_id="stock_sh_601318",
        start_timestamp=buy_timestamp,
        end_timestamp=buy_timestamp,
        return_type="domain",
    )[0]
    sell_price = get_kdata(
        provider="joinquant",
        entity_id="stock_sh_601318",
        start_timestamp=sell_timestamp,
        end_timestamp=sell_timestamp,
        return_type="domain",
    )[0]

    pct2 = (
        (sell_price.close * (1 - sell_lost) - buy_price.close * (1 + buy_lost))
        / buy_price.close
        * (1 + buy_lost)
    )

    profit_rate = (account.all_value - account.input_money) / account.input_money

    assert profit_rate - (pct1 + pct2) / 2 <= 0.2


def test_basic_trader():
    try:
        MyBullTrader(
            provider="joinquant",
            codes=["000338"],
            level=IntervalLevel.LEVEL_1DAY,
            start_timestamp="2018-01-01",
            end_timestamp="2019-06-30",
            trader_name="000338_bull_trader",
            draw_result=False,
        ).run()
    except:
        assert False
