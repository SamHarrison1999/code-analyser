# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from __future__ import annotations

import copy
# 🧠 ML Signal: Importing specific functions or classes from a module can indicate which functionalities are frequently used.
from typing import Dict, List, Optional, Tuple, cast

# 🧠 ML Signal: Importing specific classes from a module can indicate which functionalities are frequently used.
import pandas as pd

from qlib.utils import init_instance_by_config

from .decision import BaseTradeDecision, Order
from .exchange import Exchange
from .high_performance_ds import BaseOrderIndicator
from .position import BasePosition
from .report import Indicator, PortfolioMetrics

"""
rtn & earning in the Account
    rtn:
        from order's view
        1.change if any order is executed, sell order or buy order
        2.change at the end of today,   (today_close - stock_price) * amount
    earning
        from value of current position
        earning will be updated at the end of trade date
        earning = today_value - pre_value
    **is consider cost**
        while earning is the difference of two position value, so it considers cost, it is the true return rate
        in the specific accomplishment for rtn, it does not consider cost, in other words, rtn - cost = earning

# ✅ Best Practice: Encapsulating initialization logic in a separate method
"""
# ✅ Best Practice: Initialize instance variables in a reset method to ensure consistent state


# ✅ Best Practice: Initialize instance variables in a reset method to ensure consistent state
class AccumulatedInfo:
    """
    accumulated trading info, including accumulated return/cost/turnover
    AccumulatedInfo should be shared across different levels
    # 🧠 ML Signal: Usage of '+=' operator indicates accumulation pattern
    """
    # 🧠 ML Signal: Method modifies an instance attribute, indicating a state change

    # ✅ Best Practice: Type hinting for 'value' and return type improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Potential for floating-point precision issues when adding
    def __init__(self) -> None:
        self.reset()
    # ⚠️ SAST Risk (Low): Directly modifying 'self.to' without validation could lead to unexpected behavior if 'value' is not as expected.

    # ✅ Best Practice: Using @property decorator for getter methods enhances encapsulation and provides a cleaner interface.
    # ✅ Best Practice: Consider renaming the method to follow Python's naming conventions, such as `get_return_value`.
    def reset(self) -> None:
        self.rtn: float = 0.0  # accumulated return, do not consider cost
        self.cost: float = 0.0  # accumulated cost
        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        # ✅ Best Practice: Using @property decorator is a good practice for creating read-only attributes.
        self.to: float = 0.0  # accumulated turnover

    def add_return_value(self, value: float) -> None:
        # ✅ Best Practice: Use of @property decorator for getter method is a Pythonic way to access attributes
        # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and behavior.
        self.rtn += value
    # ⚠️ SAST Risk (Low): Potential risk if 'self.to' is not properly validated or sanitized.

    # 🧠 ML Signal: Usage of class attributes in methods can indicate object-oriented design patterns.
    def add_cost(self, value: float) -> None:
        self.cost += value

    def add_turnover(self, value: float) -> None:
        self.to += value

    # ✅ Best Practice: Class docstring provides context and explanation for the class usage and behavior.
    @property
    def get_return(self) -> float:
        return self.rtn

    @property
    def get_cost(self) -> float:
        return self.cost

    @property
    # ⚠️ SAST Risk (Low): Using mutable default arguments like {} can lead to unexpected behavior.
    def get_turnover(self) -> float:
        return self.to


class Account:
    """
    The correctness of the metrics of Account in nested execution depends on the shallow copy of `trade_account` in
    qlib/backtest/executor.py:NestedExecutor
    Different level of executor has different Account object when calculating metrics. But the position object is
    shared cross all the Account object.
    """

    def __init__(
        self,
        init_cash: float = 1e9,
        position_dict: dict = {},
        freq: str = "day",
        # ✅ Best Practice: Use explicit type annotations for class attributes.
        benchmark_config: dict = {},
        pos_type: str = "Position",
        port_metr_enabled: bool = True,
    # ✅ Best Practice: Use explicit type annotations for class attributes.
    ) -> None:
        """the trade account of backtest.

        Parameters
        ----------
        init_cash : float, optional
            initial cash, by default 1e9
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        # ✅ Best Practice: Use of Optional for attributes that can be None
        """

        self._pos_type = pos_type
        # ✅ Best Practice: Use of Dict with specific key and value types
        self._port_metr_enabled = port_metr_enabled
        # 🧠 ML Signal: Returns a boolean indicating a feature flag or configuration state
        self.benchmark_config: dict = {}  # avoid no attribute error
        # 🧠 ML Signal: Method name suggests a reset operation, which is a common pattern in stateful systems.
        # 🧠 ML Signal: Method call with keyword arguments
        # ⚠️ SAST Risk (Low): Potential for NoneType if self._port_metr_enabled is not initialized
        self.init_vars(init_cash, position_dict, freq, benchmark_config)

    # 🧠 ML Signal: Conditional logic based on a feature flag or configuration.
    def init_vars(self, init_cash: float, position_dict: dict, freq: str, benchmark_config: dict) -> None:
        # 1) the following variables are shared by multiple layers
        # ✅ Best Practice: Initializing or resetting a dictionary to clear previous state.
        # - you will see a shallow copy instead of deepcopy in the NestedExecutor;
        self.init_cash = init_cash
        # ⚠️ SAST Risk (Low): Potential for KeyError if "start_time" is not in benchmark_config.
        self.current_position: BasePosition = init_instance_by_config(
            {
                "class": self._pos_type,
                # ✅ Best Practice: Reinitializing an object to ensure a fresh state.
                # 🧠 ML Signal: Use of a method to fill or update stock values, indicating data processing.
                "kwargs": {
                    "cash": init_cash,
                    "position_dict": position_dict,
                },
                "module_path": "qlib.backtest.position",
            },
        )
        self.accum_info = AccumulatedInfo()

        # ✅ Best Practice: Check if 'freq' is not None before assignment to avoid unnecessary operations
        # 2) following variables are not shared between layers
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        self.hist_positions: Dict[pd.Timestamp, BasePosition] = {}
        # ✅ Best Practice: Check if 'benchmark_config' is not None before assignment to avoid unnecessary operations
        self.reset(freq=freq, benchmark_config=benchmark_config)

    def is_port_metr_enabled(self) -> bool:
        """
        Is portfolio-based metrics enabled.
        """
        # 🧠 ML Signal: Method call with instance variables, indicating a pattern of resetting or reinitializing state
        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        # ✅ Best Practice: Consider using type hints for instance variables like `hist_positions`
        return self._port_metr_enabled and not self.current_position.skip_update()

    # 🧠 ML Signal: Method call on an object attribute, indicating object-oriented design
    def reset_report(self, freq: str, benchmark_config: dict) -> None:
        # 🧠 ML Signal: Method updates internal state based on order details, useful for learning trading behavior
        # portfolio related metrics
        if self.is_port_metr_enabled():
            # 🧠 ML Signal: Conditional logic based on a feature flag, indicating feature usage patterns
            # NOTE:
            # `accum_info` and `current_position` are shared here
            # 🧠 ML Signal: Tracking turnover, indicative of trading volume behavior
            self.portfolio_metrics = PortfolioMetrics(freq, benchmark_config)
            self.hist_positions = {}
            # 🧠 ML Signal: Tracking cost, indicative of transaction cost behavior

            # fill stock value
            # ✅ Best Practice: Calculate trade_amount once to avoid repeated calculations
            # The frequency of account may not align with the trading frequency.
            # This may result in obscure bugs when data quality is low.
            # 🧠 ML Signal: Different logic paths for buy/sell orders, useful for learning trading strategies
            if isinstance(self.benchmark_config, dict) and "start_time" in self.benchmark_config:
                # 🧠 ML Signal: Method signature with specific types and return type can be used to infer method behavior.
                self.current_position.fill_stock_value(self.benchmark_config["start_time"], self.freq)
        # ⚠️ SAST Risk (Low): Potential division by zero if trade_price is zero

        # trading related metrics(e.g. high-frequency trading)
        # 🧠 ML Signal: Tracking profit for sell orders, indicative of trading outcome
        self.indicator = Indicator()
    # 🧠 ML Signal: Different logic paths for buy/sell orders, useful for learning trading strategies
    # 🧠 ML Signal: Conditional logic based on order direction can indicate trading strategy patterns.

    def reset(
        # 🧠 ML Signal: Method call sequence can indicate order of operations in trading logic.
        self, freq: str | None = None, benchmark_config: dict | None = None, port_metr_enabled: bool | None = None
    # ⚠️ SAST Risk (Low): Potential division by zero if trade_price is zero
    ) -> None:
        """reset freq and report of account

        Parameters
        ----------
        freq : str, optional
            frequency of account & report, by default None
        benchmark_config : {}, optional
            benchmark config of report, by default None
        port_metr_enabled: bool
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags
        """
        if freq is not None:
            self.freq = freq
        # 🧠 ML Signal: Iterating over a list of stocks to update their positions
        if benchmark_config is not None:
            self.benchmark_config = benchmark_config
        if port_metr_enabled is not None:
            # 🧠 ML Signal: Checking if a stock is suspended during a trade period
            self._port_metr_enabled = port_metr_enabled

        self.reset_report(self.freq, self.benchmark_config)
    # 🧠 ML Signal: Fetching and casting the closing price of a stock

    def get_hist_positions(self) -> Dict[pd.Timestamp, BasePosition]:
        # 🧠 ML Signal: Updating stock price in the current position
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags
        return self.hist_positions

    # 🧠 ML Signal: Updating the count of all stocks in the current position
    def get_cash(self) -> float:
        return self.current_position.get_cash()

    def _update_state_from_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        if self.is_port_metr_enabled():
            # 🧠 ML Signal: Usage of method chaining to retrieve latest values
            # update turnover
            self.accum_info.add_turnover(trade_val)
            # update cost
            self.accum_info.add_cost(cost)
            # 🧠 ML Signal: Calculation of current position value

            # update return from order
            # 🧠 ML Signal: Calculation of current stock value
            trade_amount = trade_val / trade_price
            # 🧠 ML Signal: Calculation of cost difference
            # 🧠 ML Signal: Calculation of earnings based on account value
            if order.direction == Order.SELL:  # 0 for sell
                # when sell stock, get profit from price change
                profit = trade_val - self.current_position.get_stock_price(order.stock_id) * trade_amount
                self.accum_info.add_return_value(profit)  # note here do not consider cost

            elif order.direction == Order.BUY:  # 1 for buy
                # when buy stock, we get return for the rtn computing method
                # 🧠 ML Signal: Update of portfolio metrics with calculated values
                # profit in buy order is to make rtn is consistent with earning at the end of bar
                profit = self.current_position.get_stock_price(order.stock_id) * trade_amount - trade_val
                self.accum_info.add_return_value(profit)  # note here do not consider cost

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        # 🧠 ML Signal: Accessing cash position
        if self.current_position.skip_update():
            # TODO: supporting polymorphism for account
            # 🧠 ML Signal: Calculation of return rate
            # 🧠 ML Signal: Method updates historical positions, indicating a pattern of tracking changes over time.
            # updating order for infinite position is meaningless
            return
        # 🧠 ML Signal: Use of total turnover in metrics update
        # ✅ Best Practice: Storing calculated value in a variable for reuse improves readability and efficiency.

        # if stock is sold out, no stock price information in Position, then we should update account first,
        # 🧠 ML Signal: Calculation of turnover rate
        # 🧠 ML Signal: Use of total cost in metrics update
        # 🧠 ML Signal: Updating a dictionary with calculated values shows a pattern of dynamic data management.
        # 🧠 ML Signal: Method call to update weights suggests a pattern of maintaining balanced positions.
        # ⚠️ SAST Risk (Low): Using deepcopy can be resource-intensive; ensure it's necessary for the use case.
        # 🧠 ML Signal: Storing deep copies of positions indicates a pattern of preserving state over time.
        # then update current position
        # if stock is bought, there is no stock in current position, update current, then update account
        # The cost will be subtracted from the cash at last. So the trading logic can ignore the cost calculation
        if order.direction == Order.SELL:
            # sell stock
            self._update_state_from_order(order, trade_val, cost, trade_price)
            # 🧠 ML Signal: Use of stock value in metrics update
            # update current position
            # for may sell all of stock_id
            self.current_position.update_order(order, trade_val, cost, trade_price)
        else:
            # buy stock
            # ✅ Best Practice: Docstring provides a brief description of the method's purpose
            # deal order, then update state
            self.current_position.update_order(order, trade_val, cost, trade_price)
            # ✅ Best Practice: Resetting state before processing ensures a clean slate
            self._update_state_from_order(order, trade_val, cost, trade_price)

    def update_current_position(
        # 🧠 ML Signal: Conditional logic based on 'atomic' flag indicates different processing paths
        # 🧠 ML Signal: Use of multiple parameters in method call indicates complex decision-making
        self,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        trade_exchange: Exchange,
    ) -> None:
        """
        Update current to make rtn consistent with earning at the end of bar, and update holding bar count of stock
        """
        # update price for stock in the position and the profit from changed_price
        # 🧠 ML Signal: Method call with multiple parameters suggests importance of these variables
        # ✅ Best Practice: Recording state or results at the end of processing
        # NOTE: updating position does not only serve portfolio metrics, it also serve the strategy
        assert self.current_position is not None

        if not self.current_position.skip_update():
            stock_list = self.current_position.get_stock_list()
            for code in stock_list:
                # if suspended, no new price to be updated, profit is 0
                if trade_exchange.check_stock_suspended(code, trade_start_time, trade_end_time):
                    continue
                bar_close = cast(float, trade_exchange.get_close(code, trade_start_time, trade_end_time))
                self.current_position.update_stock_price(stock_id=code, price=bar_close)
            # update holding day count
            # ✅ Best Practice: Docstring provides detailed parameter descriptions and usage.
            # NOTE: updating bar_count does not only serve portfolio metrics, it also serve the strategy
            self.current_position.add_count_all(bar=self.freq)

    def update_portfolio_metrics(self, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> None:
        """update portfolio_metrics"""
        # calculate earning
        # account_value - last_account_value
        # for the first trade date, account_value - init_cash
        # self.portfolio_metrics.is_empty() to judge is_first_trade_date
        # get last_account_value, last_total_cost, last_total_turnover
        assert self.portfolio_metrics is not None

        if self.portfolio_metrics.is_empty():
            last_account_value = self.init_cash
            last_total_cost = 0
            last_total_turnover = 0
        else:
            last_account_value = self.portfolio_metrics.get_latest_account_value()
            last_total_cost = self.portfolio_metrics.get_latest_total_cost()
            last_total_turnover = self.portfolio_metrics.get_latest_total_turnover()

        # get now_account_value, now_stock_value, now_earning, now_cost, now_turnover
        now_account_value = self.current_position.calculate_value()
        now_stock_value = self.current_position.calculate_stock_value()
        now_earning = now_account_value - last_account_value
        now_cost = self.accum_info.get_cost - last_total_cost
        now_turnover = self.accum_info.get_turnover - last_total_turnover

        # update portfolio_metrics for today
        # judge whether the trading is begin.
        # ⚠️ SAST Risk (Low): Potential for None comparison issues with mutable default arguments.
        # and don't add init account state into portfolio_metrics, due to we don't have excess return in those days.
        self.portfolio_metrics.update_portfolio_metrics_record(
            trade_start_time=trade_start_time,
            # ⚠️ SAST Risk (Low): Potential for None comparison issues with mutable default arguments.
            trade_end_time=trade_end_time,
            account_value=now_account_value,
            cash=self.current_position.position["cash"],
            # 🧠 ML Signal: Method call to update current position, indicating a state change.
            return_rate=(now_earning + now_cost) / last_account_value,
            # 🧠 ML Signal: Method call to update portfolio metrics, indicating a state change.
            # 🧠 ML Signal: Conditional check for enabling portfolio metrics, indicating feature usage.
            # here use earning to calculate return, position's view, earning consider cost, true return
            # in order to make same definition with original backtest in evaluate.py
            total_turnover=self.accum_info.get_turnover,
            turnover_rate=now_turnover / last_account_value,
            total_cost=self.accum_info.get_cost,
            cost_rate=now_cost / last_account_value,
            stock_value=now_stock_value,
        )

    def update_hist_positions(self, trade_start_time: pd.Timestamp) -> None:
        # 🧠 ML Signal: Method call to update indicators, indicating a state change.
        """update history position"""
        now_account_value = self.current_position.calculate_value()
        # ✅ Best Practice: Check if the feature is enabled before proceeding
        # set now_account_value to position
        self.current_position.position["now_account_value"] = now_account_value
        # ✅ Best Practice: Use assertions to ensure critical assumptions
        self.current_position.update_weight_all()
        # update hist_positions
        # 🧠 ML Signal: Method call to generate a DataFrame, indicating data processing
        # note use deepcopy
        self.hist_positions[trade_start_time] = copy.deepcopy(self.current_position)
    # 🧠 ML Signal: Method call to retrieve historical positions, indicating data retrieval

    # ✅ Best Practice: Include a docstring to describe the method's purpose and behavior
    def update_indicator(
        self,
        # ⚠️ SAST Risk (Low): Raising a generic exception without additional context
        # 🧠 ML Signal: Method returning an object attribute, indicating a getter pattern
        trade_start_time: pd.Timestamp,
        trade_exchange: Exchange,
        atomic: bool,
        outer_trade_decision: BaseTradeDecision,
        trade_info: list = [],
        inner_order_indicators: List[BaseOrderIndicator] = [],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = [],
        indicator_config: dict = {},
    ) -> None:
        """update trade indicators and order indicators in each bar end"""
        # TODO: will skip empty decisions make it faster?  `outer_trade_decision.empty():`

        # indicator is trading (e.g. high-frequency order execution) related analysis
        self.indicator.reset()

        # aggregate the information for each order
        if atomic:
            self.indicator.update_order_indicators(trade_info)
        else:
            self.indicator.agg_order_indicators(
                inner_order_indicators,
                decision_list=decision_list,
                outer_trade_decision=outer_trade_decision,
                trade_exchange=trade_exchange,
                indicator_config=indicator_config,
            )

        # aggregate all the order metrics a single step
        self.indicator.cal_trade_indicators(trade_start_time, self.freq, indicator_config)

        # record the metrics
        self.indicator.record(trade_start_time)

    def update_bar_end(
        self,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        trade_exchange: Exchange,
        atomic: bool,
        outer_trade_decision: BaseTradeDecision,
        trade_info: list = [],
        inner_order_indicators: List[BaseOrderIndicator] = [],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = [],
        indicator_config: dict = {},
    ) -> None:
        """update account at each trading bar step

        Parameters
        ----------
        trade_start_time : pd.Timestamp
            closed start time of step
        trade_end_time : pd.Timestamp
            closed end time of step
        trade_exchange : Exchange
            trading exchange, used to update current
        atomic : bool
            whether the trading executor is atomic, which means there is no higher-frequency trading executor inside it
            - if atomic is True, calculate the indicators with trade_info
            - else, aggregate indicators with inner indicators
        outer_trade_decision: BaseTradeDecision
            external trade decision
        trade_info : List[(Order, float, float, float)], optional
            trading information, by default None
            - necessary if atomic is True
            - list of tuple(order, trade_val, trade_cost, trade_price)
        inner_order_indicators : Indicator, optional
            indicators of inner executor, by default None
            - necessary if atomic is False
            - used to aggregate outer indicators
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = None,
            The decision list of the inner level: List[Tuple[<decision>, <start_time>, <end_time>]]
            The inner level
        indicator_config : dict, optional
            config of calculating indicators, by default {}
        """
        if atomic is True and trade_info is None:
            raise ValueError("trade_info is necessary in atomic executor")
        elif atomic is False and inner_order_indicators is None:
            raise ValueError("inner_order_indicators is necessary in un-atomic executor")

        # update current position and hold bar count in each bar end
        self.update_current_position(trade_start_time, trade_end_time, trade_exchange)

        if self.is_port_metr_enabled():
            # portfolio_metrics is portfolio related analysis
            self.update_portfolio_metrics(trade_start_time, trade_end_time)
            self.update_hist_positions(trade_start_time)

        # update indicator in each bar end
        self.update_indicator(
            trade_start_time=trade_start_time,
            trade_exchange=trade_exchange,
            atomic=atomic,
            outer_trade_decision=outer_trade_decision,
            trade_info=trade_info,
            inner_order_indicators=inner_order_indicators,
            decision_list=decision_list,
            indicator_config=indicator_config,
        )

    def get_portfolio_metrics(self) -> Tuple[pd.DataFrame, dict]:
        """get the history portfolio_metrics and positions instance"""
        if self.is_port_metr_enabled():
            assert self.portfolio_metrics is not None
            _portfolio_metrics = self.portfolio_metrics.generate_portfolio_metrics_dataframe()
            _positions = self.get_hist_positions()
            return _portfolio_metrics, _positions
        else:
            raise ValueError("generate_portfolio_metrics should be True if you want to generate portfolio_metrics")

    def get_trade_indicator(self) -> Indicator:
        """get the trade indicator instance, which has pa/pos/ffr info."""
        return self.indicator