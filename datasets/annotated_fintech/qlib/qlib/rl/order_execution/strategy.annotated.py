# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
from types import GeneratorType
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns

import warnings
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
import numpy as np
import pandas as pd
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
import torch
from tianshou.data import Batch
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from tianshou.policy import BasePolicy

# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.backtest import CommonInfrastructure, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWithDetails, TradeDecisionWO, TradeRange
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.backtest.exchange import Exchange
from qlib.backtest.executor import BaseExecutor
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.backtest.utils import LevelInfrastructure, get_start_end_idx
from qlib.constant import EPS, ONE_MIN, REG_CN
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.rl.data.native import IntradayBacktestData, load_backtest_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
from qlib.rl.order_execution.state import SAOEMetrics, SAOEState
from qlib.rl.order_execution.utils import dataframe_append, price_advantage
from qlib.strategy.base import RLStrategy
from qlib.utils import init_instance_by_config
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
from qlib.utils.index_data import IndexData
from qlib.utils.time import get_day_min_idx_range
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns


# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
# 🧠 ML Signal: Use of a while loop to iterate over a range of timestamps.
def _get_all_timestamps(
    start: pd.Timestamp,
    # 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
    end: pd.Timestamp,
    granularity: pd.Timedelta = ONE_MIN,
    # 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
    # ⚠️ SAST Risk (Low): Potential IndexError if 'ret' is empty; ensure 'ret' is not empty before accessing ret[-1].
    include_end: bool = True,
) -> pd.DatetimeIndex:
    # 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
    ret = []
    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Potential IndexError if 'ret' is empty; ensure 'ret' is not empty before accessing ret[-1].
    while start <= end:
        ret.append(start)
        start += granularity

    # ✅ Best Practice: Returning a pd.DatetimeIndex from a list of timestamps for better performance and functionality.
    if ret[-1] > end:
        ret.pop()
    if ret[-1] == end and not include_end:
        ret.pop()
    return pd.DatetimeIndex(ret)


def fill_missing_data(
    original_data: np.ndarray,
    fill_method: Callable = np.nanmedian,
) -> np.ndarray:
    """Fill missing data.

    Parameters
    ----------
    original_data
        Original data without missing values.
    fill_method
        Method used to fill the missing data.

    Returns
    -------
        The filled data.
    # 🧠 ML Signal: Class docstring provides detailed usage pattern and example
    """
    return np.nan_to_num(original_data, nan=fill_method(original_data))


class SAOEStateAdapter:
    """
    Maintain states of the environment. SAOEStateAdapter accepts execution results and update its internal state
    according to the execution results with additional information acquired from executors & exchange. For example,
    it gets the dealt order amount from execution results, and get the corresponding market price / volume from
    exchange.

    Example usage::

        adapter = SAOEStateAdapter(...)
        adapter.update(...)
        state = adapter.saoe_state
    # 🧠 ML Signal: Storing initial state and parameters for an object
    """

    # 🧠 ML Signal: Storing initial state and parameters for an object
    def __init__(
        self,
        # 🧠 ML Signal: Calculating start index based on trade calendar and decision
        order: Order,
        trade_decision: BaseTradeDecision,
        # 🧠 ML Signal: Calculating mean deal price from backtest data
        executor: BaseExecutor,
        exchange: Exchange,
        # 🧠 ML Signal: Initializing DataFrame with specific columns for metrics
        ticks_per_step: int,
        backtest_data: IntradayBacktestData,
        # 🧠 ML Signal: Initializing DataFrame for execution history
        # 🧠 ML Signal: Usage of pandas to handle time series data
        data_granularity: int = 1,
    ) -> None:
        # 🧠 ML Signal: Initializing DataFrame for step history
        # ✅ Best Practice: Use of integer division for clarity and to avoid float results
        self.position = order.amount
        # ✅ Best Practice: Ensures next_loc aligns with data granularity
        # 🧠 ML Signal: Initializing optional metrics attribute
        self.order = order
        self.executor = executor
        self.exchange = exchange
        self.backtest_data = backtest_data
        # 🧠 ML Signal: Determining current time based on order and backtest data
        # 🧠 ML Signal: Storing ticks per step parameter
        # ⚠️ SAST Risk (Low): Potential index out of range if next_loc is not validated
        self.start_idx, _ = get_start_end_idx(self.executor.trade_calendar, trade_decision)

        self.twap_price = self.backtest_data.get_deal_price().mean()
        # 🧠 ML Signal: Storing data granularity parameter
        # 🧠 ML Signal: Conditional logic to determine next time step
        # ⚠️ SAST Risk (Low): Use of assert for input validation, which can be disabled in production

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics: Optional[SAOEMetrics] = None
        # 🧠 ML Signal: Handling edge case by returning end_time

        self.cur_time = max(backtest_data.ticks_for_order[0], order.start_time)
        self.ticks_per_step = ticks_per_step
        self.data_granularity = data_granularity
        # 🧠 ML Signal: Iterating over execute_result to process orders
        assert self.ticks_per_step % self.data_granularity == 0

    def _next_time(self) -> pd.Timestamp:
        current_loc = self.backtest_data.ticks_index.get_loc(self.cur_time)
        # ⚠️ SAST Risk (Low): Potential floating-point precision issues with exec_vol.sum()
        next_loc = current_loc + (self.ticks_per_step // self.data_granularity)
        # ⚠️ SAST Risk (Low): Use of warnings.warn without specifying a category
        next_loc = next_loc - next_loc % (self.ticks_per_step // self.data_granularity)
        if (
            next_loc < len(self.backtest_data.ticks_index)
            and self.backtest_data.ticks_index[next_loc] < self.order.end_time
        ):
            return self.backtest_data.ticks_index[next_loc]
        else:
            return self.order.end_time

    def update(
        self,
        execute_result: list,
        last_step_range: Tuple[int, int],
    ) -> None:
        last_step_size = last_step_range[1] - last_step_range[0] + 1
        start_time = self.backtest_data.ticks_index[last_step_range[0]]
        end_time = self.backtest_data.ticks_index[last_step_range[1]]

        exec_vol = np.zeros(last_step_size)
        for order, _, __, ___ in execute_result:
            idx, _ = get_day_min_idx_range(order.start_time, order.end_time, f"{self.data_granularity}min", REG_CN)
            exec_vol[idx - last_step_range[0]] = order.deal_amount

        if exec_vol.sum() > self.position and exec_vol.sum() > 0.0:
            if exec_vol.sum() > self.position + 1.0:
                warnings.warn(
                    f"Sum of execution volume is {exec_vol.sum()} which is larger than "
                    f"position + 1.0 = {self.position} + 1.0 = {self.position + 1.0}. "
                    # ✅ Best Practice: Ensure data is in the correct format and shape
                    f"All execution volume is scaled down linearly to ensure that their sum does not position."
                )
            # ⚠️ SAST Risk (Low): Use of assert for runtime checks
            # 🧠 ML Signal: Generating trade indicators dataframe
            exec_vol *= self.position / (exec_vol.sum())

        market_volume = cast(
            IndexData,
            self.exchange.get_volume(
                self.order.stock_id,
                pd.Timestamp(start_time),
                pd.Timestamp(end_time),
                method=None,
            ),
        )
        market_price = cast(
            IndexData,
            self.exchange.get_deal_price(
                self.order.stock_id,
                pd.Timestamp(start_time),
                pd.Timestamp(end_time),
                method=None,
                direction=self.order.direction,
            ),
        )
        market_price = fill_missing_data(np.array(market_price, dtype=float).reshape(-1))
        market_volume = fill_missing_data(np.array(market_volume, dtype=float).reshape(-1))

        assert market_price.shape == market_volume.shape == exec_vol.shape

        # Get data from the current level executor's indicator
        current_trade_account = self.executor.trade_account
        current_df = current_trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
        self.history_exec = dataframe_append(
            # 🧠 ML Signal: Updating position based on executed volume
            # 🧠 ML Signal: Updating current time to the next time step
            # 🧠 ML Signal: Collecting metrics based on historical execution data
            # 🧠 ML Signal: Using the first tick index for metric calculation
            # 🧠 ML Signal: Method name suggests a post-execution metric generation pattern
            self.history_exec,
            self._collect_multi_order_metric(
                order=self.order,
                datetime=_get_all_timestamps(
                    start_time, end_time, include_end=True, granularity=ONE_MIN * self.data_granularity
                ),
                # 🧠 ML Signal: Market volume is a key feature for metric calculation
                market_vol=market_volume,
                market_price=market_price,
                # 🧠 ML Signal: Market price is a key feature for metric calculation
                # 🧠 ML Signal: Summing historical step amounts for metric calculation
                exec_vol=exec_vol,
                pa=current_df.iloc[-1]["pa"],
            ),
        )

        self.history_steps = dataframe_append(
            self.history_steps,
            [
                self._collect_single_order_metric(
                    # ✅ Best Practice: Returning a well-defined data structure (SAOEMetrics) improves code readability and maintainability.
                    # 🧠 ML Signal: Using order attributes like stock_id and direction can indicate trading behavior patterns.
                    self.order,
                    self.cur_time,
                    market_volume,
                    market_price,
                    exec_vol.sum(),
                    exec_vol,
                ),
            ],
        )

        # Do this at the end
        self.position -= exec_vol.sum()

        self.cur_time = self._next_time()

    # ⚠️ SAST Risk (Low): Multiplying arrays without validation could lead to unexpected results if dimensions mismatch.
    # ⚠️ SAST Risk (Low): Using np.cumsum without validation could lead to incorrect calculations if exec_vol is not properly formatted.
    # ⚠️ SAST Risk (Low): Division without checking for zero in order.amount could lead to division by zero errors.
    def generate_metrics_after_done(self) -> None:
        """Generate metrics once the upper level execution is done"""

        self.metrics = self._collect_single_order_metric(
            self.order,
            self.backtest_data.ticks_index[0],  # start time
            self.history_exec["market_volume"],
            self.history_exec["market_price"],
            self.history_steps["amount"].sum(),
            # ✅ Best Practice: Use of assert to ensure input arrays have the same length
            self.history_exec["deal_amount"],
        )
    # ⚠️ SAST Risk (Low): Potential division by zero if exec_vol is empty or sums to zero

    def _collect_multi_order_metric(
        self,
        order: Order,
        # ✅ Best Practice: Use of numpy average with weights for calculating average price
        datetime: pd.DatetimeIndex,
        market_vol: np.ndarray,
        # ✅ Best Practice: Use of numpy sum for efficient summation
        # ✅ Best Practice: Check if exec_avg_price has an item method and use it
        # 🧠 ML Signal: Collecting metrics for a single order, useful for model training
        market_price: np.ndarray,
        exec_vol: np.ndarray,
        pa: float,
    ) -> SAOEMetrics:
        return SAOEMetrics(
            # It should have the same keys with SAOEMetrics,
            # but the values do not necessarily have the annotated type.
            # Some values could be vectorized (e.g., exec_vol).
            stock_id=order.stock_id,
            datetime=datetime,
            direction=order.direction,
            market_volume=market_vol,
            market_price=market_price,
            amount=exec_vol,
            inner_amount=exec_vol,
            deal_amount=exec_vol,
            # ✅ Best Practice: Use of numpy sum for efficient summation
            # ✅ Best Practice: Use of numpy mean for calculating average price
            # 🧠 ML Signal: Method returning a state object, useful for state representation learning
            trade_price=market_price,
            # ✅ Best Practice: Use of numpy sum for efficient calculation of trade value
            # ⚠️ SAST Risk (Low): Potential division by zero if order.amount is zero
            # 🧠 ML Signal: Tracking position changes, useful for model training
            # 🧠 ML Signal: Instantiation of a state object, indicating a pattern of state management
            # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
            trade_value=market_price * exec_vol,
            position=self.position - np.cumsum(exec_vol),
            ffr=exec_vol / order.amount,
            pa=pa,
        )

    def _collect_single_order_metric(
        # 🧠 ML Signal: Calculating price advantage, useful for model training
        # 🧠 ML Signal: Calculation of current step, useful for temporal pattern analysis
        self,
        order: Order,
        # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
        datetime: pd.Timestamp,
        market_vol: np.ndarray,
        # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
        market_price: np.ndarray,
        amount: float,  # intended to trade such amount
        # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
        exec_vol: np.ndarray,
    # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
    ) -> SAOEMetrics:
        # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
        assert len(market_vol) == len(market_price) == len(exec_vol)

        if np.abs(np.sum(exec_vol)) < EPS:
            exec_avg_price = 0.0
        else:
            exec_avg_price = cast(float, np.average(market_price, weights=exec_vol))  # could be nan
            # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
            if hasattr(exec_avg_price, "item"):  # could be numpy scalar
                exec_avg_price = exec_avg_price.item()  # type: ignore

        # ✅ Best Practice: Use of super() to initialize the parent class ensures proper inheritance.
        # 🧠 ML Signal: Usage of class attributes to construct state, indicating feature selection
        exec_sum = exec_vol.sum()
        return SAOEMetrics(
            stock_id=order.stock_id,
            datetime=datetime,
            direction=order.direction,
            market_volume=market_vol.sum(),
            market_price=market_price.mean() if len(market_price) > 0 else np.nan,
            amount=amount,
            # 🧠 ML Signal: Storing data granularity could indicate a pattern of data processing frequency.
            inner_amount=exec_sum,
            deal_amount=exec_sum,  # in this simulator, there's no other restrictions
            # 🧠 ML Signal: Use of a dictionary to store adapters suggests a pattern of dynamic state management.
            # 🧠 ML Signal: Tracking the last step range could be used to model sequential decision-making.
            trade_price=exec_avg_price,
            trade_value=float(np.sum(market_price * exec_vol)),
            position=self.position - exec_sum,
            ffr=float(exec_sum / order.amount),
            pa=price_advantage(exec_avg_price, self.twap_price, order.direction),
        )
    # 🧠 ML Signal: Function involves creating an adapter for backtesting, indicating a pattern of preparing data for simulation or analysis.

    # ✅ Best Practice: Using a dedicated adapter class (SAOEStateAdapter) for backtesting promotes modularity and separation of concerns.
    @property
    def saoe_state(self) -> SAOEState:
        return SAOEState(
            order=self.order,
            cur_time=self.cur_time,
            cur_step=self.executor.trade_calendar.get_trade_step() - self.start_idx,
            position=self.position,
            history_exec=self.history_exec,
            history_steps=self.history_steps,
            # ✅ Best Practice: Using pd.Timedelta for time calculations ensures clarity and correctness in time-related operations.
            metrics=self.metrics,
            # ✅ Best Practice: Explicitly calling the superclass method ensures proper initialization.
            backtest_data=self.backtest_data,
            ticks_per_step=self.ticks_per_step,
            # ✅ Best Practice: Initializing adapter_dict to an empty dictionary for clarity and to avoid potential KeyErrors.
            ticks_index=self.backtest_data.ticks_index,
            ticks_for_order=self.backtest_data.ticks_for_order,
        # ✅ Best Practice: Initializing _last_step_range to a default value for clarity and consistency.
        )


# ⚠️ SAST Risk (Low): Potential risk if trade_range is None, though assert mitigates this.
class SAOEStrategy(RLStrategy):
    """RL-based strategies that use SAOEState as state."""
    # ⚠️ SAST Risk (Low): Using assert for runtime checks can be bypassed if Python is run with optimizations.

    def __init__(
        # ✅ Best Practice: Re-initializing adapter_dict to ensure no stale data is present.
        self,
        # 🧠 ML Signal: Accessing a dictionary using a key derived from an object's attribute
        policy: BasePolicy,
        # 🧠 ML Signal: Iterating over decisions could indicate a pattern of processing multiple trade decisions.
        # ✅ Best Practice: Use of type hints for return type improves code readability and maintainability
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        # 🧠 ML Signal: Casting decision to Order type suggests a pattern of handling specific object types.
        # 🧠 ML Signal: Iterating over a dictionary's values to perform operations on each item
        # ⚠️ SAST Risk (Low): Potential KeyError if order.key_by_day is not in adapter_dict
        common_infra: CommonInfrastructure | None = None,
        data_granularity: int = 1,
        # 🧠 ML Signal: Storing adapters in a dictionary keyed by day indicates a pattern of organizing data by time.
        # 🧠 ML Signal: Calling a method on each item in a collection
        **kwargs: Any,
    ) -> None:
        super(SAOEStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            # ✅ Best Practice: Use of defaultdict to handle missing keys gracefully
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )
        # 🧠 ML Signal: Iterating over a list of results to categorize them

        self._data_granularity = data_granularity
        # 🧠 ML Signal: Updating adapters with categorized results
        self.adapter_dict: Dict[tuple, SAOEStateAdapter] = {}
        self._last_step_range = (0, 0)

    def _create_qlib_backtest_adapter(
        self,
        order: Order,
        trade_decision: BaseTradeDecision,
        trade_range: TradeRange,
    ) -> SAOEStateAdapter:
        backtest_data = load_backtest_data(order, self.trade_exchange, trade_range)

        # ✅ Best Practice: Updating internal state before generating a decision ensures consistency.
        return SAOEStateAdapter(
            order=order,
            # 🧠 ML Signal: Pattern of delegating decision logic to a private method for subclass customization.
            trade_decision=trade_decision,
            executor=self.executor,
            # ✅ Best Practice: Handling generator types allows for flexible decision generation.
            exchange=self.trade_exchange,
            # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
            # ⚠️ SAST Risk (Low): Using 'yield from' can lead to unexpected behavior if not properly managed.
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / ONE_MIN),
            backtest_data=backtest_data,
            data_granularity=self._data_granularity,
        )

    # ⚠️ SAST Risk (Low): The use of NotImplementedError is generally safe, but ensure that this method is properly implemented in subclasses to avoid runtime errors.
    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        super(SAOEStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)

        self.adapter_dict = {}
        self._last_step_range = (0, 0)

        # ✅ Best Practice: The docstring provides a clear explanation of the class's purpose and behavior.
        # 🧠 ML Signal: The class is designed to delegate decision-making, which is a pattern that could be used to train models on decision delegation.
        if outer_trade_decision is not None and not outer_trade_decision.empty():
            trade_range = outer_trade_decision.trade_range
            assert trade_range is not None

            self.adapter_dict = {}
            for decision in outer_trade_decision.get_decision():
                order = cast(Order, decision)
                # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
                self.adapter_dict[order.key_by_day] = self._create_qlib_backtest_adapter(
                    # ✅ Best Practice: Default values for parameters allow for more flexible function calls.
                    order, outer_trade_decision, trade_range
                # 🧠 ML Signal: Use of generator function to yield and return values
                )
    # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of the base class.

    # 🧠 ML Signal: Interaction with trade_exchange to get order helper
    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.adapter_dict[order.key_by_day].saoe_state
    # 🧠 ML Signal: Creation of an order using order helper

    # ✅ Best Practice: Use of type hinting for function parameters and return type
    def post_upper_level_exe_step(self) -> None:
        # 🧠 ML Signal: Returning a TradeDecisionWO object
        for adapter in self.adapter_dict.values():
            # ⚠️ SAST Risk (Low): Use of assert for type checking can be bypassed; consider using explicit type checks or exceptions
            adapter.generate_metrics_after_done()

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        # 🧠 ML Signal: Checking for None before accessing attributes is a common pattern
        last_step_length = self._last_step_range[1] - self._last_step_range[0]
        if last_step_length <= 0:
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
            # ⚠️ SAST Risk (Low): Use of assert for length check can be bypassed; consider using explicit checks or exceptions
            assert not execute_result
            # 🧠 ML Signal: Accessing the first element of a list is a common pattern
            return

        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[e[0].key_by_day].append(e)

        for key, adapter in self.adapter_dict.items():
            adapter.update(results[key], self._last_step_range)

    def generate_trade_decision(
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        self,
        execute_result: list | None = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        """
        For SAOEStrategy, we need to update the `self._last_step_range` every time a decision is generated.
        This operation should be invisible to developers, so we implement it in `generate_trade_decision()`
        The concrete logic to generate decisions should be implemented in `_generate_trade_decision()`.
        In other words, all subclass of `SAOEStrategy` should overwrite `_generate_trade_decision()` instead of
        `generate_trade_decision()`.
        """
        self._last_step_range = self.get_data_cal_avail_range(rtype="step")
        # 🧠 ML Signal: Usage of an action interpreter suggests a pattern for action management in ML models.

        decision = self._generate_trade_decision(execute_result)
        if isinstance(decision, GeneratorType):
            decision = yield from decision

        return decision

    # ⚠️ SAST Risk (Medium): Assertion without exception handling can lead to crashes if the condition is not met.
    def _generate_trade_decision(
        self,
        execute_result: list | None = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        raise NotImplementedError
# 🧠 ML Signal: Updating network configuration with observation space indicates dynamic model configuration.


class ProxySAOEStrategy(SAOEStrategy):
    """Proxy strategy that uses SAOEState. It is called a 'proxy' strategy because it does not make any decisions
    by itself. Instead, when the strategy is required to generate a decision, it will yield the environment's
    information and let the outside agents to make the decision. Please refer to `_generate_trade_decision` for
    more details.
    """

    def __init__(
        # 🧠 ML Signal: Updating policy configuration with observation and action spaces indicates dynamic model configuration.
        self,
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        common_infra: CommonInfrastructure | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, outer_trade_decision, level_infra, common_infra, **kwargs)
    # ✅ Best Practice: Type hinting for parameters and return value improves code readability and maintainability
    # 🧠 ML Signal: Dynamic instance creation from configuration is a common pattern in ML frameworks.

    def _generate_trade_decision(self, execute_result: list | None = None) -> Generator[Any, Any, BaseTradeDecision]:
        # ✅ Best Practice: Using super() to call the parent class method is a good practice for code maintainability
        # Once the following line is executed, this ProxySAOEStrategy (self) will be yielded to the outside
        # ⚠️ SAST Risk (Low): No validation on 'act' and 'exec_vols' input types or lengths
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific handling can lead to ungraceful error management.
        # ⚠️ SAST Risk (Low): No exception handling for attribute access
        # level strategy through this way.
        # 🧠 ML Signal: Calling eval() on a policy indicates a pattern for setting models to evaluation mode.
        # ⚠️ SAST Risk (Low): Potential risk if 'order_list' has more elements than 'act' or 'exec_vols'
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        return TradeDecisionWO([order], self)

    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        # ⚠️ SAST Risk (Low): No validation on 'o.stock_id' existence or type
        # ⚠️ SAST Risk (Low): No exception handling for 'get_step_time' method
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        # ⚠️ SAST Risk (Low): No exception handling for 'get_freq' method
        assert isinstance(outer_trade_decision, TradeDecisionWO)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]
# ✅ Best Practice: Check for None before assigning to avoid KeyError


class SAOEIntStrategy(SAOEStrategy):
    # ⚠️ SAST Risk (Low): No validation on the structure of 'trade_details' before DataFrame creation
    """(SAOE)state based strategy with (Int)preters."""

    # ⚠️ SAST Risk (Low): Use of torch.no_grad() suppresses gradient tracking, ensure it's intended
    def __init__(
        self,
        # ⚠️ SAST Risk (Low): Ensure act is properly validated before use
        policy: dict | BasePolicy,
        state_interpreter: dict | StateInterpreter,
        # 🧠 ML Signal: Use of interpreters for actions and states
        action_interpreter: dict | ActionInterpreter,
        network: dict | torch.nn.Module | None = None,
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        common_infra: CommonInfrastructure | None = None,
        # ✅ Best Practice: Returning a well-structured object with details
        **kwargs: Any,
    ) -> None:
        super(SAOEIntStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

        self._state_interpreter: StateInterpreter = init_instance_by_config(
            state_interpreter,
            accept_types=StateInterpreter,
        )
        self._action_interpreter: ActionInterpreter = init_instance_by_config(
            action_interpreter,
            accept_types=ActionInterpreter,
        )

        if isinstance(policy, dict):
            assert network is not None

            if isinstance(network, dict):
                network["kwargs"].update(
                    {
                        "obs_space": self._state_interpreter.observation_space,
                    }
                )
                network_inst = init_instance_by_config(network)
            else:
                network_inst = network

            policy["kwargs"].update(
                {
                    "obs_space": self._state_interpreter.observation_space,
                    "action_space": self._action_interpreter.action_space,
                    "network": network_inst,
                }
            )
            self._policy = init_instance_by_config(policy)
        elif isinstance(policy, BasePolicy):
            self._policy = policy
        else:
            raise ValueError(f"Unsupported policy type: {type(policy)}.")

        if self._policy is not None:
            self._policy.eval()

    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

    def _generate_trade_details(self, act: np.ndarray, exec_vols: List[float]) -> pd.DataFrame:
        assert hasattr(self.outer_trade_decision, "order_list")

        trade_details = []
        for a, v, o in zip(act, exec_vols, getattr(self.outer_trade_decision, "order_list")):
            trade_details.append(
                {
                    "instrument": o.stock_id,
                    "datetime": self.trade_calendar.get_step_time()[0],
                    "freq": self.trade_calendar.get_freq(),
                    "rl_exec_vol": v,
                }
            )
            if a is not None:
                trade_details[-1]["rl_action"] = a
        return pd.DataFrame.from_records(trade_details)

    def _generate_trade_decision(self, execute_result: list | None = None) -> BaseTradeDecision:
        states = []
        obs_batch = []
        for decision in self.outer_trade_decision.get_decision():
            order = cast(Order, decision)
            state = self.get_saoe_state_by_order(order)

            states.append(state)
            obs_batch.append({"obs": self._state_interpreter.interpret(state)})

        with torch.no_grad():
            policy_out = self._policy(Batch(obs_batch))
        act = policy_out.act.numpy() if torch.is_tensor(policy_out.act) else policy_out.act
        exec_vols = [self._action_interpreter.interpret(s, a) for s, a in zip(states, act)]

        oh = self.trade_exchange.get_order_helper()
        order_list = []
        for decision, exec_vol in zip(self.outer_trade_decision.get_decision(), exec_vols):
            if exec_vol != 0:
                order = cast(Order, decision)
                order_list.append(oh.create(order.stock_id, exec_vol, order.direction))

        return TradeDecisionWithDetails(
            order_list=order_list,
            strategy=self,
            details=self._generate_trade_details(act, exec_vols),
        )