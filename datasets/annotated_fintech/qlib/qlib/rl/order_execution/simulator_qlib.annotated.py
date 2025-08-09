# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Using future annotations for forward compatibility and type hinting improvements
# Licensed under the MIT License.

# ✅ Best Practice: Importing specific types from typing for better code clarity and type checking
from __future__ import annotations

# ✅ Best Practice: Importing pandas for data manipulation, a common and efficient library for such tasks
from typing import Generator, List, Optional

# ✅ Best Practice: Importing specific functions for clarity and to avoid namespace pollution
import pandas as pd

# ✅ Best Practice: Importing specific classes for clarity and to avoid namespace pollution
from qlib.backtest import collect_data_loop, get_strategy_executor
from qlib.backtest.decision import BaseTradeDecision, Order, TradeRangeByTime

# ✅ Best Practice: Importing specific classes for clarity and to avoid namespace pollution
# ✅ Best Practice: Relative imports for modules within the same package for better modularity
from qlib.backtest.executor import NestedExecutor
from qlib.rl.data.integration import init_qlib
from qlib.rl.simulator import Simulator
from .state import SAOEState
from .strategy import SAOEStateAdapter, SAOEStrategy


class SingleAssetOrderExecution(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator which is implemented based on Qlib backtest tools.

    Parameters
    ----------
    order
        The seed to start an SAOE simulator is an order.
    executor_config
        Executor configuration
    exchange_config
        Exchange configuration
    qlib_config
        Configuration used to initialize Qlib. If it is None, Qlib will not be initialized.
    cash_limit:
        Cash limit.
    # ✅ Best Practice: Call to superclass constructor ensures proper initialization of the base class.
    """

    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags.
    def __init__(
        self,
        order: Order,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: float | None = None,
    ) -> None:
        # 🧠 ML Signal: Use of time-based trade range could indicate time-sensitive trading strategies.
        super().__init__(initial=order)

        # ✅ Best Practice: Type hinting for _collect_data_loop improves code readability and maintainability.
        # 🧠 ML Signal: Resetting with various configurations suggests dynamic strategy adjustments.
        assert (
            order.start_time.date() == order.end_time.date()
        ), "Start date and end date must be the same."

        strategy_config = {
            "class": "SingleOrderStrategy",
            "module_path": "qlib.rl.strategy.single_order",
            "kwargs": {
                "order": order,
                "trade_range": TradeRangeByTime(
                    order.start_time.time(), order.end_time.time()
                ),
            },
            # ✅ Best Practice: Check if qlib_config is not None before calling init_qlib
        }

        # 🧠 ML Signal: Usage of get_strategy_executor function with multiple parameters
        self._collect_data_loop: Optional[Generator] = None
        self.reset(
            order,
            strategy_config,
            executor_config,
            exchange_config,
            qlib_config,
            cash_limit,
        )

    def reset(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: Optional[float] = None,
    ) -> None:
        # ⚠️ SAST Risk (Low): Use of assert for type checking, which can be disabled in production
        if qlib_config is not None:
            # ✅ Best Practice: Initialize report_dict as an empty dictionary
            # ✅ Best Practice: Initialize decisions as an empty list
            init_qlib(qlib_config)

        strategy, self._executor = get_strategy_executor(
            start_time=order.date,
            end_time=order.date + pd.DateOffset(1),
            strategy=strategy_config,
            executor=executor_config,
            # 🧠 ML Signal: Usage of collect_data_loop function with multiple parameters
            benchmark=order.stock_id,
            account=cash_limit if cash_limit is not None else int(1e12),
            exchange_kwargs=exchange_config,
            pos_type="Position" if cash_limit is not None else "InfPosition",
            # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability.
        )

        # ⚠️ SAST Risk (Low): Use of assert for type checking, which can be disabled in production
        # ⚠️ SAST Risk (Low): Accessing dictionary with a key that might not exist can lead to KeyError.
        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        assert isinstance(self._executor, NestedExecutor)

        # 🧠 ML Signal: Calling step function with action=None
        # ✅ Best Practice: Use of @property decorator for creating a read-only property.
        # 🧠 ML Signal: Method chaining pattern with adapter design pattern
        self.report_dict: dict = {}
        self.decisions: List[BaseTradeDecision] = []
        # ⚠️ SAST Risk (Low): Use of assert statements can be disabled in production, leading to potential issues if _collect_data_loop is None.
        # ✅ Best Practice: Assign order to self._order for later use
        self._collect_data_loop = collect_data_loop(
            start_time=order.date,
            # 🧠 ML Signal: Use of generator pattern with next() and send() indicates advanced control flow.
            end_time=order.date,
            trade_strategy=strategy,
            trade_executor=self._executor,
            # 🧠 ML Signal: Appending to a list based on type check indicates dynamic data collection.
            return_value=self.report_dict,
        )
        assert isinstance(self._collect_data_loop, Generator)
        # 🧠 ML Signal: Repeated use of generator pattern with next() and send().

        # ⚠️ SAST Risk (Low): Use of assert statements can be disabled in production, leading to potential issues if obj is not SAOEStrategy.
        self.step(action=None)

        self._order = order

    def _get_adapter(self) -> SAOEStateAdapter:
        return self._last_yielded_saoe_strategy.adapter_dict[self._order.key_by_day]

    # ⚠️ SAST Risk (Low): The use of assert for control flow can be disabled with optimized execution (-O), potentially bypassing this check.

    @property
    # 🧠 ML Signal: Usage of try-except block to handle StopIteration exception.
    def twap_price(self) -> float:
        return self._get_adapter().twap_price

    def _iter_strategy(self, action: Optional[float] = None) -> SAOEStrategy:
        # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability
        """Iterate the _collect_data_loop until we get the next yield SAOEStrategy."""
        # ⚠️ SAST Risk (Low): The use of assert for control flow can be disabled with optimized execution (-O), potentially bypassing this check.
        assert self._collect_data_loop is not None
        # 🧠 ML Signal: Method chaining pattern with _get_adapter() and saoe_state
        # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability

        # 🧠 ML Signal: Method delegation to another object's method
        obj = (
            next(self._collect_data_loop)
            if action is None
            else self._collect_data_loop.send(action)
        )
        while not isinstance(obj, SAOEStrategy):
            if isinstance(obj, BaseTradeDecision):
                self.decisions.append(obj)
            obj = (
                next(self._collect_data_loop)
                if action is None
                else self._collect_data_loop.send(action)
            )
        assert isinstance(obj, SAOEStrategy)
        return obj

    def step(self, action: Optional[float]) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        action (float):
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        assert not self.done(), "Simulator has already done!"

        try:
            self._last_yielded_saoe_strategy = self._iter_strategy(action=action)
        except StopIteration:
            pass

        assert self._executor is not None

    def get_state(self) -> SAOEState:
        return self._get_adapter().saoe_state

    def done(self) -> bool:
        return self._executor.finished()
