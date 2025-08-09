# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from typing import Any, cast, List, Optional

import numpy as np
import pandas as pd

from pathlib import Path
from qlib.backtest.decision import Order, OrderDir
from qlib.constant import EPS, EPS_T, float_or_ndarray
# ✅ Best Practice: Relative imports can make the code more modular and easier to refactor.
from qlib.rl.data.base import BaseIntradayBacktestData
from qlib.rl.data.native import DataframeIntradayBacktestData, load_handler_intraday_processed_data
# ✅ Best Practice: Using __all__ to define public API of the module.
from qlib.rl.data.pickle_styled import load_simple_intraday_backtest_data
from qlib.rl.simulator import Simulator
from qlib.rl.utils import LogLevel
from .state import SAOEMetrics, SAOEState

__all__ = ["SingleAssetOrderExecutionSimple"]


class SingleAssetOrderExecutionSimple(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator.

    As there's no "calendar" in the simple simulator, ticks are used to trade.
    A tick is a record (a line) in the pickle-styled data file.
    Each tick is considered as a individual trading opportunity.
    If such fine granularity is not needed, use ``ticks_per_step`` to
    lengthen the ticks for each step.

    In each step, the traded amount are "equally" separated to each tick,
    then bounded by volume maximum execution volume (i.e., ``vol_threshold``),
    and if it's the last step, try to ensure all the amount to be executed.

    Parameters
    ----------
    order
        The seed to start an SAOE simulator is an order.
    data_dir
        Path to load backtest data.
    feature_columns_today
        Columns of today's feature.
    feature_columns_yesterday
        Columns of yesterday's feature.
    data_granularity
        Number of ticks between consecutive data entries.
    ticks_per_step
        How many ticks per step.
    vol_threshold
        Maximum execution volume (divided by market execution volume).
    """

    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    history_exec: pd.DataFrame
    """All execution history at every possible time ticks. See :class:`SAOEMetrics` for available columns.
    Index is ``datetime``.
    """
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

    history_steps: pd.DataFrame
    """Positions at each step. The position before first step is also recorded.
    See :class:`SAOEMetrics` for available columns.
    Index is ``datetime``, which is the **starting** time of each step."""

    metrics: Optional[SAOEMetrics]
    """Metrics. Only available when done."""

    twap_price: float
    # ✅ Best Practice: Call to superclass constructor ensures proper initialization of the base class.
    """This price is used to compute price advantage.
    It"s defined as the average price in the period from order"s start time to end time."""
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags.

    ticks_index: pd.DatetimeIndex
    """All available ticks for the day (not restricted to order)."""

    # 🧠 ML Signal: Use of feature columns indicates feature engineering for ML models.
    ticks_for_order: pd.DatetimeIndex
    """Ticks that is available for trading (sliced by order)."""
    # 🧠 ML Signal: Use of feature columns indicates feature engineering for ML models.

    def __init__(
        self,
        order: Order,
        # 🧠 ML Signal: Loading backtest data suggests simulation or model evaluation.
        data_dir: Path,
        feature_columns_today: List[str] = [],
        feature_columns_yesterday: List[str] = [],
        data_granularity: int = 1,
        ticks_per_step: int = 30,
        vol_threshold: Optional[float] = None,
    # 🧠 ML Signal: Calculation of average price could be used as a feature or target in ML models.
    ) -> None:
        super().__init__(initial=order)

        # 🧠 ML Signal: Use of metric keys suggests tracking or evaluation of model performance.
        assert ticks_per_step % data_granularity == 0

        # ✅ Best Practice: Use of pandas DataFrame for structured data storage and manipulation.
        self.order = order
        # ✅ Best Practice: Use of pandas DataFrame for structured data storage and manipulation.
        self.data_dir = data_dir
        self.feature_columns_today = feature_columns_today
        self.feature_columns_yesterday = feature_columns_yesterday
        self.ticks_per_step: int = ticks_per_step // data_granularity
        self.vol_threshold = vol_threshold

        self.backtest_data = self.get_backtest_data()
        self.ticks_index = self.backtest_data.get_time_index()

        # 🧠 ML Signal: Usage of a data loading function with specific parameters
        # Get time index available for trading
        self.ticks_for_order = self._get_ticks_slice(self.order.start_time, self.order.end_time)
        # ⚠️ SAST Risk (Low): Broad exception handling may hide other issues
        # ✅ Best Practice: Consider logging the exception for better traceability

        self.cur_time = self.ticks_for_order[0]
        self.cur_step = 0
        # NOTE: astype(float) is necessary in some systems.
        # this will align the precision with `.to_numpy()` in `_split_exec_vol`
        self.twap_price = float(self.backtest_data.get_deal_price().loc[self.ticks_for_order].astype(float).mean())

        self.position = order.amount

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        # NOTE: can empty dataframe contain index?
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics = None
        # ⚠️ SAST Risk (Low): Use of assert statements for runtime checks can be disabled with optimization flags.

        self.market_price: Optional[np.ndarray] = None
        self.market_vol: Optional[np.ndarray] = None
        # 🧠 ML Signal: Splitting execution volume could indicate a strategy pattern.
        self.market_vol_limit: Optional[np.ndarray] = None

    # ⚠️ SAST Risk (Low): Use of assert statements for runtime checks can be disabled with optimization flags.
    def get_backtest_data(self) -> BaseIntradayBacktestData:
        try:
            # ⚠️ SAST Risk (Low): Use of assert statements for runtime checks can be disabled with optimization flags.
            data = load_handler_intraday_processed_data(
                data_dir=self.data_dir,
                # 🧠 ML Signal: Tracking position changes could indicate trading behavior.
                stock_id=self.order.stock_id,
                date=pd.Timestamp(self.order.start_time.date()),
                feature_columns_today=self.feature_columns_today,
                # ✅ Best Practice: Use a constant for small value comparisons to improve readability.
                # ⚠️ SAST Risk (Medium): Potential for negative execution volumes, which may be unintended.
                # 🧠 ML Signal: Appending to history could indicate a pattern of tracking past actions.
                feature_columns_yesterday=self.feature_columns_yesterday,
                backtest=True,
                index_only=False,
            )
            return DataframeIntradayBacktestData(data.today)
        except (AttributeError, FileNotFoundError):
            # TODO: For compatibility with older versions of test scripts (tests/rl/test_saoe_simple.py)
            # TODO: In the future, we should modify the data format used by the test script,
            # TODO: and then delete this branch.
            return load_simple_intraday_backtest_data(
                self.data_dir / "backtest",
                self.order.stock_id,
                pd.Timestamp(self.order.start_time.date()),
                "close",
                self.order.direction,
            )

    def step(self, amount: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        amount
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        # 🧠 ML Signal: Collecting metrics could indicate a pattern of performance evaluation.
        assert not self.done()

        self.market_price = self.market_vol = None  # avoid misuse
        exec_vol = self._split_exec_vol(amount)
        assert self.market_price is not None
        assert self.market_vol is not None

        # 🧠 ML Signal: Checking for completion could indicate a pattern of iterative processes.
        ticks_position = self.position - np.cumsum(exec_vol)
        # 🧠 ML Signal: Logging history could indicate a pattern of tracking and analysis.

        self.position -= exec_vol.sum()
        if abs(self.position) < 1e-6:
            self.position = 0.0
        if self.position < -EPS or (exec_vol < -EPS).any():
            raise ValueError(f"Execution volume is invalid: {exec_vol} (position = {self.position})")

        # 🧠 ML Signal: Method returning an object with multiple attributes, indicating a complex state representation
        # Get time index available for this step
        # 🧠 ML Signal: Logging metrics could indicate a pattern of performance tracking.
        # 🧠 ML Signal: Usage of self attributes to construct a state object
        # 🧠 ML Signal: Returning an instance of a class, which may be used for state management or serialization
        time_index = self._get_ticks_slice(self.cur_time, self._next_time())

        self.history_exec = self._dataframe_append(
            self.history_exec,
            SAOEMetrics(
                # It should have the same keys with SAOEMetrics,
                # but the values do not necessarily have the annotated type.
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                # Some values could be vectorized (e.g., exec_vol).
                stock_id=self.order.stock_id,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                datetime=time_index,
                direction=self.order.direction,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                market_volume=self.market_vol,
                market_price=self.market_price,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability
                amount=exec_vol,
                inner_amount=exec_vol,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                # 🧠 ML Signal: Usage of comparison operators to determine a boolean condition
                deal_amount=exec_vol,
                # ⚠️ SAST Risk (Low): Potential risk if EPS is not defined or is user-controlled
                trade_price=self.market_price,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                # 🧠 ML Signal: Usage of pandas index location to determine the next time step
                trade_value=self.market_price * exec_vol,
                position=ticks_position,
                # 🧠 ML Signal: Usage of self attributes to construct a state object
                # 🧠 ML Signal: Incrementing index location by a fixed step size
                ffr=exec_vol / self.order.amount,
                pa=price_advantage(self.market_price, self.twap_price, self.order.direction),
            # 🧠 ML Signal: Usage of self attributes to construct a state object
            # ✅ Best Practice: Aligning next_loc to the nearest step boundary
            ),
        )
        # 🧠 ML Signal: Usage of self attributes to construct a state object
        # ⚠️ SAST Risk (Low): Potential index out of bounds if next_loc is not validated

        # ✅ Best Practice: Include type hints for better code readability and maintainability
        self.history_steps = self._dataframe_append(
            self.history_steps,
            [self._metrics_collect(self.cur_time, self.market_vol, self.market_price, amount, exec_vol)],
        # 🧠 ML Signal: Use of subtraction operation on datetime objects
        )

        if self.done():
            if self.env is not None:
                self.env.logger.add_any("history_steps", self.history_steps, loglevel=LogLevel.DEBUG)
                # 🧠 ML Signal: Usage of time-based volume splitting strategy (TWAP) for trading
                self.env.logger.add_any("history_exec", self.history_exec, loglevel=LogLevel.DEBUG)

            # 🧠 ML Signal: Accessing historical market volume data for decision making
            self.metrics = self._metrics_collect(
                self.ticks_index[0],  # start time
                # 🧠 ML Signal: Accessing historical market price data for decision making
                self.history_exec["market_volume"],
                self.history_exec["market_price"],
                # ⚠️ SAST Risk (Low): Assert statements can be disabled in production, leading to potential issues
                self.history_steps["amount"].sum(),
                self.history_exec["deal_amount"],
            # 🧠 ML Signal: Repeating execution volume based on market price length
            )

            # 🧠 ML Signal: Applying volume threshold constraints to execution volume
            # NOTE (yuge): It looks to me that it's the "correct" decision to
            # 🧠 ML Signal: Adjusting execution volume based on market volume limits
            # ✅ Best Practice: Handling edge case when next_time exceeds order end time
            # put all the logs here, because only components like simulators themselves
            # have the knowledge about what could appear in the logs, and what's the format.
            # But I admit it's not necessarily the most convenient way.
            # I'll rethink about it when we have the second environment
            # Maybe some APIs like self.logger.enable_auto_log() ?
            # 🧠 ML Signal: Adjusting final execution volume to match position

            if self.env is not None:
                for key, value in self.metrics.items():
                    # 🧠 ML Signal: Reapplying volume constraints after adjustment
                    # ✅ Best Practice: Use of assert to ensure input arrays have the same length
                    if isinstance(value, float):
                        # 🧠 ML Signal: Returning the calculated execution volumes
                        self.env.logger.add_scalar(key, value)
                    # ✅ Best Practice: Use of EPS to handle floating-point precision issues
                    else:
                        self.env.logger.add_any(key, value)

        self.cur_time = self._next_time()
        # ✅ Best Practice: Use of numpy average with weights for calculating weighted average
        self.cur_step += 1
    # ✅ Best Practice: Check if exec_avg_price has 'item' method to convert numpy scalar to Python float
    # 🧠 ML Signal: Use of self.order attributes indicates a pattern of accessing order-related data

    def get_state(self) -> SAOEState:
        return SAOEState(
            order=self.order,
            cur_time=self.cur_time,
            cur_step=self.cur_step,
            position=self.position,
            history_exec=self.history_exec,
            history_steps=self.history_steps,
            metrics=self.metrics,
            backtest_data=self.backtest_data,
            ticks_per_step=self.ticks_per_step,
            ticks_index=self.ticks_index,
            ticks_for_order=self.ticks_for_order,
        )
    # 🧠 ML Signal: Use of self.order attributes indicates a pattern of accessing order-related data
    # ✅ Best Practice: Use of numpy sum for efficient array summation

    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    def done(self) -> bool:
        # ✅ Best Practice: Use of numpy sum for efficient array summation
        return self.position < EPS or self.cur_time >= self.order.end_time
    # ⚠️ SAST Risk (Low): Potential issue if EPS_T is not defined or not a valid timedelta, which could lead to runtime errors.

    def _next_time(self) -> pd.Timestamp:
        # ✅ Best Practice: Use of numpy sum and element-wise multiplication for efficient computation
        # ✅ Best Practice: Type hint for 'df' parameter improves code readability and maintainability
        """The "current time" (``cur_time``) for next step."""
        # ✅ Best Practice: Type hint for 'other' parameter improves code readability and maintainability
        # 🧠 ML Signal: Usage of pandas' slice_indexer method indicates a pattern of working with time series data.
        # Look for next time on time index
        # 🧠 ML Signal: Use of self.position indicates a pattern of accessing position-related data
        current_loc = self.ticks_index.get_loc(self.cur_time)
        # 🧠 ML Signal: Conversion of 'other' to DataFrame indicates dynamic data handling
        # ✅ Best Practice: Use of @staticmethod decorator indicates that the method does not modify class state, improving code clarity.
        next_loc = current_loc + self.ticks_per_step
        # ⚠️ SAST Risk (Low): Potential division by zero if self.order.amount is zero
        # ✅ Best Practice: Explicitly setting index name improves code clarity
        # 🧠 ML Signal: Use of price_advantage function indicates a pattern of calculating price advantage

        # Calibrate the next location to multiple of ticks_per_step.
        # This is to make sure that:
        # 🧠 ML Signal: Use of pd.concat suggests data aggregation pattern
        # as long as ticks_per_step is a multiple of something, each step won't cross morning and afternoon.
        next_loc = next_loc - next_loc % self.ticks_per_step

        # ⚠️ SAST Risk (Medium): Division by zero check, but returning zero might not be appropriate for all contexts.
        if next_loc < len(self.ticks_index) and self.ticks_index[next_loc] < self.order.end_time:
            return self.ticks_index[next_loc]
        else:
            return self.order.end_time

    def _cur_duration(self) -> pd.Timedelta:
        # 🧠 ML Signal: Pattern of calculating percentage advantage for BUY orders.
        """The "duration" of this step (step that is about to happen)."""
        return self._next_time() - self.cur_time

    # 🧠 ML Signal: Pattern of calculating percentage advantage for SELL orders.
    def _split_exec_vol(self, exec_vol_sum: float) -> np.ndarray:
        """
        Split the volume in each step into minutes, considering possible constraints.
        This follows TWAP strategy.
        """
        # ✅ Best Practice: Use of np.nan_to_num to handle NaN values in calculations.
        # ✅ Best Practice: Use of item() to convert single-element arrays to scalar.
        # ✅ Best Practice: Use of cast for type hinting and clarity.
        next_time = self._next_time()

        # get the backtest data for next interval
        self.market_vol = self.backtest_data.get_volume().loc[self.cur_time : next_time - EPS_T].to_numpy()
        self.market_price = self.backtest_data.get_deal_price().loc[self.cur_time : next_time - EPS_T].to_numpy()

        assert self.market_vol is not None and self.market_price is not None

        # split the volume equally into each minute
        exec_vol = np.repeat(exec_vol_sum / len(self.market_price), len(self.market_price))

        # apply the volume threshold
        market_vol_limit = self.vol_threshold * self.market_vol if self.vol_threshold is not None else np.inf
        exec_vol = np.minimum(exec_vol, market_vol_limit)  # type: ignore

        # Complete all the order amount at the last moment.
        if next_time >= self.order.end_time:
            exec_vol[-1] += self.position - exec_vol.sum()
            exec_vol = np.minimum(exec_vol, market_vol_limit)  # type: ignore

        return exec_vol

    def _metrics_collect(
        self,
        datetime: pd.Timestamp,
        market_vol: np.ndarray,
        market_price: np.ndarray,
        amount: float,  # intended to trade such amount
        exec_vol: np.ndarray,
    ) -> SAOEMetrics:
        assert len(market_vol) == len(market_price) == len(exec_vol)

        if np.abs(np.sum(exec_vol)) < EPS:
            exec_avg_price = 0.0
        else:
            exec_avg_price = cast(float, np.average(market_price, weights=exec_vol))  # could be nan
            if hasattr(exec_avg_price, "item"):  # could be numpy scalar
                exec_avg_price = exec_avg_price.item()  # type: ignore

        return SAOEMetrics(
            stock_id=self.order.stock_id,
            datetime=datetime,
            direction=self.order.direction,
            market_volume=market_vol.sum(),
            market_price=market_price.mean(),
            amount=amount,
            inner_amount=exec_vol.sum(),
            deal_amount=exec_vol.sum(),  # in this simulator, there's no other restrictions
            trade_price=exec_avg_price,
            trade_value=float(np.sum(market_price * exec_vol)),
            position=self.position,
            ffr=float(exec_vol.sum() / self.order.amount),
            pa=price_advantage(exec_avg_price, self.twap_price, self.order.direction),
        )

    def _get_ticks_slice(self, start: pd.Timestamp, end: pd.Timestamp, include_end: bool = False) -> pd.DatetimeIndex:
        if not include_end:
            end = end - EPS_T
        return self.ticks_index[self.ticks_index.slice_indexer(start, end)]

    @staticmethod
    def _dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
        # dataframe.append is deprecated
        other_df = pd.DataFrame(other).set_index("datetime")
        other_df.index.name = "datetime"
        return pd.concat([df, other_df], axis=0)


def price_advantage(
    exec_price: float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        if isinstance(exec_price, float):
            return 0.0
        else:
            return np.zeros_like(exec_price)
    if direction == OrderDir.BUY:
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(float_or_ndarray, res_wo_nan)