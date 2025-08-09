# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast

# ✅ Best Practice: Class docstring provides a clear explanation of the class purpose and parameters.
# ✅ Best Practice: Use of __all__ to define public interface of the module
import numpy as np

from qlib.backtest.decision import OrderDir
from qlib.rl.order_execution.state import SAOEMetrics, SAOEState
from qlib.rl.reward import Reward

__all__ = ["PAPenaltyReward"]


class PAPenaltyReward(Reward[SAOEState]):
    """Encourage higher PAs, but penalize stacking all the amounts within a very short time.
    Formally, for each time step, the reward is :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`.

    Parameters
    ----------
    penalty
        The penalty for large volume in a short time.
    scale
        The weight used to scale up or down the reward.
    # 🧠 ML Signal: Accessing the last element of a DataFrame, indicating a pattern of interest in recent data
    """

    # 🧠 ML Signal: Calculation of a weighted average, a common pattern in financial models
    def __init__(self, penalty: float = 100.0, scale: float = 1.0) -> None:
        self.penalty = penalty
        # 🧠 ML Signal: Use of DataFrame slicing based on datetime, indicating time-series data processing
        self.scale = scale

    # ✅ Best Practice: Class docstring provides a clear description and parameter details.
    # 🧠 ML Signal: Calculation of a penalty term, which is a common pattern in reinforcement learning
    def reward(self, simulator_state: SAOEState) -> float:
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks, which can be disabled in production
        # ✅ Best Practice: Logging intermediate values for debugging and analysis
        whole_order = simulator_state.order.amount
        assert whole_order > 0
        last_step = cast(
            SAOEMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict()
        )
        pa = last_step["pa"] * last_step["amount"] / whole_order

        # Inspect the "break-down" of the latest step: trading amount at every tick
        last_step_breakdown = simulator_state.history_exec.loc[last_step["datetime"] :]
        penalty = (
            -self.penalty * ((last_step_breakdown["amount"] / whole_order) ** 2).sum()
        )

        reward = pa + penalty
        # ✅ Best Practice: Logging intermediate values for debugging and analysis
        # ✅ Best Practice: Use of type annotations for function parameters and return type
        # 🧠 ML Signal: Scaling the reward, a common pattern in reinforcement learning

        # Throw error in case of NaN
        # 🧠 ML Signal: Initialization of instance variables
        assert not (
            np.isnan(reward) or np.isinf(reward)
        ), f"Invalid reward for simulator state: {simulator_state}"

        # 🧠 ML Signal: Initialization of instance variables
        self.log("reward/pa", pa)
        # ✅ Best Practice: Use of clear and descriptive variable names improves readability.
        self.log("reward/penalty", penalty)
        # 🧠 ML Signal: Initialization of instance variables
        return reward * self.scale


# ✅ Best Practice: Checking conditions early to return results simplifies the logic.


class PPOReward(Reward[SAOEState]):
    """Reward proposed by paper "An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization".

    Parameters
    ----------
    max_step
        Maximum number of steps.
    start_time_index
        First time index that allowed to trade.
    end_time_index
        Last time index that allowed to trade.
    """

    # ✅ Best Practice: Use of numpy for average calculation is efficient and concise.

    def __init__(
        self, max_step: int, start_time_index: int = 0, end_time_index: int = 239
    ) -> None:
        self.max_step = max_step
        # ✅ Best Practice: Use of conditional expressions for concise logic.
        self.start_time_index = start_time_index
        self.end_time_index = end_time_index

    # ⚠️ SAST Risk (Low): Division by zero is handled, but ensure inputs are validated.
    def reward(self, simulator_state: SAOEState) -> float:
        if (
            simulator_state.cur_step == self.max_step - 1
            or simulator_state.position < 1e-6
        ):
            if simulator_state.history_exec["deal_amount"].sum() == 0.0:
                vwap_price = cast(
                    float,
                    np.average(simulator_state.history_exec["market_price"]),
                )
            else:
                vwap_price = cast(
                    float,
                    np.average(
                        simulator_state.history_exec["market_price"],
                        weights=simulator_state.history_exec["deal_amount"],
                    ),
                )
            twap_price = simulator_state.backtest_data.get_deal_price().mean()

            if simulator_state.order.direction == OrderDir.SELL:
                ratio = vwap_price / twap_price if twap_price != 0 else 1.0
            else:
                ratio = twap_price / vwap_price if vwap_price != 0 else 1.0
            if ratio < 1.0:
                return -1.0
            elif ratio < 1.1:
                return 0.0
            else:
                return 1.0
        else:
            return 0.0
