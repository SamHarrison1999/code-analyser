# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.constant import EPS
from qlib.rl.data.base import ProcessedDataProvider
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution.state import SAOEState
from qlib.typehint import TypedDict

# ✅ Best Practice: Grouping imports logically (standard, third-party, application-specific) improves readability.
__all__ = [
    "FullHistoryStateInterpreter",
    "CurrentStepStateInterpreter",
    # ✅ Best Practice: Check for specific types using isinstance for better readability and maintainability.
    "CategoricalActionInterpreter",
    "TwapRelativeActionInterpreter",
    # 🧠 ML Signal: Conversion of DataFrame to numpy array indicates data preprocessing for ML.
    "FullHistoryObs",
]
# ✅ Best Practice: Use of isinstance to handle both native and numpy float types.

from qlib.utils import init_instance_by_config

# 🧠 ML Signal: Conversion to float32 is common in ML for reducing memory usage.

# ✅ Best Practice: Use of isinstance to handle both native and numpy integer types.


def canonicalize(
    value: int | float | np.ndarray | pd.DataFrame | dict,
) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    # 🧠 ML Signal: Conversion to int32 is common in ML for reducing memory usage.
    # ✅ Best Practice: Use of TypedDict for type hinting improves code readability and maintainability
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    # ✅ Best Practice: Clearly defined attributes improve code readability
    if isinstance(value, (float, np.floating)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == "f"
    ):
        # 🧠 ML Signal: Recursive processing of dictionary values indicates complex data structure handling.
        return np.array(value, dtype=np.float32)
    # ✅ Best Practice: Clearly defined attributes improve code readability
    elif isinstance(value, (int, bool, np.integer)) or (
        isinstance(value, np.ndarray) and value.dtype.kind == "i"
    ):
        return np.array(value, dtype=np.int32)
    # ✅ Best Practice: Return the value as is if it doesn't match any expected type.
    # ✅ Best Practice: Clearly defined attributes improve code readability
    elif isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    # ✅ Best Practice: Clearly defined attributes improve code readability
    else:
        return value


# ✅ Best Practice: Clearly defined attributes improve code readability

# ✅ Best Practice: Include a docstring to describe the purpose and usage of the class


# ✅ Best Practice: Clearly defined attributes improve code readability
class FullHistoryObs(TypedDict):
    # ✅ Best Practice: Clearly defined attributes improve code readability
    # 🧠 ML Signal: Method returns a dictionary with a fixed key-value pair
    data_processed: Any
    data_processed_prev: Any
    # ✅ Best Practice: Use type hinting for function return type for better readability and maintainability
    acquiring: Any
    # ✅ Best Practice: Clearly defined attributes improve code readability
    cur_tick: Any
    # 🧠 ML Signal: Use of spaces.Dict and spaces.Box indicates reinforcement learning environment setup
    # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
    cur_step: Any
    # ✅ Best Practice: Clearly defined attributes improve code readability
    # ⚠️ SAST Risk (Low): Using np.inf can lead to unexpected behavior if not handled properly
    num_step: Any
    target: Any
    position: Any
    position_history: Any


class DummyStateInterpreter(StateInterpreter[SAOEState, dict]):
    """Dummy interpreter for policies that do not need inputs (for example, AllOne)."""

    def interpret(self, state: SAOEState) -> dict:
        # TODO: A fake state, used to pass `check_nan_observation`. Find a better way in the future.
        return {"DUMMY": _to_int32(1)}

    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {"DUMMY": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.int32)}
        )


class FullHistoryStateInterpreter(StateInterpreter[SAOEState, FullHistoryObs]):
    """The observation of all the history, including today (until this moment), and yesterday.

    Parameters
    ----------
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    data_ticks
        Equal to the total number of records. For example, in SAOE per minute,
        the total ticks is the length of day in minutes.
    data_dim
        Number of dimensions in data.
    processed_data_provider
        Provider of the processed data.
    """

    def __init__(
        self,
        # 🧠 ML Signal: Tracking position history for state analysis
        max_step: int,
        data_ticks: int,
        # ✅ Best Practice: Use of type casting for clarity and correctness
        # ✅ Best Practice: Canonicalize data for consistent structure
        # 🧠 ML Signal: Masking future information for current state
        data_dim: int,
        processed_data_provider: dict | ProcessedDataProvider,
    ) -> None:
        super().__init__()

        self.max_step = max_step
        self.data_ticks = data_ticks
        self.data_dim = data_dim
        self.processed_data_provider: ProcessedDataProvider = init_instance_by_config(
            processed_data_provider,
            accept_types=ProcessedDataProvider,
            # 🧠 ML Signal: Encoding order direction as integer
        )

    # 🧠 ML Signal: Calculating current tick index
    def interpret(self, state: SAOEState) -> FullHistoryObs:
        processed = self.processed_data_provider.get_data(
            # 🧠 ML Signal: Calculating current step index
            stock_id=state.order.stock_id,
            date=pd.Timestamp(state.order.start_time.date()),
            # 🧠 ML Signal: Encoding number of steps
            feature_dim=self.data_dim,
            # 🧠 ML Signal: Encoding target amount
            # 🧠 ML Signal: Encoding current position
            # 🧠 ML Signal: Encoding position history
            # 🧠 ML Signal: Defines the structure of the observation space, useful for reinforcement learning models
            # 🧠 ML Signal: Continuous space for processed data, indicating use of Box space for RL
            # 🧠 ML Signal: Continuous space for previous processed data, indicating time-series or sequential data handling
            time_index=state.ticks_index,
        )

        position_history = np.full(self.max_step + 1, 0.0, dtype=np.float32)
        position_history[0] = state.order.amount
        position_history[1 : len(state.history_steps) + 1] = state.history_steps[
            "position"
        ].to_numpy()

        # The min, slice here are to make sure that indices fit into the range,
        # even after the final step of the simulator (in the done step),
        # to make network in policy happy.
        return cast(
            # 🧠 ML Signal: Discrete space for binary state, indicating categorical data handling
            # 🧠 ML Signal: Continuous space for current tick, indicating use of Box space for RL
            FullHistoryObs,
            canonicalize(
                {
                    # ✅ Best Practice: Use of copy(deep=True) to avoid modifying the original DataFrame
                    # 🧠 ML Signal: Continuous space for target, indicating use of Box space for RL
                    "data_processed": np.array(
                        self._mask_future_info(processed.today, state.cur_time)
                    ),
                    "data_processed_prev": np.array(processed.yesterday),
                    # ⚠️ SAST Risk (Low): Overwriting data with 0.0 could lead to data loss if not intended
                    # 🧠 ML Signal: Continuous space for position, indicating use of Box space for RL
                    "acquiring": _to_int32(state.order.direction == state.order.BUY),
                    # ✅ Best Practice: Use of TypedDict for type-safe dictionaries
                    "cur_tick": _to_int32(
                        min(
                            int(np.sum(state.ticks_index < state.cur_time)),
                            self.data_ticks - 1,
                        )
                    ),
                    # 🧠 ML Signal: Continuous space for position history, indicating use of Box space for RL
                    "cur_step": _to_int32(min(state.cur_step, self.max_step - 1)),
                    # ✅ Best Practice: Clear and descriptive attribute naming
                    "num_step": _to_int32(self.max_step),
                    "target": _to_float32(state.order.amount),
                    # ✅ Best Practice: Returns a dictionary space, ensuring structured and organized observation space
                    # ✅ Best Practice: Clear and descriptive attribute naming
                    "position": _to_float32(state.position),
                    "position_history": _to_float32(position_history[: self.max_step]),
                    # ✅ Best Practice: Class docstring provides a clear explanation of the class purpose and usage
                    # ✅ Best Practice: Clear and descriptive attribute naming
                },
                # ✅ Best Practice: Clear and descriptive attribute naming
            ),
        )

    @property
    # ✅ Best Practice: Clear and descriptive attribute naming
    def observation_space(self) -> spaces.Dict:
        # ✅ Best Practice: Type hinting for function parameters and return values improves code readability and maintainability.
        space = {
            "data_processed": spaces.Box(
                -np.inf, np.inf, shape=(self.data_ticks, self.data_dim)
            ),
            "data_processed_prev": spaces.Box(
                -np.inf, np.inf, shape=(self.data_ticks, self.data_dim)
            ),
            # 🧠 ML Signal: Storing a parameter as an instance variable, indicating object state management.
            # 🧠 ML Signal: Method defining observation space for reinforcement learning environment
            "acquiring": spaces.Discrete(2),
            # ✅ Best Practice: Using @property decorator for getter methods enhances encapsulation and provides a cleaner interface.
            # ✅ Best Practice: Use of descriptive dictionary keys for clarity
            # 🧠 ML Signal: Discrete space indicating binary state (e.g., acquiring or not)
            "cur_tick": spaces.Box(0, self.data_ticks - 1, shape=(), dtype=np.int32),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            # TODO: support arbitrary length index
            "num_step": spaces.Box(
                self.max_step, self.max_step, shape=(), dtype=np.int32
            ),
            "target": spaces.Box(-EPS, np.inf, shape=()),
            "position": spaces.Box(-EPS, np.inf, shape=()),
            "position_history": spaces.Box(-EPS, np.inf, shape=(self.max_step,)),
            # 🧠 ML Signal: Continuous space for current step within a range
            # 🧠 ML Signal: Continuous space for a fixed number of steps
        }
        return spaces.Dict(space)

    # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in production, leading to potential issues.
    # 🧠 ML Signal: Continuous space for target value with lower bound

    # 🧠 ML Signal: Use of a class method to interpret and transform state data into an observation.
    # 🧠 ML Signal: Continuous space for position value with lower bound
    # ✅ Best Practice: Explicitly naming parameters in object instantiation improves readability.
    @staticmethod
    def _mask_future_info(arr: pd.DataFrame, current: pd.Timestamp) -> pd.DataFrame:
        arr = arr.copy(deep=True)
        arr.loc[current:] = 0.0  # mask out data after this moment (inclusive)
        return arr


# ✅ Best Practice: Returning a dictionary of spaces for structured observation
# 🧠 ML Signal: Checking order direction to determine acquiring status.
# 🧠 ML Signal: Mapping current step and maximum step to observation.
class CurrentStateObs(TypedDict):
    # ✅ Best Practice: Class docstring provides a clear explanation of the class purpose and parameters.
    acquiring: bool
    # 🧠 ML Signal: Mapping order amount to target in observation.
    # 🧠 ML Signal: Mapping current position to observation.
    # 🧠 ML Signal: Returning a structured observation from the interpreted state.
    cur_step: int
    num_step: int
    target: float
    position: float


class CurrentStepStateInterpreter(StateInterpreter[SAOEState, CurrentStateObs]):
    """The observation of current step.

    Used when policy only depends on the latest state, but not history.
    The key list is not full. You can add more if more information is needed by your policy.
    """

    # ✅ Best Practice: Call to super() in __init__ ensures proper initialization of the base class

    def __init__(self, max_step: int) -> None:
        # 🧠 ML Signal: Type checking and conversion based on input type
        super().__init__()

        # 🧠 ML Signal: List comprehension for generating sequences
        self.max_step = max_step

    # 🧠 ML Signal: Storing input values in instance variables
    # 🧠 ML Signal: Method defining action space, useful for RL model training
    @property
    def observation_space(self) -> spaces.Dict:
        # 🧠 ML Signal: Storing optional parameters in instance variables
        # ✅ Best Practice: Return statement directly uses spaces.Discrete for clarity
        space = {
            # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations
            "acquiring": spaces.Discrete(2),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            # ✅ Best Practice: Check if max_step is not None before comparing with cur_step
            "num_step": spaces.Box(
                self.max_step, self.max_step, shape=(), dtype=np.int32
            ),
            "target": spaces.Box(-EPS, np.inf, shape=()),
            "position": spaces.Box(-EPS, np.inf, shape=()),
        }
        # 🧠 ML Signal: Use of min function to limit the position based on action values
        return spaces.Dict(space)

    def interpret(self, state: SAOEState) -> CurrentStateObs:
        assert state.cur_step <= self.max_step
        obs = CurrentStateObs(
            acquiring=state.order.direction == state.order.BUY,
            cur_step=state.cur_step,
            # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
            # ✅ Best Practice: Use of @property decorator for getter method
            num_step=self.max_step,
            target=state.order.amount,
            # 🧠 ML Signal: Returning a Box space with infinite bounds is common in reinforcement learning environments.
            # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
            position=state.position,
        )
        # ✅ Best Practice: Use descriptive variable names for better readability.
        return obs


# ✅ Best Practice: Consider adding input validation to ensure 'val' is a valid integer
# 🧠 ML Signal: Calculation of TWAP (Time Weighted Average Price) volume indicates a financial trading strategy.


# ⚠️ SAST Risk (Low): Potential ValueError if 'val' cannot be converted to an integer
class CategoricalActionInterpreter(ActionInterpreter[SAOEState, int, float]):
    """Convert a discrete policy action to a continuous action, then multiplied by ``order.amount``.

    Parameters
    ----------
    values
        It can be a list of length $L$: $[a_1, a_2, \\ldots, a_L]$.
        Then when policy givens decision $x$, $a_x$ times order amount is the output.
        It can also be an integer $n$, in which case the list of length $n+1$ is auto-generated,
        i.e., $[0, 1/n, 2/n, \\ldots, n/n]$.
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    """

    def __init__(
        self, values: int | List[float], max_step: Optional[int] = None
    ) -> None:
        super().__init__()

        if isinstance(values, int):
            values = [i / values for i in range(0, values + 1)]
        self.action_values = values
        self.max_step = max_step

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_values))

    def interpret(self, state: SAOEState, action: int) -> float:
        assert 0 <= action < len(self.action_values)
        if self.max_step is not None and state.cur_step >= self.max_step - 1:
            return state.position
        else:
            return min(state.position, state.order.amount * self.action_values[action])


class TwapRelativeActionInterpreter(ActionInterpreter[SAOEState, float, float]):
    """Convert a continuous ratio to deal amount.

    The ratio is relative to TWAP on the remainder of the day.
    For example, there are 5 steps left, and the left position is 300.
    With TWAP strategy, in each position, 60 should be traded.
    When this interpreter receives action $a$, its output is $60 \\cdot a$.
    """

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(), dtype=np.float32)

    def interpret(self, state: SAOEState, action: float) -> float:
        estimated_total_steps = math.ceil(
            len(state.ticks_for_order) / state.ticks_per_step
        )
        twap_volume = state.position / (estimated_total_steps - state.cur_step)
        return min(state.position, twap_volume * action)


def _to_int32(val):
    return np.array(int(val), dtype=np.int32)


def _to_float32(val):
    return np.array(val, dtype=np.float32)
