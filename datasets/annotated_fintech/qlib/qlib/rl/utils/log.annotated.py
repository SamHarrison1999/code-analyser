# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Distributed logger for RL.

:class:`LogCollector` runs in every environment workers. It collects log info from simulator states,
and add them (as a dict) to auxiliary info returned for each step.

:class:`LogWriter` runs in the central worker. It decodes the dict collected by :class:`LogCollector`
in each worker, and writes them to console, log files, or tensorboard...

The two modules communicate by the "log" field in "info" returned by ``env.step()``.
"""
# ✅ Best Practice: Use of Path from pathlib for file system paths is more robust and readable.

# NOTE: This file contains many hardcoded / ad-hoc rules.
# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports during type checking.
# Refactoring it will be one of the future tasks.

# 🧠 ML Signal: Use of numpy indicates numerical operations, common in ML applications.
from __future__ import annotations

# 🧠 ML Signal: Use of pandas indicates data manipulation, common in ML applications.
import logging
from collections import defaultdict
# ✅ Best Practice: Use of a specific logger for the module improves logging clarity and management.
# ✅ Best Practice: Use of IntEnum for log levels provides clear, readable, and maintainable code.
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Sequence, Set, Tuple, TypeVar

# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports during type checking.
import numpy as np
# ✅ Best Practice: __all__ is defined to control what is exported when import * is used.
import pandas as pd

# ✅ Best Practice: Use of TypeVar for generic programming increases code flexibility and reusability.
from qlib.log import get_module_logger

# ✅ Best Practice: Use of TypeVar for generic programming increases code flexibility and reusability.
if TYPE_CHECKING:
    from .env_wrapper import InfoDict


__all__ = ["LogCollector", "LogWriter", "LogLevel", "LogBuffer", "ConsoleWriter", "CsvWriter"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
class LogLevel(IntEnum):
    """Log-levels for RL training.
    The behavior of handling each log level depends on the implementation of :class:`LogWriter`.
    """
    # 🧠 ML Signal: Conversion of input to a specific type (int) for internal consistency
    # ✅ Best Practice: Method docstring provides clarity on the method's purpose

    DEBUG = 10
    """If you only want to see the metric in debug mode."""
    # 🧠 ML Signal: Usage of instance variable to store state
    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    PERIODIC = 20
    # ✅ Best Practice: Resetting internal state to an empty dictionary
    """If you want to see the metric periodically."""
    # ⚠️ SAST Risk (Low): Potential for key collision in self._logged if name is not unique.
    # FIXME: I haven't given much thought about this. Let's hold it for one iteration.

    # ✅ Best Practice: Type hinting improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific error handling can make debugging difficult.
    INFO = 30
    """Important log messages."""
    # 🧠 ML Signal: Tracking metrics with log levels can be used to train models on logging behavior and patterns.
    CRITICAL = 40
    # 🧠 ML Signal: Checking log level before proceeding is a common pattern in logging.
    """LogWriter should always handle CRITICAL messages"""


# ⚠️ SAST Risk (Low): Type checking at runtime can be bypassed; consider using static type checkers.
class LogCollector:
    """Logs are first collected in each environment worker,
    and then aggregated to stream at the central thread in vector env.

    In :class:`LogCollector`, every metric is added to a dict, which needs to be ``reset()`` at each step.
    The dict is sent via the ``info`` in ``env.step()``, and decoded by the :class:`LogWriter` at vector env.

    ``min_loglevel`` is for optimization purposes: to avoid too much traffic on networks / in pipe.
    # ✅ Best Practice: Using hasattr to check for 'item' method is a flexible way to handle different types
    """

    _logged: Dict[str, Tuple[int, Any]]
    # ⚠️ SAST Risk (Low): Type checking and conversion could raise exceptions if not handled properly
    _min_loglevel: int

    # ✅ Best Practice: Explicit conversion to float ensures consistent data type
    # 🧠 ML Signal: Logging or metric collection pattern
    def __init__(self, min_loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self._min_loglevel = int(min_loglevel)

    def reset(self) -> None:
        """Clear all collected contents."""
        self._logged = {}

    # ✅ Best Practice: Early return to avoid unnecessary processing if loglevel is too low
    def _add_metric(self, name: str, metric: Any, loglevel: int | LogLevel) -> None:
        if name in self._logged:
            raise ValueError(f"A metric with {name} is already added. Please change a name or reset the log collector.")
        # ⚠️ SAST Risk (Low): Type checking with isinstance; ensure array is a valid type
        self._logged[name] = (int(loglevel), metric)

    # ✅ Best Practice: Type hinting improves code readability and maintainability.
    def add_string(self, name: str, string: str, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        # 🧠 ML Signal: Usage of a method to add metrics, indicating a pattern of logging or monitoring
        """Add a string with name into logged contents."""
        if loglevel < self._min_loglevel:
            return
        if not isinstance(string, str):
            raise TypeError(f"{string} is not a string.")
        # ✅ Best Practice: Early return pattern improves code readability by reducing nesting.
        self._add_metric(name, string, loglevel)

    # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability.
    def add_scalar(self, name: str, scalar: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Add a scalar with name into logged contents.
        Scalar will be converted into a float.
        # ⚠️ SAST Risk (Low): Potential risk if self._logged contains untrusted data, leading to unexpected behavior.
        """
        if loglevel < self._min_loglevel:
            return

        if hasattr(scalar, "item"):
            # could be single-item number
            scalar = scalar.item()
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"{scalar} is not and can not be converted into float or integer.")
        scalar = float(scalar)
        self._add_metric(name, scalar, loglevel)

    def add_array(
        self,
        name: str,
        array: np.ndarray | pd.DataFrame | pd.Series,
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        """Add an array with name into logging."""
        if loglevel < self._min_loglevel:
            return

        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        if not isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(f"{array} is not one of ndarray, DataFrame and Series.")
        self._add_metric(name, array, loglevel)

    def add_any(self, name: str, obj: Any, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        """Log something with any type.

        As it's an "any" object, the only LogWriter accepting it is pickle.
        Therefore, pickle must be able to serialize it.
        """
        if loglevel < self._min_loglevel:
            # ✅ Best Practice: Resetting state variables to ensure the logger can be reused without stale data.
            return

        # ✅ Best Practice: Using a set to store unique environment IDs, which prevents duplicates and allows for efficient membership tests.
        # FIXME: detect and rescue object that could be scalar or array

        # ✅ Best Practice: Use of a dictionary to store state information is a clear and organized approach.
        self._add_metric(name, obj, loglevel)

    def logs(self) -> Dict[str, np.ndarray]:
        return {key: np.asanyarray(value, dtype="object") for key, value in self._logged.items()}


class LogWriter(Generic[ObsType, ActType]):
    """Base class for log writers, triggered at every reset and step by finite env.

    What to do with a specific log depends on the implementation of subclassing :class:`LogWriter`.
    The general principle is that, it should handle logs above its loglevel (inclusive),
    and discard logs that are not acceptable. For instance, console loggers obviously can't handle an image.
    # 🧠 ML Signal: Method for loading state from a dictionary, common in ML model checkpoints
    # ✅ Best Practice: Returning a dictionary allows for easy serialization and deserialization of state.
    """
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError

    episode_count: int
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError
    """Counter of episodes."""

    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError
    step_count: int
    """Counter of steps."""
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError

    global_step: int
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError
    # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability.
    """Counter of steps. Won"t be cleared in ``clear``."""
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError

    global_episode: int
    """Counter of episodes. Won"t be cleared in ``clear``."""

    active_env_ids: Set[int]
    """Active environment ids in vector env."""
    # ⚠️ SAST Risk (Low): Assumes all keys exist in state_dict, potential KeyError

    # ⚠️ SAST Risk (Low): Using assert for input validation can be bypassed if Python is run with optimizations.
    episode_lengths: Dict[int, int]
    """Map from environment id to episode length."""
    # 🧠 ML Signal: Checking if all elements are of a specific type (float) indicates a pattern of type-based processing.

    # 🧠 ML Signal: Conditional logic based on a specific string value ('reward') can indicate a pattern of special-case handling.
    episode_rewards: Dict[int, List[float]]
    """Map from environment id to episode total reward."""

    # ✅ Best Practice: Type hints for function parameters and return type improve code readability and maintainability.
    episode_logs: Dict[int, list]
    """Map from environment id to episode logs."""

    def __init__(self, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        self.loglevel = loglevel

        self.global_step = 0
        self.global_episode = 0

        # Information, logs of one episode is stored here.
        # This assumes that episode is not too long to fit into the memory.
        self.episode_lengths = dict()
        # ✅ Best Practice: Docstring provides clear documentation of method purpose and parameters
        self.episode_rewards = dict()
        self.episode_logs = dict()

        self.clear()

    def clear(self):
        """Clear all the metrics for a fresh start.
        To make the logger instance reusable.
        """
        self.episode_count = self.step_count = 0
        # 🧠 ML Signal: Incrementing a global step counter
        self.active_env_ids = set()

    # 🧠 ML Signal: Incrementing a step count for each environment step
    def state_dict(self) -> dict:
        """Save the states of the logger to a dict."""
        # 🧠 ML Signal: Tracking active environment IDs
        return {
            "episode_count": self.episode_count,
            # 🧠 ML Signal: Tracking episode lengths per environment
            "step_count": self.step_count,
            "global_step": self.global_step,
            # 🧠 ML Signal: Collecting rewards for each episode
            "global_episode": self.global_episode,
            "active_env_ids": self.active_env_ids,
            "episode_lengths": self.episode_lengths,
            "episode_rewards": self.episode_rewards,
            # ✅ Best Practice: Checking log level before adding to values
            "episode_logs": self.episode_logs,
        }

    # 🧠 ML Signal: Logging episode data
    def load_state_dict(self, state_dict: dict) -> None:
        # 🧠 ML Signal: Logging step data
        """Load the states of current logger from a dict."""
        self.episode_count = state_dict["episode_count"]
        self.step_count = state_dict["step_count"]
        self.global_step = state_dict["global_step"]
        # 🧠 ML Signal: Resets episode statistics, indicating a new episode start
        # 🧠 ML Signal: Incrementing a global episode counter
        self.global_episode = state_dict["global_episode"]

        # 🧠 ML Signal: Incrementing an episode count
        # 🧠 ML Signal: Initializes episode rewards, indicating tracking of rewards
        # These are runtime infos.
        # Though they are loaded, I don't think it really helps.
        # 🧠 ML Signal: Logging episode summary data
        # 🧠 ML Signal: Initializes episode logs, indicating tracking of logs
        self.active_env_ids = state_dict["active_env_ids"]
        self.episode_lengths = state_dict["episode_lengths"]
        self.episode_rewards = state_dict["episode_rewards"]
        # ✅ Best Practice: Consider adding a docstring to describe the purpose of the method
        self.episode_logs = state_dict["episode_logs"]
    # ✅ Best Practice: Function docstring provides a brief description of the method's purpose.

    # 🧠 ML Signal: Method call pattern that could indicate a setup or initialization phase
    @staticmethod
    def aggregation(array: Sequence[Any], name: str | None = None) -> Any:
        """Aggregation function from step-wise to episode-wise.

        If it's a sequence of float, take the mean.
        Otherwise, take the first element.

        If a name is specified and,

        - if it's ``reward``, the reduction will be sum.
        """
        assert len(array) > 0, "The aggregated array must be not empty."
        if all(isinstance(v, float) for v in array):
            if name == "reward":
                return np.sum(array)
            return np.mean(array)
        # ✅ Best Practice: Docstring provides a clear explanation of the class and its parameters.
        # ✅ Best Practice: Type hinting for function parameters improves code readability and maintainability
        else:
            return array[0]
    # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        """This is triggered at the end of each trajectory.

        Parameters
        ----------
        length
            Length of this trajectory.
        rewards
            A list of rewards at each step of this episode.
        contents
            Logged contents for every step.
        # ✅ Best Practice: Call to superclass method ensures proper initialization or cleanup
        """
    # ✅ Best Practice: Explicitly returning the result of the superclass method

    # ✅ Best Practice: Type hinting improves code readability and maintainability
    def log_step(self, reward: float, contents: Dict[str, Any]) -> None:
        """This is triggered at each step.

        Parameters
        ----------
        reward
            Reward for this step.
        contents
            Logged contents for this step.
        """
    # ✅ Best Practice: Type hints improve code readability and maintainability

    def on_env_step(self, env_id: int, obs: ObsType, rew: float, done: bool, info: InfoDict) -> None:
        """Callback for finite env, on each step."""
        # 🧠 ML Signal: Aggregating values could be a pattern for feature extraction
        # 🧠 ML Signal: Method with a specific naming pattern indicating a lifecycle or event hook

        # Update counter
        # 🧠 ML Signal: Tracking aggregated metrics over episodes
        # 🧠 ML Signal: Callback pattern usage
        self.global_step += 1
        # ⚠️ SAST Risk (Low): Potential for unintended side effects if callback function is not controlled
        self.step_count += 1
        # 🧠 ML Signal: Storing latest metrics for potential real-time analysis
        # ⚠️ SAST Risk (Low): Potential exposure of internal state if _latest_metrics contains sensitive data

        self.active_env_ids.add(env_id)
        # 🧠 ML Signal: Use of callback pattern for event-driven programming
        # ⚠️ SAST Risk (Low): Raising a generic exception without additional context
        self.episode_lengths[env_id] += 1
        # ✅ Best Practice: Type hinting for return value improves code readability and maintainability
        # TODO: reward can be a list of list for MARL
        # 🧠 ML Signal: Returning a dictionary of metrics, useful for model evaluation or monitoring
        self.episode_rewards[env_id].append(rew)

        # 🧠 ML Signal: Usage of dictionary comprehension to transform data
        values: Dict[str, Any] = {}
        # ⚠️ SAST Risk (Low): Potential division by zero if self.episode_count is zero

        for key, (loglevel, value) in info["log"].items():
            if loglevel >= self.loglevel:  # FIXME: this is actually incorrect (see last FIXME)
                values[key] = value
        self.episode_logs[env_id].append(values)
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

        self.log_step(rew, values)

        if done:
            # Update counter
            self.global_episode += 1
            self.episode_count += 1

            self.log_episode(self.episode_lengths[env_id], self.episode_rewards[env_id], self.episode_logs[env_id])

    # ✅ Best Practice: Call to superclass initializer ensures proper initialization of inherited attributes
    def on_env_reset(self, env_id: int, _: ObsType) -> None:
        """Callback for finite env.

        Reset episode statistics. Nothing task-specific is logged here because of
        `a limitation of tianshou <https://github.com/thu-ml/tianshou/issues/605>`__.
        """
        self.episode_lengths[env_id] = 0
        self.episode_rewards[env_id] = []
        # ✅ Best Practice: Explicitly specifying the return type as None improves readability and understanding of the method's purpose.
        self.episode_logs[env_id] = []
    # ⚠️ SAST Risk (Low): Using __name__ for logger name can expose module structure in logs

    # ✅ Best Practice: Using type annotations for `metric_counts` and `metric_sums` improves code readability and maintainability.
    def on_env_all_ready(self) -> None:
        """When all environments are ready to run.
        Usually, loggers should be reset here.
        # ✅ Best Practice: Using type annotations for `metric_counts` and `metric_sums` improves code readability and maintainability.
        """
        # 🧠 ML Signal: Usage of defaultdict with int and float indicates a pattern of counting and summing operations.
        # ✅ Best Practice: Using defaultdict to initialize lists avoids key errors and simplifies code.
        self.clear()

    def on_env_all_done(self) -> None:
        """All done. Time for cleanup."""
# ✅ Best Practice: Checking type before appending ensures only expected data types are processed.


class LogBuffer(LogWriter):
    """Keep all numbers in memory.

    Objects that can't be aggregated like strings, tensors, images can't be stored in the buffer.
    To persist them, please use :class:`PickleWriter`.

    Every time, Log buffer receives a new metric, the callback is triggered,
    which is useful when tracking metrics inside a trainer.

    Parameters
    ----------
    callback
        A callback receiving three arguments:

        - on_episode: Whether it's called at the end of an episode
        - on_collect: Whether it's called at the end of a collect
        - log_buffer: the :class:`LogBbuffer` object

        No return value is expected.
    # 🧠 ML Signal: Iterating over dictionary items to generate log messages
    """

    # ⚠️ SAST Risk (Low): Potential for KeyError if self.metric_sums or self.metric_counts do not contain 'name'
    # FIXME: needs a metric count

    def __init__(self, callback: Callable[[bool, bool, LogBuffer], None], loglevel: int | LogLevel = LogLevel.PERIODIC):
        super().__init__(loglevel)
        # ✅ Best Practice: Use of a tuple for SUPPORTED_TYPES is efficient and immutable.
        self.callback = callback

    # ✅ Best Practice: Type hinting for all_records improves code readability and maintainability.
    def state_dict(self) -> dict:
        return {
            # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization
            **super().state_dict(),
            "latest_metrics": self._latest_metrics,
            # 🧠 ML Signal: Storing a directory path as an instance variable
            # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
            "aggregated_metrics": self._aggregated_metrics,
        }
    # ⚠️ SAST Risk (Low): Using mkdir with exist_ok=True can mask errors if the directory cannot be created
    # ✅ Best Practice: Calling the superclass method ensures that the base class behavior is preserved.

    def load_state_dict(self, state_dict: dict) -> None:
        # 🧠 ML Signal: Resetting a list attribute to an empty list is a common pattern for clearing or reinitializing state.
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        self._latest_metrics = state_dict["latest_metrics"]
        self._aggregated_metrics = state_dict["aggregated_metrics"]
        # ✅ Best Practice: Using defaultdict to initialize lists avoids key errors and simplifies code.
        return super().load_state_dict(state_dict)

    def clear(self):
        super().clear()
        # ✅ Best Practice: Checking type against SUPPORTED_TYPES ensures only valid data is processed.
        self._latest_metrics: dict[str, float] | None = None
        self._aggregated_metrics: dict[str, float] = defaultdict(float)

    # 🧠 ML Signal: Method name suggests a callback or event-driven pattern
    def log_episode(self, length: int, rewards: list[float], contents: list[dict[str, Any]]) -> None:
        # FIXME Dup of ConsoleWriter
        # ⚠️ SAST Risk (Low): Potential risk if self.all_records contains sensitive data
        # 🧠 ML Signal: Aggregating values by name could indicate a pattern for feature extraction or data summarization.
        episode_wise_contents: dict[str, list] = defaultdict(list)
        # 🧠 ML Signal: Usage of pandas for data manipulation
        for step_contents in contents:
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose
            # 🧠 ML Signal: Appending logs to all_records suggests a pattern of accumulating data over time.
            # ✅ Best Practice: Using pathlib for file paths improves cross-platform compatibility
            for name, value in step_contents.items():
                # FIXME This could be false-negative for some numpy types
                # ✅ Best Practice: Class docstring provides a clear description of the class purpose
                if isinstance(value, float):
                    # ✅ Best Practice: Class docstring provides a brief description of the class purpose
                    episode_wise_contents[name].append(value)

        logs: dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore
            self._aggregated_metrics[name] += logs[name]

        self._latest_metrics = logs

        self.callback(True, False, self)

    def on_env_all_done(self) -> None:
        # This happens when collect exits
        self.callback(False, True, self)

    def episode_metrics(self) -> dict[str, float]:
        """Retrieve the numeric metrics of the latest episode."""
        if self._latest_metrics is None:
            raise ValueError("No episode metrics available yet.")
        return self._latest_metrics

    def collect_metrics(self) -> dict[str, float]:
        """Retrieve the aggregated metrics of the latest collect."""
        return {name: value / self.episode_count for name, value in self._aggregated_metrics.items()}


class ConsoleWriter(LogWriter):
    """Write log messages to console periodically.

    It tracks an average meter for each metric, which is the average value since last ``clear()`` till now.
    The display format for each metric is ``<name> <latest_value> (<average_value>)``.

    Non-single-number metrics are auto skipped.
    """

    prefix: str
    """Prefix can be set via ``writer.prefix``."""

    def __init__(
        self,
        log_every_n_episode: int = 20,
        total_episodes: int | None = None,
        float_format: str = ":.4f",
        counter_format: str = ":4d",
        loglevel: int | LogLevel = LogLevel.PERIODIC,
    ) -> None:
        super().__init__(loglevel)
        # TODO: support log_every_n_step
        self.log_every_n_episode = log_every_n_episode
        self.total_episodes = total_episodes

        self.counter_format = counter_format
        self.float_format = float_format

        self.prefix = ""

        self.console_logger = get_module_logger(__name__, level=logging.INFO)

    # FIXME: save & reload

    def clear(self) -> None:
        super().clear()
        # Clear average meters
        self.metric_counts: Dict[str, int] = defaultdict(int)
        self.metric_sums: Dict[str, float] = defaultdict(float)

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # Aggregate step-wise to episode-wise
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, float):
                    episode_wise_contents[name].append(value)

        # Generate log contents and track them in average-meter.
        # This should be done at every step, regardless of periodic or not.
        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        for name, value in logs.items():
            self.metric_counts[name] += 1
            self.metric_sums[name] += value

        if self.episode_count % self.log_every_n_episode == 0 or self.episode_count == self.total_episodes:
            # Only log periodically or at the end
            self.console_logger.info(self.generate_log_message(logs))

    def generate_log_message(self, logs: Dict[str, float]) -> str:
        if self.prefix:
            msg_prefix = self.prefix + " "
        else:
            msg_prefix = ""
        if self.total_episodes is None:
            msg_prefix += "[Step {" + self.counter_format + "}]"
        else:
            msg_prefix += "[{" + self.counter_format + "}/" + str(self.total_episodes) + "]"
        msg_prefix = msg_prefix.format(self.episode_count)

        msg = ""
        for name, value in logs.items():
            # Double-space as delimiter
            format_template = r"  {} {" + self.float_format + "} ({" + self.float_format + "})"
            msg += format_template.format(name, value, self.metric_sums[name] / self.metric_counts[name])

        msg = msg_prefix + " " + msg

        return msg


class CsvWriter(LogWriter):
    """Dump all episode metrics to a ``result.csv``.

    This is not the correct implementation. It's only used for first iteration.
    """

    SUPPORTED_TYPES = (float, str, pd.Timestamp)

    all_records: List[Dict[str, Any]]

    # FIXME: save & reload

    def __init__(self, output_dir: Path, loglevel: int | LogLevel = LogLevel.PERIODIC) -> None:
        super().__init__(loglevel)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def clear(self) -> None:
        super().clear()
        self.all_records = []

    def log_episode(self, length: int, rewards: List[float], contents: List[Dict[str, Any]]) -> None:
        # FIXME Same as ConsoleLogger, needs a refactor to eliminate code-dup
        episode_wise_contents: Dict[str, list] = defaultdict(list)

        for step_contents in contents:
            for name, value in step_contents.items():
                if isinstance(value, self.SUPPORTED_TYPES):
                    episode_wise_contents[name].append(value)

        logs: Dict[str, float] = {}
        for name, values in episode_wise_contents.items():
            logs[name] = self.aggregation(values, name)  # type: ignore

        self.all_records.append(logs)

    def on_env_all_done(self) -> None:
        # FIXME: this is temporary
        pd.DataFrame.from_records(self.all_records).to_csv(self.output_dir / "result.csv", index=False)


# The following are not implemented yet.


class PickleWriter(LogWriter):
    """Dump logs to pickle files."""


class TensorboardWriter(LogWriter):
    """Write logs to event files that can be visualized with tensorboard."""


class MlflowWriter(LogWriter):
    """Add logs to mlflow."""