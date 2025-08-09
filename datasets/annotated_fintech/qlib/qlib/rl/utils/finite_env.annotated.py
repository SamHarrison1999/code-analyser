# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This is to support finite env in vector env.
See https://github.com/thu-ml/tianshou/issues/322 for details.
"""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union, cast

import gym
import numpy as np
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from qlib.typehint import Literal

from .log import LogWriter

__all__ = [
    # ✅ Best Practice: Use of Literal for type safety and clarity in specifying allowed string values
    "generate_nan_observation",
    "check_nan_observation",
    # ✅ Best Practice: Use of Union for type hinting to allow multiple types, improving code flexibility
    # ✅ Best Practice: Type hinting improves code readability and maintainability.
    "FiniteVectorEnv",
    "FiniteDummyVectorEnv",
    # 🧠 ML Signal: Use of isinstance to check types is a common pattern.
    "FiniteSubprocVectorEnv",
    "FiniteShmemVectorEnv",
    # ⚠️ SAST Risk (Low): Recursive call with np.array(obj) could lead to unexpected behavior if obj is not a scalar.
    "FiniteEnvType",
    "vectorize_env",
# 🧠 ML Signal: Use of hasattr to check for attributes is a common pattern.
]

# 🧠 ML Signal: Use of isinstance to check for specific class instances.
FiniteEnvType = Literal["dummy", "subproc", "shmem"]
T = Union[dict, list, tuple, np.ndarray]
# 🧠 ML Signal: Use of np.issubdtype to check numpy data types.


# ✅ Best Practice: Use of np.full_like for creating arrays with the same shape and type.
def fill_invalid(obj: int | float | bool | T) -> T:
    if isinstance(obj, (int, float, bool)):
        return fill_invalid(np.array(obj))
    if hasattr(obj, "dtype"):
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # 🧠 ML Signal: Use of isinstance to check for dictionary type.
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating):
                # 🧠 ML Signal: Dictionary comprehension is a common pattern.
                # 🧠 ML Signal: Checking for numpy array type
                return np.full_like(obj, np.nan)
            return np.full_like(obj, np.iinfo(obj.dtype).max)
        # 🧠 ML Signal: Use of isinstance to check for list type.
        # 🧠 ML Signal: Checking for floating point type in numpy array
        # dealing with corner cases that numpy number is not supported by tianshou's sharray
        return fill_invalid(np.array(obj))
    # 🧠 ML Signal: List comprehension is a common pattern.
    # 🧠 ML Signal: Using np.isnan to check for NaN values
    elif isinstance(obj, dict):
        return {k: fill_invalid(v) for k, v in obj.items()}
    # 🧠 ML Signal: Use of isinstance to check for tuple type.
    # 🧠 ML Signal: Using np.iinfo to get max value of integer type
    elif isinstance(obj, list):
        # 🧠 ML Signal: Checking if all elements in array are the max integer value
        return [fill_invalid(v) for v in obj]
    # 🧠 ML Signal: Tuple comprehension is a common pattern.
    elif isinstance(obj, tuple):
        # 🧠 ML Signal: Checking for dictionary type
        return tuple(fill_invalid(v) for v in obj)
    # ⚠️ SAST Risk (Low): Raising a ValueError without specific handling could lead to unhandled exceptions.
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
    raise ValueError(f"Unsupported value to fill with invalid: {obj}")
# 🧠 ML Signal: Recursively checking all values in dictionary
# 🧠 ML Signal: Checking for list or tuple type


def is_invalid(arr: int | float | bool | T) -> bool:
    if isinstance(arr, np.ndarray):
        # 🧠 ML Signal: Recursively checking all elements in list or tuple
        if np.issubdtype(arr.dtype, np.floating):
            # 🧠 ML Signal: Sampling from a space could indicate exploration of state/action spaces in RL
            return np.isnan(arr).all()
        # 🧠 ML Signal: Checking for primitive types and numpy number
        return cast(bool, cast(np.ndarray, np.iinfo(arr.dtype).max == arr).all())
    # ✅ Best Practice: Add type hint for the return value for better readability and maintainability
    # 🧠 ML Signal: Function call to fill_invalid suggests data preprocessing or cleaning step
    if isinstance(arr, dict):
        # 🧠 ML Signal: Converting primitive types to numpy array for validation
        return all(is_invalid(o) for o in arr.values())
    if isinstance(arr, (list, tuple)):
        # 🧠 ML Signal: Default return value for invalid input
        # 🧠 ML Signal: Function usage pattern could indicate how often NaN checks are performed
        # 🧠 ML Signal: Custom environment class for reinforcement learning, indicating a specific use case or pattern.
        return all(is_invalid(o) for o in arr)
    # ✅ Best Practice: Class docstring provides detailed explanation of the class's purpose and usage.
    if isinstance(arr, (int, float, bool, np.number)):
        return is_invalid(np.array(arr))
    return True


def generate_nan_observation(obs_space: gym.Space) -> Any:
    """The NaN observation that indicates the environment receives no seed.

    We assume that obs is complex and there must be something like float.
    Otherwise this logic doesn't work.
    """

    sample = obs_space.sample()
    sample = fill_invalid(sample)
    return sample


def check_nan_observation(obs: Any) -> bool:
    """Check whether obs is generated by :func:`generate_nan_observation`."""
    return is_invalid(obs)


class FiniteVectorEnv(BaseVectorEnv):
    """To allow the paralleled env workers consume a single DataQueue until it's exhausted.

    See `tianshou issue #322 <https://github.com/thu-ml/tianshou/issues/322>`_.

    The requirement is to make every possible seed (stored in :class:`qlib.rl.utils.DataQueue` in our case)
    consumed by exactly one environment. This is not possible by tianshou's native VectorEnv and Collector,
    because tianshou is unaware of this "exactly one" constraint, and might launch extra workers.

    Consider a corner case, where concurrency is 2, but there is only one seed in DataQueue.
    The reset of two workers must be both called according to the logic in collect.
    The returned results of two workers are collected, regardless of what they are.
    The problem is, one of the reset result must be invalid, or repeated,
    because there's only one need in queue, and collector isn't aware of such situation.

    Luckily, we can hack the vector env, and make a protocol between single env and vector env.
    The single environment (should be :class:`qlib.rl.utils.EnvWrapper` in our case) is responsible for
    reading from queue, and generate a special observation when the queue is exhausted. The special obs
    is called "nan observation", because simply using none causes problems in shared-memory vector env.
    :class:`FiniteVectorEnv` then read the observations from all workers, and select those non-nan
    observation. It also maintains an ``_alive_env_ids`` to track which workers should never be
    called again. When also the environments are exhausted, it will raise StopIteration exception.

    The usage of this vector env in collector are two parts:

    1. If the data queue is finite (usually when inference), collector should collect "infinity" number of
       episodes, until the vector env exhausts by itself.
    2. If the data queue is infinite (usually in training), collector can set number of episodes / steps.
       In this case, data would be randomly ordered, and some repetitions wouldn't matter.

    One extra function of this vector env is that it has a logger that explicitly collects logs
    from child workers. See :class:`qlib.rl.utils.LogWriter`.
    """
    # ✅ Best Practice: Use of deepcopy to avoid unintended mutations of the original object
    # ⚠️ SAST Risk (Low): Using deepcopy can be expensive in terms of performance

    # 🧠 ML Signal: Usage of deepcopy indicates handling of complex data structures
    _logger: list[LogWriter]
    # 🧠 ML Signal: Method returns a deep copy of an internal attribute, indicating encapsulation and immutability practices

    # ✅ Best Practice: Use of deepcopy to avoid mutable default argument issues
    def __init__(
        self, logger: LogWriter | list[LogWriter] | None, env_fns: list[Callable[..., gym.Env]], **kwargs: Any
    ) -> None:
        # ✅ Best Practice: Check for None or NaN values to handle invalid observations
        super().__init__(env_fns, **kwargs)

        if isinstance(logger, list):
            self._logger = logger
        elif isinstance(logger, LogWriter):
            self._logger = [logger]
        else:
            self._logger = []
        self._alive_env_ids: Set[int] = set()
        self._reset_alive_envs()
        self._default_obs = self._default_info = self._default_rew = None
        self._zombie = False

        self._collector_guarded: bool = False

    # ✅ Best Practice: Use of a flag to manage the state of the collector guard
    def _reset_alive_envs(self) -> None:
        if not self._alive_env_ids:
            # 🧠 ML Signal: Iterating over loggers to perform actions, indicating a pattern of event handling
            # starting or running out
            self._alive_env_ids = set(range(self.env_num))
    # 🧠 ML Signal: Method call on logger to indicate readiness

    # to workaround with tianshou's buffer and batch
    # 🧠 ML Signal: Use of yield in a context manager pattern
    def _set_default_obs(self, obs: Any) -> None:
        if obs is not None and self._default_obs is None:
            self._default_obs = copy.deepcopy(obs)

    # ✅ Best Practice: Catching and ignoring specific exceptions to control flow
    def _set_default_info(self, info: Any) -> None:
        # ✅ Best Practice: Ensuring the flag is reset in the finally block
        if info is not None and self._default_info is None:
            self._default_info = copy.deepcopy(info)

    def _set_default_rew(self, rew: Any) -> None:
        # ⚠️ SAST Risk (Low): Use of assert statement for runtime checks can be disabled with optimization flags.
        # 🧠 ML Signal: Iterating over loggers to perform actions, indicating a pattern of event handling
        if rew is not None and self._default_rew is None:
            self._default_rew = copy.deepcopy(rew)
    # 🧠 ML Signal: Method call on logger to indicate completion
    # ⚠️ SAST Risk (Low): Warnings are used for notifying potential issues but do not prevent execution.

    def _get_default_obs(self) -> Any:
        return copy.deepcopy(self._default_obs)

    def _get_default_info(self) -> Any:
        return copy.deepcopy(self._default_info)

    def _get_default_rew(self) -> Any:
        # 🧠 ML Signal: Usage of a method to wrap or transform input IDs.
        return copy.deepcopy(self._default_rew)

    # 🧠 ML Signal: Resetting internal state or environment.
    # END

    # 🧠 ML Signal: Filtering or selecting based on a condition.
    @staticmethod
    def _postproc_env_obs(obs: Any) -> Optional[Any]:
        # ✅ Best Practice: Pre-allocating list with None for expected size.
        # reserved for shmem vector env to restore empty observation
        if obs is None or check_nan_observation(obs):
            # 🧠 ML Signal: Mapping IDs to indices for later reference.
            return None
        return obs

    # 🧠 ML Signal: Iterating over paired elements from two lists.
    @contextmanager
    def collector_guard(self) -> Generator[FiniteVectorEnv, None, None]:
        """Guard the collector. Recommended to guard every collect.

        This guard is for two purposes.

        1. Catch and ignore the StopIteration exception, which is the stopping signal
           thrown by FiniteEnv to let tianshou know that ``collector.collect()`` should exit.
        2. Notify the loggers that the collect is ready / done what it's ready / done.

        Examples
        --------
        >>> with finite_env.collector_guard():
        ...     collector.collect(n_episode=INF)
        # 🧠 ML Signal: Setting default values for observations.
        """
        self._collector_guarded = True
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags

        for logger in self._logger:
            # ✅ Best Practice: Wrapping ID handling in a separate method improves readability and maintainability
            # 🧠 ML Signal: Handling missing or None values.
            logger.on_env_all_ready()

        # ✅ Best Practice: Dictionary comprehension for mapping IDs to indices is concise and efficient
        try:
            # 🧠 ML Signal: Changing state to indicate no active environments.
            yield self
        # ✅ Best Practice: Using filter with lambda for list comprehension is clear and concise
        except StopIteration:
            # ⚠️ SAST Risk (Low): Raising StopIteration can be unexpected if not handled properly.
            pass
        # ✅ Best Practice: Initializing result with default values ensures all elements are defined
        finally:
            # 🧠 ML Signal: Returning a stacked array of observations.
            self._collector_guarded = False

        # ✅ Best Practice: Using np.stack to handle arrays is efficient and clear
        # At last trigger the loggers
        for logger in self._logger:
            logger.on_env_all_done()

    # ✅ Best Practice: Post-processing observations in a separate method enhances modularity
    def reset(
        self,
        id: int | List[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        # 🧠 ML Signal: Logging each environment step can be used for monitoring and analysis
        assert not self._zombie

        # Check whether it's guarded by collector_guard()
        if not self._collector_guarded:
            # ✅ Best Practice: Setting default values in separate methods improves code clarity
            warnings.warn(
                "Collector is not guarded by FiniteEnv. "
                "This may cause unexpected problems, like unexpected StopIteration exception, "
                # ✅ Best Practice: Use of inheritance to create a new class by combining FiniteVectorEnv and DummyVectorEnv
                "or missing logs.",
                RuntimeWarning,
            # ✅ Best Practice: Using a method to get default observation enhances code reuse
            # ✅ Best Practice: Use of inheritance to combine functionality from multiple classes
            )

        # ✅ Best Practice: Use of inheritance to combine functionality from multiple parent classes.
        wrapped_id = self._wrap_id(id)
        # ✅ Best Practice: Using a method to get default reward enhances code reuse
        # 🧠 ML Signal: Demonstrates use of multiple inheritance, which can be a feature to learn class design patterns.
        self._reset_alive_envs()
        # ✅ Best Practice: Using a method to get default info enhances code reuse

        # ask super to reset alive envs and remap to current index
        request_id = [i for i in wrapped_id if i in self._alive_env_ids]
        obs = [None] * len(wrapped_id)
        # ✅ Best Practice: Using map and np.stack for result transformation is efficient
        id2idx = {i: k for k, i in enumerate(wrapped_id)}
        if request_id:
            # ✅ Best Practice: Using cast for type hinting ensures the return type is clear
            # ✅ Best Practice: Docstring provides clear usage examples and warnings.
            for i, o in zip(request_id, super().reset(request_id)):
                obs[id2idx[i]] = self._postproc_env_obs(o)

        for i, o in zip(wrapped_id, obs):
            if o is None and i in self._alive_env_ids:
                self._alive_env_ids.remove(i)

        # logging
        for i, o in zip(wrapped_id, obs):
            if i in self._alive_env_ids:
                for logger in self._logger:
                    logger.on_env_reset(i, obs)

        # fill empty observation with default(fake) observation
        for o in obs:
            self._set_default_obs(o)
        for i, o in enumerate(obs):
            if o is None:
                obs[i] = self._get_default_obs()

        if not self._alive_env_ids:
            # comment this line so that the env becomes indispensable
            # self.reset()
            self._zombie = True
            raise StopIteration

        return np.stack(obs)

    def step(
        self,
        # ✅ Best Practice: Use of a dictionary for mapping types to classes improves readability and maintainability.
        action: np.ndarray,
        id: int | List[int] | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert not self._zombie
        wrapped_id = self._wrap_id(id)
        id2idx = {i: k for k, i in enumerate(wrapped_id)}
        # ⚠️ SAST Risk (Medium): Potential KeyError if env_type is not in env_type_cls_mapping.
        # 🧠 ML Signal: Pattern of creating multiple instances using a factory function.
        request_id = list(filter(lambda i: i in self._alive_env_ids, wrapped_id))
        result = [[None, None, False, None] for _ in range(len(wrapped_id))]

        # ask super to step alive envs and remap to current index
        if request_id:
            valid_act = np.stack([action[id2idx[i]] for i in request_id])
            for i, r in zip(request_id, zip(*super().step(valid_act, request_id))):
                result[id2idx[i]] = list(r)
                result[id2idx[i]][0] = self._postproc_env_obs(result[id2idx[i]][0])

        # logging
        for i, r in zip(wrapped_id, result):
            if i in self._alive_env_ids:
                for logger in self._logger:
                    logger.on_env_step(i, *r)

        # fill empty observation/info with default(fake)
        for _, r, ___, i in result:
            self._set_default_info(i)
            self._set_default_rew(r)
        for i, r in enumerate(result):
            if r[0] is None:
                result[i][0] = self._get_default_obs()
            if r[1] is None:
                result[i][1] = self._get_default_rew()
            if r[3] is None:
                result[i][3] = self._get_default_info()

        ret = list(map(np.stack, zip(*result)))
        return cast(Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ret)


class FiniteDummyVectorEnv(FiniteVectorEnv, DummyVectorEnv):
    pass


class FiniteSubprocVectorEnv(FiniteVectorEnv, SubprocVectorEnv):
    pass


class FiniteShmemVectorEnv(FiniteVectorEnv, ShmemVectorEnv):
    pass


def vectorize_env(
    env_factory: Callable[..., gym.Env],
    env_type: FiniteEnvType,
    concurrency: int,
    logger: LogWriter | List[LogWriter],
) -> FiniteVectorEnv:
    """Helper function to create a vector env. Can be used to replace usual VectorEnv.

    For example, once you wrote: ::

        DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])

    Now you can replace it with: ::

        finite_env_factory(lambda: gym.make(task), "dummy", env_num, my_logger)

    By doing such replacement, you have two additional features enabled (compared to normal VectorEnv):

    1. The vector env will check for NaN observation and kill the worker when its found.
       See :class:`FiniteVectorEnv` for why we need this.
    2. A logger to explicit collect logs from environment workers.

    Parameters
    ----------
    env_factory
        Callable to instantiate one single ``gym.Env``.
        All concurrent workers will have the same ``env_factory``.
    env_type
        dummy or subproc or shmem. Corresponding to
        `parallelism in tianshou <https://tianshou.readthedocs.io/en/master/api/tianshou.env.html#vectorenv>`_.
    concurrency
        Concurrent environment workers.
    logger
        Log writers.

    Warnings
    --------
    Please do not use lambda expression here for ``env_factory`` as it may create incorrectly-shared instances.

    Don't do: ::

        vectorize_env(lambda: EnvWrapper(...), ...)

    Please do: ::

        def env_factory(): ...
        vectorize_env(env_factory, ...)
    """
    env_type_cls_mapping: Dict[str, Type[FiniteVectorEnv]] = {
        "dummy": FiniteDummyVectorEnv,
        "subproc": FiniteSubprocVectorEnv,
        "shmem": FiniteShmemVectorEnv,
    }

    finite_env_cls = env_type_cls_mapping[env_type]

    return finite_env_cls(logger, [env_factory for _ in range(concurrency)])