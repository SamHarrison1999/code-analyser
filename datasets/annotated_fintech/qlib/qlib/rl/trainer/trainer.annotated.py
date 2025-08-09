# ✅ Best Practice: Importing specific functions or classes instead of entire modules can improve readability and reduce memory usage.
# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Using `from __future__ import annotations` can help with forward references and improve type hinting in Python 3.7+.
# Licensed under the MIT License.
# ✅ Best Practice: Grouping standard library imports together and third-party imports separately improves readability.

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from __future__ import annotations

# ✅ Best Practice: Using type hints like `TypeVar` improves code readability and helps with static analysis.

# 🧠 ML Signal: The use of `torch` indicates potential machine learning or deep learning operations.
import collections

# 🧠 ML Signal: The use of `get_module_logger` suggests logging practices that could be analyzed for ML model training.
import copy

# 🧠 ML Signal: The import of `vectorize_env` and `FiniteVectorEnv` suggests reinforcement learning environment handling, which is relevant for ML models.
from contextlib import AbstractContextManager, contextmanager

# 🧠 ML Signal: The use of `Callback` and `TrainingVesselBase` indicates a pattern of using callbacks and training vessels, common in ML training loops.
from datetime import datetime

# ✅ Best Practice: Using `_logger` for module-level logging is a common pattern for consistent logging practices.
from pathlib import Path

# ✅ Best Practice: Using `TypeVar` for generic programming allows for more flexible and reusable code.
from typing import Any, Dict, Iterable, List, OrderedDict, Sequence, TypeVar, cast

import torch

from qlib.log import get_module_logger
from qlib.rl.simulator import InitialStateType
from qlib.rl.utils import (
    EnvWrapper,
    FiniteEnvType,
    LogBuffer,
    LogCollector,
    LogLevel,
    LogWriter,
    vectorize_env,
)
from qlib.rl.utils.finite_env import FiniteVectorEnv
from qlib.typehint import Literal

from .callbacks import Callback
from .vessel import TrainingVesselBase

_logger = get_module_logger(__name__)


T = TypeVar("T")


class Trainer:
    """
    Utility to train a policy on a particular task.

    Different from traditional DL trainer, the iteration of this trainer is "collect",
    rather than "epoch", or "mini-batch".
    In each collect, :class:`Collector` collects a number of policy-env interactions, and accumulates
    them into a replay buffer. This buffer is used as the "data" to train the policy.
    At the end of each collect, the policy is *updated* several times.

    The API has some resemblence with `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`__,
    but it's essentially different because this trainer is built for RL applications, and thus
    most configurations are under RL context.
    We are still looking for ways to incorporate existing trainer libraries, because it looks like
    big efforts to build a trainer as powerful as those libraries, and also, that's not our primary goal.

    It's essentially different
    `tianshou's built-in trainers <https://tianshou.readthedocs.io/en/master/api/tianshou.trainer.html>`__,
    as it's far much more complicated than that.

    Parameters
    ----------
    max_iters
        Maximum iterations before stopping.
    val_every_n_iters
        Perform validation every n iterations (i.e., training collects).
    logger
        Logger to record the backtest results. Logger must be present because
        without logger, all information will be lost.
    finite_env_type
        Type of finite env implementation.
    concurrency
        Parallel workers.
    fast_dev_run
        Create a subset for debugging.
        How this is implemented depends on the implementation of training vessel.
        For :class:`~qlib.rl.vessel.TrainingVessel`, if greater than zero,
        a random subset sized ``fast_dev_run`` will be used
        instead of ``train_initial_states`` and ``val_initial_states``.
    """

    should_stop: bool
    """Set to stop the training."""

    metrics: dict
    """Numeric metrics of produced in train/val/test.
    In the middle of training / validation, metrics will be of the latest episode.
    When each iteration of training / validation finishes, metrics will be the aggregation
    of all episodes encountered in this iteration.

    Cleared on every new iteration of training.

    In fit, validation metrics will be prefixed with ``val/``.
    """

    current_iter: int
    """Current iteration (collect) of training."""

    loggers: List[LogWriter]
    # ✅ Best Practice: Append a default logger to ensure logging functionality is always available.
    """A list of log writers."""

    # ✅ Best Practice: Use default empty list for callbacks to avoid mutable default arguments.
    def __init__(
        self,
        *,
        max_iters: int | None = None,
        # ✅ Best Practice: Use type annotations for better code readability and type checking.
        val_every_n_iters: int | None = None,
        loggers: LogWriter | List[LogWriter] | None = None,
        callbacks: List[Callback] | None = None,
        # ✅ Best Practice: Initialize flags and counters to manage the training process state.
        # ✅ Best Practice: Use 'cast' to explicitly indicate type conversion, improving code clarity.
        finite_env_type: FiniteEnvType = "subproc",
        concurrency: int = 2,
        # 🧠 ML Signal: Tracking the current iteration can be useful for monitoring training progress.
        fast_dev_run: int | None = None,
    ):
        # 🧠 ML Signal: Tracking the current episode can be useful for episodic training processes.
        # ✅ Best Practice: Method docstring provides a clear description of the method's purpose
        self.max_iters = max_iters
        self.val_every_n_iters = val_every_n_iters
        # 🧠 ML Signal: Tracking the current stage can be useful for managing different phases of training.

        # 🧠 ML Signal: Usage of dictionary to store metrics, indicating a pattern of data collection
        if isinstance(loggers, list):
            self.loggers = loggers
        elif isinstance(loggers, LogWriter):
            self.loggers = [loggers]
        else:
            self.loggers = []
        # 🧠 ML Signal: Collecting state information into a dictionary is a common pattern in ML for checkpointing.
        # 🧠 ML Signal: Storing the state of a model or component is a common pattern in ML for resuming training.
        # 🧠 ML Signal: Using callbacks is a common pattern in ML for extending functionality.

        self.loggers.append(
            LogBuffer(self._metrics_callback, loglevel=self._min_loglevel())
        )

        self.callbacks: List[Callback] = callbacks if callbacks is not None else []
        self.finite_env_type = finite_env_type
        self.concurrency = concurrency
        # 🧠 ML Signal: Using loggers is a common pattern in ML for tracking experiments.
        self.fast_dev_run = fast_dev_run

        # 🧠 ML Signal: Tracking stopping conditions is a common pattern in ML for early stopping.
        self.current_stage: Literal["train", "val", "test"] = "train"

        self.vessel: TrainingVesselBase = cast(TrainingVesselBase, None)

    # 🧠 ML Signal: Tracking iterations is a common pattern in ML for managing training loops.
    # ✅ Best Practice: Type hint for function return value improves code readability and maintainability

    # 🧠 ML Signal: Tracking episodes is a common pattern in reinforcement learning.
    def initialize(self):
        """Initialize the whole training process.

        The states here should be synchronized with state_dict.
        # 🧠 ML Signal: Tracking metrics is a common pattern in ML for evaluating model performance.
        """
        # 🧠 ML Signal: Accessing nested dictionary keys, common in model state management
        self.should_stop = False
        # 🧠 ML Signal: Loading state from a dictionary is a common pattern in ML for model checkpoints.
        self.current_iter = 0
        self.current_episode = 0
        # 🧠 ML Signal: Iterating over callbacks to load their states is a common pattern in ML frameworks.
        self.current_stage = "train"

    def initialize_iter(self):
        # 🧠 ML Signal: Iterating over loggers to load their states is a common pattern in ML frameworks.
        """Initialize one iteration / collect."""
        self.metrics = {}

    # 🧠 ML Signal: Restoring training control variables is a common pattern in ML for resuming training.
    def state_dict(self) -> dict:
        """Putting every states of current training into a dict, at best effort.

        It doesn't try to handle all the possible kinds of states in the middle of one training collect.
        For most cases at the end of each iteration, things should be usually correct.

        Note that it's also intended behavior that replay buffer data in the collector will be lost.
        # ✅ Best Practice: Type hinting for return type improves code readability and maintainability
        # 🧠 ML Signal: Usage of a helper function to retrieve named collections
        """
        return {
            "vessel": self.vessel.state_dict(),
            "callbacks": {
                name: callback.state_dict()
                for name, callback in self.named_callbacks().items()
            },
            "loggers": {
                name: logger.state_dict()
                for name, logger in self.named_loggers().items()
            },
            # 🧠 ML Signal: Usage of a helper function to retrieve a collection of objects
            "should_stop": self.should_stop,
            "current_iter": self.current_iter,
            "current_episode": self.current_episode,
            "current_stage": self.current_stage,
            "metrics": self.metrics,
        }

    @staticmethod
    def get_policy_state_dict(ckpt_path: Path) -> OrderedDict:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "vessel" in state_dict:
            # 🧠 ML Signal: Use of checkpointing to resume training
            state_dict = state_dict["vessel"]["policy"]
        return state_dict

    # ⚠️ SAST Risk (Medium): Loading model state from an external file can introduce security risks if the file is tampered with

    def load_state_dict(self, state_dict: dict) -> None:
        """Load all states into current trainer."""
        self.vessel.load_state_dict(state_dict["vessel"])
        for name, callback in self.named_callbacks().items():
            callback.load_state_dict(state_dict["callbacks"][name])
        # 🧠 ML Signal: Iterative training loop pattern
        for name, logger in self.named_loggers().items():
            logger.load_state_dict(state_dict["loggers"][name])
        self.should_stop = state_dict["should_stop"]
        self.current_iter = state_dict["current_iter"]
        self.current_episode = state_dict["current_episode"]
        self.current_stage = state_dict["current_stage"]
        self.metrics = state_dict["metrics"]

    # 🧠 ML Signal: Use of context manager for resource management during training
    def named_callbacks(self) -> Dict[str, Callback]:
        """Retrieve a collection of callbacks where each one has a name.
        Useful when saving checkpoints.
        """
        return _named_collection(self.callbacks)

    # 🧠 ML Signal: Conditional validation during training
    def named_loggers(self) -> Dict[str, LogWriter]:
        """Retrieve a collection of loggers where each one has a name.
        Useful when saving checkpoints.
        """
        return _named_collection(self.loggers)

    def fit(self, vessel: TrainingVesselBase, ckpt_path: Path | None = None) -> None:
        """Train the RL policy upon the defined simulator.

        Parameters
        ----------
        vessel
            A bundle of all elements used in training.
        ckpt_path
            Load a pre-trained / paused training checkpoint.
        """
        self.vessel = vessel
        vessel.assign_trainer(self)
        # ✅ Best Practice: Ensure the vessel is assigned to the trainer before proceeding

        if ckpt_path is not None:
            # ✅ Best Practice: Initialize iterators before starting the test
            _logger.info("Resuming states from %s", str(ckpt_path))
            self.load_state_dict(torch.load(ckpt_path, weights_only=False))
        else:
            # 🧠 ML Signal: Use of callback hooks indicates a pattern for extensibility and monitoring
            self.initialize()

        # ⚠️ SAST Risk (Low): Ensure _wrap_context handles exceptions properly to avoid resource leaks
        self._call_callback_hooks("on_fit_start")

        # 🧠 ML Signal: Use of vectorized environments for testing indicates a pattern for efficiency
        while not self.should_stop:
            # ✅ Best Practice: Include a docstring to describe the purpose and usage of the function
            msg = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tTrain iteration {self.current_iter + 1}/{self.max_iters}"
            # 🧠 ML Signal: Testing the vessel with a vector environment suggests a pattern for parallel processing
            _logger.info(msg)
            # 🧠 ML Signal: Conditional logic based on 'finite_env_type' can indicate different environment setups.

            # ✅ Best Practice: Explicitly delete objects to free up resources
            self.initialize_iter()
            # ⚠️ SAST Risk (Low): Deep copy can be expensive in terms of memory and performance.

            # 🧠 ML Signal: Use of callback hooks indicates a pattern for extensibility and monitoring
            self._call_callback_hooks("on_iter_start")

            self.current_stage = "train"
            self._call_callback_hooks("on_train_start")

            # ✅ Best Practice: Using a wrapper class to encapsulate environment setup.
            # TODO
            # Add a feature that supports reloading the training environment every few iterations.
            with _wrap_context(vessel.train_seed_iterator()) as iterator:
                vector_env = self.venv_from_iterator(iterator)
                self.vessel.train(vector_env)
                del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.

            self._call_callback_hooks("on_train_end")
            # ✅ Best Practice: Use of logging with configurable log level.

            if (
                self.val_every_n_iters is not None
                and (self.current_iter + 1) % self.val_every_n_iters == 0
            ):
                # Implementation of validation loop
                self.current_stage = "val"
                # 🧠 ML Signal: Use of vectorized environments can indicate parallel processing or batch processing.
                self._call_callback_hooks("on_validate_start")
                with _wrap_context(vessel.val_seed_iterator()) as iterator:
                    vector_env = self.venv_from_iterator(iterator)
                    # 🧠 ML Signal: Use of callback pattern for metrics collection
                    self.vessel.validate(vector_env)
                    del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
                # 🧠 ML Signal: Conditional logic based on episode state

                self._call_callback_hooks("on_validate_end")
            # 🧠 ML Signal: Accessing episode metrics from a log buffer

            # This iteration is considered complete.
            # Bumping the current iteration counter.
            # 🧠 ML Signal: Conditional logic based on collect state
            self.current_iter += 1

            # 🧠 ML Signal: Iterating over a list of callbacks to invoke methods dynamically
            if self.max_iters is not None and self.current_iter >= self.max_iters:
                # 🧠 ML Signal: Metrics transformation based on validation stage
                self.should_stop = True
            # ⚠️ SAST Risk (Medium): Using getattr to dynamically call methods can lead to security risks if hook_name is not validated

            # ✅ Best Practice: Use of dictionary update method for merging metrics
            self._call_callback_hooks("on_iter_end")
        # ✅ Best Practice: Check for empty list before processing
        # 🧠 ML Signal: Passing a mix of positional and keyword arguments to a function

        self._call_callback_hooks("on_fit_end")

    # 🧠 ML Signal: Use of min function to determine minimum log level
    def test(self, vessel: TrainingVesselBase) -> None:
        """Test the RL policy against the simulator.

        The simulator will be fed with data generated in ``test_seed_iterator``.

        Parameters
        ----------
        vessel
            A bundle of all related elements.
        # ✅ Best Practice: Type hinting for the function parameters and return type improves code readability and maintainability.
        """
        self.vessel = vessel
        # ✅ Best Practice: Yielding the object directly if it's not a context manager allows for flexible usage.
        vessel.assign_trainer(self)

        # ✅ Best Practice: Using collections.Counter for counting occurrences is efficient and improves code readability.
        self.initialize_iter()

        self.current_stage = "test"
        # 🧠 ML Signal: Using type names as keys in a dictionary could indicate a pattern of dynamic typing or runtime type analysis.
        self._call_callback_hooks("on_test_start")
        with _wrap_context(vessel.test_seed_iterator()) as iterator:
            # ✅ Best Practice: Using f-strings for string formatting is more readable and concise.
            vector_env = self.venv_from_iterator(iterator)
            self.vessel.test(vector_env)
            del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
        self._call_callback_hooks("on_test_end")

    def venv_from_iterator(
        self, iterator: Iterable[InitialStateType]
    ) -> FiniteVectorEnv:
        """Create a vectorized environment from iterator and the training vessel."""

        def env_factory():
            # FIXME: state_interpreter and action_interpreter are stateful (having a weakref of env),
            # and could be thread unsafe.
            # I'm not sure whether it's a design flaw.
            # I'll rethink about this when designing the trainer.

            if self.finite_env_type == "dummy":
                # We could only experience the "threading-unsafe" problem in dummy.
                state = copy.deepcopy(self.vessel.state_interpreter)
                action = copy.deepcopy(self.vessel.action_interpreter)
                rew = copy.deepcopy(self.vessel.reward)
            else:
                state = self.vessel.state_interpreter
                action = self.vessel.action_interpreter
                rew = self.vessel.reward

            return EnvWrapper(
                self.vessel.simulator_fn,
                state,
                action,
                iterator,
                rew,
                logger=LogCollector(min_loglevel=self._min_loglevel()),
            )

        return vectorize_env(
            env_factory,
            self.finite_env_type,
            self.concurrency,
            self.loggers,
        )

    def _metrics_callback(
        self, on_episode: bool, on_collect: bool, log_buffer: LogBuffer
    ) -> None:
        if on_episode:
            # Update the global counter.
            self.current_episode = log_buffer.global_episode
            metrics = log_buffer.episode_metrics()
        elif on_collect:
            # Update the latest metrics.
            metrics = log_buffer.collect_metrics()
        if self.current_stage == "val":
            metrics = {"val/" + name: value for name, value in metrics.items()}
        self.metrics.update(metrics)

    def _call_callback_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            fn(self, self.vessel, *args, **kwargs)

    def _min_loglevel(self):
        if not self.loggers:
            return LogLevel.PERIODIC
        else:
            # To save bandwidth
            return min(lg.loglevel for lg in self.loggers)


@contextmanager
def _wrap_context(obj):
    """Make any object a (possibly dummy) context manager."""

    if isinstance(obj, AbstractContextManager):
        # obj has __enter__ and __exit__
        with obj as ctx:
            yield ctx
    else:
        yield obj


def _named_collection(seq: Sequence[T]) -> Dict[str, T]:
    """Convert a list into a dict, where each item is named with its type."""
    res = {}
    retry_cnt: collections.Counter = collections.Counter()
    for item in seq:
        typename = type(item).__name__.lower()
        key = (
            typename if retry_cnt[typename] == 0 else f"{typename}{retry_cnt[typename]}"
        )
        retry_cnt[typename] += 1
        res[key] = item
    return res
