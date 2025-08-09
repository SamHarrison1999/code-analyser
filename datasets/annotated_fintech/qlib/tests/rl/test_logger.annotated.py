# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Grouping imports by standard library, third-party, and local modules improves readability.
# Licensed under the MIT License.
from random import randint, choice
from pathlib import Path
import logging

import re
from typing import Any, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces
from tianshou.data import Collector, Batch
from tianshou.policy import BasePolicy

from qlib.log import set_log_with_config
from qlib.config import C
from qlib.constant import INF
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter

# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase.
from qlib.rl.simulator import Simulator
from qlib.rl.utils.data_queue import DataQueue

# 🧠 ML Signal: Initialization of a logger object, indicating logging behavior
from qlib.rl.utils.env_wrapper import InfoDict, EnvWrapper
from qlib.rl.utils.log import LogLevel, LogCollector, CsvWriter, ConsoleWriter

# 🧠 ML Signal: Initialization of observation space, indicating environment setup for RL
from qlib.rl.utils.finite_env import vectorize_env

# ✅ Best Practice: Use of *args and **kwargs allows for flexible function signatures

# 🧠 ML Signal: Initialization of action space, indicating environment setup for RL


# 🧠 ML Signal: Resets internal state, indicating a stateful object
class SimpleEnv(gym.Env[int, int]):
    def __init__(self) -> None:
        # ✅ Best Practice: Resetting logger at the start of the step ensures clean state for each step
        # ✅ Best Practice: Explicit return type annotation improves code readability
        self.logger = LogCollector()
        self.observation_space = gym.spaces.Discrete(2)
        # 🧠 ML Signal: Logging reward values can be used to track performance over time
        self.action_space = gym.spaces.Discrete(2)

    # 🧠 ML Signal: Logging random values can be used to analyze variability in actions or states
    def reset(self, *args: Any, **kwargs: Any) -> int:
        # ⚠️ SAST Risk (Low): Potentially large data structures being logged could lead to performance issues
        self.step_count = 0
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        # 🧠 ML Signal: Randomly setting 'done' can be used to simulate different episode lengths
        self.logger.reset()

        self.logger.add_scalar("reward", 42.0)
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.

        self.logger.add_scalar("a", randint(1, 10))
        # 🧠 ML Signal: Conditional logging based on step count can be used to track specific events
        self.logger.add_array("b", pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        # 🧠 ML Signal: Method signature suggests this is part of a model's forward pass, common in ML frameworks.

        # ✅ Best Practice: Incrementing step count at the end of the step ensures correct sequence tracking
        if self.step_count >= 3:
            # 🧠 ML Signal: Use of np.stack indicates data manipulation, a common operation in ML data processing.
            # 🧠 ML Signal: Method name 'learn' suggests a machine learning training or update operation
            done = choice([False, True])
        # ✅ Best Practice: Returning a tuple with clear structure improves readability and maintainability
        # ⚠️ SAST Risk (Low): Assumes batch is iterable and len(batch) is valid, which could lead to errors if batch is not as expected.
        else:
            # ✅ Best Practice: Consider adding type hints for batch and state for better readability and maintainability.
            done = False
        # 🧠 ML Signal: Testing function for logging and environment setup

        if 2 <= self.step_count <= 3:
            self.logger.add_scalar("c", randint(11, 20))
        # ⚠️ SAST Risk (Low): Potential risk if logging configuration is not properly sanitized

        self.step_count += 1

        # 🧠 ML Signal: Iterating over different environment class names
        return 1, 42.0, done, InfoDict(log=self.logger.logs(), aux_info={})

    def render(self, mode: str = "human") -> None:
        # 🧠 ML Signal: Vectorizing environment with different writers
        pass


class AnyPolicy(BasePolicy):
    # 🧠 ML Signal: Collecting data for a specified number of episodes
    def forward(self, batch, state=None):
        return Batch(act=np.stack([1] * len(batch)))

    # ✅ Best Practice: Asserting expected columns in the output file
    def learn(self, batch):
        pass


# ✅ Best Practice: Ensuring the output file has a minimum number of entries


# ✅ Best Practice: Class definition should be followed by a docstring explaining its purpose and usage
def test_simple_env_logger(caplog):
    set_log_with_config(C.logging_config)
    # ✅ Best Practice: Use of type hints for function parameters and return type
    # In order for caplog to capture log messages, we configure it here:
    # allow logs from the qlib logger to be passed to the parent logger.
    # ✅ Best Practice: Explicit call to superclass initializer
    C.logging_config["loggers"]["qlib"]["propagate"] = True
    # ⚠️ SAST Risk (Low): Regular expression usage can be risky if not properly controlled
    logging.config.dictConfig(C.logging_config)
    # ✅ Best Practice: Consider importing at the top of the file for better readability and maintainability
    # 🧠 ML Signal: Conversion of integer to float, indicating potential need for precision
    for venv_cls_name in ["dummy", "shmem", "subproc"]:
        # ✅ Best Practice: Asserting a minimum number of log lines
        writer = ConsoleWriter()
        # 🧠 ML Signal: Logging scalar values can be used to track model performance or environment changes
        csv_writer = CsvWriter(Path(__file__).parent / ".output")
        # ✅ Best Practice: Include a docstring to describe the purpose and behavior of the function
        venv = vectorize_env(
            lambda: SimpleEnv(), venv_cls_name, 4, [writer, csv_writer]
        )
        # ⚠️ SAST Risk (Low): Ensure that the logger handles data securely to prevent information leakage
        with venv.collector_guard():
            # ✅ Best Practice: Include a docstring to describe the purpose and behavior of the function
            # 🧠 ML Signal: Logging scalar values can be used to track model performance or environment changes
            # ✅ Best Practice: Consider using a property decorator if this method is intended to be an attribute accessor
            collector = Collector(AnyPolicy(), venv)
            collector.collect(n_episode=30)
        # 🧠 ML Signal: Use of modulus operator to determine a condition
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
        # ⚠️ SAST Risk (Low): Ensure that the logger handles data securely to prevent information leakage

        # ⚠️ SAST Risk (Low): Potential misuse of modulus operator if 'initial' is not a number
        output_file = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
        # ✅ Best Practice: Type annotations are used for function parameters and return type
        assert output_file.columns.tolist() == ["reward", "a", "c"]
        assert len(output_file) >= 30
    # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
    line_counter = 0
    # ✅ Best Practice: Use of @property decorator for creating a read-only attribute
    for line in caplog.text.splitlines():
        # ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage
        # 🧠 ML Signal: Use of spaces.Box suggests this is likely part of a reinforcement learning environment.
        line = line.strip()
        # ⚠️ SAST Risk (Low): Using np.inf as a bound can lead to unexpected behavior if not handled properly.
        if line:
            # ⚠️ SAST Risk (Low): Division by zero risk if action is 0
            line_counter += 1
            assert re.match(
                r".*reward .* {2}a .* \(([456])\.\d+\) {2}c .* \((14|15|16)\.\d+\)",
                line,
            )
    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
    assert line_counter >= 3


# ⚠️ SAST Risk (Low): Returning a different type (spaces.Discrete) than the annotated return type (spaces.Box) can lead to runtime errors


# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
# 🧠 ML Signal: Method signature suggests this is part of a model's forward pass, common in ML frameworks
class SimpleSimulator(Simulator[int, float, float]):
    def __init__(self, initial: int, **kwargs: Any) -> None:
        # ⚠️ SAST Risk (Low): Use of np.random can lead to non-deterministic behavior, which might be undesirable in some ML applications
        # 🧠 ML Signal: Method name 'learn' suggests a machine learning training or update operation
        super(SimpleSimulator, self).__init__(initial, **kwargs)
        # 🧠 ML Signal: Random action selection indicates this might be part of a reinforcement learning setup
        self.initial = float(initial)

    # 🧠 ML Signal: Usage of DataQueue with shuffle parameter
    # ✅ Best Practice: Consider setting a random seed for reproducibility

    # ✅ Best Practice: Use of context manager for DataQueue
    def step(self, action: float) -> None:
        import torch

        # 🧠 ML Signal: Factory pattern for creating environment wrappers

        self.initial += action
        self.env.logger.add_scalar("test_a", torch.tensor(233.0))
        self.env.logger.add_scalar("test_b", np.array(200))

    def get_state(self) -> float:
        return self.initial

    # ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
    def done(self) -> bool:
        return self.initial % 1 > 0.5


# 🧠 ML Signal: Use of vectorized environments for parallel processing


class DummyStateInterpreter(StateInterpreter[float, float]):
    # 🧠 ML Signal: Use of a collector pattern for gathering data
    def interpret(self, state: float) -> float:
        return state

    # ⚠️ SAST Risk (Low): Potential infinite loop if INF is not properly defined

    @property
    # 🧠 ML Signal: Use of assertions for validating data integrity
    # 🧠 ML Signal: Use of numpy for efficient numerical operations
    # 🧠 ML Signal: Use of assertions for validating specific conditions in data
    # 🧠 ML Signal: Checking for specific columns in a DataFrame
    # ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(), dtype=np.float32)


class DummyActionInterpreter(ActionInterpreter[float, int, float]):
    def interpret(self, state: float, action: int) -> float:
        return action / 100

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Discrete(5)


class RandomFivePolicy(BasePolicy):
    def forward(self, batch, state=None):
        return Batch(act=np.random.randint(5, size=len(batch)))

    def learn(self, batch):
        pass


def test_logger_with_env_wrapper():
    with DataQueue(list(range(20)), shuffle=False) as data_iterator:

        def env_wrapper_factory():
            return EnvWrapper(
                SimpleSimulator,
                DummyStateInterpreter(),
                DummyActionInterpreter(),
                data_iterator,
                logger=LogCollector(LogLevel.DEBUG),
            )

        # loglevel can be debugged here because metrics can all dump into csv
        # otherwise, csv writer might crash
        csv_writer = CsvWriter(
            Path(__file__).parent / ".output", loglevel=LogLevel.DEBUG
        )
        venv = vectorize_env(env_wrapper_factory, "shmem", 4, csv_writer)
        with venv.collector_guard():
            collector = Collector(RandomFivePolicy(), venv)
            collector.collect(n_episode=INF * len(venv))

    output_df = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(output_df) == 20
    # obs has an increasing trend
    assert (
        output_df["obs"].to_numpy()[:10].sum() < output_df["obs"].to_numpy()[10:].sum()
    )
    assert (output_df["test_a"] == 233).all()
    assert (output_df["test_b"] == 200).all()
    assert "steps_per_episode" in output_df and "reward" in output_df
