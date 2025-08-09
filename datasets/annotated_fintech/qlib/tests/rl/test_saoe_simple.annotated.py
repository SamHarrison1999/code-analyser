# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import torch
from tianshou.data import Batch

from qlib.backtest import Order
from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.data import pickle_styled
# 🧠 ML Signal: Conditional test skipping based on Python version
from qlib.rl.data.pickle_styled import PickleProcessedDataProvider
from qlib.rl.order_execution import *
from qlib.rl.trainer import backtest, train
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
from qlib.rl.utils import ConsoleWriter, CsvWriter, EnvWrapperStatus

# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
pytestmark = pytest.mark.skipif(sys.version_info < (3, 8), reason="Pickle styled data only supports Python >= 3.8")

# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility

DATA_ROOT_DIR = Path(__file__).parent.parent / ".data" / "rl" / "intraday_saoe"
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
DATA_DIR = DATA_ROOT_DIR / "us"
BACKTEST_DATA_DIR = DATA_DIR / "backtest"
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# 🧠 ML Signal: Usage of a custom function to load data
FEATURE_DATA_DIR = DATA_DIR / "processed"
ORDER_DIR = DATA_DIR / "order" / "valid_bidir"
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# ✅ Best Practice: Asserting expected length of data for validation

CN_DATA_DIR = DATA_ROOT_DIR / "cn"
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# 🧠 ML Signal: Instantiation of a custom data provider class
# 🧠 ML Signal: Function name indicates a test case, useful for identifying test functions
CN_FEATURE_DATA_DIR = CN_DATA_DIR / "processed"
CN_ORDER_DIR = CN_DATA_DIR / "order" / "test"
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# 🧠 ML Signal: Usage of a method to retrieve data with specific parameters
# 🧠 ML Signal: Order creation with specific parameters, useful for learning order patterns
CN_POLICY_WEIGHTS_DIR = CN_DATA_DIR / "weights"

# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
# ✅ Best Practice: Asserting expected length of data for validation
# 🧠 ML Signal: Simulator initialization with order and data directory, useful for learning initialization patterns

def test_pickle_data_inspect():
    # 🧠 ML Signal: Retrieving state from simulator, useful for learning state management
    data = pickle_styled.load_simple_intraday_backtest_data(BACKTEST_DATA_DIR, "AAL", "2013-12-11", "close", 0)
    assert len(data) == 390
    # 🧠 ML Signal: Assertion to check initial state time, useful for learning expected state transitions

    provider = PickleProcessedDataProvider(DATA_DIR / "processed")
    # 🧠 ML Signal: Assertion to check initial position, useful for learning expected state transitions
    data = provider.get_data("AAL", "2013-12-11", 5, data.get_time_index())
    assert len(data.today) == len(data.yesterday) == 390
# 🧠 ML Signal: Simulator step function call, useful for learning step execution patterns


# 🧠 ML Signal: Retrieving state after step, useful for learning state management
def test_simulator_first_step():
    order = Order("AAL", 30.0, 0, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))
    # 🧠 ML Signal: Assertion to check history execution length, useful for learning expected state transitions

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    # 🧠 ML Signal: Assertion to check first index of history execution, useful for learning expected state transitions
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2013-12-11 09:30:00")
    # 🧠 ML Signal: Assertion to check market volume, useful for learning expected state transitions
    assert state.position == 30.0

    # 🧠 ML Signal: Assertion to check market price with tolerance, useful for learning expected state transitions
    simulator.step(15.0)
    state = simulator.get_state()
    assert len(state.history_exec) == 30
    assert state.history_exec.index[0] == pd.Timestamp("2013-12-11 09:30:00")
    # 🧠 ML Signal: Assertion to check deal amount consistency, useful for learning expected state transitions
    assert state.history_exec["market_volume"].iloc[0] == 450072.0
    assert abs(state.history_exec["market_price"].iloc[0] - 25.370001) < 1e-4
    # 🧠 ML Signal: Assertion to check trade price with tolerance, useful for learning expected state transitions
    # 🧠 ML Signal: Function name indicates a test case, useful for identifying test patterns
    assert (state.history_exec["amount"] == 0.5).all()
    assert (state.history_exec["deal_amount"] == 0.5).all()
    # 🧠 ML Signal: Assertion to check trade value with tolerance, useful for learning expected state transitions
    # 🧠 ML Signal: Order creation pattern, useful for understanding object initialization
    assert abs(state.history_exec["trade_price"].iloc[0] - 25.370001) < 1e-4
    assert abs(state.history_exec["trade_value"].iloc[0] - 12.68500) < 1e-4
    # 🧠 ML Signal: Assertion to check position after execution, useful for learning expected state transitions
    # 🧠 ML Signal: Simulator initialization pattern, useful for understanding object initialization
    assert state.history_exec["position"].iloc[0] == 29.5
    assert state.history_exec["ffr"].iloc[0] == 1 / 60
    # 🧠 ML Signal: Assertion to check fill factor rate, useful for learning expected state transitions
    # 🧠 ML Signal: Loop pattern, useful for understanding iteration over a fixed range

    assert state.history_steps["market_volume"].iloc[0] == 5041147.0
    # 🧠 ML Signal: Assertion to check market volume in steps, useful for learning expected state transitions
    # 🧠 ML Signal: Method call pattern, useful for understanding object behavior
    assert state.history_steps["amount"].iloc[0] == 15.0
    assert state.history_steps["deal_amount"].iloc[0] == 15.0
    # 🧠 ML Signal: Assertion to check amount in steps, useful for learning expected state transitions
    # 🧠 ML Signal: Method call pattern, useful for understanding object behavior
    assert state.history_steps["ffr"].iloc[0] == 0.5
    assert (
        # 🧠 ML Signal: Assertion to check deal amount in steps, useful for learning expected state transitions
        # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
        state.history_steps["pa"].iloc[0]
        == (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000
    # 🧠 ML Signal: Assertion to check fill factor rate in steps, useful for learning expected state transitions
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
    )

    # 🧠 ML Signal: Function name indicates a test, useful for identifying test patterns
    # 🧠 ML Signal: Assertion to check price adjustment calculation, useful for learning expected state transitions
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
    assert state.position == 15.0
    assert state.cur_time == pd.Timestamp("2013-12-11 10:00:00")
# 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
# 🧠 ML Signal: Usage of Order class with specific parameters, useful for learning object initialization patterns


# 🧠 ML Signal: Use of pytest.raises to test for exceptions, useful for learning testing patterns
# 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
def test_simulator_stop_twap():
    # 🧠 ML Signal: Assertion to check position after step, useful for learning expected state transitions
    order = Order("AAL", 13.0, 0, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))
    # 🧠 ML Signal: Instantiation of SingleAssetOrderExecutionSimple, useful for learning object creation patterns
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes

    # 🧠 ML Signal: Assertion to check current time after step, useful for learning expected state transitions
    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    # 🧠 ML Signal: Method call with specific argument, useful for learning method usage patterns
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
    for _ in range(13):
        # 🧠 ML Signal: Usage of assert statements to validate expected outcomes
        simulator.step(1.0)
    # 🧠 ML Signal: Re-instantiation of the simulator, useful for learning object lifecycle patterns
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes

    state = simulator.get_state()
    # 🧠 ML Signal: Method call with specific argument, useful for learning method usage patterns
    # 🧠 ML Signal: Assertion pattern, useful for understanding expected outcomes
    # 🧠 ML Signal: Instantiation of a simulator object with specific parameters
    assert len(state.history_exec) == 390
    assert (state.history_exec["deal_amount"] == 13 / 390).all()
    # 🧠 ML Signal: Use of pytest.raises to test for exceptions, useful for learning testing patterns
    # 🧠 ML Signal: Method call pattern, useful for understanding object behavior
    # 🧠 ML Signal: Validation of simulator's initial state
    assert state.history_steps["position"].iloc[0] == 12 and state.history_steps["position"].iloc[-1] == 0

    # 🧠 ML Signal: Method call with specific argument, useful for learning method usage patterns
    # 🧠 ML Signal: Validation of simulator's initial time
    assert (state.metrics["ffr"] - 1) < 1e-3
    assert abs(state.metrics["market_price"] - state.backtest_data.get_deal_price().mean()) < 1e-4
    # 🧠 ML Signal: Simulator step function usage with a parameter
    assert np.isclose(state.metrics["market_volume"], state.backtest_data.get_volume().sum())
    assert state.position == 0.0
    # 🧠 ML Signal: Validation of simulator's time after stepping
    assert abs(state.metrics["trade_price"] - state.metrics["market_price"]) < 1e-4
    assert abs(state.metrics["pa"]) < 1e-2

    # 🧠 ML Signal: Repeated simulator step function usage
    # 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
    assert simulator.done()

# 🧠 ML Signal: Instantiation of an Order object with specific parameters

# 🧠 ML Signal: Validation of simulator's execution history length
def test_simulator_stop_early():
    # 🧠 ML Signal: Instantiation of a simulator object with specific parameters
    order = Order("AAL", 1.0, 1, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))
    # 🧠 ML Signal: Validation of simulator's completion state

    # ✅ Best Practice: Use of assert statements for testing expected outcomes
    with pytest.raises(ValueError):
        # 🧠 ML Signal: Validation of simulator's execution history amount
        # 🧠 ML Signal: Usage of a specific interpreter with parameters, indicating a pattern for model training
        simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
        # ✅ Best Practice: Use of assert statements for testing expected outcomes
        simulator.step(2.0)
    # 🧠 ML Signal: Validation of simulator's metrics
    # 🧠 ML Signal: Usage of a specific interpreter with parameters, indicating a pattern for model training

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    # 🧠 ML Signal: Usage of a specific interpreter with parameters, indicating a pattern for model training
    simulator.step(1.0)

    # 🧠 ML Signal: Usage of a specific interpreter, indicating a pattern for model training
    with pytest.raises(AssertionError):
        simulator.step(1.0)
# ✅ Best Practice: Using a dictionary for keyword arguments improves readability and maintainability


# 🧠 ML Signal: Setting up environment wrapper with specific status, indicating a pattern for model training
def test_simulator_start_middle():
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 10:15:00"), pd.Timestamp("2013-12-11 15:44:59"))
    # 🧠 ML Signal: Observing the state of the simulator, indicating a pattern for model training

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert len(simulator.ticks_for_order) == 330
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:15:00")
    # ✅ Best Practice: Using assertions to validate expected outcomes
    simulator.step(2.0)
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:30:00")
    # ✅ Best Practice: Using assertions to validate expected outcomes

    for _ in range(10):
        # ✅ Best Practice: Using assertions to validate expected outcomes
        simulator.step(1.0)

    # ✅ Best Practice: Using assertions to validate expected outcomes
    simulator.step(2.0)
    assert len(simulator.history_exec) == 330
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert simulator.done()
    assert abs(simulator.history_exec["amount"].iloc[-1] - (1 + 2 / 15)) < 1e-4
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert abs(simulator.metrics["ffr"] - 1) < 1e-4

# 🧠 ML Signal: Setting up environment wrapper with specific status, indicating a pattern for model training

# 🧠 ML Signal: Observing the state of the simulator, indicating a pattern for model training
def test_interpreter():
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 10:15:00"), pd.Timestamp("2013-12-11 15:44:59"))

    # ✅ Best Practice: Using assertions to validate expected outcomes
    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    assert len(simulator.ticks_for_order) == 330
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:15:00")

    # 🧠 ML Signal: Stepping the simulator, indicating a pattern for model training
    # emulate a env status
    # 🧠 ML Signal: Setting up environment wrapper with specific status, indicating a pattern for model training
    class EmulateEnvWrapper(NamedTuple):
        status: EnvWrapperStatus

    # 🧠 ML Signal: Observing the state of the simulator, indicating a pattern for model training
    interpreter = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    interpreter_step = CurrentStepStateInterpreter(13)
    # ✅ Best Practice: Using assertions to validate expected outcomes
    interpreter_action = CategoricalActionInterpreter(20)
    interpreter_action_twap = TwapRelativeActionInterpreter()
    # ✅ Best Practice: Using assertions to validate expected outcomes

    wrapper_status_kwargs = dict(initial_state=order, obs_history=[], action_history=[], reward_history=[])
    # ✅ Best Practice: Using assertions to validate expected outcomes

    # first step
    # 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
    # ✅ Best Practice: Using assertions to validate expected outcomes
    interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=0, done=False, **wrapper_status_kwargs))

    # 🧠 ML Signal: Instantiation of an Order object with specific parameters
    # ✅ Best Practice: Using assertions to validate expected outcomes
    obs = interpreter(simulator.get_state())
    assert obs["cur_tick"] == 45
    # 🧠 ML Signal: Instantiation of a simulator object with specific parameters
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert obs["cur_step"] == 0
    assert obs["position"] == 15.0
    # ✅ Best Practice: Use of assert statement for testing expected outcomes
    # 🧠 ML Signal: Interpreting action from the simulator state, indicating a pattern for model training
    assert obs["position_history"][0] == 15.0
    # 🧠 ML Signal: Usage of a specific action interpreter for categorical actions
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(45))
    # ✅ Best Practice: Using assertions to validate expected outcomes
    assert np.sum(obs["data_processed"][45:]) == 0
    # 🧠 ML Signal: Use of a dictionary to store initial state and history information
    assert obs["data_processed_prev"].shape == (390, 5)
    # 🧠 ML Signal: Setting up environment wrapper with specific status, indicating a pattern for model training

    # 🧠 ML Signal: Initialization of a recurrent network with a specific observation space
    # first step: second interpreter
    interpreter_step.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=0, done=False, **wrapper_status_kwargs))
    # 🧠 ML Signal: Use of PPO algorithm with specific network and action space

    # 🧠 ML Signal: Interpreting action from the simulator state, indicating a pattern for model training
    obs = interpreter_step(simulator.get_state())
    assert obs["acquiring"] == 1
    # ✅ Best Practice: Using assertions to validate expected outcomes
    # 🧠 ML Signal: Iterative environment emulation with step tracking
    assert obs["position"] == 15.0

    # 🧠 ML Signal: Stepping the simulator multiple times, indicating a pattern for model training
    # 🧠 ML Signal: Creation of a batch with observations for policy input
    # second step
    simulator.step(5.0)
    interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=1, done=False, **wrapper_status_kwargs))
    # 🧠 ML Signal: Stepping the simulator, indicating a pattern for model training

    # ⚠️ SAST Risk (Low): Potential risk if output["act"] is not within expected range
    # 🧠 ML Signal: Setting up environment wrapper with specific status, indicating a pattern for model training
    obs = interpreter(simulator.get_state())
    assert obs["cur_tick"] == 60
    # 🧠 ML Signal: Function definition for testing a trading strategy
    assert obs["cur_step"] == 1
    assert obs["position"] == 10.0
    # 🧠 ML Signal: Logging configuration setup
    assert obs["position_history"][:2].tolist() == [15.0, 10.0]
    # ✅ Best Practice: Using assertions to validate expected outcomes
    # ⚠️ SAST Risk (Low): Assertions without exception handling could lead to unhandled exceptions
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(60))
    # 🧠 ML Signal: Loading orders from a specific directory
    assert np.sum(obs["data_processed"][60:]) == 0
    # 🧠 ML Signal: Observing the state of the simulator, indicating a pattern for model training

    # ⚠️ SAST Risk (Low): Assertion without error message
    # second step: action
    # ✅ Best Practice: Using assertions to validate expected outcomes
    # ✅ Best Practice: Use of parameterized tests to cover multiple scenarios
    action = interpreter_action(simulator.get_state(), 1)
    # ✅ Best Practice: Using assertions to validate expected outcomes
    # 🧠 ML Signal: State interpreter initialization with specific parameters
    # 🧠 ML Signal: Action interpreter initialization
    # 🧠 ML Signal: Policy initialization with state and action spaces
    assert action == 15 / 20

    interpreter_action_twap.env = EmulateEnvWrapper(
        status=EnvWrapperStatus(cur_step=1, done=False, **wrapper_status_kwargs)
    )
    action = interpreter_action_twap(simulator.get_state(), 1.5)
    assert action == 1.5

    # fast-forward
    for _ in range(10):
        # ✅ Best Practice: Using assertions to validate expected outcomes
        # 🧠 ML Signal: CSV writer setup for output
        # 🧠 ML Signal: Backtesting function call with specific parameters
        simulator.step(0.0)

    # last step
    simulator.step(5.0)
    interpreter.env = EmulateEnvWrapper(
        # 🧠 ML Signal: Function definition for testing a specific strategy
        status=EnvWrapperStatus(cur_step=12, done=simulator.done(), **wrapper_status_kwargs)
    )
    # 🧠 ML Signal: Setting logging configuration

    assert interpreter.env.status["done"]
    # 🧠 ML Signal: Reading metrics from CSV output
    # ⚠️ SAST Risk (Low): Loading orders from a potentially untrusted source

    obs = interpreter(simulator.get_state())
    # 🧠 ML Signal: Asserting the number of orders loaded
    # ⚠️ SAST Risk (Low): Assertion without error message
    assert obs["cur_tick"] == 375
    assert obs["cur_step"] == 12
    # 🧠 ML Signal: Initializing state interpreter with specific parameters
    # ⚠️ SAST Risk (Low): Assertion without error message
    assert obs["position"] == 0.0
    assert obs["position_history"][1:11].tolist() == [10.0] * 10
    # 🧠 ML Signal: Initializing action interpreter with specific parameters
    # ⚠️ SAST Risk (Low): Assertion without error message
    # 🧠 ML Signal: Creating a recurrent network with a specific observation space
    # 🧠 ML Signal: Initializing PPO policy with specific parameters
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(375))
    assert np.sum(obs["data_processed"][375:]) == 0


def test_network_sanity():
    # we won't check the correctness of networks here
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 9:30:00"), pd.Timestamp("2013-12-11 15:59:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    # ⚠️ SAST Risk (Medium): Loading model state from a file, potential for model tampering
    # ✅ Best Practice: Using a CSV writer to log output
    assert len(simulator.ticks_for_order) == 390
    # 🧠 ML Signal: Running a backtest with specific parameters and concurrency

    class EmulateEnvWrapper(NamedTuple):
        status: EnvWrapperStatus

    interpreter = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    # 🧠 ML Signal: Function definition for testing a PPO training process
    action_interp = CategoricalActionInterpreter(13)

    # 🧠 ML Signal: Setting logging configuration, indicating logging is important for this process
    wrapper_status_kwargs = dict(initial_state=order, obs_history=[], action_history=[], reward_history=[])

    # ⚠️ SAST Risk (Low): Potential deserialization of untrusted data with pickle
    network = Recurrent(interpreter.observation_space)
    # ⚠️ SAST Risk (Low): Reading metrics from a file, potential for data integrity issues
    policy = PPO(network, interpreter.observation_space, action_interp.action_space, 1e-3)
    # 🧠 ML Signal: Asserting the number of orders, indicating expected data size

    # 🧠 ML Signal: Asserting the length of metrics matches the number of orders
    for i in range(14):
        # 🧠 ML Signal: Asserting specific statistical properties of the metrics
        # 🧠 ML Signal: Initialization of state interpreter with specific parameters
        # 🧠 ML Signal: Network initialization with observation space
        # 🧠 ML Signal: PPO policy initialization with network and spaces
        # 🧠 ML Signal: Training function call with multiple parameters and configurations
        # 🧠 ML Signal: Partial function application for environment setup
        # 🧠 ML Signal: Reward function specification
        # 🧠 ML Signal: Vessel configuration for training episodes and updates
        # 🧠 ML Signal: Trainer configuration for iterations and logging
        interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=i, done=False, **wrapper_status_kwargs))
        obs = interpreter(simulator.get_state())
        batch = Batch(obs=[obs])
        output = policy(batch)
        assert 0 <= output["act"].item() <= 13
        if i < 13:
            simulator.step(1.0)
        else:
            assert obs["cur_tick"] == 389
            assert obs["cur_step"] == 12
            assert obs["position_history"][-1] == 3


@pytest.mark.parametrize("finite_env_type", ["dummy", "subproc", "shmem"])
def test_twap_strategy(finite_env_type):
    set_log_with_config(C.logging_config)
    orders = pickle_styled.load_orders(ORDER_DIR)
    assert len(orders) == 248

    state_interp = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    action_interp = TwapRelativeActionInterpreter()
    policy = AllOne(state_interp.observation_space, action_interp.action_space)
    csv_writer = CsvWriter(Path(__file__).parent / ".output")

    backtest(
        partial(SingleAssetOrderExecutionSimple, data_dir=DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        [ConsoleWriter(total_episodes=len(orders)), csv_writer],
        concurrency=4,
        finite_env_type=finite_env_type,
    )

    metrics = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(metrics) == 248
    assert np.isclose(metrics["ffr"].mean(), 1.0)
    assert np.isclose(metrics["pa"].mean(), 0.0)
    assert np.allclose(metrics["pa"], 0.0, atol=2e-3)


def test_cn_ppo_strategy():
    set_log_with_config(C.logging_config)
    # The data starts with 9:31 and ends with 15:00
    orders = pickle_styled.load_orders(CN_ORDER_DIR, start_time=pd.Timestamp("9:31"), end_time=pd.Timestamp("14:58"))
    assert len(orders) == 40

    state_interp = FullHistoryStateInterpreter(8, 240, 6, PickleProcessedDataProvider(CN_FEATURE_DATA_DIR))
    action_interp = CategoricalActionInterpreter(4)
    network = Recurrent(state_interp.observation_space)
    policy = PPO(network, state_interp.observation_space, action_interp.action_space, 1e-4)
    policy.load_state_dict(torch.load(CN_POLICY_WEIGHTS_DIR / "ppo_recurrent_30min.pth", map_location="cpu"))
    csv_writer = CsvWriter(Path(__file__).parent / ".output")

    backtest(
        partial(SingleAssetOrderExecutionSimple, data_dir=CN_DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        [ConsoleWriter(total_episodes=len(orders)), csv_writer],
        concurrency=4,
    )

    metrics = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(metrics) == len(orders)
    assert np.isclose(metrics["ffr"].mean(), 1.0)
    assert np.isclose(metrics["pa"].mean(), -16.21578303474833)
    assert np.isclose(metrics["market_price"].mean(), 58.68277690875527)
    assert np.isclose(metrics["trade_price"].mean(), 58.76063985000002)


def test_ppo_train():
    set_log_with_config(C.logging_config)
    # The data starts with 9:31 and ends with 15:00
    orders = pickle_styled.load_orders(CN_ORDER_DIR, start_time=pd.Timestamp("9:31"), end_time=pd.Timestamp("14:58"))
    assert len(orders) == 40

    state_interp = FullHistoryStateInterpreter(8, 240, 6, PickleProcessedDataProvider(CN_FEATURE_DATA_DIR))
    action_interp = CategoricalActionInterpreter(4)
    network = Recurrent(state_interp.observation_space)
    policy = PPO(network, state_interp.observation_space, action_interp.action_space, 1e-4)

    train(
        partial(SingleAssetOrderExecutionSimple, data_dir=CN_DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        PAPenaltyReward(),
        vessel_kwargs={"episode_per_iter": 100, "update_kwargs": {"batch_size": 64, "repeat": 5}},
        trainer_kwargs={"max_iters": 2, "loggers": ConsoleWriter(total_episodes=100)},
    )