# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

# 🧠 ML Signal: Constant value that might be used as a feature or label in ML models
from qlib.backtest.decision import Order, OrderDir

# ✅ Best Practice: Type annotations are used for function parameters and return type
from qlib.backtest.executor import SimulatorExecutor

# ✅ Best Practice: Default value for epsilon is provided, making the function flexible
from qlib.rl.order_execution import CategoricalActionInterpreter

# ✅ Best Practice: Use of pytest.mark.skipif to conditionally skip tests based on Python version
# 🧠 ML Signal: Function checks for closeness of two floating-point numbers, a common pattern in numerical computations
# 🧠 ML Signal: Function returning a specific object pattern
from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution

# ✅ Best Practice: Use of abs() function for calculating absolute difference
# ⚠️ SAST Risk (Low): Hardcoded stock_id and timestamps may lead to inflexibility or misuse

TOTAL_POSITION = 2100.0

python_version_request = pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
# ⚠️ SAST Risk (Low): Use of undefined variable TOTAL_POSITION


# 🧠 ML Signal: Consistent use of enum for direction
def is_close(a: float, b: float, epsilon: float = 1e-4) -> bool:
    # 🧠 ML Signal: Function signature and return type hint can be used to infer function behavior and expected output.
    return abs(a - b) <= epsilon


# 🧠 ML Signal: Use of specific timestamps for start and end time
# ✅ Best Practice: Use of dictionary to store configuration settings improves readability and maintainability.


def get_order() -> Order:
    return Order(
        stock_id="SH600000",
        amount=TOTAL_POSITION,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2019-03-04 09:30:00"),
        end_time=pd.Timestamp("2019-03-04 14:29:00"),
    )


def get_configs(order: Order) -> Tuple[dict, dict]:
    executor_config = {
        "class": "NestedExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "1day",
            "inner_strategy": {
                "class": "ProxySAOEStrategy",
                "module_path": "qlib.rl.order_execution.strategy",
            },
            "track_data": True,
            "inner_executor": {
                "class": "NestedExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "30min",
                    "inner_strategy": {
                        "class": "TWAPStrategy",
                        "module_path": "qlib.contrib.strategy.rule_strategy",
                    },
                    "inner_executor": {
                        "class": "SimulatorExecutor",
                        "module_path": "qlib.backtest.executor",
                        "kwargs": {
                            "time_per_step": "1min",
                            # ⚠️ SAST Risk (Low): Potential timezone issues with pd.Timestamp if order.start_time is naive.
                            # ✅ Best Practice: Use of dictionary to store configuration settings improves readability and maintainability.
                            "verbose": False,
                            "trade_type": SimulatorExecutor.TT_SERIAL,
                            "generate_report": False,
                            "track_data": True,
                        },
                    },
                    "track_data": True,
                },
            },
            "start_time": pd.Timestamp(order.start_time.date()),
            "end_time": pd.Timestamp(order.start_time.date()),
        },
    }

    exchange_config = {
        "freq": "1min",
        "codes": [order.stock_id],
        # ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility.
        "limit_threshold": ("$ask == 0", "$bid == 0"),
        # ✅ Best Practice: Clear and descriptive dictionary keys improve code readability.
        # 🧠 ML Signal: Returning a tuple of configurations can indicate a pattern of configuration management.
        # 🧠 ML Signal: Use of specific feature columns indicates a pattern for feature selection in ML models.
        "deal_price": ("If($ask == 0, $bid, $ask)", "If($bid == 0, $ask, $bid)"),
        "volume_threshold": {
            "all": ("cum", "0.2 * DayCumsum($volume, '9:30', '14:29')"),
            "buy": ("current", "$askV1"),
            "sell": ("current", "$bidV1"),
        },
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
        "trade_unit": None,
    }

    return executor_config, exchange_config


# 🧠 ML Signal: Use of historical feature columns suggests a pattern for time-series data handling in ML models.

# 🧠 ML Signal: Function call to get_configs indicates a pattern for dynamic configuration retrieval.


def get_simulator(order: Order) -> SingleAssetOrderExecution:
    DATA_ROOT_DIR = Path(__file__).parent.parent / ".data" / "rl" / "qlib_simulator"

    # fmt: off
    # 🧠 ML Signal: Returning an instance of SingleAssetOrderExecution suggests a pattern for order execution in trading systems.
    qlib_config = {
        "provider_uri_day": DATA_ROOT_DIR / "qlib_1d",
        "provider_uri_1min": DATA_ROOT_DIR / "qlib_1min",
        "feature_root_dir": DATA_ROOT_DIR / "qlib_handler_stock",
        # 🧠 ML Signal: Usage of a simulator pattern, common in testing environments
        "feature_columns_today": [
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
            # 🧠 ML Signal: Retrieving state from an object, indicating stateful behavior
            "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
        ],
        # ⚠️ SAST Risk (Low): Hardcoded timestamp, could lead to brittle tests
        "feature_columns_yesterday": [
            "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
            # 🧠 ML Signal: Use of constants in assertions, indicating expected behavior
            "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
        ],
    }
    # 🧠 ML Signal: Simulator step function, indicating iterative process
    # fmt: on

    # 🧠 ML Signal: Re-fetching state after an operation, indicating state change
    executor_config, exchange_config = get_configs(order)

    # ⚠️ SAST Risk (Low): Hardcoded timestamp, could lead to brittle tests
    return SingleAssetOrderExecution(
        order=order,
        # 🧠 ML Signal: Use of constants in assertions, indicating expected behavior
        qlib_config=qlib_config,
        executor_config=executor_config,
        # 🧠 ML Signal: Checking length of a collection, indicating expected data size
        exchange_config=exchange_config,
    )


# ⚠️ SAST Risk (Low): Hardcoded timestamp, could lead to brittle tests


# 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
@python_version_request
def test_simulator_first_step():
    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    # 🧠 ML Signal: Use of vectorized operations, indicating data processing
    order = get_order()
    simulator = get_simulator(order)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2019-03-04 09:30:00")
    assert state.position == TOTAL_POSITION
    # 🧠 ML Signal: Use of vectorized operations, indicating data processing

    # 🧠 ML Signal: Function to test simulator behavior, useful for ML models to learn expected outcomes
    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    AMOUNT = 300.0
    simulator.step(AMOUNT)
    # 🧠 ML Signal: Simulator setup with an order, indicating a pattern of initializing test scenarios
    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2019-03-04 10:00:00")
    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    assert state.position == TOTAL_POSITION - AMOUNT
    assert len(state.history_exec) == 30
    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    # 🧠 ML Signal: Iterative process to simulate steps, useful for learning loop patterns
    assert state.history_exec.index[0] == pd.Timestamp("2019-03-04 09:30:00")

    # 🧠 ML Signal: Use of is_close for floating-point comparison, indicating precision handling
    assert is_close(state.history_exec["market_volume"].iloc[0], 109382.382812)
    assert is_close(state.history_exec["market_price"].iloc[0], 149.566483)
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    # 🧠 ML Signal: Use of constants in assertions, indicating expected behavior
    assert (state.history_exec["amount"] == AMOUNT / 30).all()
    assert (state.history_exec["deal_amount"] == AMOUNT / 30).all()
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    # 🧠 ML Signal: Use of constants in assertions, indicating expected behavior
    assert is_close(state.history_exec["trade_price"].iloc[0], 149.566483)
    assert is_close(state.history_exec["trade_value"].iloc[0], 1495.664825)
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    # 🧠 ML Signal: Use of constants in assertions, indicating expected behavior
    assert is_close(
        state.history_exec["position"].iloc[0], TOTAL_POSITION - AMOUNT / 30
    )
    assert is_close(state.history_exec["ffr"].iloc[0], AMOUNT / TOTAL_POSITION / 30)
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    # 🧠 ML Signal: Use of conditional logic in assertions, indicating complex behavior

    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    assert is_close(state.history_steps["market_volume"].iloc[0], 1254848.5756835938)
    assert state.history_steps["amount"].iloc[0] == AMOUNT
    assert state.history_steps["deal_amount"].iloc[0] == AMOUNT
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    assert state.history_steps["ffr"].iloc[0] == AMOUNT / TOTAL_POSITION
    # 🧠 ML Signal: Usage of a function to get an order, indicating a pattern of dynamic input retrieval
    assert is_close(
        # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
        state.history_steps["pa"].iloc[0]
        * (1.0 if order.direction == OrderDir.SELL else -1.0),
        # 🧠 ML Signal: Usage of a function to get a simulator, indicating a pattern of dynamic environment setup
        (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000,
        # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    )


# 🧠 ML Signal: Instantiation of an interpreter with specific values, indicating a pattern of parameterized action interpretation

# ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production


@python_version_request
# ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
# 🧠 ML Signal: Initial state retrieval from a simulator, indicating a pattern of state-based simulation
def test_simulator_stop_twap() -> None:
    order = get_order()
    # ⚠️ SAST Risk (Low): Use of assert statements, which can be disabled in production
    # ✅ Best Practice: Initialize lists before loops to avoid repeated initialization
    simulator = get_simulator(order)
    # 🧠 ML Signal: Step function usage with dynamic actions, indicating a pattern of iterative simulation
    # 🧠 ML Signal: State update pattern after each simulation step
    # ✅ Best Practice: Use append to add elements to a list, ensuring clarity and maintainability
    # ⚠️ SAST Risk (Low): Potential floating-point precision issues in equality comparison
    NUM_STEPS = 7
    for i in range(NUM_STEPS):
        simulator.step(TOTAL_POSITION / NUM_STEPS)

    HISTORY_STEP_LENGTH = 30 * NUM_STEPS
    state = simulator.get_state()
    assert len(state.history_exec) == HISTORY_STEP_LENGTH

    assert (
        state.history_exec["deal_amount"] == TOTAL_POSITION / HISTORY_STEP_LENGTH
    ).all()
    assert is_close(
        state.history_steps["position"].iloc[0],
        TOTAL_POSITION * (NUM_STEPS - 1) / NUM_STEPS,
    )
    assert is_close(state.history_steps["position"].iloc[-1], 0.0)
    assert is_close(state.position, 0.0)
    assert is_close(state.metrics["ffr"], 1.0)

    assert is_close(
        state.metrics["market_price"], state.backtest_data.get_deal_price().mean()
    )
    assert is_close(
        state.metrics["market_volume"], state.backtest_data.get_volume().sum()
    )
    assert is_close(state.metrics["trade_price"], state.metrics["market_price"])
    assert is_close(state.metrics["pa"], 0.0)

    assert simulator.done()


@python_version_request
def test_interpreter() -> None:
    NUM_EXECUTION = 3
    order = get_order()
    simulator = get_simulator(order)
    interpreter_action = CategoricalActionInterpreter(values=NUM_EXECUTION)

    NUM_STEPS = 7
    state = simulator.get_state()
    position_history = []
    for i in range(NUM_STEPS):
        simulator.step(interpreter_action(state, 1))
        state = simulator.get_state()
        position_history.append(state.position)

        assert position_history[-1] == max(
            TOTAL_POSITION - TOTAL_POSITION / NUM_EXECUTION * (i + 1), 0.0
        )
