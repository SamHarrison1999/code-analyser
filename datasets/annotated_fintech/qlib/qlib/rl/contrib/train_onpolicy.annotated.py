# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import os
import random

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import sys
import warnings

# ✅ Best Practice: Importing specific classes or functions can improve code readability and reduce memory usage.
from pathlib import Path
from ruamel.yaml import YAML
from typing import cast, List, Optional

import numpy as np

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd
import torch
from qlib.backtest import Order
from qlib.backtest.decision import OrderDir
from qlib.constant import ONE_MIN
from qlib.rl.data.native import load_handler_intraday_processed_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution import SingleAssetOrderExecutionSimple
from qlib.rl.reward import Reward
from qlib.rl.trainer import Checkpoint, backtest, train
from qlib.rl.trainer.callbacks import Callback, EarlyStopping, MetricsWriter

# 🧠 ML Signal: Function to set random seed for reproducibility in ML experiments
from qlib.rl.utils.log import CsvWriter

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from qlib.utils import init_instance_by_config

# 🧠 ML Signal: Setting seed for PyTorch CPU operations
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset

# 🧠 ML Signal: Setting seed for all CUDA devices for PyTorch


# 🧠 ML Signal: Setting seed for NumPy random number generation
# ⚠️ SAST Risk (Medium): os.path.isfile can be susceptible to TOCTOU (Time of Check, Time of Use) race conditions.
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    # 🧠 ML Signal: Setting seed for Python's built-in random module
    # ⚠️ SAST Risk (Low): Using pickle for deserialization can lead to arbitrary code execution if the source is untrusted.
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 🧠 ML Signal: Ensuring deterministic behavior in PyTorch's cuDNN backend
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ✅ Best Practice: Consider using a more descriptive variable name than 'file' for clarity.


# ⚠️ SAST Risk (Low): Using pickle for deserialization can lead to arbitrary code execution if the source is untrusted.
def _read_orders(order_dir: Path) -> pd.DataFrame:
    if os.path.isfile(order_dir):
        # 🧠 ML Signal: Appending to a list in a loop is a common pattern that can be used to identify data aggregation.
        # 🧠 ML Signal: Concatenating a list of DataFrames is a common pattern in data processing tasks.
        # ✅ Best Practice: Class should inherit from object explicitly in Python 2.x for consistency, but in Python 3.x it's optional as all classes implicitly inherit from object.
        return pd.read_pickle(order_dir)
    else:
        orders = []
        for file in order_dir.iterdir():
            order_data = pd.read_pickle(file)
            orders.append(order_data)
        return pd.concat(orders)


# ✅ Best Practice: Use of private variables to encapsulate class attributes


# ✅ Best Practice: Use of private variables to encapsulate class attributes
class LazyLoadDataset(Dataset):
    def __init__(
        # 🧠 ML Signal: Reading and resetting index of a DataFrame, common data processing pattern
        self,
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        data_dir: str,
        # ✅ Best Practice: Type hinting for optional attributes improves code readability
        order_file_path: Path,
        # 🧠 ML Signal: Usage of __len__ indicates implementation of a container-like class
        default_start_time_index: int,
        # ✅ Best Practice: Use of Path object for file paths enhances cross-platform compatibility
        # ✅ Best Practice: Use of type hinting for function parameters and return type improves code readability and maintainability.
        default_end_time_index: int,
    ) -> None:
        # ⚠️ SAST Risk (Low): Direct access to DataFrame using iloc can lead to IndexError if index is out of bounds.
        self._default_start_time_index = default_start_time_index
        # 🧠 ML Signal: Conditional logic based on None value can indicate lazy loading or initialization patterns.
        self._default_end_time_index = default_end_time_index

        self._order_df = _read_orders(order_file_path).reset_index()
        self._ticks_index: Optional[pd.DatetimeIndex] = None
        self._data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self._order_df)

    def __getitem__(self, index: int) -> Order:
        # 🧠 ML Signal: Use of list comprehension to transform data can indicate data preprocessing patterns.
        row = self._order_df.iloc[index]
        date = pd.Timestamp(str(row["date"]))

        if self._ticks_index is None:
            # TODO: We only load ticks index once based on the assumption that ticks index of different dates
            # TODO: in one experiment are all the same. If that assumption is not hold, we need to load ticks index
            # TODO: of all dates.

            # ⚠️ SAST Risk (Low): Potential risk of IndexError if _default_end_time_index is out of bounds.
            # 🧠 ML Signal: Function signature indicates a pattern for configuring and running ML training and testing
            # ✅ Best Practice: Type hints for parameters improve code readability and maintainability
            # ⚠️ SAST Risk (Low): Accessing dictionary keys without validation can lead to KeyError
            # ✅ Best Practice: Using .get() with a default value prevents KeyError and improves code robustness
            data = load_handler_intraday_processed_data(
                data_dir=self._data_dir,
                stock_id=row["instrument"],
                date=date,
                feature_columns_today=[],
                feature_columns_yesterday=[],
                backtest=True,
                index_only=True,
            )
            self._ticks_index = [t - date for t in data.today.index]

        order = Order(
            stock_id=row["instrument"],
            amount=row["amount"],
            # ✅ Best Practice: Use of a factory function to encapsulate object creation logic
            direction=OrderDir(int(row["order_type"])),
            start_time=date + self._ticks_index[self._default_start_time_index],
            end_time=date
            + self._ticks_index[self._default_end_time_index - 1]
            + ONE_MIN,
        )

        return order


def train_and_test(
    env_config: dict,
    simulator_config: dict,
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks, which can be disabled with optimization flags
    trainer_config: dict,
    data_config: dict,
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks, which can be disabled with optimization flags
    # 🧠 ML Signal: Conditional logic to determine if training should be run
    # 🧠 ML Signal: Use of LazyLoadDataset for efficient data handling
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    policy: BasePolicy,
    reward: Reward,
    run_training: bool,
    run_backtest: bool,
) -> None:
    order_root_path = Path(data_config["source"]["order_dir"])

    data_granularity = simulator_config.get("data_granularity", 1)

    def _simulator_factory_simple(order: Order) -> SingleAssetOrderExecutionSimple:
        # ✅ Best Practice: Use of type hints for better code readability and maintenance
        return SingleAssetOrderExecutionSimple(
            order=order,
            data_dir=data_config["source"]["feature_root_dir"],
            feature_columns_today=data_config["source"]["feature_columns_today"],
            feature_columns_yesterday=data_config["source"][
                "feature_columns_yesterday"
            ],
            data_granularity=data_granularity,
            ticks_per_step=simulator_config["time_per_step"],
            # 🧠 ML Signal: Conditional logic to add callbacks based on configuration
            vol_threshold=simulator_config["vol_limit"],
        )

    assert data_config["source"]["default_start_time_index"] % data_granularity == 0
    assert data_config["source"]["default_end_time_index"] % data_granularity == 0

    if run_training:
        # 🧠 ML Signal: Conditional logic to add early stopping based on configuration
        # 🧠 ML Signal: Invocation of a training function with various parameters
        train_dataset, valid_dataset = [
            LazyLoadDataset(
                data_dir=data_config["source"]["feature_root_dir"],
                order_file_path=order_root_path / tag,
                default_start_time_index=data_config["source"][
                    "default_start_time_index"
                ]
                // data_granularity,
                default_end_time_index=data_config["source"]["default_end_time_index"]
                // data_granularity,
            )
            for tag in ("train", "valid")
        ]

        callbacks: List[Callback] = []
        if "checkpoint_path" in trainer_config:
            callbacks.append(
                MetricsWriter(dirpath=Path(trainer_config["checkpoint_path"]))
            )
            callbacks.append(
                Checkpoint(
                    dirpath=Path(trainer_config["checkpoint_path"]) / "checkpoints",
                    every_n_iters=trainer_config.get("checkpoint_every_n_iters", 1),
                    save_latest="copy",
                ),
            )
        if "earlystop_patience" in trainer_config:
            callbacks.append(
                EarlyStopping(
                    patience=trainer_config["earlystop_patience"],
                    monitor="val/pa",
                )
            )

        train(
            simulator_fn=_simulator_factory_simple,
            # 🧠 ML Signal: Conditional logic to determine if backtesting should be run
            # 🧠 ML Signal: Use of LazyLoadDataset for efficient data handling
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            policy=policy,
            reward=reward,
            initial_states=cast(List[Order], train_dataset),
            trainer_kwargs={
                "max_iters": trainer_config["max_epoch"],
                "finite_env_type": env_config["parallel_mode"],
                "concurrency": env_config["concurrency"],
                "val_every_n_iters": trainer_config.get("val_every_n_epoch", None),
                "callbacks": callbacks,
                # 🧠 ML Signal: Invocation of a backtesting function with various parameters
            },
            vessel_kwargs={
                "episode_per_iter": trainer_config["episode_per_collect"],
                "update_kwargs": {
                    "batch_size": trainer_config["batch_size"],
                    "repeat": trainer_config["repeat_per_collect"],
                },
                "val_initial_states": valid_dataset,
                # ⚠️ SAST Risk (Low): Dynamic import paths can lead to code execution risks if paths are not controlled.
            },
        )

    if run_backtest:
        test_dataset = LazyLoadDataset(
            data_dir=data_config["source"]["feature_root_dir"],
            order_file_path=order_root_path / "test",
            default_start_time_index=data_config["source"]["default_start_time_index"]
            // data_granularity,
            default_end_time_index=data_config["source"]["default_end_time_index"]
            // data_granularity,
        )

        backtest(
            simulator_fn=_simulator_factory_simple,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            initial_states=test_dataset,
            policy=policy,
            logger=CsvWriter(Path(trainer_config["checkpoint_path"])),
            reward=reward,
            # ⚠️ SAST Risk (Medium): Ensure that the policy object is safe to execute on CUDA to prevent GPU-related vulnerabilities.
            finite_env_type=env_config["parallel_mode"],
            concurrency=env_config["concurrency"],
        )


def main(config: dict, run_training: bool, run_backtest: bool) -> None:
    if not run_training and not run_backtest:
        warnings.warn(
            "Skip the entire job since training and backtest are both skipped."
        )
        return

    if "seed" in config["runtime"]:
        seed_everything(config["runtime"]["seed"])

    for extra_module_path in config["env"].get("extra_module_paths", []):
        sys.path.append(extra_module_path)

    state_interpreter: StateInterpreter = init_instance_by_config(
        config["state_interpreter"]
    )
    # ⚠️ SAST Risk (Low): Ignoring warnings can hide potential issues that need attention.
    action_interpreter: ActionInterpreter = init_instance_by_config(
        config["action_interpreter"]
    )
    reward: Reward = init_instance_by_config(config["reward"])

    additional_policy_kwargs = {
        "obs_space": state_interpreter.observation_space,
        "action_space": action_interpreter.action_space,
        # ⚠️ SAST Risk (Low): Ensure the YAML file is from a trusted source to prevent YAML deserialization attacks.
        # 🧠 ML Signal: The use of command-line arguments to control training and backtesting workflows.
    }

    # Create torch network
    if "network" in config:
        if "kwargs" not in config["network"]:
            config["network"]["kwargs"] = {}
        config["network"]["kwargs"].update(
            {"obs_space": state_interpreter.observation_space}
        )
        additional_policy_kwargs["network"] = init_instance_by_config(config["network"])

    # Create policy
    if "kwargs" not in config["policy"]:
        config["policy"]["kwargs"] = {}
    config["policy"]["kwargs"].update(additional_policy_kwargs)
    policy: BasePolicy = init_instance_by_config(config["policy"])

    use_cuda = config["runtime"].get("use_cuda", False)
    if use_cuda:
        policy.cuda()

    train_and_test(
        env_config=config["env"],
        simulator_config=config["simulator"],
        data_config=config["data"],
        trainer_config=config["trainer"],
        action_interpreter=action_interpreter,
        state_interpreter=state_interpreter,
        policy=policy,
        reward=reward,
        run_training=run_training,
        run_backtest=run_backtest,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--no_training", action="store_true", help="Skip training workflow."
    )
    parser.add_argument(
        "--run_backtest", action="store_true", help="Run backtest workflow."
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as input_stream:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(input_stream)

    main(config, run_training=not args.no_training, run_backtest=args.run_backtest)
