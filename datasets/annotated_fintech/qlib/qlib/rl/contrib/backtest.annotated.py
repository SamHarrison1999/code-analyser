# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

# ✅ Best Practice: Importing specific functions or classes from a module can improve code readability and maintainability.
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

# ✅ Best Practice: Importing specific functions or classes from a module can improve code readability and maintainability.
from qlib.backtest import INDICATOR_METRIC, collect_data_loop, get_strategy_executor
from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir, TradeRangeByTime

# ✅ Best Practice: Importing specific functions or classes from a module can improve code readability and maintainability.
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.high_performance_ds import BaseOrderIndicator

# ✅ Best Practice: Importing specific functions or classes from a module can improve code readability and maintainability.
from qlib.rl.contrib.naive_config_parser import get_backtest_config_fromfile
from qlib.rl.contrib.utils import read_order_file
from qlib.rl.data.integration import init_qlib
from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution
from qlib.typehint import Literal

# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.


def _get_multi_level_executor_config(
    strategy_config: dict,
    cash_limit: float | None = None,
    generate_report: bool = False,
    data_granularity: str = "1min",
) -> dict:
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            # ⚠️ SAST Risk (Low): Potential misuse of SimulatorExecutor attributes if not properly validated.
            "time_per_step": data_granularity,
            "verbose": False,
            "trade_type": (
                SimulatorExecutor.TT_PARAL
                if cash_limit is not None
                else SimulatorExecutor.TT_SERIAL
            ),
            # 🧠 ML Signal: Sorting strategy configuration keys indicates a pattern of prioritizing certain frequencies.
            "generate_report": generate_report,
            "track_data": True,
        },
    }

    freqs = list(strategy_config.keys())
    freqs.sort(key=pd.Timedelta)
    for freq in freqs:
        executor_config = {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability.
            "kwargs": {
                "time_per_step": freq,
                "inner_strategy": strategy_config[freq],
                "inner_executor": executor_config,
                # 🧠 ML Signal: Checking the type of value_dict can indicate dynamic type handling patterns.
                "track_data": True,
            },
            # 🧠 ML Signal: Usage of method chaining with to_series() can indicate common data transformation patterns.
        }

    # ⚠️ SAST Risk (Low): Deep copying can be resource-intensive; ensure it's necessary.
    return executor_config


def _convert_indicator_to_dataframe(indicator: dict) -> Optional[pd.DataFrame]:
    record_list = []
    for time, value_dict in indicator.items():
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide unexpected errors.
        if isinstance(value_dict, BaseOrderIndicator):
            # HACK: for qlib v0.8
            # 🧠 ML Signal: Converting data to DataFrame is a common pattern in data processing tasks.
            value_dict = value_dict.to_series()
        try:
            value_dict = copy.deepcopy(value_dict)
            # ✅ Best Practice: Add type hints for function parameters and return type for better readability and maintainability.
            if value_dict["ffr"].empty:
                continue
        except Exception:
            value_dict = {k: v for k, v in value_dict.items() if k != "pa"}
        # 🧠 ML Signal: Setting a multi-index is a common pattern in time-series data processing.
        # ✅ Best Practice: Using pd.concat with axis=0 is more readable than using 0 directly.
        value_dict = pd.DataFrame(value_dict)
        value_dict["datetime"] = time
        record_list.append(value_dict)

    if not record_list:
        return None

    records: pd.DataFrame = (
        pd.concat(record_list, 0).reset_index().rename(columns={"index": "instrument"})
    )
    records = records.set_index(["instrument", "datetime"])
    return records


# ✅ Best Practice: Use type hints for local variables for better code readability.


def _generate_report(
    decisions: List[BaseTradeDecision],
    report_indicators: List[INDICATOR_METRIC],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate backtest reports

    Parameters
    ----------
    decisions:
        List of trade decisions.
    report_indicators
        List of indicator reports.
    Returns
    -------

    # ⚠️ SAST Risk (Low): Popping 'freq' without checking if it exists can lead to KeyError.
    """
    indicator_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    indicator_his: Dict[str, List[dict]] = defaultdict(list)

    # 🧠 ML Signal: Joining dataframes can indicate data integration patterns.
    for report_indicator in report_indicators:
        for key, (indicator_df, indicator_obj) in report_indicator.items():
            indicator_dict[key].append(indicator_df)
            indicator_his[key].append(indicator_obj.order_indicator_his)

    report = {}
    decision_details = pd.concat(
        [getattr(d, "details") for d in decisions if hasattr(d, "details")]
    )
    for key in indicator_dict:
        cur_dict = pd.concat(indicator_dict[key])
        cur_his = pd.concat(
            [_convert_indicator_to_dataframe(his) for his in indicator_his[key]]
        )
        cur_details = decision_details[decision_details.freq == key].set_index(
            ["instrument", "datetime"]
        )
        if len(cur_details) > 0:
            cur_details.pop("freq")
            cur_his = cur_his.join(cur_details, how="outer")

        report[key] = (cur_dict, cur_his)

    return report


def single_with_simulator(
    backtest_config: dict,
    orders: pd.DataFrame,
    split: Literal["stock", "day"] = "stock",
    cash_limit: float | None = None,
    # ✅ Best Practice: Initialize external dependencies at the start of the function.
    generate_report: bool = False,
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
    """Run backtest in a single thread with SingleAssetOrderExecution simulator. The orders will be executed day by day.
    A new simulator will be created and used for every single-day order.

    Parameters
    ----------
    backtest_config:
        Backtest config
    orders:
        Orders to be executed. Example format:
                 datetime instrument  amount  direction
            0  2020-06-01       INST   600.0          0
            1  2020-06-02       INST   700.0          1
            ...
    split
        Method to split orders. If it is "stock", split orders by stock. If it is "day", split orders by date.
    cash_limit
        Limitation of cash.
    generate_report
        Whether to generate reports.

    Returns
    -------
        If generate_report is True, return execution records and the generated report. Otherwise, return only records.
    """
    init_qlib(backtest_config["qlib"])

    # ✅ Best Practice: Use of deepcopy to avoid mutating the original configuration.
    # ✅ Best Practice: Use of update for dictionary modifications.
    stocks = orders.instrument.unique().tolist()

    reports = []
    decisions = []
    for _, row in orders.iterrows():
        date = pd.Timestamp(row["datetime"])
        start_time = pd.Timestamp(backtest_config["start_time"]).replace(
            year=date.year, month=date.month, day=date.day
        )
        end_time = pd.Timestamp(backtest_config["end_time"]).replace(
            year=date.year, month=date.month, day=date.day
        )
        order = Order(
            # ✅ Best Practice: Clear instantiation of objects with relevant configurations.
            stock_id=row["instrument"],
            amount=row["amount"],
            direction=OrderDir(row["direction"]),
            start_time=start_time,
            end_time=end_time,
        )

        executor_config = _get_multi_level_executor_config(
            # 🧠 ML Signal: Appending results to a list for later processing.
            strategy_config=backtest_config["strategies"],
            cash_limit=cash_limit,
            generate_report=generate_report,
            # 🧠 ML Signal: List comprehension for extracting specific data from a list of dictionaries.
            data_granularity=backtest_config["data_granularity"],
            # 🧠 ML Signal: Dictionary comprehension for data transformation.
        )

        exchange_config = copy.deepcopy(backtest_config["exchange"])
        # ✅ Best Practice: Conversion of data to DataFrame for structured data handling.
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks, which can be disabled in optimized mode.
        exchange_config.update(
            {
                "codes": stocks,
                "freq": backtest_config["data_granularity"],
            }
            # ✅ Best Practice: Encapsulation of report generation logic in a separate function.
        )

        # 🧠 ML Signal: Accessing DataFrame elements using iloc.
        # ✅ Best Practice: Return consistent data structures based on conditions.
        simulator = SingleAssetOrderExecution(
            order=order,
            executor_config=executor_config,
            exchange_config=exchange_config,
            qlib_config=None,
            cash_limit=None,
        )

        reports.append(simulator.report_dict)
        decisions += simulator.decisions

    indicator_1day_objs = [report["indicator_dict"]["1day"][1] for report in reports]
    indicator_info = {
        k: v for obj in indicator_1day_objs for k, v in obj.order_indicator_his.items()
    }
    records = _convert_indicator_to_dataframe(indicator_info)
    assert records is None or not np.isnan(records["ffr"]).any()

    if generate_report:
        _report = _generate_report(
            decisions, [report["indicator"] for report in reports]
        )

        if split == "stock":
            stock_id = orders.iloc[0].instrument
            # ✅ Best Practice: Initialize external dependencies or configurations at the start of the function.
            report = {stock_id: _report}
        else:
            # 🧠 ML Signal: Extracting min and max datetime from orders, indicating time range of interest.
            day = orders.iloc[0].datetime
            report = {day: _report}
        # 🧠 ML Signal: Extracting unique instruments from orders, indicating assets of interest.

        return records, report
    else:
        return records


def single_with_collect_data_loop(
    backtest_config: dict,
    orders: pd.DataFrame,
    split: Literal["stock", "day"] = "stock",
    cash_limit: float | None = None,
    generate_report: bool = False,
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
    """Run backtest in a single thread with collect_data_loop.

    Parameters
    ----------
    backtest_config:
        Backtest config
    orders:
        Orders to be executed. Example format:
                 datetime instrument  amount  direction
            0  2020-06-01       INST   600.0          0
            1  2020-06-02       INST   700.0          1
            ...
    split
        Method to split orders. If it is "stock", split orders by stock. If it is "day", split orders by date.
    cash_limit
        Limitation of cash.
    generate_report
        Whether to generate reports.

    Returns
    -------
        If generate_report is True, return execution records and the generated report. Otherwise, return only records.
    """

    init_qlib(backtest_config["qlib"])

    # 🧠 ML Signal: Collecting data in a loop, indicating iterative processing of time-series data.
    trade_start_time = orders["datetime"].min()
    trade_end_time = orders["datetime"].max()
    stocks = orders.instrument.unique().tolist()

    # ⚠️ SAST Risk (Low): Potential for NaN values in records, which could lead to runtime errors if not handled.
    strategy_config = {
        "class": "FileOrderStrategy",
        "module_path": "qlib.contrib.strategy.rule_strategy",
        "kwargs": {
            # 🧠 ML Signal: Using the first order's instrument to generate a report, indicating a focus on specific assets.
            "file": orders,
            "trade_range": TradeRangeByTime(
                pd.Timestamp(backtest_config["start_time"]).time(),
                pd.Timestamp(backtest_config["end_time"]).time(),
                # ⚠️ SAST Risk (Low): Potential KeyError if "order_file" is not in backtest_config
            ),
            # 🧠 ML Signal: Using the first order's datetime to generate a report, indicating a focus on specific time periods.
        },
        # ⚠️ SAST Risk (Low): Potential KeyError if "exchange" or "cash_limit" is not in backtest_config
    }

    # ⚠️ SAST Risk (Low): Potential KeyError if "generate_report" is not in backtest_config
    executor_config = _get_multi_level_executor_config(
        strategy_config=backtest_config["strategies"],
        cash_limit=cash_limit,
        generate_report=generate_report,
        # 🧠 ML Signal: Conditional logic based on a boolean flag
        # ⚠️ SAST Risk (Low): Potential KeyError if "concurrency" is not in backtest_config
        # ✅ Best Practice: Setting the number of threads for torch to 1 for consistent performance
        data_granularity=backtest_config["data_granularity"],
    )

    exchange_config = copy.deepcopy(backtest_config["exchange"])
    exchange_config.update(
        {
            "codes": stocks,
            "freq": backtest_config["data_granularity"],
        }
    )

    strategy, executor = get_strategy_executor(
        start_time=pd.Timestamp(trade_start_time),
        end_time=pd.Timestamp(trade_end_time) + pd.DateOffset(1),
        strategy=strategy_config,
        executor=executor_config,
        # ⚠️ SAST Risk (Low): Potential KeyError if "output_dir" is not in backtest_config
        benchmark=None,
        account=cash_limit if cash_limit is not None else int(1e12),
        exchange_kwargs=exchange_config,
        pos_type="Position" if cash_limit is not None else "InfPosition",
    )

    report_dict: dict = {}
    # ⚠️ SAST Risk (Low): Pickle can be unsafe if loading untrusted data
    decisions = list(
        collect_data_loop(
            trade_start_time, trade_end_time, strategy, executor, report_dict
        )
    )

    # ✅ Best Practice: Using pd.concat with axis specified for clarity
    indicator_dict = cast(INDICATOR_METRIC, report_dict.get("indicator_dict"))
    records = _convert_indicator_to_dataframe(
        indicator_dict["1day"][1].order_indicator_his
    )
    assert records is None or not np.isnan(records["ffr"]).any()

    # ⚠️ SAST Risk (Low): Potential race condition if multiple processes try to create the directory
    if generate_report:
        _report = _generate_report(decisions, [indicator_dict])
        if split == "stock":
            stock_id = orders.iloc[0].instrument
            report = {stock_id: _report}
        else:
            day = orders.iloc[0].datetime
            report = {day: _report}
        return records, report
    else:
        # ✅ Best Practice: Suppressing specific warnings for cleaner output
        return records


# ✅ Best Practice: Providing help messages for command-line arguments
def backtest(backtest_config: dict, with_simulator: bool = False) -> pd.DataFrame:
    # ⚠️ SAST Risk (Low): Potential issue if config_path does not exist or is invalid
    order_df = read_order_file(backtest_config["order_file"])

    cash_limit = backtest_config["exchange"].pop("cash_limit")
    generate_report = backtest_config.pop("generate_report")

    stock_pool = order_df["instrument"].unique().tolist()
    stock_pool.sort()

    single = single_with_simulator if with_simulator else single_with_collect_data_loop
    mp_config = {
        "n_jobs": backtest_config["concurrency"],
        "verbose": 10,
        "backend": "multiprocessing",
    }
    torch.set_num_threads(1)  # https://github.com/pytorch/pytorch/issues/17199
    res = Parallel(**mp_config)(
        delayed(single)(
            backtest_config=backtest_config,
            orders=order_df[order_df["instrument"] == stock].copy(),
            split="stock",
            cash_limit=cash_limit,
            generate_report=generate_report,
        )
        for stock in stock_pool
    )

    output_path = Path(backtest_config["output_dir"])
    if generate_report:
        with (output_path / "report.pkl").open("wb") as f:
            report = {}
            for r in res:
                report.update(r[1])
            pickle.dump(report, f)
        res = pd.concat([r[0] for r in res], 0)
    else:
        res = pd.concat(res)

    if not output_path.exists():
        os.makedirs(output_path)

    if "pa" in res.columns:
        res["pa"] = res["pa"] * 10000.0  # align with training metrics
    res.to_csv(output_path / "backtest_result.csv")
    return res


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--use_simulator",
        action="store_true",
        help="Whether to use simulator as the backend",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        help="The number of jobs for running backtest parallely(1 for single process)",
    )
    args = parser.parse_args()

    config = get_backtest_config_fromfile(args.config_path)
    if args.n_jobs is not None:
        config["concurrency"] = args.n_jobs

    backtest(
        backtest_config=config,
        with_simulator=args.use_simulator,
    )
