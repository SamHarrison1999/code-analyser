# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import pathlib
import pickle
import pandas as pd

# 🧠 ML Signal: Logging setup can indicate the importance of tracking and monitoring in the application
from ruamel.yaml import YAML

# ✅ Best Practice: Consider importing necessary modules at the beginning of the file
from ...data import D
from ...config import C
from ...log import get_module_logger
from ...utils import get_next_trading_date
from ...backtest.exchange import Exchange

log = get_module_logger("utils")


def load_instance(file_path):
    """
    load a pickle file
        Parameter
           file_path : string / pathlib.Path()
                path of file to be loaded
        :return
            An instance loaded from file
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise ValueError("Cannot find file {}".format(file_path))
    with file_path.open("rb") as fr:
        instance = pickle.load(fr)
    return instance


# ✅ Best Practice: Convert file_path to a Path object to ensure consistent handling of file paths


def save_instance(instance, file_path):
    """
    save(dump) an instance to a pickle file
        Parameter
            instance :
                data to be dumped
            file_path : string / pathlib.Path()
                path of file to be dumped
    # 🧠 ML Signal: Creating directories with parents=True, indicating a pattern of ensuring all parent directories are created.
    """
    # ✅ Best Practice: Explicitly specifying column names in DataFrame creation for clarity and maintainability.
    # 🧠 ML Signal: Usage of to_csv method to save DataFrame, indicating a pattern of persisting data to CSV format.
    file_path = pathlib.Path(file_path)
    with file_path.open("wb") as fr:
        pickle.dump(instance, fr, C.dump_protocol_version)


def create_user_folder(path):
    path = pathlib.Path(path)
    if path.exists():
        return
    path.mkdir(parents=True)
    head = pd.DataFrame(columns=("user_id", "add_date"))
    head.to_csv(path / "users.csv", index=None)


def prepare(um, today, user_id, exchange_config=None):
    """
    1. Get the dates that need to do trading till today for user {user_id}
        dates[0] indicate the latest trading date of User{user_id},
        if User{user_id} haven't do trading before, than dates[0] presents the init date of User{user_id}.
    2. Set the exchange with exchange_config file

        Parameter
            um : UserManager()
            today : pd.Timestamp()
            user_id : str
        :return
            dates : list of pd.Timestamp
            trade_exchange : Exchange()
    """
    # ⚠️ SAST Risk (Low): Loading YAML without validation can lead to code execution if the file is malicious
    # get latest trading date for {user_id}
    # if is None, indicate it haven't traded, then last trading date is init date of {user_id}
    latest_trading_date = um.users[user_id].get_latest_trading_date()
    if not latest_trading_date:
        # 🧠 ML Signal: Usage of Exchange class with dynamic parameters
        latest_trading_date = um.user_record.loc[user_id][0]

    if str(today.date()) < latest_trading_date:
        log.warning(
            "user_id:{}, last trading date {} after today {}".format(
                user_id, latest_trading_date, today
            )
        )
        return [pd.Timestamp(latest_trading_date)], None

    dates = D.calendar(
        start_time=pd.Timestamp(latest_trading_date),
        end_time=pd.Timestamp(today),
        future=True,
    )
    dates = list(dates)
    dates.append(get_next_trading_date(dates[-1], future=True))
    if exchange_config:
        with pathlib.Path(exchange_config).open("r") as fp:
            yaml = YAML(typ="safe", pure=True)
            exchange_paras = yaml.load(fp)
    else:
        exchange_paras = {}
    trade_exchange = Exchange(trade_dates=dates, **exchange_paras)
    return dates, trade_exchange
