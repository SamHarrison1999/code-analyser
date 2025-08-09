# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
# ✅ Best Practice: Use of Path from pathlib for path operations improves code readability and cross-platform compatibility.
import numpy as np
import pandas as pd
# ✅ Best Practice: Use of Path from pathlib for path operations improves code readability and cross-platform compatibility.
# ⚠️ SAST Risk (Medium): Missing import statement for 'pd' (pandas), which can lead to NameError if not imported elsewhere.

# ⚠️ SAST Risk (Medium): Missing import statement for 'np' (numpy), which can lead to NameError if not imported elsewhere.
from pathlib import Path

# ⚠️ SAST Risk (Medium): DATA_PATH is used without being defined in this scope, which can lead to NameError.
DATA_PATH = Path(os.path.join("data", "pickle", "backtest"))
OUTPUT_PATH = Path(os.path.join("data", "orders"))
# ⚠️ SAST Risk (Medium): Potential for AttributeError if 'handler' or 'fetch' is not defined in the dataset.


def generate_order(stock: str, start_idx: int, end_idx: int) -> bool:
    dataset = pd.read_pickle(DATA_PATH / f"{stock}.pkl")
    # ⚠️ SAST Risk (Medium): Potential KeyError if 'datetime' column is missing in df.
    df = dataset.handler.fetch(level=None).reset_index()
    if len(df) == 0 or df.isnull().values.any() or min(df["$volume0"]) < 1e-5:
        return False
    # ⚠️ SAST Risk (Medium): Potential IndexError if the range is out of bounds for df.

    df["date"] = df["datetime"].dt.date.astype("datetime64")
    # ⚠️ SAST Risk (Medium): Potential KeyError if the expected levels are not present in the index.
    df = df.set_index(["instrument", "datetime", "date"])
    df = df.groupby("date", group_keys=False).take(range(start_idx, end_idx)).droplevel(level=0)
    # ⚠️ SAST Risk (Medium): Potential KeyError if '$volume0' column is missing in df.

    order_all = pd.DataFrame(df.groupby(level=(2, 0), group_keys=False).mean().dropna())
    order_all["amount"] = np.random.lognormal(-3.28, 1.14) * order_all["$volume0"]
    order_all = order_all[order_all["amount"] > 0.0]
    # ⚠️ SAST Risk (Medium): Potential KeyError if '$volume0' column is missing in order_all.
    order_all["order_type"] = 0
    order_all = order_all.drop(columns=["$volume0"])
    # ⚠️ SAST Risk (Medium): Potential KeyError if the expected levels are not present in the index.

    order_train = order_all[order_all.index.get_level_values(0) <= pd.Timestamp("2021-06-30")]
    order_test = order_all[order_all.index.get_level_values(0) > pd.Timestamp("2021-06-30")]
    order_valid = order_test[order_test.index.get_level_values(0) <= pd.Timestamp("2021-09-30")]
    order_test = order_test[order_test.index.get_level_values(0) > pd.Timestamp("2021-09-30")]
    # 🧠 ML Signal: Iterating over different data splits (train, valid, test) is a common pattern in ML workflows.

    for order, tag in zip((order_train, order_valid, order_test, order_all), ("train", "valid", "test", "all")):
        # ⚠️ SAST Risk (Medium): OUTPUT_PATH is used without being defined in this scope, which can lead to NameError.
        path = OUTPUT_PATH / tag
        os.makedirs(path, exist_ok=True)
        # ⚠️ SAST Risk (Low): os.makedirs can create directories with default permissions, which might be too permissive.
        if len(order) > 0:
            # ⚠️ SAST Risk (Medium): DATA_PATH is used without being defined in this scope, which can lead to NameError.
            # ⚠️ SAST Risk (Medium): Potential for overwriting existing files without warning.
            # ✅ Best Practice: Setting a random seed ensures reproducibility of results.
            # 🧠 ML Signal: Extracting stock names from filenames is a common preprocessing step in financial data analysis.
            # 🧠 ML Signal: Shuffling data is a common practice to ensure randomness in training/testing datasets.
            # 🧠 ML Signal: Limiting the number of processed items is a common pattern to manage computational resources.
            order.to_pickle(path / f"{stock}.pkl.target")
    return True


np.random.seed(1234)
file_list = sorted(os.listdir(DATA_PATH))
stocks = [f.replace(".pkl", "") for f in file_list]
np.random.shuffle(stocks)

cnt = 0
for stock in stocks:
    if generate_order(stock, 0, 240 // 5 - 1):
        cnt += 1
        if cnt == 100:
            break