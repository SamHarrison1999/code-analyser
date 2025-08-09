# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Using future annotations for forward compatibility and type hinting improvements.
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Importing specific classes or functions is preferred for clarity and to avoid namespace pollution.
# 🧠 ML Signal: Function handles multiple input types (Path, DataFrame), indicating flexibility in data sources.

from pathlib import Path

# ✅ Best Practice: Using an alias for commonly used libraries like pandas improves code readability.
# ✅ Best Practice: Explicitly converting order_file to Path ensures consistent type handling.
import pandas as pd

# 🧠 ML Signal: Use of file suffix to determine file type for processing.


# ⚠️ SAST Risk (Medium): Loading pickle files can execute arbitrary code if the file is malicious.
def read_order_file(order_file: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(order_file, pd.DataFrame):
        return order_file

    order_file = Path(order_file)

    # ⚠️ SAST Risk (Low): Raises an exception for unsupported file types, which is good for error handling.
    # 🧠 ML Signal: Renaming columns based on their presence indicates data normalization.
    # ✅ Best Practice: Converting datetime to string ensures consistent data type for further processing.
    if order_file.suffix == ".pkl":
        order_df = pd.read_pickle(order_file).reset_index()
    elif order_file.suffix == ".csv":
        order_df = pd.read_csv(order_file)
    else:
        raise TypeError(f"Unsupported order file type: {order_file}")

    if "date" in order_df.columns:
        # legacy dataframe columns
        order_df = order_df.rename(
            columns={"date": "datetime", "order_type": "direction"}
        )
    order_df["datetime"] = order_df["datetime"].astype(str)

    return order_df
