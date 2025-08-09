# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# REGION CONST
# ✅ Best Practice: Use of constants for region codes improves readability and maintainability
from typing import TypeVar

import numpy as np
import pandas as pd

# ✅ Best Practice: Use of constants for small values like EPS improves readability and maintainability
REG_CN = "cn"
REG_US = "us"
# ✅ Best Practice: Use of constants for large values like INF improves readability and maintainability
# ✅ Best Practice: Use of TypeVar for type hinting improves code readability and type safety
REG_TW = "tw"

# Epsilon for avoiding division by zero.
EPS = 1e-12

# Infinity in integer
INF = int(1e18)
ONE_DAY = pd.Timedelta("1day")
ONE_MIN = pd.Timedelta("1min")
EPS_T = pd.Timedelta("1s")  # use 1 second to exclude the right interval point
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)