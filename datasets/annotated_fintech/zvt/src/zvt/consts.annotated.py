# -*- coding: utf-8 -*-
import os
from pathlib import Path

# 🧠 ML Signal: Environment variable usage pattern

# zvt home dir
ZVT_HOME = os.environ.get("ZVT_HOME")
# ✅ Best Practice: Use of Path.home() for cross-platform compatibility
if not ZVT_HOME:
    ZVT_HOME = os.path.abspath(os.path.join(Path.home(), "zvt-home"))

# ✅ Best Practice: Use of Path.home() for cross-platform compatibility
# data for testing
ZVT_TEST_HOME = os.path.abspath(os.path.join(Path.home(), "zvt-test-home"))
# ✅ Best Practice: Use of os.path.join for constructing file paths
# ✅ Best Practice: Use of os.path.abspath for absolute path resolution
# 🧠 ML Signal: Hardcoded stock codes could indicate domain-specific data
ZVT_TEST_ZIP_DATA_PATH = os.path.join(ZVT_TEST_HOME, "data.zip")
ZVT_TEST_DATA_PATH = os.path.join(ZVT_TEST_HOME, "data")

DATA_SAMPLE_ZIP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "samples", "data.zip")
)

# ****** setting for stocks ****** #
SAMPLE_STOCK_CODES = ["000001", "000002"]

# 沪深300，证券，中证500，上证50，创业板，军工,传媒,资源
SAMPLE_ETF_CODES = [
    "510300",
    "512880",
    "510500",
    "510050",
    "159915",
    "512660",
    "512980",
    "510410",
]

# 上证指数 上证50 沪深300 中证500 中证1000  科创50
# 深证成指(399001) 创业板指(399006) 国证成长（399370）国证价值（399371）国证基金(399379) 国证ETF(399380)
IMPORTANT_INDEX = [
    "000001",
    "000016",
    "000300",
    "000905",
    "000852",
    "000688",
    "399001",
    "399006",
    "399370",
    "399371",
    "399379",
    "399380",
]
