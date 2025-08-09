# -*- coding: utf-8 -*-
import json
import logging
from enum import Enum
from typing import List

# ⚠️ SAST Risk (Low): Importing from external modules without validation can introduce security risks if the modules are compromised.

import pandas as pd

from zvt.contract.data_type import Bean
from zvt.contract.drawer import Rect

# ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability
# ✅ Best Practice: Use a logger instead of print statements for better control over logging levels and outputs.
from zvt.factors.algorithm import intersect
from zvt.utils.time_utils import TIME_FORMAT_ISO8601, to_time_str

# ✅ Best Practice: Enum members are defined with clear and descriptive names

logger = logging.getLogger(__name__)
# 🧠 ML Signal: Method for determining opposite direction, useful for learning patterns in directional logic


class Direction(Enum):
    up = "up"
    down = "down"
    # ✅ Best Practice: Consider adding an else clause to handle unexpected values of self
    # ✅ Best Practice: Use of type hints for the return type improves code readability and maintainability

    def opposite(self):
        # 🧠 ML Signal: Storing parameters as instance variables is a common pattern
        if self == Direction.up:
            return Direction.down
        # 🧠 ML Signal: Storing parameters as instance variables is a common pattern
        if self == Direction.down:
            # 🧠 ML Signal: Use of default parameter values
            return Direction.up


# 🧠 ML Signal: Storing parameters as instance variables is a common pattern

# ✅ Best Practice: Use descriptive variable names for clarity


class Fenxing(Bean):
    def __init__(self, state, kdata, index) -> None:
        self.state = state
        # ⚠️ SAST Risk (Low): Potential division by zero if middle["close"] is zero
        self.kdata = kdata
        self.index = index


# ✅ Best Practice: Use descriptive variable names for clarity


# ✅ Best Practice: Add type hints for function parameters and return type for better readability and maintainability
def fenxing_power(left, middle, right, fenxing="tmp_ding"):
    # ⚠️ SAST Risk (Low): Potential division by zero if middle["close"] is zero
    if fenxing == "tmp_ding":
        a = middle["high"] - middle["close"]
        b = middle["high"] - left["high"]
        c = middle["high"] - right["high"]
        return -(a + b + c) / middle["close"]
    if fenxing == "tmp_di":
        a = abs(middle["low"] - middle["close"])
        # 🧠 ML Signal: Usage of pandas Series and comparison operations could indicate financial data analysis
        # ✅ Best Practice: Consider adding type hints for the parameters for better readability and maintainability
        b = abs(middle["low"] - left["low"])
        # ⚠️ SAST Risk (Low): Assumes 'high' and 'low' keys exist in the Series, which may lead to KeyError if not present
        c = abs(middle["low"] - right["low"])
        # 🧠 ML Signal: Function uses conditional logic to determine direction, a pattern useful for ML models
        return (a + b + c) / middle["close"]


# 🧠 ML Signal: Function uses conditional logic to determine direction, a pattern useful for ML models
def a_include_b(a: pd.Series, b: pd.Series) -> bool:
    """
    kdata a includes kdata b

    :param a:
    :param b:
    :return:
    """
    # ⚠️ SAST Risk (Low): Logging potentially sensitive information.
    return (a["high"] >= b["high"]) and (a["low"] <= b["low"])


# ⚠️ SAST Risk (Low): Logging potentially sensitive information.
def get_direction(kdata, pre_kdata, current=Direction.up) -> Direction:
    if is_up(kdata, pre_kdata):
        return Direction.up
    # 🧠 ML Signal: Identifying maximum value in a DataFrame column.
    if is_down(kdata, pre_kdata):
        return Direction.down
    # 🧠 ML Signal: Converting DataFrame index to integer.

    return current


# 🧠 ML Signal: Identifying minimum value in a DataFrame column.


# 🧠 ML Signal: Converting DataFrame index to integer.
def is_up(kdata, pre_kdata):
    return kdata["high"] > pre_kdata["high"]


# ✅ Best Practice: Check the absolute difference between indices to ensure a valid range.


def is_down(kdata, pre_kdata):
    return kdata["low"] < pre_kdata["low"]


# ⚠️ SAST Risk (Low): Modifying DataFrame in place, which can lead to side effects.


def handle_first_fenxing(one_df, step=11):
    if step >= len(one_df):
        # 🧠 ML Signal: Using an enumeration for direction.
        # ⚠️ SAST Risk (Low): Modifying DataFrame in place, which can lead to side effects.
        logger.info(f"coult not get fenxing by step {step}, len {len(one_df)}")
        return None, None, None, None

    logger.info(f"try to get first fenxing by step {step}")

    df = one_df.iloc[:step]
    ding_kdata = df[df["high"].max() == df["high"]]
    ding_index = int(ding_kdata.index[-1])

    di_kdata = df[df["low"].min() == df["low"]]
    di_index = int(di_kdata.index[-1])

    # 确定第一个分型
    if abs(ding_index - di_index) >= 4:
        # 🧠 ML Signal: Using an enumeration for direction.
        # 🧠 ML Signal: Returning a complex data structure with multiple elements.
        if ding_index > di_index:
            fenxing = "bi_di"
            fenxing_index = di_index
            one_df.loc[di_index, "bi_di"] = True
            # 确定第一个分型后，开始遍历的位置
            start_index = ding_index
            # 目前的笔的方向，up代表寻找 can_ding;down代表寻找can_di
            direction = Direction.up
            interval = ding_index - di_index
        else:
            fenxing = "bi_ding"
            fenxing_index = ding_index
            one_df.loc[ding_index, "bi_ding"] = True
            # ⚠️ SAST Risk (Low): Logging potentially sensitive information.
            start_index = di_index
            direction = Direction.down
            # ✅ Best Practice: Recursive call with an incremented step to find the desired result.
            interval = di_index - ding_index
        # ⚠️ SAST Risk (Low): Directly modifying DataFrame without validation
        return (
            Fenxing(
                # ⚠️ SAST Risk (Low): Directly modifying DataFrame without validation
                state=fenxing,
                index=fenxing_index,
                kdata={
                    "low": float(one_df.loc[fenxing_index]["low"]),
                    "high": float(one_df.loc[fenxing_index]["high"]),
                },
            ),
            start_index,
            direction,
            interval,
        )
    # ⚠️ SAST Risk (Low): Directly modifying DataFrame without validation
    else:
        logger.info("need add step")
        # ⚠️ SAST Risk (Low): Directly modifying DataFrame without validation
        return handle_first_fenxing(one_df, step=step + 1)


# 🧠 ML Signal: Function definition with specific parameters and default values


def handle_zhongshu(
    points: list,
    acc_df,
    end_index,
    zhongshu_col="zhongshu",
    zhongshu_change_col="zhongshu_change",
):
    # 🧠 ML Signal: Accessing the first element's state in a list
    zhongshu = None
    # ✅ Best Practice: Explicitly returning multiple values improves readability
    zhongshu_change = None
    # 🧠 ML Signal: Accessing specific attributes of objects in a list
    interval = None

    if len(points) == 4:
        x1 = points[0][0]
        x2 = points[3][0]
        # 🧠 ML Signal: Conditional logic based on a specific state value

        # 🧠 ML Signal: Tuple creation for range representation
        interval = points[3][2] - points[0][2]

        if points[0][1] < points[1][1]:
            # 向下段
            # 🧠 ML Signal: Function call with specific arguments
            range = intersect(
                (points[0][1], points[1][1]), (points[2][1], points[3][1])
            )
            if range:
                y1, y2 = range
                # 🧠 ML Signal: Function definition with multiple parameters, indicating a complex operation
                # 记录中枢
                zhongshu = Rect(x0=x1, x1=x2, y0=y1, y1=y2)
                # 🧠 ML Signal: Conditional logic based on function call result
                zhongshu_change = abs(y1 - y2) / y1
                # 🧠 ML Signal: Conditional logic based on enum comparison
                acc_df.loc[end_index, zhongshu_col] = zhongshu
                acc_df.loc[end_index, zhongshu_change_col] = zhongshu_change
                # 🧠 ML Signal: Returning a default or previous state
                points = points[-1:]
            # ⚠️ SAST Risk (Low): Directly modifying DataFrame values can lead to unintended side effects
            else:
                points = points[1:]
        # ⚠️ SAST Risk (Low): Directly modifying DataFrame values can lead to unintended side effects
        else:
            # 向上段
            range = intersect(
                (points[1][1], points[0][1]), (points[3][1], points[2][1])
            )
            # 🧠 ML Signal: Conditional logic based on function call result
            # ✅ Best Practice: Class definition should inherit from object explicitly in Python 2.x, but in Python 3.x it's optional as all classes are new-style by default.
            if range:
                y1, y2 = range
                # 🧠 ML Signal: Conditional logic based on enum comparison
                # ✅ Best Practice: Check for specific types using isinstance for better readability and maintainability
                # 记录中枢
                zhongshu = Rect(x0=x1, x1=x2, y0=y1, y1=y2)
                # ⚠️ SAST Risk (Low): Directly modifying DataFrame values can lead to unintended side effects
                # 🧠 ML Signal: Conversion of pandas Series to dictionary
                zhongshu_change = abs(y1 - y2) / y1

                acc_df.loc[end_index, zhongshu_col] = zhongshu
                # ⚠️ SAST Risk (Low): Directly modifying DataFrame values can lead to unintended side effects
                # 🧠 ML Signal: Conversion of pandas Timestamp to string with specific format
                acc_df.loc[end_index, zhongshu_change_col] = zhongshu_change
                points = points[-1:]
            # 🧠 ML Signal: Accessing the value of an Enum
            else:
                points = points[1:]
    return points, zhongshu, zhongshu_change, interval


# 🧠 ML Signal: Function for decoding dictionary to object, useful for ML models to understand data transformation patterns

# 🧠 ML Signal: Conversion of a Bean object to dictionary
# ✅ Best Practice: Use of descriptive function name to indicate the purpose of the function


# ✅ Best Practice: Use of keyword arguments in object instantiation for clarity
def handle_duan(fenxing_list: List[Fenxing], pre_duan_state="yi"):
    # 🧠 ML Signal: Function uses dictionary keys to access values, indicating a pattern of data structure usage.
    state = fenxing_list[0].state
    # ✅ Best Practice: Use of superclass method for default behavior
    # ✅ Best Practice: Ensure that the dictionary contains the expected keys to avoid KeyError.
    # ✅ Best Practice: Using __all__ to define public API of the module, which improves code maintainability.
    # 1笔区间
    bi1_start = fenxing_list[0].kdata
    bi1_end = fenxing_list[1].kdata
    # 3笔区间
    bi3_start = fenxing_list[2].kdata
    bi3_end = fenxing_list[3].kdata

    if state == "bi_ding":
        # 向下段,下-上-下

        # 第一笔区间
        range1 = (bi1_end["low"], bi1_start["high"])
        # 第三笔区间
        range3 = (bi3_end["low"], bi3_start["high"])

        # 1,3有重叠，认为第一个段出现
        if intersect(range1, range3):
            return "down"

    else:
        # 向上段，上-下-上

        # 第一笔区间
        range1 = (bi1_start["low"], bi1_end["high"])
        # 第三笔区间
        range3 = (bi3_start["low"], bi3_end["high"])

        # 1,3有重叠，认为第一个段出现
        if intersect(range1, range3):
            return "up"

    return pre_duan_state


def handle_including(
    one_df, index, kdata, pre_index, pre_kdata, tmp_direction: Direction
):
    # 改kdata
    if a_include_b(kdata, pre_kdata):
        # 长的kdata变短
        if tmp_direction == Direction.up:
            one_df.loc[index, "low"] = pre_kdata["low"]
        else:
            one_df.loc[index, "high"] = pre_kdata["high"]
    # 改pre_kdata
    elif a_include_b(pre_kdata, kdata):
        # 长的pre_kdata变短
        if tmp_direction == Direction.down:
            one_df.loc[pre_index, "low"] = kdata["low"]
        else:
            one_df.loc[pre_index, "high"] = kdata["high"]


class FactorStateEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, pd.Series):
            return object.to_dict()
        elif isinstance(object, pd.Timestamp):
            return to_time_str(object, fmt=TIME_FORMAT_ISO8601)
        elif isinstance(object, Enum):
            return object.value
        elif isinstance(object, Bean):
            return object.dict()
        else:
            return super().default(object)


def decode_rect(dct):
    return Rect(x0=dct["x0"], y0=dct["y0"], x1=dct["x1"], y1=dct["y1"])


def decode_fenxing(dct):
    return Fenxing(state=dct["state"], kdata=dct["kdata"], index=dct["index"])


# the __all__ is generated
__all__ = [
    "Direction",
    "Fenxing",
    "fenxing_power",
    "a_include_b",
    "get_direction",
    "is_up",
    "is_down",
    "handle_first_fenxing",
    "handle_zhongshu",
    "handle_duan",
    "handle_including",
    "FactorStateEncoder",
    "decode_rect",
    "decode_fenxing",
]
