# -*- coding: utf-8 -*-
import json
import logging
from enum import Enum
from typing import List
from typing import Union, Optional, Type

# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
import numpy as np
import pandas as pd

# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract import TradableEntity
from zvt.contract.api import get_schema_by_name
from zvt.contract.data_type import Bean
# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
from zvt.contract.drawer import Rect
from zvt.contract.factor import Accumulator
# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
from zvt.contract.factor import Transformer
from zvt.domain import Stock, Index, Index1dKdata
from zvt.factors.algorithm import intersect, combine
from zvt.factors.shape import (
    Fenxing,
    Direction,
    handle_first_fenxing,
    decode_rect,
    get_direction,
    handle_including,
    fenxing_power,
    handle_duan,
)
from zvt.factors.technical_factor import TechnicalFactor
from zvt.utils.decorator import to_string
# ✅ Best Practice: Inheriting from json.JSONEncoder to create a custom encoder
from zvt.utils.pd_utils import pd_is_not_null
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
from zvt.utils.time_utils import TIME_FORMAT_ISO8601, to_time_str

# ✅ Best Practice: Grouping imports from the same module in a single line improves readability.
# ✅ Best Practice: Using isinstance to check the type of an object is a common and clear pattern.
logger = logging.getLogger(__name__)

# ✅ Best Practice: Using a logger is a good practice for handling log messages.
# 🧠 ML Signal: Converting a pandas Series to a dictionary is a common pattern.

class FactorStateEncoder(json.JSONEncoder):
    def default(self, object):
        # 🧠 ML Signal: Converting a pandas Timestamp to a string with a specific format is a common pattern.
        if isinstance(object, pd.Series):
            return object.to_dict()
        elif isinstance(object, pd.Timestamp):
            # 🧠 ML Signal: Accessing the value of an Enum is a common pattern.
            return to_time_str(object, fmt=TIME_FORMAT_ISO8601)
        elif isinstance(object, Enum):
            # ✅ Best Practice: Use isinstance() instead of type() for type checking
            return object.value
        # 🧠 ML Signal: Converting a custom object to a dictionary is a common pattern.
        elif isinstance(object, Bean):
            return object.dict()
        # ✅ Best Practice: Using super() to call a method from the parent class is a common and clear pattern.
        # 🧠 ML Signal: Usage of string formatting to create dynamic schema names
        else:
            return super().default(object)
# 🧠 ML Signal: Function call pattern to retrieve schema by name

# 🧠 ML Signal: Inheritance from a class named 'Bean' suggests a pattern of using a specific framework or library

# ⚠️ SAST Risk (Low): Potential misuse of decorators if not properly defined
# ✅ Best Practice: Use of default mutable arguments (like dict) can lead to unexpected behavior. Consider using None and initializing inside the function.
def get_zen_factor_schema(entity_type: str, level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY):
    if type(level) == str:
        level = IntervalLevel(level)

    # 🧠 ML Signal: List comprehension used to transform data, indicating a pattern of data processing.
    # z factor schema rule
    # 1)name:{SecurityType.value.capitalize()}{IntervalLevel.value.upper()}ZFactor
    schema_str = "{}{}ZenFactor".format(entity_type.capitalize(), level.value.capitalize())
    # 🧠 ML Signal: Conditional logic based on dictionary keys, indicating a pattern of optional configuration.

    return get_schema_by_name(schema_str)


@to_string
class ZenState(Bean):
    def __init__(self, state: dict = None) -> None:
        super().__init__()

        if not state:
            state = dict()

        # 用于计算未完成段的分型
        self.fenxing_list = state.get("fenxing_list", [])
        fenxing_list = [Fenxing(item["state"], item["kdata"], item["index"]) for item in self.fenxing_list]
        self.fenxing_list = fenxing_list

        # 目前的方向
        if state.get("direction"):
            self.direction = Direction(state.get("direction"))
        else:
            # ✅ Best Practice: Initialize variables to None for clarity and to avoid potential reference before assignment errors.
            self.direction = None

        # 候选分型(candidate)
        self.can_fenxing = state.get("can_fenxing")
        # 🧠 ML Signal: Checking the length of a list to determine logic flow is a common pattern.
        self.can_fenxing_index = state.get("can_fenxing_index")
        # 反方向count
        self.opposite_count = state.get("opposite_count", 0)
        # 目前段的方向
        self.current_duan_state = state.get("current_duan_state", "yi")
        # 🧠 ML Signal: Conditional logic based on list element comparison is a common pattern.

        # 记录用于计算中枢的段
        # ⚠️ SAST Risk (Low): Potential risk if intersect function is not properly validated.
        # list of (timestamp,value)
        self.duans = state.get("duans", [])
        self.bis = state.get("bis", [])

        # ✅ Best Practice: Use of named arguments in object creation improves readability.
        # 前一个点
        self.pre_bi = state.get("pre_bi")
        # ⚠️ SAST Risk (Low): Division by zero risk if y1 is zero.
        self.pre_duan = state.get("pre_duan")

        # ⚠️ SAST Risk (Low): Direct assignment to DataFrame without validation can lead to data integrity issues.
        # 目前的merge_zhongshu
        self.merge_zhongshu = state.get("merge_zhongshu")
        self.merge_zhongshu_level = state.get("merge_zhongshu_level")
        self.merge_zhongshu_interval = state.get("merge_zhongshu_interval")


def handle_zhongshu(
    # ⚠️ SAST Risk (Low): Potential risk if intersect function is not properly validated.
    points: list,
    acc_df,
    end_index,
    zhongshu_col="zhongshu",
    # ✅ Best Practice: Use of named arguments in object creation improves readability.
    # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which this does.
    zhongshu_change_col="zhongshu_change",
):
    # ⚠️ SAST Risk (Low): Division by zero risk if y1 is zero.
    # ✅ Best Practice: Docstring provides detailed explanation of the algorithm and concepts used.
    # ⚠️ SAST Risk (Low): Direct assignment to DataFrame without validation can lead to data integrity issues.
    zhongshu = None
    zhongshu_change = None
    interval = None

    if len(points) == 4:
        x1 = points[0][0]
        x2 = points[3][0]

        interval = points[3][2] - points[0][2]

        if points[0][1] < points[1][1]:
            # 向下段
            range = intersect((points[0][1], points[1][1]), (points[2][1], points[3][1]))
            if range:
                y1, y2 = range
                # 记录中枢
                zhongshu = Rect(x0=x1, x1=x2, y0=y1, y1=y2)
                zhongshu_change = abs(y1 - y2) / abs(y1)
                acc_df.loc[end_index, zhongshu_col] = zhongshu
                acc_df.loc[end_index, zhongshu_change_col] = zhongshu_change
                points = points[-1:]
            else:
                points = points[1:]
        else:
            # 向上段
            # ✅ Best Practice: Calls the superclass's __init__ method to ensure proper initialization.
            range = intersect((points[1][1], points[0][1]), (points[3][1], points[2][1]))
            # 🧠 ML Signal: Logging the start of processing for a specific entity
            if range:
                y1, y2 = range
                # 记录中枢
                zhongshu = Rect(x0=x1, x1=x2, y0=y1, y1=y2)
                zhongshu_change = abs(y1 - y2) / abs(y1)
                # 🧠 ML Signal: Logging the timestamp from which computation starts

                acc_df.loc[end_index, zhongshu_col] = zhongshu
                acc_df.loc[end_index, zhongshu_change_col] = zhongshu_change
                # ✅ Best Practice: Using pd.concat to append dataframes
                points = points[-1:]
            else:
                points = points[1:]
    # ✅ Best Practice: Resetting index for a clean DataFrame
    return points, zhongshu, zhongshu_change, interval


class ZenAccumulator(Accumulator):
    # 🧠 ML Signal: Logging when no computation is needed
    def __init__(self, acc_window: int = 1) -> None:
        """
        算法和概念
        <实体> 某种状态的k线
        [实体] 连续实体排列

        两k线的关系有三种: 上涨，下跌，包含
        上涨: k线高点比之前高，低点比之前高
        下跌: k线低点比之前低，高点比之前低
        包含: k线高点比之前高，低点比之前低;反方向，即被包含
        处理包含关系，长的k线缩短，上涨时，低点取max(low1,low2)；下跌时，高点取min(high1,high2)

        第一个顶(底)分型: 出现连续4根下跌(上涨)k线
        之后开始寻找 候选底(顶)分型，寻找的过程中有以下状态

        <临时顶>: 中间k线比两边的高点高,是一条特定的k线
        <临时底>: 中间k线比两边的高点高,是一条特定的k线

        <候选顶分型>: 连续的<临时顶>取最大
        <候选底分型>:  连续的<临时底>取最小
        任何时刻只能有一个候选，其之前是一个确定的分型

        <上升k线>:
        <下降k线>:
        <连接k线>: 分型之间的k线都可以认为是连接k线，以上为演化过程的中间态
        distance(<候选顶分型>, <连接k线>)>=4 则 <候选顶分型> 变成顶分型
        distance(<候选底分型>, <连接k线>)>=4 则 <候选底分型> 变成底分型

        <顶分型><连接k线><候选底分型>
        <底分型><连接k线><候选顶分型>
        """
        super().__init__(acc_window)

    def acc_one(self, entity_id, df: pd.DataFrame, acc_df: pd.DataFrame, state: dict) -> (pd.DataFrame, dict):
        self.logger.info(f"acc_one:{entity_id}")
        if pd_is_not_null(acc_df):
            df = df[df.index > acc_df.index[-1]]
            if pd_is_not_null(df):
                self.logger.info(f'compute from {df.iloc[0]["timestamp"]}')
                # 遍历的开始位置
                start_index = len(acc_df)
                # ✅ Best Practice: Resetting index for a clean DataFrame

                acc_df = pd.concat([acc_df, df])

                zen_state = ZenState(state)

                acc_df = acc_df.reset_index(drop=True)
                current_interval = acc_df.iloc[start_index - 1]["current_interval"]
            else:
                self.logger.info("no need to compute")
                return acc_df, state
        else:
            acc_df = df
            # 笔的底
            acc_df["bi_di"] = False
            # 笔的顶
            acc_df["bi_ding"] = False
            # 记录笔顶/底分型的值，bi_di取low,bi_ding取high,其他为None,绘图时取有值的连线即为 笔
            acc_df["bi_value"] = np.nan
            # 笔的变化
            acc_df["bi_change"] = np.nan
            # 笔的斜率
            acc_df["bi_slope"] = np.nan
            # 持续的周期
            acc_df["bi_interval"] = np.nan

            # 记录临时分型，不变
            acc_df["tmp_ding"] = False
            acc_df["tmp_di"] = False
            # 分型的力度
            acc_df["fenxing_power"] = np.nan

            # 目前分型确定的方向
            acc_df["current_direction"] = None
            acc_df["current_change"] = np.nan
            acc_df["current_interval"] = np.nan
            acc_df["current_slope"] = np.nan
            # 最近的一个笔中枢
            # acc_df['current_zhongshu'] = np.nan
            acc_df["current_zhongshu_change"] = np.nan
            acc_df["current_zhongshu_y0"] = np.nan
            acc_df["current_zhongshu_y1"] = np.nan

            acc_df["current_merge_zhongshu_change"] = np.nan
            acc_df["current_merge_zhongshu_y0"] = np.nan
            acc_df["current_merge_zhongshu_y1"] = np.nan
            acc_df["current_merge_zhongshu_level"] = np.nan
            acc_df["current_merge_zhongshu_interval"] = np.nan

            # 目前走势的临时方向 其跟direction的的关系 确定了下一个分型
            acc_df["tmp_direction"] = None
            acc_df["opposite_change"] = np.nan
            acc_df["opposite_interval"] = np.nan
            acc_df["opposite_slope"] = np.nan

            acc_df["duan_state"] = "yi"

            # 段的底
            acc_df["duan_di"] = False
            # 段的顶
            acc_df["duan_ding"] = False
            # 记录段顶/底的值，为duan_di时取low,为duan_ding时取high,其他为None,绘图时取有值的连线即为 段
            acc_df["duan_value"] = np.nan
            # 段的变化
            acc_df["duan_change"] = np.nan
            # 段的斜率
            acc_df["duan_slope"] = np.nan
            # 持续的周期
            acc_df["duan_interval"] = np.nan

            # 记录在确定中枢的最后一个段的终点x1，值为Rect(x0,y0,x1,y1)
            acc_df["zhongshu"] = None
            acc_df["zhongshu_change"] = np.nan

            acc_df["bi_zhongshu"] = None
            acc_df["bi_zhongshu_change"] = np.nan

            acc_df["merge_zhongshu"] = None
            acc_df["merge_zhongshu_change"] = np.nan
            acc_df["merge_zhongshu_level"] = np.nan
            acc_df["merge_zhongshu_interval"] = np.nan

            acc_df = acc_df.reset_index(drop=True)

            zen_state = ZenState(
                dict(
                    fenxing_list=[],
                    direction=None,
                    can_fenxing=None,
                    can_fenxing_index=None,
                    opposite_count=0,
                    current_duan_state="yi",
                    duans=[],
                    pre_bi=None,
                    pre_duan=None,
                    merge_zhongshu=None,
                )
            )

            zen_state.fenxing_list: List[Fenxing] = []

            # 取前11条k线，至多出现一个顶分型+底分型
            # 注:只是一种方便的确定第一个分型的办法，有了第一个分型，后面的处理就比较统一
            # start_index 为遍历开始的位置
            # direction为一个确定分型后的方向，即顶分型后为:down，底分型后为:up
            fenxing, start_index, direction, current_interval = handle_first_fenxing(acc_df, step=11)
            if not fenxing:
                return None, None

            zen_state.fenxing_list.append(fenxing)
            zen_state.direction = direction

            # list of (timestamp,value)
            zen_state.duans = []
            zen_state.bis = []

        pre_kdata = acc_df.iloc[start_index - 1]
        pre_index = start_index - 1

        tmp_direction = zen_state.direction
        current_merge_zhongshu = decode_rect(zen_state.merge_zhongshu) if zen_state.merge_zhongshu else None
        current_merge_zhongshu_change = None
        current_merge_zhongshu_interval = zen_state.merge_zhongshu_interval
        current_merge_zhongshu_level = zen_state.merge_zhongshu_level

        current_zhongshu = None
        current_zhongshu_change = None
        for index, kdata in acc_df.iloc[start_index:].iterrows():
            # print(f'timestamp: {kdata.timestamp}')
            # 临时方向
            tmp_direction = get_direction(kdata, pre_kdata, current=tmp_direction)

            # current states
            current_interval = current_interval + 1
            if zen_state.direction == Direction.up:
                pre_value = acc_df.loc[zen_state.fenxing_list[0].index, "low"]
                current_value = kdata["high"]
            else:
                pre_value = acc_df.loc[zen_state.fenxing_list[0].index, "high"]
                current_value = kdata["low"]
            acc_df.loc[index, "current_direction"] = zen_state.direction.value
            acc_df.loc[index, "current_interval"] = current_interval
            change = (current_value - pre_value) / abs(pre_value)
            acc_df.loc[index, "current_change"] = change
            acc_df.loc[index, "current_slope"] = change / current_interval
            if current_zhongshu:
                # acc_df.loc[index, 'current_zhongshu'] = current_zhongshu
                acc_df.loc[index, "current_zhongshu_y0"] = current_zhongshu.y0
                acc_df.loc[index, "current_zhongshu_y1"] = current_zhongshu.y1
                acc_df.loc[index, "current_zhongshu_change"] = current_zhongshu_change
            else:
                # acc_df.loc[index, 'current_zhongshu'] = acc_df.loc[index - 1, 'current_zhongshu']
                acc_df.loc[index, "current_zhongshu_y0"] = acc_df.loc[index - 1, "current_zhongshu_y0"]
                acc_df.loc[index, "current_zhongshu_y1"] = acc_df.loc[index - 1, "current_zhongshu_y1"]
                acc_df.loc[index, "current_zhongshu_change"] = acc_df.loc[index - 1, "current_zhongshu_change"]

            if current_merge_zhongshu:
                # acc_df.loc[index, 'current_merge_zhongshu'] = current_merge_zhongshu
                acc_df.loc[index, "current_merge_zhongshu_y0"] = current_merge_zhongshu.y0
                acc_df.loc[index, "current_merge_zhongshu_y1"] = current_merge_zhongshu.y1
                acc_df.loc[index, "current_merge_zhongshu_change"] = current_merge_zhongshu_change
                acc_df.loc[index, "current_merge_zhongshu_level"] = current_merge_zhongshu_level
                acc_df.loc[index, "current_merge_zhongshu_interval"] = current_merge_zhongshu_interval
            else:
                # acc_df.loc[index, 'current_merge_zhongshu'] = acc_df.loc[index - 1, 'current_merge_zhongshu']
                acc_df.loc[index, "current_merge_zhongshu_y0"] = acc_df.loc[index - 1, "current_merge_zhongshu_y0"]
                acc_df.loc[index, "current_merge_zhongshu_y1"] = acc_df.loc[index - 1, "current_merge_zhongshu_y1"]
                acc_df.loc[index, "current_merge_zhongshu_change"] = acc_df.loc[
                    index - 1, "current_merge_zhongshu_change"
                ]
                acc_df.loc[index, "current_merge_zhongshu_level"] = acc_df.loc[
                    index - 1, "current_merge_zhongshu_level"
                ]
                acc_df.loc[index, "current_merge_zhongshu_interval"] = acc_df.loc[
                    index - 1, "current_merge_zhongshu_interval"
                ]

            # 处理包含关系
            handle_including(
                one_df=acc_df,
                index=index,
                kdata=kdata,
                pre_index=pre_index,
                pre_kdata=pre_kdata,
                tmp_direction=tmp_direction,
            )

            # 根据方向，寻找对应的分型 和 段
            if zen_state.direction == Direction.up:
                tmp_fenxing_col = "tmp_ding"
                fenxing_col = "bi_ding"
            else:
                tmp_fenxing_col = "tmp_di"
                fenxing_col = "bi_di"

            # 方向一致，延续中
            if tmp_direction == zen_state.direction:
                zen_state.opposite_count = 0
            # 反向，寻找反 分型
            else:
                zen_state.opposite_count = zen_state.opposite_count + 1

                # opposite states
                current_interval = zen_state.opposite_count
                if tmp_direction == Direction.up:
                    pre_value = acc_df.loc[index - zen_state.opposite_count, "low"]
                    current_value = kdata["high"]
                else:
                    pre_value = acc_df.loc[index - zen_state.opposite_count, "high"]
                    current_value = kdata["low"]
                acc_df.loc[index, "tmp_direction"] = tmp_direction.value
                acc_df.loc[index, "opposite_interval"] = current_interval
                change = (current_value - pre_value) / abs(pre_value)
                acc_df.loc[index, "opposite_change"] = change
                acc_df.loc[index, "opposite_slope"] = change / current_interval

                # 第一次反向
                if zen_state.opposite_count == 1:
                    acc_df.loc[pre_index, tmp_fenxing_col] = True
                    acc_df.loc[pre_index, "fenxing_power"] = fenxing_power(
                        acc_df.loc[pre_index - 1],
                        pre_kdata,
                        kdata,
                        fenxing=tmp_fenxing_col,
                    )

                    if zen_state.can_fenxing is not None:
                        # 候选底分型
                        if tmp_direction == Direction.up:
                            # 取小的
                            if pre_kdata["low"] <= zen_state.can_fenxing["low"]:
                                zen_state.can_fenxing = pre_kdata[["low", "high"]]
                                zen_state.can_fenxing_index = pre_index

                        # 候选顶分型
                        else:
                            # 取大的
                            if pre_kdata["high"] >= zen_state.can_fenxing["high"]:
                                zen_state.can_fenxing = pre_kdata[["low", "high"]]
                                zen_state.can_fenxing_index = pre_index
                    else:
                        zen_state.can_fenxing = pre_kdata[["low", "high"]]
                        zen_state.can_fenxing_index = pre_index

                # 分型确立
                if zen_state.can_fenxing is not None:
                    if zen_state.opposite_count >= 4 or (index - zen_state.can_fenxing_index >= 8):
                        acc_df.loc[zen_state.can_fenxing_index, fenxing_col] = True

                        # 记录笔的值
                        if fenxing_col == "bi_ding":
                            bi_value = acc_df.loc[zen_state.can_fenxing_index, "high"]
                        else:
                            bi_value = acc_df.loc[zen_state.can_fenxing_index, "low"]
                        acc_df.loc[zen_state.can_fenxing_index, "bi_value"] = bi_value

                        # 计算笔斜率
                        if zen_state.pre_bi:
                            change = (bi_value - zen_state.pre_bi[1]) / abs(zen_state.pre_bi[1])
                            interval = zen_state.can_fenxing_index - zen_state.pre_bi[0]
                            bi_slope = change / interval
                            acc_df.loc[zen_state.can_fenxing_index, "bi_change"] = change
                            acc_df.loc[zen_state.can_fenxing_index, "bi_slope"] = bi_slope
                            acc_df.loc[zen_state.can_fenxing_index, "bi_interval"] = interval
                        # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is CamelCase.

                        # 记录用于计算笔中枢的笔
                        # ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.
                        # 🧠 ML Signal: Use of a class attribute to hold an instance of another class, indicating a composition relationship.
                        zen_state.bis.append(
                            (
                                acc_df.loc[zen_state.can_fenxing_index, "timestamp"],
                                bi_value,
                                zen_state.can_fenxing_index,
                            )
                        )

                        # 计算笔中枢，当下来说这个 中枢 是确定的，并且是不可变的
                        # 但标记的点为 过去，注意在回测时最近的一个中枢可能用到未来函数，前一个才是 已知的
                        # 所以记了一个 current_zhongshu_y0 current_zhongshu_y1 这个是可直接使用的
                        end_index = zen_state.can_fenxing_index

                        (
                            zen_state.bis,
                            current_zhongshu,
                            current_zhongshu_change,
                            current_zhongshu_interval,
                        ) = handle_zhongshu(
                            points=zen_state.bis,
                            acc_df=acc_df,
                            end_index=end_index,
                            zhongshu_col="bi_zhongshu",
                            zhongshu_change_col="bi_zhongshu_change",
                        )

                        if not current_merge_zhongshu:
                            current_merge_zhongshu = current_zhongshu
                            current_merge_zhongshu_change = current_zhongshu_change
                            current_merge_zhongshu_level = 1
                            # ✅ Best Practice: Use of descriptive variable names improves code readability.
                            current_merge_zhongshu_interval = current_zhongshu_interval
                        # 🧠 ML Signal: Use of a schema pattern indicates structured data handling.
                        # ✅ Best Practice: Type annotations help with code clarity and static analysis.
                        else:
                            if current_zhongshu:
                                range_a = (
                                    current_merge_zhongshu.y0,
                                    current_merge_zhongshu.y1,
                                )
                                range_b = (current_zhongshu.y0, current_zhongshu.y1)
                                combine_range = combine(range_a, range_b)
                                if combine_range:
                                    y0 = combine_range[0]
                                    y1 = combine_range[1]
                                    current_merge_zhongshu = Rect(
                                        x0=current_merge_zhongshu.x0,
                                        x1=current_zhongshu.x1,
                                        y0=y0,
                                        y1=y1,
                                    )
                                    current_merge_zhongshu_change = abs(y0 - y1) / abs(y0)
                                    current_merge_zhongshu_level = current_merge_zhongshu_level + 1
                                    current_merge_zhongshu_interval = (
                                        current_merge_zhongshu_interval + current_zhongshu_interval
                                    )
                                else:
                                    current_merge_zhongshu = current_zhongshu
                                    current_merge_zhongshu_change = current_zhongshu_change
                                    current_merge_zhongshu_level = 1
                                    current_merge_zhongshu_interval = current_zhongshu_interval

                                acc_df.loc[end_index, "merge_zhongshu"] = current_merge_zhongshu
                                # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
                                acc_df.loc[end_index, "merge_zhongshu_change"] = current_merge_zhongshu_change
                                # ✅ Best Practice: Calling the superclass constructor ensures proper initialization.
                                # 🧠 ML Signal: Use of a dictionary to map keys to functions
                                acc_df.loc[end_index, "merge_zhongshu_level"] = current_merge_zhongshu_level
                                acc_df.loc[end_index, "merge_zhongshu_interval"] = current_merge_zhongshu_interval

                        # 🧠 ML Signal: Consistent use of the same function for multiple keys
                        zen_state.merge_zhongshu = current_merge_zhongshu
                        zen_state.merge_zhongshu_interval = current_merge_zhongshu_interval
                        # 🧠 ML Signal: Method returning a class or function, indicating a factory pattern
                        zen_state.merge_zhongshu_level = current_merge_zhongshu_level

                        # 🧠 ML Signal: Returning a class from a method, useful for dynamic behavior analysis
                        # ✅ Best Practice: Type hinting improves code readability and maintainability
                        zen_state.pre_bi = (zen_state.can_fenxing_index, bi_value)

                        # 🧠 ML Signal: Usage of pandas DataFrame operations like dropna
                        zen_state.opposite_count = 0
                        # ✅ Best Practice: Specify the return type for better readability and maintainability
                        zen_state.direction = zen_state.direction.opposite()
                        # 🧠 ML Signal: Returning a list containing a DataFrame
                        zen_state.can_fenxing = None
                        # 🧠 ML Signal: Usage of DataFrame dropna method to handle missing data

                        # 确定第一个段
                        # 🧠 ML Signal: Conversion of DataFrame column to list
                        if zen_state.fenxing_list != None:
                            # 🧠 ML Signal: Usage of dropna() indicates data cleaning, which is a common preprocessing step in ML pipelines.
                            zen_state.fenxing_list.append(
                                Fenxing(
                                    state=fenxing_col,
                                    kdata={
                                        # 🧠 ML Signal: Hardcoded entity IDs suggest a specific use case or dataset, which can be a feature for ML models.
                                        # ⚠️ SAST Risk (Low): Potential risk if entity_ids are user-controlled, leading to data injection.
                                        "low": float(acc_df.loc[zen_state.can_fenxing_index]["low"]),
                                        "high": float(acc_df.loc[zen_state.can_fenxing_index]["high"]),
                                    },
                                    index=zen_state.can_fenxing_index,
                                )
                            )

                            if len(zen_state.fenxing_list) == 4:
                                # 🧠 ML Signal: Visualization function call, indicating a pattern of data exploration or result presentation.
                                # ✅ Best Practice: Use of __all__ to define public API of the module, improving code maintainability and readability.
                                # 🧠 ML Signal: Instantiation of ZenFactor with specific parameters can indicate a pattern for model configuration.
                                duan_state = handle_duan(
                                    fenxing_list=zen_state.fenxing_list,
                                    pre_duan_state=zen_state.current_duan_state,
                                )

                                change = duan_state != zen_state.current_duan_state

                                if change:
                                    zen_state.current_duan_state = duan_state

                                    # 确定状态
                                    acc_df.loc[
                                        zen_state.fenxing_list[0].index : zen_state.fenxing_list[-1].index,
                                        "duan_state",
                                    ] = zen_state.current_duan_state

                                    duan_index = zen_state.fenxing_list[0].index
                                    if zen_state.current_duan_state == "up":
                                        acc_df.loc[duan_index, "duan_di"] = True
                                        duan_value = acc_df.loc[duan_index, "low"]
                                    else:
                                        duan_index = zen_state.fenxing_list[0].index
                                        acc_df.loc[duan_index, "duan_ding"] = True
                                        duan_value = acc_df.loc[duan_index, "high"]
                                    # 记录段的值
                                    acc_df.loc[duan_index, "duan_value"] = duan_value

                                    # 计算段斜率
                                    if zen_state.pre_duan:
                                        change = (duan_value - zen_state.pre_duan[1]) / abs(zen_state.pre_duan[1])
                                        interval = duan_index - zen_state.pre_duan[0]
                                        duan_slope = change / interval
                                        acc_df.loc[duan_index, "duan_change"] = change
                                        acc_df.loc[duan_index, "duan_slope"] = duan_slope
                                        acc_df.loc[duan_index, "duan_interval"] = interval

                                    zen_state.pre_duan = (duan_index, duan_value)

                                    # 记录用于计算中枢的段
                                    zen_state.duans.append(
                                        (
                                            acc_df.loc[duan_index, "timestamp"],
                                            duan_value,
                                            duan_index,
                                        )
                                    )

                                    # 计算中枢
                                    zen_state.duans, _, _, _ = handle_zhongshu(
                                        points=zen_state.duans,
                                        acc_df=acc_df,
                                        end_index=duan_index,
                                        zhongshu_col="zhongshu",
                                        zhongshu_change_col="zhongshu_change",
                                    )

                                    # 只留最后一个
                                    zen_state.fenxing_list = zen_state.fenxing_list[-1:]
                                else:
                                    # 保持之前的状态并踢出候选
                                    acc_df.loc[zen_state.fenxing_list[0].index, "duan_state"] = (
                                        zen_state.current_duan_state
                                    )
                                    zen_state.fenxing_list = zen_state.fenxing_list[1:]

            pre_kdata = kdata
            pre_index = index

        acc_df = acc_df.set_index("timestamp", drop=False)
        return acc_df, zen_state


class ZenFactor(TechnicalFactor):
    accumulator = ZenAccumulator()

    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = None,
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        fill_method: str = "ffill",
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
    ) -> None:
        self.factor_schema = get_zen_factor_schema(entity_type=entity_schema.__name__, level=level)
        super().__init__(
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            keep_all_timestamp,
            fill_method,
            effective_number,
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
        )

    def factor_col_map_object_hook(self) -> dict:
        return {
            "zhongshu": decode_rect,
            "bi_zhongshu": decode_rect,
            "merge_zhongshu": decode_rect,
        }

    def state_encoder(self):
        return FactorStateEncoder

    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        bi_value = self.factor_df[["bi_value"]].dropna()
        # duan_value = self.factor_df[['duan_value']].dropna()
        return [bi_value]

    def drawer_rects(self) -> List[Rect]:
        df1 = self.factor_df[["merge_zhongshu"]].dropna()
        return df1["merge_zhongshu"].tolist()

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        # bi_slope = self.factor_df[['bi_slope']].dropna()
        # duan_slope = self.factor_df[['duan_slope']].dropna()
        # power = self.factor_df[['fenxing_power']].dropna()
        # zhongshu_change = self.factor_df[['zhongshu_change']].dropna()
        # return [bi_slope, duan_slope, power, zhongshu_change]
        # change1 = self.factor_df[['current_merge_zhongshu_level']].dropna()
        # change2 = self.factor_df[['opposite_change']].dropna()
        current_slope = self.factor_df[["current_slope"]].dropna()
        return [current_slope]


if __name__ == "__main__":
    entity_ids = ["index_sh_000001"]
    Index1dKdata.record_data(entity_ids=entity_ids)

    f = ZenFactor(
        entity_schema=Index,
        entity_ids=entity_ids,
        need_persist=False,
        provider="em",
        entity_provider="exchange",
    )
    f.draw(show=True)


# the __all__ is generated
__all__ = ["FactorStateEncoder", "get_zen_factor_schema", "ZenState", "handle_zhongshu", "ZenAccumulator", "ZenFactor"]