# -*- coding: utf-8 -*-
# ✅ Best Practice: Import only necessary components to avoid namespace pollution
from typing import List
# 🧠 ML Signal: Definition of a class with multiple attributes, useful for understanding data structure patterns

from zvt.contract.model import MixinModel
# 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types


# 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
class PositionModel(MixinModel):
    #: 机器人名字
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    trader_name: str
    #: 做多数量
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    long_amount: float
    #: 可平多数量
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    available_long: float
    #: 平均做多价格
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    average_long_price: float
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    #: 做空数量
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    short_amount: float
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    #: 可平空数量
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    available_short: float
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    #: 平均做空价格
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    average_short_price: float
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    #: 盈亏
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    profit: float
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    # 🧠 ML Signal: Use of type annotations for class attributes, indicating expected data types
    #: 盈亏比例
    profit_rate: float
    #: 市值 或者 占用的保证金(方便起见，总是100%)
    value: float
    #: 交易类型(0代表T+0,1代表T+1)
    trading_t: int


class AccountStatsModel(MixinModel):
    #: 投入金额
    input_money: float
    #: 机器人名字
    trader_name: str
    #: 具体仓位
    positions: List[PositionModel]
    #: 市值
    value: float
    #: 可用现金
    cash: float
    #: value + cash
    all_value: float

    #: 盈亏
    profit: float
    #: 盈亏比例
    profit_rate: float

    #: 收盘计算
    closing: bool