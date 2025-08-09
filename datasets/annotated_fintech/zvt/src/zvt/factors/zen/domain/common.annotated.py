# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability and maintainability.
from sqlalchemy import Column, Float, String, Boolean, Integer

# 🧠 ML Signal: Class definition with multiple attributes can be used to train models on class structure and attribute usage

from zvt.contract import Mixin

# ✅ Best Practice: Importing specific classes or functions is preferred over wildcard imports for clarity.
# 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

# ⚠️ SAST Risk (Low): Potential risk if user input is directly mapped to these columns without validation


class ZenFactorCommon(Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    level = Column(String(length=32))
    # 开盘价
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    open = Column(Float)
    # 收盘价
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    close = Column(Float)
    # 最高价
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    high = Column(Float)
    # 最低价
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    low = Column(Float)
    # 成交量
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    volume = Column(Float)
    # 成交金额
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    turnover = Column(Float)

    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 笔的底
    bi_di = Column(Boolean)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 笔的顶
    bi_ding = Column(Boolean)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 记录笔顶/底分型的值，bi_di取low,bi_ding取high,其他为None,绘图时取有值的连线即为 笔
    bi_value = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 笔的变化
    bi_change = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 笔的斜率
    bi_slope = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 持续的周期
    bi_interval = Column(Integer)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    # 记录临时分型，不变
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    tmp_ding = Column(Boolean)
    tmp_di = Column(Boolean)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 分型的力度
    fenxing_power = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    # 目前分型确定的方向
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    current_direction = Column(String(length=16))
    current_change = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    current_interval = Column(Integer)
    current_slope = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 最近的一个笔中枢
    # current_zhongshu = Column(String(length=512))
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    current_zhongshu_y0 = Column(Float)
    current_zhongshu_y1 = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    current_zhongshu_change = Column(Float)
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    current_merge_zhongshu_y0 = Column(Float)
    current_merge_zhongshu_y1 = Column(Float)
    current_merge_zhongshu_change = Column(Float)
    current_merge_zhongshu_level = Column(Integer)
    current_merge_zhongshu_interval = Column(Integer)

    # 目前走势的临时方向 其跟direction的的关系 确定了下一个分型
    tmp_direction = Column(String(length=16))
    # 已经确定分型，目前反向才有值
    opposite_change = Column(Float)
    opposite_slope = Column(Float)
    opposite_interval = Column(Integer)

    duan_state = Column(String(length=32))

    # 段的底
    duan_di = Column(Boolean)
    # 段的顶
    duan_ding = Column(Boolean)
    # 记录段顶/底的值，为duan_di时取low,为duan_ding时取high,其他为None,绘图时取有值的连线即为 段
    duan_value = Column(Float)
    # 段的变化
    duan_change = Column(Float)
    # 段的斜率
    duan_slope = Column(Float)
    # 持续的周期
    duan_interval = Column(Integer)

    # 记录在确定中枢的最后一个段的终点x1，值为Rect(x0,y0,x1,y1)
    zhongshu = Column(String(length=512))
    zhongshu_change = Column(Float)

    # 记录在确定中枢的最后一个笔的终点x1，值为Rect(x0,y0,x1,y1)
    bi_zhongshu = Column(String(length=512))
    bi_zhongshu_change = Column(Float)

    # 从前往后，合并相邻的有重叠的笔中枢
    merge_zhongshu = Column(String(length=512))
    merge_zhongshu_change = Column(Float)
    merge_zhongshu_level = Column(Integer)
    merge_zhongshu_interval = Column(Integer)


# the __all__ is generated
__all__ = ["ZenFactorCommon"]
