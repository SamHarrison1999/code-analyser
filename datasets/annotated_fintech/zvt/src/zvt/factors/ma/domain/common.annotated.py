# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy import Column, Float, Integer

# 🧠 ML Signal: Use of financial indicators as class attributes
from zvt.contract import Mixin

# 🧠 ML Signal: Use of financial indicators as class attributes

class MaStatsFactorCommon(Mixin):
    # 🧠 ML Signal: Use of financial indicators as class attributes
    open = Column(Float)
    close = Column(Float)
    # 🧠 ML Signal: Use of financial indicators as class attributes
    high = Column(Float)
    low = Column(Float)
    # 🧠 ML Signal: Use of financial indicators as class attributes
    turnover = Column(Float)

    # 🧠 ML Signal: Use of moving averages as class attributes
    ma5 = Column(Float)
    ma10 = Column(Float)
    # 🧠 ML Signal: Use of moving averages as class attributes

    ma34 = Column(Float)
    # 🧠 ML Signal: Use of moving averages as class attributes
    ma55 = Column(Float)
    ma89 = Column(Float)
    # 🧠 ML Signal: Use of moving averages as class attributes
    ma144 = Column(Float)

    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Use of moving averages as class attributes
    # 🧠 ML Signal: Use of integer attribute for status or flag
    # 🧠 ML Signal: Use of integer attribute for counting
    # 🧠 ML Signal: Use of distance as a float attribute
    ma120 = Column(Float)
    ma250 = Column(Float)

    vol_ma30 = Column(Float)

    live = Column(Integer)
    count = Column(Integer)
    distance = Column(Float)
    area = Column(Float)


# the __all__ is generated
__all__ = ["MaStatsFactorCommon"]