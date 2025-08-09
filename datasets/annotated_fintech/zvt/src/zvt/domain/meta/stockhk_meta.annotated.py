# -*- coding: utf-8 -*-
from sqlalchemy import Column, Boolean

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import TradableEntity
from zvt.contract.register import register_schema, register_entity

# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.

StockhkMetaBase = declarative_base()
# 🧠 ML Signal: Use of decorators to register entities indicates a pattern for extensibility and plugin-like architecture.
# ✅ Best Practice: Use of SQLAlchemy's Column to define table columns


#: 港股
@register_entity(entity_type="stockhk")
class Stockhk(StockhkMetaBase, TradableEntity):
    __tablename__ = "stockhk"
    #: 是否属于港股通
    south = Column(Boolean)
    # ✅ Best Practice: Consider adding a return type hint for better readability and maintainability

    @classmethod
    def get_trading_t(cls):
        """
        0 means t+0
        1 means t+1

        :return:
        # 🧠 ML Signal: Conditional logic based on a boolean parameter
        """
        return 0

    @classmethod
    # ⚠️ SAST Risk (Low): Potential risk if `register_schema` is not properly validated or sanitized
    # ✅ Best Practice: Explicitly defining `__all__` to control module exports
    def get_trading_intervals(cls, include_bidding_time=False):
        """
        overwrite it to get the trading intervals of the entity

        :return: list of time intervals, in format [(start,end)]
        """
        if include_bidding_time:
            return [("09:15", "12:00"), ("13:00", "16:00")]
        else:
            return [("09:30", "12:00"), ("13:00", "16:00")]


register_schema(providers=["em"], db_name="stockhk_meta", schema_base=StockhkMetaBase)


# the __all__ is generated
__all__ = ["Stockhk"]
