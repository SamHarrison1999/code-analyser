# -*- coding: utf-8 -*-

# 🧠 ML Signal: Usage of custom imports from a specific package
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Usage of custom imports from a specific package

# 🧠 ML Signal: Usage of custom imports from a specific package
from zvt.contract import Portfolio, PortfolioStockHistory
from zvt.contract.register import register_schema, register_entity
from zvt.utils.time_utils import now_pd_timestamp

# 🧠 ML Signal: Inheritance from multiple classes, indicating a complex class structure

# ✅ Best Practice: Naming convention for base class
# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
EtfMetaBase = declarative_base()

# ✅ Best Practice: Consider adding type hints for better code readability and maintainability


# 🧠 ML Signal: Usage of a decorator for entity registration
# 🧠 ML Signal: Definition of a database column with a specific data type and length
#: etf
# ✅ Best Practice: Import statements should be at the top of the file
@register_entity(entity_type="etf")
# 🧠 ML Signal: Use of class method decorator indicating a method that operates on the class itself
class Etf(EtfMetaBase, Portfolio):
    # 🧠 ML Signal: Function call with multiple parameters, useful for learning API usage patterns
    # ✅ Best Practice: Define a class-level attribute for the table name to improve maintainability and readability.
    __tablename__ = "etf"
    category = Column(String(length=64))
    # 🧠 ML Signal: Registering schema with specific providers and database names indicates usage patterns for database interactions.
    # ✅ Best Practice: Use __all__ to explicitly declare the public API of the module, improving code readability and maintainability.

    @classmethod
    def get_stocks(
        cls,
        code=None,
        codes=None,
        ids=None,
        timestamp=now_pd_timestamp(),
        provider=None,
    ):
        from zvt.api.portfolio import get_etf_stocks

        return get_etf_stocks(
            code=code, codes=codes, ids=ids, timestamp=timestamp, provider=provider
        )


class EtfStock(EtfMetaBase, PortfolioStockHistory):
    __tablename__ = "etf_stock"


register_schema(
    providers=["exchange", "joinquant"], db_name="etf_meta", schema_base=EtfMetaBase
)


# the __all__ is generated
__all__ = ["Etf", "EtfStock"]
