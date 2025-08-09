# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin

# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
from zvt.contract.register import register_schema

# 🧠 ML Signal: Inheritance from multiple classes indicates a pattern of combining functionalities.

TradeDayBase = declarative_base()
# 🧠 ML Signal: Use of a class attribute to define a database table name is a common pattern in ORM usage.
# 🧠 ML Signal: Registering a schema with specific providers and database name indicates a pattern of dynamic schema management.
# ⚠️ SAST Risk (Low): Ensure that the 'providers' and 'db_name' values are validated to prevent injection attacks.
# ✅ Best Practice: Defining __all__ to specify public symbols of the module improves code readability and maintainability.


class StockTradeDay(TradeDayBase, Mixin):
    __tablename__ = "stock_trade_day"


register_schema(providers=["joinquant"], db_name="trade_day", schema_base=TradeDayBase)


# the __all__ is generated
__all__ = ["StockTradeDay"]
