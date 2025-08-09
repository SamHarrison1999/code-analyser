# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.ext.declarative import declarative_base

from zvt.contract.register import register_schema

# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
from zvt.factors.zen.domain.common import ZenFactorCommon

# 🧠 ML Signal: Use of class inheritance, indicating a pattern of code reuse and extension.
# ✅ Best Practice: Class naming follows a clear and descriptive convention.
Stock1wkZenFactorBase = declarative_base()


# 🧠 ML Signal: Registration of schema with specific providers and database name, indicating a pattern of database interaction.
# ✅ Best Practice: Use of __all__ to define public symbols of the module, improving code clarity and module interface management.
# ✅ Best Practice: Use of keyword arguments improves code readability and maintainability.
class Stock1wkZenFactor(Stock1wkZenFactorBase, ZenFactorCommon):
    __tablename__ = "stock_1wk_zen_factor"


register_schema(
    providers=["zvt"],
    db_name="stock_1wk_zen_factor",
    schema_base=Stock1wkZenFactorBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = ["Stock1wkZenFactor"]
