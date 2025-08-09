# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract.register import register_schema
# 🧠 ML Signal: Usage of SQLAlchemy's declarative_base indicates ORM pattern.
from zvt.factors.ma.domain.common import MaStatsFactorCommon

# 🧠 ML Signal: Use of class inheritance and tablename for ORM mapping
Stock1dMaStatsFactorBase = declarative_base()
# 🧠 ML Signal: Registration of schema with specific providers and database name
# ⚠️ SAST Risk (Low): Ensure that the 'providers' and 'db_name' are validated to prevent injection attacks
# ✅ Best Practice: Use of __all__ to define public API of the module


class Stock1dMaStatsFactor(Stock1dMaStatsFactorBase, MaStatsFactorCommon):
    __tablename__ = "stock_1d_ma_stats_factor"


register_schema(providers=["zvt"], db_name="stock_1d_ma_stats_factor", schema_base=Stock1dMaStatsFactorBase)


# the __all__ is generated
__all__ = ["Stock1dMaStatsFactor"]