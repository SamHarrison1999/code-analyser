# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, DateTime, Float
# ✅ Best Practice: Grouping related imports together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin
# ✅ Best Practice: Naming convention for base classes should be consistent and descriptive.
from zvt.contract.register import register_schema

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
DividendFinancingBase = declarative_base()

# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

class DividendFinancing(DividendFinancingBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    __tablename__ = "dividend_financing"

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    provider = Column(String(length=32))
    code = Column(String(length=32))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    #: 分红总额
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # ✅ Best Practice: Define a table name for ORM mapping to ensure clarity and avoid errors.
    dividend_money = Column(Float)

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.
    #: 新股
    ipo_issues = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.
    ipo_raising_fund = Column(Float)

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    # 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.
    #: 增发
    # ✅ Best Practice: Class inherits from DividendFinancingBase and Mixin, indicating a structured design.
    spo_issues = Column(Float)
    # 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.
    spo_raising_fund = Column(Float)
    # ✅ Best Practice: Use of __tablename__ to explicitly define the table name in the database.
    #: 配股
    # 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.
    rights_issues = Column(Float)
    # ✅ Best Practice: Use of Column with String type for provider, ensuring consistent data type.
    rights_raising_fund = Column(Float)
# 🧠 ML Signal: Usage of SQLAlchemy's Column to define database schema.

# ✅ Best Practice: Use of Column with String type for code, ensuring consistent data type.

class DividendDetail(DividendFinancingBase, Mixin):
    # ✅ Best Practice: Use of Column with Float type for spo_issues, ensuring consistent data type.
    __tablename__ = "dividend_detail"
    # ✅ Best Practice: Define column types and constraints for database schema

    # ✅ Best Practice: Use of Column with Float type for spo_price, ensuring consistent data type.
    provider = Column(String(length=32))
    # ✅ Best Practice: Define column types and constraints for database schema
    code = Column(String(length=32))
    # ✅ Best Practice: Use of Column with Float type for spo_raising_fund, ensuring consistent data type.

    # ✅ Best Practice: Define column types and constraints for database schema
    #: 公告日
    # ✅ Best Practice: Define column types and constraints for database schema
    announce_date = Column(DateTime)
    #: 股权登记日
    record_date = Column(DateTime)
    # ✅ Best Practice: Define column types and constraints for database schema
    # 🧠 ML Signal: Usage of register_schema function indicates schema registration pattern
    # ✅ Best Practice: Use of __all__ to define public API of the module
    #: 除权除息日
    dividend_date = Column(DateTime)

    #: 方案
    dividend = Column(String(length=128))


class SpoDetail(DividendFinancingBase, Mixin):
    __tablename__ = "spo_detail"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    spo_issues = Column(Float)
    spo_price = Column(Float)
    spo_raising_fund = Column(Float)


class RightsIssueDetail(DividendFinancingBase, Mixin):
    __tablename__ = "rights_issue_detail"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    #: 配股
    rights_issues = Column(Float)
    rights_issue_price = Column(Float)
    rights_raising_fund = Column(Float)


register_schema(
    providers=["eastmoney"], db_name="dividend_financing", schema_base=DividendFinancingBase, entity_type="stock"
)


# the __all__ is generated
__all__ = ["DividendFinancing", "DividendDetail", "SpoDetail", "RightsIssueDetail"]