# -*- coding: utf-8 -*-

# 🧠 ML Signal: Usage of custom mixin class for ORM models
from sqlalchemy import Column, String, JSON, Boolean, Float, Integer
from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Custom schema registration pattern

# 🧠 ML Signal: Use of SQLAlchemy ORM for database modeling
from zvt.contract import Mixin

# 🧠 ML Signal: Declarative base pattern for SQLAlchemy ORM
from zvt.contract.register import register_schema

# 🧠 ML Signal: Use of SQLAlchemy ORM for database table definition

StockTagsBase = declarative_base()
# 🧠 ML Signal: Use of SQLAlchemy Column for defining table columns

# ⚠️ SAST Risk (Low): Potential SQL Injection if user input is not sanitized
# ✅ Best Practice: Class should inherit from object for compatibility with Python 2 and 3


class IndustryInfo(StockTagsBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy Column for defining table columns
    # ✅ Best Practice: Use a class variable to define the table name for clarity
    __tablename__ = "industry_info"
    # ⚠️ SAST Risk (Low): Potential SQL Injection if user input is not sanitized

    # 🧠 ML Signal: Usage of unique constraint on a column
    # ✅ Best Practice: Class should inherit from object explicitly in Python 2.x for new-style classes, but in Python 3.x it's optional.
    industry_name = Column(String, unique=True)
    # 🧠 ML Signal: Use of SQLAlchemy Column for defining table columns
    description = Column(String)
    # ⚠️ SAST Risk (Low): Potential SQL Injection if user input is not sanitized
    # 🧠 ML Signal: Definition of a column without constraints
    # 🧠 ML Signal: Use of __tablename__ indicates interaction with a database ORM.
    # related main tag
    main_tag = Column(String)


# 🧠 ML Signal: Column definition with unique constraint indicates a need for unique values in the database.

# ✅ Best Practice: Class names should follow the CapWords convention for readability


# 🧠 ML Signal: Column definition indicates a mapping to a database field.
class MainTagInfo(StockTagsBase, Mixin):
    # ✅ Best Practice: Use of __tablename__ is a common pattern in SQLAlchemy for table naming
    __tablename__ = "main_tag_info"
    # 🧠 ML Signal: Column definition indicates a mapping to a database field.

    # 🧠 ML Signal: Use of unique constraint on a column indicates a need for distinct values
    tag = Column(String, unique=True)
    tag_reason = Column(String)


class SubTagInfo(StockTagsBase, Mixin):
    # ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage.
    __tablename__ = "sub_tag_info"

    # ✅ Best Practice: Specify a maximum length for string columns to prevent excessive data storage.
    tag = Column(String, unique=True)
    tag_reason = Column(String)
    # ⚠️ SAST Risk (Low): Consider adding constraints or validation to ensure data integrity for string fields.

    # related main tag
    # ⚠️ SAST Risk (Low): Consider adding constraints or validation to ensure data integrity for string fields.
    main_tag = Column(String)


# ⚠️ SAST Risk (Low): JSON fields can store arbitrary data; ensure proper validation and sanitization.


class HiddenTagInfo(StockTagsBase, Mixin):
    # ⚠️ SAST Risk (Low): Consider adding constraints or validation to ensure data integrity for string fields.
    __tablename__ = "hidden_tag_info"

    # ⚠️ SAST Risk (Low): Consider adding constraints or validation to ensure data integrity for string fields.
    # ✅ Best Practice: Use of __tablename__ to explicitly define the table name in SQLAlchemy
    tag = Column(String, unique=True)
    tag_reason = Column(String)


# ⚠️ SAST Risk (Low): JSON fields can store arbitrary data; ensure proper validation and sanitization.
# 🧠 ML Signal: Use of a fixed-length string for 'code' suggests a standardized identifier


# ⚠️ SAST Risk (Low): JSON fields can store arbitrary data; ensure proper validation and sanitization.
# 🧠 ML Signal: Use of a longer string for 'name' suggests it holds descriptive text
class StockTags(StockTagsBase, Mixin):
    """
    Schema for storing stock tags
    """

    # 🧠 ML Signal: Boolean fields can indicate binary states, useful for classification models.
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models

    __tablename__ = "stock_tags"
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models

    code = Column(String(length=64))
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    name = Column(String(length=128))

    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    # ✅ Best Practice: Class inherits from multiple base classes, indicating use of mixins for shared functionality
    main_tag = Column(String)
    main_tag_reason = Column(String)
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    # 🧠 ML Signal: Use of __tablename__ suggests ORM pattern, common in database interaction
    main_tags = Column(JSON)

    # 🧠 ML Signal: Integer field could represent a count or frequency, useful for ML models
    # 🧠 ML Signal: Column definitions indicate ORM usage for database schema mapping
    # ✅ Best Practice: Class inherits from multiple base classes, ensure MRO is correct
    sub_tag = Column(String)
    sub_tag_reason = Column(String)
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    # 🧠 ML Signal: Unique constraint on stock_pool_name suggests importance of this field for identification
    # 🧠 ML Signal: Custom table name for ORM mapping
    sub_tags = Column(JSON)

    # 🧠 ML Signal: Column definition for ORM, indicates schema design
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    active_hidden_tags = Column(JSON)
    hidden_tags = Column(JSON)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
    # 🧠 ML Signal: Boolean fields indicate binary features that can be used in ML models
    # 🧠 ML Signal: Use of JSON type for a column, indicates flexible data storage
    set_by_user = Column(Boolean, default=False)


# 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
# 🧠 ML Signal: JSON field can store complex data structures, useful for ML feature extraction


class StockSystemTags(StockTagsBase, Mixin):
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
    __tablename__ = "stock_system_tags"
    #: 编码
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
    code = Column(String(length=64))
    #: 名字
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
    name = Column(String(length=128))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database schema definition
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Registering schema with specific providers and database name
    #: 减持
    recent_reduction = Column(Boolean)
    #: 增持
    recent_acquisition = Column(Boolean)
    #: 解禁
    recent_unlock = Column(Boolean)
    #: 增发配股
    recent_additional_or_rights_issue = Column(Boolean)
    #: 业绩利好
    recent_positive_earnings_news = Column(Boolean)
    #: 业绩利空
    recent_negative_earnings_news = Column(Boolean)
    #: 上榜次数
    recent_dragon_and_tiger_count = Column(Integer)
    #: 违规行为
    recent_violation_alert = Column(Boolean)
    #: 利好
    recent_positive_news = Column(Boolean)
    #: 利空
    recent_negative_news = Column(Boolean)
    #: 新闻总结
    recent_news_summary = Column(JSON)


class StockPoolInfo(StockTagsBase, Mixin):
    __tablename__ = "stock_pool_info"
    stock_pool_type = Column(String)
    stock_pool_name = Column(String, unique=True)


class StockPools(StockTagsBase, Mixin):
    __tablename__ = "stock_pools"
    stock_pool_name = Column(String)
    entity_ids = Column(JSON)


class TagStats(StockTagsBase, Mixin):
    __tablename__ = "tag_stats"

    stock_pool_name = Column(String)
    main_tag = Column(String)
    turnover = Column(Float)
    entity_count = Column(Integer)
    position = Column(Integer)
    is_main_line = Column(Boolean)
    main_line_continuous_days = Column(Integer)
    entity_ids = Column(JSON)


register_schema(providers=["zvt"], db_name="stock_tags", schema_base=StockTagsBase)


# the __all__ is generated
__all__ = [
    "IndustryInfo",
    "MainTagInfo",
    "SubTagInfo",
    "HiddenTagInfo",
    "StockTags",
    "StockSystemTags",
    "StockPoolInfo",
    "StockPools",
    "TagStats",
]
