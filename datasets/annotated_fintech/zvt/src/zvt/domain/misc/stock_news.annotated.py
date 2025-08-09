# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, JSON, Boolean, DateTime, Integer

# ✅ Best Practice: Grouping related imports together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract import Mixin

# ✅ Best Practice: Naming convention for base classes is clear and descriptive.
from zvt.contract.register import register_schema

# ✅ Best Practice: Use of class-level variable for table name improves maintainability and readability.

NewsBase = declarative_base()
# 🧠 ML Signal: Use of String type for news_code indicates it might be used as a unique identifier or key.


# 🧠 ML Signal: Use of String type for news_url suggests it stores URLs, which can be used for web scraping or link analysis.
class StockNews(NewsBase, Mixin):
    __tablename__ = "stock_news"
    # 🧠 ML Signal: Use of String type for news_title indicates it stores titles, which can be used for NLP tasks.

    #: 新闻编号
    # 🧠 ML Signal: Use of String type for news_content suggests it stores large text data, useful for text analysis or NLP.
    news_code = Column(String)
    #: 新闻地址
    # 🧠 ML Signal: Use of JSON type for news_analysis indicates structured data storage, useful for ML model predictions or analysis.
    # ✅ Best Practice: Use of Integer for position ensures efficient storage and comparison.
    news_url = Column(String)
    #: 新闻标题
    # 🧠 ML Signal: Use of Boolean type for ignore_by_user suggests binary classification or filtering by user preference.
    # ✅ Best Practice: Use of JSON for entity_ids allows flexible storage of structured data.
    news_title = Column(String)
    # ✅ Best Practice: Default value for ignore_by_user improves data integrity and consistency.
    #: 新闻内容
    # ✅ Best Practice: Use of String for news_code is appropriate for storing short text identifiers.
    news_content = Column(String)
    #: 新闻解读
    # ✅ Best Practice: Use of String for news_title is appropriate for storing short text data.
    news_analysis = Column(JSON)
    # ✅ Best Practice: Use of String for news_content is appropriate for storing longer text data.
    # ✅ Best Practice: Use of JSON for news_analysis allows flexible storage of structured data.
    # 🧠 ML Signal: Registering schema with specific providers and entity types can indicate data source and domain.
    # ✅ Best Practice: Defining __all__ helps in controlling the import of module components.
    #: 用户设置为忽略
    ignore_by_user = Column(Boolean, default=False)


class StockHotTopic(NewsBase, Mixin):
    __tablename__ = "stock_hot_topic"

    #: 出现时间
    created_timestamp = Column(DateTime)
    #: 热度排行
    position = Column(Integer)
    #: 相关标的
    entity_ids = Column(JSON)

    #: 新闻编号
    news_code = Column(String)
    #: 新闻标题
    news_title = Column(String)
    #: 新闻内容
    news_content = Column(String)
    #: 新闻解读
    news_analysis = Column(JSON)


register_schema(
    providers=["em"], db_name="stock_news", schema_base=NewsBase, entity_type="stock"
)


# the __all__ is generated
__all__ = ["StockNews", "StockHotTopic"]
