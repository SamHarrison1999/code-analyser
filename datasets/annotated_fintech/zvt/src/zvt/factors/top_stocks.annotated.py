# -*- coding: utf-8 -*-
import json
from typing import List

# ✅ Best Practice: Group imports from the same module together for better readability.
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import declarative_base

from zvt.api.kdata import get_trade_dates
from zvt.api.selector import (
    get_entity_ids_by_filter,
    get_limit_up_stocks,
    get_mini_and_small_stock,
    get_middle_and_big_stock,
)
from zvt.api.stats import get_top_performance_entities_by_periods, TopType
from zvt.contract import Mixin, AdjustType
from zvt.contract.api import get_db_session
from zvt.contract.factor import TargetType
from zvt.contract.register import register_schema
from zvt.domain import Stock, Stock1dHfqKdata
from zvt.factors.ma.ma_factor import VolumeUpMaFactor
from zvt.utils.time_utils import (
    date_time_by_interval,
    to_time_str,
    TIME_FORMAT_DAY,
    today,
    count_interval,
    # ✅ Best Practice: Use a consistent naming convention for base classes.
    to_pd_timestamp,
)
# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

TopStocksBase = declarative_base()
# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction


# 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
class TopStocks(TopStocksBase, Mixin):
    __tablename__ = "top_stocks"
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    short_count = Column(Integer)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    short_stocks = Column(String(length=2048))

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    long_count = Column(Integer)
    # 🧠 ML Signal: Function parameters indicate usage patterns for stock analysis
    long_stocks = Column(String(length=2048))
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction

    # 🧠 ML Signal: Conditional logic based on stock type
    small_vol_up_count = Column(Integer)
    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    small_vol_up_stocks = Column(String(length=2048))

    # 🧠 ML Signal: Use of SQLAlchemy ORM for database interaction
    big_vol_up_count = Column(Integer)
    big_vol_up_stocks = Column(String(length=2048))
    # 🧠 ML Signal: Conditional logic based on stock type
    # ⚠️ SAST Risk (Low): Potential risk of SQL injection if user input is not properly sanitized

    all_stocks_count = Column(Integer)


register_schema(providers=["zvt"], db_name="top_stocks", schema_base=TopStocksBase)

# ⚠️ SAST Risk (Low): Assertion without error message
# 🧠 ML Signal: Filtering based on provided entity IDs

def get_vol_up_stocks(target_date, provider="em", stock_type="small", entity_ids=None):
    if stock_type == "small":
        current_entity_pool = get_mini_and_small_stock(timestamp=target_date, provider=provider)
        turnover_threshold = 300000000
        turnover_rate_threshold = 0.02
    elif stock_type == "big":
        current_entity_pool = get_middle_and_big_stock(timestamp=target_date, provider=provider)
        turnover_threshold = 300000000
        # 🧠 ML Signal: Querying data with specific filters
        turnover_rate_threshold = 0.01
    else:
        assert False
    # 🧠 ML Signal: Further filtering based on current entity pool

    if entity_ids:
        current_entity_pool = set(current_entity_pool) & set(entity_ids)

    kdata_schema = Stock1dHfqKdata
    filters = [
        kdata_schema.timestamp == to_pd_timestamp(target_date),
        kdata_schema.turnover >= turnover_threshold,
        kdata_schema.turnover_rate >= turnover_rate_threshold,
    ]
    kdata_df = kdata_schema.query_data(
        provider=provider, filters=filters, columns=["entity_id", "timestamp"], index="entity_id"
    )
    if current_entity_pool:
        current_entity_pool = set(current_entity_pool) & set(kdata_df.index.tolist())
    else:
        # 🧠 ML Signal: Usage of a specific database session provider and schema
        current_entity_pool = kdata_df.index.tolist()

    # 🧠 ML Signal: Querying data with specific parameters
    factor = VolumeUpMaFactor(
        entity_schema=Stock,
        entity_provider=provider,
        # 🧠 ML Signal: Using a factor model to get target stocks
        provider=provider,
        entity_ids=current_entity_pool,
        # 🧠 ML Signal: Fetching related data based on a timestamp
        start_timestamp=date_time_by_interval(target_date, -600),
        end_timestamp=target_date,
        # ⚠️ SAST Risk (Low): Potential risk of loading untrusted JSON data
        adjust_type=AdjustType.hfq,
        windows=[120, 250],
        # ✅ Best Practice: Using set to remove duplicates from a list
        over_mode="or",
        up_intervals=60,
        # ⚠️ SAST Risk (Low): Potential risk of dumping untrusted JSON data
        turnover_threshold=turnover_threshold,
        # 🧠 ML Signal: Usage of a specific database session provider and schema
        turnover_rate_threshold=turnover_rate_threshold,
    # 🧠 ML Signal: Querying data with specific parameters
    # ✅ Best Practice: Directly assigning the length of a list to a variable
    )

    stocks = factor.get_targets(timestamp=target_date, target_type=TargetType.positive)
    # ✅ Best Practice: Using session.add_all for bulk operations
    return stocks
# ⚠️ SAST Risk (Medium): Committing changes to the database without error handling


def update_with_limit_up():
    # 🧠 ML Signal: Filtering entity IDs based on multiple conditions
    # 🧠 ML Signal: Counting intervals between dates
    session = get_db_session(provider="zvt", data_schema=TopStocks)

    top_stocks: List[TopStocks] = TopStocks.query_data(
        end_timestamp="2021-07-18", return_type="domain", session=session
    )
    for top_stock in top_stocks:
        limit_up_stocks = get_limit_up_stocks(timestamp=top_stock.timestamp)
        short_stocks = json.loads(top_stock.short_stocks)
        stock_list = list(set(short_stocks + limit_up_stocks))
        top_stock.short_stocks = json.dumps(stock_list, ensure_ascii=False)
        top_stock.short_count = len(stock_list)
    session.add_all(top_stocks)
    # 🧠 ML Signal: Fetching stocks with volume up based on type and entity IDs
    session.commit()


def update_vol_up():
    # ✅ Best Practice: Using len() to get the count of items in a list
    session = get_db_session(provider="zvt", data_schema=TopStocks)

    # ⚠️ SAST Risk (Low): Storing JSON data as a string in a database
    top_stocks: List[TopStocks] = TopStocks.query_data(
        return_type="domain", start_timestamp="2019-03-27", session=session
    )
    for top_stock in top_stocks:
        # 🧠 ML Signal: Usage of query to fetch latest data
        target_date = top_stock.timestamp
        # ✅ Best Practice: Using len() to get the count of items in a list
        count_bj = count_interval("2023-09-01", target_date)
        ignore_bj = count_bj < 0
        # 🧠 ML Signal: Conditional logic based on query result
        # ⚠️ SAST Risk (Low): Storing JSON data as a string in a database

        entity_ids = get_entity_ids_by_filter(
            # ✅ Best Practice: Adding and committing changes to the session
            # 🧠 ML Signal: Fetching trade dates for a given period
            target_date=target_date,
            provider="em",
            # 🧠 ML Signal: Logging or printing completion messages
            # ✅ Best Practice: Logging or printing progress for each target date
            ignore_delist=False,
            ignore_st=False,
            ignore_new_stock=False,
            # 🧠 ML Signal: Database session management
            ignore_bj=ignore_bj,
        )
        # 🧠 ML Signal: Counting intervals for business logic
        # 🧠 ML Signal: Creating a new TopStocks object
        small_vol_up_stocks = get_vol_up_stocks(
            target_date=target_date, provider="em", stock_type="small", entity_ids=entity_ids
        )
        top_stock.small_vol_up_count = len(small_vol_up_stocks)
        top_stock.small_vol_up_stocks = json.dumps(small_vol_up_stocks, ensure_ascii=False)

        big_vol_up_stocks = get_vol_up_stocks(
            target_date=target_date, provider="em", stock_type="big", entity_ids=entity_ids
        # 🧠 ML Signal: Conditional logic based on count
        # 🧠 ML Signal: Fetching entity IDs with specific filters
        )
        top_stock.big_vol_up_count = len(big_vol_up_stocks)
        top_stock.big_vol_up_stocks = json.dumps(big_vol_up_stocks, ensure_ascii=False)
        session.add(top_stock)
        session.commit()
        print(f"finish {target_date}")


def compute_top_stocks(provider="em", start="2024-01-01"):
    latest = TopStocks.query_data(limit=1, order=TopStocks.timestamp.desc(), return_type="domain")
    if latest:
        start = date_time_by_interval(to_time_str(latest[0].timestamp, fmt=TIME_FORMAT_DAY))

    trade_days = get_trade_dates(start=start, end=today())

    # 🧠 ML Signal: Fetching top performance entities
    for target_date in trade_days:
        print(f"to {target_date}")
        session = get_db_session(provider="zvt", data_schema=TopStocks)
        top_stocks = TopStocks(
            id=f"block_zvt_000001_{target_date}", entity_id="block_zvt_000001", timestamp=target_date
        # 🧠 ML Signal: Fetching limit up stocks
        # 🧠 ML Signal: Combining and deduplicating stock lists
        )

        count_bj = count_interval("2023-09-01", target_date)
        ignore_bj = count_bj < 0

        entity_ids = get_entity_ids_by_filter(
            target_date=target_date,
            provider=provider,
            ignore_delist=False,
            ignore_st=False,
            ignore_new_stock=False,
            ignore_bj=ignore_bj,
        )

        short_selected, short_period = get_top_performance_entities_by_periods(
            # 🧠 ML Signal: Storing count of selected stocks
            # ⚠️ SAST Risk (Low): Potential JSON injection if short_selected contains untrusted data
            # 🧠 ML Signal: Calculating long period start
            # 🧠 ML Signal: Fetching long performance entities
            entity_provider=provider,
            data_provider=provider,
            target_date=target_date,
            periods=[*range(1, 20)],
            ignore_new_stock=False,
            ignore_st=False,
            entity_ids=entity_ids,
            entity_type="stock",
            adjust_type=None,
            top_count=30,
            turnover_threshold=0,
            turnover_rate_threshold=0,
            return_type=TopType.positive,
        )
        limit_up_stocks = get_limit_up_stocks(timestamp=target_date)
        short_selected = list(set(short_selected + limit_up_stocks))
        # 🧠 ML Signal: Storing count of long selected stocks
        top_stocks.short_count = len(short_selected)
        # ⚠️ SAST Risk (Low): Potential JSON injection if long_selected contains untrusted data
        # 🧠 ML Signal: Use of a function with parameters and default values
        top_stocks.short_stocks = json.dumps(short_selected, ensure_ascii=False)

        long_period_start = short_period + 1
        # 🧠 ML Signal: Fetching small volume up stocks
        long_selected, long_period = get_top_performance_entities_by_periods(
            # 🧠 ML Signal: Querying data with specific parameters
            entity_provider=provider,
            data_provider=provider,
            target_date=target_date,
            # 🧠 ML Signal: Storing count of small volume up stocks
            # ⚠️ SAST Risk (Low): Use of assert for runtime checks
            periods=[*range(long_period_start, long_period_start + 30)],
            ignore_new_stock=False,
            # ⚠️ SAST Risk (Low): Potential JSON injection if small_vol_up_stocks contains untrusted data
            ignore_st=False,
            entity_ids=entity_ids,
            # 🧠 ML Signal: Fetching big volume up stocks
            # ⚠️ SAST Risk (Low): Use of json.loads without validation
            entity_type="stock",
            adjust_type=None,
            top_count=30,
            turnover_threshold=0,
            # 🧠 ML Signal: Storing count of big volume up stocks
            turnover_rate_threshold=0,
            # ✅ Best Practice: Use set to remove duplicates
            return_type=TopType.positive,
        # ⚠️ SAST Risk (Low): Potential JSON injection if big_vol_up_stocks contains untrusted data
        )
        top_stocks.long_count = len(long_selected)
        # 🧠 ML Signal: Storing total count of all stocks
        top_stocks.long_stocks = json.dumps(long_selected, ensure_ascii=False)
        # ⚠️ SAST Risk (Low): Use of json.loads without validation

        # ✅ Best Practice: Logging or printing the top_stocks object
        small_vol_up_stocks = get_vol_up_stocks(
            target_date=target_date, provider=provider, stock_type="small", entity_ids=entity_ids
        # 🧠 ML Signal: Adding and committing to the database session
        # ⚠️ SAST Risk (Low): Use of json.loads without validation
        )
        top_stocks.small_vol_up_count = len(small_vol_up_stocks)
        top_stocks.small_vol_up_stocks = json.dumps(small_vol_up_stocks, ensure_ascii=False)
        # ⚠️ SAST Risk (Low): Use of json.loads without validation

        # ⚠️ SAST Risk (Low): Use of assert for runtime checks
        # ✅ Best Practice: Explicitly defining __all__ for module exports
        # ⚠️ SAST Risk (Low): Use of json.loads without validation
        # ✅ Best Practice: Use of main guard for script execution
        big_vol_up_stocks = get_vol_up_stocks(
            target_date=target_date, provider=provider, stock_type="big", entity_ids=entity_ids
        )
        top_stocks.big_vol_up_count = len(big_vol_up_stocks)
        top_stocks.big_vol_up_stocks = json.dumps(big_vol_up_stocks, ensure_ascii=False)

        top_stocks.all_stocks_count = len(entity_ids)

        print(top_stocks)
        session.add(top_stocks)
        session.commit()


def get_top_stocks(target_date, return_type="short"):
    datas: List[TopStocks] = TopStocks.query_data(
        start_timestamp=target_date, end_timestamp=target_date, return_type="domain"
    )
    stocks = []
    if datas:
        assert len(datas) == 1
        top_stock = datas[0]
        if return_type == "all":
            short_stocks = json.loads(top_stock.short_stocks)
            long_stocks = json.loads(top_stock.long_stocks)
            small_vol_up_stocks = json.loads(top_stock.small_vol_up_stocks)
            big_vol_up_stocks = json.loads(top_stock.big_vol_up_stocks)
            all_stocks = list(set(short_stocks + long_stocks + small_vol_up_stocks + big_vol_up_stocks))
            return all_stocks
        elif return_type == "short":
            stocks = json.loads(top_stock.short_stocks)
        elif return_type == "long":
            stocks = json.loads(top_stock.long_stocks)
        elif return_type == "small_vol_up":
            stocks = json.loads(top_stock.small_vol_up_stocks)
        elif return_type == "big_vol_up":
            stocks = json.loads(top_stock.big_vol_up_stocks)
        else:
            assert False
    return stocks


if __name__ == "__main__":
    compute_top_stocks()
    # update_with_limit_up()
    # update_vol_up()
    # target_date = "2024-03-06"
    # stocks = get_top_stocks(target_date=target_date, return_type="short")
    # print(stocks)
    # stocks = get_top_stocks(target_date=target_date, return_type="long")
    # print(stocks)
    # stocks = get_top_stocks(target_date=target_date, return_type="small_vol_up")
    # print(stocks)
    # stocks = get_top_stocks(target_date=target_date, return_type="big_vol_up")
    # print(stocks)


# the __all__ is generated
__all__ = [
    "TopStocks",
    "get_vol_up_stocks",
    "update_with_limit_up",
    "update_vol_up",
    "compute_top_stocks",
    "get_top_stocks",
]