# -*- coding: utf-8 -*-
import logging
from typing import List
# 🧠 ML Signal: Usage of external library for pagination

import pandas as pd
from fastapi_pagination.ext.sqlalchemy import paginate

import zvt.api.kdata as kdata_api
import zvt.contract.api as contract_api
from zvt.common.query_models import TimeUnit
from zvt.domain import Stock, StockQuote, Stock1mQuote
from zvt.tag.tag_schemas import StockTags, StockPools
from zvt.trading.common import ExecutionStatus
from zvt.trading.trading_models import (
    BuildTradingPlanModel,
    QueryTradingPlanModel,
    QueryTagQuoteModel,
    QueryStockQuoteModel,
    BuildQueryStockQuoteSettingModel,
    KdataRequestModel,
    TSRequestModel,
# 🧠 ML Signal: Usage of utility function for checking DataFrame null values
)
from zvt.trading.trading_schemas import TradingPlan, QueryStockQuoteSetting, TagQuoteStats
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import (
    to_time_str,
    to_pd_timestamp,
    now_pd_timestamp,
    date_time_by_interval,
    current_date,
    # 🧠 ML Signal: Function signature and parameter types can be used to infer usage patterns.
    date_and_time,
# ✅ Best Practice: Use of logging for tracking and debugging
# 🧠 ML Signal: API call pattern can be used to understand data retrieval methods.
)

logger = logging.getLogger(__name__)


def query_kdata(kdata_request_model: KdataRequestModel):
    kdata_df = kdata_api.get_kdata(
        entity_ids=kdata_request_model.entity_ids,
        provider=kdata_request_model.data_provider,
        # ✅ Best Practice: Check for null data before processing to avoid errors.
        # ✅ Best Practice: Convert timestamps to a consistent format for easier processing.
        start_timestamp=kdata_request_model.start_timestamp,
        end_timestamp=kdata_request_model.end_timestamp,
        adjust_type=kdata_request_model.adjust_type,
    )
    if pd_is_not_null(kdata_df):
        kdata_df["timestamp"] = kdata_df["timestamp"].apply(lambda x: int(x.timestamp()))
        # ✅ Best Practice: Use apply for row-wise operations to maintain readability.
        kdata_df["data"] = kdata_df.apply(
            lambda x: x[
                ["timestamp", "open", "high", "low", "close", "volume", "turnover", "change_pct", "turnover_rate"]
            ].values.tolist(),
            axis=1,
        )
        # ✅ Best Practice: Use groupby and agg for efficient data aggregation.
        df = kdata_df.groupby("entity_id").agg(
            code=("code", "first"),
            name=("name", "first"),
            # 🧠 ML Signal: Usage of external API to fetch trading dates
            level=("level", "first"),
            # ✅ Best Practice: Reset index after groupby to maintain DataFrame structure.
            # 🧠 ML Signal: Querying data based on entity IDs and provider
            datas=("data", lambda data: list(data)),
        )
        df = df.reset_index(drop=False)
        return df.to_dict(orient="records")

# 🧠 ML Signal: Returning data as a dictionary can indicate data serialization patterns.

# ⚠️ SAST Risk (Low): Potential risk if ts_df is None or not a DataFrame
# ✅ Best Practice: Using apply with lambda for row-wise operations
def query_ts(ts_request_model: TSRequestModel):
    trading_dates = kdata_api.get_recent_trade_dates(days_count=ts_request_model.days_count)
    ts_df = Stock1mQuote.query_data(
        entity_ids=ts_request_model.entity_ids,
        provider=ts_request_model.data_provider,
        start_timestamp=trading_dates[0],
    )
    if pd_is_not_null(ts_df):
        ts_df["data"] = ts_df.apply(
            lambda x: x[
                ["time", "price", "avg_price", "change_pct", "volume", "turnover", "turnover_rate"]
            # ✅ Best Practice: Using groupby and agg for data aggregation
            ].values.tolist(),
            axis=1,
        )
        # ✅ Best Practice: Use of context manager for session ensures proper resource management
        df = ts_df.groupby("entity_id").agg(
            code=("code", "first"),
            # ✅ Best Practice: Resetting index for a clean DataFrame
            name=("name", "first"),
            # 🧠 ML Signal: Conversion of date to string and timestamp for consistent date handling
            datas=("data", lambda data: list(data)),
        # 🧠 ML Signal: Converting DataFrame to dictionary for record-oriented data
        )
        df = df.reset_index(drop=False)
        # 🧠 ML Signal: Use of trading signal type as a value for plan identification
        return df.to_dict(orient="records")


# 🧠 ML Signal: Unique plan ID generation pattern
def build_trading_plan(build_trading_plan_model: BuildTradingPlanModel):
    # ✅ Best Practice: Use of query_data method for data retrieval
    with contract_api.DBSession(provider="zvt", data_schema=TradingPlan)() as session:
        stock_id = build_trading_plan_model.stock_id
        trading_date_str = to_time_str(build_trading_plan_model.trading_date)
        trading_date = to_pd_timestamp(trading_date_str)
        signal = build_trading_plan_model.trading_signal_type.value
        # ✅ Best Practice: Use of query_data method for data retrieval
        # ⚠️ SAST Risk (Low): Assertion can raise exceptions if condition is not met
        # ✅ Best Practice: Use of ORM model for data manipulation
        plan_id = f"{stock_id}_{trading_date_str}_{signal}"

        datas = TradingPlan.query_data(
            session=session, filters=[TradingPlan.id == plan_id], limit=1, return_type="domain"
        )
        if datas:
            assert len(datas) == 1
            plan = datas[0]
        else:
            datas = Stock.query_data(provider="em", entity_id=stock_id, return_type="domain")
            stock = datas[0]
            plan = TradingPlan(
                id=plan_id,
                entity_id=stock_id,
                stock_id=stock_id,
                stock_code=stock.code,
                stock_name=stock.name,
                trading_date=trading_date,
                expected_open_pct=build_trading_plan_model.expected_open_pct,
                # 🧠 ML Signal: Function definition with specific input type can be used to infer usage patterns
                buy_price=build_trading_plan_model.buy_price,
                sell_price=build_trading_plan_model.sell_price,
                # ⚠️ SAST Risk (Low): Use of external session management, ensure proper handling of session lifecycle
                trading_reason=build_trading_plan_model.trading_reason,
                # 🧠 ML Signal: Timestamping for tracking plan creation or update time
                trading_signal_type=signal,
                # ✅ Best Practice: Adding and committing changes to the session
                # 🧠 ML Signal: Conditional logic based on object attributes can indicate decision-making patterns
                status=ExecutionStatus.init.value,
            )
        plan.timestamp = now_pd_timestamp()
        # ✅ Best Practice: Refreshing the session to get the latest state of the object
        # 🧠 ML Signal: Use of date and time manipulation functions can indicate temporal data handling
        session.add(plan)
        session.commit()
        session.refresh(plan)
        return plan


def query_trading_plan(query_trading_plan_model: QueryTradingPlanModel):
    with contract_api.DBSession(provider="zvt", data_schema=TradingPlan)() as session:
        # 🧠 ML Signal: Querying data with specific parameters can indicate data access patterns
        # 🧠 ML Signal: Use of a database session pattern
        time_range = query_trading_plan_model.time_range
        if time_range.relative_time_range:
            # 🧠 ML Signal: Use of pagination function can indicate handling of large datasets
            # 🧠 ML Signal: Querying data with specific filters and ordering
            # ⚠️ SAST Risk (Low): Potential for SQL injection if filters are not properly sanitized
            start_timestamp = date_time_by_interval(
                current_date(), time_range.relative_time_range.interval, time_range.relative_time_range.time_unit
            )
            end_timestamp = None
        else:
            start_timestamp = time_range.absolute_time_range.start_timestamp
            # 🧠 ML Signal: Use of a database session pattern
            end_timestamp = time_range.absolute_time_range.end_timestamp
        # ✅ Best Practice: Use of context manager for database session
        selectable = TradingPlan.query_data(
            # 🧠 ML Signal: Querying data with specific filters and order
            # ⚠️ SAST Risk (Low): Potential SQL injection if filters are not properly sanitized
            session=session, start_timestamp=start_timestamp, end_timestamp=end_timestamp, return_type="select"
        )
        return paginate(session, selectable)


def get_current_trading_plan():
    with contract_api.DBSession(provider="zvt", data_schema=TradingPlan)() as session:
        # ✅ Best Practice: Using a context manager for session management ensures that resources are properly managed and released.
        return TradingPlan.query_data(
            # 🧠 ML Signal: Querying data with specific filters and ordering can indicate patterns in data retrieval.
            session=session,
            filters=[TradingPlan.status == ExecutionStatus.pending.value],
            order=TradingPlan.trading_date.asc(),
            return_type="domain",
        )

# 🧠 ML Signal: Ordering by trading date can indicate a preference for processing data in chronological order.
# 🧠 ML Signal: Filtering by status and trading date can be used to identify specific trading plan conditions.

# 🧠 ML Signal: Usage of a specific filter range for change_pct
def get_future_trading_plan():
    # 🧠 ML Signal: Returning data as a specific type can indicate how the data is intended to be used.
    with contract_api.DBSession(provider="zvt", data_schema=TradingPlan)() as session:
        return TradingPlan.query_data(
            session=session,
            filters=[TradingPlan.status == ExecutionStatus.init.value],
            order=TradingPlan.trading_date.asc(),
            # 🧠 ML Signal: Calculation of statistics from a DataFrame
            # ✅ Best Practice: Logging the current plans can help in debugging and tracking the flow of data.
            return_type="domain",
        )


def check_trading_plan():
    with contract_api.DBSession(provider="zvt", data_schema=TradingPlan)() as session:
        plans = TradingPlan.query_data(
            session=session,
            # ✅ Best Practice: Check if the list is not empty before accessing the first element
            filters=[TradingPlan.status == ExecutionStatus.init.value, TradingPlan.trading_date == current_date()],
            order=TradingPlan.trading_date.asc(),
            return_type="domain",
        )

        logger.debug(f"current plans:{plans}")
# 🧠 ML Signal: Usage of a specific filter range for change_pct


def query_quote_stats():
    quote_df = StockQuote.query_data(
        return_type="df",
        filters=[StockQuote.change_pct >= -0.31, StockQuote.change_pct <= 0.31],
        columns=["timestamp", "entity_id", "time", "change_pct", "turnover", "is_limit_up", "is_limit_down"],
    )
    current_stats = cal_quote_stats(quote_df)
    # ✅ Best Practice: Check for null DataFrame before processing
    start_timestamp = current_stats["timestamp"]

    # 🧠 ML Signal: Calculation of statistics from a DataFrame
    pre_date_df = Stock1mQuote.query_data(
        # 🧠 ML Signal: Usage of DataFrame operations to calculate statistics
        filters=[Stock1mQuote.timestamp < to_time_str(start_timestamp)],
        # ✅ Best Practice: Avoid using magic numbers; consider using a named constant for clarity
        order=Stock1mQuote.timestamp.desc(),
        # 🧠 ML Signal: Grouping data by a constant to aggregate over the entire DataFrame
        # 🧠 ML Signal: Custom aggregation function to count positive changes
        limit=1,
        columns=["timestamp"],
    )
    pre_date = pre_date_df["timestamp"].tolist()[0]

    if start_timestamp.hour >= 15:
        start_timestamp = date_and_time(pre_date, "15:00")
    else:
        start_timestamp = date_and_time(pre_date, f"{start_timestamp.hour}:{start_timestamp.minute}")
    end_timestamp = date_time_by_interval(start_timestamp, 1, TimeUnit.minute)

    pre_df = Stock1mQuote.query_data(
        return_type="df",
        start_timestamp=start_timestamp,
        # 🧠 ML Signal: Custom aggregation function to count non-positive changes
        # 🧠 ML Signal: Summing turnover values
        # 🧠 ML Signal: Calculating mean of percentage changes
        end_timestamp=end_timestamp,
        # 🧠 ML Signal: Counting occurrences of limit up events
        # 🧠 ML Signal: Function name suggests a specific task related to stock data processing
        filters=[Stock1mQuote.change_pct >= -0.31, Stock1mQuote.change_pct <= 0.31],
        # 🧠 ML Signal: Custom aggregation function to count limit down events
        # 🧠 ML Signal: Querying data from a database using specific filters
        columns=["timestamp", "entity_id", "time", "change_pct", "turnover", "is_limit_up", "is_limit_down"],
    )

    if pd_is_not_null(pre_df):
        pre_stats = cal_quote_stats(pre_df)
        current_stats["pre_turnover"] = pre_stats["turnover"]
        # 🧠 ML Signal: Converting DataFrame to dictionary for record-based access
        current_stats["turnover_change"] = current_stats["turnover"] - current_stats["pre_turnover"]
    return current_stats


# 🧠 ML Signal: Querying data with specific columns and filters
def cal_quote_stats(quote_df):
    quote_df["ss"] = 1

    df = (
        quote_df.groupby("ss")
        .agg(
            timestamp=("timestamp", "last"),
            time=("time", "last"),
            up_count=("change_pct", lambda x: (x > 0).sum()),
            down_count=("change_pct", lambda x: (x <= 0).sum()),
            # 🧠 ML Signal: Converting a DataFrame column to a list
            turnover=("turnover", "sum"),
            # 🧠 ML Signal: Concatenating DataFrames
            # 🧠 ML Signal: Querying data with specific return type and index
            # ⚠️ SAST Risk (Low): Assumes 'timestamp' column exists and has at least one entry
            change_pct=("change_pct", "mean"),
            limit_up_count=("is_limit_up", "sum"),
            limit_down_count=("is_limit_down", lambda x: (x == True).sum()),
        )
        .reset_index(drop=True)
    )

    return df.to_dict(orient="records")[0]


def cal_tag_quote_stats(stock_pool_name):
    stock_pools: List[StockPools] = StockPools.query_data(
        filters=[StockPools.stock_pool_name == stock_pool_name],
        # 🧠 ML Signal: Grouping and aggregating DataFrame data
        order=StockPools.timestamp.desc(),
        limit=1,
        return_type="domain",
    )
    if stock_pools:
        entity_ids = stock_pools[0].entity_ids
    else:
        entity_ids = None
    # 🧠 ML Signal: Adding new columns to DataFrame

    # 🧠 ML Signal: Applying a function to DataFrame rows
    tag_df = StockTags.query_data(
        entity_ids=entity_ids,
        filters=[StockTags.main_tag.isnot(None)],
        columns=[StockTags.entity_id, StockTags.main_tag],
        # 🧠 ML Signal: Creating unique identifiers for DataFrame rows
        # 🧠 ML Signal: Usage of a specific query pattern to retrieve stock pool data
        # ⚠️ SAST Risk (Low): Potential risk if `query_data` method is vulnerable to injection attacks
        return_type="df",
        index="entity_id",
    )

    entity_ids = tag_df["entity_id"].tolist()

    # ✅ Best Practice: Using descending order to get the latest timestamp
    # ⚠️ SAST Risk (Low): Printing potentially sensitive data to the console
    quote_df = StockQuote.query_data(entity_ids=entity_ids, return_type="df", index="entity_id")
    # 🧠 ML Signal: Storing DataFrame to a database
    timestamp = quote_df["timestamp"].tolist()[0]

    df = pd.concat([tag_df, quote_df], axis=1)
    # 🧠 ML Signal: Filtering data based on main tags
    grouped_df = (
        df.groupby("main_tag")
        .agg(
            up_count=("change_pct", lambda x: (x > 0).sum()),
            down_count=("change_pct", lambda x: (x <= 0).sum()),
            turnover=("turnover", "sum"),
            change_pct=("change_pct", "mean"),
            # ⚠️ SAST Risk (Low): Potential risk if `in_` method is vulnerable to injection attacks
            limit_up_count=("is_limit_up", "sum"),
            limit_down_count=("is_limit_down", lambda x: (x == True).sum()),
            total_count=("main_tag", "size"),  # 添加计数，计算每个分组的总行数
        # ✅ Best Practice: Concatenating DataFrames for combined analysis
        # 🧠 ML Signal: Conversion of DataFrame column to list
        # 🧠 ML Signal: Querying stock quotes based on entity IDs
        )
        .reset_index(drop=False)
    )
    grouped_df["stock_pool_name"] = stock_pool_name

    grouped_df["entity_id"] = grouped_df[["stock_pool_name", "main_tag"]].apply(
        lambda se: "{}_{}".format(se["stock_pool_name"], se["main_tag"]), axis=1
    )
    grouped_df["timestamp"] = timestamp
    grouped_df["id"] = grouped_df[["entity_id", "timestamp"]].apply(
        lambda se: "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"])), axis=1
    )

    # 🧠 ML Signal: Grouping and aggregating data for analysis
    # 🧠 ML Signal: Counting positive changes
    print(grouped_df)
    # 🧠 ML Signal: Counting non-positive changes

    contract_api.df_to_db(
        # 🧠 ML Signal: Summing turnover values
        df=grouped_df, data_schema=TagQuoteStats, provider="zvt", force_update=True, drop_duplicates=False
    )
# 🧠 ML Signal: Usage of query pattern with filters and ordering
# 🧠 ML Signal: Calculating mean percentage change
# 🧠 ML Signal: Counting limit up occurrences


def query_tag_quotes(query_tag_quote_model: QueryTagQuoteModel):
    stock_pools: List[StockPools] = StockPools.query_data(
        filters=[StockPools.stock_pool_name == query_tag_quote_model.stock_pool_name],
        order=StockPools.timestamp.desc(),
        # 🧠 ML Signal: Counting limit down occurrences
        limit=1,
        return_type="domain",
    )
    if stock_pools:
        entity_ids = stock_pools[0].entity_ids
    # ✅ Best Practice: Sorting DataFrame for prioritized analysis
    # 🧠 ML Signal: Converting DataFrame to dictionary for output
    # 🧠 ML Signal: Conditional query based on main_tag presence
    else:
        entity_ids = None

    tag_df = StockTags.query_data(
        entity_ids=entity_ids,
        filters=[StockTags.main_tag.in_(query_tag_quote_model.main_tags)],
        columns=[StockTags.entity_id, StockTags.main_tag],
        return_type="df",
        index="entity_id",
    )

    entity_ids = tag_df["entity_id"].tolist()
    # 🧠 ML Signal: Query without filters

    quote_df = StockQuote.query_data(entity_ids=entity_ids, return_type="df", index="entity_id")

    # 🧠 ML Signal: Mapping entity_ids to tags
    df = pd.concat([tag_df, quote_df], axis=1)
    grouped_df = (
        # ⚠️ SAST Risk (High): Use of eval() with dynamic input can lead to code injection
        df.groupby("main_tag")
        .agg(
            # 🧠 ML Signal: Query pattern with dynamic ordering
            # 🧠 ML Signal: Accessing dictionary keys to retrieve values
            up_count=("change_pct", lambda x: (x > 0).sum()),
            down_count=("change_pct", lambda x: (x <= 0).sum()),
            # 🧠 ML Signal: Accessing dictionary keys to retrieve values
            turnover=("turnover", "sum"),
            change_pct=("change_pct", "mean"),
            # 🧠 ML Signal: Accessing dictionary keys to retrieve values
            limit_up_count=("is_limit_up", "sum"),
            limit_down_count=("is_limit_down", lambda x: (x == True).sum()),
            total_count=("main_tag", "size"),  # 添加计数，计算每个分组的总行数
        # 🧠 ML Signal: Conditional logic based on dictionary content
        )
        .reset_index(drop=False)
    )
    sorted_df = grouped_df.sort_values(by=["turnover", "total_count"], ascending=[False, False])
    # 🧠 ML Signal: Returning a pandas Series from a function

    return sorted_df.to_dict(orient="records")
# 🧠 ML Signal: Applying a function across DataFrame rows


# 🧠 ML Signal: Counting occurrences based on a condition
def query_stock_quotes(query_stock_quote_model: QueryStockQuoteModel):
    # 🧠 ML Signal: Counting occurrences based on a condition
    # 🧠 ML Signal: Summing a DataFrame column
    entity_ids = None
    if query_stock_quote_model.stock_pool_name:
        stock_pools: List[StockPools] = StockPools.query_data(
            filters=[StockPools.stock_pool_name == query_stock_quote_model.stock_pool_name],
            order=StockPools.timestamp.desc(),
            limit=1,
            return_type="domain",
        )
        if stock_pools:
            # 🧠 ML Signal: Calculating the mean of a DataFrame column
            # 🧠 ML Signal: Counting occurrences based on a condition
            entity_ids = stock_pools[0].entity_ids
    # 🧠 ML Signal: Function definition with a financial context, useful for identifying domain-specific functions
    else:
        # 🧠 ML Signal: Converting DataFrame to a list of dictionaries
        entity_ids = query_stock_quote_model.entity_ids
    # ✅ Best Practice: Define a function to encapsulate stock selling logic for reusability and clarity

    # 🧠 ML Signal: Constructing a dictionary with computed values
    if query_stock_quote_model.main_tag:
        # ✅ Best Practice: Use 'pass' as a placeholder for future implementation
        # 🧠 ML Signal: Function definition with a specific model parameter indicates a pattern for ML model usage
        tags_dict = StockTags.query_data(
            entity_ids=entity_ids,
            # ⚠️ SAST Risk (Low): Hardcoded ID value could lead to potential issues if not managed properly
            filters=[StockTags.main_tag == query_stock_quote_model.main_tag],
            return_type="dict",
        )
        # ⚠️ SAST Risk (Medium): Querying data without input validation can lead to SQL injection risks
        if not tags_dict:
            return None
        # ⚠️ SAST Risk (Low): Potential risk of large data exposure if quotes is too large
        entity_ids = [item["entity_id"] for item in tags_dict]
    else:
        tags_dict = StockTags.query_data(
            # 🧠 ML Signal: Returning a dictionary from a function
            # ✅ Best Practice: Initialize object with default values to ensure consistency
            return_type="dict",
        )
    # 🧠 ML Signal: Updating timestamp indicates a pattern of tracking changes over time

    entity_tags_map = {item["entity_id"]: item for item in tags_dict}
    # 🧠 ML Signal: Assigning model attributes to object properties shows a pattern of data transformation

    order = eval(f"StockQuote.{query_stock_quote_model.order_by_field}.{query_stock_quote_model.order_by_type.value}()")
    # 🧠 ML Signal: Function to build default settings if not present

    # ⚠️ SAST Risk (Low): Adding objects to session without validation can lead to data integrity issues
    df = StockQuote.query_data(order=order, entity_ids=entity_ids, return_type="df")

    # ⚠️ SAST Risk (Medium): Committing session without exception handling can lead to unhandled errors
    if not pd_is_not_null(df):
        # 🧠 ML Signal: Function call with specific parameters
        return None
    # ✅ Best Practice: Refreshing session to ensure the object is updated with the latest database state

    # 🧠 ML Signal: Returning the updated object indicates a pattern of function output for further processing
    # ✅ Best Practice: Use of __name__ guard for script execution
    # ✅ Best Practice: Explicitly defining __all__ for module exports
    def set_tags(quote):
        entity_id = quote["entity_id"]
        main_tag = entity_tags_map.get(entity_id, {}).get("main_tag", None)
        sub_tag = entity_tags_map.get(entity_id, {}).get("sub_tag", None)
        active_hidden_tags = entity_tags_map.get(entity_id, {}).get("active_hidden_tags", None)
        if active_hidden_tags:
            hidden_tags = list(active_hidden_tags.keys())
        else:
            hidden_tags = None
        return pd.Series({"main_tag": main_tag, "sub_tag": sub_tag, "hidden_tags": hidden_tags})

    df[["main_tag", "sub_tag", "hidden_tags"]] = df.apply(set_tags, axis=1)

    up_count = (df["change_pct"] > 0).sum()
    down_count = (df["change_pct"] < 0).sum()
    turnover = df["turnover"].sum()
    change_pct = df["change_pct"].mean()
    limit_up_count = df["is_limit_up"].sum()
    limit_down_count = df["is_limit_down"].sum()

    quotes = df.to_dict(orient="records")

    result = {
        "up_count": up_count,
        "down_count": down_count,
        "turnover": turnover,
        "change_pct": change_pct,
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "quotes": quotes[: query_stock_quote_model.limit],
    }
    return result


def buy_stocks():
    pass


def sell_stocks():
    pass


def build_query_stock_quote_setting(build_query_stock_quote_setting_model: BuildQueryStockQuoteSettingModel):
    with contract_api.DBSession(provider="zvt", data_schema=QueryStockQuoteSetting)() as session:
        the_id = "admin_setting"
        datas = QueryStockQuoteSetting.query_data(ids=[the_id], session=session, return_type="domain")
        if datas:
            query_setting = datas[0]
        else:
            query_setting = QueryStockQuoteSetting(entity_id="admin", id=the_id)
        query_setting.timestamp = current_date()
        query_setting.stock_pool_name = build_query_stock_quote_setting_model.stock_pool_name
        query_setting.main_tags = build_query_stock_quote_setting_model.main_tags
        session.add(query_setting)
        session.commit()
        session.refresh(query_setting)
        return query_setting


def build_default_query_stock_quote_setting():
    datas = QueryStockQuoteSetting.query_data(ids=["admin_setting"], return_type="domain")
    if datas:
        return
    build_query_stock_quote_setting(BuildQueryStockQuoteSettingModel(stock_pool_name="all", main_tags=["消费电子"]))


if __name__ == "__main__":
    # print(query_tag_quotes(QueryTagQuoteModel(stock_pool_name="all", main_tags=["低空经济", "半导体", "化工", "消费电子"])))
    # print(query_stock_quotes(QueryStockQuoteModel(stock_pool_name="all", main_tag="半导体")))
    print(query_quote_stats())
# the __all__ is generated
__all__ = [
    "build_trading_plan",
    "query_trading_plan",
    "get_current_trading_plan",
    "get_future_trading_plan",
    "check_trading_plan",
    "query_stock_quotes",
    "buy_stocks",
    "sell_stocks",
    "build_query_stock_quote_setting",
]