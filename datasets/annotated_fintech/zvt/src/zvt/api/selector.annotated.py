# -*- coding: utf-8 -*-
import logging

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
import pandas as pd
from sqlalchemy import or_, and_

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.api.kdata import default_adjust_type, get_kdata_schema, get_latest_kdata_date, get_recent_trade_dates
from zvt.contract import IntervalLevel, AdjustType
# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.contract.api import get_entity_ids
from zvt.domain import DragonAndTiger, Stock1dHfqKdata, Stock, LimitUpInfo, StockQuote, StockQuoteLog
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import to_pd_timestamp, date_time_by_interval, current_date, now_timestamp
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

logger = logging.getLogger(__name__)
# ✅ Best Practice: Use a consistent logger naming convention

# 🧠 ML Signal: Constants for market capitalization thresholds indicate domain-specific usage
# 🧠 ML Signal: Constants for department names indicate domain-specific usage
# 500亿
BIG_CAP = 50000000000
# 150亿
MIDDLE_CAP = 15000000000
# 40亿
SMALL_CAP = 4000000000

# 买入榜单
IN_DEPS = ["dep1", "dep2", "dep3", "dep4", "dep5"]
# 卖出入榜单
# ✅ Best Practice: Initialize filters as a list to collect filter conditions
OUT_DEPS = ["dep_1", "dep_2", "dep_3", "dep_4", "dep_5"]


def get_entity_ids_by_filter(
    # ✅ Best Practice: Use a default function to get the current date if target_date is not provided
    provider="em",
    # ✅ Best Practice: Calculate a date one year before the target_date
    ignore_delist=True,
    ignore_st=True,
    ignore_new_stock=False,
    # ✅ Best Practice: Append filter condition to the filters list
    target_date=None,
    entity_schema=Stock,
    # ✅ Best Practice: Append filter condition to the filters list
    entity_ids=None,
    ignore_bj=False,
):
    filters = []
    if ignore_new_stock:
        # ✅ Best Practice: Append multiple filter conditions to the filters list
        if not target_date:
            target_date = current_date()
        pre_year = date_time_by_interval(target_date, -365)
        filters += [entity_schema.timestamp <= pre_year]
    else:
        if target_date:
            # ✅ Best Practice: Append multiple filter conditions to the filters list
            filters += [entity_schema.timestamp <= target_date]
    # 🧠 ML Signal: Function definition with a specific purpose, useful for understanding code behavior
    if ignore_delist:
        filters += [
            # 🧠 ML Signal: Querying data with specific parameters, indicating a pattern of data retrieval
            entity_schema.name.not_like("%退%"),
            entity_schema.name.not_like("%PT%"),
        # ⚠️ SAST Risk (Low): No validation on the 'timestamp' input, potential for incorrect data retrieval
        # 🧠 ML Signal: Function with multiple parameters and default values
        ]
    # ✅ Best Practice: Append filter condition to the filters list
    # 🧠 ML Signal: Conditional check for data presence, common pattern in data processing

    # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed in optimized mode
    if ignore_st:
        # 🧠 ML Signal: Converting DataFrame column to list, a common data transformation pattern
        # 🧠 ML Signal: Function call with dynamic filters and parameters
        filters += [
            entity_schema.name.not_like("%ST%"),
            entity_schema.name.not_like("%*ST%"),
        ]
    if ignore_bj:
        filters += [entity_schema.exchange != "bj"]

    return get_entity_ids(provider=provider, entity_schema=entity_schema, filters=filters, entity_ids=entity_ids)
# 🧠 ML Signal: Querying data with dynamic filters


def get_limit_up_stocks(timestamp):
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    df = LimitUpInfo.query_data(start_timestamp=timestamp, end_timestamp=timestamp, columns=[LimitUpInfo.entity_id])
    # 🧠 ML Signal: Grouping and sorting data
    if pd_is_not_null(df):
        # 🧠 ML Signal: Function call with named arguments, useful for learning API usage patterns
        return df["entity_id"].tolist()

# ✅ Best Practice: Convert index to list before slicing for clarity

def get_dragon_and_tigger_player(start_timestamp, end_timestamp=None, direction="in"):
    # ✅ Best Practice: Convert index to list before slicing for clarity
    assert direction in ("in", "out")

    # ✅ Best Practice: Convert index to list before slicing for clarity
    filters = None
    if direction == "in":
        # ✅ Best Practice: Use set to remove duplicates before returning
        filters = [DragonAndTiger.change_pct > 0]
        columns = ["dep1", "dep2", "dep3"]
    elif direction == "out":
        filters = [DragonAndTiger.change_pct > 0]
        columns = ["dep_1", "dep_2", "dep_3"]

    df = DragonAndTiger.query_data(start_timestamp=start_timestamp, end_timestamp=end_timestamp, filters=filters)
    counts = []
    for col in columns:
        counts.append(df[[col, f"{col}_rate"]].groupby(col).count().sort_values(f"{col}_rate", ascending=False))
    return counts


def get_big_players(start_timestamp, end_timestamp=None, count=40):
    dep1, dep2, dep3 = get_dragon_and_tigger_player(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    # 榜1前40
    bang1 = dep1.index.tolist()[:count]

    # 榜2前40
    bang2 = dep2.index.tolist()[:count]
    # ⚠️ SAST Risk (Low): Potential risk of duplicated index entries if not handled properly

    # 榜3前40
    bang3 = dep3.index.tolist()[:count]

    # 🧠 ML Signal: Usage of date_time_by_interval function to calculate end_date
    return list(set(bang1 + bang2 + bang3))


def get_player_performance(start_timestamp, end_timestamp=None, days=5, players="机构专用", provider="em", buy_rate=5):
    filters = []
    if isinstance(players, str):
        players = [players]

    if isinstance(players, list):
        for player in players:
            # ⚠️ SAST Risk (Low): Logging potentially sensitive information
            filters.append(
                or_(
                    and_(DragonAndTiger.dep1 == player, DragonAndTiger.dep1_rate >= buy_rate),
                    and_(DragonAndTiger.dep2 == player, DragonAndTiger.dep2_rate >= buy_rate),
                    # 🧠 ML Signal: Calculation of change_pct as a performance metric
                    # ✅ Best Practice: Using pandas DataFrame for structured data handling
                    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
                    and_(DragonAndTiger.dep3 == player, DragonAndTiger.dep3_rate >= buy_rate),
                    and_(DragonAndTiger.dep4 == player, DragonAndTiger.dep4_rate >= buy_rate),
                    and_(DragonAndTiger.dep5 == player, DragonAndTiger.dep5_rate >= buy_rate),
                )
            )
    else:
        raise AssertionError("players should be list or str type")

    df = DragonAndTiger.query_data(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        # 🧠 ML Signal: Usage of a function to fetch player performance data, which could be a pattern for data retrieval.
        filters=filters,
        index=["entity_id", "timestamp"],
        provider=provider,
    )
    df = df[~df.index.duplicated(keep="first")]
    records = []
    for entity_id, timestamp in df.index:
        end_date = date_time_by_interval(timestamp, days + round(days + days * 2 / 5 + 30))
        kdata = Stock1dHfqKdata.query_data(
            # ⚠️ SAST Risk (Low): Potential division by zero if df is empty, consider adding a check.
            entity_id=entity_id,
            start_timestamp=timestamp,
            end_timestamp=end_date,
            # 🧠 ML Signal: Use of default parameters can indicate common usage patterns
            provider=provider,
            # 🧠 ML Signal: Conversion of records to a DataFrame, indicating a pattern of data processing and transformation.
            index="timestamp",
        )
        if len(kdata) <= days:
            # 🧠 ML Signal: Iteration over a fixed range can indicate a pattern in data processing
            logger.warning(f"ignore {timestamp} -> end_timestamp: {end_date}")
            break
        close = kdata["close"]
        change_pct = (close[days] - close[0]) / close[0]
        records.append({"entity_id": entity_id, "timestamp": timestamp, f"change_pct": change_pct})
    # 🧠 ML Signal: Iteration over a fixed range can indicate a pattern in data processing
    return pd.DataFrame.from_records(records)


def get_player_success_rate(
    start_timestamp,
    end_timestamp=None,
    intervals=(3, 5, 10, 60),
    players=("机构专用", "东方财富证券股份有限公司拉萨团结路第二证券营业部"),
    provider="em",
):
    records = []
    # ✅ Best Practice: Initialize lists outside of loops for clarity
    for player in players:
        record = {"player": player}
        for days in intervals:
            df = get_player_performance(
                # 🧠 ML Signal: Consistent data transformation pattern
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                days=days,
                players=player,
                provider=provider,
            )
            # 🧠 ML Signal: Consistent data transformation pattern
            # ⚠️ SAST Risk (Low): Using a mutable default value for `timestamp` can lead to unexpected behavior if `current_date()` returns a mutable object.
            rate = len(df[df["change_pct"] > 0]) / len(df)
            record[f"rate_{days}"] = rate
        # ✅ Best Practice: Use descriptive variable names for clarity and maintainability.
        records.append(record)
    return pd.DataFrame.from_records(records, index="player")
# ✅ Best Practice: Use of pd.concat for combining DataFrames is efficient and clear

# ✅ Best Practice: Use logging instead of print for better control over output and log levels.

# ✅ Best Practice: Sorting DataFrame by index for organized data output
# 🧠 ML Signal: The function `get_big_players` is used to filter or retrieve a subset of data, indicating a pattern of data selection.
def get_players(entity_id, start_timestamp, end_timestamp, provider="em", direction="in", buy_rate=5):
    columns = ["entity_id", "timestamp"]
    if direction == "in":
        # ✅ Best Practice: Use logging to record information, which is more flexible and appropriate for production environments.
        for i in range(5):
            columns.append(f"dep{i + 1}")
            # 🧠 ML Signal: The function `get_player_success_rate` is used to calculate or retrieve performance metrics, indicating a pattern of data analysis.
            columns.append(f"dep{i + 1}_rate")
    elif direction == "out":
        for i in range(5):
            # ✅ Best Practice: Use of default parameter values for better function flexibility
            columns.append(f"dep_{i + 1}")
            # 🧠 ML Signal: Filtering data based on conditions is a common pattern in data processing and analysis.
            columns.append(f"dep_{i + 1}_rate")

    df = DragonAndTiger.query_data(
        entity_id=entity_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        provider=provider,
        columns=columns,
        # 🧠 ML Signal: Checking for non-null DataFrame to handle data availability
        index=["entity_id", "timestamp"],
    )
    # ✅ Best Practice: Calculating new column 'cap' for further filtering
    dfs = []
    if direction == "in":
        for i in range(5):
            p_df = df[[f"dep{i + 1}", f"dep{i + 1}_rate"]].copy()
            # ✅ Best Practice: Filtering DataFrame based on 'cap_start' for targeted results
            p_df.columns = ["player", "buy_rate"]
            dfs.append(p_df)
    elif direction == "out":
        # ✅ Best Practice: Filtering DataFrame based on 'cap_end' for targeted results
        for i in range(5):
            # 🧠 ML Signal: Returning list of entity IDs as a result
            # ⚠️ SAST Risk (Low): Potential infinite recursion if data is never available
            p_df = df[[f"dep_{i + 1}", f"dep_{i + 1}_rate"]].copy()
            p_df.columns = ["player", "buy_rate"]
            dfs.append(p_df)

    player_df = pd.concat(dfs, sort=True)
    return player_df.sort_index(level=[0, 1])


def get_good_players(timestamp=current_date(), recent_days=400, intervals=(3, 5, 10)):
    # 🧠 ML Signal: Recursive call pattern with retry mechanism
    # 🧠 ML Signal: Function with default parameter value, indicating common usage pattern
    end_timestamp = date_time_by_interval(timestamp, -intervals[-1] - 30)
    # ✅ Best Practice: Use of named parameters for clarity and maintainability
    # recent year
    start_timestamp = date_time_by_interval(end_timestamp, -recent_days)
    print(f"{start_timestamp} to {end_timestamp}")
    # ✅ Best Practice: Explicit parameter passing improves readability
    # ✅ Best Practice: Provide a default value for the 'provider' parameter to enhance function usability.
    # 最近一年牛x的营业部
    # 🧠 ML Signal: Use of a constant (BIG_CAP) suggests a threshold or boundary condition
    # 🧠 ML Signal: Function calls another function with specific parameters, indicating a pattern of usage.
    players = get_big_players(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    logger.info(players)
    df = get_player_success_rate(
        # 🧠 ML Signal: Function with default parameter value indicating common usage pattern
        # ✅ Best Practice: Use of None to indicate no upper limit for cap_end
        # 🧠 ML Signal: Use of specific constants (MIDDLE_CAP, BIG_CAP) can indicate domain-specific knowledge.
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, intervals=intervals, players=players
    # ✅ Best Practice: Use of named parameters improves readability and maintainability.
    # 🧠 ML Signal: Hardcoded string "stock" indicates specific entity type filtering
    # 🧠 ML Signal: Function call with specific parameters indicating a pattern of usage
    )
    good_players = df[(df["rate_3"] > 0.4) & (df["rate_5"] > 0.3) & (df["rate_10"] > 0.3)].index.tolist()
    return good_players
# ✅ Best Practice: Passing provider as a parameter allows for flexibility and reuse
# 🧠 ML Signal: Function with default parameter value indicating common usage pattern
# 🧠 ML Signal: Named parameter usage indicating common practice

# 🧠 ML Signal: Use of constants indicating a pattern of usage
# ✅ Best Practice: Using named arguments for clarity and maintainability

def get_entity_list_by_cap(
    timestamp, cap_start, cap_end, entity_type="stock", provider=None, adjust_type=None, retry_times=20
# 🧠 ML Signal: Use of constants indicating a pattern of usage
# 🧠 ML Signal: Function with default parameter values indicating common usage patterns
):
    # 🧠 ML Signal: Hardcoded string indicating a specific usage pattern
    # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
    if not adjust_type:
        adjust_type = default_adjust_type(entity_type=entity_type)
    # 🧠 ML Signal: Function call with keyword arguments indicating common usage patterns

    # ✅ Best Practice: Use of keyword arguments for clarity and maintainability
    # 🧠 ML Signal: Function with default parameter value, indicating common usage pattern
    # 🧠 ML Signal: Passing a variable as a parameter indicating flexibility in usage
    kdata_schema = get_kdata_schema(entity_type, level=IntervalLevel.LEVEL_1DAY, adjust_type=adjust_type)
    # ✅ Best Practice: Use of descriptive function name for clarity
    # ✅ Best Practice: Use of default parameter value for flexibility
    df = kdata_schema.query_data(
        provider=provider,
        filters=[kdata_schema.timestamp == to_pd_timestamp(timestamp)],
        # 🧠 ML Signal: Function call with specific parameters, indicating common usage pattern
        # 🧠 ML Signal: Function to query and filter stock data for limit up stocks
        index="entity_id",
    # ✅ Best Practice: Use of named arguments for clarity
    )
    # 🧠 ML Signal: Querying stock data with specific filters and columns
    if pd_is_not_null(df):
        df["cap"] = df["turnover"] / df["turnover_rate"]
        # ⚠️ SAST Risk (Low): Potential risk if pd_is_not_null is not properly defined or validated
        # 🧠 ML Signal: Function parameter 'n' controls the number of top stocks, indicating user preference for data size
        df_result = df.copy()
        if cap_start:
            # 🧠 ML Signal: Converting DataFrame column to list for further processing
            # 🧠 ML Signal: Querying stock data with specific columns and order indicates user interest in top-performing stocks
            df_result = df_result.loc[(df["cap"] >= cap_start)]
        # ⚠️ SAST Risk (Low): Potential SQL injection risk if 'query_data' method does not properly sanitize inputs
        if cap_end:
            df_result = df_result.loc[(df["cap"] <= cap_end)]
        # 🧠 ML Signal: Usage of current timestamp for time-based operations
        # 🧠 ML Signal: Checking if DataFrame is not null indicates handling of empty or invalid data scenarios
        return df_result.index.tolist()
    # 🧠 ML Signal: Querying data with specific columns and order
    # 🧠 ML Signal: Converting DataFrame column to list shows interest in specific data format for further processing
    else:
        if retry_times == 0:
            return []
        return get_entity_list_by_cap(
            timestamp=date_time_by_interval(timestamp, 1),
            # 🧠 ML Signal: Accessing specific data from a DataFrame
            cap_start=cap_start,
            cap_end=cap_end,
            entity_type=entity_type,
            # ⚠️ SAST Risk (Low): Potential information exposure through logging
            provider=provider,
            adjust_type=adjust_type,
            # ⚠️ SAST Risk (Low): Logging potentially sensitive timing information
            retry_times=retry_times - 1,
        )


# 🧠 ML Signal: Filtering data based on time range
def get_big_cap_stock(timestamp, provider="em"):
    # 🧠 ML Signal: Querying data with filters and specific columns
    return get_entity_list_by_cap(
        timestamp=timestamp, cap_start=BIG_CAP, cap_end=None, entity_type="stock", provider=provider
    )


def get_middle_cap_stock(timestamp, provider="em"):
    # ✅ Best Practice: Checking if DataFrame is not null before processing
    return get_entity_list_by_cap(
        # ✅ Best Practice: Sorting DataFrame for consistent processing
        timestamp=timestamp, cap_start=MIDDLE_CAP, cap_end=BIG_CAP, entity_type="stock", provider=provider
    )
# 🧠 ML Signal: Grouping and aggregating data for analysis


def get_small_cap_stock(timestamp, provider="em"):
    return get_entity_list_by_cap(
        timestamp=timestamp, cap_start=SMALL_CAP, cap_end=MIDDLE_CAP, entity_type="stock", provider=provider
    )

# ⚠️ SAST Risk (Low): Potential information exposure through logging
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use

# 🧠 ML Signal: Identifying significant changes in data
# 🧠 ML Signal: Conditional logic based on provider type
def get_mini_cap_stock(timestamp, provider="em"):
    return get_entity_list_by_cap(
        timestamp=timestamp, cap_start=None, cap_end=SMALL_CAP, entity_type="stock", provider=provider
    )


# ✅ Best Practice: Returning results in a clear and structured format
def get_mini_and_small_stock(timestamp, provider="em"):
    return get_entity_list_by_cap(
        # 🧠 ML Signal: Conversion of DataFrame column to list
        timestamp=timestamp, cap_start=None, cap_end=MIDDLE_CAP, entity_type="stock", provider=provider
    )
# 🧠 ML Signal: Handling of optional parameters
# ⚠️ SAST Risk (Low): Potential timezone issues with date handling


def get_middle_and_big_stock(timestamp, provider="em"):
    return get_entity_list_by_cap(
        timestamp=timestamp, cap_start=MIDDLE_CAP, cap_end=None, entity_type="stock", provider=provider
    )


def get_limit_up_today():
    # 🧠 ML Signal: Default parameter value for 'n' indicates typical usage pattern
    df = StockQuote.query_data(filters=[StockQuote.is_limit_up], columns=[StockQuote.entity_id])
    if pd_is_not_null(df):
        # 🧠 ML Signal: Querying data with specific columns and order indicates a pattern of data retrieval
        return df["entity_id"].to_list()

# ⚠️ SAST Risk (Low): Potential risk if pd_is_not_null is not properly handling null checks
# 🧠 ML Signal: Conversion of DataFrame column to list
# 🧠 ML Signal: Function to query and filter stock data for limit down events

def get_top_up_today(n=100):
    # 🧠 ML Signal: Converting DataFrame column to list indicates a pattern of data transformation
    # ⚠️ SAST Risk (Low): Ensure that the query_data method handles input sanitization to prevent injection attacks
    df = StockQuote.query_data(columns=[StockQuote.entity_id], order=StockQuote.change_pct.desc(), limit=n)
    if pd_is_not_null(df):
        # ✅ Best Practice: Check for null data before processing to avoid runtime errors
        return df["entity_id"].to_list()
# 🧠 ML Signal: Usage of default parameters and function calls as default values

# 🧠 ML Signal: Conversion of DataFrame column to list for further processing

def get_shoot_today(up_change_pct=0.03, down_change_pct=-0.03, interval=2):
    current_time = now_timestamp()
    latest = StockQuoteLog.query_data(
        columns=[StockQuoteLog.time], return_type="df", limit=1, order=StockQuoteLog.time.desc()
    )
    # 🧠 ML Signal: Querying data with dynamic filters
    latest_time = int(latest["time"][0])
    print(latest_time)

    delay = (current_time - latest_time) / (60 * 1000)
    if delay > 2:
        logger.warning(f"delay {delay} minutes")

    # ✅ Best Practice: Sorting data for consistent processing
    # interval minutes
    start_time = latest_time - (interval * 60 * 1000)
    # ✅ Best Practice: Dropping duplicates to ensure unique entity processing
    filters = [StockQuoteLog.time > start_time]
    df = StockQuoteLog.query_data(
        # 🧠 ML Signal: Mapping entities to specific attributes
        # ⚠️ SAST Risk (Low): Direct print statements can expose data in production environments
        # ✅ Best Practice: Using __all__ to define public API of the module
        filters=filters, columns=[StockQuoteLog.entity_id, StockQuoteLog.time, StockQuoteLog.price], return_type="df"
    )
    if pd_is_not_null(df):
        df.sort_values(by=["entity_id", "time"], inplace=True)

        g_df = df.groupby("entity_id").agg(
            first_price=("price", "first"),
            last_price=("price", "last"),
            last_time=("time", "last"),
            change_pct=("price", lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]),
        )
        print(g_df.sort_values(by=["change_pct"]))
        up = g_df[g_df["change_pct"] > up_change_pct]
        down = g_df[g_df["change_pct"] < down_change_pct]
        return up.index.tolist(), down.index.tolist()


def get_top_vol(
    entity_ids,
    target_date=None,
    limit=500,
    provider="qmt",
):
    if provider == "qmt":
        df = StockQuote.query_data(
            entity_ids=entity_ids,
            columns=[StockQuote.entity_id],
            order=StockQuote.turnover.desc(),
            limit=limit,
        )
        return df["entity_id"].to_list()
    else:
        if not target_date:
            target_date = get_latest_kdata_date(provider="em", entity_type="stock", adjust_type=AdjustType.hfq)
        df = Stock1dHfqKdata.query_data(
            provider="em",
            filters=[Stock1dHfqKdata.timestamp == to_pd_timestamp(target_date)],
            entity_ids=entity_ids,
            columns=[Stock1dHfqKdata.entity_id],
            order=Stock1dHfqKdata.turnover.desc(),
            limit=limit,
        )
        return df["entity_id"].to_list()


def get_top_down_today(n=100):
    df = StockQuote.query_data(columns=[StockQuote.entity_id], order=StockQuote.change_pct.asc(), limit=n)
    if pd_is_not_null(df):
        return df["entity_id"].to_list()


def get_limit_down_today():
    df = StockQuote.query_data(filters=[StockQuote.is_limit_down], columns=[StockQuote.entity_id])
    if pd_is_not_null(df):
        return df["entity_id"].to_list()


def get_high_days_count(entity_ids=None, target_date=current_date(), days_count=10, high_days_count=None):
    recent_days = get_recent_trade_dates(target_date=target_date, days_count=days_count)

    if recent_days:
        filters = [LimitUpInfo.timestamp >= recent_days[0]]
    else:
        filters = [LimitUpInfo.timestamp >= target_date]

    if high_days_count:
        filters = filters + [LimitUpInfo.high_days_count >= high_days_count]

    df = LimitUpInfo.query_data(
        entity_ids=entity_ids,
        filters=filters,
        columns=[LimitUpInfo.timestamp, LimitUpInfo.entity_id, LimitUpInfo.high_days, LimitUpInfo.high_days_count],
    )
    df_sorted = df.sort_values(by=["entity_id", "timestamp"])
    df_latest = df_sorted.drop_duplicates(subset="entity_id", keep="last").reset_index(drop=True)

    entity_id_to_high_days_map = df_latest.set_index("entity_id")["high_days"].to_dict()
    return entity_id_to_high_days_map


if __name__ == "__main__":
    # stocks = get_top_vol(entity_ids=None, provider="em")
    # assert len(stocks) == 500
    # Index1dKdata.record_data(provider="em",sleeping_time=0)
    # print(get_recent_trade_dates(days_count=10))
    print(get_high_days_count(days_count=3, high_days_count=3))


# the __all__ is generated
__all__ = [
    "get_entity_ids_by_filter",
    "get_limit_up_stocks",
    "get_dragon_and_tigger_player",
    "get_big_players",
    "get_player_performance",
    "get_player_success_rate",
    "get_players",
    "get_good_players",
    "get_entity_list_by_cap",
    "get_big_cap_stock",
    "get_middle_cap_stock",
    "get_small_cap_stock",
    "get_mini_cap_stock",
    "get_mini_and_small_stock",
    "get_middle_and_big_stock",
]