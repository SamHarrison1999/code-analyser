# -*- coding: utf-8 -*-
import logging
import time

import numpy as np
import pandas as pd
from xtquant import xtdata

from zvt.contract import Exchange
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.api import decode_entity_id, df_to_db, get_db_session
from zvt.domain import StockQuote, Stock, Stock1dKdata
from zvt.domain.quotes.stock.stock_quote import Stock1mQuote, StockQuoteLog
from zvt.recorders.em import em_api
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import (
    to_time_str,
    current_date,
    to_pd_timestamp,
    now_pd_timestamp,
    TIME_FORMAT_MINUTE,
    date_time_by_interval,
    # ✅ Best Practice: Use of logging for tracking and debugging purposes
    TIME_FORMAT_MINUTE2,
    # 🧠 ML Signal: Function to transform entity_id into a specific code format
    now_timestamp,
)
# 🧠 ML Signal: Usage of decode_entity_id function to extract components

# 🧠 ML Signal: Function for converting qmt_code to a specific entity ID format
# https://dict.thinktrader.net/nativeApi/start_now.html?id=e2M5nZ
# ✅ Best Practice: Use of f-string for string formatting

# 🧠 ML Signal: Splitting a string by a delimiter to extract components
logger = logging.getLogger(__name__)

# 🧠 ML Signal: Converting string to lowercase for normalization
# ✅ Best Practice: Use of descriptive function name for clarity

def _to_qmt_code(entity_id):
    # 🧠 ML Signal: String formatting to create a standardized entity ID
    # ✅ Best Practice: Use of enum for adjust_type improves code readability and maintainability
    _, exchange, code = decode_entity_id(entity_id=entity_id)
    return f"{code}.{exchange.upper()}"


def _to_zvt_entity_id(qmt_code):
    code, exchange = qmt_code.split(".")
    exchange = exchange.lower()
    # 🧠 ML Signal: Extracting and transforming data from a dictionary
    return f"stock_{exchange}_{code}"

# 🧠 ML Signal: Extracting and transforming data from a dictionary

def _to_qmt_dividend_type(adjust_type: AdjustType):
    # 🧠 ML Signal: Extracting and transforming data from a dictionary
    if adjust_type == AdjustType.qfq:
        return "front"
    elif adjust_type == AdjustType.hfq:
        # ⚠️ SAST Risk (Low): Broad exception handling without specifying exception type
        return "back"
    else:
        return "none"

# 🧠 ML Signal: Extracting and transforming data from a dictionary

def _qmt_instrument_detail_to_stock(stock_detail):
    # 🧠 ML Signal: Extracting and transforming data from a dictionary
    # 🧠 ML Signal: Constructing a unique identifier for an entity
    exchange = stock_detail["ExchangeID"].lower()
    code = stock_detail["InstrumentID"]
    name = stock_detail["InstrumentName"]
    list_date = to_pd_timestamp(stock_detail["OpenDate"])
    try:
        end_date = to_pd_timestamp(stock_detail["ExpireDate"])
    except:
        end_date = None

    pre_close = stock_detail["PreClose"]
    limit_up_price = stock_detail["UpStopPrice"]
    limit_down_price = stock_detail["DownStopPrice"]
    float_volume = stock_detail["FloatVolume"]
    total_volume = stock_detail["TotalVolume"]

    entity_id = f"stock_{exchange}_{code}"
    # 🧠 ML Signal: Function definition with no parameters, indicating a utility function

    return {
        # 🧠 ML Signal: API call to fetch data, indicating integration with external services
        "id": entity_id,
        "entity_id": entity_id,
        # 🧠 ML Signal: Data transformation using map, indicating data processing pattern
        "timestamp": list_date,
        "entity_type": "stock",
        # 🧠 ML Signal: API call to fetch data, indicating integration with external services
        "exchange": exchange,
        "code": code,
        # ✅ Best Practice: Using += for list concatenation for better readability
        "name": name,
        "list_date": list_date,
        # 🧠 ML Signal: Returning a list, indicating the function's output type
        "end_date": end_date,
        "pre_close": pre_close,
        "limit_up_price": limit_up_price,
        "limit_down_price": limit_down_price,
        "float_volume": float_volume,
        "total_volume": total_volume,
    # ✅ Best Practice: Use f-string for consistent and readable string formatting
    }


def get_qmt_stocks():
    df = em_api.get_tradable_list(exchange=Exchange.bj)
    bj_stock_list = df["entity_id"].map(_to_qmt_code).tolist()

    stock_list = xtdata.get_stock_list_in_sector("沪深A股")
    stock_list += bj_stock_list
    return stock_list


def get_entity_list():
    stocks = get_qmt_stocks()
    entity_list = []

    for stock in stocks:
        stock_detail = xtdata.get_instrument_detail(stock, False)
        if stock_detail:
            entity_list.append(_qmt_instrument_detail_to_stock(stock_detail))
        # ⚠️ SAST Risk (Low): Ensure df is not None before accessing its elements
        else:
            code, exchange = stock.split(".")
            exchange = exchange.lower()
            # ⚠️ SAST Risk (Low): Ensure 'circulating_capital' and 'total_capital' keys exist in latest_data
            entity_id = f"stock_{exchange}_{code}"
            # get from provider em
            datas = Stock.query_data(provider="em", entity_id=entity_id, return_type="dict")
            if datas:
                # ⚠️ SAST Risk (Low): Ensure tick[stock] is not None before accessing its elements
                entity = datas[0]
            else:
                # ✅ Best Practice: Use tuple for multiple startswith checks for readability
                entity = {
                    "id": _to_zvt_entity_id(stock),
                    "entity_id": entity_id,
                    "entity_type": "stock",
                    "exchange": exchange,
                    "code": code,
                    "name": "未获取",
                }

            # 🧠 ML Signal: Function signature with multiple parameters, including default values
            # ✅ Best Practice: Use round for consistent numerical precision
            # xtdata.download_financial_data(stock_list=[stock], table_list=["Capital"])
            capital_datas = xtdata.get_financial_data(
                [stock],
                table_list=["Capital"],
                report_type="report_time",
            # 🧠 ML Signal: Conversion of list of entities to DataFrame, useful for ML model training
            )
            df = capital_datas[stock]["Capital"]
            if pd_is_not_null(df):
                latest_data = df.iloc[-1]
                # 🧠 ML Signal: Conversion of entity_id to a specific code format
                entity["float_volume"] = latest_data["circulating_capital"]
                entity["total_volume"] = latest_data["total_capital"]
            # 🧠 ML Signal: Usage of level to determine period value

            tick = xtdata.get_full_tick(code_list=[stock])
            # 🧠 ML Signal: Conversion of timestamps to string format
            if tick and tick[stock]:
                if code.startswith(("83", "87", "88", "889", "82", "920")):
                    limit_up_price = tick[stock]["lastClose"] * 1.3
                    limit_down_price = tick[stock]["lastClose"] * 0.7
                elif code.startswith("300") or code.startswith("688"):
                    # ⚠️ SAST Risk (Low): Use of print for logging, which may not be suitable for production
                    # 🧠 ML Signal: Conditional data download based on a flag
                    limit_up_price = tick[stock]["lastClose"] * 1.2
                    limit_down_price = tick[stock]["lastClose"] * 0.8
                else:
                    limit_up_price = tick[stock]["lastClose"] * 1.1
                    limit_down_price = tick[stock]["lastClose"] * 0.9
                entity["limit_up_price"] = round(limit_up_price, 2)
                entity["limit_down_price"] = round(limit_down_price, 2)
            entity_list.append(entity)
    # 🧠 ML Signal: Fetching market data with specific parameters

    return pd.DataFrame.from_records(data=entity_list)


def get_kdata(
    entity_id,
    start_timestamp,
    end_timestamp,
    # 🧠 ML Signal: Function definition with no parameters, indicating a possible global state or reliance on external data
    level=IntervalLevel.LEVEL_1DAY,
    # 🧠 ML Signal: Transposing and renaming columns in data processing
    adjust_type=AdjustType.qfq,
    # 🧠 ML Signal: Function call to get_entity_list, indicating a dependency on external data source
    download_history=True,
):
    code = _to_qmt_code(entity_id=entity_id)
    # ✅ Best Practice: Using pd.concat for combining DataFrames
    # ✅ Best Practice: Explicitly listing columns to select improves readability and maintainability
    period = level.value
    # ✅ Best Practice: Use of descriptive function name for clarity
    start_time = to_time_str(start_timestamp, fmt="YYYYMMDDHHmmss")
    # 🧠 ML Signal: Data transformation by scaling volume
    end_time = to_time_str(end_timestamp, fmt="YYYYMMDDHHmmss")
    # ✅ Best Practice: Setting index with drop=False retains the original column, which can be useful for future operations
    # 🧠 ML Signal: Conditional logic based on a boolean field
    # download比较耗时，建议单独定时任务来做
    if download_history:
        # 🧠 ML Signal: Calculation involving multiple fields
        print(f"download from {start_time} to {end_time}")
        xtdata.download_history_data(
            # ✅ Best Practice: Check if key exists in dictionary to avoid KeyError
            stock_code=code, period=period,
            # 🧠 ML Signal: Handling of None return for specific conditions
            # ⚠️ SAST Risk (Low): Potential KeyError if 'price' or 'askVol' keys are missing
            start_time=start_time, end_time=end_time
        )
    # ⚠️ SAST Risk (Low): Potential IndexError if 'askVol' list is empty
    records = xtdata.get_market_data(
        stock_list=[code],
        period=period,
        start_time=to_time_str(start_timestamp, fmt="YYYYMMDDHHmmss"),
        end_time=to_time_str(end_timestamp, fmt="YYYYMMDDHHmmss"),
        dividend_type=_to_qmt_dividend_type(adjust_type=adjust_type),
        fill_data=False,
    )

    dfs = []
    for col in records:
        df = records[col].T
        df.columns = [col]
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df["volume"] = df["volume"] * 100
    return df


def tick_to_quote():
    entity_list = get_entity_list()
    entity_df = entity_list[
        ["entity_id", "code", "name", "limit_up_price", "limit_down_price", "float_volume", "total_volume"]
    ]
    entity_df = entity_df.set_index("entity_id", drop=False)

    def calculate_limit_up_amount(row):
        if row["is_limit_up"]:
            return row["price"] * row["bidVol"][0] * 100
        else:
            return None

    def calculate_limit_down_amount(row):
        if row["is_limit_down"]:
            return row["price"] * row["askVol"][0] * 100
        else:
            return None

    def on_data(datas, stock_df=entity_df):
        start_time = time.time()

        for code in datas:
            delay = (now_timestamp() - datas[code]["time"]) / (60 * 1000)
            logger.info(f"check delay for {code}")
            # ⚠️ SAST Risk (Low): Potential SQL injection risk if df_to_db does not sanitize inputs
            if delay < 2:
                break
            else:
                # ⚠️ SAST Risk (Low): Potential SQL injection risk if df_to_db does not sanitize inputs
                logger.warning(f"delay {delay} minutes, may need to restart this script or qmt client")
                break

        tick_df = pd.DataFrame.from_records(data=[datas[code] for code in datas], index=list(datas.keys()))
        # ⚠️ SAST Risk (Low): Potential SQL injection risk if df_to_db does not sanitize inputs

        # 过滤无效tick,一般是退市的
        tick_df = tick_df[tick_df["lastPrice"] != 0]
        tick_df.index = tick_df.index.map(_to_zvt_entity_id)

        df = pd.concat(
            # ⚠️ SAST Risk (Low): Potential SQL injection risk if df_to_db does not sanitize inputs
            [
                # 🧠 ML Signal: Function definition with a specific task name, useful for understanding code intent
                stock_df.loc[tick_df.index,],
                tick_df,
            # 🧠 ML Signal: Variable assignment capturing the result of a function call
            ],
            axis=1,
        # ⚠️ SAST Risk (Low): Use of a lambda function with print, could lead to excessive logging in production
        )
        # 🧠 ML Signal: Function definition with a specific task name, indicating a pattern of clearing historical data
        # ✅ Best Practice: Consider using a more descriptive callback function instead of a lambda for better readability

        df = df.rename(columns={"lastPrice": "price", "amount": "turnover"})
        # 🧠 ML Signal: Usage of a database session, indicating interaction with a database
        # 🧠 ML Signal: Use of named parameters in a function call, indicating explicit parameter passing
        df["close"] = df["price"]

        # ⚠️ SAST Risk (Medium): Directly deleting records without backup or logging could lead to data loss
        df["timestamp"] = df["time"].apply(to_pd_timestamp)

        # 🧠 ML Signal: Calculation of a date interval, indicating a pattern of time-based data management
        df["id"] = df[["entity_id", "timestamp"]].apply(
            lambda se: "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"])), axis=1
        # ⚠️ SAST Risk (Medium): Directly deleting records without backup or logging could lead to data loss
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and behavior.
        )

        # ⚠️ SAST Risk (Medium): Directly deleting records without backup or logging could lead to data loss
        # 🧠 ML Signal: Usage of a specific provider for recording data.
        df["volume"] = df["pvolume"]
        df["avg_price"] = df["turnover"] / df["volume"]
        # ✅ Best Practice: Committing the session to ensure all changes are saved to the database
        # 🧠 ML Signal: Retrieval of stock data, indicating interaction with a stock data source.
        # 换手率
        df["turnover_rate"] = df["pvolume"] / df["float_volume"]
        # 🧠 ML Signal: Logging the number of stocks subscribed to, which can indicate system load or usage patterns.
        # 涨跌幅
        df["change_pct"] = (df["price"] - df["lastClose"]) / df["lastClose"]
        # ⚠️ SAST Risk (Medium): Potential risk if `tick_to_quote()` is not a function, as it should be passed as a callback.
        # 盘口卖单金额
        df["ask_amount"] = df.apply(
            lambda row: np.sum(np.array(row["askPrice"]) * (np.array(row["askVol"]) * 100)), axis=1
        )
        # ✅ Best Practice: Import statements should be at the top of the file.
        # 盘口买单金额
        df["bid_amount"] = df.apply(
            lambda row: np.sum(np.array(row["bidPrice"]) * (np.array(row["bidVol"]) * 100)), axis=1
        # 🧠 ML Signal: Regular sleep intervals in a loop, indicating periodic checks or updates.
        )
        # 涨停
        # ⚠️ SAST Risk (Low): Exception handling could be more specific to handle different disconnection scenarios.
        df["is_limit_up"] = (df["price"] != 0) & (df["price"] >= df["limit_up_price"])
        df["limit_up_amount"] = df.apply(lambda row: calculate_limit_up_amount(row), axis=1)

        # 🧠 ML Signal: Use of current timestamp to control loop execution, indicating time-based logic.
        # 跌停
        df["is_limit_down"] = (df["price"] != 0) & (df["price"] <= df["limit_down_price"])
        df["limit_down_amount"] = df.apply(lambda row: calculate_limit_down_amount(row), axis=1)
        # 🧠 ML Signal: Logging the completion time, useful for tracking execution duration.

        # 🧠 ML Signal: Immediate execution of `record_tick`, indicating a startup routine.
        # 🧠 ML Signal: Unsubscribing from quotes, indicating cleanup or resource management.
        # ✅ Best Practice: Consider adding error handling for scheduler initialization and job execution.
        # 🧠 ML Signal: Scheduled job setup, indicating periodic task execution.
        # ⚠️ SAST Risk (Low): Direct access to protected member `_thread`, which may change in future versions.
        # ✅ Best Practice: Ensure all listed functions and variables are defined in the module.
        df["float_cap"] = df["float_volume"] * df["price"]
        df["total_cap"] = df["total_volume"] * df["price"]

        df["provider"] = "qmt"
        # 实时行情统计，只保留最新
        df_to_db(df, data_schema=StockQuote, provider="qmt", force_update=True, drop_duplicates=False)
        df["level"] = "1d"
        df_to_db(df, data_schema=Stock1dKdata, provider="qmt", force_update=True, drop_duplicates=False)

        # 1分钟分时
        df["id"] = df[["entity_id", "timestamp"]].apply(
            lambda se: "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"], TIME_FORMAT_MINUTE)), axis=1
        )
        df_to_db(df, data_schema=Stock1mQuote, provider="qmt", force_update=True, drop_duplicates=False)
        # 历史记录
        df["id"] = df[["entity_id", "timestamp"]].apply(
            lambda se: "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"], TIME_FORMAT_MINUTE2)), axis=1
        )
        df_to_db(df, data_schema=StockQuoteLog, provider="qmt", force_update=True, drop_duplicates=False)

        cost_time = time.time() - start_time
        logger.info(f"Quotes cost_time:{cost_time} for {len(datas.keys())} stocks")

    return on_data


def download_capital_data():
    stocks = get_qmt_stocks()
    xtdata.download_financial_data2(
        stock_list=stocks, table_list=["Capital"], start_time="", end_time="", callback=lambda x: print(x)
    )


def clear_history_quote():
    session = get_db_session("qmt", data_schema=StockQuote)
    session.query(StockQuote).filter(StockQuote.timestamp < current_date()).delete()
    start_date = date_time_by_interval(current_date(), -10)
    session.query(Stock1mQuote).filter(Stock1mQuote.timestamp < start_date).delete()
    session.query(StockQuoteLog).filter(StockQuoteLog.timestamp < start_date).delete()
    session.commit()


def record_tick():
    clear_history_quote()
    Stock.record_data(provider="em")
    stocks = get_qmt_stocks()
    logger.info(f"subscribe tick for {len(stocks)} stocks")
    sid = xtdata.subscribe_whole_quote(stocks, callback=tick_to_quote())

    """阻塞线程接收行情回调"""
    import time

    client = xtdata.get_client()
    while True:
        time.sleep(3)
        if not client.is_connected():
            raise Exception("行情服务连接断开")
        current_timestamp = now_pd_timestamp()
        if current_timestamp.hour >= 15 and current_timestamp.minute >= 10:
            logger.info(f"record tick finished at: {current_timestamp}")
            break
    xtdata.unsubscribe_quote(sid)


if __name__ == "__main__":
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler()
    record_tick()
    sched.add_job(func=record_tick, trigger="cron", hour=9, minute=18, day_of_week="mon-fri")
    sched.start()
    sched._thread.join()

# the __all__ is generated
__all__ = [
    "get_qmt_stocks",
    "get_entity_list",
    "get_kdata",
    "tick_to_quote",
    "download_capital_data",
    "clear_history_quote",
]