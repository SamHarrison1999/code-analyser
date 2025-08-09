# -*- coding: utf-8 -*-
import logging
import random
from typing import Union

import demjson3
import pandas as pd
import requests

# ✅ Best Practice: Grouping imports by standard library, third-party, and local modules improves readability.
import sqlalchemy
from requests import Session

from zvt.api.kdata import generate_kdata_id
from zvt.api.utils import value_to_pct, china_stock_code_to_id
from zvt.contract import (
    ActorType,
    AdjustType,
    IntervalLevel,
    Exchange,
    TradableType,
    get_entity_exchanges,
    tradable_type_map_exchanges,
)
from zvt.contract.api import decode_entity_id, df_to_db
from zvt.domain import BlockCategory, StockHotTopic
from zvt.recorders.consts import DEFAULT_HEADER
from zvt.utils.time_utils import (
    to_pd_timestamp,
    now_timestamp,
    to_time_str,
    current_date,
    now_pd_timestamp,
    # 🧠 ML Signal: Use of logging indicates tracking and debugging practices.
    # 🧠 ML Signal: Function signature and parameters can be used to understand API usage patterns
)

# 🧠 ML Signal: API call pattern with specific parameters
from zvt.utils.utils import to_float, json_callback_param

logger = logging.getLogger(__name__)


# 获取中美国债收益率
def get_treasury_yield(pn=1, ps=2000, fetch_all=True):
    results = get_em_data(
        request_type="RPTA_WEB_TREASURYYIELD",
        source=None,
        fields="ALL",
        sort_by="SOLAR_DATE",
        # ✅ Best Practice: Initialize an empty list before appending items
        sort="desc",
        # ✅ Best Practice: Use descriptive variable names for clarity
        # 🧠 ML Signal: Pattern of constructing dictionary with specific keys
        pn=pn,
        ps=ps,
        fetch_all=fetch_all,
    )
    yields = []
    for item in results:
        date = item["SOLAR_DATE"]
        # 中国
        yields.append(
            {
                "id": f"country_galaxy_CN_{to_time_str(date)}",
                "entity_id": "country_galaxy_CN",
                # 🧠 ML Signal: Use of f-string for dynamic string formatting
                # 🧠 ML Signal: Conversion of date to timestamp
                # ⚠️ SAST Risk (Low): Potential for missing keys in item dictionary
                "timestamp": to_pd_timestamp(date),
                "code": "CN",
                "yield_2": item.get("EMM00588704"),
                "yield_5": item.get("EMM00166462"),
                "yield_10": item.get("EMM00166466"),
                "yield_30": item.get("EMM00166469"),
            }
        )
        yields.append(
            {
                "id": f"country_galaxy_US_{to_time_str(date)}",
                "entity_id": "country_galaxy_US",
                # 🧠 ML Signal: Use of f-string for dynamic string formatting
                "timestamp": to_pd_timestamp(date),
                # 🧠 ML Signal: Conversion of date to timestamp
                # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
                "code": "US",
                # ⚠️ SAST Risk (Low): Potential for missing keys in item dictionary
                # 🧠 ML Signal: Function call with keyword arguments, useful for learning API usage patterns
                "yield_2": item.get("EMG00001306"),
                "yield_5": item.get("EMG00001308"),
                "yield_10": item.get("EMG00001310"),
                "yield_30": item.get("EMG00001312"),
            }
            # 🧠 ML Signal: Function call with dynamic argument, useful for learning how functions are composed
        )
    return yields


# 🧠 ML Signal: Function with date range parameters, common in time-series data processing


# ✅ Best Practice: Return statement at the end of the function
# ✅ Best Practice: Convert date to string format for consistency
# 机构持仓日期
def get_ii_holder_report_dates(code):
    # ✅ Best Practice: Default value handling for optional parameters
    return get_em_data(
        # ✅ Best Practice: Convert date to string format for consistency
        # 🧠 ML Signal: Use of current timestamp as a default value
        request_type="RPT_F10_MAIN_ORGHOLD",
        fields="REPORT_DATE,IS_COMPLETE",
        filters=generate_filters(code=code),
        sort_by="REPORT_DATE",
        sort="desc",
        # 🧠 ML Signal: Function call with multiple parameters, indicating complex data retrieval
    )


# 🧠 ML Signal: Use of specific request type for data retrieval


# 🧠 ML Signal: Function name suggests a specific domain-related operation, useful for domain-specific model training
def get_dragon_and_tiger_list(start_date, end_date=None):
    # 🧠 ML Signal: Specifying data source, indicating data provenance
    # ⚠️ SAST Risk (Low): Potential for SQL injection if filters are not properly sanitized
    # 🧠 ML Signal: Requesting all fields, indicating comprehensive data usage
    # ✅ Best Practice: Use of descriptive parameter names improves code readability
    # 🧠 ML Signal: Hardcoded request type indicates a specific API usage pattern
    start_date = to_time_str(start_date)
    if not end_date:
        end_date = now_timestamp()
    end_date = to_time_str(end_date)
    return get_em_data(
        # 🧠 ML Signal: Specific fields requested can indicate data importance or relevance
        request_type="RPT_DAILYBILLBOARD_DETAILS",
        # 🧠 ML Signal: Sorting by multiple fields, indicating data organization preferences
        fields="ALL",
        # 🧠 ML Signal: Use of a filter function indicates a pattern of data selection
        source="DataCenter",
        # 🧠 ML Signal: Function name suggests a pattern of retrieving report dates for a holder
        # 🧠 ML Signal: Ascending sort order, indicating preference for chronological data
        filters=f"(TRADE_DATE>='{start_date}')(TRADE_DATE<='{end_date}')",
        # ⚠️ SAST Risk (Low): Potential risk if params are constructed from untrusted input
        # ✅ Best Practice: Using named parameters improves readability
        # 🧠 ML Signal: Sorting preferences can indicate data processing patterns
        # ✅ Best Practice: Using a descriptive function name for `get_em_data` indicates its purpose
        sort_by="TRADE_DATE,SECURITY_CODE",
        sort="asc,asc",
    )


# 龙虎榜
# 🧠 ML Signal: Use of a helper function to generate filters based on code
def get_dragon_and_tiger(code, start_date=None):
    # 🧠 ML Signal: Function definition with parameters indicating a pattern for data retrieval
    return get_em_data(
        # 🧠 ML Signal: Usage of a specific request type indicating a pattern for data retrieval
        # ✅ Best Practice: Using named arguments improves readability and maintainability
        request_type="RPT_OPERATEDEPT_TRADE",
        fields="TRADE_ID,TRADE_DATE,EXPLANATION,SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,ACCUM_AMOUNT,CHANGE_RATE,NET_BUY,BUY_BUY_TOTAL,BUY_SELL_TOTAL,BUY_RATIO_TOTAL,SELL_BUY_TOTAL,SELL_SELL_TOTAL,SELL_RATIO_TOTAL,TRADE_DIRECTION,RANK,OPERATEDEPT_NAME,BUY_AMT_REAL,SELL_AMT_REAL,BUY_RATIO,SELL_RATIO,BUY_TOTAL,SELL_TOTAL,BUY_TOTAL_NET,SELL_TOTAL_NET,NET",
        filters=generate_filters(
            code=code, trade_date=start_date, field_op={"trade_date": ">="}
        ),
        params='(groupField=TRADE_ID)(groupedFields=TRADE_DIRECTION,RANK,OPERATEDEPT_NAME,BUY_AMT_REAL,SELL_AMT_REAL,BUY_RATIO,SELL_RATIO,NET")(groupListName="LIST")',
        sort_by="TRADE_DATE,RANK",
        sort="asc,asc",
        # 🧠 ML Signal: Function call with dynamic filters based on input parameter
    )


# 🧠 ML Signal: Function definition with a specific pattern of parameters

# ✅ Best Practice: Sorting parameters are explicitly defined, improving clarity
# 🧠 ML Signal: Usage of a specific API call pattern


# 十大股东持仓日期
def get_holder_report_dates(code):
    # 🧠 ML Signal: Use of hardcoded request type indicating a specific data retrieval pattern
    return get_em_data(
        request_type="RPT_F10_EH_HOLDERSDATE",
        # 🧠 ML Signal: Use of specific fields indicating a pattern in data selection
        # 🧠 ML Signal: Function with multiple parameters, indicating a pattern of data processing or transformation
        fields="END_DATE,IS_DEFAULT,IS_REPORTDATE",
        # 🧠 ML Signal: Use of a specific request type, indicating a pattern of data retrieval
        # 🧠 ML Signal: Use of specific fields, indicating a pattern of data selection
        # 🧠 ML Signal: Use of a function to generate filters, indicating a pattern in data filtering
        filters=generate_filters(code=code),
        sort_by="END_DATE",
        sort="desc",
    )


# 🧠 ML Signal: Function with multiple parameters, indicating a pattern for ML models to learn parameter usage.


# 🧠 ML Signal: Use of a filter generation function, indicating a pattern of data filtering
# 🧠 ML Signal: Usage of a specific request type string, which could be a pattern for ML models to learn API usage.
# ✅ Best Practice: Use of named parameters improves readability and maintainability.
# 十大流通股东日期
def get_free_holder_report_dates(code):
    return get_em_data(
        request_type="RPT_F10_EH_FREEHOLDERSDATE",
        fields="END_DATE,IS_DEFAULT,IS_REPORTDATE",
        # ✅ Best Practice: Use of a single string for fields improves readability and maintainability.
        # 🧠 ML Signal: Function name suggests a specific data retrieval pattern
        filters=generate_filters(code=code),
        # 🧠 ML Signal: Usage of a specific request type for data retrieval
        # 🧠 ML Signal: Function call with parameters, indicating a pattern for ML models to learn function usage.
        # ⚠️ SAST Risk (Low): Hardcoded request type could lead to inflexibility
        sort_by="END_DATE",
        sort="desc",
    )


# 🧠 ML Signal: Specific fields requested indicate data usage patterns
# https://datacenter.eastmoney.com/securities/api/data/get?type=RPT_F10_EH_RELATION&sty=SECUCODE%2CHOLDER_NAME%2CRELATED_RELATION%2CHOLD_RATIO&filter=(SECUCODE%3D%22601162.SH%22)&client=APP&source=SECURITIES&p=1&ps=200&rdm=rnd_01BE6995104944ED99B70EEB7FFC0353&v=012649539724458458
# https://datacenter.eastmoney.com/securities/api/data/get?type=RPT_F10_FREE_TOTALHOLDNUM&sty=SECUCODE%2CSECURITY_CODE%2CEND_DATE%2CHOLD_NUM_COUNT%2CHOLD_RATIO_COUNT%2CHOLD_RATIO_CHANGE&filter=(SECUCODE%3D%22601162.SH%22)(END_DATE%3D%272024-09-30%27)&client=APP&source=SECURITIES&p=1&ps=200&sr=1&st=&rdm=rnd_FA1943FA30474E3AA0CCF206EA1B5749&v=032098454407366983
# 🧠 ML Signal: Function parameter 'code' indicates a pattern of processing financial or stock data
# ✅ Best Practice: Use of a helper function to generate filters improves readability
def get_controlling_shareholder(code):
    return get_em_data(
        # 🧠 ML Signal: Sorting by a specific field indicates importance of data order
        request_type="RPT_F10_EH_RELATION",
        # 🧠 ML Signal: Use of 'datas[0]' suggests a pattern of accessing the most recent or first element
        fields="SECUCODE,CHOLDER_NAME,CRELATED_RELATION,CHOLD_RATIO",
        filters=generate_filters(code=code),
    )


# 🧠 ML Signal: 'request_type' and 'fields' parameters indicate a pattern of querying specific data
# 机构持仓
# 🧠 ML Signal: Use of 'generate_filters' suggests a pattern of filtering data based on parameters
def get_ii_holder(code, report_date, org_type):
    return get_em_data(
        request_type="RPT_MAIN_ORGHOLDDETAIL",
        fields="SECURITY_CODE,REPORT_DATE,HOLDER_CODE,HOLDER_NAME,TOTAL_SHARES,HOLD_VALUE,FREESHARES_RATIO,ORG_TYPE,SECUCODE,FUND_DERIVECODE",
        # 🧠 ML Signal: Accessing 'holders[0]' indicates a pattern of using the first result from a query
        filters=generate_filters(code=code, report_date=report_date, org_type=org_type),
    )


# 机构持仓汇总
def get_ii_summary(code, report_date, org_type):
    # ⚠️ SAST Risk (Low): Potential division by zero if 'HOLD_RATIO_COUNT' is zero
    return get_em_data(
        request_type="RPT_F10_MAIN_ORGHOLDDETAILS",
        fields="SECURITY_CODE,SECUCODE,REPORT_DATE,ORG_TYPE,TOTAL_ORG_NUM,TOTAL_FREE_SHARES,TOTAL_MARKET_CAP,TOTAL_SHARES_RATIO,CHANGE_RATIO,IS_COMPLETE",
        filters=generate_filters(code=code, report_date=report_date, org_type=org_type),
    )


# ⚠️ SAST Risk (Low): Catching broad exceptions can hide unexpected errors


# 🧠 ML Signal: Function to retrieve controlling shareholder data based on a code
# ✅ Best Practice: Use logging with exception information for better debugging
def get_free_holders(code, end_date):
    return get_em_data(
        request_type="RPT_F10_EH_FREEHOLDERS",
        fields="SECUCODE,END_DATE,HOLDER_NAME,HOLDER_CODE,HOLDER_CODE_OLD,HOLD_NUM,FREE_HOLDNUM_RATIO,FREE_RATIO_QOQ,IS_HOLDORG,HOLDER_RANK",
        filters=generate_filters(code=code, end_date=end_date),
        # 🧠 ML Signal: Returning a dictionary with specific keys indicates a pattern of structured data output
        # ✅ Best Practice: Using a helper function to generate filters improves code readability and maintainability
        sort_by="HOLDER_RANK",
    )


# ✅ Best Practice: Initializing a dictionary to store control information


def get_top_ten_free_holder_stats(code):
    datas = get_holder_report_dates(code=code)
    # 🧠 ML Signal: Pattern of checking specific relationship types in data
    if datas:
        end_date = to_time_str(datas[0]["END_DATE"])
        holders = get_em_data(
            request_type="RPT_F10_FREE_TOTALHOLDNUM",
            # 🧠 ML Signal: Function definition with parameters indicating a pattern of data retrieval based on code and date
            fields="SECUCODE,SECURITY_CODE,END_DATE,HOLD_NUM_COUNT,HOLD_RATIO_COUNT,HOLD_RATIO_CHANGE,",
            # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if request_type is not validated
            # ⚠️ SAST Risk (Low): Potential risk if "HOLD_RATIO" is not a number; consider validating input
            # 🧠 ML Signal: Use of a specific request type indicating a pattern of accessing holder data
            filters=generate_filters(code=code, end_date=end_date),
        )
        if holders:
            holder = holders[0]
            ratio = 0
            # 🧠 ML Signal: Specific fields requested indicating a pattern of data usage
            change = 0
            try:
                # ✅ Best Practice: Check if 'order' is not None or empty before processing
                # 🧠 ML Signal: Use of a filter function indicating a pattern of data filtering
                if holder["HOLD_RATIO_COUNT"]:
                    # ⚠️ SAST Risk (Low): Potential injection risk if filters are not properly sanitized
                    ratio = holder["HOLD_RATIO_COUNT"] / 100
                # 🧠 ML Signal: Splitting a string by a delimiter to process each element
                if holder["HOLD_RATIO_CHANGE"]:
                    # 🧠 ML Signal: Sorting by a specific field indicating a pattern of data ordering
                    change = holder["HOLD_RATIO_CHANGE"] / 100
            # 🧠 ML Signal: List comprehension used for conditional transformation of list elements
            # ✅ Best Practice: Consider using more descriptive parameter names for better readability.
            except Exception as e:
                logger.warning(f"Wrong holder {holder}", e)
            # ✅ Best Practice: Return the original 'order' if it is None or empty
            # ✅ Best Practice: Use a helper function to encapsulate order parameter logic.

            return {
                # ⚠️ SAST Risk (Low): Use of random number generation without a seed can lead to non-deterministic behavior.
                "code": code,
                "timestamp": end_date,
                # 🧠 ML Signal: Conditional logic based on input parameters.
                "ratio": ratio,
                "change": change,
                # 🧠 ML Signal: URL construction pattern with multiple query parameters.
            }


# 🧠 ML Signal: Function definition with a single responsibility


# 🧠 ML Signal: URL construction pattern with multiple query parameters.
def get_controlling_shareholder(code):
    # ✅ Best Practice: Convert input to integer for consistent comparison
    holders = get_em_data(
        # 🧠 ML Signal: Appending additional parameters to a URL.
        request_type="RPT_F10_EH_RELATION",
        # ✅ Best Practice: Use of chained comparison for readability
        fields="SECUCODE,HOLDER_NAME,RELATED_RELATION,HOLD_RATIO",
        # ✅ Best Practice: Return statement should be the last line of the function for clarity.
        filters=generate_filters(code=code),
    )
    # ✅ Best Practice: Clear and concise conditional logic

    if holders:
        # ✅ Best Practice: Consider using a dictionary for mapping to improve readability and maintainability.
        control = {"ratio": 0}

        for holder in holders:
            if holder["RELATED_RELATION"] == "控股股东":
                control["holder"] = holder["HOLDER_NAME"]
            elif holder["RELATED_RELATION"] == "实际控制人":
                control["parent"] = holder["HOLDER_NAME"]
            if holder["HOLD_RATIO"]:
                control["ratio"] = control["ratio"] + holder["HOLD_RATIO"]
        return control


def get_holders(code, end_date):
    return get_em_data(
        request_type="RPT_F10_EH_HOLDERS",
        # ⚠️ SAST Risk (Low): Using assert for control flow can be disabled with optimization flags.
        fields="SECUCODE,END_DATE,HOLDER_NAME,HOLDER_CODE,HOLDER_CODE_OLD,HOLD_NUM,HOLD_NUM_RATIO,HOLD_RATIO_QOQ,HOLDER_RANK,IS_HOLDORG",
        # ✅ Best Practice: Using list comprehension for filtering and processing items in a dictionary
        filters=generate_filters(code=code, end_date=end_date),
        sort_by="HOLDER_RANK",
    )


# 🧠 ML Signal: Conditional logic based on the presence of 'code'


# ⚠️ SAST Risk (Low): Potential injection risk if 'code' is not properly sanitized
def _order_param(order: str):
    if order:
        orders = order.split(",")
        # 🧠 ML Signal: Conditional logic based on the presence of 'org_type'
        return ",".join(["1" if item == "asc" else "-1" for item in orders])
    return order


def get_url(
    type,
    sty,
    source="SECURITIES",
    filters=None,
    order_by="",
    order="asc",
    pn=1,
    ps=2000,
    params=None,
):
    # 根据 url 映射如下
    # 🧠 ML Signal: Use of a dictionary to determine operation type
    # type=RPT_F10_MAIN_ORGHOLDDETAILS
    # ⚠️ SAST Risk (Low): Potential injection risk if 'value' is not properly sanitized
    # sty=SECURITY_CODE,SECUCODE,REPORT_DATE,ORG_TYPE,TOTAL_ORG_NUM,TOTAL_FREE_SHARES,TOTAL_MARKET_CAP,TOTAL_SHARES_RATIO,CHANGE_RATIO,IS_COMPLETE
    # filter=(SECUCODE="000338.SZ")(REPORT_DATE=\'2021-03-31\')(ORG_TYPE="01")
    # sr=1
    # st=
    sr = _order_param(order=order)
    v = random.randint(1000000000000000, 9000000000000000)

    if filters or source:
        url = f"https://datacenter.eastmoney.com/securities/api/data/get?type={type}&sty={sty}&filter={filters}&client=APP&source={source}&p={pn}&ps={ps}&sr={sr}&st={order_by}&v=0{v}"
    else:
        url = f"https://datacenter.eastmoney.com/api/data/get?type={type}&sty={sty}&st={order_by}&sr={sr}&p={pn}&ps={ps}&_={now_timestamp()}"

    if params:
        url = url + f"&params={params}"
    # 🧠 ML Signal: Function call with multiple parameters, including optional ones

    return url


def get_exchange(code):
    code_ = int(code)
    if 800000 >= code_ >= 600000:
        return "SH"
    elif code_ >= 400000:
        return "BJ"
    else:
        return "SZ"


# 🧠 ML Signal: Logging usage pattern


def actor_type_to_org_type(actor_type: ActorType):
    # ⚠️ SAST Risk (Low): Potential misuse of session object if not properly managed
    if actor_type == ActorType.raised_fund:
        return "01"
    if actor_type == ActorType.qfii:
        # ⚠️ SAST Risk (Low): No timeout specified in requests.get, which can lead to hanging
        return "02"
    if actor_type == ActorType.social_security:
        return "03"
    # ⚠️ SAST Risk (Low): Assumes JSON response without exception handling
    if actor_type == ActorType.broker:
        return "04"
    # ✅ Best Practice: Explicitly closing the response to free up resources
    if actor_type == ActorType.insurance:
        return "05"
    if actor_type == ActorType.trust:
        return "06"
    if actor_type == ActorType.corporation:
        return "07"
    assert False


# 🧠 ML Signal: Recursive function call pattern


def generate_filters(
    code=None,
    trade_date=None,
    report_date=None,
    end_date=None,
    org_type=None,
    field_op: dict = None,
):
    args = [
        item
        for item in locals().items()
        if item[1] and (item[0] not in ("code", "org_type", "field_op"))
    ]

    result = ""
    if code:
        result += f'(SECUCODE="{code}.{get_exchange(code)}")'
    if org_type:
        result += f'(ORG_TYPE="{org_type}")'

    for arg in args:
        field = arg[0]
        value = arg[1]
        if field_op:
            op = field_op.get(field, "=")
        else:
            op = "="
        result += f"({field.upper()}{op}'{value}')"

    return result


# ✅ Best Practice: Using list concatenation
def get_em_data(
    request_type,
    fields,
    session=None,
    # ⚠️ SAST Risk (High): The function does not return any value or perform any operations, leading to potential logical errors.
    # ✅ Best Practice: Consider returning the dictionaries or performing operations on them.
    # ⚠️ SAST Risk (Low): Raises a generic RuntimeError without specific exception handling
    source="SECURITIES",
    filters=None,
    sort_by="",
    sort="asc",
    pn=1,
    ps=2000,
    fetch_all=True,
    fetch_count=1,
    params=None,
):
    url = get_url(
        type=request_type,
        sty=fields,
        source=source,
        filters=filters,
        order_by=sort_by,
        order=sort,
        pn=pn,
        ps=ps,
        params=params,
    )
    logger.debug(f"current url: {url}")
    # ⚠️ SAST Risk (High): The function does not return any value or perform any operations, leading to potential logical errors.
    # ✅ Best Practice: Consider returning the dictionaries or performing operations on them.
    if session:
        resp = session.get(url)
    else:
        resp = requests.get(url)
    if resp.status_code == 200:
        json_result = resp.json()
        resp.close()

        if json_result:
            if json_result.get("result"):
                data: list = json_result["result"]["data"]
                need_next = pn < json_result["result"]["pages"]
            elif json_result.get("data"):
                data: list = json_result["data"]
                need_next = json_result["hasNext"] == 1
            else:
                data = []
                need_next = False
            if fetch_all or fetch_count - 1 > 0:
                if need_next:
                    next_data = get_em_data(
                        session=session,
                        # ⚠️ SAST Risk (High): The function does not return any value or perform any operations, leading to potential logical errors.
                        # ✅ Best Practice: Consider returning the dictionaries or performing operations on them.
                        request_type=request_type,
                        fields=fields,
                        source=source,
                        filters=filters,
                        sort_by=sort_by,
                        sort=sort,
                        pn=pn + 1,
                        ps=ps,
                        fetch_all=fetch_all,
                        fetch_count=fetch_count - 1,
                        params=params,
                    )
                    if next_data:
                        data = data + next_data
                        return data
                    else:
                        return data
                else:
                    return data
            else:
                return data
        return None
    raise RuntimeError(f"request em data code: {resp.status_code}, error: {resp.text}")


# 🧠 ML Signal: Function signature with default parameters


def get_quotes():
    {
        # 市场,2 A股, 3 港股
        "f1": 2,
        # 🧠 ML Signal: URL construction with dynamic parameters
        # 最新价 660/100=6.6
        "f2": 660,
        # 涨幅 2000/10000=20%
        # ⚠️ SAST Risk (Low): Potential misuse of session object
        "f3": 2000,
        # 涨跌额 110/100=1.1
        "f4": 110,
        # ⚠️ SAST Risk (Low): No timeout specified in requests.get
        # 总手
        "f5": 112596,
        # ⚠️ SAST Risk (Low): raise_for_status() can raise an exception
        # 成交额
        "f6": 74313472.2,
        # ⚠️ SAST Risk (Low): Assumes response is always JSON
        # 换手率 239/10000
        "f8": 239,
        # ✅ Best Practice: Explicitly close the response
        # 市盈率 110
        "f9": 11000,
        # code
        "f12": "300175",
        #
        "f13": 0,
        # name
        "f14": "朗源股份",
        "f18": 550,
        "f19": 80,
        "f30": -215,
        # 买入价
        # 🧠 ML Signal: Appending structured data to a list
        "f31": 660,
        # 卖出价
        "f32": None,
        "f125": 0,
        "f139": 5,
        "f148": 1,
        "f152": 2,
    }
    {
        "f1": 2,
        "f2": 1515,
        "f3": 1002,
        "f4": 138,
        "f5": 547165,
        "f6": 804705199.0,
        "f8": 241,
        "f9": 1575,
        "f12": "601233",
        "f13": 1,
        "f14": "桐昆股份",
        "f18": 1377,
        "f19": 2,
        # 🧠 ML Signal: Function to decode entity IDs, indicating a pattern of handling different entity types
        "f30": -1281,
        # 买入价
        # 🧠 ML Signal: Decoding entity IDs to determine type, exchange, and code
        "f31": 1515,
        # 卖出价
        # ✅ Best Practice: Using if-elif-else for clear conditional logic
        "f32": None,
        "f125": 0,
        # 🧠 ML Signal: Conversion of list of dicts to DataFrame
        "f139": 2,
        "f148": 577,
        "f152": 2,
    }
    {
        "f1": 2,
        "f2": 611,
        "f3": 338,
        "f4": 20,
        # ⚠️ SAST Risk (Low): Using assert for control flow can be disabled in production
        "f5": 478746,
        "f6": 293801314.14,
        # 🧠 ML Signal: Constructing data payload for API request
        "f8": 803,
        # ⚠️ SAST Risk (Medium): URL may be vulnerable to injection if `now_timestamp()` is not properly sanitized
        "f9": 2067,
        # ⚠️ SAST Risk (Medium): No exception handling for network request failures
        "f12": "000788",
        # ⚠️ SAST Risk (Medium): No exception handling for network errors
        "f13": 0,
        # ⚠️ SAST Risk (Low): No handling for potential HTTP errors other than raising exceptions
        "f14": "北大医药",
        # ⚠️ SAST Risk (Low): Potential for unhandled HTTP errors if `raise_for_status` is not called
        "f18": 591,
        # ✅ Best Practice: Closing response to free up resources
        "f19": 6,
        # ⚠️ SAST Risk (Medium): `json_callback_param` function may be vulnerable to injection if not properly sanitized
        "f30": -4015,
        # ⚠️ SAST Risk (Low): Assumes JSON response structure without validation
        "f31": 611,
        # ✅ Best Practice: Ensure the response is closed to free up resources
        "f32": 612,
        "f125": 0,
        "f139": 2,
        "f148": 1,
        # 🧠 ML Signal: Pattern of splitting strings to extract multiple values
        "f152": 2,
    }


# quote
# 🧠 ML Signal: Pattern of transforming data using a specific function
# url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?'
# 日线      klt=101
# 周线      klt=102
# 🧠 ML Signal: Usage of enum-like pattern for value conversion
# 月线      klt=103
#
# limit    lmt=2000
#
# 结束时间   end=20500000
#
# ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in optimized mode
# 复权      fqt 0 不复权 1 前复权 2 后复权
#          iscca
# 🧠 ML Signal: Pattern of converting strings to uppercase
#
# 字段
# f51,f52,f53,f54,f55,
# 🧠 ML Signal: Function to calculate limits based on specific code patterns
# ⚠️ SAST Risk (Low): Logging error without handling may expose sensitive information
# timestamp,open,close,high,low
# f56,f57,f58,f59,f60,f61,f62,f63,f64
# volume,turnover,震幅,change_pct,change,turnover_rate
# 🧠 ML Signal: Accessing dictionary values by key
# 深圳
# secid=0.399001&klt=101&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
# 🧠 ML Signal: Pattern of constructing unique identifiers
# 🧠 ML Signal: Pattern matching with specific code prefixes
# secid=0.399001&klt=102&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
# 🧠 ML Signal: Conditional logic based on code patterns
# secid=0.000338&klt=101&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
#
# 港股
# ✅ Best Practice: Use of pandas for structured data manipulation
# 🧠 ML Signal: Pattern matching with specific code prefixes
# ⚠️ SAST Risk (Medium): No exception handling for network request failures
# secid=116.01024&klt=102&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
# ✅ Best Practice: Use a constant or configuration for the URL to improve maintainability
# 美股
# 🧠 ML Signal: Conditional logic based on code patterns
# secid=106.BABA&klt=102&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
# ⚠️ SAST Risk (Medium): No validation or sanitization of the response data
#
# 🧠 ML Signal: Usage of external API for data retrieval
# 上海
# 🧠 ML Signal: Default case for codes not matching specific patterns
# secid=1.512660&klt=101&fqt=1&lmt=66&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1
# ⚠️ SAST Risk (Low): Potential for unhandled HTTP errors
def get_kdata(
    entity_id,
    session=None,
    level=IntervalLevel.LEVEL_1DAY,
    adjust_type=AdjustType.qfq,
    limit=10000,
):
    # ⚠️ SAST Risk (Low): URL construction with user-controlled parameters can lead to SSRF or information disclosure if not properly validated.
    entity_type, exchange, code = decode_entity_id(entity_id)
    # ⚠️ SAST Risk (Medium): Assumes the JSON response structure without validation
    level = IntervalLevel(level)
    # ⚠️ SAST Risk (Medium): No timeout specified for the HTTP request, which can lead to hanging requests.

    # ✅ Best Practice: Use a context manager (with statement) for handling resources like network connections
    sec_id = to_em_sec_id(entity_id)
    # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if the response does not contain expected JSON structure.
    fq_flag = to_em_fq_flag(adjust_type)
    # 🧠 ML Signal: Function returns data from an external API
    level_flag = to_em_level_flag(level)
    # ✅ Best Practice: Ensure the response is closed to free up system resources.
    # f131 结算价
    # f133 持仓
    # 🧠 ML Signal: Usage of pandas DataFrame for data manipulation.
    # 目前未获取
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={sec_id}&klt={level_flag}&fqt={fq_flag}&lmt={limit}&end=20500000&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1"
    # 🧠 ML Signal: Conditional logic based on function parameters.
    # 🧠 ML Signal: DataFrame column selection and renaming.

    if session:
        resp = session.get(url, headers=DEFAULT_HEADER)
    else:
        resp = requests.get(url, headers=DEFAULT_HEADER)
    resp.raise_for_status()
    results = resp.json()
    resp.close()
    data = results["data"]

    kdatas = []

    if data:
        klines = data["klines"]
        name = data["name"]

        for result in klines:
            # "2000-01-28,1005.26,1012.56,1173.12,982.13,3023326,3075552000.00"
            # ✅ Best Practice: Dropping NaN values to ensure data integrity.
            # "2021-08-27,19.39,20.30,20.30,19.25,1688497,3370240912.00,5.48,6.01,1.15,3.98,0,0,0"
            # time,open,close,high,low,volume,turnover
            # ✅ Best Practice: Filtering out invalid data entries.
            # "2022-04-13,10708,10664,10790,10638,402712,43124771328,1.43,0.57,60,0.00,4667112399583576064,4690067230254170112,1169270784"
            fields = result.split(",")
            the_timestamp = to_pd_timestamp(fields[0])

            # ✅ Best Practice: Explicit type conversion for DataFrame columns.
            the_id = generate_kdata_id(
                entity_id=entity_id, timestamp=the_timestamp, level=level
            )

            # 🧠 ML Signal: Data normalization by scaling values.
            open = to_float(fields[1])
            close = to_float(fields[2])
            high = to_float(fields[3])
            low = to_float(fields[4])
            # 🧠 ML Signal: Use of lambda functions for row-wise operations.
            volume = to_float(fields[5])
            turnover = to_float(fields[6])
            # 7 振幅
            # 🧠 ML Signal: Conditional logic based on entity type.
            change_pct = value_to_pct(to_float(fields[8]))
            # 9 变动
            turnover_rate = value_to_pct(to_float(fields[10]))

            # ✅ Best Practice: Use of to_numeric with error handling for data conversion.
            kdatas.append(
                # ✅ Best Practice: Use of default parameter value for limit increases function flexibility.
                dict(
                    id=the_id,
                    # 🧠 ML Signal: Use of specific entity_flag pattern could indicate a common configuration or filter criteria.
                    timestamp=the_timestamp,
                    entity_id=entity_id,
                    # 🧠 ML Signal: Specific fields requested could indicate common data attributes of interest.
                    # 🧠 ML Signal: Conditional assignment based on function parameters.
                    provider="em",
                    # 🧠 ML Signal: Function with default parameter value, indicating common usage pattern
                    code=code,
                    name=name,
                    # 🧠 ML Signal: Adding new columns to DataFrame.
                    # 🧠 ML Signal: Hardcoded string values, indicating specific API usage pattern
                    level=level.value,
                    # 🧠 ML Signal: Use of lambda functions for row-wise string operations.
                    # ✅ Best Practice: Explicit parameter naming improves readability and maintainability.
                    # 🧠 ML Signal: Hardcoded string values, indicating specific API usage pattern
                    open=open,
                    close=close,
                    high=high,
                    # ✅ Best Practice: Use of named arguments improves readability and maintainability
                    # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
                    # 🧠 ML Signal: Returning a DataFrame as a function result.
                    # 🧠 ML Signal: Use of specific enum value, indicating common usage pattern
                    low=low,
                    volume=volume,
                    turnover=turnover,
                    turnover_rate=turnover_rate,
                    change_pct=change_pct,
                )
            )
    if kdatas:
        # ✅ Best Practice: Convert entity_type to TradableType early to ensure consistent type usage throughout the function.
        df = pd.DataFrame.from_records(kdatas)
        return df


# ✅ Best Practice: Retrieve exchanges based on entity_type to ensure valid exchange options.
def get_basic_info(entity_id):
    entity_type, exchange, code = decode_entity_id(entity_id)
    if entity_type == "stock":
        # ⚠️ SAST Risk (Low): Using assert for input validation can be bypassed if Python is run with optimizations.
        url = "https://emh5.eastmoney.com/api/GongSiGaiKuang/GetJiBenZiLiao"
        result_field = "JiBenZiLiao"
    elif entity_type == "stockus":
        url = "https://emh5.eastmoney.com/api/MeiGu/GaiKuang/GetZhengQuanZiLiao"
        result_field = "ZhengQuanZiLiao"
    # ✅ Best Practice: Convert exchange to Exchange type to ensure consistent type usage.
    elif entity_type == "stockhk":
        url = "https://emh5.eastmoney.com/api/GangGu/GaiKuang/GetZhengQuanZiLiao"
        result_field = "ZhengQuanZiLiao"
    else:
        assert False

    data = {"fc": to_em_fc(entity_id=entity_id), "color": "w"}
    resp = requests.post(url=url, json=data, headers=DEFAULT_HEADER)

    resp.raise_for_status()
    resp.close()

    return resp.json()["Result"][result_field]


def get_future_list():
    # 主连
    url = f"https://futsseapi.eastmoney.com/list/filter/2?fid=sp_all&mktid=0&typeid=0&pageSize=1000&pageIndex=0&callbackName=jQuery34106875017735118845_1649736551642&sort=asc&orderBy=idx&_={now_timestamp()}"
    resp = requests.get(url, headers=DEFAULT_HEADER)
    # ⚠️ SAST Risk (Low): Using assert for control flow can be bypassed if Python is run with optimizations.
    resp.raise_for_status()
    result = json_callback_param(resp.text)
    resp.close()
    # [['DCE', 'im'], ['SHFE', 'rbm'], ['SHFE', 'hcm'], ['SHFE', 'ssm'], ['CZCE', 'SFM'], ['CZCE', 'SMM'], ['SHFE', 'wrm'], ['SHFE', 'cum'], ['SHFE', 'alm'], ['SHFE', 'znm'], ['SHFE', 'pbm'], ['SHFE', 'nim'], ['SHFE', 'snm'], ['INE', 'bcm'], ['SHFE', 'aum'], ['SHFE', 'agm'], ['DCE', 'am'], ['DCE', 'bm'], ['DCE', 'ym'], ['DCE', 'mm'], ['CZCE', 'RSM'], ['CZCE', 'OIM'], ['CZCE', 'RMM'], ['DCE', 'pm'], ['DCE', 'cm'], ['DCE', 'csm'], ['DCE', 'jdm'], ['CZCE', 'CFM'], ['CZCE', 'CYM'], ['CZCE', 'SRM'], ['CZCE', 'APM'], ['CZCE', 'CJM'], ['CZCE', 'PKM'], ['CZCE', 'PMM'], ['CZCE', 'WHM'], ['DCE', 'rrm'], ['CZCE', 'JRM'], ['CZCE', 'RIM'], ['CZCE', 'LRM'], ['DCE', 'lhm'], ['INE', 'scm'], ['SHFE', 'fum'], ['DCE', 'pgm'], ['INE', 'lum'], ['SHFE', 'bum'], ['CZCE', 'MAM'], ['DCE', 'egm'], ['DCE', 'lm'], ['CZCE', 'TAM'], ['DCE', 'vm'], ['DCE', 'ppm'], ['DCE', 'ebm'], ['CZCE', 'SAM'], ['CZCE', 'FGM'], ['CZCE', 'URM'], ['SHFE', 'rum'], ['INE', 'nrm'], ['SHFE', 'spm'], ['DCE', 'fbm'], ['DCE', 'bbm'], ['CZCE', 'PFM'], ['DCE', 'jmm'], ['DCE', 'jm'], ['CZCE', 'ZCM'], ['8', '060120'], ['8', '040120'], ['8', '070120'], ['8', '110120'], ['8', '050120'], ['8', '130120']]
    futures = []
    for item in result["list"]:
        entity = {}
        entity["exchange"], entity["code"] = item["uid"].split("|")

        # {'8', 'CZCE', 'DCE', 'INE', 'SHFE'}
        if entity["exchange"] == "8":
            entity["exchange"] = "cffex"
            entity["code"] = to_zvt_code(entity["code"])
        else:
            try:
                entity["exchange"] = Exchange(entity["exchange"].lower()).value
                if entity["code"][-1].lower() == "m":
                    entity["code"] = entity["code"][:-1]
                else:
                    assert False
                entity["code"] = entity["code"].upper()
            except Exception as e:
                logger.error(f"wrong item: {item}", e)
                continue

        # ⚠️ SAST Risk (Low): Using assert for control flow can be bypassed if Python is run with optimizations.
        entity["entity_type"] = "future"
        entity["name"] = item["name"]
        entity["id"] = f"future_{entity['exchange']}_{entity['code']}"
        entity["entity_id"] = entity["id"]
        # 🧠 ML Signal: The function get_top_tradable_list is called with parameters that could be used to learn patterns of tradable entities.
        futures.append(entity)
    # 🧠 ML Signal: Function uses a URL with query parameters, indicating a pattern of API usage.
    df = pd.DataFrame.from_records(data=futures)
    # ⚠️ SAST Risk (Medium): URL contains dynamic query parameters, which could be manipulated if not properly validated.
    return df


# ⚠️ SAST Risk (Medium): URL construction with dynamic parameters can lead to injection attacks if inputs are not sanitized.


def _calculate_limit(row):
    code = row["code"]
    # 🧠 ML Signal: The use of pd.concat to combine dataframes could be a pattern for learning data aggregation techniques.
    # ✅ Best Practice: Use session object for HTTP requests to leverage connection pooling.
    change_pct = row["change_pct"]
    if code.startswith(("83", "87", "88", "889", "82", "920")):
        return change_pct >= 0.29, change_pct <= -0.29
    # ⚠️ SAST Risk (Low): Direct use of requests.get without session can lead to inefficient network usage.
    elif code.startswith("300") or code.startswith("301") or code.startswith("688"):
        return change_pct >= 0.19, change_pct <= -0.19
    # ⚠️ SAST Risk (Medium): Potential risk of JSON injection if the response is not properly validated.
    else:
        return change_pct > 0.09, change_pct < -0.09


# 🧠 ML Signal: Pattern of converting stock codes to IDs, useful for entity recognition models.


def get_stock_turnover():
    sz_url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2&fields2=f51,f57&ut=fa5fd1943c7b386f172d6893dbfba10b&iscr=0&iscca=0&secid=0.399001&time=0&ndays=2"
    resp = requests.get(sz_url, headers=DEFAULT_HEADER)

    resp.raise_for_status()

    data = resp.json()["data"]["trends"]
    resp.close()
    return data


def get_top_tradable_list(
    entity_type, fields, limit, entity_flag, exchange=None, return_quote=False
):
    url = f"https://push2.eastmoney.com/api/qt/clist/get?np=1&fltt=2&invt=2&fields={fields}&pn=1&pz={limit}&fid=f3&po=1&{entity_flag}&ut=f057cbcbce2a86e2866ab8877db1d059&forcect=1&cb=cbCallbackMore&&callback=jQuery34109676853980006124_{now_timestamp() - 1}&_={now_timestamp()}"
    # ✅ Best Practice: Use of format method for string formatting improves readability.
    resp = requests.get(url, headers=DEFAULT_HEADER)

    # 🧠 ML Signal: Conditional logic based on market values indicates a pattern for market-specific processing.
    # 🧠 ML Signal: Use of current date as a timestamp, indicating a pattern of time-based data logging.
    resp.raise_for_status()

    # 🧠 ML Signal: Function call with specific market code suggests a pattern for handling Chinese stock codes.
    result = json_callback_param(resp.text)
    resp.close()
    # 🧠 ML Signal: Specific market codes mapped to formatted strings indicate a pattern for US stock exchanges.
    data = result["data"]["diff"]
    df = pd.DataFrame.from_records(data=data)

    if return_quote:
        df = df[
            [
                "f12",
                "f14",
                "f2",
                "f3",
                "f5",
                "f8",
                "f6",
                "f15",
                "f16",
                "f17",
                "f20",
                "f21",
            ]
        ]
        # 🧠 ML Signal: Specific market code for Hong Kong stock exchange indicates a pattern for HK market handling.
        df.columns = [
            "code",
            "name",
            "price",
            # 🧠 ML Signal: Iterating over exchange map suggests a pattern for dynamic market handling.
            "change_pct",
            "volume",
            # 🧠 ML Signal: Matching market flag to exchange indicates a pattern for exchange-specific processing.
            "turnover_rate",
            "turnover",
            # 🧠 ML Signal: Nested iteration over tradable types suggests a pattern for entity type mapping.
            # 🧠 ML Signal: Checking if exchange is in exchanges indicates a pattern for valid exchange filtering.
            # 🧠 ML Signal: Hardcoded URL can be used to identify API usage patterns
            "high",
            "low",
            "open",
            "total_cap",
            "float_cap",
            # ⚠️ SAST Risk (Low): Returning the code directly if no conditions match may lead to unexpected results.
        ]
        # 🧠 ML Signal: Logging usage can indicate debugging or monitoring practices

        df = df.dropna()
        df = df[df.change_pct != "-"]
        # ⚠️ SAST Risk (Low): Potential misuse of session object if not properly validated
        df = df[df.turnover_rate != "-"]
        df = df[df.turnover != "-"]

        # ⚠️ SAST Risk (Low): No exception handling for network request failures
        df = df.astype(
            {
                "change_pct": "float",
                "turnover_rate": "float",
                "turnover": "float",
                "volume": "float",
            }
        )

        df["change_pct"] = df["change_pct"] / 100
        # ⚠️ SAST Risk (Low): Assumes JSON response without validation
        df["turnover_rate"] = df["turnover_rate"] / 100
        df["volume"] = df["volume"] * 100

        df[["is_limit_up", "is_limit_down"]] = df.apply(
            lambda row: _calculate_limit(row), axis=1, result_type="expand"
        )

    else:
        # 🧠 ML Signal: Usage of list comprehensions can indicate coding style
        if entity_type == TradableType.stock:
            df = df[["f12", "f13", "f14", "f20", "f21", "f9", "f23"]]
            df.columns = ["code", "exchange", "name", "cap", "cap1", "pe", "pb"]
            df[["cap", "cap1", "pe", "pb"]] = df[["cap", "cap1", "pe", "pb"]].apply(
                pd.to_numeric, errors="coerce"
            )
        else:
            df = df[["f12", "f13", "f14"]]
            df.columns = ["code", "exchange", "name"]
        if exchange:
            df["exchange"] = exchange.value
        df["entity_type"] = entity_type.value
        df["id"] = df[["entity_type", "exchange", "code"]].apply(
            lambda x: "_".join(x.astype(str)), axis=1
        )
        df["entity_id"] = df["id"]

    return df


# ✅ Best Practice: Use of timestamp functions for consistent time handling
def get_top_stocks(limit=100):
    # 沪深和北交所
    # 🧠 ML Signal: Function definition and naming pattern
    entity_flag = "fs=m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:81+s:2048"

    # 🧠 ML Signal: Function call pattern
    fields = "f2,f3,f5,f6,f8,f12,f14,f15,f16,f17,f20,f21"
    return get_top_tradable_list(
        # 🧠 ML Signal: Logging usage pattern
        entity_type=TradableType.stock,
        fields=fields,
        limit=limit,
        entity_flag=entity_flag,
        return_quote=True,
        # 🧠 ML Signal: Error logging can be used to identify error handling practices
    )


# 🧠 ML Signal: DataFrame creation from records


def get_top_stockhks(limit=20):
    # 🧠 ML Signal: Function call with multiple parameters
    # ✅ Best Practice: Use of a helper function to convert entity_id to sec_id improves code readability and reusability.
    entity_flag = "fs=b:DLMK0144,b:DLMK0146"
    # ⚠️ SAST Risk (Low): Potential SQL injection if inputs are not sanitized
    fields = "f2,f3,f5,f6,f8,f12,f14,f15,f16,f17,f20,f21"
    # ⚠️ SAST Risk (Low): URL construction with f-strings can lead to injection if inputs are not sanitized.
    return get_top_tradable_list(
        entity_type=TradableType.stockhk,
        fields=fields,
        limit=limit,
        entity_flag=entity_flag,
        return_quote=True,
        # ✅ Best Practice: Logging the URL for debugging purposes.
    )


# ✅ Best Practice: Use of session for requests can improve performance by reusing connections.
def get_tradable_list(
    entity_type: Union[TradableType, str] = "stock",
    exchange: Union[Exchange, str] = None,
    limit: int = 10000,
    hk_south=False,
    # ⚠️ SAST Risk (Medium): Directly manipulating response text without validation can lead to security issues.
    block_category=BlockCategory.concept,
    # ✅ Best Practice: Closing the response to free up resources.
    # 🧠 ML Signal: Pattern of filtering and transforming data from an API response.
):
    entity_type = TradableType(entity_type)
    if entity_type == TradableType.future:
        return get_future_list()

    exchanges = get_entity_exchanges(entity_type=entity_type)

    if exchange is not None:
        assert exchange in exchanges
        exchanges = [exchange]

    dfs = []
    for exchange in exchanges:
        exchange = Exchange(exchange)
        ex_flag = to_em_entity_flag(exchange=exchange)
        entity_flag = f"fs=m:{ex_flag}"

        if entity_type == TradableType.index:
            if exchange == Exchange.sh:
                entity_flag = "fs=i:1.000001,i:1.000002,i:1.000003,i:1.000009,i:1.000010,i:1.000011,i:1.000012,i:1.000016,i:1.000300,i:1.000903,i:1.000905,i:1.000906,i:1.000688,i:1.000852,i:2.932000"
            if exchange == Exchange.sz:
                entity_flag = "fs=i:0.399001,i:0.399002,i:0.399003,i:0.399004,i:0.399005,i:0.399006,i:0.399100,i:0.399106,i:0.399305,i:0.399550"
        elif entity_type == TradableType.currency:
            entity_flag = "fs=m:119,m:120"
        elif entity_type == TradableType.indexus:
            # 纳斯达克，道琼斯，标普500，美元指数
            # 🧠 ML Signal: Recursive function call pattern for pagination.
            entity_flag = "fs=i:100.NDX,i:100.DJIA,i:100.SPX,i:100.UDI"
        elif entity_type == TradableType.cbond:
            # 🧠 ML Signal: Function decodes entity_id into components and processes based on type and exchange
            if exchange == Exchange.sz:
                entity_flag = "fs=m:0+e:11"
            elif exchange == Exchange.sh:
                entity_flag = "fs=m:1+e:11"
            else:
                assert False
        # ✅ Best Practice: Logging errors with detailed information for troubleshooting.
        # m为交易所代码，t为交易类型
        elif entity_type in [
            TradableType.block,
            TradableType.stock,
            TradableType.stockus,
            TradableType.stockhk,
        ]:
            if exchange == Exchange.sh:
                # t=2 主板
                # t=23 科创板
                entity_flag = "fs=m:1+t:2,m:1+t:23"
            if exchange == Exchange.sz:
                # 🧠 ML Signal: Mapping of exchanges to specific flags, useful for feature extraction
                # ⚠️ SAST Risk (Low): Potential risk if Exchange is not validated or sanitized
                # t=6 主板
                # t=80 创业板
                entity_flag = "fs=m:0+t:6,m:0+t:13,m:0+t:80"
            if exchange == Exchange.bj:
                entity_flag = "fs=m:0+t:81+s:2048"
            if exchange == Exchange.hk:
                if hk_south:
                    # 港股通
                    entity_flag = "fs=b:DLMK0144,b:DLMK0146"
                else:
                    # t=3 主板
                    # t=4 创业板
                    entity_flag = "fs=m:116+t:3,m:116+t:4"
            if exchange == Exchange.nasdaq:
                # t=1
                # t=3 中概股
                # ✅ Best Practice: Consider adding type hints for the return type of the function
                entity_flag = "fs=m:105+t:1,m:105+t:3"
            if exchange == Exchange.nyse:
                # 🧠 ML Signal: Conversion of input to a specific type (Exchange) indicates a pattern of type normalization
                # t=1
                # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
                # t=3 中概股
                # 🧠 ML Signal: Use of a dictionary for mapping suggests a pattern of key-value retrieval
                entity_flag = "fs=m:106+t:1,m:105+t:3"
            # ✅ Best Practice: Ensure the input is of the correct type by converting it to AdjustType.
            if exchange == Exchange.cn:
                if block_category == BlockCategory.industry:
                    # 🧠 ML Signal: Using conditional checks to map enum values to integers.
                    entity_flag = entity_flag + "+t:2"
                elif block_category == BlockCategory.concept:
                    entity_flag = entity_flag + "+t:3"
                else:
                    # 🧠 ML Signal: Function converts interval levels to numeric flags, useful for feature extraction
                    assert False

        # ✅ Best Practice: Explicitly converting level to IntervalLevel ensures type consistency
        # f2, f3, f4, f12, f13, f14, f19, f111, f148
        fields = "f1,f2,f3,f4,f12,f13,f14"
        # 🧠 ML Signal: Mapping specific interval levels to numeric values
        if entity_type in (TradableType.stock, TradableType.stockhk):
            # 市值,流通市值,pe,pb
            fields = fields + ",f20,f21,f9,f23"

        df = get_top_tradable_list(
            entity_type=entity_type,
            fields=fields,
            limit=limit,
            entity_flag=entity_flag,
            exchange=exchange,
        )
        if entity_type == TradableType.block:
            df["category"] = block_category.value

        dfs.append(df)

    return pd.concat(dfs)


# 🧠 ML Signal: Function definition with specific naming pattern
def get_block_stocks(block_id, name="", session=None):
    entity_type, exchange, code = decode_entity_id(block_id)
    # ⚠️ SAST Risk (Low): Using assert for control flow can be disabled in production
    # 🧠 ML Signal: Unpacking tuple from function return
    category_stocks_url = f"http://48.push2.eastmoney.com/api/qt/clist/get?cb=jQuery11240710111145777397_{now_timestamp() - 1}&pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=4668014655929990|0|1|0|web&fid=f3&fs=b:{code}+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45&_={now_timestamp()}"
    if session:
        # 🧠 ML Signal: Conditional logic based on entity type
        resp = session.get(category_stocks_url, headers=DEFAULT_HEADER)
    else:
        # ✅ Best Practice: Clear and concise string manipulation
        resp = requests.get(category_stocks_url, headers=DEFAULT_HEADER)

    # 🧠 ML Signal: Conditional logic based on entity type and substring check
    # 🧠 ML Signal: Function uses a series of conditional checks to map input codes to specific values.
    data = json_callback_param(resp.text)["data"]
    the_list = []
    # ✅ Best Practice: Use of f-string for string formatting
    if data:
        results = data["diff"]
        # ✅ Best Practice: Use of f-string for string formatting
        for result in results:
            stock_code = result["f12"]
            stock_name = result["f14"]
            stock_id = china_stock_code_to_id(stock_code)

            the_list.append(
                {
                    "id": "{}_{}".format(block_id, stock_id),
                    "entity_id": block_id,
                    "entity_type": "block",
                    "exchange": exchange,
                    # ⚠️ SAST Risk (High): The function get_stock_turnover() is called without being defined or imported, leading to a potential NameError.
                    # ✅ Best Practice: Using __all__ to define public API of the module, which improves code maintainability and readability.
                    "code": code,
                    "name": name,
                    "timestamp": current_date(),
                    "stock_id": stock_id,
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                }
            )
    return the_list


def market_code_to_entity_id(market, code):
    if market in (0, 1):
        return china_stock_code_to_id(code)
    elif market == 105:
        return f"stockus_nasdaq_{code}"
    elif market == 106:
        return f"stockus_nyse_{code}"
    elif market == 116:
        return f"stockhk_hk_{code}"
    else:
        for exchange, flag in exchange_map_em_flag.items():
            if flag == market:
                for entity_type, exchanges in tradable_type_map_exchanges.items():
                    if exchange in exchanges:
                        return f"{entity_type.value}_{exchange.value}_{code}"
    return code


def get_hot_topic(session: Session = None):
    url = "https://emcreative.eastmoney.com/FortuneApi/GuBaApi/common"
    data = {
        "url": "newctopic/api/Topic/HomeTopicRead?deviceid=IPHONE&version=10001000&product=Guba&plat=Iphone&p=1&ps=20&needPkPost=true",
        "type": "get",
        "parm": "",
    }
    logger.debug(f"get hot topic from: {url}")
    if session:
        resp = session.post(url=url, json=data, headers=DEFAULT_HEADER)
    else:
        resp = requests.post(url=url, json=data, headers=DEFAULT_HEADER)

    if resp.status_code == 200:
        data_list = resp.json().get("re")
        if data_list:
            hot_topics = []
            for position, data in enumerate(data_list):
                if data["stockList"]:
                    entity_ids = [
                        market_code_to_entity_id(
                            market=stock["qMarket"], code=stock["qCode"]
                        )
                        for stock in data["stockList"]
                    ]
                else:
                    entity_ids = []
                topic_id = data["topicid"]
                entity_id = f"hot_topic_{topic_id}"
                hot_topics.append(
                    {
                        "id": entity_id,
                        "entity_id": entity_id,
                        "timestamp": now_pd_timestamp(),
                        "created_timestamp": to_pd_timestamp(data["cTime"]),
                        "position": position,
                        "entity_ids": entity_ids,
                        "news_code": topic_id,
                        "news_title": data["name"],
                        "news_content": data["summary"],
                    }
                )
            return hot_topics

    logger.error(f"request em data code: {resp.status_code}, error: {resp.text}")


def record_hot_topic():
    hot_topics = get_hot_topic()
    logger.debug(hot_topics)
    if hot_topics:
        df = pd.DataFrame.from_records(hot_topics)
        df_to_db(
            df=df,
            data_schema=StockHotTopic,
            provider="em",
            force_update=True,
            dtype={"entity_ids": sqlalchemy.JSON},
        )


def get_news(
    entity_id, ps=200, index=1, start_timestamp=None, session=None, latest_code=None
):
    sec_id = to_em_sec_id(entity_id=entity_id)
    url = f"https://np-listapi.eastmoney.com/comm/wap/getListInfo?cb=callback&client=wap&type=1&mTypeAndCode={sec_id}&pageSize={ps}&pageIndex={index}&callback=jQuery1830017478247906740352_{now_timestamp() - 1}&_={now_timestamp()}"
    logger.debug(f"get news from: {url}")
    if session:
        resp = session.get(url)
    else:
        resp = requests.get(url)
    # {
    #     "Art_ShowTime": "2022-02-11 14:29:25",
    #     "Art_Image": "",
    #     "Art_MediaName": "每日经济新闻",
    #     "Art_Code": "202202112274017262",
    #     "Art_Title": "潍柴动力：巴拉德和锡里斯不纳入合并财务报表范围",
    #     "Art_SortStart": "1644560965017262",
    #     "Art_VideoCount": 0,
    #     "Art_OriginUrl": "http://finance.eastmoney.com/news/1354,202202112274017262.html",
    #     "Art_Url": "http://finance.eastmoney.com/a/202202112274017262.html",
    # }
    if resp.status_code == 200:
        json_text = resp.text[resp.text.index("(") + 1 : resp.text.rindex(")")]
        if "list" in demjson3.decode(json_text)["data"]:
            json_result = demjson3.decode(json_text)["data"]["list"]
            resp.close()
            if json_result:
                news = [
                    {
                        "id": f'{entity_id}_{item.get("Art_ShowTime", "")}',
                        "entity_id": entity_id,
                        "timestamp": to_pd_timestamp(item.get("Art_ShowTime", "")),
                        "news_code": item.get("Art_Code", ""),
                        "news_url": item.get("Art_Url", ""),
                        "news_title": item.get("Art_Title", ""),
                        "ignore_by_user": False,
                    }
                    for index, item in enumerate(json_result)
                    if not start_timestamp
                    or (
                        (to_pd_timestamp(item["Art_ShowTime"]) >= start_timestamp)
                        and (item.get("Art_Code", "") != latest_code)
                    )
                ]
                if len(news) < len(json_result):
                    return news
                next_data = get_news(entity_id=entity_id, ps=ps, index=index + 1)
                if next_data:
                    return news + next_data
                else:
                    return news
        else:
            return None

    logger.error(f"request em data code: {resp.status_code}, error: {resp.text}")


# utils to transform zvt entity to em entity
def to_em_fc(entity_id):
    entity_type, exchange, code = decode_entity_id(entity_id)
    if entity_type == "stock":
        if exchange == "sh":
            return f"{code}01"
        if exchange == "sz":
            return f"{code}02"

    if entity_type == "stockhk":
        return code

    if entity_type == "stockus":
        if exchange == "nyse":
            return f"{code}.N"
        if exchange == "nasdaq":
            return f"{code}.O"


exchange_map_em_flag = {
    #: 深证交易所
    Exchange.sz: 0,
    #: 上证交易所
    Exchange.sh: 1,
    #: 北交所
    Exchange.bj: 0,
    #: 纳斯达克
    Exchange.nasdaq: 105,
    #: 纽交所
    Exchange.nyse: 106,
    #: 中国金融期货交易所
    Exchange.cffex: 8,
    #: 上海期货交易所
    Exchange.shfe: 113,
    #: 大连商品交易所
    Exchange.dce: 114,
    #: 郑州商品交易所
    Exchange.czce: 115,
    #: 上海国际能源交易中心
    Exchange.ine: 142,
    #: 港交所
    Exchange.hk: 116,
    #: 中国行业/概念板块
    Exchange.cn: 90,
    #: 美国指数
    Exchange.us: 100,
    #: 汇率
    Exchange.forex: 119,
}


def to_em_entity_flag(exchange: Union[Exchange, str]):
    exchange = Exchange(exchange)
    return exchange_map_em_flag.get(exchange, exchange)


def to_em_fq_flag(adjust_type: AdjustType):
    adjust_type = AdjustType(adjust_type)
    if adjust_type == AdjustType.bfq:
        return 0
    if adjust_type == AdjustType.qfq:
        return 1
    if adjust_type == AdjustType.hfq:
        return 2


def to_em_level_flag(level: IntervalLevel):
    level = IntervalLevel(level)
    if level == IntervalLevel.LEVEL_1MIN:
        return 1
    elif level == IntervalLevel.LEVEL_5MIN:
        return 5
    elif level == IntervalLevel.LEVEL_15MIN:
        return 15
    elif level == IntervalLevel.LEVEL_30MIN:
        return 30
    elif level == IntervalLevel.LEVEL_1HOUR:
        return 60
    elif level == IntervalLevel.LEVEL_1DAY:
        return 101
    elif level == IntervalLevel.LEVEL_1WEEK:
        return 102
    elif level == IntervalLevel.LEVEL_1MON:
        return 103

    assert False


def to_em_sec_id(entity_id):
    entity_type, exchange, code = decode_entity_id(entity_id)
    # 主力合约
    if entity_type == "future" and code[-1].isalpha():
        code = code + "m"
    if entity_type == "currency" and "CNYC" in code:
        return f"120.{code}"
    return f"{to_em_entity_flag(exchange)}.{code}"


def to_zvt_code(code):
    #  ('中证当月连续', '8|060120'),
    #  ('沪深当月连续', '8|040120'),
    #  ('上证当月连续', '8|070120'),
    #  ('十债当季连续', '8|110120'),
    #  ('五债当季连续', '8|050120'),
    #  ('二债当季连续', '8|130120')]
    if code == "060120":
        return "IC"
    elif code == "040120":
        return "IF"
    elif code == "070120":
        return "IH"
    elif code == "110120":
        return "T"
    elif code == "050120":
        return "TF"
    elif code == "130120":
        return "TS"
    return code


if __name__ == "__main__":
    # from pprint import pprint
    # pprint(get_free_holder_report_dates(code='000338'))
    # pprint(get_holder_report_dates(code='000338'))
    # pprint(get_holders(code='000338', end_date='2021-03-31'))
    # pprint(get_free_holders(code='000338', end_date='2021-03-31'))
    # pprint(get_ii_holder(code='000338', report_date='2021-03-31',
    #                      org_type=actor_type_to_org_type(ActorType.corporation)))
    # print(
    #     get_ii_summary(code="600519", report_date="2021-03-31", org_type=actor_type_to_org_type(ActorType.corporation))
    # )
    # df = get_kdata(entity_id="index_sz_399370", level="1wk")
    # df = get_tradable_list(entity_type="cbond")
    # print(df)
    # df = get_news("stock_sz_300999", ps=1)
    # print(df)
    # print(len(df))
    # df = get_tradable_list(entity_type="block")
    # print(df)
    # df = get_tradable_list(entity_type="indexus")
    # print(df)
    # df = get_tradable_list(entity_type="currency")
    # print(df)
    # df = get_tradable_list(entity_type="index")
    # print(df)
    # df = get_kdata(entity_id="index_us_SPX", level="1d")
    # df = get_treasury_yield(pn=1, ps=50, fetch_all=False)
    # print(df)
    # df = get_future_list()
    # print(df)
    # df = get_kdata(entity_id="future_dce_I", level="1d")
    # print(df)
    # df = get_dragon_and_tiger(code="000989", start_date="2018-10-31")
    # df = get_dragon_and_tiger_list(start_date="2022-04-25")
    # # df = get_tradable_list()
    # # df_delist = df[df["name"].str.contains("退")]
    # print(df_delist[["id", "name"]].values.tolist())
    # print(get_block_stocks(block_id="block_cn_BK1144"))
    # df = get_tradable_list(entity_type="index")
    # print(df)
    # df = get_kdata(entity_id="stock_bj_873693", level="1d")
    # print(df)
    # print(get_controlling_shareholder(code="000338"))
    # events = get_events(entity_id="stock_sz_300684")
    # print(events)
    # print(get_hot_topic())
    # record_hot_topic()
    # df = StockHotTopic.query_data(
    #     filters=[func.json_extract(StockHotTopic.entity_ids, "$").contains("stock_sh_600809")],
    # )
    # print(df)
    # print(get_top_stocks(limit=10))
    # print(get_top_stockhks(limit=10))
    # print(get_controlling_shareholder(code="000338"))
    # print(get_top_ten_free_holder_stats(code="000338"))
    print(get_stock_turnover())


# the __all__ is generated
__all__ = [
    "get_treasury_yield",
    "get_ii_holder_report_dates",
    "get_dragon_and_tiger_list",
    "get_dragon_and_tiger",
    "get_holder_report_dates",
    "get_free_holder_report_dates",
    "get_controlling_shareholder",
    "get_ii_holder",
    "get_ii_summary",
    "get_free_holders",
    "get_top_ten_free_holder_stats",
    "get_controlling_shareholder",
    "get_holders",
    "get_url",
    "get_exchange",
    "actor_type_to_org_type",
    "generate_filters",
    "get_em_data",
    "get_quotes",
    "get_kdata",
    "get_basic_info",
    "get_future_list",
    "get_top_tradable_list",
    "get_top_stocks",
    "get_top_stockhks",
    "get_tradable_list",
    "get_block_stocks",
    "market_code_to_entity_id",
    "get_hot_topic",
    "record_hot_topic",
    "get_news",
    "to_em_fc",
    "to_em_entity_flag",
    "to_em_fq_flag",
    "to_em_level_flag",
    "to_em_sec_id",
    "to_zvt_code",
]
