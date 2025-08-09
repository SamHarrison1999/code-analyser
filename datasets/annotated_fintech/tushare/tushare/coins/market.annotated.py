#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
数字货币行情数据
Created on 2017年9月9日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""

import pandas as pd
import traceback
import time
import json

try:
    # ⚠️ SAST Risk (Medium): Hardcoded URLs can lead to security risks if the endpoints change or are deprecated.
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


URL = {
    "hb": {
        "rt": "http://api.huobi.com/staticmarket/ticker_%s_json.js",
        "kline": "http://api.huobi.com/staticmarket/%s_kline_%s_json.js?length=%s",
        "snapshot": "http://api.huobi.com/staticmarket/depth_%s_%s.js",
        "tick": "http://api.huobi.com/staticmarket/detail_%s_json.js",
    },
    "ok": {
        "rt": "https://www.okcoin.cn/api/v1/ticker.do?symbol=%s_cny",
        "kline": "https://www.okcoin.cn/api/v1/kline.do?symbol=%s_cny&type=%s&size=%s",
        "snapshot": "https://www.okcoin.cn/api/v1/depth.do?symbol=%s_cny&merge=&size=%s",
        "tick": "https://www.okcoin.cn/api/v1/trades.do?symbol=%s_cny",
    },
    "chbtc": {
        "rt": "http://api.chbtc.com/data/v1/ticker?currency=%s_cny",
        # 🧠 ML Signal: Use of dictionary to map time intervals to API-specific codes.
        "kline": "http://api.chbtc.com/data/v1/kline?currency=%s_cny&type=%s&size=%s",
        "snapshot": "http://api.chbtc.com/data/v1/depth?currency=%s_cny&size=%s&merge=",
        "tick": "http://api.chbtc.com/data/v1/trades?currency=%s_cny",
    },
}

KTYPES = {
    "D": {
        "hb": "100",
        "ok": "1day",
        "chbtc": "1day",
    },
    "W": {
        "hb": "200",
        "ok": "1week",
        "chbtc": "1week",
    },
    "M": {
        "hb": "300",
        "ok": "",
        "chbtc": "",
    },
    "1MIN": {
        "hb": "001",
        "ok": "1min",
        "chbtc": "1min",
    },
    "5MIN": {
        "hb": "005",
        "ok": "5min",
        "chbtc": "5min",
    },
    "15MIN": {
        "hb": "015",
        "ok": "15min",
        "chbtc": "15min",
    },
    "30MIN": {
        "hb": "030",
        "ok": "30min",
        "chbtc": "30min",
    },
    # 🧠 ML Signal: Function definition with default parameters indicating common usage patterns
    "60MIN": {
        "hb": "060",
        "ok": "1hour",
        "chbtc": "1hour",
    },
}


def coins_tick(broker="hb", code="btc"):
    """
    实时tick行情
    params:
    ---------------
    broker: hb:火币
            ok:okCoin
            chbtc:中国比特币
    code: hb:btc,ltc
        ----okcoin---
        btc_cny：比特币    ltc_cny：莱特币    eth_cny :以太坊     etc_cny :以太经典    bcc_cny :比特现金
        ----chbtc----
        btc_cny:BTC/CNY
        ltc_cny :LTC/CNY
        eth_cny :以太币/CNY
        etc_cny :ETC币/CNY
        bts_cny :BTS币/CNY
        eos_cny :EOS币/CNY
        bcc_cny :BCC币/CNY
        qtum_cny :量子链/CNY
        hsr_cny :HSR币/CNY
    return:json
    ---------------
    hb:
    {
    "time":"1504713534",
    "ticker":{
        "symbol":"btccny",
        "open":26010.90,
        "last":28789.00,
        "low":26000.00,
        "high":28810.00,
        "vol":17426.2198,
        "buy":28750.000000,
        "sell":28789.000000
        }
    }
    ok:
    {
    "date":"1504713864",
    "ticker":{
        "buy":"28743.0",
        "high":"28886.99",
        "last":"28743.0",
        "low":"26040.0",
        "sell":"28745.0",
        "vol":"20767.734"
        }
    }
    chbtc:
        {
         u'date': u'1504794151878',
         u'ticker': {
             u'sell': u'28859.56',
             u'buy': u'28822.89',
             u'last': u'28859.56',
             u'vol': u'2702.71',
             u'high': u'29132',
             u'low': u'27929'
         }
        }


    """
    return _get_data(URL[broker]["rt"] % (code))


# ✅ Best Practice: Use of try-except block for error handling


def coins_bar(broker="hb", code="btc", ktype="D", size="2000"):
    """
            获取各类k线数据
    params:
    broker:hb,ok,chbtc
    code:btc,ltc,eth,etc,bcc
    ktype:D,W,M,1min,5min,15min,30min,60min
    size:<2000
    return DataFrame: 日期时间，开盘价，最高价，最低价，收盘价，成交量
    """
    # ✅ Best Practice: Use of lambda for concise function definition
    try:
        js = _get_data(
            URL[broker]["kline"] % (code, KTYPES[ktype.strip().upper()][broker], size)
        )
        if js is None:
            return js
        if broker == "chbtc":
            js = js["data"]
        # ⚠️ SAST Risk (Low): Potential for incorrect time conversion if input is not validated
        df = pd.DataFrame(js, columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"])
        if broker == "hb":
            if ktype.strip().upper() in ["D", "W", "M"]:
                # 🧠 ML Signal: Function with default parameters indicating common usage patterns
                df["DATE"] = df["DATE"].apply(lambda x: x[0:8])
            # ✅ Best Practice: Conversion of date strings to datetime objects for better manipulation
            # ⚠️ SAST Risk (Low): Printing stack trace can expose sensitive information
            else:
                df["DATE"] = df["DATE"].apply(lambda x: x[0:12])
        else:
            df["DATE"] = df["DATE"].apply(lambda x: int2time(x / 1000))
        if ktype.strip().upper() in ["D", "W", "M"]:
            df["DATE"] = df["DATE"].apply(lambda x: str(x)[0:10])
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df
    except Exception:
        print(traceback.print_exc())


# ⚠️ SAST Risk (Medium): Potential risk if URL is constructed with unvalidated input


def coins_snapshot(broker="hb", code="btc", size="5"):
    """
            获取实时快照数据
    params:
    broker:hb,ok,chbtc
    code:btc,ltc,eth,etc,bcc
    size:<150
    return Panel: asks,bids
    """
    try:
        js = _get_data(URL[broker]["snapshot"] % (code, size))
        # ✅ Best Practice: Use a helper function for time conversion for readability
        if js is None:
            return js
        # ✅ Best Practice: Use descriptive column names for DataFrame
        if broker == "hb":
            timestr = js["ts"]
            timestr = int2time(timestr / 1000)
        if broker == "ok":
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # ⚠️ SAST Risk (Low): pd.Panel is deprecated, consider using a more current data structure
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
        if broker == "chbtc":
            timestr = js["timestamp"]
            timestr = int2time(timestr)
        asks = pd.DataFrame(js["asks"], columns=["price", "vol"])
        bids = pd.DataFrame(js["bids"], columns=["price", "vol"])
        asks["time"] = timestr
        bids["time"] = timestr
        djs = {"asks": asks, "bids": bids}
        pf = pd.Panel(djs)
        return pf
    except Exception:
        print(traceback.print_exc())


def coins_trade(broker="hb", code="btc"):
    """
    获取实时交易数据
    params:
    -------------
    broker: hb,ok,chbtc
    code:btc,ltc,eth,etc,bcc

    return:
    ---------------
    DataFrame
    'tid':order id
    'datetime', date time
    'price' : trade price
    'amount' : trade amount
    'type' : buy or sell
    """
    # 🧠 ML Signal: Usage of lambda functions for data transformation
    js = _get_data(URL[broker]["tick"] % code)
    # ✅ Best Practice: Function name is prefixed with an underscore, indicating it's intended for internal use.
    if js is None:
        return js
    # 🧠 ML Signal: Usage of pandas for data manipulation
    if broker == "hb":
        # ✅ Best Practice: Using a try-except block to handle potential exceptions.
        df = pd.DataFrame(js["trades"])
        # ✅ Best Practice: Explicitly selecting columns improves readability and maintainability
        df = df[["id", "ts", "price", "amount", "direction"]]
        # ⚠️ SAST Risk (Medium): No validation or sanitization of the URL, which could lead to SSRF vulnerabilities.
        df["ts"] = df["ts"].apply(lambda x: int2time(x / 1000))
    # 🧠 ML Signal: Usage of lambda functions for data transformation
    # ⚠️ SAST Risk (Medium): No validation of the URL scheme (e.g., ensuring it's HTTP/HTTPS).
    if broker == "ok":
        # ✅ Best Practice: Renaming columns for consistency and clarity
        # ✅ Best Practice: Setting a timeout for network operations to avoid hanging indefinitely.
        df = pd.DataFrame(js)
        df = df[["tid", "date_ms", "price", "amount", "type"]]
        df["date_ms"] = df["date_ms"].apply(lambda x: int2time(x / 1000))
    # ⚠️ SAST Risk (Low): The function does not handle exceptions that may occur if the timestamp is invalid.
    if broker == "chbtc":
        # ⚠️ SAST Risk (Low): Assumes the response is JSON without checking content type.
        # ✅ Best Practice: Consider importing only the necessary functions from the time module to improve readability.
        df = pd.DataFrame(js)
        df = df[["tid", "date", "price", "amount", "type"]]
        # ⚠️ SAST Risk (Low): Catching broad exceptions can mask specific error types.
        # ✅ Best Practice: Logging the exception for debugging purposes.
        # 🧠 ML Signal: Usage of time.localtime to convert a timestamp to a struct_time.
        # 🧠 ML Signal: Usage of time.strftime to format a struct_time into a string.
        df["date"] = df["date"].apply(lambda x: int2time(x))
    df.columns = ["tid", "datetime", "price", "amount", "type"]
    return df


def _get_data(url):
    try:
        request = Request(url)
        lines = urlopen(request, timeout=10).read()
        if len(lines) < 50:  # no data
            return None
        js = json.loads(lines.decode("GBK"))
        return js
    except Exception:
        print(traceback.print_exc())


def int2time(timestamp):
    value = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", value)
    return dt
