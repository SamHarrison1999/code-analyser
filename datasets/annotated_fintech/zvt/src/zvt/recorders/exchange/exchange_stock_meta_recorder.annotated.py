# -*- coding: utf-8 -*-

import io
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.

import pandas as pd
import requests

from zvt.contract.api import df_to_db
# ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and structure
from zvt.contract.recorder import Recorder
from zvt.domain import Stock, StockDetail
# ✅ Best Practice: Class variables should be defined at the top of the class for clarity
from zvt.recorders.consts import DEFAULT_SH_HEADER, DEFAULT_SZ_HEADER
from zvt.utils.time_utils import to_pd_timestamp
# 🧠 ML Signal: Hardcoded provider name could be used to identify data source patterns

# ⚠️ SAST Risk (Low): Hardcoded URL may lead to maintenance issues if the URL changes

class ExchangeStockMetaRecorder(Recorder):
    data_schema = Stock
    # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security issues if the endpoint changes or is deprecated.
    provider = "exchange"

    # ⚠️ SAST Risk (Low): No error handling for the HTTP request, which may lead to unhandled exceptions.
    # 🧠 ML Signal: Usage of requests.get to fetch data from a URL.
    original_page_url = "http://www.sse.com.cn/assortment/stock/list/share/"

    def run(self):
        # 🧠 ML Signal: Method call with specific parameters indicating a pattern of downloading stock lists.
        url = (
            "http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=1"
        )
        resp = requests.get(url, headers=DEFAULT_SH_HEADER)
        # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security issues if the endpoint changes or is deprecated.
        self.download_stock_list(response=resp, exchange="sh")

        # ⚠️ SAST Risk (Low): No error handling for the HTTP request, which may lead to unhandled exceptions.
        url = (
            # 🧠 ML Signal: Usage of requests.get to fetch data from a URL.
            "http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=8"
        # 🧠 ML Signal: Method call with specific parameters indicating a pattern of downloading stock lists.
        # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security issues if the endpoint changes or is deprecated.
        )
        resp = requests.get(url, headers=DEFAULT_SH_HEADER)
        self.download_stock_list(response=resp, exchange="sh")

        url = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1&random=0.20932135244582617"
        resp = requests.get(url, headers=DEFAULT_SZ_HEADER)
        self.download_stock_list(response=resp, exchange="sz")

    def download_stock_list(self, response, exchange):
        # ⚠️ SAST Risk (Low): No error handling for the HTTP request, which may lead to unhandled exceptions.
        # 🧠 ML Signal: Usage of requests.get to fetch data from a URL.
        # 🧠 ML Signal: Method call with specific parameters indicating a pattern of downloading stock lists.
        df = None
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        if exchange == "sh":
            df = pd.read_csv(
                io.BytesIO(response.content),
                sep="\s+",
                encoding="GB2312",
                dtype=str,
                parse_dates=["上市日期"],
                date_format="%Y-m-d",
                on_bad_lines="skip",
            )
            print(df)
            if df is not None:
                df = df.loc[:, ["公司代码", "公司简称", "上市日期"]]

        elif exchange == "sz":
            df = pd.read_excel(
                io.BytesIO(response.content),
                sheet_name="A股列表",
                # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
                dtype=str,
                parse_dates=["A股上市日期"],
                # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
                date_format="%Y-m-d",
            )
            if df is not None:
                df = df.loc[:, ["A股代码", "A股简称", "A股上市日期"]]

        if df is not None:
            df.columns = ["code", "name", "list_date"]

            df = df.dropna(subset=["code"])

            # 🧠 ML Signal: Usage of a database function to persist data
            # handle the dirty data
            # 600996,贵广网络,2016-12-26,2016-12-26,sh,stock,stock_sh_600996,,次新股,贵州,,
            # 🧠 ML Signal: Usage of a database function to persist data
            df.loc[df["code"] == "600996", "list_date"] = "2016-12-26"
            print(df[df["list_date"] == "-"])
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # ⚠️ SAST Risk (Low): Logging data frames can expose sensitive data in logs
            # 🧠 ML Signal: Instantiation and usage of a class
            # 🧠 ML Signal: Execution of a method within a class
            print(df["list_date"])
            df["list_date"] = df["list_date"].apply(lambda x: to_pd_timestamp(x))
            df["exchange"] = exchange
            df["entity_type"] = "stock"
            df["id"] = df[["entity_type", "exchange", "code"]].apply(lambda x: "_".join(x.astype(str)), axis=1)
            df["entity_id"] = df["id"]
            df["timestamp"] = df["list_date"]
            df = df.dropna(axis=0, how="any")
            df = df.drop_duplicates(subset=("id"), keep="last")
            df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=False)
            # persist StockDetail too
            df_to_db(df=df, data_schema=StockDetail, provider=self.provider, force_update=False)
            self.logger.info(df.tail())
            self.logger.info("persist stock list successs")


__all__ = ["ExchangeStockMetaRecorder"]

if __name__ == "__main__":
    recorder = ExchangeStockMetaRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["ExchangeStockMetaRecorder"]