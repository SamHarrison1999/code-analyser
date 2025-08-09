# -*- coding: utf-8 -*-

import io
import re

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import demjson3
import pandas as pd
import requests

from zvt.api.utils import china_stock_code_to_id
from zvt.contract.api import df_to_db

# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
from zvt.contract.recorder import Recorder
from zvt.domain import EtfStock, Etf

# ✅ Best Practice: Class attributes should be documented or initialized in __init__
from zvt.recorders.consts import DEFAULT_SH_ETF_LIST_HEADER

# ✅ Best Practice: Use of default parameter values for flexibility and ease of use
from zvt.utils.time_utils import now_pd_timestamp

# ✅ Best Practice: Proper use of super() to initialize the parent class


class ChinaETFListSpider(Recorder):
    # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security risks if the endpoint changes or is compromised.
    data_schema = EtfStock

    # ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which can lead to unhandled exceptions.
    def __init__(
        self, force_update=False, sleeping_time=10.0, provider="exchange"
    ) -> None:
        self.provider = provider
        # ⚠️ SAST Risk (Medium): No validation of the response before decoding, which can lead to security issues.
        super().__init__(force_update, sleeping_time)

    # 🧠 ML Signal: Usage of pandas DataFrame indicates data processing or analysis.
    def run(self):
        # 抓取沪市 ETF 列表
        # 🧠 ML Signal: Method call with a DataFrame and exchange parameter, indicating data persistence.
        url = "http://query.sse.com.cn/commonQuery.do?sqlId=COMMON_SSE_ZQPZ_ETFLB_L_NEW"
        response = requests.get(url, headers=DEFAULT_SH_ETF_LIST_HEADER)
        # ✅ Best Practice: Logging provides traceability and debugging information.
        response_dict = demjson3.decode(response.text)

        # 🧠 ML Signal: Method call with a DataFrame, indicating further data processing or downloading.
        df = pd.DataFrame(response_dict.get("result", []))
        self.persist_etf_list(df, exchange="sh")
        # ✅ Best Practice: Logging provides traceability and debugging information.
        self.logger.info("沪市 ETF 列表抓取完成...")
        # ✅ Best Practice: Check for None to avoid processing invalid data

        # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security risks if the endpoint changes or is compromised.
        # 抓取沪市 ETF 成分股
        self.download_sh_etf_component(df)
        # ✅ Best Practice: Use copy to avoid modifying the original DataFrame
        # ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which can lead to unhandled exceptions.
        self.logger.info("沪市 ETF 成分股抓取完成...")

        # 🧠 ML Signal: Conditional logic based on exchange type
        # ⚠️ SAST Risk (Medium): No validation of the response content before reading it as an Excel file.
        # 抓取深市 ETF 列表
        # 🧠 ML Signal: Usage of pandas to read Excel files indicates data processing or analysis.
        url = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1945"
        response = requests.get(url)
        # 🧠 ML Signal: Conditional logic based on exchange type
        # 🧠 ML Signal: Method call with a DataFrame and exchange parameter, indicating data persistence.

        df = pd.read_excel(io.BytesIO(response.content), dtype=str)
        # ✅ Best Practice: Logging provides traceability and debugging information.
        self.persist_etf_list(df, exchange="sz")
        # ✅ Best Practice: Standardize column names for consistency
        self.logger.info("深市 ETF 列表抓取完成...")
        # 🧠 ML Signal: Method call with a DataFrame, indicating further data processing or downloading.

        # 🧠 ML Signal: Use of lambda function for ID generation
        # 抓取深市 ETF 成分股
        # ✅ Best Practice: Logging provides traceability and debugging information.
        self.download_sz_etf_component(df)
        # ✅ Best Practice: Assigning new columns for clarity and future use
        self.logger.info("深市 ETF 成分股抓取完成...")

    def persist_etf_list(self, df: pd.DataFrame, exchange: str):
        if df is None:
            return
        # ✅ Best Practice: Remove rows with any NaN values to ensure data integrity
        # ✅ Best Practice: Use parentheses for multi-line strings for better readability.

        df = df.copy()
        # ✅ Best Practice: Remove duplicate entries to maintain unique records
        # 🧠 ML Signal: Filtering DataFrame based on specific column values.
        if exchange == "sh":
            df = df[["FUND_ID", "FUND_NAME"]]
        # ⚠️ SAST Risk (Low): Ensure df_to_db handles SQL injection and data validation
        # 🧠 ML Signal: Method call to populate additional data in DataFrame.
        elif exchange == "sz":
            df = df[["证券代码", "证券简称"]]
        # 🧠 ML Signal: Iterating over DataFrame rows.

        df.columns = ["code", "name"]
        # ⚠️ SAST Risk (Low): Potential for URL injection if ETF_TYPE or ETF_CLASS are not validated.
        df["id"] = df["code"].apply(lambda code: f"etf_{exchange}_{code}")
        df["entity_id"] = df["id"]
        # ⚠️ SAST Risk (Medium): No error handling for network request failures.
        df["exchange"] = exchange
        df["entity_type"] = "etf"
        # ⚠️ SAST Risk (Medium): No error handling for JSON decoding.
        df["category"] = "etf"

        # 🧠 ML Signal: Creating DataFrame from JSON response.
        df = df.dropna(axis=0, how="any")
        df = df.drop_duplicates(subset="id", keep="last")

        # 🧠 ML Signal: String formatting to create unique identifiers.
        df_to_db(df=df, data_schema=Etf, provider=self.provider, force_update=False)

    # ✅ Best Practice: Use .copy() to avoid SettingWithCopyWarning.
    def download_sh_etf_component(self, df: pd.DataFrame):
        query_url = (
            # ✅ Best Practice: Use rename with inplace=True for clarity and efficiency.
            "http://query.sse.com.cn/infodisplay/queryConstituentStockInfo.do?"
            "isPagination=false&type={}&etfClass={}"
            # ✅ Best Practice: Consider adding type hints for the method parameters and return type for better readability and maintainability.
        )

        etf_df = df[(df["ETF_CLASS"] == "1") | (df["ETF_CLASS"] == "2")]
        # 🧠 ML Signal: Logging usage pattern for information messages.
        etf_df = self.populate_sh_etf_type(etf_df)

        for _, etf in etf_df.iterrows():
            # ⚠️ SAST Risk (Low): Ensure now_pd_timestamp() is timezone-aware if needed.
            url = query_url.format(etf["ETF_TYPE"], etf["ETF_CLASS"])
            response = requests.get(url, headers=DEFAULT_SH_ETF_LIST_HEADER)
            # 🧠 ML Signal: Applying function to DataFrame column.
            response_dict = demjson3.decode(response.text)
            # 🧠 ML Signal: Logging usage pattern for information messages.
            response_df = pd.DataFrame(response_dict.get("result", []))
            # 🧠 ML Signal: Generating unique IDs for DataFrame entries.

            etf_code = etf["FUND_ID"]
            # ⚠️ SAST Risk (Medium): Ensure df_to_db handles SQL injection and data validation.
            etf_id = f"etf_sh_{etf_code}"
            # ⚠️ SAST Risk (Medium): No error handling for network request failures.
            # 🧠 ML Signal: Logging information about processed data.
            response_df = response_df[["instrumentId", "instrumentName"]].copy()
            response_df.rename(
                columns={"instrumentId": "stock_code", "instrumentName": "stock_name"},
                inplace=True,
            )

            # 🧠 ML Signal: Usage of sleep to manage request rate.
            response_df["entity_id"] = etf_id
            # ⚠️ SAST Risk (Low): Potentially unsafe HTML parsing without validation.
            response_df["entity_type"] = "etf"
            response_df["exchange"] = "sh"
            response_df["code"] = etf_code
            # 🧠 ML Signal: Logging usage pattern for error messages.
            response_df["name"] = etf["FUND_NAME"]
            response_df["timestamp"] = now_pd_timestamp()

            response_df["stock_id"] = response_df["stock_code"].apply(
                lambda code: china_stock_code_to_id(code)
            )
            response_df["id"] = response_df["stock_id"].apply(lambda x: f"{etf_id}_{x}")

            df_to_db(
                data_schema=self.data_schema, df=response_df, provider=self.provider
            )
            # ✅ Best Practice: Use f-strings for better readability and performance.
            self.logger.info(f'{etf["FUND_NAME"]} - {etf_code} 成分股抓取完成...')

            self.sleep()

    def download_sz_etf_component(self, df: pd.DataFrame):
        query_url = "http://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/indexid/{}.phtml"

        self.parse_sz_etf_underlying_index(df)
        for _, etf in df.iterrows():
            underlying_index = etf["拟合指数"]
            # ⚠️ SAST Risk (Low): Potential timezone issues if `now_pd_timestamp` is not timezone-aware.
            etf_code = etf["证券代码"]

            if len(underlying_index) == 0:
                self.logger.info(
                    f'{etf["证券简称"]} - {etf_code} 非 A 股市场指数，跳过...'
                )
                continue

            # ⚠️ SAST Risk (Medium): Ensure `df_to_db` handles SQL injection and data validation.
            # 🧠 ML Signal: Logging usage pattern for information messages.
            # ⚠️ SAST Risk (Low): Ensure `self.sleep()` does not introduce significant delays or blocking.
            # 🧠 ML Signal: Usage of a specific URL pattern for querying data
            url = query_url.format(underlying_index)
            response = requests.get(url)
            response.encoding = "gbk"

            try:
                # ✅ Best Practice: Initialize an empty DataFrame for concatenation
                dfs = pd.read_html(response.text, header=1)
            except ValueError as error:
                self.logger.error(
                    f"HTML parse error: {error}, response: {response.text}"
                )
                # 🧠 ML Signal: Iterating over a fixed set of ETF classes
                continue

            # ⚠️ SAST Risk (Low): No error handling for network request
            if len(dfs) < 4:
                continue
            # ⚠️ SAST Risk (Low): No error handling for JSON decoding

            response_df = dfs[3].copy()
            # ✅ Best Practice: Use .get() to safely access dictionary keys
            response_df = response_df.dropna(axis=1, how="any")
            response_df["品种代码"] = response_df["品种代码"].apply(
                lambda x: f"{x:06d}"
            )
            # ✅ Best Practice: Explicitly select relevant columns

            etf_id = f"etf_sz_{etf_code}"
            # ✅ Best Practice: Use pd.concat for DataFrame concatenation
            response_df = response_df[["品种代码", "品种名称"]].copy()
            # ✅ Best Practice: Use .copy() to avoid modifying the original DataFrame
            response_df.rename(
                columns={"品种代码": "stock_code", "品种名称": "stock_name"},
                inplace=True,
            )

            response_df["entity_id"] = etf_id
            response_df["entity_type"] = "etf"
            response_df["exchange"] = "sz"
            # ✅ Best Practice: Sort DataFrame by a specific column
            # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability.
            response_df["code"] = etf_code
            # ⚠️ SAST Risk (Low): Assumes type_df and result_df have the same length
            # ✅ Best Practice: Check for empty input to avoid unnecessary processing
            response_df["name"] = etf["证券简称"]
            response_df["timestamp"] = now_pd_timestamp()

            # ⚠️ SAST Risk (Low): Use of regular expressions can lead to ReDoS if not properly constrained
            response_df["stock_id"] = response_df["stock_code"].apply(
                lambda code: china_stock_code_to_id(code)
            )
            response_df["id"] = response_df["stock_id"].apply(lambda x: f"{etf_id}_{x}")

            df_to_db(
                data_schema=self.data_schema, df=response_df, provider=self.provider
            )
            self.logger.info(f'{etf["证券简称"]} - {etf_code} 成分股抓取完成...')
            # 🧠 ML Signal: Extracting numeric patterns from text

            self.sleep()

    # 🧠 ML Signal: Applying a function to a DataFrame column

    @staticmethod
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Instantiating and running a spider class
    # 🧠 ML Signal: Running a method on a class instance
    def populate_sh_etf_type(df: pd.DataFrame):
        """
        填充沪市 ETF 代码对应的 TYPE 到列表数据中
        :param df: ETF 列表数据
        :return: 包含 ETF 对应 TYPE 的列表数据
        """
        query_url = (
            "http://query.sse.com.cn/infodisplay/queryETFNewAllInfo.do?"
            "isPagination=false&type={}&pageHelp.pageSize=25"
        )

        type_df = pd.DataFrame()
        for etf_class in [1, 2]:
            url = query_url.format(etf_class)
            response = requests.get(url, headers=DEFAULT_SH_ETF_LIST_HEADER)
            response_dict = demjson3.decode(response.text)
            response_df = pd.DataFrame(response_dict.get("result", []))
            response_df = response_df[["fundid1", "etftype"]]

            type_df = pd.concat([type_df, response_df])

        result_df = df.copy()
        result_df = result_df.sort_values(by="FUND_ID").reset_index(drop=True)
        type_df = type_df.sort_values(by="fundid1").reset_index(drop=True)

        result_df["ETF_TYPE"] = type_df["etftype"]

        return result_df

    @staticmethod
    def parse_sz_etf_underlying_index(df: pd.DataFrame):
        """
        解析深市 ETF 对应跟踪的指数代码
        :param df: ETF 列表数据
        :return: 解析完成 ETF 对应指数代码的列表数据
        """

        def parse_index(text):
            if len(text) == 0:
                return ""

            result = re.search(r"(\d+).*", text)
            if result is None:
                return ""
            else:
                return result.group(1)

        df["拟合指数"] = df["拟合指数"].apply(parse_index)


__all__ = ["ChinaETFListSpider"]

if __name__ == "__main__":
    spider = ChinaETFListSpider(provider="exchange")
    spider.run()


# the __all__ is generated
__all__ = ["ChinaETFListSpider"]
