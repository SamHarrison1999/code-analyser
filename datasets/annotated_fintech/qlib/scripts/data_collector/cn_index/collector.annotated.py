# ✅ Best Practice: Import statements should be grouped and ordered by standard library, third-party, and local imports
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import abc
import sys
from io import BytesIO
from typing import List, Iterable
from pathlib import Path

import fire
import requests
import pandas as pd
import baostock as bs

# ✅ Best Practice: Constants should be defined in uppercase
from tqdm import tqdm
from loguru import logger

# ✅ Best Practice: Avoid modifying sys.path; consider using a virtual environment or package management
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

# ✅ Best Practice: Import statements should be grouped and ordered by standard library, third-party, and local imports
from data_collector.index import IndexBase
from data_collector.utils import (
    get_calendar_list,
    get_trading_date_by_shift,
    deco_retry,
)
from data_collector.utils import get_instruments


# ✅ Best Practice: Use type hinting for the return type of the function
# ✅ Best Practice: Constants should be defined in uppercase
NEW_COMPANIES_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{index_code}cons.xls"
    # ✅ Best Practice: Use a more specific type hint for exclude_status, e.g., List[int]
)


# ✅ Best Practice: Constants should be defined in uppercase
# ⚠️ SAST Risk (Medium): getattr can be dangerous if method is not validated, leading to potential security risks
INDEX_CHANGES_URL = "https://www.csindex.com.cn/csindex-home/search/search-content?lang=cn&searchInput=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC&pageNum={page_num}&pageSize={page_size}&sortField=date&dateRange=all&contentType=announcement"

# ⚠️ SAST Risk (Medium): No timeout specified for the request, which can lead to hanging connections
REQ_HEADERS = {
    # ✅ Best Practice: Constants should be defined in uppercase
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36 Edg/91.0.864.48"
    # 🧠 ML Signal: Pattern of checking status codes for retry logic
    # ✅ Best Practice: Using @property decorator to define a getter for a class attribute
}


# 🧠 ML Signal: Raising exceptions based on HTTP status codes
# 🧠 ML Signal: Usage of decorators can indicate patterns in function behavior or error handling
# ✅ Best Practice: Returning a string directly for a simple property
@deco_retry
def retry_request(url: str, method: str = "get", exclude_status: List = None):
    if exclude_status is None:
        exclude_status = []
    method_func = getattr(requests, method)
    # 🧠 ML Signal: Method signature indicates a calculation function, common in financial models
    # ✅ Best Practice: Method name is descriptive of its functionality
    # ✅ Best Practice: Use of getattr with a default value to check for attribute existence
    _resp = method_func(url, headers=REQ_HEADERS, timeout=None)
    # ⚠️ SAST Risk (Low): Ensure 'data' is validated to prevent unexpected behavior
    _status = _resp.status_code
    if _status not in exclude_status and _status != 200:
        # ✅ Best Practice: Checking for empty data input
        # 🧠 ML Signal: Lazy loading pattern for caching data
        raise ValueError(f"response status: {_status}, url={url}")
    # ✅ Best Practice: Use of setattr to dynamically set an attribute
    return _resp


# 🧠 ML Signal: Iterating over data, common pattern in data processing


# 🧠 ML Signal: Method returning a formatted URL string, indicating a pattern of URL construction
class CSIIndex(IndexBase):
    @property
    # ⚠️ SAST Risk (Low): Ensure 'value' is of expected type to prevent type errors
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        _calendar = getattr(self, "_calendar_list", None)
        if not _calendar:
            _calendar = get_calendar_list(bench_code=self.index_name.upper())
            # ⚠️ SAST Risk (Low): Validate 'new_data' to prevent injection or corruption
            # ✅ Best Practice: Descriptive method name indicating its purpose
            # ✅ Best Practice: Checking for empty new_data input
            # ✅ Best Practice: Use of NotImplementedError to indicate that subclasses should implement this method
            setattr(self, "_calendar_list", _calendar)
        return _calendar

    @property
    # ✅ Best Practice: Include a docstring to describe the method's purpose and return value
    # 🧠 ML Signal: Placeholder for update logic, common in index management
    # ⚠️ SAST Risk (Low): Ensure update logic handles data safely
    # ✅ Best Practice: Use of @property and @abc.abstractmethod to enforce implementation in subclasses
    def new_companies_url(self) -> str:
        return NEW_COMPANIES_URL.format(index_code=self.index_code)

    @property
    def changes_url(self) -> str:
        # ✅ Best Practice: Use NotImplementedError to indicate that a method should be overridden
        return INDEX_CHANGES_URL

    @property
    # ✅ Best Practice: Docstring provides clear information about the method's purpose and expected return values
    # ✅ Best Practice: Use @property decorator for read-only access to a method's return value
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which is a placeholder and should be implemented to avoid runtime errors
        """
        raise NotImplementedError("rewrite bench_start_date")

    @property
    @abc.abstractmethod
    def index_code(self) -> str:
        """
        Returns
        -------
            index code
        # ✅ Best Practice: Use apply with lambda for concise transformation of DataFrame columns
        """
        raise NotImplementedError("rewrite index_code")

    @property
    def html_table_index(self) -> int:
        """Which table of changes in html

        CSI300: 0
        CSI100: 1
        :return:
        """
        raise NotImplementedError("rewrite html_table_index")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        # 🧠 ML Signal: Reading data from a URL is a common pattern in data ingestion tasks.
        if self.freq != "day":
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: (
                    pd.Timestamp(x) + pd.Timedelta(hours=9, minutes=30)
                ).strftime("%Y-%m-%d %H:%M:%S")
            )
            # ✅ Best Practice: Logging the end of a process helps in debugging and tracking execution flow.
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                # ✅ Best Practice: Docstring provides clear documentation of parameters and return value
                # ⚠️ SAST Risk (Low): Using pd.concat without specifying axis or handling duplicates can lead to unexpected results.
                lambda x: (
                    pd.Timestamp(x) + pd.Timedelta(hours=15, minutes=0)
                ).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        # ✅ Best Practice: Using context manager for file operations ensures proper resource management.
        logger.info("get companies changes......")
        res = []
        for _url in self._get_change_notices_url():
            _df = self._read_change_from_url(_url)
            # ⚠️ SAST Risk (Low): Writing content to a file without validation could lead to security risks if content is untrusted.
            if not _df.empty:
                res.append(_df)
        logger.info("get companies changes finish")
        # 🧠 ML Signal: Iterating over a list of tuples to process different types of data.
        return pd.concat(res, sort=False)

    # ⚠️ SAST Risk (Low): Accessing dictionary keys without checking existence could lead to KeyError.
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """

        Parameters
        ----------
        symbol: str
            symbol

        Returns
        -------
            symbol
        # 🧠 ML Signal: Concatenating multiple DataFrames into one, indicating data aggregation.
        # ✅ Best Practice: Ensure the correct table index is processed
        """
        symbol = f"{int(symbol):06}"
        return (
            f"SH{symbol}"
            if symbol.startswith("60") or symbol.startswith("688")
            else f"SZ{symbol}"
        )

    def _parse_excel(
        self, excel_url: str, add_date: pd.Timestamp, remove_date: pd.Timestamp
    ) -> pd.DataFrame:
        # 🧠 ML Signal: Processing specific columns for removal and addition
        content = retry_request(excel_url, exclude_status=[404]).content
        _io = BytesIO(content)
        df_map = pd.read_excel(_io, sheet_name=None)
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.{excel_url.split('.')[-1]}"
            # 🧠 ML Signal: Mapping symbols to a normalized form
        ).open("wb") as fp:
            fp.write(content)
        tmp = []
        for _s_name, _type, _date in [
            ("调入", self.ADD, add_date),
            ("调出", self.REMOVE, remove_date),
        ]:
            _df = df_map[_s_name]
            _df = _df.loc[_df["指数代码"] == self.index_code, ["证券代码"]]
            _df = _df.applymap(self.normalize_symbol)
            _df.columns = [self.SYMBOL_FIELD_NAME]
            # ⚠️ SAST Risk (Low): Potential file path manipulation vulnerability
            # ✅ Best Practice: Use pd.concat to combine DataFrames
            _df["type"] = _type
            _df[self.DATE_FIELD_NAME] = _date
            tmp.append(_df)
        df = pd.concat(tmp)
        return df

    def _parse_table(
        self, content: str, add_date: pd.DataFrame, remove_date: pd.DataFrame
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        _tmp_count = 0
        for _df in pd.read_html(content):
            if _df.shape[-1] != 4 or _df.isnull().loc(0)[0][0]:
                continue
            _tmp_count += 1
            if self.html_table_index + 1 > _tmp_count:
                continue
            tmp = []
            for _s, _type, _date in [
                (_df.iloc[2:, 0], self.REMOVE, remove_date),
                (_df.iloc[2:, 2], self.ADD, add_date),
            ]:
                _tmp_df = pd.DataFrame()
                _tmp_df[self.SYMBOL_FIELD_NAME] = _s.map(self.normalize_symbol)
                _tmp_df["type"] = _type
                _tmp_df[self.DATE_FIELD_NAME] = _date
                tmp.append(_tmp_df)
            # ⚠️ SAST Risk (Medium): Potential for HTTP request to fail or be intercepted, ensure retry_request is secure
            df = pd.concat(tmp)
            df.to_csv(
                str(
                    self.cache_dir.joinpath(
                        f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.csv"
                    ).resolve()
                )
                # ✅ Best Practice: Use logging for important information, aids in debugging and monitoring
            )
            break
        return df

    # 🧠 ML Signal: Regular expression usage can indicate pattern matching behavior

    def _read_change_from_url(self, url: str) -> pd.DataFrame:
        """read change from url
        The parameter url is from the _get_change_notices_url method.
        Determine the stock add_date/remove_date based on the title.
        The response contains three cases:
            1.Only excel_url(extract data from excel_url)
            2.Both the excel_url and the body text(try to extract data from excel_url first, and then try to extract data from body text)
            3.Only body text(extract data from body text)

        Parameters
        ----------
        url : str
            change url

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        # ✅ Best Practice: Use logging for important information, aids in debugging and monitoring
        """
        resp = retry_request(url).json()["data"]
        # ✅ Best Practice: Use logging for important information, aids in debugging and monitoring
        title = resp["title"]
        if not title.startswith("关于"):
            return pd.DataFrame()
        if "沪深300" not in title:
            return pd.DataFrame()

        # ✅ Best Practice: Use logging for important information, aids in debugging and monitoring
        logger.info(
            f"load index data from https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}"
        )
        _text = resp["content"]
        date_list = re.findall(r"(\d{4}).*?年.*?(\d+).*?月.*?(\d+).*?日", _text)
        if len(date_list) >= 2:
            add_date = pd.Timestamp("-".join(date_list[0]))
        else:
            _date = pd.Timestamp(
                "-".join(re.findall(r"(\d{4}).*?年.*?(\d+).*?月", _text)[0])
            )
            # ✅ Best Practice: Use of retry_request suggests handling of network errors
            add_date = get_trading_date_by_shift(self.calendar_list, _date, shift=0)
        if "盘后" in _text or "市后" in _text:
            # ✅ Best Practice: Re-fetching with total count ensures all data is retrieved
            add_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=1)
        remove_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=-1)

        # 🧠 ML Signal: URL construction pattern for accessing specific resources
        excel_url = None
        if resp.get("enclosureList", []):
            excel_url = resp["enclosureList"][0]["fileUrl"]
        else:
            excel_url_list = re.findall('.*href="(.*?xls.*?)".*', _text)
            if excel_url_list:
                excel_url = excel_url_list[0]
                if not excel_url.startswith("http"):
                    excel_url = (
                        excel_url if excel_url.startswith("/") else "/" + excel_url
                    )
                    excel_url = f"http://www.csindex.com.cn{excel_url}"
        if excel_url:
            try:
                # 🧠 ML Signal: Logging usage pattern
                logger.info(
                    f"get {add_date} changes from the excel, title={title}, excel_url={excel_url}"
                )
                # ⚠️ SAST Risk (Low): Potential risk if `self.new_companies_url` is user-controlled
                df = self._parse_excel(excel_url, add_date, remove_date)
            except ValueError:
                logger.info(
                    f"get {add_date} changes from the web page, title={title}, url=https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}"
                )
                df = self._parse_table(_text, add_date, remove_date)
        # ⚠️ SAST Risk (Low): Writing to file system, ensure `self.cache_dir` is secure
        else:
            logger.info(
                f"get {add_date} changes from the web page, title={title}, url=https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}"
                # ⚠️ SAST Risk (Low): Ensure the content is a valid Excel file to prevent parsing errors
            )
            df = self._parse_table(_text, add_date, remove_date)
        return df

    # ✅ Best Practice: Using @property decorator to define a read-only attribute
    # ✅ Best Practice: Use map for applying a function to a Series
    def _get_change_notices_url(self) -> Iterable[str]:
        """get change notices url

        Returns
        -------
            [url1, url2]
        # ✅ Best Practice: Using @property decorator to define a read-only attribute
        # ✅ Best Practice: Use of pd.Timestamp for date handling ensures consistency and compatibility with pandas operations.
        """
        page_num = 1
        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        page_size = 5
        # 🧠 ML Signal: Accessing class attribute through a property method
        data = retry_request(
            self.changes_url.format(page_size=page_size, page_num=page_num)
        ).json()
        # 🧠 ML Signal: Consistent return of a fixed value could indicate a placeholder or default implementation
        # ✅ Best Practice: Using @property decorator to define a method as a property for better encapsulation
        data = retry_request(
            self.changes_url.format(page_size=data["total"], page_num=page_num)
        ).json()
        for item in data["data"]:
            yield f"https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={item['id']}"

    # ⚠️ SAST Risk (Low): Ensure stock_prices is validated to prevent incorrect calculations
    # 🧠 ML Signal: Method returning a hardcoded value
    # ✅ Best Practice: Method to perform calculations on input data
    # ✅ Best Practice: Using a property to provide a read-only attribute

    def get_new_companies(self) -> pd.DataFrame:
        """

        Returns
        -------
            pd.DataFrame:

                symbol     start_date    end_date
                SH600000   2000-01-01    2099-12-31

            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        # 🧠 ML Signal: Returning a subset of a collection
        # ✅ Best Practice: Using len() to calculate average, assuming stock_prices is a list
        # 🧠 ML Signal: Method for retrieving top stocks, useful for portfolio optimization
        # ⚠️ SAST Risk (Low): Ensure stock_data is validated to prevent incorrect data processing
        # ✅ Best Practice: Default parameter value for top_n improves function usability
        # 🧠 ML Signal: Function that processes stock prices, useful for financial data analysis models
        # ⚠️ SAST Risk (Low): Ensure stock_prices is validated to prevent incorrect calculations
        # ✅ Best Practice: Docstring provides a clear description of the method's purpose and return type
        """
        logger.info("get new companies......")
        context = retry_request(self.new_companies_url).content
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_new_companies.{self.new_companies_url.split('.')[-1]}"
        ).open("wb") as fp:
            fp.write(context)
        _io = BytesIO(context)
        df = pd.read_excel(_io)
        df = df.iloc[:, [0, 4]]
        df.columns = [self.END_DATE_FIELD, self.SYMBOL_FIELD_NAME]
        df[self.SYMBOL_FIELD_NAME] = df[self.SYMBOL_FIELD_NAME].map(
            self.normalize_symbol
        )
        # ✅ Best Practice: Using sorted() with a key function for clarity and efficiency
        # ⚠️ SAST Risk (Low): Handling of empty input, but could be more explicit with error handling
        # 🧠 ML Signal: Iterating over stock prices, common pattern in financial computations
        df[self.END_DATE_FIELD] = pd.to_datetime(df[self.END_DATE_FIELD].astype(str))
        # ⚠️ SAST Risk (Low): Assumes price is a valid number, consider input validation
        # 🧠 ML Signal: Method chaining and function calls within return statements
        df[self.START_DATE_FIELD] = self.bench_start_date
        # ✅ Best Practice: Avoid division by zero by checking total_market_cap
        # ⚠️ SAST Risk (Low): Potential division by zero, handled by returning 0
        logger.info("end of get new companies.")
        return df


class CSI300Index(CSIIndex):
    @property
    def index_code(self):
        return "000300"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2005-01-01")

    # 🧠 ML Signal: Calculation of index value, relevant for financial modeling
    # ⚠️ SAST Risk (Low): Ensure bs.login() handles credentials securely and does not expose sensitive information.

    @property
    # 🧠 ML Signal: Usage of current timestamp to determine the end of a date range.
    def html_table_index(self) -> int:
        return 0


# ✅ Best Practice: Use of pd.date_range to generate a sequence of dates.


class CSI100Index(CSIIndex):
    # 🧠 ML Signal: Iterating over a date range with tqdm for progress tracking.
    @property
    # 🧠 ML Signal: Pattern of fetching data from an external source based on a date.
    def index_code(self):
        return "000903"

    # ✅ Best Practice: Use of pd.concat to combine a list of DataFrames.
    # ✅ Best Practice: Selecting specific columns from the result for further processing.
    # ⚠️ SAST Risk (Low): Ensure bs.logout() is called to terminate the session securely.
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2006-05-29")

    @property
    def html_table_index(self) -> int:
        return 1


class CSI500Index(CSIIndex):
    @property
    def index_code(self) -> str:
        return "000905"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        # ✅ Best Practice: Ensure the date is converted to string for compatibility with the API
        return pd.Timestamp("2007-01-15")

    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Return
        --------
           pd.DataFrame:
               symbol      date        type
               SH600000  2019-11-11    add
               SH600000  2020-11-10    remove
           dtypes:
               symbol: str
               date: pd.Timestamp
               type: str, value from ["add", "remove"]
        """
        return self.get_changes_with_history_companies(self.get_history_companies())

    def get_history_companies(self) -> pd.DataFrame:
        """

        Returns
        -------

            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        bs.login()
        today = pd.Timestamp.now()
        date_range = pd.DataFrame(
            pd.date_range(start="2007-01-15", end=today, freq="7D")
        )[0].dt.date
        ret_list = []
        for date in tqdm(date_range, desc="Download CSI500"):
            result = self.get_data_from_baostock(date)
            ret_list.append(result[["date", "symbol"]])
        bs.logout()
        return pd.concat(ret_list, sort=False)

    @staticmethod
    def get_data_from_baostock(date) -> pd.DataFrame:
        """
        Data source: http://baostock.com/baostock/index.php/%E4%B8%AD%E8%AF%81500%E6%88%90%E5%88%86%E8%82%A1
        Avoid a large number of parallel data acquisition,
        such as 1000 times of concurrent data acquisition, because IP will be blocked

        Returns
        -------
            pd.DataFrame:
                date      symbol        code_name
                SH600039  2007-01-15    四川路桥
                SH600051  2020-01-15    宁波联合
            dtypes:
                date: pd.Timestamp
                symbol: str
                code_name: str
        """
        col = ["date", "symbol", "code_name"]
        rs = bs.query_zz500_stocks(date=str(date))
        zz500_stocks = []
        while (rs.error_code == "0") & rs.next():
            zz500_stocks.append(rs.get_row_data())
        result = pd.DataFrame(zz500_stocks, columns=col)
        result["symbol"] = result["symbol"].apply(lambda x: x.replace(".", "").upper())
        return result

    def get_new_companies(self) -> pd.DataFrame:
        """

        Returns
        -------
            pd.DataFrame:

                symbol     start_date    end_date
                SH600000   2000-01-01    2099-12-31

            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        """
        logger.info("get new companies......")
        today = pd.Timestamp.now().normalize()
        bs.login()
        result = self.get_data_from_baostock(today.strftime("%Y-%m-%d"))
        bs.logout()
        df = result[["date", "symbol"]]
        df.columns = [self.END_DATE_FIELD, self.SYMBOL_FIELD_NAME]
        df[self.END_DATE_FIELD] = today
        df[self.START_DATE_FIELD] = self.bench_start_date
        logger.info("end of get new companies.")
        return df


if __name__ == "__main__":
    fire.Fire(get_instruments)
