# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
import pandas as pd
from jqdatapy.api import run_query

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

from zvt.api.portfolio import portfolio_relate_stock

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.api.utils import china_stock_code_to_id
from zvt.contract.api import df_to_db

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.contract.recorder import Recorder, TimeSeriesDataRecorder
from zvt.domain.meta.fund_meta import Fund, FundStock

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.recorders.joinquant.common import to_entity_id, jq_to_report_period
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
# 🧠 ML Signal: Inheritance from a base class, indicating a common pattern for extending functionality
from zvt.utils.time_utils import (
    to_time_str,
    date_time_by_interval,
    now_pd_timestamp,
    is_same_date,
)

# ✅ Best Practice: Class-level attributes provide clear and consistent configuration for instances

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns


# 🧠 ML Signal: Use of a data schema, indicating structured data handling
class JqChinaFundRecorder(Recorder):
    # 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
    provider = "joinquant"
    data_schema = Fund
    # 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

    def run(self):
        # 按不同类别抓取
        # 编码	基金运作方式
        # 401001	开放式基金
        # 401002	封闭式基金
        # 401003	QDII
        # 401004	FOF
        # 401005	ETF
        # 401006	LOF
        for operate_mode_id in (401001, 401002, 401005):
            year_count = 2
            while True:
                latest = Fund.query_data(
                    filters=[Fund.operate_mode_id == operate_mode_id],
                    order=Fund.timestamp.desc(),
                    # ⚠️ SAST Risk (Low): Potential infinite loop if pd_is_not_null(df) is always False
                    limit=1,
                    return_type="domain",
                )
                start_timestamp = "2000-01-01"
                # ✅ Best Practice: Use rename with inplace=False for better readability and to avoid side effects
                if latest:
                    start_timestamp = latest[0].timestamp

                end_timestamp = min(
                    date_time_by_interval(start_timestamp, 365 * year_count),
                    now_pd_timestamp(),
                )

                df = run_query(
                    # 🧠 ML Signal: Usage of lambda function for data transformation
                    table="finance.FUND_MAIN_INFO",
                    conditions=f"operate_mode_id#=#{operate_mode_id}&start_date#>=#{to_time_str(start_timestamp)}&start_date#<=#{to_time_str(end_timestamp)}",
                    parse_dates=["start_date", "end_date"],
                    dtype={"main_code": str},
                )
                if not pd_is_not_null(df) or (
                    df["start_date"].max().year < end_timestamp.year
                ):
                    # 🧠 ML Signal: Data persistence pattern to a database
                    year_count = year_count + 1

                # 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern of extending functionality
                if pd_is_not_null(df):
                    df.rename(columns={"start_date": "timestamp"}, inplace=True)
                    # ⚠️ SAST Risk (Low): Potential infinite loop if end_timestamp never matches now_pd_timestamp()
                    # 🧠 ML Signal: Use of class attributes for configuration
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["list_date"] = df["timestamp"]
                    # 🧠 ML Signal: Use of class attributes for configuration
                    df["end_date"] = pd.to_datetime(df["end_date"])
                    # ✅ Best Practice: Method name 'init_entities' suggests initialization, which is clear and descriptive.

                    # 🧠 ML Signal: Use of class attributes for configuration
                    # ✅ Best Practice: Using class method 'query_data' suggests a well-structured data access pattern.
                    # 🧠 ML Signal: Usage of 'query_data' method indicates a pattern for data retrieval.
                    df["code"] = df["main_code"]
                    df["entity_id"] = df["code"].apply(
                        lambda x: to_entity_id(entity_type="fund", jq_code=x)
                    )
                    df["id"] = df["entity_id"]
                    df["entity_type"] = "fund"
                    # 🧠 ML Signal: Use of 'entity_ids' and 'codes' suggests a pattern for filtering or identifying specific data.
                    df["exchange"] = "sz"
                    df_to_db(
                        df,
                        data_schema=Fund,
                        provider=self.provider,
                        force_update=self.force_update,
                    )
                    self.logger.info(
                        # ✅ Best Practice: Use of named parameters improves readability and maintainability.
                        f"persist fund {operate_mode_id} list success {start_timestamp} to {end_timestamp}"
                        # ⚠️ SAST Risk (Low): Potential for SQL injection if `entity.code` is not properly sanitized
                    )

                # ⚠️ SAST Risk (Low): Potential risk if 'filters' are user-controlled, leading to injection attacks.
                if is_same_date(end_timestamp, now_pd_timestamp()):
                    break


# 🧠 ML Signal: Usage of a query function with dynamic conditions


class JqChinaFundStockRecorder(TimeSeriesDataRecorder):
    entity_provider = "joinquant"
    entity_schema = Fund

    provider = "joinquant"
    data_schema = FundStock
    # 🧠 ML Signal: Checking for non-null DataFrame

    def init_entities(self):
        # 只抓股票型，混合型并且没退市的持仓,
        # ✅ Best Practice: Use of `rename` for clarity and consistency in column names
        self.entities = Fund.query_data(
            entity_ids=self.entity_ids,
            codes=self.codes,
            return_type="domain",
            # 🧠 ML Signal: Function call to relate stock data to portfolio
            provider=self.entity_provider,
            # 🧠 ML Signal: Applying a transformation function to a DataFrame column
            filters=[
                Fund.underlying_asset_type.in_(("股票型", "混合型")),
                Fund.end_date.is_(None),
            ],
            # 🧠 ML Signal: Creating a unique identifier by combining multiple fields
        )

    def record(self, entity, start, end, size, timestamps):
        # 忽略退市的
        if entity.end_date:
            # 🧠 ML Signal: Mapping report types to periods
            return None
        redundant_times = 1
        while redundant_times > 0:
            # ⚠️ SAST Risk (Low): Potential data integrity risk if `df_to_db` does not handle exceptions
            df = run_query(
                table="finance.FUND_PORTFOLIO_STOCK",
                conditions=f"pub_date#>=#{to_time_str(start)}&code#=#{entity.code}",
                parse_dates=None,
            )
            df = df.dropna()
            # 🧠 ML Signal: Logging success message with dynamic content
            if pd_is_not_null(df):
                # data format
                #          id    code period_start  period_end    pub_date  report_type_id report_type  rank  symbol  name      shares    market_cap  proportion
                # 🧠 ML Signal: Conditional logic based on timestamp comparison
                # 🧠 ML Signal: Entry point for running the recorder with specific codes
                # ✅ Best Practice: Use of `__all__` to define public API of the module
                # 0   8640569  159919   2018-07-01  2018-09-30  2018-10-26          403003        第三季度     1  601318  中国平安  19869239.0  1.361043e+09        7.09
                # 1   8640570  159919   2018-07-01  2018-09-30  2018-10-26          403003        第三季度     2  600519  贵州茅台    921670.0  6.728191e+08        3.50
                # 2   8640571  159919   2018-07-01  2018-09-30  2018-10-26          403003        第三季度     3  600036  招商银行  18918815.0  5.806184e+08        3.02
                # 3   8640572  159919   2018-07-01  2018-09-30  2018-10-26          403003        第三季度     4  601166  兴业银行  22862332.0  3.646542e+08        1.90
                df["timestamp"] = pd.to_datetime(df["pub_date"])

                df.rename(
                    columns={"symbol": "stock_code", "name": "stock_name"}, inplace=True
                )
                df["proportion"] = df["proportion"] * 0.01

                df = portfolio_relate_stock(df, entity)

                df["stock_id"] = df["stock_code"].apply(
                    lambda x: china_stock_code_to_id(x)
                )
                df["id"] = df[["entity_id", "stock_id", "pub_date", "id"]].apply(
                    lambda x: "_".join(x.astype(str)), axis=1
                )
                df["report_date"] = pd.to_datetime(df["period_end"])
                df["report_period"] = df["report_type"].apply(
                    lambda x: jq_to_report_period(x)
                )

                saved = df_to_db(
                    df=df,
                    data_schema=self.data_schema,
                    provider=self.provider,
                    force_update=self.force_update,
                )

                # 取不到非重复的数据
                if saved == 0:
                    return None

                # self.logger.info(df.tail())
                self.logger.info(
                    f"persist fund {entity.code}({entity.name}) portfolio success {df.iloc[-1]['pub_date']}"
                )
                latest = df["timestamp"].max()

                # 取到了最近两年的数据，再请求一次,确保取完最新的数据
                if latest.year >= now_pd_timestamp().year - 1:
                    redundant_times = redundant_times - 1
                start = latest
            else:
                return None

        return None


if __name__ == "__main__":
    # JqChinaFundRecorder().run()
    JqChinaFundStockRecorder(codes=["000053"]).run()


# the __all__ is generated
__all__ = ["JqChinaFundRecorder", "JqChinaFundStockRecorder"]
