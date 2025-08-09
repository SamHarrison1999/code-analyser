# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
import pandas as pd
from jqdatapy.api import get_all_securities, run_query

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

from zvt.api.portfolio import portfolio_relate_stock

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.api.utils import china_stock_code_to_id
from zvt.contract.api import df_to_db, get_entity_exchange, get_entity_code

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.contract.recorder import Recorder, TimeSeriesDataRecorder
from zvt.domain import EtfStock, Stock, Etf, StockDetail

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.recorders.joinquant.common import to_entity_id, jq_to_report_period
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
# 🧠 ML Signal: Inheritance from a base class, indicating a common pattern for extending functionality
from zvt.utils.time_utils import to_time_str

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
# 🧠 ML Signal: Use of a class attribute to store constant values, indicating a pattern for configuration
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use


class BaseJqChinaMetaRecorder(Recorder):
    # 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
    # ✅ Best Practice: Setting the index to a specific column for better data manipulation
    # ✅ Best Practice: Proper use of super() to initialize the parent class
    provider = "joinquant"

    # 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
    def __init__(self, force_update=True, sleeping_time=10) -> None:
        # ✅ Best Practice: Resetting index to make 'entity_id' a column again
        super().__init__(force_update, sleeping_time)

    # ✅ Best Practice: Renaming columns for consistency and clarity
    def to_zvt_entity(self, df, entity_type, category=None):
        df = df.set_index("code")
        # ✅ Best Practice: Converting date strings to datetime objects for better date manipulation
        df.index.name = "entity_id"
        df = df.reset_index()
        # ✅ Best Practice: Assigning 'timestamp' to 'list_date' for clarity
        # 上市日期
        df.rename(columns={"start_date": "timestamp"}, inplace=True)
        # ✅ Best Practice: Converting date strings to datetime objects for better date manipulation
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["list_date"] = df["timestamp"]
        # 🧠 ML Signal: Usage of lambda function for transformation
        df["end_date"] = pd.to_datetime(df["end_date"])

        # ✅ Best Practice: Assigning 'entity_id' to 'id' for clarity
        df["entity_id"] = df["entity_id"].apply(
            lambda x: to_entity_id(entity_type=entity_type, jq_code=x)
        )
        df["id"] = df["entity_id"]
        # ✅ Best Practice: Storing 'entity_type' for each row for clarity
        # 🧠 ML Signal: Inheritance from a base class indicates a pattern of code reuse and specialization
        df["entity_type"] = entity_type
        df["exchange"] = df["entity_id"].apply(lambda x: get_entity_exchange(x))
        # 🧠 ML Signal: Usage of lambda function for transformation
        # 🧠 ML Signal: Assignment of a class attribute suggests a pattern of defining schema or structure
        # 🧠 ML Signal: Method call pattern for converting securities to a specific entity type
        df["code"] = df["entity_id"].apply(lambda x: get_entity_code(x))
        df["name"] = df["display_name"]
        # 🧠 ML Signal: Usage of lambda function for transformation
        # 🧠 ML Signal: Pattern of persisting data to a database

        # ⚠️ SAST Risk (Low): Potential risk of SQL injection if inputs are not sanitized
        if category:
            # ✅ Best Practice: Assigning 'display_name' to 'name' for clarity
            # ✅ Best Practice: Class definition should follow PEP 8 naming conventions for readability
            df["category"] = category
        # 🧠 ML Signal: Repeated pattern of persisting data with different schemas

        # 🧠 ML Signal: Method invocation pattern for running a process
        # ⚠️ SAST Risk (Low): Potential risk of SQL injection if inputs are not sanitized
        # 🧠 ML Signal: Usage of class-level attributes for schema definition
        return df


# ✅ Best Practice: Conditionally adding a column if 'category' is provided

# ✅ Best Practice: Logging success messages for better traceability and debugging
# 🧠 ML Signal: Data transformation pattern using a method call


class JqChinaStockRecorder(BaseJqChinaMetaRecorder):
    # 🧠 ML Signal: Data persistence pattern to a database
    data_schema = Stock
    # 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern of extending functionality

    # ✅ Best Practice: Logging for success confirmation
    def run(self):
        # 🧠 ML Signal: Use of class attributes for configuration
        # 抓取股票列表
        df_stock = self.to_zvt_entity(
            get_all_securities(code="stock"), entity_type="stock"
        )
        # 🧠 ML Signal: Use of class attributes for configuration
        df_to_db(
            df_stock,
            data_schema=Stock,
            provider=self.provider,
            force_update=self.force_update,
        )
        # 🧠 ML Signal: Use of class attributes for configuration
        # 🧠 ML Signal: Usage of a specific database table and query conditions
        # persist StockDetail too
        df_to_db(
            df=df_stock,
            data_schema=StockDetail,
            provider=self.provider,
            force_update=self.force_update,
        )

        # self.logger.info(df_stock)
        self.logger.info("persist stock list success")


# 🧠 ML Signal: Use of class attributes for configuration

# ⚠️ SAST Risk (Low): Potential SQL injection if `entity.code` is not sanitized


class JqChinaEtfRecorder(BaseJqChinaMetaRecorder):
    # ✅ Best Practice: Convert date strings to datetime objects for consistency
    data_schema = Etf

    # ✅ Best Practice: Use of `rename` for clarity and consistency in column names
    def run(self):
        # 抓取etf列表
        # ✅ Best Practice: Adjusting proportions to a standard scale
        df_index = self.to_zvt_entity(
            get_all_securities(code="etf"), entity_type="etf", category="etf"
        )
        df_to_db(
            df_index,
            data_schema=Etf,
            provider=self.provider,
            force_update=self.force_update,
        )
        # 🧠 ML Signal: Transformation of data with a specific function

        # self.logger.info(df_index)
        # 🧠 ML Signal: Mapping stock codes to IDs
        self.logger.info("persist etf list success")


# ⚠️ SAST Risk (Low): Potential for ID collision if not unique


# 🧠 ML Signal: Mapping report types to periods
# 🧠 ML Signal: Logging successful operations
# 🧠 ML Signal: Definition of module exports
# ✅ Best Practice: Convert date strings to datetime objects for consistency
# 🧠 ML Signal: Persisting transformed data to a database
# 🧠 ML Signal: Entry point for running the recorder with specific codes
class JqChinaStockEtfPortfolioRecorder(TimeSeriesDataRecorder):
    entity_provider = "joinquant"
    entity_schema = Etf

    # 数据来自jq
    provider = "joinquant"

    data_schema = EtfStock

    def record(self, entity, start, end, size, timestamps):
        df = run_query(
            table="finance.FUND_PORTFOLIO_STOCK",
            conditions=f"pub_date#>=#{to_time_str(start)}&code#=#{entity.code}",
            parse_dates=None,
        )
        if pd_is_not_null(df):
            #          id    code period_start  period_end    pub_date  report_type_id report_type  rank  symbol  name      shares    market_cap  proportion
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

            df["stock_id"] = df["stock_code"].apply(lambda x: china_stock_code_to_id(x))
            df["id"] = df[["entity_id", "stock_id", "pub_date", "id"]].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )
            df["report_date"] = pd.to_datetime(df["period_end"])
            df["report_period"] = df["report_type"].apply(
                lambda x: jq_to_report_period(x)
            )

            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )

            # self.logger.info(df.tail())
            self.logger.info(
                f"persist etf {entity.code} portfolio success {df.iloc[-1]['pub_date']}"
            )

        return None


if __name__ == "__main__":
    # JqChinaEtfRecorder().run()
    JqChinaStockEtfPortfolioRecorder(codes=["510050"]).run()


# the __all__ is generated
__all__ = [
    "BaseJqChinaMetaRecorder",
    "JqChinaStockRecorder",
    "JqChinaEtfRecorder",
    "JqChinaStockEtfPortfolioRecorder",
]
