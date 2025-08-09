# -*- coding: utf-8 -*-
import pandas as pd
from jqdatapy import get_token, get_money_flow

from zvt import zvt_config
from zvt.api.kdata import generate_kdata_id
# ✅ Best Practice: Group related imports together for better readability and organization
from zvt.contract import IntervalLevel
from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import StockMoneyFlow, Stock
from zvt.recorders.joinquant.common import to_jq_entity_id
from zvt.recorders.joinquant.misc.jq_index_money_flow_recorder import JoinquantIndexMoneyFlowRecorder
from zvt.utils.pd_utils import pd_is_not_null
# 🧠 ML Signal: Inheritance from FixedCycleDataRecorder indicates a pattern of extending functionality
from zvt.utils.time_utils import TIME_FORMAT_DAY, to_time_str

# 🧠 ML Signal: Use of a specific data provider suggests a pattern in data source preference

class JoinquantStockMoneyFlowRecorder(FixedCycleDataRecorder):
    # 🧠 ML Signal: Association with a specific schema indicates a pattern in data structure usage
    # 🧠 ML Signal: Repeated use of the same provider suggests a strong dependency or preference
    # 🧠 ML Signal: Use of a specific data schema indicates a pattern in data handling and processing
    entity_provider = "joinquant"
    entity_schema = Stock

    provider = "joinquant"
    data_schema = StockMoneyFlow

    def __init__(
        self,
        force_update=True,
        sleeping_time=10,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        code=None,
        codes=None,
        day_data=False,
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="ignore",
        start_timestamp=None,
        end_timestamp=None,
        # ✅ Best Practice: Call to superclass constructor ensures proper initialization
        level=IntervalLevel.LEVEL_1DAY,
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        compute_index_money_flow=False,
        return_unfinished=False,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            code,
            codes,
            day_data,
            entity_filters,
            ignore_failed,
            real_time,
            fix_duplicate_way,
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: Tracking the use of compute_index_money_flow for feature importance
            level,
            # 🧠 ML Signal: Function definition with parameters indicating a pattern of generating IDs
            kdata_use_begin_time,
            # ⚠️ SAST Risk (Medium): Storing sensitive information like username and password in code
            one_day_trading_minutes,
            # 🧠 ML Signal: Usage of a function to generate an ID based on entity and timestamp
            return_unfinished,
        # ✅ Best Practice: Using descriptive function and parameter names for clarity
        # 🧠 ML Signal: Method that triggers actions based on a condition
        )
        self.compute_index_money_flow = compute_index_money_flow
        # 🧠 ML Signal: Instantiation and execution of a specific recorder class
        get_token(zvt_config["jq_username"], zvt_config["jq_password"], force=True)
    # ⚠️ SAST Risk (Low): Potential for unhandled exceptions during execution
    # ⚠️ SAST Risk (Low): Potential risk if `self.end_timestamp` is not properly validated before use.

    # 🧠 ML Signal: Usage of external function `get_money_flow` with specific parameters.
    def generate_domain_id(self, entity, original_data):
        return generate_kdata_id(entity_id=entity.id, timestamp=original_data["timestamp"], level=self.level)

    def on_finish(self):
        # 🧠 ML Signal: Conditional logic affecting function call parameters.
        # 根据 个股资金流 计算 大盘资金流
        if self.compute_index_money_flow:
            # ✅ Best Practice: Dropping NaN values to ensure data integrity.
            # ⚠️ SAST Risk (Low): Assumes `pd_is_not_null` correctly identifies non-null DataFrames.
            # 🧠 ML Signal: Adding a new column with a constant value.
            # ✅ Best Practice: Renaming columns for clarity and consistency.
            JoinquantIndexMoneyFlowRecorder().run()

    def record(self, entity, start, end, size, timestamps):
        if not self.end_timestamp:
            df = get_money_flow(code=to_jq_entity_id(entity), date=to_time_str(start))
        else:
            df = get_money_flow(code=to_jq_entity_id(entity), date=start, end_date=to_time_str(self.end_timestamp))

        df = df.dropna()

        if pd_is_not_null(df):
            df["name"] = entity.name
            df.rename(
                columns={
                    "date": "timestamp",
                    "net_amount_main": "net_main_inflows",
                    "net_pct_main": "net_main_inflow_rate",
                    "net_amount_xl": "net_huge_inflows",
                    "net_pct_xl": "net_huge_inflow_rate",
                    "net_amount_l": "net_big_inflows",
                    "net_pct_l": "net_big_inflow_rate",
                    "net_amount_m": "net_medium_inflows",
                    "net_pct_m": "net_medium_inflow_rate",
                    # ✅ Best Practice: Using a list to manage related column names.
                    "net_amount_s": "net_small_inflows",
                    "net_pct_s": "net_small_inflow_rate",
                },
                inplace=True,
            )

            # ✅ Best Practice: Converting data to numeric type with error handling.
            # 转换到标准float
            inflows_cols = [
                "net_main_inflows",
                "net_huge_inflows",
                "net_big_inflows",
                "net_medium_inflows",
                "net_small_inflows",
            # ✅ Best Practice: Dropping NaN values to ensure data integrity.
            # ⚠️ SAST Risk (Low): Assumes `pd_is_not_null` correctly identifies non-null DataFrames.
            ]
            for col in inflows_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # 🧠 ML Signal: Applying a transformation to a DataFrame column.
            df = df.dropna()

            # ✅ Best Practice: Using a list to manage related column names.
            if not pd_is_not_null(df):
                return None

            df[inflows_cols] = df[inflows_cols].apply(lambda x: x * 10000)

            inflow_rate_cols = [
                "net_main_inflow_rate",
                "net_huge_inflow_rate",
                "net_big_inflow_rate",
                # ✅ Best Practice: Converting data to numeric type with error handling.
                "net_medium_inflow_rate",
                "net_small_inflow_rate",
            # ✅ Best Practice: Dropping NaN values to ensure data integrity.
            # ✅ Best Practice: Use of string formatting for constructing unique identifiers
            ]
            for col in inflow_rate_cols:
                # ⚠️ SAST Risk (Low): Assumes `pd_is_not_null` correctly identifies non-null DataFrames.
                # 🧠 ML Signal: Applying a function across DataFrame rows
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            # 🧠 ML Signal: Dropping duplicates in a DataFrame
            if not pd_is_not_null(df):
                # 🧠 ML Signal: Applying a transformation to a DataFrame column.
                return None
            # 🧠 ML Signal: Saving DataFrame to a database

            # ⚠️ SAST Risk (Low): Potential division by zero if `df["net_main_inflow_rate"]` contains zeros.
            # ✅ Best Practice: Converting string to datetime for consistency and operations.
            # 🧠 ML Signal: Creating a new column based on arithmetic operations on existing columns.
            # 🧠 ML Signal: Adding a new column with a constant value.
            # 🧠 ML Signal: Running a specific function or class with a set of parameters
            # 🧠 ML Signal: Defining __all__ for module exports
            df[inflow_rate_cols] = df[inflow_rate_cols].apply(lambda x: x / 100)

            # 计算总流入
            df["net_inflows"] = (
                df["net_huge_inflows"] + df["net_big_inflows"] + df["net_medium_inflows"] + df["net_small_inflows"]
            )
            # 计算总流入率
            amount = df["net_main_inflows"] / df["net_main_inflow_rate"]
            df["net_inflow_rate"] = df["net_inflows"] / amount

            df["entity_id"] = entity.id
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["provider"] = "joinquant"
            df["code"] = entity.code

            def generate_kdata_id(se):
                return "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"], fmt=TIME_FORMAT_DAY))

            df["id"] = df[["entity_id", "timestamp"]].apply(generate_kdata_id, axis=1)

            df = df.drop_duplicates(subset="id", keep="last")

            df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)

        return None


if __name__ == "__main__":
    JoinquantStockMoneyFlowRecorder(codes=["000578"]).run()


# the __all__ is generated
__all__ = ["JoinquantStockMoneyFlowRecorder"]