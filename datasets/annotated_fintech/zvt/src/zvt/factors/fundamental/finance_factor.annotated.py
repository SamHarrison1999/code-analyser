# -*- coding: utf-8 -*-
import operator
from itertools import accumulate
from typing import List, Union, Type
# ⚠️ SAST Risk (Low): Importing from external modules without validation can introduce security risks if the modules are compromised.

import pandas as pd

from zvt.contract import IntervalLevel, Mixin, TradableEntity
from zvt.contract.factor import Factor, Transformer, Accumulator
from zvt.domain import FinanceFactor, BalanceSheet, Stock


class FinanceBaseFactor(Factor):
    def __init__(
        self,
        data_schema: Type[Mixin] = FinanceFactor,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = None,
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = None,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        fill_method: str = "ffill",
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        # ✅ Best Practice: Check if 'columns' is None before assigning a default value
        only_compute_factor: bool = False,
        factor_name: str = None,
        # 🧠 ML Signal: Usage of a method to get important columns from a schema
        # ✅ Best Practice: Use of super() to initialize the parent class
        clear_state: bool = False,
        only_load_factor: bool = False,
    ) -> None:
        if not columns:
            columns = data_schema.important_cols()
        super().__init__(
            data_schema,
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            keep_all_timestamp,
            fill_method,
            effective_number,
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            # 🧠 ML Signal: Inheritance from a base class, indicating use of object-oriented design patterns
            factor_name,
            clear_state,
            only_load_factor,
        )


class GoodCompanyFactor(FinanceBaseFactor):
    def __init__(
        self,
        data_schema: Type[Mixin] = FinanceFactor,
        entity_schema: TradableEntity = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = (
            FinanceFactor.roe,
            FinanceFactor.op_income_growth_yoy,
            FinanceFactor.net_profit_growth_yoy,
            FinanceFactor.report_period,
            FinanceFactor.op_net_cash_flow_per_op_income,
            FinanceFactor.sales_net_cash_flow_per_op_income,
            FinanceFactor.current_ratio,
            FinanceFactor.debt_asset_ratio,
        ),
        filters: List = (
            FinanceFactor.roe >= 0.02,
            FinanceFactor.op_income_growth_yoy >= 0.05,
            FinanceFactor.net_profit_growth_yoy >= 0.05,
            FinanceFactor.op_net_cash_flow_per_op_income >= 0.1,
            FinanceFactor.sales_net_cash_flow_per_op_income >= 0.3,
            FinanceFactor.current_ratio >= 1,
            FinanceFactor.debt_asset_ratio <= 0.5,
        ),
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = True,
        fill_method: str = "ffill",
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        window="1095d",
        count=8,
        # ✅ Best Practice: Check for conditions before modifying lists to avoid unintended side effects.
        col_period_threshold={"roe": 0.02},
    ) -> None:
        # ⚠️ SAST Risk (Low): Modifying a default mutable argument (columns) can lead to unexpected behavior.
        # 🧠 ML Signal: Logging usage patterns can be used to train models for log analysis or anomaly detection.
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
        self.window = window
        self.count = count

        # 对于根据年度计算才有意义的指标，比如roe,我们会对不同季度的值区别处理,传入的参数为季度值
        self.col_period_threshold = col_period_threshold
        if self.col_period_threshold:
            if "report_period" not in columns and (data_schema.report_period not in columns):
                columns.append(data_schema.report_period)

        self.logger.info(f"using data_schema:{data_schema.__name__}")

        super().__init__(
            data_schema,
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            keep_all_timestamp,
            fill_method,
            # ✅ Best Practice: Initialize Series with a specific dtype to avoid deprecation warnings
            effective_number,
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
        )

    def compute_factor(self):
        def filter_df(df):
            se = pd.Series(index=df.index)
            # ⚠️ SAST Risk (High): Use of eval() can lead to code injection vulnerabilities
            for index, row in df.iterrows():

                if row.report_period == "year":
                    mul = 4
                elif row.report_period == "season3":
                    mul = 3
                # 🧠 ML Signal: Conditional logic based on attribute presence
                elif row.report_period == "half_year":
                    mul = 2
                else:
                    # ✅ Best Practice: Consider using meaningful variable names for readability
                    mul = 1

                filters = []
                for col in self.col_period_threshold:
                    # 🧠 ML Signal: Usage of DataFrame apply method with lambda function
                    col_se = eval(f"row.{col}")
                    filters.append(col_se >= mul * self.col_period_threshold[col])
                # 🧠 ML Signal: Logging information about data processing
                # ✅ Best Practice: Explicitly setting column names for clarity
                se[index] = list(accumulate(filters, func=operator.__and__))[-1]

            # 🧠 ML Signal: Logging information with dynamic content
            # 🧠 ML Signal: Instantiation of GoodCompanyFactor with specific parameters
            return se

        if self.col_period_threshold:
            self.factor_df = self.data_df.loc[lambda df: filter_df(df), :]

        self.factor_df = pd.DataFrame(index=self.data_df.index, columns=["count"], data=1)

        self.factor_df = self.factor_df.reset_index(level=1)
        # 🧠 ML Signal: Printing the result of a DataFrame
        # ✅ Best Practice: Defining __all__ for module exports

        self.factor_df = self.factor_df.groupby(level=0).rolling(window=self.window, on=self.time_field).count()

        self.factor_df = self.factor_df.reset_index(level=0, drop=True)
        self.factor_df = self.factor_df.set_index(self.time_field, append=True)

        self.factor_df = self.factor_df.loc[(slice(None), slice(self.start_timestamp, self.end_timestamp)), :]

        self.logger.info("factor:{},factor_df:\n{}".format(self.name, self.factor_df))

    def compute_result(self):
        self.result_df = self.factor_df.apply(lambda x: x >= self.count)
        self.result_df.columns = ["filter_score"]

        self.logger.info("factor:{},result_df:\n{}".format(self.name, self.result_df))


if __name__ == "__main__":
    # f1 = GoodCompanyFactor(keep_all_timestamp=False)
    # print(f1.result_df)

    # 高股息 低应收
    factor2 = GoodCompanyFactor(
        data_schema=BalanceSheet,
        columns=[BalanceSheet.accounts_receivable],
        filters=[BalanceSheet.accounts_receivable <= 0.2 * BalanceSheet.total_current_assets],
        keep_all_timestamp=False,
        col_period_threshold=None,
    )
    print(factor2.result_df)


# the __all__ is generated
__all__ = ["FinanceBaseFactor", "GoodCompanyFactor"]