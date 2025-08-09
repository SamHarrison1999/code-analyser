# -*- coding: utf-8 -*-
from zvt.domain import BalanceSheet
from zvt.factors.fundamental.finance_factor import GoodCompanyFactor
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is followed here.
from zvt.factors.target_selector import TargetSelector

# 🧠 ML Signal: Usage of a specific provider "eastmoney" could indicate a preference or dependency on this data source.

class FundamentalSelector(TargetSelector):
    def init_factors(self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, level):
        # 核心资产=(高ROE 高现金流 高股息 低应收 低资本开支 低财务杠杆 有增长)
        # 高roe 高现金流 低财务杠杆 有增长
        factor1 = GoodCompanyFactor(
            entity_ids=entity_ids,
            codes=codes,
            # ✅ Best Practice: Consider checking if self.factors is initialized before appending to avoid potential AttributeError.
            # 🧠 ML Signal: Usage of specific financial metrics and thresholds can indicate domain-specific knowledge or strategies.
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            provider="eastmoney",
        )

        self.factors.append(factor1)

        # 高股息 低应收
        factor2 = GoodCompanyFactor(
            data_schema=BalanceSheet,
            entity_ids=entity_ids,
            codes=codes,
            columns=[BalanceSheet.accounts_receivable],
            filters=[BalanceSheet.accounts_receivable <= 0.3 * BalanceSheet.total_current_assets],
            start_timestamp=start_timestamp,
            # ✅ Best Practice: Ensure the script is being run as the main module before executing main logic.
            # 🧠 ML Signal: The date range used here could be indicative of a specific period of interest for analysis.
            # 🧠 ML Signal: The run method invocation suggests a pattern of executing a sequence of operations.
            # 🧠 ML Signal: Retrieving targets for a specific date could indicate a focus on end-of-period analysis.
            end_timestamp=end_timestamp,
            provider="eastmoney",
            col_period_threshold=None,
        )
        self.factors.append(factor2)


if __name__ == "__main__":
    selector: TargetSelector = FundamentalSelector(start_timestamp="2015-01-01", end_timestamp="2019-06-30")
    selector.run()
    print(selector.get_targets("2019-06-30"))