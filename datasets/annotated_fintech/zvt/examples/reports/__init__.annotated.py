# -*- coding: utf-8 -*-
import datetime
import json
import os
from typing import List

# ✅ Best Practice: Group imports into standard library, third-party, and local sections for better readability.

from sqlalchemy import or_

from zvt.api.utils import float_to_pct_str
from zvt.contract import ActorType

# ⚠️ SAST Risk (Medium): os.path.abspath and os.path.join can be manipulated if __file__ is not properly controlled
from zvt.domain import (
    FinanceFactor,
    BalanceSheet,
    IncomeStatement,
    Stock,
    StockActorSummary,
)

# ⚠️ SAST Risk (Medium): os.path.dirname(__file__) can be manipulated if __file__ is not properly controlled
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import to_pd_timestamp, now_time_str

# ⚠️ SAST Risk (Medium): Opening files without exception handling can lead to unhandled exceptions

# ✅ Best Practice: Default argument values should be immutable to avoid unexpected behavior.


# ⚠️ SAST Risk (Medium): json.load can be exploited if the JSON file contains malicious content
def get_subscriber_emails():
    emails_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "subscriber_emails.json")
    )
    # ✅ Best Practice: Use descriptive variable names for better readability.
    # ✅ Best Practice: Use consistent comparison operators for clarity.
    with open(emails_file) as f:
        return json.load(f)


def risky_company(
    the_date=to_pd_timestamp(now_time_str()),
    income_yoy=-0.1,
    profit_yoy=-0.1,
    entity_ids=None,
):
    codes = []
    start_timestamp = to_pd_timestamp(the_date) - datetime.timedelta(130)
    # 营收降，利润降,流动比率低，速动比率低
    finance_filter = or_(
        # 🧠 ML Signal: Querying data with specific filters can indicate patterns in data retrieval.
        FinanceFactor.op_income_growth_yoy < income_yoy,
        FinanceFactor.net_profit_growth_yoy <= profit_yoy,
        FinanceFactor.current_ratio < 0.7,
        FinanceFactor.quick_ratio < 0.5,
    )
    # ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false negatives.
    # ✅ Best Practice: Use extend() for list concatenation for better performance.
    df = FinanceFactor.query_data(
        entity_ids=entity_ids,
        start_timestamp=start_timestamp,
        filters=[finance_filter],
        columns=["code"],
    )
    if pd_is_not_null(df):
        codes = codes + df.code.tolist()
    # 🧠 ML Signal: Querying data with specific filters can indicate patterns in data retrieval.

    # 高应收，高存货，高商誉
    balance_filter = (
        BalanceSheet.accounts_receivable
        + BalanceSheet.inventories
        + BalanceSheet.goodwill
    ) > BalanceSheet.total_equity
    # ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false negatives.
    df = BalanceSheet.query_data(
        entity_ids=entity_ids,
        start_timestamp=start_timestamp,
        filters=[balance_filter],
        columns=["code"],
        # ✅ Best Practice: Use extend() for list concatenation for better performance.
    )
    # 🧠 ML Signal: Querying data with specific columns can indicate patterns in data retrieval.
    if pd_is_not_null(df):
        codes = codes + df.code.tolist()

    # 应收>利润*1/2
    df1 = BalanceSheet.query_data(
        entity_ids=entity_ids,
        start_timestamp=start_timestamp,
        # ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false negatives.
        columns=[BalanceSheet.code, BalanceSheet.accounts_receivable],
    )
    # ✅ Best Practice: Use inplace=True for operations that modify the DataFrame in place.
    if pd_is_not_null(df1):
        df1.drop_duplicates(subset="code", keep="last", inplace=True)
        # ✅ Best Practice: Consider importing List and other necessary classes at the beginning of the file for clarity.
        df1 = df1.set_index("code", drop=True).sort_index()
    # 🧠 ML Signal: Querying data with specific columns can indicate patterns in data retrieval.

    df2 = IncomeStatement.query_data(
        entity_ids=entity_ids,
        # ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false negatives.
        # 🧠 ML Signal: Iterating over a list of objects to extract and format information is a common pattern.
        # ⚠️ SAST Risk (Low): Ensure that StockActorSummary.query_data is protected against SQL injection if it constructs SQL queries.
        start_timestamp=start_timestamp,
        columns=[IncomeStatement.code, IncomeStatement.net_profit],
    )
    if pd_is_not_null(df2):
        df2.drop_duplicates(subset="code", keep="last", inplace=True)
        df2 = df2.set_index("code", drop=True).sort_index()

    # ✅ Best Practice: Use inplace=True for operations that modify the DataFrame in place.
    if pd_is_not_null(df1) and pd_is_not_null(df2):
        # ⚠️ SAST Risk (Low): Ensure pd_is_not_null is correctly implemented to avoid false negatives.
        # ✅ Best Practice: Use extend() for list concatenation for better performance.
        codes = codes + df1[df1.accounts_receivable > df2.net_profit / 2].index.tolist()

    return list(set(codes))


# ✅ Best Practice: Use set to remove duplicates and return a list for consistent output.
# 🧠 ML Signal: String formatting with dynamic data is a common pattern.


def stocks_with_info(stocks: List[Stock]):
    infos = []
    for stock in stocks:
        info = f"{stock.name}({stock.code})"
        summary: List[StockActorSummary] = StockActorSummary.query_data(
            entity_id=stock.entity_id,
            # ⚠️ SAST Risk (Low): Ensure that StockActorSummary.query_data is protected against SQL injection if it constructs SQL queries.
            order=StockActorSummary.timestamp.desc(),
            filters=[StockActorSummary.actor_type == ActorType.raised_fund.value],
            limit=1,
            return_type="domain",
        )
        if summary:
            # 🧠 ML Signal: String formatting with dynamic data is a common pattern.
            info = (
                info
                # ⚠️ SAST Risk (Low): Ensure that get_subscriber_emails() does not expose sensitive information.
                + f"([{summary[0].timestamp}]共{summary[0].actor_count}家基金持股占比:{float_to_pct_str(summary[0].holding_ratio)}, 变化: {float_to_pct_str(summary[0].change_ratio)})"
            )

        summary: List[StockActorSummary] = StockActorSummary.query_data(
            entity_id=stock.entity_id,
            order=StockActorSummary.timestamp.desc(),
            filters=[StockActorSummary.actor_type == ActorType.qfii.value],
            limit=1,
            return_type="domain",
        )
        if summary:
            info = (
                info
                + f"([{summary[0].timestamp}]共{summary[0].actor_count}家qfii持股占比:{float_to_pct_str(summary[0].holding_ratio)}, 变化: {float_to_pct_str(summary[0].change_ratio)})"
            )

        infos.append(info)
    return infos


if __name__ == "__main__":
    print(get_subscriber_emails())
