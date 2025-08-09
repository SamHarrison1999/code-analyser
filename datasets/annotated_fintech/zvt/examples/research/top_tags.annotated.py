# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.stats import get_top_performance_by_month
from zvt.domain import Stock1dHfqKdata
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.utils.time_utils import date_time_by_interval, month_end_date, is_same_date

# 🧠 ML Signal: Iterating over a function that returns data based on a date range

# 每月涨幅前30，市值90%分布在100亿以下
# 重复上榜的有1/4左右
# 连续两个月上榜的1/10左右
# 🧠 ML Signal: Accessing top 30 entities by index
def top_tags(data_provider="em", start_timestamp="2020-01-01", end_timestamp="2021-01-01"):
    records = []
    # 🧠 ML Signal: Querying data with specific parameters
    for _, timestamp, df in get_top_performance_by_month(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, list_days=250, data_provider=data_provider
    ):
        for entity_id in df.index[:30]:
            query_timestamp = timestamp
            while True:
                kdata = Stock1dHfqKdata.query_data(
                    provider=data_provider,
                    entity_id=entity_id,
                    start_timestamp=query_timestamp,
                    order=Stock1dHfqKdata.timestamp.asc(),
                    # ⚠️ SAST Risk (Low): Potential infinite loop if kdata is always None or turnover_rate is always 0
                    limit=1,
                    return_type="domain",
                # 🧠 ML Signal: Checking if a date is the end of the month
                )
                if not kdata or kdata[0].turnover_rate == 0:
                    # 🧠 ML Signal: Adjusting timestamp by interval
                    if is_same_date(query_timestamp, month_end_date(query_timestamp)):
                        break
                    query_timestamp = date_time_by_interval(query_timestamp)
                    continue
                # 🧠 ML Signal: Calculating market cap from turnover and turnover rate
                cap = kdata[0].turnover / kdata[0].turnover_rate
                # ✅ Best Practice: Use of __name__ guard to allow or prevent parts of code from being run when modules are imported
                # 🧠 ML Signal: Appending calculated data to records
                # 🧠 ML Signal: Printing the result of a function call
                break

            records.append(
                {"entity_id": entity_id, "timestamp": timestamp, "cap": cap, "score": df.loc[entity_id, "score"]}
            )

    return records


if __name__ == "__main__":
    print(top_tags())