# -*- coding: utf-8 -*-

import demjson3
import pandas as pd
import requests

from zvt import init_log
from zvt.api.kdata import generate_kdata_id, get_kdata
from zvt.contract import IntervalLevel
from zvt.contract.recorder import FixedCycleDataRecorder
# 🧠 ML Signal: Class definition for a specific data recorder, indicating a pattern of inheritance and specialization
from zvt.domain import Etf, Etf1dKdata
from zvt.recorders.consts import EASTMONEY_ETF_NET_VALUE_HEADER
# 🧠 ML Signal: Class attribute indicating the source of the entity data
from zvt.utils.time_utils import to_time_str

# 🧠 ML Signal: Class attribute indicating the schema used for the entity

# 🧠 ML Signal: Class attribute indicating the data provider
class ChinaETFDayKdataRecorder(FixedCycleDataRecorder):
    entity_provider = "exchange"
    entity_schema = Etf

    # ✅ Best Practice: Method should have a docstring to describe its purpose
    # 🧠 ML Signal: Class attribute indicating the schema used for the data
    provider = "sina"
    # 🧠 ML Signal: URL pattern for accessing data, indicating a pattern of constructing URLs for API requests
    data_schema = Etf1dKdata
    # ✅ Best Practice: Consider returning a more descriptive data structure if applicable
    # 🧠 ML Signal: Function definition with parameters indicating a pattern for generating domain IDs
    url = (
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?"
        # 🧠 ML Signal: Usage of a function call with specific parameters to generate an ID
        "symbol={}{}&scale=240&&datalen={}&ma=no"
    # ⚠️ SAST Risk (Low): Potential risk if `generate_kdata_id` is not properly validated or sanitized
    # ✅ Best Practice: Use of named parameters improves readability and maintainability
    # 🧠 ML Signal: Function processes financial data for an entity
    )

    def get_data_map(self):
        return {}

    def generate_domain_id(self, entity, original_data):
        return generate_kdata_id(entity_id=entity.id, timestamp=original_data["timestamp"], level=self.level)

    def on_finish_entity(self, entity):
        # ✅ Best Practice: Check if kdatas is not None before accessing its length
        kdatas = get_kdata(
            entity_id=entity.id,
            level=IntervalLevel.LEVEL_1DAY.value,
            order=Etf1dKdata.timestamp.asc(),
            # 🧠 ML Signal: Fetching cumulative net value for a date range
            return_type="domain",
            session=self.session,
            # ✅ Best Practice: Check if df is not None and not empty before processing
            filters=[Etf1dKdata.cumulative_net_value.is_(None)],
        )

        # ✅ Best Practice: Check if timestamp exists in df index before accessing
        if kdatas and len(kdatas) > 0:
            start = kdatas[0].timestamp
            end = kdatas[-1].timestamp

            # 从东方财富获取基金累计净值
            # ⚠️ SAST Risk (Low): Committing to the session without exception handling
            # ✅ Best Practice: Initialize variables outside the loop to avoid reinitialization
            df = self.fetch_cumulative_net_value(entity, start, end)
            # 🧠 ML Signal: Logging information about the update process

            if df is not None and not df.empty:
                for kdata in kdatas:
                    # ⚠️ SAST Risk (Medium): Potential for URL injection if security_item.code is not validated
                    if kdata.timestamp in df.index:
                        kdata.cumulative_net_value = df.loc[kdata.timestamp, "LJJZ"]
                        # ⚠️ SAST Risk (Medium): No error handling for network request failures
                        kdata.change_pct = df.loc[kdata.timestamp, "JZZZL"]
                self.session.commit()
                # ⚠️ SAST Risk (Medium): No error handling for JSON decoding
                self.logger.info(f"{entity.code} - {entity.name}累计净值更新完成...")

    # 🧠 ML Signal: Usage of external API and data fetching patterns
    def fetch_cumulative_net_value(self, security_item, start, end) -> pd.DataFrame:
        query_url = (
            "http://api.fund.eastmoney.com/f10/lsjz?" "fundCode={}&pageIndex={}&pageSize=200&startDate={}&endDate={}"
        )
        # ✅ Best Practice: Convert data types explicitly for consistency

        page = 1
        df = pd.DataFrame()
        while True:
            # ✅ Best Practice: Handle missing data to prevent errors in data processing
            url = query_url.format(security_item.code, page, to_time_str(start), to_time_str(end))
            # ⚠️ SAST Risk (Low): No validation on 'entity', 'start', 'end', 'size', and 'timestamps' inputs

            # ✅ Best Practice: Set index for DataFrame for efficient data manipulation
            response = requests.get(url, headers=EASTMONEY_ETF_NET_VALUE_HEADER)
            response_json = demjson3.decode(response.text)
            # ✅ Best Practice: Use pd.concat to efficiently append DataFrames
            response_df = pd.DataFrame(response_json["Data"]["LSJZList"])

            # 最后一页
            # 🧠 ML Signal: Custom sleep function usage pattern
            # ⚠️ SAST Risk (Medium): URL formatting with unvalidated 'security_item' could lead to SSRF
            if response_df.empty:
                break
            # ⚠️ SAST Risk (Medium): No error handling for network request

            response_df["FSRQ"] = pd.to_datetime(response_df["FSRQ"])
            # ⚠️ SAST Risk (Medium): No error handling for JSON decoding
            response_df["JZZZL"] = pd.to_numeric(response_df["JZZZL"], errors="coerce")
            response_df["LJJZ"] = pd.to_numeric(response_df["LJJZ"], errors="coerce")
            response_df = response_df.fillna(0)
            response_df.set_index("FSRQ", inplace=True, drop=True)
            # ✅ Best Practice: Use of pandas for data manipulation

            df = pd.concat([df, response_df])
            page += 1
            # ✅ Best Practice: Converting string to datetime for 'timestamp'

            self.sleep()

        return df
    # 🧠 ML Signal: Returning data as a list of dictionaries
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # ⚠️ SAST Risk (Low): No validation or error handling for logging initialization
    # 🧠 ML Signal: Pattern of running a main function in a script

    def record(self, entity, start, end, size, timestamps):
        # 此 url 不支持分页，如果超过我们想取的条数，则只能取最大条数
        if start is None or size > self.default_size:
            size = 8000

        param = {"security_item": entity, "level": self.level.value, "size": size}

        security_item = param["security_item"]
        size = param["size"]

        url = ChinaETFDayKdataRecorder.url.format(security_item.exchange, security_item.code, size)

        response = requests.get(url)
        response_json = demjson3.decode(response.text)

        if response_json is None or len(response_json) == 0:
            return []

        df = pd.DataFrame(response_json)
        df.rename(columns={"day": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["name"] = security_item.name
        df["provider"] = "sina"
        df["level"] = param["level"]

        return df.to_dict(orient="records")


__all__ = ["ChinaETFDayKdataRecorder"]

if __name__ == "__main__":
    init_log("sina_china_etf_day_kdata.log")
    ChinaETFDayKdataRecorder(level=IntervalLevel.LEVEL_1DAY).run()


# the __all__ is generated
__all__ = ["ChinaETFDayKdataRecorder"]