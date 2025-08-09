# -*- coding: utf-8 -*-

import time

import pandas as pd
import requests

from zvt.api.kdata import generate_kdata_id, get_kdata_schema

# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.recorder import FixedCycleDataRecorder

# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
from zvt.domain import Index, IndexKdataCommon
from zvt.utils.time_utils import get_year_quarters, is_same_date

# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations


# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
# ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if not validated or sanitized
# 🧠 ML Signal: URL pattern, useful for understanding data source endpoints
class ChinaIndexDayKdataRecorder(FixedCycleDataRecorder):
    entity_provider = "exchange"
    entity_schema = Index

    provider = "sina"
    data_schema = IndexKdataCommon
    url = "http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/{}/type/S.phtml?year={}&jidu={}"

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
        # ✅ Best Practice: Convert level to IntervalLevel to ensure type consistency
        fix_duplicate_way="ignore",
        start_timestamp=None,
        # ✅ Best Practice: Set default adjust_type for clarity and consistency
        end_timestamp=None,
        level=IntervalLevel.LEVEL_1DAY,
        # ✅ Best Practice: Use of lower() for consistent entity type naming
        # ✅ Best Practice: Use of a function to get schema promotes modularity
        # ✅ Best Practice: Use of super() to ensure proper inheritance
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        return_unfinished=False,
    ) -> None:
        level = IntervalLevel(level)
        self.adjust_type = AdjustType.qfq
        self.entity_type = self.entity_schema.__name__.lower()

        self.data_schema = get_kdata_schema(
            entity_type=self.entity_type, level=level, adjust_type=self.adjust_type
        )

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
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and return value of the function
            ignore_failed,
            real_time,
            # ✅ Best Practice: Returning a dictionary directly is clear and concise
            # 🧠 ML Signal: Method signature and parameter usage pattern
            fix_duplicate_way,
            start_timestamp,
            # 🧠 ML Signal: Usage of external function with specific parameters
            end_timestamp,
            # ⚠️ SAST Risk (Low): Potential information exposure if entity.id or original_data["timestamp"] contains sensitive data
            level,
            kdata_use_begin_time,
            one_day_trading_minutes,
            return_unfinished,
        )

    def get_data_map(self):
        return {}

    def generate_domain_id(self, entity, original_data):
        # ⚠️ SAST Risk (Medium): Potentially unsafe external URL request without timeout
        return generate_kdata_id(
            entity.id, timestamp=original_data["timestamp"], level=self.level
        )

    def record(self, entity, start, end, size, timestamps):
        # ⚠️ SAST Risk (Low): Potentially unsafe HTML parsing
        the_quarters = get_year_quarters(start)
        if not is_same_date(entity.timestamp, start) and len(the_quarters) > 1:
            the_quarters = the_quarters[1:]

        param = {
            "security_item": entity,
            "quarters": the_quarters,
            "level": self.level.value,
        }
        # ✅ Best Practice: Consider using a backoff strategy instead of fixed sleep

        security_item = param["security_item"]
        quarters = param["quarters"]
        level = param["level"]
        # ✅ Best Practice: Consider using a backoff strategy instead of fixed sleep

        result_df = pd.DataFrame()
        for year, quarter in quarters:
            query_url = self.url.format(security_item.code, year, quarter)
            response = requests.get(query_url)
            response.encoding = "gbk"

            try:
                # 🧠 ML Signal: Usage of pandas for data manipulation
                dfs = pd.read_html(response.text)
            except ValueError as error:
                self.logger.error(
                    f"skip ({year}-{quarter:02d}){security_item.code}{security_item.name}({error})"
                )
                # 🧠 ML Signal: Usage of pandas for data concatenation
                time.sleep(10.0)
                continue

            # 🧠 ML Signal: Conversion of DataFrame to dictionary
            # ✅ Best Practice: Consider using a backoff strategy instead of fixed sleep
            # 🧠 ML Signal: Usage of pandas for data sorting
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # 🧠 ML Signal: Pattern for running a script directly
            if len(dfs) < 5:
                time.sleep(10.0)
                continue

            df = dfs[4].copy()
            df = df.iloc[1:]
            df.columns = [
                "timestamp",
                "open",
                "high",
                "close",
                "low",
                "volume",
                "turnover",
            ]
            df["name"] = security_item.name
            df["level"] = level
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["provider"] = "sina"

            result_df = pd.concat([result_df, df])

            self.logger.info(
                f"({security_item.code}{security_item.name})({year}-{quarter:02d})"
            )
            time.sleep(10.0)

        result_df = result_df.sort_values(by="timestamp")

        return result_df.to_dict(orient="records")


__all__ = ["ChinaIndexDayKdataRecorder"]

if __name__ == "__main__":
    ChinaIndexDayKdataRecorder().run()


# the __all__ is generated
__all__ = ["ChinaIndexDayKdataRecorder"]
