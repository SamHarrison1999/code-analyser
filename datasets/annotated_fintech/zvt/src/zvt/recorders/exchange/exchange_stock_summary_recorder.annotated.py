import demjson3
import pandas as pd
import requests

# ✅ Best Practice: Group imports into standard library, third-party, and local imports for better readability

from zvt.contract.recorder import TimestampsDataRecorder
from zvt.domain import Index
from zvt.domain.misc import StockSummary
from zvt.recorders.consts import DEFAULT_SH_SUMMARY_HEADER
from zvt.utils.time_utils import to_time_str
from zvt.utils.utils import to_float

# 🧠 ML Signal: Inheritance from TimestampsDataRecorder indicates a pattern of extending functionality


# 🧠 ML Signal: Use of class-level attributes for configuration
class ExchangeStockSummaryRecorder(TimestampsDataRecorder):
    entity_provider = "exchange"
    # 🧠 ML Signal: Use of class-level attributes for configuration
    entity_schema = Index

    # 🧠 ML Signal: Use of class-level attributes for configuration
    # ⚠️ SAST Risk (Low): Hardcoded URL can lead to inflexibility and potential security risks if the URL changes or is deprecated
    # ⚠️ SAST Risk (Low): Hardcoded URL with query parameters can expose sensitive data or be manipulated if not handled properly
    provider = "exchange"
    data_schema = StockSummary

    original_page_url = "http://www.sse.com.cn/market/stockdata/overview/day/"

    url = "http://query.sse.com.cn/marketdata/tradedata/queryTradingByProdTypeData.do?jsonCallBack=jsonpCallback30731&searchDate={}&prodType=gp&_=1515717065511"

    def __init__(
        self,
        force_update=False,
        sleeping_time=5,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        code=None,
        day_data=False,
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method
        # 🧠 ML Signal: Use of default parameters in function signature
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            code,
            ["000001"],
            day_data,
            # 🧠 ML Signal: Use of default parameters in function signature
            # 🧠 ML Signal: Use of hardcoded list in function call
            # 🧠 ML Signal: Use of pandas for date range generation
            entity_filters,
            # ✅ Best Practice: Use of pd.date_range for generating a list of business days
            ignore_failed,
            # 🧠 ML Signal: Use of default parameters in function signature
            real_time,
            # ⚠️ SAST Risk (Low): Potential timezone issues with pd.Timestamp.now()
            fix_duplicate_way,
            # 🧠 ML Signal: Use of default parameters in function signature
            start_timestamp,
            # 🧠 ML Signal: Iterating over timestamps to fetch and process data
            end_timestamp,
            # 🧠 ML Signal: Use of default parameters in function signature
        )

    # 🧠 ML Signal: URL formatting with dynamic timestamp

    # 🧠 ML Signal: Use of default parameters in function signature
    def init_timestamps(self, entity):
        # ⚠️ SAST Risk (Medium): No error handling for network request
        return pd.date_range(
            start=entity.timestamp, end=pd.Timestamp.now(), freq="B"
        ).tolist()

    # 🧠 ML Signal: Use of default parameters in function signature

    # ⚠️ SAST Risk (Medium): No error handling for JSON decoding
    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Use of default parameters in function signature
        # 🧠 ML Signal: Filtering results based on productType
        # 🧠 ML Signal: Appending processed data to json_results
        json_results = []
        for timestamp in timestamps:
            timestamp_str = to_time_str(timestamp)
            url = self.url.format(timestamp_str)
            response = requests.get(url=url, headers=DEFAULT_SH_SUMMARY_HEADER)

            results = demjson3.decode(
                response.text[response.text.index("(") + 1 : response.text.index(")")]
            )["result"]
            result = [result for result in results if result["productType"] == "1"]
            if result and len(result) == 1:
                result_json = result[0]
                # 有些较老的数据不存在,默认设为0.0
                json_results.append(
                    {
                        # ✅ Best Practice: Use of helper function to convert values
                        "provider": "exchange",
                        "timestamp": timestamp,
                        "name": "上证指数",
                        "pe": to_float(result_json["profitRate"], 0.0),
                        "total_value": to_float(
                            result_json["marketValue1"] + "亿", 0.0
                        ),
                        # ✅ Best Practice: Use of __name__ == "__main__" to allow or prevent parts of code from being run when the modules are imported.
                        "total_tradable_vaule": to_float(
                            result_json["negotiableValue1"] + "亿", 0.0
                        ),
                        "volume": to_float(result_json["trdVol1"] + "万", 0.0),
                        # ✅ Best Practice: Early return to avoid unnecessary processing
                        # ⚠️ SAST Risk (High): Potential use of an undefined class 'ExchangeStockSummaryRecorder', which could lead to runtime errors.
                        # ✅ Best Practice: Use of __all__ to define the public interface of the module, controlling what is exported when import * is used.
                        "turnover": to_float(result_json["trdAmt1"] + "亿", 0.0),
                        "turnover_rate": to_float(result_json["exchangeRate"], 0.0),
                    }
                )

                if len(json_results) > 30:
                    return json_results

        return json_results

    def get_data_map(self):
        return None


if __name__ == "__main__":
    ExchangeStockSummaryRecorder().run()


# the __all__ is generated
__all__ = ["ExchangeStockSummaryRecorder"]
