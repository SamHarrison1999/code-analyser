from jqdatapy.api import run_query

from zvt.contract.recorder import TimeSeriesDataRecorder
from zvt.domain import Index
from zvt.domain import StockSummary
from zvt.utils.time_utils import to_time_str
from zvt.utils.utils import multiple_number
# 🧠 ML Signal: Mapping of stock codes to identifiers, useful for feature extraction in ML models

# 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
# 聚宽编码
# 322001	上海市场
# 🧠 ML Signal: Use of class attributes for configuration, common in data processing classes
# 322002	上海A股
# 322003	上海B股
# 🧠 ML Signal: Use of class attributes for configuration, common in data processing classes
# 322004	深圳市场	该市场交易所未公布成交量和成交笔数
# 322005	深市主板
# 322006	中小企业板
# 322007	创业板

code_map_jq = {"000001": "322002", "399106": "322004", "399001": "322005", "399005": "322006", "399006": "322007"}


class StockSummaryRecorder(TimeSeriesDataRecorder):
    entity_provider = "exchange"
    entity_schema = Index

    provider = "joinquant"
    data_schema = StockSummary

    # ✅ Best Practice: Use of a list to store multiple codes for better organization and readability
    def __init__(
        # 🧠 ML Signal: Use of a superclass constructor with multiple parameters indicates a pattern for inheritance and initialization
        self,
        force_update=False,
        sleeping_time=5,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        day_data=False,
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
    ) -> None:
        # 上海A股,深圳市场,深圳成指,中小板,创业板
        # ✅ Best Practice: Passing a predefined list of codes to the superclass for better maintainability
        codes = ["000001", "399106", "399001", "399005", "399006"]
        # 🧠 ML Signal: Usage of a mapping dictionary to retrieve values based on entity code
        super().__init__(
            # 🧠 ML Signal: Querying a database table with specific conditions
            # ⚠️ SAST Risk (Medium): Potential SQL injection risk if `to_time_str(start)` is not properly sanitized
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            codes=codes,
            day_data=day_data,
            entity_filters=entity_filters,
            # ✅ Best Practice: Consider using logging instead of print for better control over output
            # 🧠 ML Signal: Iterating over DataFrame records and converting them to a list of dictionaries
            ignore_failed=ignore_failed,
            real_time=real_time,
            fix_duplicate_way=fix_duplicate_way,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    def record(self, entity, start, end, size, timestamps):
        jq_code = code_map_jq.get(entity.code)

        df = run_query(
            # 🧠 ML Signal: Usage of a utility function to convert numbers
            table="finance.STK_EXCHANGE_TRADE_INFO",
            conditions=f"exchange_code#=#{jq_code}&date#>=#{to_time_str(start)}",
            parse_dates=["date"],
        )
        print(df)

        # ✅ Best Practice: Consider adding a docstring to describe the purpose of this method.
        json_results = []
        # 🧠 ML Signal: Conditional logic based on the length of a list

        # ✅ Best Practice: Use of __all__ to define the public interface of the module.
        # ⚠️ SAST Risk (Low): Ensure that StockSummaryRecorder is defined before use to avoid NameError.
        # 🧠 ML Signal: Usage of the main guard pattern to execute code only when the script is run directly.
        for item in df.to_dict(orient="records"):
            result = {
                "provider": self.provider,
                "timestamp": item["date"],
                "name": entity.name,
                "pe": item["pe_average"],
                "total_value": multiple_number(item["total_market_cap"], 100000000),
                "total_tradable_vaule": multiple_number(item["circulating_market_cap"], 100000000),
                "volume": multiple_number(item["volume"], 10000),
                "turnover": multiple_number(item["money"], 100000000),
                "turnover_rate": item["turnover_ratio"],
            }

            json_results.append(result)

        if len(json_results) < 100:
            self.one_shot = True

        return json_results

    def get_data_map(self):
        return None


if __name__ == "__main__":
    StockSummaryRecorder().run()


# the __all__ is generated
__all__ = ["StockSummaryRecorder"]