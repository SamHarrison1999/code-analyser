from jqdatapy.api import run_query
# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns

# ⚠️ SAST Risk (Low): Ensure that the imported function is from a trusted source to avoid executing malicious code
from zvt.contract.recorder import TimeSeriesDataRecorder
from zvt.domain import Index, MarginTradingSummary
from zvt.utils.time_utils import to_time_str
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
# 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern of extending functionality for time series data.

# ⚠️ SAST Risk (Low): Ensure that the imported class is from a trusted source to avoid executing malicious code
# 聚宽编码
# 🧠 ML Signal: Use of class attributes to define metadata for the recorder.
# XSHG-上海证券交易所
# XSHE-深圳证券交易所
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
# 🧠 ML Signal: Association with a specific schema (Index) for data structure.

# 🧠 ML Signal: Importing specific utility functions indicates usage patterns
# ⚠️ SAST Risk (Low): Ensure that the imported classes are from a trusted source to avoid executing malicious code
# 🧠 ML Signal: Usage of a dictionary to map codes indicates a pattern of data transformation or lookup
# 🧠 ML Signal: Use of a specific data provider indicates a pattern of data source selection.
# 🧠 ML Signal: Association with a specific data schema (MarginTradingSummary) for data structure.
code_map_jq = {"000001": "XSHG", "399106": "XSHE"}


class MarginTradingSummaryRecorder(TimeSeriesDataRecorder):
    entity_provider = "exchange"
    entity_schema = Index

    provider = "joinquant"
    # ✅ Best Practice: Use descriptive variable names for better readability and maintainability
    data_schema = MarginTradingSummary

    def __init__(
        self,
        force_update=False,
        sleeping_time=5,
        exchanges=None,
        # ✅ Best Practice: Use of a list to store multiple codes for better organization and potential scalability
        entity_id=None,
        # 🧠 ML Signal: Use of a superclass constructor with multiple parameters indicates inheritance and polymorphism
        # ✅ Best Practice: Explicitly passing parameters to the superclass constructor improves code readability and maintainability
        entity_ids=None,
        day_data=False,
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
    ) -> None:
        # 上海A股,深圳市场
        codes = ["000001", "399106"]
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            # 🧠 ML Signal: Usage of a dictionary to map entity codes to jq_code
            entity_ids,
            # 🧠 ML Signal: Querying a database table with specific conditions
            codes=codes,
            day_data=day_data,
            entity_filters=entity_filters,
            ignore_failed=ignore_failed,
            real_time=real_time,
            # ⚠️ SAST Risk (Low): Potential SQL injection if conditions are not properly sanitized
            fix_duplicate_way=fix_duplicate_way,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        # 🧠 ML Signal: Converting DataFrame to a list of dictionaries
        )

    def record(self, entity, start, end, size, timestamps):
        jq_code = code_map_jq.get(entity.code)

        df = run_query(
            table="finance.STK_MT_TOTAL",
            conditions=f"exchange_code#=#{jq_code}&date#>=#{to_time_str(start)}",
            parse_dates=["date"],
        )
        print(df)

        json_results = []

        for item in df.to_dict(orient="records"):
            result = {
                # 🧠 ML Signal: Conditional logic based on the length of results
                # ✅ Best Practice: Use a main guard to ensure code is only executed when the script is run directly
                "provider": self.provider,
                "timestamp": item["date"],
                # ⚠️ SAST Risk (Low): Potential NameError if MarginTradingSummaryRecorder is not defined elsewhere
                # 🧠 ML Signal: Indicates the entry point of the script, useful for understanding script execution flow
                # ✅ Best Practice: Define __all__ to explicitly declare the public API of the module
                "name": entity.name,
                "margin_value": item["fin_value"],
                "margin_buy": item["fin_buy_value"],
                "short_value": item["sec_value"],
                "short_volume": item["sec_sell_volume"],
                "total_value": item["fin_sec_value"],
            }

            json_results.append(result)

        if len(json_results) < 100:
            self.one_shot = True

        return json_results

    def get_data_map(self):
        return None


if __name__ == "__main__":
    MarginTradingSummaryRecorder().run()


# the __all__ is generated
__all__ = ["MarginTradingSummaryRecorder"]