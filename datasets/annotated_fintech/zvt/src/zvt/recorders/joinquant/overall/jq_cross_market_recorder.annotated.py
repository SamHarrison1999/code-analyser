from jqdatapy.api import run_query
# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies

from zvt.contract.recorder import TimeSeriesDataRecorder
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies
from zvt.domain import Index, CrossMarketSummary
from zvt.utils.time_utils import to_time_str
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies
from zvt.utils.utils import multiple_number
# ✅ Best Practice: Class attributes are defined at the top for clarity and maintainability

# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies

# 🧠 ML Signal: Use of a specific entity schema indicates a pattern in data handling
class CrossMarketSummaryRecorder(TimeSeriesDataRecorder):
    # 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies
    entity_provider = "joinquant"
    # 🧠 ML Signal: Consistent use of provider name suggests a pattern in data source usage
    entity_schema = Index
    # ✅ Best Practice: Use of default values for function parameters improves usability and flexibility.

    # 🧠 ML Signal: Use of a specific data schema indicates a pattern in data handling
    # 🧠 ML Signal: Hardcoded list of codes could indicate specific domain knowledge or usage pattern.
    provider = "joinquant"
    data_schema = CrossMarketSummary

    def __init__(self, force_update=False, sleeping_time=5, real_time=False, fix_duplicate_way="add") -> None:

        # 聚宽编码
        # 市场通编码	市场通名称
        # 310001	沪股通
        # 310002	深股通
        # 310003	港股通（沪）
        # ✅ Best Practice: Use of super() to call a method from the parent class ensures proper initialization in inheritance.
        # 310004	港股通（深）

        # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
        codes = ["310001", "310002", "310003", "310004"]
        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if `entity.code` or `start` are not properly sanitized
        # 🧠 ML Signal: The parameters passed to super().__init__() can indicate configuration or setup patterns.
        super().__init__(
            force_update,
            # ✅ Best Practice: Consider using logging instead of print for better control over output
            sleeping_time,
            ["cn"],
            # 🧠 ML Signal: Iterating over DataFrame records to transform data
            None,
            codes=codes,
            day_data=True,
            real_time=real_time,
            fix_duplicate_way=fix_duplicate_way,
        )

    def init_entities(self):
        super().init_entities()

    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Applying transformation to numerical data
        df = run_query(table="finance.STK_ML_QUOTA", conditions=f"link_id#=#{entity.code}&day#>=#{to_time_str(start)}")
        print(df)
        # 🧠 ML Signal: Applying transformation to numerical data

        json_results = []
        # 🧠 ML Signal: Applying transformation to numerical data

        for item in df.to_dict(orient="records"):
            # ✅ Best Practice: Use a main guard to ensure code is only executed when the script is run directly
            result = {
                "provider": self.provider,
                # ✅ Best Practice: Define __all__ to explicitly declare the public API of the module
                # 🧠 ML Signal: Conditional logic based on the length of results
                # ⚠️ SAST Risk (Low): Potential risk if CrossMarketSummaryRecorder is not defined or imported
                "timestamp": item["day"],
                "name": entity.name,
                "buy_amount": multiple_number(item["buy_amount"], 100000000),
                "buy_volume": item["buy_volume"],
                "sell_amount": multiple_number(item["sell_amount"], 100000000),
                "sell_volume": item["sell_volume"],
                "quota_daily": multiple_number(item["quota_daily"], 100000000),
                "quota_daily_balance": multiple_number(item["quota_daily_balance"], 100000000),
            }

            json_results.append(result)

        if len(json_results) < 100:
            self.one_shot = True

        return json_results

    def get_data_map(self):
        return None


if __name__ == "__main__":
    CrossMarketSummaryRecorder().run()


# the __all__ is generated
__all__ = ["CrossMarketSummaryRecorder"]