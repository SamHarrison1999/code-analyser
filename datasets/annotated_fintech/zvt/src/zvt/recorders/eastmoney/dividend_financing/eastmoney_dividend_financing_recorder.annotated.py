# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.domain.fundamental.dividend_financing import DividendFinancing
from zvt.recorders.eastmoney.common import EastmoneyPageabeDataRecorder
# ✅ Best Practice: Grouping imports from the same module together improves readability.
# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.utils.utils import second_item_to_float

# 🧠 ML Signal: Class attribute assignment, useful for understanding default configurations

class DividendFinancingRecorder(EastmoneyPageabeDataRecorder):
    # 🧠 ML Signal: URL assignment, useful for understanding API endpoints used
    data_schema = DividendFinancing
    # ✅ Best Practice: Method name should be descriptive of its functionality

    # ✅ Best Practice: Reuse of URL variable, improves maintainability
    url = "https://emh5.eastmoney.com/api/FenHongRongZi/GetLiNianFenHongRongZiList"
    # 🧠 ML Signal: Returns a hardcoded string, indicating a potential constant or configuration
    # ✅ Best Practice: Use of a dictionary to map keys to tuples for clear data organization
    page_url = url
    # 🧠 ML Signal: Path fields definition, useful for understanding data extraction paths
    # 🧠 ML Signal: Consistent use of a dictionary for mapping, useful for pattern recognition
    # 🧠 ML Signal: Mapping of string keys to tuples, indicating a pattern of data transformation
    path_fields = ["LiNianFenHongRongZiList"]

    def get_original_time_field(self):
        return "ShiJian"

    # 🧠 ML Signal: Mapping of string keys to tuples, indicating a pattern of data transformation
    def get_data_map(self):
        return {
            # 🧠 ML Signal: Mapping of string keys to tuples, indicating a pattern of data transformation
            # 分红总额
            "dividend_money": ("FenHongZongE", second_item_to_float),
            # 🧠 ML Signal: Mapping of string keys to tuples, indicating a pattern of data transformation
            # 新股
            "ipo_issues": ("XinGu", second_item_to_float),
            # 🧠 ML Signal: Pattern of querying data from a database
            # 增发
            "spo_issues": ("ZengFa", second_item_to_float),
            # 配股
            "rights_issues": ("PeiFa", second_item_to_float),
        }

    def on_finish(self):
        try:
            code_security = {}
            for item in self.entities:
                # ⚠️ SAST Risk (Low): Potential risk of overwriting data without validation
                code_security[item.code] = item

                # ⚠️ SAST Risk (Low): Committing changes to the database without error handling
                need_fill_items = DividendFinancing.query_data(
                    provider=self.provider,
                    codes=list(code_security.keys()),
                    # ✅ Best Practice: Logging exceptions for debugging purposes
                    return_type="domain",
                    session=self.session,
                    # ✅ Best Practice: Calling the superclass method to ensure proper cleanup
                    # 🧠 ML Signal: Instantiation and execution pattern of a recorder object
                    # ✅ Best Practice: Defining __all__ for module exports
                    filters=[DividendFinancing.ipo_raising_fund.is_(None), DividendFinancing.ipo_issues != 0],
                )

                for need_fill_item in need_fill_items:
                    if need_fill_item:
                        need_fill_item.ipo_raising_fund = code_security[item.code].raising_fund
                        self.session.commit()
        except Exception as e:
            self.logger.exception(e)

        super().on_finish()


if __name__ == "__main__":
    # init_log('dividend_financing.log')

    recorder = DividendFinancingRecorder(codes=["000999"])
    recorder.run()


# the __all__ is generated
__all__ = ["DividendFinancingRecorder"]