# -*- coding: utf-8 -*-
from zvt.api.utils import to_report_period_type
from zvt.domain.misc.holder import TopTenHolder
from zvt.recorders.eastmoney.common import EastmoneyTimestampsDataRecorder, get_fc
from zvt.utils.time_utils import to_time_str, to_pd_timestamp

# 🧠 ML Signal: Importing specific functions and classes indicates usage patterns and dependencies
# 🧠 ML Signal: Inherits from a base class, indicating a pattern of using inheritance for code reuse
from zvt.utils.utils import to_float

# 🧠 ML Signal: Class attribute indicating a constant value, useful for identifying configuration patterns


class TopTenHolderRecorder(EastmoneyTimestampsDataRecorder):
    # 🧠 ML Signal: Class attribute indicating a constant value, useful for identifying configuration patterns
    provider = "eastmoney"
    data_schema = TopTenHolder
    # 🧠 ML Signal: Class attribute indicating a constant value, useful for identifying configuration patterns

    url = "https://emh5.eastmoney.com/api/GuBenGuDong/GetShiDaGuDong"
    # 🧠 ML Signal: Class attribute indicating a constant value, useful for identifying configuration patterns
    path_fields = ["ShiDaGuDongList"]
    # ✅ Best Practice: Use of a dictionary to map keys to tuples for structured data transformation
    # 🧠 ML Signal: Class attribute indicating a constant value, useful for identifying configuration patterns
    # ✅ Best Practice: Clear mapping of data fields to transformation functions

    timestamps_fetching_url = (
        "https://emh5.eastmoney.com/api/GuBenGuDong/GetFirstRequest2Data"
    )
    timestamp_list_path_fields = ["SDGDBGQ", "ShiDaGuDongBaoGaoQiList"]
    timestamp_path_fields = ["BaoGaoQi"]

    def get_data_map(self):
        return {
            "report_period": ("timestamp", to_report_period_type),
            "report_date": ("timestamp", to_pd_timestamp),
            # 股东代码
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
            "holder_code": ("GuDongDaiMa", str),
            # 股东名称
            # ✅ Best Practice: Use descriptive keys in the dictionary to improve readability
            # ✅ Best Practice: Use of descriptive function name for clarity
            "holder_name": ("GuDongMingCheng", str),
            # 🧠 ML Signal: Usage of dictionary to structure request parameters
            # 持股数
            # ✅ Best Practice: Use of descriptive variable name for clarity
            # 🧠 ML Signal: Function call pattern with specific parameters
            "shareholding_numbers": ("ChiGuShu", to_float),
            # 持股比例
            # ✅ Best Practice: Use of method to retrieve time field increases flexibility
            "shareholding_ratio": ("ChiGuBiLi", to_float),
            # 变动
            # ✅ Best Practice: Use of string formatting for constructing the_id
            "change": ("ZengJian", to_float),
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # 🧠 ML Signal: Entry point for script execution
            # 🧠 ML Signal: Instantiation and method call pattern
            # 变动比例
            "change_ratio": ("BianDongBiLi", to_float),
        }

    def generate_request_param(self, security_item, start, end, size, timestamp):
        return {
            "color": "w",
            "fc": get_fc(security_item),
            "BaoGaoQi": to_time_str(timestamp),
        }

    def generate_domain_id(self, entity, original_data):
        the_name = original_data.get("GuDongMingCheng")
        timestamp = original_data[self.get_original_time_field()]
        the_id = "{}_{}_{}".format(entity.id, timestamp, the_name)
        return the_id


if __name__ == "__main__":
    # init_log('top_ten_holder.log')

    TopTenHolderRecorder(codes=["002572"]).run()


# the __all__ is generated
__all__ = ["TopTenHolderRecorder"]
