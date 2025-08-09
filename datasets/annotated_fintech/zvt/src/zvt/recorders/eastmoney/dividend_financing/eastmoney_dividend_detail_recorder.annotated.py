# -*- coding: utf-8 -*-
from zvt.domain import DividendDetail

# ⚠️ SAST Risk (Low): Importing modules without validation can lead to dependency confusion if the module names are not unique or are typo-squatted.
from zvt.recorders.eastmoney.common import EastmoneyPageabeDataRecorder
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Inheritance from a specific base class indicates a pattern of extending functionality


# 🧠 ML Signal: Assignment of a schema to a variable suggests a pattern of data structure usage
class DividendDetailRecorder(EastmoneyPageabeDataRecorder):
    data_schema = DividendDetail
    # ⚠️ SAST Risk (Low): Hardcoded URL can lead to inflexibility and potential security risks if the URL changes or is deprecated

    # 🧠 ML Signal: Hardcoded API endpoint indicates a pattern of accessing external resources
    # ✅ Best Practice: Method name is descriptive and follows naming conventions
    url = "https://emh5.eastmoney.com/api/FenHongRongZi/GetFenHongSongZhuanList"
    page_url = url
    # ✅ Best Practice: Reusing the `url` variable improves maintainability and reduces redundancy
    # 🧠 ML Signal: Returns a hardcoded string, indicating a constant or fixed value
    # ✅ Best Practice: Use of a dictionary to map keys to tuples for clear data structure
    # 🧠 ML Signal: Use of a list to define path fields suggests a pattern of structured data access
    path_fields = ["FenHongSongZhuanList"]

    def get_original_time_field(self):
        return "GongGaoRiQi"

    def get_data_map(self):
        return {
            # 公告日
            # 🧠 ML Signal: Entry point for script execution
            "announce_date": ("GongGaoRiQi", to_pd_timestamp),
            # 🧠 ML Signal: Instantiation of a class with specific parameters
            # 🧠 ML Signal: Method call on an object
            # ✅ Best Practice: Use of __all__ to define public interface of the module
            # 股权登记日
            "record_date": ("GuQuanDengJiRi", to_pd_timestamp),
            # 除权除息日
            "dividend_date": ("ChuQuanChuXiRi", to_pd_timestamp),
            # 方案
            "dividend": ("FengHongFangAn", str),
        }


if __name__ == "__main__":
    # init_log('dividend_detail.log')

    recorder = DividendDetailRecorder(codes=["601318"])
    recorder.run()


# the __all__ is generated
__all__ = ["DividendDetailRecorder"]
