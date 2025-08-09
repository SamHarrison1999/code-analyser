# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.domain import HolderTrading
from zvt.recorders.eastmoney.common import EastmoneyMoreDataRecorder
# ✅ Best Practice: Class definition should follow the naming conventions (CamelCase).
from zvt.utils.utils import to_float

# ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.

class HolderTradingRecorder(EastmoneyMoreDataRecorder):
    # 🧠 ML Signal: URL patterns can be used to identify API endpoints for web scraping or data collection.
    # ✅ Best Practice: Method name should reflect its purpose and be descriptive
    data_schema = HolderTrading

    # ✅ Best Practice: Use of descriptive variable names improves code readability.
    # ✅ Best Practice: Use of a dictionary to map keys to tuples for structured data representation
    # 🧠 ML Signal: Hardcoded return values can indicate fixed configurations or constants
    url = "https://emh5.eastmoney.com/api/JiaoYiShuJu/GetGuDongZengJian"
    # 🧠 ML Signal: Mapping of human-readable keys to internal data representation
    path_fields = ["GuDongZengJianList"]

    def get_original_time_field(self):
        return "RiQi"
    # 🧠 ML Signal: Use of a custom function 'to_float' for data conversion

    def get_data_map(self):
        # 🧠 ML Signal: Use of a custom function 'to_float' for data conversion
        return {
            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            "holder_name": ("GuDongMingCheng", str),
            # 🧠 ML Signal: Use of a custom function 'to_float' for data conversion
            "volume": ("BianDongShuLiang", to_float),
            # 🧠 ML Signal: Accessing dictionary values using the get method, which is a common pattern.
            "change_pct": ("BianDongBiLi", to_float),
            "holding_pct": ("BianDongHouChiGuBiLi", to_float),
        # 🧠 ML Signal: String formatting using format method, a common pattern in Python.
        }

    def generate_domain_id(self, entity, original_data):
        # 🧠 ML Signal: Instantiation of a class with specific parameters.
        # 🧠 ML Signal: Use of __all__ to define public symbols of a module.
        # 🧠 ML Signal: Common pattern for checking if the script is run as the main program.
        # 🧠 ML Signal: Invocation of a method on an object.
        the_name = original_data.get("GuDongMingCheng")
        timestamp = original_data[self.get_original_time_field()]
        the_id = "{}_{}_{}".format(entity.id, timestamp, the_name)
        return the_id


if __name__ == "__main__":
    # init_log('holder_trading.log')

    recorder = HolderTradingRecorder(codes=["002572"])
    recorder.run()


# the __all__ is generated
__all__ = ["HolderTradingRecorder"]