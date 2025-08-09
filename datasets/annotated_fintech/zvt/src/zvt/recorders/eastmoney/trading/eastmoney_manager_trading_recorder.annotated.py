# -*- coding: utf-8 -*-
from zvt.domain import ManagerTrading
from zvt.recorders.eastmoney.common import EastmoneyMoreDataRecorder
# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies
# 🧠 ML Signal: Inheritance from a specific base class indicates a pattern of extending functionality
from zvt.utils.utils import to_float

# 🧠 ML Signal: Class attribute assignment indicates a pattern of setting schema for data handling

class ManagerTradingRecorder(EastmoneyMoreDataRecorder):
    # 🧠 ML Signal: URL assignment indicates a pattern of accessing external resources
    # ✅ Best Practice: Method name should be descriptive of its functionality
    data_schema = ManagerTrading

    # ✅ Best Practice: Use of a dictionary to map keys to tuples for structured data representation
    # 🧠 ML Signal: Path fields assignment indicates a pattern of specifying data extraction paths
    # 🧠 ML Signal: Returns a hardcoded string, indicating a potential constant or configuration
    url = "https://emh5.eastmoney.com/api/JiaoYiShuJu/GetGaoGuanZengJian"
    # 🧠 ML Signal: Mapping of keys to data types and conversion functions indicates data processing patterns
    # 🧠 ML Signal: Use of string keys and conversion functions suggests a pattern for data transformation
    # 🧠 ML Signal: Use of a custom function 'to_float' for data conversion indicates a specific data handling pattern
    path_fields = ["GaoGuanZengJianList"]

    def get_original_time_field(self):
        return "RiQi"

    def get_data_map(self):
        return {
            # 🧠 ML Signal: Consistent use of 'to_float' for numerical data conversion
            "trading_person": ("BianDongRen", str),
            "volume": ("BianDongShuLiang", to_float),
            # 🧠 ML Signal: Consistent use of 'to_float' for numerical data conversion
            "price": ("JiaoYiJunJia", to_float),
            "holding": ("BianDongHouShuLiang", to_float),
            # 🧠 ML Signal: Use of string keys and conversion functions suggests a pattern for data transformation
            # 🧠 ML Signal: Usage of dictionary get method to access data
            "trading_way": ("JiaoYiTuJing", str),
            "manager": ("GaoGuanMingCheng", str),
            # 🧠 ML Signal: Use of string keys and conversion functions suggests a pattern for data transformation
            # 🧠 ML Signal: Accessing a method to retrieve a specific field name
            "manager_position": ("GaoGuanZhiWei", str),
            "relationship_with_manager": ("GaoGuanGuanXi", str),
        # 🧠 ML Signal: Use of string keys and conversion functions suggests a pattern for data transformation
        # 🧠 ML Signal: String formatting pattern for ID generation
        }

    # 🧠 ML Signal: Use of string keys and conversion functions suggests a pattern for data transformation
    def generate_domain_id(self, entity, original_data):
        # 🧠 ML Signal: Instantiation of a class with specific parameters
        # ⚠️ SAST Risk (Low): Direct execution of code in the main block without input validation
        # 🧠 ML Signal: Invocation of a run method on an object
        # ✅ Best Practice: Explicitly defining __all__ for module exports
        the_name = original_data.get("BianDongRen")
        timestamp = original_data[self.get_original_time_field()]
        the_id = "{}_{}_{}".format(entity.id, timestamp, the_name)
        return the_id


if __name__ == "__main__":
    # init_log('manager_trading.log')

    recorder = ManagerTradingRecorder(codes=["002572"])
    recorder.run()


# the __all__ is generated
__all__ = ["ManagerTradingRecorder"]