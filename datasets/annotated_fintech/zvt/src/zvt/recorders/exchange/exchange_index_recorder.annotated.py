# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.domain import Index
# 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
from zvt.recorders.exchange.api import cn_index_api, cs_index_api

# 🧠 ML Signal: Use of class attribute to define a constant value

# 🧠 ML Signal: Method calls with string literals indicate specific actions or configurations
class ExchangeIndexRecorder(Recorder):
    # 🧠 ML Signal: Use of class attribute to define a schema or data structure
    provider = "exchange"
    # 🧠 ML Signal: Method calls with string literals indicate specific actions or configurations
    data_schema = Index

    # 🧠 ML Signal: Method calls with string literals indicate specific actions or configurations
    # 🧠 ML Signal: Method for recording index data, useful for learning data processing patterns
    def run(self):
        # 深圳
        # 🧠 ML Signal: Method calls with string literals indicate specific actions or configurations
        # 🧠 ML Signal: API call pattern for fetching data
        self.record_cn_index("sz")
        # 国证
        # ⚠️ SAST Risk (Low): Ensure df_to_db handles data safely to prevent injection attacks
        self.record_cn_index("cni")
        # 🧠 ML Signal: Conditional logic based on input parameter
        # 🧠 ML Signal: Pattern for storing data into a database

        # 上海
        # ✅ Best Practice: Use of logging for tracking execution and debugging
        self.record_cs_index("sh")
        # 中证
        self.record_cs_index("csi")

    # ⚠️ SAST Risk (Low): Use of assert for control flow, which can be disabled in production
    # 中证，上海
    def record_cs_index(self, index_type):
        df = cs_index_api.get_cs_index(index_type=index_type)
        # 🧠 ML Signal: Iterating over items in a dictionary
        df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)
        self.logger.info(f"finish record {index_type} index")
    # 🧠 ML Signal: API call with dynamic parameters

    # 国证，深圳
    # 🧠 ML Signal: Logging operation with dynamic content
    # ✅ Best Practice: Standard Python entry point check
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Data persistence operation
    # 🧠 ML Signal: Execution of main functionality
    def record_cn_index(self, index_type):
        if index_type == "cni":
            category_map_url = cn_index_api.cni_category_map_url
        elif index_type == "sz":
            category_map_url = cn_index_api.sz_category_map_url
        else:
            self.logger.error(f"not support index_type: {index_type}")
            assert False

        for category, _ in category_map_url.items():
            df = cn_index_api.get_cn_index(index_type=index_type, category=category)
            df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)
            self.logger.info(f"finish record {index_type} index:{category.value}")


if __name__ == "__main__":
    # init_log('china_stock_category.log')
    ExchangeIndexRecorder().run()


# the __all__ is generated
__all__ = ["ExchangeIndexRecorder"]