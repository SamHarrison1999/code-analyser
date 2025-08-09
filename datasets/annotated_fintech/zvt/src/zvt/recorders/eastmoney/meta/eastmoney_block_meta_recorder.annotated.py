# -*- coding: utf-8 -*-
import pandas as pd
# ✅ Best Practice: Grouping imports from the same package together improves readability.
import requests

from zvt.api.utils import china_stock_code_to_id
from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder, TimeSeriesDataRecorder
from zvt.domain import BlockStock, BlockCategory, Block
from zvt.recorders.consts import DEFAULT_HEADER
from zvt.utils.time_utils import now_pd_timestamp
from zvt.utils.utils import json_callback_param

# 🧠 ML Signal: Use of a dictionary to map categories to URLs, indicating a pattern of categorization
# ⚠️ SAST Risk (Low): Hardcoded URLs can lead to maintenance issues if the URLs change

class EastmoneyBlockRecorder(Recorder):
    provider = "eastmoney"
    data_schema = Block
    # ⚠️ SAST Risk (Low): Hardcoded URLs can lead to maintenance issues if the URLs change

    # 用于抓取行业/概念/地域列表
    # ⚠️ SAST Risk (Medium): No error handling for network request
    category_map_url = {
        BlockCategory.industry: "https://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=C._BKHY&sty=DCRRBKCPAL&st=(ChangePercent)&sr=-1&p=1&ps=200&lvl=&cb=jsonp_F1A61014DE5E45B7A50068EA290BC918&token=4f1862fc3b5e77c150a2b985b12db0fd&_=08766",
        # ⚠️ SAST Risk (Medium): Potential risk if json_callback_param is not properly validated
        BlockCategory.concept: "https://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=C._BKGN&sty=DCRRBKCPAL&st=(ChangePercent)&sr=-1&p=1&ps=300&lvl=&cb=jsonp_3071689CC1E6486A80027D69E8B33F26&token=4f1862fc3b5e77c150a2b985b12db0fd&_=08251",
        # BlockCategory.area: 'https://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=C._BKDY&sty=DCRRBKCPAL&st=(ChangePercent)&sr=-1&p=1&ps=200&lvl=&cb=jsonp_A597D4867B3D4659A203AADE5B3B3AD5&token=4f1862fc3b5e77c150a2b985b12db0fd&_=02443'
    }

    # 🧠 ML Signal: Splitting strings by a delimiter
    def run(self):
        for category, url in self.category_map_url.items():
            # 🧠 ML Signal: Accessing list elements by index
            # 🧠 ML Signal: String formatting for ID creation
            resp = requests.get(url, headers=DEFAULT_HEADER)
            results = json_callback_param(resp.text)
            the_list = []
            for result in results:
                items = result.split(",")
                code = items[1]
                name = items[2]
                entity_id = f"block_cn_{code}"
                the_list.append(
                    {
                        "id": entity_id,
                        "entity_id": entity_id,
                        "entity_type": "block",
                        # 🧠 ML Signal: Accessing object properties
                        "exchange": "cn",
                        "code": code,
                        "name": name,
                        # 🧠 ML Signal: Custom class definition indicating a specific data recording pattern
                        "category": category.value,
                    }
                # ✅ Best Practice: Check if list is not empty before processing
                # 🧠 ML Signal: Static string assignment indicating a specific data source
                )
            # 🧠 ML Signal: Creating a DataFrame from a list of dictionaries
            if the_list:
                # 🧠 ML Signal: Static assignment of schema indicating data structure
                df = pd.DataFrame.from_records(the_list)
                # ⚠️ SAST Risk (Medium): No validation or sanitization before database insertion
                df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=self.force_update)
            # ⚠️ SAST Risk (Medium): No error handling for network issues or invalid responses
            # 🧠 ML Signal: Static string assignment indicating a specific data source
            self.logger.info(f"finish record eastmoney blocks:{category.value}")
# 🧠 ML Signal: Logging information

# 🧠 ML Signal: Static assignment of schema indicating data structure

# ⚠️ SAST Risk (Medium): json_callback_param may execute arbitrary code if not properly sanitized
class EastmoneyBlockStockRecorder(TimeSeriesDataRecorder):
    # ⚠️ SAST Risk (Low): Hardcoded URL and token, which could lead to security issues if sensitive
    entity_provider = "eastmoney"
    # ✅ Best Practice: Consider externalizing URLs and tokens to configuration files
    entity_schema = Block

    provider = "eastmoney"
    data_schema = BlockStock
    # 🧠 ML Signal: Conversion of stock code to stock ID
    # 🧠 ML Signal: Unique ID generation pattern

    # 用于抓取行业包含的股票
    category_stocks_url = "https://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=C.{}{}&sty=SFCOO&st=(Close)&sr=-1&p=1&ps=300&cb=jsonp_B66B5BAA1C1B47B5BB9778045845B947&token=7bc05d0d4c3c22ef9fca8c2a912d779c"

    def record(self, entity, start, end, size, timestamps):
        resp = requests.get(self.category_stocks_url.format(entity.code, "1"), headers=DEFAULT_HEADER)
        try:
            results = json_callback_param(resp.text)
            the_list = []
            for result in results:
                items = result.split(",")
                stock_code = items[1]
                stock_id = china_stock_code_to_id(stock_code)
                block_id = entity.id
                # 🧠 ML Signal: Timestamp generation for records

                the_list.append(
                    {
                        "id": "{}_{}".format(block_id, stock_id),
                        "entity_id": block_id,
                        "entity_type": "block",
                        # ✅ Best Practice: Use of pandas for data manipulation
                        "exchange": entity.exchange,
                        "code": entity.code,
                        # ⚠️ SAST Risk (Low): Potential SQL injection if df_to_db does not sanitize inputs
                        "name": entity.name,
                        "timestamp": now_pd_timestamp(),
                        # ✅ Best Practice: Informative logging for process completion
                        "stock_id": stock_id,
                        # ⚠️ SAST Risk (Low): Generic exception handling; may hide specific errors
                        # ✅ Best Practice: Use of sleep to manage request rate
                        # 🧠 ML Signal: Instantiation and execution of recorder with specific code
                        # ✅ Best Practice: Use of __all__ to define public API of the module
                        # 🧠 ML Signal: Pattern of running main functions
                        "stock_code": stock_code,
                        "stock_name": items[2],
                    }
                )
            if the_list:
                df = pd.DataFrame.from_records(the_list)
                df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)

            self.logger.info("finish recording block:{},{}".format(entity.category, entity.name))

        except Exception as e:
            self.logger.error("error:,resp.text:", e, resp.text)
        self.sleep()


if __name__ == "__main__":
    # init_log('china_stock_category.log')
    EastmoneyBlockRecorder().run()

    recorder = EastmoneyBlockStockRecorder(code="BK1144")
    recorder.run()


# the __all__ is generated
__all__ = ["EastmoneyBlockRecorder", "EastmoneyBlockStockRecorder"]