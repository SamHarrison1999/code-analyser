# -*- coding: utf-8 -*-
import json

import demjson3
# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd
import requests
# ✅ Best Practice: Grouping imports from the same package together improves readability.

from zvt.api.utils import china_stock_code_to_id
# ✅ Best Practice: Grouping imports from the same package together improves readability.
from zvt.contract.api import df_to_db
# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.contract.recorder import Recorder, TimeSeriesDataRecorder
# ✅ Best Practice: Grouping imports from the same package together improves readability.
from zvt.domain import BlockStock, BlockCategory, Block
# 🧠 ML Signal: Class attribute indicating a constant value, useful for understanding class-specific configurations
from zvt.utils.time_utils import now_pd_timestamp
# 🧠 ML Signal: Class attribute indicating a schema, useful for understanding data structure expectations
# ✅ Best Practice: Grouping imports from the same package together improves readability.


class SinaBlockRecorder(Recorder):
    provider = "sina"
    # 🧠 ML Signal: Mapping of categories to URLs, useful for understanding data source configurations
    data_schema = Block
    # 🧠 ML Signal: Enum usage, useful for understanding categorical data handling

    # ⚠️ SAST Risk (Medium): No error handling for network request failures
    # 用于抓取行业/概念/地域列表
    # ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if URLs change or are insecure
    category_map_url = {
        # ✅ Best Practice: Set encoding explicitly to ensure correct text decoding
        BlockCategory.industry: "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php",
        BlockCategory.concept: "http://money.finance.sina.com.cn/q/view/newFLJK.php?param=class"
        # StockCategory.area: 'http://money.finance.sina.com.cn/q/view/newFLJK.php?param=area',
    # ⚠️ SAST Risk (High): Potential for ValueError if "{" or "}" not found in string
    }

    # ⚠️ SAST Risk (Medium): No error handling for JSON decoding
    def run(self):
        # get stock blocks from sina
        # 🧠 ML Signal: Pattern of extracting and processing data from JSON
        # 🧠 ML Signal: Pattern of constructing unique identifiers
        for category, url in self.category_map_url.items():
            resp = requests.get(url)
            resp.encoding = "GBK"

            tmp_str = resp.text
            json_str = tmp_str[tmp_str.index("{") : tmp_str.index("}") + 1]
            tmp_json = json.loads(json_str)

            the_list = []

            for code in tmp_json:
                name = tmp_json[code].split(",")[1]
                entity_id = f"block_cn_{code}"
                the_list.append(
                    {
                        # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
                        "id": entity_id,
                        "entity_id": entity_id,
                        # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
                        "entity_type": "block",
                        # 🧠 ML Signal: Pattern of converting list of dicts to DataFrame
                        "exchange": "cn",
                        # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
                        "code": code,
                        # ⚠️ SAST Risk (Medium): No error handling for database operations
                        "name": name,
                        # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
                        "category": category.value,
                    # ✅ Best Practice: Use logging for tracking execution flow
                    }
                # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
                # 🧠 ML Signal: Iterating over a fixed range of pages
                )
            if the_list:
                # ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if not validated or sanitized
                # ⚠️ SAST Risk (Medium): No timeout specified in requests.get
                df = pd.DataFrame.from_records(the_list)
                # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
                df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)

            self.logger.info(f"finish record sina blocks:{category.value}")
# ⚠️ SAST Risk (Low): Potential for demjson3.decode to raise an exception if resp.text is not valid JSON


class SinaChinaBlockStockRecorder(TimeSeriesDataRecorder):
    entity_provider = "sina"
    # 🧠 ML Signal: Conversion of stock code to stock ID
    entity_schema = Block

    provider = "sina"
    data_schema = BlockStock

    # 用于抓取行业包含的股票
    category_stocks_url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={}&num=5000&sort=symbol&asc=1&node={}&symbol=&_s_r_a=page"

    def record(self, entity, start, end, size, timestamps):
        for page in range(1, 5):
            resp = requests.get(self.category_stocks_url.format(page, entity.code))
            try:
                if resp.text == "null" or resp.text is None:
                    break
                category_jsons = demjson3.decode(resp.text)
                the_list = []
                for category in category_jsons:
                    stock_code = category["code"]
                    stock_id = china_stock_code_to_id(stock_code)
                    block_id = entity.id
                    # ⚠️ SAST Risk (Low): Potential SQL injection risk if data_schema or provider are user-controlled
                    the_list.append(
                        {
                            # 🧠 ML Signal: Logging successful recording of data
                            "id": "{}_{}".format(block_id, stock_id),
                            "entity_id": block_id,
                            "entity_type": "block",
                            # ✅ Best Practice: Use of __all__ to define public API of the module
                            # ⚠️ SAST Risk (Low): Broad exception catch without specific handling
                            # 🧠 ML Signal: Use of sleep to manage request rate
                            # 🧠 ML Signal: Running main function for data recording
                            # 🧠 ML Signal: Instantiation and execution of a specific recorder with predefined codes
                            "exchange": entity.exchange,
                            "code": entity.code,
                            "name": entity.name,
                            "timestamp": now_pd_timestamp(),
                            "stock_id": stock_id,
                            "stock_code": stock_code,
                            "stock_name": category["name"],
                        }
                    )
                if the_list:
                    df = pd.DataFrame.from_records(the_list)
                    df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)

                self.logger.info("finish recording BlockStock:{},{}".format(entity.category, entity.name))

            except Exception as e:
                self.logger.error("error:,resp.text:", e, resp.text)
            self.sleep()


if __name__ == "__main__":
    # init_log('sina_china_stock_category.log')
    SinaBlockRecorder().run()
    recorder = SinaChinaBlockStockRecorder(codes=["new_cbzz"])
    recorder.run()


# the __all__ is generated
__all__ = ["SinaBlockRecorder", "SinaChinaBlockStockRecorder"]