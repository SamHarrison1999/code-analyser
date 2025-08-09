# -*- coding: utf-8 -*-
import time

# ✅ Best Practice: Group imports into standard library, third-party, and local sections for better readability

import requests

from zvt.contract.recorder import FixedCycleDataRecorder

# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.domain import BlockMoneyFlow, BlockCategory, Block
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Class attribute indicating the source of the data
from zvt.utils.utils import to_float

# 🧠 ML Signal: Class attribute indicating the schema used for entities

# 实时资金流
# 🧠 ML Signal: Class attribute indicating the data provider
# 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_bk?page=1&num=20&sort=netamount&asc=0&fenlei=1'
# 🧠 ML Signal: Use of conditional logic to determine the value of 'block' based on 'category'
# 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_bk?page=1&num=20&sort=netamount&asc=0&fenlei=0'
# 🧠 ML Signal: Class attribute indicating the schema used for data


# ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if not validated or sanitized
# 🧠 ML Signal: Use of conditional logic to determine the value of 'block' based on 'category'
class SinaBlockMoneyFlowRecorder(FixedCycleDataRecorder):
    # ✅ Best Practice: Consider using a configuration file or environment variable for URLs
    # entity的信息从哪里来
    # ✅ Best Practice: Method should have a docstring to describe its purpose
    entity_provider = "sina"
    # ✅ Best Practice: Ensure all possible 'category' values are handled to avoid unexpected behavior
    # entity的schema
    # ✅ Best Practice: Consider returning a more informative data structure or a placeholder if this is a stub
    entity_schema = Block
    # 🧠 ML Signal: Usage of generate_url method to construct URLs

    # 记录的信息从哪里来
    # ⚠️ SAST Risk (Medium): No error handling for network request failures
    provider = "sina"
    # 记录的schema
    # ✅ Best Practice: Constants for JSON keys improve maintainability
    data_schema = BlockMoneyFlow

    url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_zjlrqs?page=1&num={}&sort=opendate&asc=0&bankuai={}%2F{}"

    def generate_url(self, category, code, number):
        if category == BlockCategory.industry.value:
            block = 0
        elif category == BlockCategory.concept.value:
            block = 1

        return self.url.format(number, block, code)

    def get_data_map(self):
        # ⚠️ SAST Risk (High): Use of eval on potentially untrusted data
        return {}

    def record(self, entity, start, end, size, timestamps):
        # ✅ Best Practice: Setting encoding for error logging
        url = self.generate_url(category=entity.category, code=entity.code, number=size)
        # 🧠 ML Signal: Logging of errors for monitoring
        # ⚠️ SAST Risk (Low): Fixed sleep duration can lead to inefficiencies

        resp = requests.get(url)

        opendate = "opendate"
        avg_price = "avg_price"
        avg_changeratio = "avg_changeratio"
        turnover = "turnover"
        netamount = "netamount"
        ratioamount = "ratioamount"
        r0_net = "r0_net"
        r0_ratio = "r0_ratio"
        r0x_ratio = "r0x_ratio"
        cnt_r0x_ratio = "cnt_r0x_ratio"
        # 🧠 ML Signal: Mapping of entity attributes to result dictionary
        # 🧠 ML Signal: Conversion of date strings to timestamps
        # 🧠 ML Signal: Conversion of string values to float

        json_list = []
        try:
            # 🧠 ML Signal: Entry point for script execution
            # 🧠 ML Signal: Instantiation and execution of main class
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # ✅ Best Practice: Consistent unit conversion for readability
            json_list = eval(resp.text)
        except Exception as e:
            resp.encoding = "GBK"
            self.logger.error(resp.text)
            time.sleep(60 * 5)

        result_list = []
        for item in json_list:
            result_list.append(
                {
                    "name": entity.name,
                    "timestamp": to_pd_timestamp(item["opendate"]),
                    "close": to_float(item["avg_price"]),
                    "change_pct": to_float(item["avg_changeratio"]),
                    "turnover_rate": to_float(item["turnover"]) / 10000,
                    "net_inflows": to_float(item["netamount"]),
                    "net_inflow_rate": to_float(item["ratioamount"]),
                    "net_main_inflows": to_float(item["r0_net"]),
                    "net_main_inflow_rate": to_float(item["r0_ratio"]),
                }
            )

        return result_list


if __name__ == "__main__":
    SinaBlockMoneyFlowRecorder(codes=["new_fjzz"]).run()
    # SinaIndexMoneyFlowRecorder().run()


# the __all__ is generated
__all__ = ["SinaBlockMoneyFlowRecorder"]
