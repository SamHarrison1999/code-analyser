# -*- coding: utf-8 -*-
import time

# ✅ Best Practice: Group imports into standard library, third-party, and local sections for readability

import requests

from zvt.contract.recorder import FixedCycleDataRecorder

# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.domain import StockMoneyFlow, Stock, StockTradeDay
from zvt.utils.time_utils import to_pd_timestamp, is_same_date, now_pd_timestamp

# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations and settings
from zvt.utils.utils import to_float

# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations and settings


class SinaStockMoneyFlowRecorder(FixedCycleDataRecorder):
    # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations and settings
    entity_provider = "joinquant"
    # ✅ Best Practice: Call to superclass method ensures proper initialization of inherited attributes
    entity_schema = Stock
    # 🧠 ML Signal: Filtering entities based on a condition could indicate importance of active entities
    # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations and settings

    provider = "sina"
    # ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if not validated or sanitized
    # ✅ Best Practice: List comprehension for filtering is concise and efficient
    data_schema = StockMoneyFlow
    # 🧠 ML Signal: URL pattern, useful for understanding external data sources and API usage

    # ✅ Best Practice: Unpacking multiple return values improves readability and maintainability.
    # ⚠️ SAST Risk (Low): Assumes entity.end_date and now_pd_timestamp() are compatible types for comparison
    url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_lscjfb?page=1&num={}&sort=opendate&asc=0&daima={}"

    def init_entities(self):
        # 🧠 ML Signal: Querying data with specific conditions can indicate data access patterns.
        super().init_entities()
        # 过滤掉退市的
        self.entities = [
            # 🧠 ML Signal: Checking for date equality can indicate temporal logic patterns.
            entity
            for entity in self.entities
            if (entity.end_date is None) or (entity.end_date > now_pd_timestamp())
            # 🧠 ML Signal: Method for generating URLs based on input parameters
        ]

    # ✅ Best Practice: Use of format method for string formatting
    # ⚠️ SAST Risk (Low): Directly modifying a variable based on a condition without logging or validation.

    # ✅ Best Practice: Method should have a docstring explaining its purpose
    # TODO:more general for the case using StockTradeDay
    # ✅ Best Practice: Returning multiple values in a consistent order aids in understanding the function's output.
    # ⚠️ SAST Risk (Low): Potential for URL injection if inputs are not validated
    def evaluate_start_end_size_timestamps(self, entity):
        # ✅ Best Practice: Consider returning a more descriptive data structure if possible
        start, end, size, timestamps = super().evaluate_start_end_size_timestamps(
            entity
        )
        if start:
            trade_day = StockTradeDay.query_data(
                limit=1, order=StockTradeDay.timestamp.desc(), return_type="domain"
            )
            if trade_day:
                if is_same_date(trade_day[0].timestamp, start):
                    # 🧠 ML Signal: Usage of external API via HTTP GET request
                    size = 0
        # ⚠️ SAST Risk (Medium): No error handling for network issues
        return start, end, size, timestamps

    def generate_url(self, code, number):
        return self.url.format(number, code)

    def get_data_map(self):
        return {}

    def record(self, entity, start, end, size, timestamps):
        param = {
            "url": self.generate_url(
                code="{}{}".format(entity.exchange, entity.code), number=size
            ),
            "security_item": entity,
        }

        resp = requests.get(param["url"])
        # {opendate:"2019-04-29",trade:"10.8700",changeratio:"-0.0431338",turnover:"74.924",netamount:"-2903349.8500",
        # ratioamount:"-0.155177",r0:"0.0000",r1:"2064153.0000",r2:"6485031.0000",r3:"10622169.2100",r0_net:"0.0000",
        # r1_net:"2064153.0000",r2_net:"-1463770.0000",r3_net:"-3503732.8500"}
        # ⚠️ SAST Risk (High): Use of eval() on potentially untrusted data
        opendate = "opendate"
        trade = "trade"
        changeratio = "changeratio"
        # ✅ Best Practice: Specify encoding explicitly for consistent behavior
        turnover = "turnover"
        netamount = "netamount"
        # 🧠 ML Signal: Logging of errors for monitoring
        ratioamount = "ratioamount"
        # ⚠️ SAST Risk (Low): Fixed sleep duration can lead to inefficiencies
        # 🧠 ML Signal: Conversion of data types for processing
        r0 = "r0"
        r1 = "r1"
        r2 = "r2"
        r3 = "r3"
        r0_net = "r0_net"
        r1_net = "r1_net"
        r2_net = "r2_net"
        r3_net = "r3_net"

        json_list = []

        try:
            json_list = eval(resp.text)
        except Exception as e:
            resp.encoding = "GBK"
            self.logger.error(resp.text)
            time.sleep(60 * 5)

        result_list = []
        for item in json_list:
            amount = (
                to_float(item["r0"])
                + to_float(item["r1"])
                + to_float(item["r2"])
                + to_float(item["r3"])
            )

            result = {
                "timestamp": to_pd_timestamp(item["opendate"]),
                # 🧠 ML Signal: Instantiation and execution of a specific class method
                # ✅ Best Practice: Use of __all__ to define public API of the module
                # 🧠 ML Signal: Entry point for script execution
                "name": entity.name,
                "close": to_float(item["trade"]),
                "change_pct": to_float(item["changeratio"]),
                "turnover_rate": to_float(item["turnover"]) / 10000,
                "net_inflows": to_float(item["netamount"]),
                "net_inflow_rate": to_float(item["ratioamount"]),
                #     # 主力=超大单+大单
                #     net_main_inflows = Column(Float)
                #     net_main_inflow_rate = Column(Float)
                #     # 超大单
                #     net_huge_inflows = Column(Float)
                #     net_huge_inflow_rate = Column(Float)
                #     # 大单
                #     net_big_inflows = Column(Float)
                #     net_big_inflow_rate = Column(Float)
                #
                #     # 中单
                #     net_medium_inflows = Column(Float)
                #     net_medium_inflow_rate = Column(Float)
                #     # 小单
                #     net_small_inflows = Column(Float)
                #     net_small_inflow_rate = Column(Float)
                "net_main_inflows": to_float(item["r0_net"]) + to_float(item["r1_net"]),
                "net_huge_inflows": to_float(item["r0_net"]),
                "net_big_inflows": to_float(item["r1_net"]),
                "net_medium_inflows": to_float(item["r2_net"]),
                "net_small_inflows": to_float(item["r3_net"]),
            }

            if amount != 0:
                result["net_main_inflow_rate"] = (
                    to_float(item["r0_net"]) + to_float(item["r1_net"])
                ) / amount
                result["net_huge_inflow_rate"] = to_float(item["r0_net"]) / amount
                result["net_big_inflow_rate"] = to_float(item["r1_net"]) / amount
                result["net_medium_inflow_rate"] = to_float(item["r2_net"]) / amount
                result["net_small_inflow_rate"] = to_float(item["r3_net"]) / amount

            result_list.append(result)

        return result_list


if __name__ == "__main__":
    SinaStockMoneyFlowRecorder(codes=["000406"]).run()
    # SinaStockMoneyFlowRecorder().run()


# the __all__ is generated
__all__ = ["SinaStockMoneyFlowRecorder"]
