# -*- coding: utf-8 -*-
# ✅ Best Practice: Group imports into standard library, third-party, and local sections for readability

import requests

from zvt.contract.api import get_entities
from zvt.contract.recorder import Recorder
from zvt.domain.meta.stock_meta import StockDetail, Stock
# 🧠 ML Signal: Custom class definition for stock recording
from zvt.recorders.exchange.exchange_stock_meta_recorder import ExchangeStockMetaRecorder
from zvt.utils.time_utils import to_pd_timestamp
# 🧠 ML Signal: Class attribute for data schema
from zvt.utils.utils import to_float, pct_to_float
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase for class names.

# 🧠 ML Signal: Class attribute for provider information

# ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.
class EastmoneyStockRecorder(ExchangeStockMetaRecorder):
    data_schema = Stock
    # ✅ Best Practice: Explicitly call the superclass's __init__ method to ensure proper initialization.
    # ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.
    provider = "eastmoney"

# 🧠 ML Signal: Instantiating and running a specific recorder class, indicating a pattern of data collection.

# ✅ Best Practice: Use of conditional logic to handle mutually exclusive parameters.
class EastmoneyStockDetailRecorder(Recorder):
    provider = "eastmoney"
    data_schema = StockDetail

    def __init__(self, force_update=False, sleeping_time=5, code=None, codes=None) -> None:
        super().__init__(force_update, sleeping_time)
        # ✅ Best Practice: Use of conditional logic to set filters based on the force_update flag.
        # 🧠 ML Signal: Fetching entities with specific filters and parameters, indicating a pattern of data retrieval.

        # get list at first
        EastmoneyStockRecorder().run()

        if codes is None and code is not None:
            self.codes = [code]
        else:
            self.codes = codes
        filters = None
        if not self.force_update:
            filters = [StockDetail.profile.is_(None)]
        self.entities = get_entities(
            # ⚠️ SAST Risk (Low): Use of assert for type checking can be bypassed if Python is run with optimizations
            session=self.session,
            entity_schema=StockDetail,
            exchanges=None,
            codes=self.codes,
            filters=filters,
            return_type="domain",
            provider=self.provider,
        )

    def run(self):
        for security_item in self.entities:
            assert isinstance(security_item, StockDetail)

            if security_item.exchange == "sh":
                # ⚠️ SAST Risk (Medium): Potential for injection if securities_code is not properly sanitized
                fc = "{}01".format(security_item.code)
            if security_item.exchange == "sz":
                fc = "{}02".format(security_item.code)
            # ⚠️ SAST Risk (Low): No error handling for network request failures

            # 基本资料
            # param = {"color": "w", "fc": fc, "SecurityCode": "SZ300059"}
            # ⚠️ SAST Risk (Low): No error handling for JSON parsing or missing keys

            securities_code = f"{security_item.code}.{security_item.exchange.upper()}"
            param = {
                "type": "RPT_F10_ORG_BASICINFO",
                "sty": "ORG_PROFIE,MAIN_BUSINESS,FOUND_DATE,EM2016,BLGAINIAN,REGIONBK",
                "filter": f"(SECUCODE=\"{securities_code}\")",
                "client": "app",
                "source": "SECURITIES",
                "pageNumber": 1,
                "pageSize": 1
            }
            resp = requests.get("https://datacenter.eastmoney.com/securities/api/data/get", params=param)
            resp.encoding = "utf8"

            # ⚠️ SAST Risk (Medium): Potential for injection if securities_code is not properly sanitized
            resp_json = resp.json()["result"]["data"][0]

            security_item.profile = resp_json["ORG_PROFIE"]
            security_item.main_business = resp_json["MAIN_BUSINESS"]
            security_item.date_of_establishment = to_pd_timestamp(resp_json["FOUND_DATE"])

            # ⚠️ SAST Risk (Low): No error handling for network request failures
            # 关联行业
            industries = ",".join(resp_json["EM2016"].split("-"))
            security_item.industries = industries
            # ⚠️ SAST Risk (Low): No error handling for JSON parsing or missing keys

            # 关联概念
            security_item.concept_indices = resp_json["BLGAINIAN"]

            # 关联地区
            security_item.area_indices = resp_json["REGIONBK"]
            # ⚠️ SAST Risk (Low): No error handling for database commit failures
            # ✅ Best Practice: Use f-string for better readability
            # 🧠 ML Signal: Instantiation and usage of a custom class
            # 🧠 ML Signal: Method call with specific parameters
            # 🧠 ML Signal: Definition of module exports

            self.sleep()

            # 发行相关
            param = {
                "reportName": "RPT_F10_ORG_ISSUEINFO",
                "columns": "AFTER_ISSUE_PE,ISSUE_PRICE,TOTAL_ISSUE_NUM,NET_RAISE_FUNDS,ONLINE_ISSUE_LWR",
                "filter": f"(SECUCODE=\"{securities_code}\")(TYPENEW=\"4\")",
                "client": "app",
                "source": "SECURITIES",
                "pageNumber": 1,
                "pageSize": 1
            }
            resp = requests.get("https://datacenter.eastmoney.com/securities/api/data/v1/get", params=param)
            resp.encoding = "utf8"

            resp_json = resp.json()["result"]["data"][0]

            security_item.issue_pe = resp_json["AFTER_ISSUE_PE"]
            security_item.price = resp_json["ISSUE_PRICE"]
            security_item.issues = resp_json["TOTAL_ISSUE_NUM"]
            security_item.raising_fund = resp_json.get("NET_RAISE_FUNDS")
            security_item.net_winning_rate = resp_json["ONLINE_ISSUE_LWR"]

            self.session.commit()

            self.logger.info("finish recording stock meta for:{}".format(security_item.code))

            self.sleep()


if __name__ == "__main__":
    # init_log('china_stock_meta.log')

    recorder = EastmoneyStockRecorder()
    recorder.run()
    StockDetail.record_data(codes=["000338", "000777"], provider="eastmoney")


# the __all__ is generated
__all__ = ["EastmoneyStockRecorder", "EastmoneyStockDetailRecorder"]