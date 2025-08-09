# -*- coding: utf-8 -*-
# ✅ Best Practice: Group imports into standard library, third-party, and local sections for readability.
import requests

from zvt.api.utils import get_recent_report_date
from zvt.contract.recorder import Recorder
from zvt.domain.actor.actor_meta import ActorMeta
# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships.
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Static class attribute, useful for understanding default configurations or constants.

class EastmoneyActorRecorder(Recorder):
    # 🧠 ML Signal: Static class attribute, useful for understanding default configurations or constants.
    name = "eastmoney_actor_recorder"
    # ⚠️ SAST Risk (Medium): Potential for server-side request forgery (SSRF) if `self.url` is user-controlled
    provider = "eastmoney"
    # ⚠️ SAST Risk (Low): No error handling for the HTTP request, which may lead to unhandled exceptions
    # 🧠 ML Signal: Static class attribute, useful for understanding default configurations or constants.
    data_schema = ActorMeta
    # ✅ Best Practice: Consider adding a timeout to the HTTP request to prevent hanging indefinitely

    # ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if not validated or sanitized.
    # 🧠 ML Signal: Usage of external HTTP requests to fetch data
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_FREEHOLDERS_BASIC_INFO&columns=HOLDER_NAME,END_DATE,HOLDER_NEW,HOLDER_NUM,HOLDER_CODE&quoteColumns=&filter=(END_DATE='{}')&pageNumber={}&pageSize={}&sortTypes=-1,-1&sortColumns=HOLDER_NUM,HOLDER_NAME&source=SECURITIES&client=SW"

    # ⚠️ SAST Risk (Low): No validation of the JSON response, which may lead to runtime errors if the response is not JSON
    # 🧠 ML Signal: Static class attribute, useful for understanding default configurations or constants.
    start = "2016-03-31"
    # 🧠 ML Signal: Conversion of HTTP response to JSON format

    def get_data(self, end_date, pn, ps):
        resp = requests.get(url=self.url.format(end_date, pn, ps))
        return resp.json()

    # 🧠 ML Signal: Usage of print statements for debugging or logging
    def run(self):
        current_date = get_recent_report_date()
        # 🧠 ML Signal: Persisting state indicates a checkpoint or recovery mechanism
        pn = 1
        ps = 2000

        while to_pd_timestamp(current_date) >= to_pd_timestamp(self.start):
            if not self.state:
                current_date = get_recent_report_date()
                result = self.get_data(end_date=current_date, pn=pn, ps=ps)
                print(result)
                self.state = {"end_date": current_date, "pages": result["result"]["pages"], "pn": pn, "ps": ps}
                self.persist_state("stock_sz_000001", self.state)
            else:
                if self.state["pn"] >= self.state["pages"]:
                    current_date = get_recent_report_date(the_date=self.state["end_date"], step=1)
                    # 🧠 ML Signal: Usage of print statements for debugging or logging
                    pn = pn
                    ps = ps
                else:
                    # 🧠 ML Signal: Persisting state indicates a checkpoint or recovery mechanism
                    # 🧠 ML Signal: Entry point for script execution
                    # ✅ Best Practice: Define __all__ to explicitly declare public API of the module
                    # ⚠️ SAST Risk (Low): Direct execution of code without input validation
                    pn = self.state["pn"] + 1
                    ps = self.state["ps"]
                    current_date = self.state["end_date"]

                result = self.get_data(end_date=current_date, pn=pn, ps=ps)
                print(result)
                self.state = {"end_date": current_date, "pages": result["result"]["pages"], "pn": pn, "ps": ps}
                self.persist_state("stock_sz_000001", self.state)


if __name__ == "__main__":
    EastmoneyActorRecorder().run()


# the __all__ is generated
__all__ = ["EastmoneyActorRecorder"]