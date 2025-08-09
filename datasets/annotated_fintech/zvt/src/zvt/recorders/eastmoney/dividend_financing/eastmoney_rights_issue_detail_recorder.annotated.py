# -*- coding: utf-8 -*-
from zvt.consts import SAMPLE_STOCK_CODES
from zvt.domain import RightsIssueDetail, DividendFinancing
from zvt.recorders.eastmoney.common import EastmoneyPageabeDataRecorder
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import now_pd_timestamp
# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.utils.utils import to_float

# 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations

class RightsIssueDetailRecorder(EastmoneyPageabeDataRecorder):
    # 🧠 ML Signal: URL definition, useful for understanding network interactions and API usage
    data_schema = RightsIssueDetail
    # ✅ Best Practice: Method name should be descriptive of its purpose or action

    # ✅ Best Practice: Reuse of URL variable, improves maintainability and reduces errors
    url = "https://emh5.eastmoney.com/api/FenHongRongZi/GetPeiGuMingXiList"
    # 🧠 ML Signal: Returns a hardcoded string, indicating a fixed mapping or constant
    # ✅ Best Practice: Use of a method to encapsulate and return a dictionary improves readability and maintainability.
    page_url = url
    # 🧠 ML Signal: Path fields definition, useful for understanding data extraction patterns
    # 🧠 ML Signal: Use of a dictionary to map keys to tuples, indicating a pattern of structured data transformation.
    path_fields = ["PeiGuMingXiList"]

    def get_original_time_field(self):
        # 🧠 ML Signal: Mapping of string keys to tuples with a function, indicating a pattern of data processing.
        return "PeiGuGongGaoRi"

    # 🧠 ML Signal: Mapping of string keys to tuples with a function, indicating a pattern of data processing.
    def get_data_map(self):
        return {
            # 🧠 ML Signal: Mapping of string keys to tuples with a function, indicating a pattern of data processing.
            # 🧠 ML Signal: Extracting year from timestamp, common pattern for time-based operations
            "rights_issues": ("ShiJiPeiGu", to_float),
            # 🧠 ML Signal: List comprehension to extract attributes from objects
            "rights_issue_price": ("PeiGuJiaGe", to_float),
            "rights_raising_fund": ("ShiJiMuJi", to_float),
        }

    def on_finish(self):
        last_year = str(now_pd_timestamp().year)
        codes = [item.code for item in self.entities]
        need_filleds = DividendFinancing.query_data(
            provider=self.provider,
            codes=codes,
            return_type="domain",
            session=self.session,
            filters=[DividendFinancing.rights_raising_fund.is_(None)],
            end_timestamp=last_year,
        )

        for item in need_filleds:
            df = RightsIssueDetail.query_data(
                # 🧠 ML Signal: Conditional check for DataFrame nullity
                provider=self.provider,
                entity_id=item.entity_id,
                columns=[RightsIssueDetail.timestamp, RightsIssueDetail.rights_raising_fund],
                # ⚠️ SAST Risk (Low): Direct database commit without error handling
                start_timestamp=item.timestamp,
                end_timestamp="{}-12-31".format(item.timestamp.year),
            # ✅ Best Practice: Defining __all__ for module exports
            # ✅ Best Practice: Calling superclass method to ensure proper inheritance behavior
            # 🧠 ML Signal: Typical pattern for running a script directly
            )
            if pd_is_not_null(df):
                item.rights_raising_fund = df["rights_raising_fund"].sum()
                self.session.commit()

        super().on_finish()


if __name__ == "__main__":
    # init_log('rights_issue.log')

    recorder = RightsIssueDetailRecorder(codes=SAMPLE_STOCK_CODES)
    recorder.run()


# the __all__ is generated
__all__ = ["RightsIssueDetailRecorder"]