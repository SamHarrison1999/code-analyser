# -*- coding: utf-8 -*-
from zvt.domain import SpoDetail, DividendFinancing
from zvt.recorders.eastmoney.common import EastmoneyPageabeDataRecorder
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import now_pd_timestamp
from zvt.utils.utils import to_float
# ✅ Best Practice: Define class variables for schema and URLs to improve readability and maintainability.


# ✅ Best Practice: Use descriptive variable names for URLs to clarify their purpose.
class SPODetailRecorder(EastmoneyPageabeDataRecorder):
    data_schema = SpoDetail
    # ✅ Best Practice: Method name should be descriptive of its functionality
    # ✅ Best Practice: Reuse the URL variable to avoid duplication and potential errors if the URL changes.

    url = "https://emh5.eastmoney.com/api/FenHongRongZi/GetZengFaMingXiList"
    # 🧠 ML Signal: Returns a hardcoded string, indicating a constant value
    # ✅ Best Practice: Use descriptive variable names for path fields to clarify their purpose.
    # ✅ Best Practice: Method name is descriptive and follows snake_case naming convention
    page_url = url
    # ✅ Best Practice: Using a dictionary to map keys to tuples for structured data representation
    path_fields = ["ZengFaMingXiList"]

    def get_original_time_field(self):
        # 🧠 ML Signal: Mapping keys to functions for data transformation
        return "ZengFaShiJian"

    # 🧠 ML Signal: Mapping keys to functions for data transformation
    def get_data_map(self):
        return {
            # 🧠 ML Signal: Mapping keys to functions for data transformation
            # 🧠 ML Signal: Extracting year from timestamp, common pattern for time-based operations
            "spo_issues": ("ShiJiZengFa", to_float),
            # 🧠 ML Signal: List comprehension to extract attributes from objects
            "spo_price": ("ZengFaJiaGe", to_float),
            "spo_raising_fund": ("ShiJiMuJi", to_float),
        }

    def on_finish(self):
        last_year = str(now_pd_timestamp().year)
        codes = [item.code for item in self.entities]
        need_filleds = DividendFinancing.query_data(
            provider=self.provider,
            codes=codes,
            return_type="domain",
            session=self.session,
            filters=[DividendFinancing.spo_raising_fund.is_(None)],
            end_timestamp=last_year,
        )

        for item in need_filleds:
            df = SpoDetail.query_data(
                # 🧠 ML Signal: Conditional check for DataFrame nullity
                provider=self.provider,
                entity_id=item.entity_id,
                columns=[SpoDetail.timestamp, SpoDetail.spo_raising_fund],
                # ⚠️ SAST Risk (Low): Direct database commit without error handling
                start_timestamp=item.timestamp,
                end_timestamp="{}-12-31".format(item.timestamp.year),
            # ✅ Best Practice: Explicit call to superclass method
            # 🧠 ML Signal: Instantiation and execution of a class with specific parameters
            # ✅ Best Practice: Use of __all__ to define public API of the module
            )
            if pd_is_not_null(df):
                item.spo_raising_fund = df["spo_raising_fund"].sum()
                self.session.commit()
        super().on_finish()


if __name__ == "__main__":
    # init_log('spo_detail.log')

    recorder = SPODetailRecorder(codes=["000999"])
    recorder.run()


# the __all__ is generated
__all__ = ["SPODetailRecorder"]