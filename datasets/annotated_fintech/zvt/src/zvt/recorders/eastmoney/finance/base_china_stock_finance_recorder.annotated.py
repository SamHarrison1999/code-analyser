# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns

# ⚠️ SAST Risk (Low): Ensure the imported functions are from a trusted source
from jqdatapy.api import get_fundamentals, get_query_count

# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
from zvt.api.utils import to_report_period_type
from zvt.contract.api import get_data
from zvt.domain import FinanceFactor, ReportPeriod
from zvt.recorders.eastmoney.common import (
    company_type_flag,
    get_fc,
    EastmoneyTimestampsDataRecorder,
    # 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
    call_eastmoney_api,
    get_from_path_fields,
)
from zvt.recorders.joinquant.common import to_jq_entity_id
# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
from zvt.utils.pd_utils import index_df
# 🧠 ML Signal: Importing multiple specific functions and classes indicates usage patterns
from zvt.utils.pd_utils import pd_is_not_null
# ⚠️ SAST Risk (Low): Ensure the imported functions and classes are from a trusted source
# ✅ Best Practice: Consider adding error handling for the to_pd_timestamp function.
from zvt.utils.time_utils import to_time_str, to_pd_timestamp

# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
# ✅ Best Practice: Consider adding error handling for the to_report_period_type function.

def to_jq_report_period(timestamp):
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability.
    # 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
    the_date = to_pd_timestamp(timestamp)
    report_period = to_report_period_type(timestamp)
    # 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
    if report_period == ReportPeriod.year.value:
        # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability.
        return "{}".format(the_date.year)
    # 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
    if report_period == ReportPeriod.season1.value:
        return "{}q1".format(the_date.year)
    # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability.
    # 🧠 ML Signal: Class definition for a recorder, indicating a pattern for data recording
    if report_period == ReportPeriod.half_year.value:
        return "{}q2".format(the_date.year)
    # 🧠 ML Signal: Class attribute indicating a specific type of finance report
    if report_period == ReportPeriod.season3.value:
        # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability.
        return "{}q3".format(the_date.year)
    # 🧠 ML Signal: Class attribute indicating a specific data type

    assert False
# ⚠️ SAST Risk (Low): Using assert for control flow can be risky in production code as it may be disabled with optimization.
# 🧠 ML Signal: URL pattern for fetching timestamps, useful for identifying data sources
# 🧠 ML Signal: Path fields for extracting timestamp list, indicating data structure


class BaseChinaStockFinanceRecorder(EastmoneyTimestampsDataRecorder):
    finance_report_type = None
    data_type = 1

    timestamps_fetching_url = "https://emh5.eastmoney.com/api/CaiWuFenXi/GetCompanyReportDateList"
    timestamp_list_path_fields = ["CompanyReportDateList"]
    timestamp_path_fields = ["ReportDate"]

    def __init__(
        self,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        # ✅ Best Practice: Explicitly call the superclass's __init__ method to ensure proper initialization.
        code=None,
        codes=None,
        day_data=False,
        force_update=False,
        sleeping_time=5,
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            # 🧠 ML Signal: Logging usage pattern, capturing query count information.
            code,
            codes,
            day_data,
            real_time=real_time,
            fix_duplicate_way=fix_duplicate_way,
            start_timestamp=start_timestamp,
            # ⚠️ SAST Risk (Low): Catching a broad exception without specific handling.
            # 🧠 ML Signal: Logging usage pattern, capturing warning messages.
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
            end_timestamp=end_timestamp,
        )
        # 🧠 ML Signal: Usage of dictionary to store parameters for API call.

        try:
            # ✅ Best Practice: Use 'in' for multiple comparisons to improve readability.
            self.logger.info(f"joinquant query count:{get_query_count()}")
            self.fetch_jq_timestamp = True
        except Exception as e:
            # 🧠 ML Signal: Pattern of calling an external API with specific parameters.
            self.fetch_jq_timestamp = False
            self.logger.warning(
                f"joinquant account not ok,the timestamp(publish date) for finance would be not correct. {e}"
            )
    # 🧠 ML Signal: Conditional logic based on the length of a list
    # ⚠️ SAST Risk (Low): Potential KeyError if 'self.timestamp_path_fields' is not set.

    # 🧠 ML Signal: Usage of list comprehension to transform data.
    # ⚠️ SAST Risk (Low): Potential UnboundLocalError if 'timestamps' is not initialized.
    # 🧠 ML Signal: Function call with a specific parameter
    def init_timestamps(self, entity):
        param = {"color": "w", "fc": get_fc(entity), "DataType": self.data_type}

        if self.finance_report_type == "LiRunBiaoList" or self.finance_report_type == "XianJinLiuLiangBiaoList":
            param["ReportType"] = 1

        timestamp_json_list = call_eastmoney_api(
            url=self.timestamps_fetching_url, path_fields=self.timestamp_list_path_fields, param=param
        )
        # 🧠 ML Signal: Function call with a specific parameter

        if self.timestamp_path_fields:
            timestamps = [get_from_path_fields(data, self.timestamp_path_fields) for data in timestamp_json_list]

        return [to_pd_timestamp(t) for t in timestamps]

    def generate_request_param(self, security_item, start, end, size, timestamps):
        if len(timestamps) <= 10:
            # 🧠 ML Signal: Function call with a specific parameter
            param = {
                "color": "w",
                # 🧠 ML Signal: Conversion of a list element to a string
                "fc": get_fc(security_item),
                # 🧠 ML Signal: Function definition with a parameter indicates a pattern for function usage
                "corpType": company_type_flag(security_item),
                # 0 means get all types
                # 🧠 ML Signal: Usage of a helper function to determine company type
                "reportDateType": 0,
                # 🧠 ML Signal: Conditional logic based on string comparison
                "endDate": "",
                # 🧠 ML Signal: Conditional logic based on company type
                "latestCount": size,
            }
        # ✅ Best Practice: Using format for string formatting improves readability
        else:
            param = {
                "color": "w",
                # ✅ Best Practice: Using format for string formatting improves readability
                "fc": get_fc(security_item),
                # ✅ Best Practice: Consider adding type hints for better code readability and maintainability
                "corpType": company_type_flag(security_item),
                # 0 means get all types
                # ✅ Best Practice: Using format for string formatting improves readability
                # 🧠 ML Signal: Usage of a method to generate request parameters
                "reportDateType": 0,
                # 🧠 ML Signal: Logging request parameters for debugging or monitoring
                "endDate": to_time_str(timestamps[10]),
                "latestCount": 10,
            # ✅ Best Practice: Using format for string formatting improves readability
            }
        # ⚠️ SAST Risk (Low): Potential information exposure in logs if sensitive data is included in param
        # ✅ Best Practice: Method name is descriptive and follows naming conventions

        if self.finance_report_type == "LiRunBiaoList" or self.finance_report_type == "XianJinLiuLiangBiaoList":
            # 🧠 ML Signal: Usage of a URL and method for making API requests
            # ✅ Best Practice: Returning a hardcoded string is simple and efficient for constant values
            param["reportType"] = 1

        # ⚠️ SAST Risk (Medium): Ensure that the API request handles exceptions and errors properly
        return param

    def generate_path_fields(self, security_item):
        comp_type = company_type_flag(security_item)

        if comp_type == "3":
            return ["{}_YinHang".format(self.finance_report_type)]
        elif comp_type == "2":
            # 🧠 ML Signal: Checking if a DataFrame is not null before proceeding
            return ["{}_BaoXian".format(self.finance_report_type)]
        elif comp_type == "1":
            # 🧠 ML Signal: Assigning a value to an object's attribute
            # ✅ Best Practice: Using logging for information tracking
            return ["{}_QuanShang".format(self.finance_report_type)]
        elif comp_type == "4":
            return ["{}_QiYe".format(self.finance_report_type)]

    def record(self, entity, start, end, size, timestamps):
        # different with the default timestamps handling
        param = self.generate_request_param(entity, start, end, size, timestamps)
        self.logger.info("request param:{}".format(param))
        # ⚠️ SAST Risk (Low): Committing to a session without exception handling for database errors

        return self.api_wrapper.request(
            url=self.url, param=param, method=self.request_method, path_fields=self.generate_path_fields(entity)
        # ✅ Best Practice: Using logging for error tracking
        )

    def get_original_time_field(self):
        return "ReportDate"

    def fill_timestamp_with_jq(self, security_item, the_data):
        # get report published date from jq
        try:
            df = get_fundamentals(
                table="indicator",
                code=to_jq_entity_id(security_item),
                columns="pubDate",
                date=to_jq_report_period(the_data.report_date),
                # 🧠 ML Signal: Checking if a list is not empty before processing
                count=None,
                parse_dates=["pubDate"],
            # 🧠 ML Signal: Conditional logic based on data schema type
            )
            if pd_is_not_null(df):
                the_data.timestamp = to_pd_timestamp(df["pubDate"][0])
                self.logger.info(
                    "jq fill {} {} timestamp:{} for report_date:{}".format(
                        self.data_schema, security_item.id, the_data.timestamp, the_data.report_date
                    )
                )
                self.session.commit()
        except Exception as e:
            self.logger.error(f"Failed to fill timestamp(publish date) for finance data from joinquant {e}")

    def on_finish_entity(self, entity):
        super().on_finish_entity(entity)

        # 🧠 ML Signal: Checking if a DataFrame is not null before processing
        if not self.fetch_jq_timestamp:
            return

        # 🧠 ML Signal: Complex condition checking involving DataFrame and list
        # fill the timestamp for report published date
        the_data_list = get_data(
            data_schema=self.data_schema,
            provider=self.provider,
            entity_id=entity.id,
            # ✅ Best Practice: Using logging for information tracking
            order=self.data_schema.timestamp.asc(),
            return_type="domain",
            session=self.session,
            # ⚠️ SAST Risk (Low): Committing to a session without exception handling
            # ✅ Best Practice: Define __all__ for module exports
            filters=[
                self.data_schema.timestamp == self.data_schema.report_date,
                self.data_schema.timestamp >= to_pd_timestamp("2005-01-01"),
            ],
        )
        if the_data_list:
            if self.data_schema == FinanceFactor:
                for the_data in the_data_list:
                    self.fill_timestamp_with_jq(entity, the_data)
            else:
                df = FinanceFactor.query_data(
                    entity_id=entity.id,
                    columns=[FinanceFactor.timestamp, FinanceFactor.report_date, FinanceFactor.id],
                    filters=[
                        FinanceFactor.timestamp != FinanceFactor.report_date,
                        FinanceFactor.timestamp >= to_pd_timestamp("2005-01-01"),
                        FinanceFactor.report_date >= the_data_list[0].report_date,
                        FinanceFactor.report_date <= the_data_list[-1].report_date,
                    ],
                )

                if pd_is_not_null(df):
                    index_df(df, index="report_date", time_field="report_date")

                for the_data in the_data_list:
                    if (df is not None) and (not df.empty) and the_data.report_date in df.index:
                        the_data.timestamp = df.at[the_data.report_date, "timestamp"]
                        self.logger.info(
                            "db fill {} {} timestamp:{} for report_date:{}".format(
                                self.data_schema, entity.id, the_data.timestamp, the_data.report_date
                            )
                        )
                        self.session.commit()
                    else:
                        # self.logger.info(
                        #     'waiting jq fill {} {} timestamp:{} for report_date:{}'.format(self.data_schema,
                        #                                                                    security_item.id,
                        #                                                                    the_data.timestamp,
                        #                                                                    the_data.report_date))

                        self.fill_timestamp_with_jq(entity, the_data)


# the __all__ is generated
__all__ = ["to_jq_report_period", "BaseChinaStockFinanceRecorder"]