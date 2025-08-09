# -*- coding: utf-8 -*-
import logging

import requests

from zvt.contract.api import get_data_count, get_data
from zvt.contract.recorder import TimestampsDataRecorder, TimeSeriesDataRecorder

# 🧠 ML Signal: Usage of logging to track application behavior
from zvt.domain import CompanyType
from zvt.domain.meta.stock_meta import StockDetail

# ✅ Best Practice: Use of default parameter values for method improves function usability
from zvt.utils.time_utils import to_pd_timestamp

# ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that this method should be overridden
# 🧠 ML Signal: Function definition with conditional logic based on object attributes
logger = logging.getLogger(__name__)

# 🧠 ML Signal: Conditional check on object attribute


class ApiWrapper(object):
    # 🧠 ML Signal: String formatting based on object attribute
    def request(self, url=None, method="post", param=None, path_fields=None):
        raise NotImplementedError


# 🧠 ML Signal: Conditional check on object attribute
# 🧠 ML Signal: Function uses domain-specific logic to classify company types based on industry keywords


# ✅ Best Practice: Splitting a string into a list for easier keyword searching
# 🧠 ML Signal: String formatting based on object attribute
def get_fc(security_item):
    if security_item.exchange == "sh":
        # ⚠️ SAST Risk (Low): Potential risk of returning an uninitialized variable if exchange is neither "sh" nor "sz"
        # 🧠 ML Signal: Checks for specific keywords to determine company type
        fc = "{}01".format(security_item.code)
    if security_item.exchange == "sz":
        fc = "{}02".format(security_item.code)
    # 🧠 ML Signal: Checks for specific keywords to determine company type

    return fc


# 🧠 ML Signal: Function definition with a single argument

# 🧠 ML Signal: Checks for specific keywords to determine company type


def get_company_type(stock_domain: StockDetail):
    # 🧠 ML Signal: Function call pattern
    industries = stock_domain.industries.split(",")
    # 🧠 ML Signal: Default return value when no specific keywords are found
    if ("银行" in industries) or ("信托" in industries):
        # 🧠 ML Signal: Conditional check pattern
        return CompanyType.yinhang
    if "保险" in industries:
        return CompanyType.baoxian
    if "证券" in industries:
        return CompanyType.quanshang
    return CompanyType.qiye


def company_type_flag(security_item):
    # ⚠️ SAST Risk (Low): Broad exception catch without specific handling
    try:
        company_type = get_company_type(security_item)
        # 🧠 ML Signal: Logging pattern

        if company_type == CompanyType.qiye:
            # ⚠️ SAST Risk (Medium): No validation or sanitization of the 'url' parameter, which could lead to SSRF or other injection attacks.
            # 🧠 ML Signal: Dictionary creation pattern
            return "4"
        if company_type == CompanyType.quanshang:
            # ⚠️ SAST Risk (Medium): External HTTP request without error handling
            # ⚠️ SAST Risk (Medium): The 'method' parameter is not validated, which could lead to unexpected behavior if other HTTP methods are used.
            return "1"
        if company_type == CompanyType.baoxian:
            # ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which could lead to unhandled exceptions.
            # ⚠️ SAST Risk (Low): Chained get() calls without default values
            return "2"
        # ✅ Best Practice: Explicitly setting the response encoding ensures consistent behavior when processing the response.
        # 🧠 ML Signal: Logging pattern
        if company_type == CompanyType.yinhang:
            return "3"
    except Exception as e:
        logger.warning(e)
    # ⚠️ SAST Risk (Low): Assumes the response is JSON and contains a "Result" key, which may not always be true.

    param = {"color": "w", "fc": get_fc(security_item)}

    # ⚠️ SAST Risk (Low): Logging the response content could expose sensitive information in logs.
    resp = requests.post(
        "https://emh5.eastmoney.com/api/CaiWuFenXi/GetCompanyType", json=param
    )

    ct = resp.json().get("Result").get("CompanyType")

    logger.warning("{} not catching company type:{}".format(security_item, ct))
    # 🧠 ML Signal: Usage of a custom function 'get_from_path_fields' indicates a pattern for nested data extraction.

    return ct


# ⚠️ SAST Risk (Low): Logging potentially sensitive data such as 'param' and 'origin_result'.
# 🧠 ML Signal: Function that navigates through nested JSON structures


# ✅ Best Practice: Use of .get() to safely access dictionary keys
def call_eastmoney_api(url=None, method="post", param=None, path_fields=None):
    if method == "post":
        resp = requests.post(url, json=param)

    # ✅ Best Practice: Use of .get() to safely access dictionary keys
    resp.encoding = "utf-8"

    try:
        # ✅ Best Practice: Provide default values for function parameters to improve usability and prevent errors.
        origin_result = resp.json().get("Result")
    except Exception as e:
        # ✅ Best Practice: Consider using a more descriptive class name for clarity and maintainability.
        # 🧠 ML Signal: Function calls with specific parameters can indicate usage patterns.
        logger.exception("code:{},content:{}".format(resp.status_code, resp.text))
        # ⚠️ SAST Risk (Low): Using dynamic URLs can lead to SSRF vulnerabilities if not properly validated.
        raise e
    # ✅ Best Practice: Class variables should be documented to explain their purpose and usage.

    if path_fields:
        # ✅ Best Practice: Initialize mutable class variables like lists or dictionaries in the constructor to avoid shared state across instances.
        # ✅ Best Practice: Method is defined but not implemented, indicating it's intended to be overridden in subclasses
        the_data = get_from_path_fields(origin_result, path_fields)
        if not the_data:
            # 🧠 ML Signal: Usage of a specific API wrapper class indicates a pattern of API interaction.
            # ✅ Best Practice: Explicitly raising NotImplementedError to indicate that this method should be implemented by subclasses
            logger.warning(
                # 🧠 ML Signal: Conditional logic based on the presence of timestamps
                # ✅ Best Practice: Consider dependency injection for easier testing and flexibility.
                "url:{},param:{},origin_result:{},could not get data for nested_fields:{}".format(
                    url,
                    param,
                    origin_result,
                    path_fields,
                    # ✅ Best Practice: Initialize lists outside of loops to avoid repeated allocations
                )
            )
        # 🧠 ML Signal: Usage of a method to generate request parameters
        return the_data

    return origin_result


# 🧠 ML Signal: API request pattern with specific parameters


def get_from_path_fields(the_json, path_fields):
    the_data = the_json.get(path_fields[0])
    # 🧠 ML Signal: Logging pattern with dynamic message content
    if the_data:
        for field in path_fields[1:]:
            the_data = the_data.get(field)
            if not the_data:
                # 🧠 ML Signal: Dynamic field assignment in a loop
                return None
    return the_data


# ✅ Best Practice: Use list.extend() for list concatenation


class EastmoneyApiWrapper(ApiWrapper):
    # ⚠️ SAST Risk (Low): Magic number used for list length check
    def request(self, url=None, method="post", param=None, path_fields=None):
        return call_eastmoney_api(
            url=url, method=method, param=param, path_fields=path_fields
        )


# 🧠 ML Signal: Class definition with multiple inheritance, indicating a pattern of combining functionalities.


# 🧠 ML Signal: Use of class attributes to define static configuration or metadata.
class BaseEastmoneyRecorder(object):
    # 🧠 ML Signal: Handling of cases where timestamps are not provided
    request_method = "post"
    # 🧠 ML Signal: Use of class attributes to define static configuration or metadata.
    path_fields = None
    # 🧠 ML Signal: API request pattern with specific parameters
    api_wrapper = EastmoneyApiWrapper()
    # 🧠 ML Signal: Use of class attributes to define static configuration or metadata.

    # ✅ Best Practice: Use of descriptive variable names improves code readability.
    def generate_request_param(self, security_item, start, end, size, timestamp):
        # 🧠 ML Signal: Use of class attributes to define static configuration or metadata.
        # 🧠 ML Signal: API call pattern with specific parameters.
        raise NotImplementedError

    def record(self, entity_item, start, end, size, timestamps):
        # 🧠 ML Signal: Use of class attributes to define static configuration or metadata.
        if timestamps:
            # 🧠 ML Signal: Use of class attributes to define static configuration or metadata.
            original_list = []
            # ✅ Best Practice: Checking for both existence and non-emptiness of a list.
            for the_timestamp in timestamps:
                param = self.generate_request_param(
                    entity_item, start, end, size, the_timestamp
                )
                # 🧠 ML Signal: Use of list comprehension for data transformation.
                tmp_list = self.api_wrapper.request(
                    # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
                    url=self.url,
                    param=param,
                    method=self.request_method,
                    path_fields=self.path_fields,
                    # 🧠 ML Signal: Conversion of data to a specific format (pandas timestamp).
                )
                # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
                self.logger.info(
                    # ✅ Best Practice: Returning an empty list when no data is available.
                    "record {} for entity_id:{},timestamp:{}".format(
                        self.data_schema, entity_item.id, the_timestamp
                    )
                    # 🧠 ML Signal: Method definition with parameters, useful for learning method usage patterns
                    # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
                )
                # fill timestamp field
                # 🧠 ML Signal: Dictionary creation with static and dynamic values, useful for learning data structure patterns
                # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
                for tmp in tmp_list:
                    tmp[self.get_evaluated_time_field()] = the_timestamp
                # 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
                # ⚠️ SAST Risk (Low): Potential risk if `call_eastmoney_api` is not handling input validation or sanitization
                # 🧠 ML Signal: Function signature and parameter usage can be used to understand method behavior and usage patterns.
                original_list += tmp_list
                # 🧠 ML Signal: API call pattern, useful for learning how APIs are used
                if len(original_list) == 50:
                    # 🧠 ML Signal: Conditional checks on remote_count can indicate decision-making patterns.
                    break
            return original_list

        else:
            # ✅ Best Practice: Use of descriptive variable names like local_count improves code readability.
            param = self.generate_request_param(entity_item, start, end, size, None)
            # ✅ Best Practice: Keyword arguments improve readability and maintainability.
            return self.api_wrapper.request(
                url=self.url,
                param=param,
                method=self.request_method,
                path_fields=self.path_fields,
                # ✅ Best Practice: Method name is descriptive and indicates its purpose
            )


# 🧠 ML Signal: Comparison between local_count and remote_count can indicate data synchronization logic.
# 🧠 ML Signal: Return values can be used to infer the function's purpose and output structure.
# ✅ Best Practice: Returning a dictionary is a clear and concise way to handle multiple return values


class EastmoneyTimestampsDataRecorder(BaseEastmoneyRecorder, TimestampsDataRecorder):
    entity_provider = "eastmoney"
    # 🧠 ML Signal: Usage of a function call within a dictionary value
    entity_schema = StockDetail

    provider = "eastmoney"
    # 🧠 ML Signal: Incrementing a parameter by a constant value
    # 🧠 ML Signal: Inheritance from multiple base classes, indicating a mixin or composite pattern

    # ✅ Best Practice: Class name should be descriptive of its purpose and functionality
    timestamps_fetching_url = None
    timestamp_list_path_fields = None
    # 🧠 ML Signal: Use of class-level attributes for configuration
    # 🧠 ML Signal: Function definition with parameters indicates a method that could be part of a class
    timestamp_path_fields = None

    # 🧠 ML Signal: Use of class-level attributes for configuration
    # 🧠 ML Signal: Dictionary creation with specific keys and values
    def init_timestamps(self, entity):
        param = {"color": "w", "fc": get_fc(entity)}
        # 🧠 ML Signal: Use of class-level attributes for configuration
        # ⚠️ SAST Risk (Medium): Potential risk if `call_eastmoney_api` is not properly handling input validation or output sanitization

        # 🧠 ML Signal: API call pattern with parameters
        timestamp_json_list = call_eastmoney_api(
            # 🧠 ML Signal: Tuple unpacking pattern
            # 🧠 ML Signal: Return statement indicating the end of a function
            # 🧠 ML Signal: Function name and parameters suggest a pattern for evaluating timestamps
            # 🧠 ML Signal: Usage of get_data function with specific parameters
            url=self.timestamps_fetching_url,
            path_fields=self.timestamp_list_path_fields,
            param=param,
        )

        if self.timestamp_path_fields and timestamp_json_list:
            timestamps = [
                get_from_path_fields(data, self.timestamp_path_fields)
                for data in timestamp_json_list
            ]
            return [to_pd_timestamp(t) for t in timestamps]
        return []


class EastmoneyPageabeDataRecorder(BaseEastmoneyRecorder, TimeSeriesDataRecorder):
    entity_provider = "eastmoney"
    entity_schema = StockDetail
    # 🧠 ML Signal: Conditional logic based on the presence of latest_record

    provider = "eastmoney"
    # 🧠 ML Signal: Pattern of comparing local and remote records

    page_url = None

    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    def get_remote_count(self, security_item):
        # 🧠 ML Signal: Usage of a dictionary to structure API request parameters.
        # ⚠️ SAST Risk (Low): Ensure that the 'size' parameter is validated to prevent excessive data requests.
        # 🧠 ML Signal: Explicitly defining __all__ to control module exports.
        param = {"color": "w", "fc": get_fc(security_item), "pageNum": 1, "pageSize": 1}
        return call_eastmoney_api(
            self.page_url, param=param, path_fields=["TotalCount"]
        )

    def evaluate_start_end_size_timestamps(self, entity):
        remote_count = self.get_remote_count(entity)

        if remote_count == 0:
            return None, None, 0, None

        # get local count
        local_count = get_data_count(
            data_schema=self.data_schema,
            session=self.session,
            filters=[self.data_schema.entity_id == entity.id],
        )
        # FIXME:the > case
        if local_count >= remote_count:
            return None, None, 0, None

        return None, None, remote_count - local_count, None

    def generate_request_param(self, security_item, start, end, size, timestamp):
        return {
            "color": "w",
            "fc": get_fc(security_item),
            "pageNum": 1,
            # just get more for some fixed data
            "pageSize": size + 10,
        }


class EastmoneyMoreDataRecorder(BaseEastmoneyRecorder, TimeSeriesDataRecorder):
    entity_provider = "eastmoney"
    entity_schema = StockDetail

    provider = "eastmoney"

    def get_remote_latest_record(self, security_item):
        param = {"color": "w", "fc": get_fc(security_item), "pageNum": 1, "pageSize": 1}
        results = call_eastmoney_api(
            self.url, param=param, path_fields=self.path_fields
        )
        _, result = self.generate_domain(security_item, results[0])
        return result

    def evaluate_start_end_size_timestamps(self, entity):
        # get latest record
        latest_record = get_data(
            entity_id=entity.id,
            provider=self.provider,
            data_schema=self.data_schema,
            order=self.data_schema.timestamp.desc(),
            limit=1,
            return_type="domain",
            session=self.session,
        )
        if latest_record:
            remote_record = self.get_remote_latest_record(entity)
            if not remote_record or (latest_record[0].id == remote_record.id):
                return None, None, 0, None
            else:
                return None, None, 10, None

        return None, None, 1000, None

    def generate_request_param(self, security_item, start, end, size, timestamp):
        return {
            "color": "w",
            "fc": get_fc(security_item),
            "pageNum": 1,
            "pageSize": size,
        }


# the __all__ is generated
__all__ = [
    "ApiWrapper",
    "get_fc",
    "get_company_type",
    "company_type_flag",
    "call_eastmoney_api",
    "get_from_path_fields",
    "EastmoneyApiWrapper",
    "BaseEastmoneyRecorder",
    "EastmoneyTimestampsDataRecorder",
    "EastmoneyPageabeDataRecorder",
    "EastmoneyMoreDataRecorder",
]
