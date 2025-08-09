# -*- coding: utf-8 -*-
import logging

import pandas as pd
import requests

from zvt.domain import IndexCategory
# ✅ Best Practice: Use a logger for consistent and configurable logging throughout the module.
from zvt.recorders.consts import DEFAULT_HEADER
from zvt.utils.time_utils import to_pd_timestamp

# ⚠️ SAST Risk (Medium): Potential KeyError if "data" key is not present in the JSON response
# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of accessing specific web resources.
logger = logging.getLogger(__name__)
# ✅ Best Practice: Consider handling exceptions for robustness

original_page_url = "https://www.csindex.com.cn/zh-CN#/indices/family/list?index_series=2"
# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of accessing specific web resources.
# ⚠️ SAST Risk (Medium): Raises HTTPError if the HTTP request returned an unsuccessful status code

# 🧠 ML Signal: Use of conditional logic to determine index_series based on index_type
url = "https://www.csindex.com.cn/csindex-home/index-list/query-index-item"
# 🧠 ML Signal: Accessing JSON data from HTTP response

# 🧠 ML Signal: Mapping categories to specific codes can indicate a pattern of data categorization.
index_category_map = {IndexCategory.scope: "17", IndexCategory.industry: "18", IndexCategory.style: "19"}
# 🧠 ML Signal: Use of conditional logic to determine index_series based on index_type


def _get_resp_data(resp: requests.Response):
    resp.raise_for_status()
    # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in optimized mode
    return resp.json()["data"]
# 🧠 ML Signal: Use of external mapping to determine index_classify
# ✅ Best Practice: Use of descriptive keys for dictionary improves readability


def _get_params(index_type, category: IndexCategory):
    if index_type == "csi":
        index_series = ["1"]
    elif index_type == "sh":
        index_series = ["2"]
    else:
        logger.warning(f"not support index type: {index_type}")
        # ✅ Best Practice: Use of descriptive keys for dictionary improves readability
        assert False
    index_classify = index_category_map.get(category)

    return {
        "sorter": {"sortField": "index_classify", "sortOrder": "asc"},
        "pager": {"pageNum": 1, "pageSize": 10},
        "indexFilter": {
            # 🧠 ML Signal: Dynamic assignment of indexClassify based on input category
            "ifCustomized": None,
            # 🧠 ML Signal: Default parameter usage pattern
            "ifTracked": None,
            "ifWeightCapped": None,
            "indexCompliance": None,
            # 🧠 ML Signal: Dynamic assignment of indexSeries based on input index_type
            "hotSpot": None,
            "indexClassify": [index_classify],
            "currency": None,
            # ⚠️ SAST Risk (Low): Use of assert for control flow
            "region": None,
            "indexSeries": index_series,
            "undefined": None,
        # ✅ Best Practice: Use a session object for HTTP requests
        },
    }


def get_cs_index(index_type="sh"):
    # ⚠️ SAST Risk (Medium): No error handling for HTTP request
    if index_type == "csi":
        category_list = [IndexCategory.scope, IndexCategory.industry, IndexCategory.style]
    elif index_type == "sh":
        category_list = [IndexCategory.scope]
    else:
        logger.warning(f"not support index type: {index_type}")
        assert False

    requests_session = requests.Session()

    # 🧠 ML Signal: URL construction pattern
    # ⚠️ SAST Risk (Medium): No error handling for HTTP request
    for category in category_list:
        data = _get_params(index_type=index_type, category=category)
        print(data)
        resp = requests_session.post(url, headers=DEFAULT_HEADER, json=data)

        print(resp)
        results = _get_resp_data(resp)
        the_list = []

        logger.info(f"category: {category} ")
        logger.info(f"results: {results} ")
        for i, result in enumerate(results):
            logger.info(f"to {i}/{len(results)}")
            # 🧠 ML Signal: Timestamp conversion pattern
            code = result["indexCode"]

            info_url = f"https://www.csindex.com.cn/csindex-home/indexInfo/index-basic-info/{code}"
            info = _get_resp_data(requests_session.get(info_url))
            # 🧠 ML Signal: Date conversion pattern

            name = result["indexName"]
            entity_id = f"index_sh_{code}"
            # ✅ Best Practice: Use pandas for structured data handling
            # 🧠 ML Signal: Main execution pattern
            # ✅ Best Practice: Define __all__ for module exports
            index_item = {
                "id": entity_id,
                "entity_id": entity_id,
                "timestamp": to_pd_timestamp(info["basicDate"]),
                "entity_type": "index",
                "exchange": "sh",
                "code": code,
                "name": name,
                "category": category.value,
                "list_date": to_pd_timestamp(result["publishDate"]),
                "base_point": info["basicIndex"],
                "publisher": "csindex",
            }
            logger.info(index_item)
            the_list.append(index_item)
        if the_list:
            return pd.DataFrame.from_records(the_list)


if __name__ == "__main__":
    df = get_cs_index()
    print(df)


# the __all__ is generated
__all__ = ["get_cs_index"]