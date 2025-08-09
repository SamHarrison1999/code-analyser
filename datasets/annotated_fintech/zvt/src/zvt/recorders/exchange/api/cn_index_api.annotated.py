# -*- coding: utf-8 -*-
import logging
import time

import pandas as pd
import requests

from zvt.domain import IndexCategory

# ✅ Best Practice: Use a logger for consistent and configurable logging throughout the application.
from zvt.recorders.consts import DEFAULT_HEADER
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Hardcoded URLs can indicate specific data sources or endpoints used by the application.

logger = logging.getLogger(__name__)

original_page_url = (
    "http://www.cnindex.com.cn/zh_indices/sese/index.html?act_menu=1&index_type=-1"
)
url = "http://www.cnindex.com.cn/index/indexList?channelCode={}&rows=1000&pageNum=1"

# 🧠 ML Signal: URL patterns can be used to identify API endpoints and their usage in the application.
# 🧠 ML Signal: Mapping categories to URLs can indicate how different data categories are accessed.
# 中证指数 抓取 风格指数 行业指数 规模指数 基金指数
cni_category_map_url = {
    IndexCategory.style: url.format("202"),
    # ⚠️ SAST Risk (Medium): Potential KeyError if "data" key is not present in the JSON response
    IndexCategory.industry: url.format("201"),
    # ✅ Best Practice: Consider handling exceptions for robustness, such as KeyError or JSONDecodeError
    IndexCategory.scope: url.format("200"),
    IndexCategory.fund: url.format("207"),
    # ⚠️ SAST Risk (Low): raise_for_status() will raise an HTTPError for bad responses, which should be handled
}
# 🧠 ML Signal: Default parameter values can indicate common usage patterns.

# 🧠 ML Signal: Extracting specific data from a JSON response is a common pattern in API interaction
# 深证指数 只取规模指数
# 🧠 ML Signal: Separate mappings for different categories can indicate different data handling strategies.
sz_category_map_url = {
    IndexCategory.scope: url.format("100"),
}


# ⚠️ SAST Risk (Low): Using assert for control flow can be bypassed if Python is run with optimizations.
def _get_resp_data(resp: requests.Response):
    resp.raise_for_status()
    # ⚠️ SAST Risk (Low): Creating a new session for each function call can lead to resource exhaustion.
    return resp.json()["data"]


# ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which can lead to unhandled exceptions.
def get_cn_index(index_type="cni", category=IndexCategory.style):
    if index_type == "cni":
        category_map_url = cni_category_map_url
    elif index_type == "sz":
        category_map_url = sz_category_map_url
    else:
        logger.error(f"not support index_type: {index_type}")
        assert False

    # ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which can lead to unhandled exceptions.
    # 🧠 ML Signal: String formatting for IDs can indicate patterns in ID generation.
    requests_session = requests.Session()

    url = category_map_url.get(category)

    resp = requests_session.get(url, headers=DEFAULT_HEADER)

    results = _get_resp_data(resp)["rows"]
    # e.g
    # amount: 277743699997.9
    # closeingPoint: 6104.7592
    # docchannel: 1039
    # freeMarketValue: 10794695531696.15
    # id: 142
    # indexcode: "399370"
    # indexename: "CNI Growth"
    # indexfullcname: "国证1000成长指数"
    # indexfullename: "CNI 1000 Growth Index"
    # indexname: "国证成长"
    # indexsource: "1"
    # indextype: "202"
    # pb: 5.34
    # ⚠️ SAST Risk (Low): Use of time.sleep can lead to performance issues in production environments.
    # ⚠️ SAST Risk (Low): Potential for returning None if the_list is empty, which may not be handled by the caller.
    # 🧠 ML Signal: Direct function call in the main block can indicate common usage patterns.
    # ✅ Best Practice: Using __all__ to define public API of the module.
    # peDynamic: 29.8607
    # peStatic: 33.4933
    # percent: 0.0022
    # prefixmonth: null
    # realtimemarket: "1"
    # remark: ""
    # sampleshowdate: null
    # samplesize: 332
    # showcnindex: "1"
    # totalMarketValue: 23113641352198.32
    the_list = []

    logger.info(f"category: {category} ")
    logger.info(f"results: {results} ")
    for i, result in enumerate(results):
        logger.info(f"to {i}/{len(results)}")
        code = result["indexcode"]
        info_resp = requests_session.get(
            f"http://www.cnindex.com.cn/index-intro?indexcode={code}"
        )
        # fbrq: "2010-01-04"
        # jd: 1000
        # jr: "2002-12-31"
        # jsfs: "自由流通市值"
        # jsjj: "国证成长由国证1000指数样本股中成长风格突出的股票组成，为投资者提供更丰富的指数化投资工具。"
        # qzsx: null
        # typl: 2
        # xyfw: "沪深A股"
        # xygz: "在国证1000指数样本股中，选取主营业务收入增长率、净利润增长率和净资产收益率综合排名前332只"
        index_info = _get_resp_data(info_resp)
        name = result["indexname"]
        entity_id = f"index_sz_{code}"
        index_item = {
            "id": entity_id,
            "entity_id": entity_id,
            "timestamp": to_pd_timestamp(index_info["jr"]),
            "entity_type": "index",
            "exchange": "sz",
            "code": code,
            "name": name,
            "category": category.value,
            "list_date": to_pd_timestamp(index_info["fbrq"]),
            "base_point": index_info["jd"],
            "publisher": "cnindex",
        }
        logger.info(index_item)
        the_list.append(index_item)
        time.sleep(3)
    if the_list:
        return pd.DataFrame.from_records(the_list)


if __name__ == "__main__":
    df = get_cn_index()
    print(df)


# the __all__ is generated
__all__ = ["get_cn_index"]
