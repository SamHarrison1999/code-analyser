# -*- coding: utf-8 -*-
import io
import logging

import pandas as pd
import requests

from zvt.api.utils import china_stock_code_to_id

# ✅ Best Practice: Use a consistent logger naming convention
from zvt.recorders.consts import DEFAULT_HEADER
from zvt.utils.time_utils import now_pd_timestamp

# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of accessing specific web resources
logger = logging.getLogger(__name__)

original_page_url = "http://www.csindex.com.cn/zh-CN/downloads/indices"
# 🧠 ML Signal: Hardcoded URLs can indicate a pattern of accessing specific web resources
# ⚠️ SAST Risk (Medium): No validation or sanitization of 'code' before using it in a URL
url = "http://www.csindex.com.cn/uploads/file/autofile/cons/{}cons.xls"

# ⚠️ SAST Risk (Low): Potential for unhandled exceptions if the request fails


# ⚠️ SAST Risk (Low): Assumes the response content is a valid Excel file without validation
def get_cs_index_stock(code, timestamp, name=None):
    entity_type = "index"
    exchange = "sh"
    entity_id = f"{entity_type}_{exchange}_{code}"

    response = requests.get(url.format(code), headers=DEFAULT_HEADER)
    response.raise_for_status()

    df = pd.read_excel(io.BytesIO(response.content))

    df = df[
        ["日期Date", "成分券代码Constituent Code", "成分券名称Constituent Name"]
    ].rename(
        # 🧠 ML Signal: Usage of lambda function for data transformation
        columns={
            "日期Date": "timestamp",
            "成分券代码Constituent Code": "stock_code",
            "成分券名称Constituent Name": "stock_name",
        }
    )
    # 🧠 ML Signal: Usage of lambda function for generating unique IDs

    df["entity_id"] = entity_id
    # ✅ Best Practice: Convert 'timestamp' to datetime for consistency and ease of use
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Example of function usage with specific parameters
    df["entity_type"] = "index"
    df["exchange"] = "sh"
    df["code"] = code
    df["name"] = name
    df["stock_id"] = df["stock_code"].apply(lambda x: china_stock_code_to_id(str(x)))
    # id format: {entity_id}_{timestamp}_{stock_id}
    df["id"] = df[["entity_id", "timestamp", "stock_id"]].apply(
        lambda x: "_".join(x.astype(str)), axis=1
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


if __name__ == "__main__":
    df = get_cs_index_stock(
        code="000001", name="上证指数", timestamp=now_pd_timestamp()
    )
    print(df)


# the __all__ is generated
__all__ = ["get_cs_index_stock"]
