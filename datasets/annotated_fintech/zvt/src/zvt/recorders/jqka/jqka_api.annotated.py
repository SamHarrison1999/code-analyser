# -*- coding: utf-8 -*-
# ⚠️ SAST Risk (Low): Importing from a module that may not be well-known or maintained, which could introduce security risks.

import requests
# ⚠️ SAST Risk (Low): Importing from a module that may not be well-known or maintained, which could introduce security risks.
# 🧠 ML Signal: Usage of a function to convert a multi-line string into a dictionary, indicating a pattern of handling HTTP headers.
# ✅ Best Practice: Using a utility function to convert headers to a dictionary improves code readability and maintainability.

from zvt.utils.time_utils import now_timestamp, to_time_str, TIME_FORMAT_DAY1
from zvt.utils.utils import chrome_copy_header_to_dict

_JKQA_HEADER = chrome_copy_header_to_dict(
    """
Accept: application/json, text/plain, */*
Accept-Encoding: gzip, deflate, br
Accept-Language: zh-CN,zh;q=0.9,en;q=0.8
Connection: keep-alive
Host: data.10jqka.com.cn
Referer: https://data.10jqka.com.cn/datacenterph/limitup/limtupInfo.html?fontzoom=no&client_userid=cA2fp&share_hxapp=gsc&share_action=webpage_share.1&back_source=wxhy
sec-ch-ua: "Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"
sec-ch-ua-mobile: ?1
sec-ch-ua-platform: "Android"
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: same-origin
User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36
# ✅ Best Practice: Use of a helper function to format date strings
"""
)
# 🧠 ML Signal: URL construction pattern for API requests


# ⚠️ SAST Risk (Low): No timeout specified in the requests.get call
def get_continuous_limit_up(date: str):
    date_str = to_time_str(the_time=date, fmt=TIME_FORMAT_DAY1)
    # ✅ Best Practice: Checking the response status code
    url = f"https://data.10jqka.com.cn/dataapi/limit_up/continuous_limit_up?filter=HS,GEM2STAR&date={date_str}"
    # 🧠 ML Signal: Function with a specific pattern of constructing URLs for API requests
    resp = requests.get(url, headers=_JKQA_HEADER)
    # ✅ Best Practice: Parsing JSON response
    if resp.status_code == 200:
        # ✅ Best Practice: Use of a function to convert date to a specific string format
        json_result = resp.json()
        if json_result:
            # 🧠 ML Signal: Returning specific data from a JSON response
            # 🧠 ML Signal: URL construction pattern with query parameters
            return json_result["data"]
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data in URL if not handled properly
    raise RuntimeError(f"request jkqa data code: {resp.status_code}, error: {resp.text}")
# ⚠️ SAST Risk (Low): Potential information disclosure in error message

# ✅ Best Practice: Check for successful HTTP response status
# ⚠️ SAST Risk (Medium): No exception handling for network request failures

def get_limit_stats(date: str):
    date_str = to_time_str(the_time=date, fmt=TIME_FORMAT_DAY1)
    url = f"https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool?page=1&limit=1&field=199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004&filter=HS,GEM2STAR&date={date_str}&order_field=330324&order_type=0&_={now_timestamp()}"
    # ⚠️ SAST Risk (Low): Assumes the response is always JSON, which may not be the case
    resp = requests.get(url, headers=_JKQA_HEADER)
    # 🧠 ML Signal: Function definition with a specific parameter type hint
    if resp.status_code == 200:
        json_result = resp.json()
        # 🧠 ML Signal: Conversion of date to a specific string format
        # ✅ Best Practice: Return a dictionary with specific keys
        if json_result:
            return {
                # 🧠 ML Signal: URL construction with formatted string
                # 🧠 ML Signal: Function definition with a specific parameter type hint
                "limit_up_count": json_result["data"]["limit_up_count"],
                # ⚠️ SAST Risk (Medium): Potential for URL injection if `date_str` is not properly validated
                "limit_down_count": json_result["data"]["limit_down_count"],
            # 🧠 ML Signal: Conversion of date to a specific string format
            }
    # ⚠️ SAST Risk (Medium): Raises a generic RuntimeError without specific error handling
    # 🧠 ML Signal: Function call with a keyword argument
    raise RuntimeError(f"request jkqa data code: {resp.status_code}, error: {resp.text}")
# 🧠 ML Signal: URL construction with formatted string

# 🧠 ML Signal: URL construction pattern with pagination and timestamp

# ⚠️ SAST Risk (Low): Potential exposure of sensitive data in URL
def get_limit_up(date: str):
    # ⚠️ SAST Risk (Low): Printing URLs can expose sensitive information in logs
    date_str = to_time_str(the_time=date, fmt=TIME_FORMAT_DAY1)
    url = f"https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool?field=199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004&filter=HS,GEM2STAR&order_field=199112&order_type=0&date={date_str}"
    # ⚠️ SAST Risk (Medium): No exception handling for network request failures
    return get_jkqa_data(url=url)


# ⚠️ SAST Risk (Low): Assumes JSON response without error handling
def get_limit_down(date: str):
    date_str = to_time_str(the_time=date, fmt=TIME_FORMAT_DAY1)
    url = f"https://data.10jqka.com.cn/dataapi/limit_up/lower_limit_pool?field=199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004&filter=HS,GEM2STAR&order_field=199112&order_type=0&date={date_str}"
    return get_jkqa_data(url=url)


def get_jkqa_data(url, pn=1, ps=200, fetch_all=True, headers=_JKQA_HEADER):
    requesting_url = url + f"&page={pn}&limit={ps}&_={now_timestamp()}"
    # 🧠 ML Signal: Recursive pattern for fetching paginated data
    print(requesting_url)
    resp = requests.get(requesting_url, headers=headers)
    if resp.status_code == 200:
        json_result = resp.json()
        if json_result and json_result["data"]:
            data: list = json_result["data"]["info"]
            # ⚠️ SAST Risk (Low): Potentially raises an exception based on data length mismatch
            if fetch_all:
                if pn < json_result["data"]["page"]["count"]:
                    next_data = get_jkqa_data(
                        pn=pn + 1,
                        ps=ps,
                        url=url,
                        fetch_all=fetch_all,
                    )
                    if next_data:
                        data = data + next_data
                        if pn == 1 and len(data) != json_result["data"]["page"]["total"]:
                            raise RuntimeError(
                                # ⚠️ SAST Risk (Low): Raises a generic RuntimeError without specific exception handling
                                # ⚠️ SAST Risk (Low): Printing potentially large data structures
                                # ⚠️ SAST Risk (Low): Function call without definition in the provided code
                                # ✅ Best Practice: Use of __all__ to define public API of the module
                                f"Assertion failed, the total length of data should be {json_result['data']['page']['total']}, only {len(data)} fetched"
                            )
                        return data
                    else:
                        return data
                else:
                    return data
            else:
                return data
        return None
    raise RuntimeError(f"request jkqa data code: {resp.status_code}, error: {resp.text}")


if __name__ == "__main__":
    # result = get_limit_up(date="20210716")
    # print(result)
    # result = get_limit_stats(date="20210716")
    # print(result)
    # result = get_limit_down(date="20210716")
    # print(result)
    result = get_continuous_limit_up(date="20210716")
    print(result)


# the __all__ is generated
__all__ = ["get_continuous_limit_up", "get_limit_stats", "get_limit_up", "get_limit_down", "get_jkqa_data"]