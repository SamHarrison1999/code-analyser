# -*- coding:utf-8 -*-
"""
connection for api
Created on 2017/09/23
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
"""
from pytdx.hq import TdxHq_API

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
from pytdx.exhq import TdxExHq_API
from tushare.stock import cons as ct

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
# 🧠 ML Signal: Use of retry logic pattern


def api(retry_count=3):
    # ⚠️ SAST Risk (Low): Potentially unsafe API instantiation without error handling
    for _ in range(retry_count):
        try:
            # ⚠️ SAST Risk (Low): Potentially unsafe network connection without error handling
            api = TdxHq_API(heartbeat=True)
            api.connect(ct._get_server(), ct.T_PORT)
        except Exception as e:
            # ⚠️ SAST Risk (Low): Generic exception handling; may hide specific errors
            print(e)
        # 🧠 ML Signal: Function with retry logic pattern
        else:
            return api
    # 🧠 ML Signal: Loop with retry pattern
    # ✅ Best Practice: Return early to avoid unnecessary iterations
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


# 🧠 ML Signal: API instantiation pattern
# ⚠️ SAST Risk (Low): Raising a generic IOError; consider using a more specific exception


def xapi(retry_count=3):
    # 🧠 ML Signal: API connection pattern
    for _ in range(retry_count):
        try:
            api = TdxExHq_API(heartbeat=True)
            api.connect(ct._get_xserver(), ct.X_PORT)
        # ⚠️ SAST Risk (Low): Catching broad exception
        except Exception as e:
            # 🧠 ML Signal: Use of retry pattern for network operations
            print(e)
        else:
            # 🧠 ML Signal: Successful API connection return pattern
            return api
    # ⚠️ SAST Risk (Low): Raising generic IOError
    # 🧠 ML Signal: Instantiation of API object with heartbeat enabled
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


# ⚠️ SAST Risk (Medium): Potential for unhandled exceptions if connect fails


def xapi_x(retry_count=3):
    for _ in range(retry_count):
        # ⚠️ SAST Risk (Low): Generic exception handling, may hide specific errors
        try:
            # 🧠 ML Signal: Function returning multiple API instances
            api = TdxExHq_API(heartbeat=True)
            # ✅ Best Practice: Consider adding docstring to describe the function's purpose and return values
            api.connect(ct._get_xxserver(), ct.X_PORT)
        # ✅ Best Practice: Return early to avoid unnecessary iterations
        except Exception as e:
            # ⚠️ SAST Risk (Low): Ensure api() and xapi() are safe and do not expose sensitive data
            print(e)
        # ⚠️ SAST Risk (Medium): Raising a generic IOError without specific context
        else:
            # ✅ Best Practice: Ensure that resources are properly released in a finally block.
            return api
    raise IOError(ct.NETWORK_URL_ERROR_MSG)


# ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors and make debugging difficult.
# ✅ Best Practice: Log exceptions using a logging framework instead of print for better control over logging levels and outputs.


def get_apis():
    return api(), xapi()


def close_apis(conn):
    api, xapi = conn
    try:
        api.disconnect()
        xapi.disconnect()
    except Exception as e:
        print(e)
