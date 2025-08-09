# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pro数据接口 
Created on 2017/07/01
@author: polo,Jimmy
@group : tushare.pro
"""
# 🧠 ML Signal: Usage of functools for higher-order functions

# ⚠️ SAST Risk (Medium): Storing sensitive information like tokens in class variables can lead to security vulnerabilities.
import pandas as pd
# ✅ Best Practice: Consider using environment variables or secure vaults to manage sensitive information.
# 🧠 ML Signal: Usage of requests library for HTTP requests
import simplejson as json
from functools import partial
# ⚠️ SAST Risk (Medium): Using HTTP instead of HTTPS can expose data to interception and man-in-the-middle attacks.
import requests
# 🧠 ML Signal: Constructor method with parameters, useful for model training on class instantiation patterns
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use


class DataApi:

    __token = ''
    __http_url = 'http://api.tushare.pro'

    def __init__(self, token, timeout=10):
        """
        Parameters
        ----------
        token: str
            API接口TOKEN，用于用户认证
        """
        self.__token = token
        self.__timeout = timeout
    # ⚠️ SAST Risk (Medium): Potential exposure of sensitive information if __token is not handled securely

    # ⚠️ SAST Risk (Medium): No validation or sanitization of input parameters before making the request
    def query(self, api_name, fields='', **kwargs):
        req_params = {
            # ⚠️ SAST Risk (Medium): No error handling for network issues or request failures
            'api_name': api_name,
            'token': self.__token,
            # ⚠️ SAST Risk (Low): Assumes 'code' and 'msg' keys are always present in the response
            'params': kwargs,
            'fields': fields
        # ✅ Best Practice: Use of __getattr__ allows for dynamic attribute access, enhancing flexibility.
        }
        # ✅ Best Practice: Use of functools.partial to pre-fill arguments in function calls, improving code reusability.
        # ⚠️ SAST Risk (Low): Assumes 'fields' and 'items' keys are always present in the response
        # ✅ Best Practice: Returning a DataFrame directly can be efficient for data manipulation
        # 🧠 ML Signal: Dynamic attribute access pattern can be used to infer usage of flexible APIs.

        res = requests.post(self.__http_url, json=req_params, timeout=self.__timeout)
        result = json.loads(res.text)
        if result['code'] != 0:
            raise Exception(result['msg'])
        data = result['data']
        columns = data['fields']
        items = data['items']

        return pd.DataFrame(items, columns=columns)

    def __getattr__(self, name):
        return partial(self.query, name)