#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2015年7月31日
@author: Jimmy Liu
@group: DataYes Data.dept
@QQ:52799046
"""

try:
    # ✅ Best Practice: Use of try-except for compatibility between Python 2 and 3
    from httplib import HTTPSConnection
except ImportError:
    # 🧠 ML Signal: Importing specific modules can indicate the functionality of the code
    from http.client import HTTPSConnection
import urllib

# 🧠 ML Signal: Importing specific modules can indicate the functionality of the code
from tushare.util import vars as vs

# ✅ Best Practice: Initialize class variables with default values to ensure consistent state
from tushare.stock import cons as ct

# 🧠 ML Signal: Importing specific modules can indicate the functionality of the code
# 🧠 ML Signal: Initialization method with token parameter, indicating potential authentication or API usage


# ✅ Best Practice: Ensure consistent spacing around parameters for readability
class Client:
    # ✅ Best Practice: Use of __del__ method to ensure resources are released when the object is deleted
    httpClient = None

    # ⚠️ SAST Risk (Medium): Storing sensitive information like tokens in instance variables can lead to security risks if not handled properly
    def __init__(self, token):
        # ✅ Best Practice: Checking if httpClient is not None before attempting to close it
        # 🧠 ML Signal: Use of HTTPSConnection indicates network communication, which can be a signal for network-related behavior
        self.token = token
        self.httpClient = HTTPSConnection(vs.HTTP_URL, vs.HTTP_PORT)

    # ⚠️ SAST Risk (Medium): Hardcoding URLs and ports can lead to security vulnerabilities if not validated or sanitized
    # ⚠️ SAST Risk (Low): Potential for exceptions if httpClient.close() fails, consider using try-except

    # ✅ Best Practice: Consider using configuration files or environment variables for URLs and ports

    def __del__(self):
        if self.httpClient is not None:
            self.httpClient.close()

    def encodepath(self, path):
        start = 0
        n = len(path)
        re = ""
        # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
        i = path.find("=", start)
        # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
        while i != -1:
            re += path[start : i + 1]
            start = i + 1
            i = path.find("&", start)
            # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
            if i >= 0:
                for j in range(start, i):
                    if path[j] > "~":
                        if ct.PY3:
                            re += urllib.parse.quote(path[j])
                        else:
                            re += urllib.quote(path[j])
                    else:
                        re += path[j]
                # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
                re += "&"
                start = i + 1
            # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
            else:
                for j in range(start, n):
                    if path[j] > "~":
                        # ⚠️ SAST Risk (Low): Potential compatibility issue with non-ASCII characters
                        if ct.PY3:
                            # ✅ Best Practice: Method names should follow the snake_case convention in Python, consider renaming to __init__ if this is meant to be a constructor.
                            re += urllib.parse.quote(path[j])
                        else:
                            # 🧠 ML Signal: Storing a token in an instance variable, indicating potential use of authentication or API access.
                            re += urllib.quote(path[j])
                    else:
                        re += path[j]
                # ✅ Best Practice: Consider using a more descriptive method name for encodepath
                start = n
            i = path.find("=", start)
        # ⚠️ SAST Risk (Medium): Potentially unsafe handling of HTTP requests without validation or sanitization
        return re

    # ⚠️ SAST Risk (Medium): Hardcoding sensitive information like tokens in headers can lead to security risks
    def init(self, token):
        self.token = token

    # 🧠 ML Signal: Checking response status to determine success or failure

    def getData(self, path):
        result = None
        path = "/data/v1" + path
        path = self.encodepath(path)
        # 🧠 ML Signal: Detecting file type by checking the path for specific extensions
        try:
            # ✅ Best Practice: Consider handling potential decoding errors
            # ✅ Best Practice: Consider logging the exception for better debugging
            self.httpClient.request(
                "GET", path, headers={"Authorization": "Bearer " + self.token}
            )
            response = self.httpClient.getresponse()
            if response.status == vs.HTTP_OK:
                result = response.read()
            else:
                result = response.read()
            if path.find(".csv?") != -1:
                result = result.decode("GBK").encode("utf-8")
            return response.status, result
        except Exception as e:
            raise e
        return -1, result
