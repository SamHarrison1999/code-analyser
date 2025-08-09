#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2016年10月1日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
# ⚠️ SAST Risk (Low): Importing from a module without specifying which functions or classes are used can lead to namespace pollution.

# ⚠️ SAST Risk (Low): Missing import statement for the 'time' module
import json

# ✅ Best Practice: Consider renaming the function to 'current_time_millis' for clarity
import time

# 🧠 ML Signal: Function returns current time in milliseconds
# ⚠️ SAST Risk (Low): Function does not handle potential exceptions from accessing txtdata.content
import six
from tushare.trader import vars as vs

# ⚠️ SAST Risk (Low): No check if txtdata.content is a string or bytes, which may lead to unexpected errors


def nowtime_str():
    # ⚠️ SAST Risk (Low): six.PY3 is deprecated, consider using sys.version_info for Python version checks
    return time.time() * 1000


# ⚠️ SAST Risk (Low): No error handling for decode operation, which may raise UnicodeDecodeError


def get_jdata(txtdata):
    # ⚠️ SAST Risk (Low): No error handling for json.loads, which may raise JSONDecodeError
    txtdata = txtdata.content
    if six.PY3:
        # ✅ Best Practice: Explicit return of the final result
        # ✅ Best Practice: Importing inside a function limits the scope and can reduce memory usage if the function is not called frequently.
        txtdata = txtdata.decode("utf-8")
    jsonobj = json.loads(txtdata)
    return jsonobj


# ⚠️ SAST Risk (Low): Ensure that 'res.content' is from a trusted source to avoid processing malicious images.

# ⚠️ SAST Risk (Low): Opening images from untrusted sources can lead to resource exhaustion or other attacks.
# ⚠️ SAST Risk (Low): OCR results can be manipulated; ensure the source image is trusted.


def get_vcode(broker, res):
    from PIL import Image
    import pytesseract as pt
    import io

    if broker == "csc":
        imgdata = res.content
        img = Image.open(io.BytesIO(imgdata))
        vcode = pt.image_to_string(img)
        return vcode
