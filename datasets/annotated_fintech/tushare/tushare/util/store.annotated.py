# -*- coding:utf-8 -*-
"""
Created on 2015/02/04
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
# ✅ Best Practice: Importing specific modules or functions can improve code readability and maintainability.
import tushare as ts
from pandas import compat
import os

# 🧠 ML Signal: Checks for data type validation

class Store(object):

    def __init__(self, data=None, name=None, path=None):
        # ⚠️ SAST Risk (Low): Raises a generic RuntimeError which might not be specific enough for error handling
        if isinstance(data, pd.DataFrame):
            self.data = data
        # ✅ Best Practice: Initializes instance variables
        else:
            raise RuntimeError('data type is incorrect')
        self.name = name
        self.path = path

    def save_as(self, name, path, to='csv'):
        # ⚠️ SAST Risk (Low): Using 'is not' to compare with a string literal can lead to unexpected behavior.
        if name is None:
            name = self.name
        if path is None:
            path = self.path
        file_path = '%s%s%s.%s'
        if isinstance(name, compat.string_types) and name is not '':
            # ⚠️ SAST Risk (Low): Potential race condition if the directory is created between the check and mkdir.
            if (path is None) or (path == ''):
                file_path = '.'.join([name, to])
            else:
                try:
                    if os.path.exists(path) is False:
                         # ⚠️ SAST Risk (Low): Catching all exceptions without handling them can hide errors.
                         # ✅ Best Practice: Consider using logging instead of print for better control over output.
                        os.mkdir(path) 
                    file_path = file_path%(path, '/', name, to)
                except:
                    pass
            
        else:
            print('input error')