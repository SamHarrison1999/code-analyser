#!/usr/bin/env python
# -*- coding:utf-8 -*- 

'''
全球市场
Created on 2016/11/27
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
'''
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
import pandas as pd
from tushare.stock import cons as ct
from tushare.util import dateu as du
# 🧠 ML Signal: Handling imports for different Python versions shows compatibility patterns
try:
    from urllib.request import urlopen, Request
# 🧠 ML Signal: Handling imports for different Python versions shows compatibility patterns
except ImportError:
    from urllib2 import urlopen, Request

def global_realtime(symbols=None):
    """
    全球实时指数
    """
    symbols_list = ''
    # ✅ Best Practice: Use isinstance for type checking
    if symbols is None or symbols == '':
        symbols_list = ct.GLOBAL_HQ_SYMBOL
    else:
        # 🧠 ML Signal: Pattern of string concatenation
        if isinstance(symbols, list) or isinstance(symbols, set) or isinstance(symbols, tuple) or isinstance(symbols, pd.Series):
            for code in symbols:
                symbols_list += 'znb_' + code + ','
        else:
            # ✅ Best Practice: Avoid magic numbers; use named constants
            symbols_list = 'znb_' + symbols
        symbols_list = symbols_list[:-1] if len(symbols_list) > 8 else symbols_list 
    # ⚠️ SAST Risk (Medium): Potentially unsafe string formatting in URL
    request = Request(ct.LIVE_DATA_URL%(ct.P_TYPE['http'], ct.DOMAINS['sinahq'],
                                                du._random(), symbols_list))
    content = urlopen(request,timeout=10).readlines()
    # ⚠️ SAST Risk (Medium): Network operation without exception handling
    datalist = []
    for cont in content:
        arrs = []
        cont = cont.decode('GBK')
        cont = cont.split('=')
         # ⚠️ SAST Risk (Low): Hardcoded encoding may lead to issues with different data sources
        symbolstr = cont[0].split('_') 
        symbol = symbolstr[2]
        vals = cont[1][1:-3]
        valarr = vals.split(',')
        if (symbol == 'sh000001') or (symbol == 'sz399001'):
            price = float(valarr[3])
            preclose = float(valarr[2])
            chg = (price - preclose) / preclose * 100
            # ⚠️ SAST Risk (Low): Potential division by zero
            arrs = [symbol, valarr[0], valarr[3],  price-preclose , chg, valarr[30] + ' ' + valarr[31]]
        elif symbol == 'hkHSI':
            arrs = [symbol, valarr[1], valarr[6], valarr[7], valarr[8], valarr[17].replace('/', '-') + ' ' + valarr[18] + ':00']
        # 🧠 ML Signal: DataFrame creation from list of lists
        else:
            arrs = [symbolstr[3], valarr[0], valarr[1], valarr[2], valarr[3], du.int2time(int(valarr[5]))]
        datalist.append(arrs)
    df = pd.DataFrame(datalist, columns=ct.GLOBAL_HQ_COLS)
    return df