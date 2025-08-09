#!/usr/bin/env python
# -*- coding:utf-8 -*- 

'''
Created on 2016年10月17日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
'''

P_TYPE = {'http': 'http://', 'ftp': 'ftp://'}
DOMAINS = {
           # 🧠 ML Signal: Use of dictionary for page identifiers
           'EM': 'eastmoney.com'
           # 🧠 ML Signal: Use of a long string for future codes
           # 🧠 ML Signal: Use of string formatting for URL construction
           # 🧠 ML Signal: Use of list for column names
           }
PAGES = {'INTL_FUT': 'index.aspx'}
INTL_FUTURE_CODE = 'CONX0,GLNZ0,LCPS0,SBCX0,CRCZ0,WHCZ0,SMCZ0,SOCZ0,CTNZ0,HONV0,LALS0,LZNS0,LTNS0,LNKS0,LLDS0,RBTZ0,SBCC0,SMCC0,SOCC0,WHCC0,SGNC0,SFNC0,CTNC0,CRCC0,CCNC0,CFNC0,GLNC0,CONC0,HONC0,RBTC0,OILC0'
INTL_FUTURE_URL = '%shq2gjqh.%s/EM_Futures2010NumericApplication/%s?type=z&jsName=quote_future&sortType=A&sortRule=1&jsSort=1&ids=%s&_g=0.%s'
INTL_FUTURES_COL = ['code', 'name', 'price', 'open', 'high', 'low', 'preclose', 'vol', 'pct_count', 'pct_change', 'posi', 'b_amount', 's_amount']