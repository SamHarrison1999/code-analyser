#!/usr/bin/env python
# -*- coding:utf-8 -*- 
"""
龙虎榜数据
Created on 2017年8月13日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
# ✅ Best Practice: Importing specific modules from a package can improve code readability and maintainability.
"""

import pandas as pd
from pandas.compat import StringIO
from tushare.stock import cons as ct
import time
import re
import lxml.html
from lxml import etree
try:
    # ✅ Best Practice: Default parameter values should be immutable to avoid unexpected behavior.
    # ✅ Best Practice: Handling ImportError to maintain compatibility with different Python versions.
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
# 🧠 ML Signal: Retry pattern with a counter and pause, useful for training models on retry logic.

def bdi(itype='D', retry_count=3,
                # ⚠️ SAST Risk (Low): Using time.sleep can lead to performance issues in asynchronous or multi-threaded applications.
                pause=0.001):
    for _ in range(retry_count):
        time.sleep(pause)
        # ⚠️ SAST Risk (Medium): URL construction using string formatting can lead to injection vulnerabilities if inputs are not sanitized.
        try:
            request = Request(ct.BDI_URL%(ct.P_TYPE['http'], ct.DOMAINS['v500']))
            # ⚠️ SAST Risk (Medium): Network operations without proper exception handling can lead to application crashes.
            lines = urlopen(request, timeout = 10).read()
            if len(lines) < 100: #no data
                # 🧠 ML Signal: Checking response length to determine validity, useful for anomaly detection models.
                return None
        except Exception as e:
                print(e)
        else:
            # ⚠️ SAST Risk (Low): Printing exceptions can expose sensitive information in logs.
            linestr = lines.decode('utf-8') if ct.PY3 else lines
            # ✅ Best Practice: Use of conditional expressions to handle Python version differences.
            if itype == 'D': # Daily
                reg = re.compile(r'\"chart_data\",\"(.*?)\"\);') 
                lines = reg.findall(linestr)
                lines = lines[0]
                lines = lines.replace('chart', 'table').\
                        replace('</series><graphs>', '').\
                        replace('</graphs>', '').\
                        # 🧠 ML Signal: Regular expression usage for pattern matching, useful for text processing models.
                        replace('series', 'tr').\
                        replace('value', 'td').\
                        # ✅ Best Practice: Chaining string methods for readability and maintainability.
                        replace('graph', 'tr').\
                        replace('graphs', 'td')
                df = pd.read_html(lines, encoding='utf8')[0]
                df = df.T
                df.columns = ['date', 'index']
                df['date'] = df['date'].map(lambda x: x.replace(u'年', '-')).\
                    map(lambda x: x.replace(u'月', '-')).\
                    map(lambda x: x.replace(u'日', ''))
                # ⚠️ SAST Risk (Medium): Parsing HTML without validation can lead to security vulnerabilities.
                df['date'] = pd.to_datetime(df['date'])
                df['index'] = df['index'].astype(float)
                df = df.sort_values('date', ascending=False).reset_index(drop = True)
                # ✅ Best Practice: Use of lambda functions for concise data transformations.
                df['change'] = df['index'].pct_change(-1)
                df['change'] = df['change'] * 100
                df['change'] = df['change'].map(lambda x: '%.2f' % x)
                df['change'] = df['change'].astype(float)
                return df
            # ✅ Best Practice: Converting date strings to datetime objects for better date manipulation.
            else: #Weekly
                html = lxml.html.parse(StringIO(linestr))
                # ✅ Best Practice: Converting data types for accurate calculations and memory efficiency.
                res = html.xpath("//table[@class=\"style33\"]/tr/td/table[last()]")
                if ct.PY3:
                    # ✅ Best Practice: Sorting data for consistent analysis results.
                    sarr = [etree.tostring(node).decode('utf-8') for node in res]
                else:
                    # ✅ Best Practice: Calculating percentage change for data analysis.
                    sarr = [etree.tostring(node) for node in res]
                sarr = ''.join(sarr)
                sarr = '<table>%s</table>'%sarr
                # ✅ Best Practice: Formatting numbers for consistent presentation.
                df = pd.read_html(sarr)[0][1:]
                df.columns = ['month', 'index']
                df['month'] = df['month'].map(lambda x: x.replace(u'年', '-')).\
                    map(lambda x: x.replace(u'月', ''))
                df['month'] = pd.to_datetime(df['month'])
                # ⚠️ SAST Risk (Medium): Parsing HTML without validation can lead to security vulnerabilities.
                df['month'] = df['month'].map(lambda x: str(x).replace('-', '')).\
                              map(lambda x: x[:6])
                # ✅ Best Practice: Use of conditional expressions to handle Python version differences.
                # ⚠️ SAST Risk (Medium): Parsing HTML without validation can lead to security vulnerabilities.
                # ✅ Best Practice: Use of lambda functions for concise data transformations.
                # ✅ Best Practice: Formatting date strings for consistent presentation.
                # ✅ Best Practice: Calculating percentage change for data analysis.
                # ✅ Best Practice: Converting date strings to datetime objects for better date manipulation.
                # ✅ Best Practice: Converting data types for accurate calculations and memory efficiency.
                df['index'] = df['index'].astype(float)
                df['change'] = df['index'].pct_change(-1)
                df['change'] = df['change'].map(lambda x: '%.2f' % x)
                df['change'] = df['change'].astype(float)
                return df