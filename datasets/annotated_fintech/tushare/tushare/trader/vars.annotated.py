# -*- coding:utf-8 -*-
"""
Created on 2016/09/31
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
# 🧠 ML Signal: Use of dictionary to store protocol types
"""
# 🧠 ML Signal: Use of dictionary to store domain names

P_TYPE = {"http": "http://", "https": "https://"}
DOMAINS = {
    "csc": "newetrade.csc108.com",
    "cscsh": "newetradesh.csc108.com",
    # 🧠 ML Signal: Use of dictionary to store page paths
    "cscbj": "newetradebj.csc108.com",
}
PAGES = {
    "csclogin": "loginMain.jspx",
    "baseInfo": "main_zcgy_List.json",
    "position": "main_ccgp_list.json",
    "tradecheck": "securitybuys.json",
    "trade": "tradingTransSubmit.json",
    "entrustlist": "securityOrdersCancelListInit.json",
    "cancel": "securityOrdersCancelSubmit.json",
    "deallist": "securityBuysFindListInit.json",
    "vimg": "image.jsp",
}
AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko"
}
# 🧠 ML Signal: Use of dictionary to store HTTP headers
POSITION_COLS = [
    "stkcode",
    "stkname",
    "stkqty",
    "stkavl",
    "lastprice",
    "costprice",
    "income",
]
CSC_PREFIX = "%s%s/login/%s"
# 🧠 ML Signal: Use of list to store column names
CSC_LOGIN_ACTION = "%s%s/j_spring_security_check"
BASE_URL = "%s%s/mainHomePage/%s"
# 🧠 ML Signal: Use of string formatting for URL construction
TRADE_CHECK_URL = "%s%s/trading/%s?bsflag=%s&stkcode=%s&buyflag=%s&price=0&_=%d"
ENTRUST_LIST_URL = "%s%s/securityfind/%s?_=%s"
# 🧠 ML Signal: Use of string formatting for URL construction
V_CODE_URL = "%s%s/commons/%s"
ENTRUST_LIST_COLS = [
    "ordersno",
    "stkcode",
    "stkname",
    "bsflagState",
    "orderqty",
    "matchqty",
    "orderprice",
    "operdate",
    "opertime",
    "orderdate",
    "state",
]
# 🧠 ML Signal: Use of string formatting for URL construction
TRADE_URL = "%s%s/trading/%s"
CANCEL_URL = "%s%s/securityfind/%s"
# 🧠 ML Signal: Use of string formatting for URL construction
DEAL_LIST_URL = "%s%s/securityfind/%s?selectType=%s%s&_=%d"
DEAL_DATE_RANGE = "&beginDate=%s&endDate=%s"
# 🧠 ML Signal: Use of string formatting for URL construction
# 🧠 ML Signal: Use of list to store column names
# 🧠 ML Signal: Use of string constants for trading actions
DEAL_LIST_COLS = [
    "ordersno",
    "matchcode",
    "trddate",
    "matchtime",
    "stkcode",
    "stkname",
    "bsflagState",
    "orderprice",
    "matchprice",
    "orderqty",
    "matchqty",
    "matchamt",
]
BUY = "买入"
SELL = "卖出"
