#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2016年9月25日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import six
import pandas as pd
import requests

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
import time
from threading import Thread

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
from tushare.trader import vars as vs
from tushare.trader import utils

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
# 🧠 ML Signal: Importing specific functions from a module indicates specific usage patterns
from tushare.util import upass as up
from tushare.util.upass import set_broker


class TraderAPI(object):
    """
    股票实盘交易接口
    提醒：本文涉及的思路和内容仅限于量化投资及程序化交易的研究与尝试，不作为个人或机构常规程序化交易的依据，
    不对实盘的交易风险和政策风险产生的影响负责，如有问题请与我联系。
    投资有风险，下单须谨慎。
    """

    # 🧠 ML Signal: Usage of formatted strings for constructing URLs.
    def __init__(self, broker=""):
        if broker == "":
            return None
        self.broker = broker
        self.trade_prefix = vs.CSC_PREFIX % (
            vs.P_TYPE["https"],
            vs.DOMAINS["csc"],
            # 🧠 ML Signal: Instantiation of a requests session, common in network communication.
            vs.PAGES["csclogin"],
        )
        # ⚠️ SAST Risk (Low): Compatibility check for Python 2, which is outdated and may have security implications.
        self.heart_active = True
        self.s = requests.session()
        # 🧠 ML Signal: Usage of threading for background tasks.
        if six.PY2:
            self.heart_thread = Thread(target=self.send_heartbeat)
            self.heart_thread.setDaemon(True)
        # 🧠 ML Signal: Usage of headers update pattern
        # ✅ Best Practice: Use setDaemon method consistently across Python versions.
        else:
            # 🧠 ML Signal: Pattern of constructing URLs with string formatting
            self.heart_thread = Thread(target=self.send_heartbeat, daemon=True)

    # 🧠 ML Signal: Usage of threading for background tasks.

    def login(self):
        # 🧠 ML Signal: Pattern of constructing URLs with string formatting
        self.s.headers.update(vs.AGENT)
        self.s.get(
            vs.CSC_PREFIX
            % (vs.P_TYPE["https"], vs.DOMAINS["csc"], vs.PAGES["csclogin"])
        )
        # ⚠️ SAST Risk (Low): Potential issue if utils.get_vcode or _login are not handling exceptions properly
        res = self.s.get(
            vs.V_CODE_URL
            % (
                vs.P_TYPE["https"],
                # 🧠 ML Signal: Use of a method with a name starting with an underscore, indicating a private method.
                vs.DOMAINS["cscsh"],
                # ✅ Best Practice: Consider using logging instead of print for better control over output
                vs.PAGES["vimg"],
            )
        )
        # ✅ Best Practice: Use of a dictionary to organize login parameters for clarity and maintainability.
        # 🧠 ML Signal: Keep-alive pattern for maintaining session
        # 🧠 ML Signal: Accessing dictionary values using keys, indicating a pattern of data retrieval.
        if self._login(utils.get_vcode("csc", res)) is False:
            print("请确认账号或密码是否正确 ，或券商服务器是否处于维护中。 ")
        self.keepalive()

    def _login(self, v_code):
        brokerinfo = up.get_broker(self.broker)
        user = brokerinfo["user"][0]
        login_params = dict(
            inputid=user,
            j_username=user,
            # ⚠️ SAST Risk (High): Storing and using passwords directly from a dictionary can lead to security vulnerabilities.
            j_inputid=user,
            AppendCode=v_code,
            isCheckAppendCode="false",
            logined="false",
            # ⚠️ SAST Risk (Medium): Sending sensitive information like passwords over a POST request without ensuring encryption.
            f_tdx="",
            j_cpu="",
            j_password=brokerinfo["passwd"][0],
            # 🧠 ML Signal: Checks if a thread is alive, indicating a pattern of monitoring thread status
        )
        # 🧠 ML Signal: Checking for specific text in a response to determine login success, indicating a pattern for success criteria.
        # 🧠 ML Signal: Sets a flag based on thread status, indicating a pattern of state management
        logined = self.s.post(
            vs.CSC_LOGIN_ACTION % (vs.P_TYPE["https"], vs.DOMAINS["csc"]),
            params=login_params,
        )
        if logined.text.find("消息中心") != -1:
            # ⚠️ SAST Risk (Low): Potential race condition if `start` is called on an already started thread
            # 🧠 ML Signal: Infinite loop pattern for continuous operation
            return True
        # ✅ Best Practice: Consider checking if the thread is already started before calling `start`
        return False

    # 🧠 ML Signal: Method call pattern for checking account status
    def keepalive(self):
        if self.heart_thread.is_alive():
            self.heart_active = True
        # ⚠️ SAST Risk (Medium): Catching all exceptions without specific handling
        else:
            self.heart_thread.start()

    # ✅ Best Practice: Use a constant or configuration for sleep duration
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and behavior of the method.

    def send_heartbeat(self):
        # ✅ Best Practice: Method should have a docstring explaining its purpose
        # ✅ Best Practice: Ensure that self.baseinfo is initialized and has the expected structure.
        while True:
            # ✅ Best Practice: Use a constant or configuration for sleep duration
            if self.heart_active:
                # 🧠 ML Signal: Setting a flag to control the state of an object
                try:
                    # ✅ Best Practice: Add a docstring to describe the function and its parameters
                    response = self.heartbeat()
                    self.check_account_live(response)
                except:
                    self.login()
                time.sleep(100)
            else:
                time.sleep(10)

    # ⚠️ SAST Risk (Low): Potential typo in the docstring for 'pricce', should be 'price'
    def heartbeat(self):
        return self.baseinfo

    # 🧠 ML Signal: Pattern of checking JSON response for a specific key-value pair
    def exit(self):
        # ✅ Best Practice: Simplify boolean expression to directly return the condition
        # ✅ Best Practice: Add a docstring to describe the function's purpose and parameters
        self.heart_active = False

    def buy(self, stkcode, price=0, count=0, amount=0):
        """
        买入证券
            params
            ---------
            stkcode:股票代码，string
            pricce:委托价格，int
            count:买入数量
            amount:买入金额
        """
        # 🧠 ML Signal: Pattern of checking JSON response for a specific key-value pair
        jsonobj = utils.get_jdata(
            self._trading(
                stkcode,
                price,
                # 🧠 ML Signal: Usage of external service URL for trading operations
                # ✅ Best Practice: Simplify boolean expression to directly return the condition
                count,
                amount,
                "B",
                "buy",
            )
        )
        res = True if jsonobj["result"] == "true" else False
        return res

    def sell(self, stkcode, price=0, count=0, amount=0):
        """
        卖出证券
            params
            ---------
            stkcode:股票代码，string
            pricce:委托价格，int
            count:卖出数量
            amount:卖出金额
        """
        # ⚠️ SAST Risk (Low): Potential division by zero if price is zero
        jsonobj = utils.get_jdata(
            self._trading(stkcode, price, count, amount, "S", "sell")
        )
        # ⚠️ SAST Risk (Low): Potential division by zero if price is zero
        # ✅ Best Practice: Use a dictionary literal for better readability
        res = True if jsonobj["result"] == "true" else False
        return res

    def _trading(self, stkcode, price, count, amount, tradeflag, tradetype):
        txtdata = self.s.get(
            vs.TRADE_CHECK_URL
            % (
                vs.P_TYPE["https"],
                vs.DOMAINS["csc"],
                vs.PAGES["tradecheck"],
                tradeflag,
                stkcode,
                tradetype,
                utils.nowtime_str(),
            )
        )
        jsonobj = utils.get_jdata(txtdata)
        list = jsonobj["returnList"][0]
        secuid = list["buysSecuid"]
        fundavl = list["fundavl"]
        stkname = list["stkname"]
        if secuid is not None:
            # 🧠 ML Signal: Posting data to a trading service
            if tradeflag == "B":
                buytype = vs.BUY
                # ✅ Best Practice: Include a docstring to describe the function's purpose and return value
                count = count if count else amount // price // 100 * 100
            else:
                buytype = vs.SELL
                count = count if count else amount // price

            tradeparams = dict(
                stkname=stkname,
                stkcode=stkcode,
                secuid=secuid,
                buytype=buytype,
                bsflag=tradeflag,
                maxstkqty="",
                buycount=count,
                # 🧠 ML Signal: Method call pattern could be used to understand API usage
                buyprice=price,
                # 🧠 ML Signal: Updates headers with a constant value, indicating a pattern of modifying request headers.
                _=utils.nowtime_str(),
                # ⚠️ SAST Risk (Low): Potential risk if vs.BASE_URL or its components are user-controlled, leading to SSRF.
            )
            tradeResult = self.s.post(
                vs.TRADE_URL
                % (vs.P_TYPE["https"], vs.DOMAINS["csc"], vs.PAGES["trade"]),
                params=tradeparams,
            )
            # 🧠 ML Signal: Use of a utility function to parse JSON data, indicating a pattern of JSON handling.
            return tradeResult
        return None

    # 🧠 ML Signal: Conversion of JSON data to a DataFrame, indicating a pattern of data processing.
    # ✅ Best Practice: Returns a DataFrame, which is a common and efficient data structure for data manipulation.

    def position(self):
        """
        获取持仓列表
            return:DataFrame
            ----------------------
            stkcode:证券代码
            stkname:证券名称
            stkqty :证券数量
            stkavl :可用数量
            lastprice:最新价格
            costprice:成本价
            income :参考盈亏（元）
        """
        return self._get_position()

    # 🧠 ML Signal: Use of self.s.get indicates a pattern of making HTTP requests
    # ⚠️ SAST Risk (Medium): Potential exposure to HTTP response data without validation

    def _get_position(self):
        self.s.headers.update(vs.AGENT)
        txtdata = self.s.get(
            vs.BASE_URL
            % (
                vs.P_TYPE["https"],
                vs.DOMAINS["csc"],
                # 🧠 ML Signal: Use of a utility function to parse JSON data
                vs.PAGES["position"],
            )
        )
        # ⚠️ SAST Risk (Low): Potential for JSON parsing errors if data is malformed
        jsonobj = utils.get_jdata(txtdata)
        # 🧠 ML Signal: Conversion of JSON data to a DataFrame
        # ✅ Best Practice: Explicitly specify columns when creating a DataFrame for clarity
        # ✅ Best Practice: Add a docstring to describe the function's purpose and parameters
        df = pd.DataFrame(jsonobj["data"], columns=vs.POSITION_COLS)
        return df

    def entrust_list(self):
        """
        获取委托单列表
        return:DataFrame
        ----------
        ordersno:委托单号
        stkcode:证券代码
        stkname:证券名称
        bsflagState:买卖标志
        orderqty:委托数量
        matchqty:成交数量
        orderprice:委托价格
        operdate:交易日期
        opertime:交易时间
        orderdate:下单日期
        state:状态
        """
        txtdata = self.s.get(
            vs.ENTRUST_LIST_URL
            % (
                vs.P_TYPE["https"],
                vs.DOMAINS["csc"],
                # ✅ Best Practice: Use 'and' instead of '&' for logical operations
                vs.PAGES["entrustlist"],
                utils.nowtime_str(),
            )
        )
        jsonobj = utils.get_jdata(txtdata)
        df = pd.DataFrame(jsonobj["data"], columns=vs.ENTRUST_LIST_COLS)
        return df

    def deal_list(self, begin=None, end=None):
        """
        获取成交列表
            params
            -----------
            begin:开始日期  YYYYMMDD
            end:结束日期  YYYYMMDD

            return: DataFrame
            -----------
            ordersno:委托单号
            matchcode:成交编号
            trddate:交易日期
            matchtime:交易时间
            stkcode:证券代码
            stkname:证券名称
            bsflagState:买卖标志
            orderprice:委托价格
            matchprice:成交价格
            orderqty:委托数量
            matchqty:成交数量
            matchamt:成交金额
        """
        daterange = ""
        # ⚠️ SAST Risk (Medium): Potential vulnerability if vs.CANCEL_URL or vs.P_TYPE['https'], vs.DOMAINS['csc'], vs.PAGES['cancel'] are user-controlled.
        if (begin is None) & (end is None):
            selecttype = "intraDay"
        else:
            # ⚠️ SAST Risk (Medium): Potential risk if result.text contains unexpected or malicious content.
            daterange = vs.DEAL_DATE_RANGE % (begin, end)
            # 🧠 ML Signal: Accessing specific keys in a JSON object can indicate expected data structure.
            # ✅ Best Practice: Include a docstring to describe the function's purpose and return value
            selecttype = "all"
        txtdata = self.s.get(
            vs.DEAL_LIST_URL
            % (
                vs.P_TYPE["https"],
                vs.DOMAINS["csc"],
                vs.PAGES["deallist"],
                selecttype,
                daterange,
                utils.nowtime_str(),
            )
        )
        jsonobj = utils.get_jdata(txtdata)
        df = pd.DataFrame(jsonobj["data"], columns=vs.DEAL_LIST_COLS)
        return df

    def cancel(self, ordersno="", orderdate=""):
        """
                 撤单
        params
        -----------
        ordersno:委托单号，多个以逗号分隔，e.g. 1866,1867
        orderdata:委托日期 YYYYMMDD，多个以逗号分隔，对应委托单好
        return
        ------------
        string
        # ⚠️ SAST Risk (Medium): URL formatting with user-controlled variables can lead to SSRF or other injection attacks
        """
        if (ordersno != "") & (orderdate != ""):
            # ✅ Best Practice: Check if 'return_data' has 'get' method to ensure it behaves like a dictionary.
            # 🧠 ML Signal: Usage of utility function to parse JSON data
            params = dict(
                ordersno=ordersno,
                # ⚠️ SAST Risk (Low): Raising a custom exception without additional context may obscure the error source.
                # ✅ Best Practice: Custom exception class for specific error handling
                # 🧠 ML Signal: Accessing nested JSON data
                orderdate=orderdate,
                _=utils.nowtime_str(),
                # ✅ Best Practice: Use of default parameter value for flexibility
                # 🧠 ML Signal: Modifying dictionary with additional data
            )
            # ✅ Best Practice: Explicitly calling the superclass constructor
            # 🧠 ML Signal: Returning a pandas Series object
            result = self.s.post(
                vs.CANCEL_URL
                % (vs.P_TYPE["https"], vs.DOMAINS["csc"], vs.PAGES["cancel"]),
                # 🧠 ML Signal: Storing initialization parameters as instance variables
                # 🧠 ML Signal: Method name suggests a health check or status update pattern
                # 🧠 ML Signal: Returning an attribute directly may indicate a getter pattern
                params=params,
            )
            jsonobj = utils.get_jdata(result.text)
            return jsonobj["msgMap"]["ResultSucess"]
        return None

    def baseinfo(self):
        """
        获取帐户基本信息
            return: Series
            -------------
            fundid:帐户ID
            gpsz: 股票市值
            fundvalue:基金市值
            jihelicai:集合理财
            fundbal:帐户余额
            marketvalue:总资产
            fundavl:可用余额
            daixiao:代销份额
            otc:OTC份额
        """
        return self._get_baseinfo()

    def _get_baseinfo(self):
        self.s.headers.update(vs.AGENT)
        txtdata = self.s.get(
            vs.BASE_URL % (vs.P_TYPE["https"], vs.DOMAINS["csc"], vs.PAGES["baseInfo"])
        )
        jsonobj = utils.get_jdata(txtdata)
        stkdata = jsonobj["data"]["moneytype0"]
        stkdata["fundid"] = jsonobj["fundid"]
        return pd.Series(stkdata)

    def check_login_status(self, return_data):
        if hasattr(return_data, "get") and return_data.get("error_no") == "-1":
            raise NotLoginError


class NotLoginError(Exception):
    def __init__(self, result=None):
        super(NotLoginError, self).__init__()
        self.result = result

    def heartbeat(self):
        return self.baseinfo
