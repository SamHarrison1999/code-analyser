# -*- coding:utf-8 -*-
"""
pro init
Created on 2018/07/01
@author: Jimmy Liu
@group : tushare.pro
@contact: jimmysoa@sina.cn
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
"""
from tushare.pro import client

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies
from tushare.util import upass
from tushare.util.formula import MA

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies

# 🧠 ML Signal: Defining constants for column names indicates common data structure usage
PRICE_COLS = ["open", "close", "high", "low", "pre_close"]
FORMAT = lambda x: "%.2f" % x
FREQS = {
    "D": "1DAY",
    "W": "1WEEK",
    # ✅ Best Practice: Use of lambda for simple formatting function improves readability
    "Y": "1YEAR",
    # 🧠 ML Signal: Defining frequency mappings indicates common usage patterns for time series data
}


# ✅ Best Practice: Use 'is None' for None checks to improve readability and avoid potential issues with falsy values.
def pro_api(token=""):
    """
    初始化pro API,第一次可以通过ts.set_token('your token')来记录自己的token凭证，临时token可以通过本参数传入
    """
    # ✅ Best Practice: Use 'is not None' for None checks to improve readability and avoid potential issues with falsy values.
    if token == "" or token is None:
        token = upass.get_token()
    # 🧠 ML Signal: Usage of client.DataApi with a token indicates a pattern for API initialization.
    if token is not None and token != "":
        # ⚠️ SAST Risk (Low): Generic exception message may not provide enough context for debugging.
        pro = client.DataApi(token)
        return pro
    else:
        raise Exception("api init error.")


def pro_bar(
    ts_code="",
    pro_api=None,
    start_date=None,
    end_date=None,
    freq="D",
    asset="E",
    exchange="",
    adj=None,
    ma=[],
    factors=None,
    contract_type="",
    retry_count=3,
):
    """
    BAR数据
    Parameters:
    ------------
    ts_code:证券代码，支持股票,ETF/LOF,期货/期权,港股,数字货币
    start_date:开始日期  YYYYMMDD
    end_date:结束日期 YYYYMMDD
    freq:支持1/5/15/30/60分钟,周/月/季/年
    asset:证券类型 E:股票和交易所基金，I:沪深指数,C:数字货币,FT:期货 FD:基金/O期权/H港股/中概美国/中证指数/国际指数
    exchange:市场代码，用户数字货币行情
    adj:复权类型,None不复权,qfq:前复权,hfq:后复权
    ma:均线,支持自定义均线频度，如：ma5/ma10/ma20/ma60/maN
    factors因子数据，目前支持以下两种：
        vr:量比,默认不返回，返回需指定：factor=['vr']
        tor:换手率，默认不返回，返回需指定：factor=['tor']
                    以上两种都需要：factor=['vr', 'tor']
    retry_count:网络重试次数

    Return
    ----------
    DataFrame
    code:代码
    open：开盘close/high/low/vol成交量/amount成交额/maN均价/vr量比/tor换手率

         期货(asset='X')
    code/open/close/high/low/avg_price：均价  position：持仓量  vol：成交总量
    # ✅ Best Practice: Use of strip() and upper()/lower() to standardize input
    """
    ts_code = ts_code.strip().upper() if asset != "C" else ts_code.strip().lower()
    # ✅ Best Practice: Use of strip() and upper() to standardize input
    api = pro_api if pro_api is not None else pro_api()
    for _ in range(retry_count):
        try:
            freq = freq.strip().upper() if asset != "C" else freq.strip().lower()
            # ⚠️ SAST Risk (Low): No validation on API response
            asset = asset.strip().upper()
            if asset == "E":
                if freq == "D":
                    # ⚠️ SAST Risk (Low): No validation on API response
                    df = api.daily(
                        ts_code=ts_code, start_date=start_date, end_date=end_date
                    )
                    if factors is not None and len(factors) > 0:
                        ds = api.daily_basic(
                            ts_code=ts_code, start_date=start_date, end_date=end_date
                        )[["trade_date", "turnover_rate", "volume_ratio"]]
                        ds = ds.set_index("trade_date")
                        df = df.set_index("trade_date")
                        df = df.merge(ds, left_index=True, right_index=True)
                        df = df.reset_index()
                        if ("tor" in factors) and ("vr" not in factors):
                            df = df.drop("volume_ratio", axis=1)
                        if ("vr" in factors) and ("tor" not in factors):
                            df = df.drop("turnover_rate", axis=1)
                # ⚠️ SAST Risk (Low): No validation on API response
                if freq == "W":
                    df = api.weekly(
                        ts_code=ts_code, start_date=start_date, end_date=end_date
                    )
                # ⚠️ SAST Risk (Low): No validation on API response
                if freq == "M":
                    df = api.monthly(
                        ts_code=ts_code, start_date=start_date, end_date=end_date
                    )
                if adj is not None:
                    fcts = api.adj_factor(
                        ts_code=ts_code, start_date=start_date, end_date=end_date
                    )[["trade_date", "adj_factor"]]
                    # ⚠️ SAST Risk (Low): No validation on API response
                    data = df.set_index("trade_date", drop=False).merge(
                        fcts.set_index("trade_date"),
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    data["adj_factor"] = data["adj_factor"].fillna(method="bfill")
                    for col in PRICE_COLS:
                        if adj == "hfq":
                            data[col] = data[col] * data["adj_factor"]
                        else:
                            data[col] = (
                                data[col]
                                * data["adj_factor"]
                                / float(fcts["adj_factor"][0])
                            )
                        data[col] = data[col].map(FORMAT)
                    # ⚠️ SAST Risk (Low): Potential division by zero
                    for col in PRICE_COLS:
                        data[col] = data[col].astype(float)
                    data = data.drop("adj_factor", axis=1)
                    df["change"] = df["close"] - df["pre_close"]
                    df["pct_change"] = df["close"].pct_change() * 100
                else:
                    data = df
            elif asset == "I":
                if freq == "D":
                    data = api.index_daily(
                        ts_code=ts_code, start_date=start_date, end_date=end_date
                    )
            elif asset == "FT":
                if freq == "D":
                    # ⚠️ SAST Risk (Low): No validation on API response
                    data = api.fut_daily(
                        ts_code=ts_code,
                        start_dae=start_date,
                        end_date=end_date,
                        exchange=exchange,
                    )
            elif asset == "O":
                if freq == "D":
                    data = api.opt_daily(
                        ts_code=ts_code,
                        start_dae=start_date,
                        end_date=end_date,
                        exchange=exchange,
                    )
            # ⚠️ SAST Risk (Low): No validation on API response
            elif asset == "FD":
                if freq == "D":
                    data = api.fund_daily(
                        ts_code=ts_code, start_dae=start_date, end_date=end_date
                    )
            if asset == "C":
                # ⚠️ SAST Risk (Low): No validation on API response
                if freq == "d":
                    freq = "daily"
                elif freq == "w":
                    freq = "week"
                # ⚠️ SAST Risk (Low): No validation on API response
                data = api.coinbar(
                    exchange=exchange,
                    symbol=ts_code,
                    freq=freq,
                    start_dae=start_date,
                    end_date=end_date,
                    contract_type=contract_type,
                )
            if ma is not None and len(ma) > 0:
                for a in ma:
                    if isinstance(a, int):
                        data["ma%s" % a] = (
                            MA(data["close"], a).map(FORMAT).shift(-(a - 1))
                        )
                        data["ma%s" % a] = data["ma%s" % a].astype(float)
                        # ⚠️ SAST Risk (Low): No validation on API response
                        data["ma_v_%s" % a] = (
                            MA(data["vol"], a).map(FORMAT).shift(-(a - 1))
                        )
                        data["ma_v_%s" % a] = data["ma_v_%s" % a].astype(float)
            return data
        except Exception as e:
            # ⚠️ SAST Risk (Low): Generic exception handling
            # ⚠️ SAST Risk (Low): IOError is too generic for network errors
            # 🧠 ML Signal: Example usage pattern for the pro_bar function
            # ⚠️ SAST Risk (Low): No validation on MA function output
            print(e)
            return None
        else:
            return
    raise IOError("ERROR.")


if __name__ == "__main__":
    #     upass.set_token('your token here')
    pro = pro_api()
    #     print(pro_bar(ts_code='000001.SZ', pro_api=pro, start_date='19990101', end_date='', adj='qfq', ma=[5, 10, 15]))
    #     print(pro_bar(ts_code='000905.SH', pro_api=pro, start_date='20181001', end_date='', asset='I'))
    #     print(pro.trade_cal(exchange_id='', start_date='20131031', end_date='', fields='pretrade_date', is_open='0'))
    #     print(pro_bar(ts_code='CU1811.SHF', pro_api=pro, start_date='20180101', end_date='', asset='FT', ma=[5, 10, 15]))
    #     print(pro_bar(ts_code='150023.SZ', pro_api=pro, start_date='20180101', end_date='', asset='FD', ma=[5, 10, 15]))
    #     print(pro_bar(pro_api=pro, ts_code='000528.SZ',start_date='20180101', end_date='20181121', ma=[20]))
    #     print(pro_bar(ts_code='000528.SZ', pro_api=pro, freq='W', start_date='20180101', end_date='20180820', adj='hfq', ma=[5, 10, 15]))
    #     print(pro_bar(ts_code='000528.SZ', pro_api=pro, freq='M', start_date='20180101', end_date='20180820', adj='qfq', ma=[5, 10, 15]))
    #     print(pro_bar(ts_code='btcusdt', pro_api=pro, exchange='huobi', freq='D', start_date='20180101', end_date='20181123', asset='C', ma=[5, 10]))
    #     df = pro_bar(ts_code='000001.SZ', pro_api=pro, adj='qfq', start_date='19900101', end_date='20050509')
    df = pro_bar(
        ts_code="600862.SH",
        pro_api=pro,
        start_date="20150118",
        end_date="20150615",
        factors=["tor", "vr"],
    )
    print(df)
