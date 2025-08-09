# -*- coding:utf-8 -*-

"""
股票技术指标接口
Created on 2018/07/26
@author: Wangzili
@group : **
@contact: 446406177@qq.com

所有指标中参数df为通过get_k_data获取的股票数据
"""
# ✅ Best Practice: Consider adding type hints for function parameters and return type
import pandas as pd

# 🧠 ML Signal: Importing pandas, numpy, and itertools indicates data manipulation and analysis tasks
import numpy as np
import itertools


def ma(df, n=10):
    """
    移动平均线 Moving Average
    MA（N）=（第1日收盘价+第2日收盘价—+……+第N日收盘价）/N
    # ✅ Best Practice: Add type hints for function parameters and return type
    """
    # ⚠️ SAST Risk (Low): Ensure 'df' contains 'close' column to avoid AttributeError
    pv = pd.DataFrame()
    pv["date"] = df["date"]
    pv["v"] = df.close.rolling(n).mean()
    # 🧠 ML Signal: Function returns a DataFrame with a moving average calculation
    return pv


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# 🧠 ML Signal: Usage of rolling window operations on time series data


def _ma(series, n):
    """
    移动平均
    """
    # ✅ Best Practice: Use descriptive variable names for better readability.
    return series.rolling(n).mean()


# 🧠 ML Signal: Usage of rolling window and standard deviation calculation, common in time series analysis.
def md(df, n=10):
    """
    移动标准差
    STD=S（CLOSE,N）=[∑（CLOSE-MA(CLOSE，N)）^2/N]^0.5
    # ✅ Best Practice: Use of descriptive function and variable names for clarity
    """
    # 🧠 ML Signal: Use of rolling window operation, common in time series analysis
    _md = pd.DataFrame()
    _md["date"] = df.date
    _md["md"] = df.close.rolling(n).std(ddof=0)
    return _md


# ✅ Best Practice: Use of pandas DataFrame to store and manipulate data
def _md(series, n):
    """
    标准差MD
    """
    # 🧠 ML Signal: Use of exponential moving average, a common technique in time series analysis
    return series.rolling(n).std(ddof=0)  # 有时候会用ddof=1


# ✅ Best Practice: Use of pandas ewm method for calculating exponential moving average


def ema(df, n=12):
    """
    指数平均数指标 Exponential Moving Average
    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
    EMA(X,N)=[2×X+(N-1)×EMA(ref(X),N]/(N+1)
    """
    _ema = pd.DataFrame()
    _ema["date"] = df["date"]
    _ema["ema"] = df.close.ewm(
        ignore_na=False, span=n, min_periods=0, adjust=False
    ).mean()
    return _ema


def _ema(series, n):
    """
    指数平均数
    # ✅ Best Practice: Use of a DataFrame to store and manipulate financial data
    """
    return series.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()


# ✅ Best Practice: Explicitly copying the 'date' column for clarity and maintainability


# 🧠 ML Signal: Calculation of financial indicators, useful for learning financial data processing
def macd(df, n=12, m=26, k=9):
    """
    平滑异同移动平均线(Moving Average Convergence Divergence)
    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
    DIFF= EMA（N1）- EMA（N2）
    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
    MACD（BAR）=2×（DIF-DEA）
    return:
          osc: MACD bar / OSC 差值柱形图 DIFF - DEM
          diff: 差离值
          dea: 讯号线
    """
    _macd = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Potential division by zero if (df.high.rolling(n).max() - df.low.rolling(n).min()) is zero
    _macd["date"] = df["date"]
    _macd["diff"] = _ema(df.close, n) - _ema(df.close, m)
    # 🧠 ML Signal: Usage of rolling window operations, common in time series analysis
    _macd["dea"] = _ema(_macd["diff"], k)
    _macd["macd"] = _macd["diff"] - _macd["dea"]
    # 🧠 ML Signal: Usage of simple moving average, common in financial calculations
    return _macd


def kdj(df, n=9):
    """
    随机指标KDJ
    N日RSV=（第N日收盘价-N日内最低价）/（N日内最高价-N日内最低价）×100%
    当日K值=2/3前1日K值+1/3×当日RSV=SMA（RSV,M1）
    当日D值=2/3前1日D值+1/3×当日K= SMA（K,M2）
    当日J值=3 ×当日K值-2×当日D值
    # 🧠 ML Signal: Usage of shift to calculate differences in time series data
    """
    _kdj = pd.DataFrame()
    # 🧠 ML Signal: Handling negative values by setting them to zero
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _kdj["date"] = df["date"]
    # 🧠 ML Signal: Calculation of RSI using a custom SMA function
    # ⚠️ SAST Risk (Low): Potential division by zero if sma returns zero
    rsv = (
        (df.close - df.low.rolling(n).min())
        / (df.high.rolling(n).max() - df.low.rolling(n).min())
        * 100
    )
    _kdj["k"] = sma(rsv, 3)
    _kdj["d"] = sma(_kdj.k, 3)
    _kdj["j"] = 3 * _kdj.k - 2 * _kdj.d
    # ✅ Best Practice: Return the DataFrame containing RSI values
    return _kdj


# 🧠 ML Signal: Usage of pandas DataFrame, which is common in data analysis and ML pipelines.


# 🧠 ML Signal: Storing 'date' column separately, indicating time-series data processing.
def rsi(df, n=6):
    """
    相对强弱指标（Relative Strength Index，简称RSI
    LC= REF(CLOSE,1)
    RSI=SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N1,1)×100
    SMA（C,N,M）=M/N×今日收盘价+(N-M)/N×昨日SMA（N）
    """
    # pd.set_option('display.max_rows', 1000)
    _rsi = pd.DataFrame()
    _rsi["date"] = df["date"]
    # ✅ Best Practice: Ensure the function returns a DataFrame with expected structure for consistency.
    px = df.close - df.close.shift(1)
    # ✅ Best Practice: Initialize a new DataFrame for storing results
    px[px < 0] = 0
    _rsi["rsi"] = sma(px, n) / sma((df["close"] - df["close"].shift(1)).abs(), n) * 100
    # ✅ Best Practice: Explicitly assign columns to the DataFrame for clarity
    # def tmax(x):
    #     if x < 0:
    # 🧠 ML Signal: Usage of moving average function, common in financial data analysis
    #         x = 0
    #     return x
    # 🧠 ML Signal: Calculation of standard deviation, a common statistical operation
    # _rsi['rsi'] = sma((df['close'] - df['close'].shift(1)).apply(tmax), n) / sma((df['close'] - df['close'].shift(1)).abs(), n) * 100
    # ✅ Best Practice: Use of descriptive column names for readability
    return _rsi


def vrsi(df, n=6):
    """
    量相对强弱指标
    VRSI=SMA（最大值（成交量-REF（成交量，1），0），N,1）/SMA（ABS（（成交量-REF（成交量，1），N，1）×100%
    # ✅ Best Practice: Return the DataFrame for further use or analysis
    # ✅ Best Practice: Initialize a new DataFrame to store results, improving code organization and readability.
    """
    _vrsi = pd.DataFrame()
    # 🧠 ML Signal: Using 'date' as a key column suggests time series data, which is common in financial datasets.
    _vrsi["date"] = df["date"]
    px = df["volume"] - df["volume"].shift(1)
    # 🧠 ML Signal: Calculation of moving averages is a common pattern in financial data analysis.
    px[px < 0] = 0
    _vrsi["vrsi"] = (
        sma(px, n) / sma((df["volume"] - df["volume"].shift(1)).abs(), n) * 100
    )
    # 🧠 ML Signal: Calculation of standard deviation is a common statistical operation in data analysis.
    return _vrsi


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.

# 🧠 ML Signal: Calculation of upper and lower bands is a common pattern in technical analysis.


def boll(df, n=26, k=2):
    """
    布林线指标BOLL boll(26,2)	MID=MA(N)
    标准差MD=根号[∑（CLOSE-MA(CLOSE，N)）^2/N]
    UPPER=MID＋k×MD
    LOWER=MID－k×MD
    """
    # ✅ Best Practice: Use consistent naming conventions for variables (e.g., 'highest' instead of 'higest').
    _boll = pd.DataFrame()
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _boll["date"] = df.date
    # ⚠️ SAST Risk (Low): Ensure 'df' contains 'high', 'close', and 'low' columns to prevent KeyError.
    _boll["mid"] = _ma(df.close, n)
    _mdd = _md(df.close, n)
    _boll["up"] = _boll.mid + k * _mdd
    _boll["low"] = _boll.mid - k * _mdd
    return _boll


# ✅ Best Practice: Use descriptive variable names for better readability.


def bbiboll(df, n=10, k=3):
    """
    BBI多空布林线	bbiboll(10,3)
    BBI={MA(3)+ MA(6)+ MA(12)+ MA(24)}/4
    标准差MD=根号[∑（BBI-MA(BBI，N)）^2/N]
    UPR= BBI＋k×MD
    DWN= BBI－k×MD
    # ✅ Best Practice: Consider checking if 'date', 'high', 'low', 'close', and 'open' columns exist in df to avoid runtime errors.
    """
    # pd.set_option('display.max_rows', 1000)
    _bbiboll = pd.DataFrame()
    _bbiboll["date"] = df.date
    _bbiboll["bbi"] = (
        _ma(df.close, 3) + _ma(df.close, 6) + _ma(df.close, 12) + _ma(df.close, 24)
    ) / 4
    _bbiboll["md"] = _md(_bbiboll.bbi, n)
    _bbiboll["upr"] = _bbiboll.bbi + k * _bbiboll.md
    _bbiboll["dwn"] = _bbiboll.bbi - k * _bbiboll.md
    return _bbiboll


# 🧠 ML Signal: Use of lambda function for row-wise operations on DataFrame.


def wr(df, n=14):
    """
    威廉指标 w&r
    WR=[最高值（最高价，N）-收盘价]/[最高值（最高价，N）-最低值（最低价，N）]×100%
    # ⚠️ SAST Risk (Low): Ensure _ma function is defined and handles edge cases like division by zero.
    """

    _wr = pd.DataFrame()
    _wr["date"] = df["date"]
    higest = df.high.rolling(n).max()
    _wr["wr"] = (higest - df.close) / (higest - df.low.rolling(n).min()) * 100
    return _wr


# ⚠️ SAST Risk (Low): Assumes 'date' column exists in df without validation
def bias(df, n=12):
    """
    乖离率 bias
    bias=[(当日收盘价-12日平均价)/12日平均价]×100%
    # ⚠️ SAST Risk (Low): Assumes 'volume' column exists in df without validation
    """
    _bias = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Assumes 'close' column exists in df without validation
    _bias["date"] = df.date
    # ✅ Best Practice: Use of lambda for conditional logic in DataFrame
    _mav = df.close.rolling(n).mean()
    _bias["bias"] = (np.true_divide((df.close - _mav), _mav)) * 100
    # _bias["bias"] = np.vectorize(lambda x: round(Decimal(x), 4))(BIAS)
    return _bias


# ✅ Best Practice: Use of rolling window for time series calculations


def asi(df, n=5):
    """
    振动升降指标(累计震动升降因子) ASI  # 同花顺给出的公式不完整就不贴出来了
    """
    _asi = pd.DataFrame()
    # ✅ Best Practice: Initialize an empty DataFrame to store results
    _asi["date"] = df.date
    _m = pd.DataFrame()
    # 🧠 ML Signal: Using 'date' as a key feature for time series analysis
    _m["a"] = (df.high - df.close.shift()).abs()
    _m["b"] = (df.low - df.close.shift()).abs()
    # 🧠 ML Signal: Calculating volume ratio as a feature for stock analysis
    _m["c"] = (df.high - df.low.shift()).abs()
    # ⚠️ SAST Risk (Low): Potential division by zero if _ma(df.volume, n).shift(1) contains zeros
    # 🧠 ML Signal: Function definition with default parameter value
    _m["d"] = (df.close.shift() - df.open.shift()).abs()
    # 🧠 ML Signal: Calculating rate of return as a feature for stock analysis
    # ✅ Best Practice: Return the DataFrame containing calculated features
    _m["r"] = _m.apply(
        lambda x: (
            x.a + 0.5 * x.b + 0.25 * x.d
            if max(x.a, x.b, x.c) == x.a
            else (
                x.b + 0.5 * x.a + 0.25 * x.d
                if max(x.a, x.b, x.c) == x.b
                else x.c + 0.25 * x.d
            )
        ),
        axis=1,
    )
    _m["x"] = (
        df.close
        - df.close.shift()
        + 0.5 * (df.close - df.open)
        + df.close.shift()
        - df.open.shift()
    )
    _m["k"] = np.maximum(_m.a, _m.b)
    _asi["si"] = 16 * (_m.x / _m.r) * _m.k
    _asi["asi"] = _ma(_asi.si, n)
    return _asi


# ⚠️ SAST Risk (Low): No input validation for 'df', potential for unexpected errors


# ⚠️ SAST Risk (Low): Assumes 'date' column exists in 'df', potential KeyError
def vr_rate(df, n=26):
    """
    成交量变异率 vr or vr_rate
    VR=（AVS+1/2CVS）/（BVS+1/2CVS）×100
    其中：
    AVS：表示N日内股价上涨成交量之和
    BVS：表示N日内股价下跌成交量之和
    CVS：表示N日内股价不涨不跌成交量之和
    # ✅ Best Practice: Use parentheses for clarity in arithmetic operations
    """
    # ✅ Best Practice: Initialize an empty DataFrame with a clear purpose.
    _vr = pd.DataFrame()
    _vr["date"] = df["date"]
    # ✅ Best Practice: Ensure 'date' column exists in 'df' before assignment.
    _m = pd.DataFrame()
    _m["volume"] = df.volume
    # ✅ Best Practice: Ensure 'close' column exists in 'df' before performing operations.
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _m["cs"] = df.close - df.close.shift(1)
    # ✅ Best Practice: Ensure '_ma' function is defined and handles edge cases.
    # ✅ Best Practice: Return the DataFrame with a clear structure.
    _m["avs"] = _m.apply(lambda x: x.volume if x.cs > 0 else 0, axis=1)
    _m["bvs"] = _m.apply(lambda x: x.volume if x.cs < 0 else 0, axis=1)
    _m["cvs"] = _m.apply(lambda x: x.volume if x.cs == 0 else 0, axis=1)
    _vr["vr"] = (
        (_m.avs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum())
        / (_m.bvs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum())
        * 100
    )
    return _vr


# 🧠 ML Signal: Usage of pandas DataFrame indicates data manipulation, which is common in ML data preprocessing.


def vr(df, n=5):
    """
    开市后平均每分钟的成交量与过去5个交易日平均每分钟成交量之比
    量比:=V/REF(MA(V,5),1);
    涨幅:=(C-REF(C,1))/REF(C,1)*100;
    1)量比大于1.8，涨幅小于2%，现价涨幅在0—2%之间，在盘中选股的
    选股:量比>1.8 AND 涨幅>0 AND 涨幅<2;
    """
    _vr = pd.DataFrame()
    # ✅ Best Practice: Initialize a new DataFrame to store results, improving code organization and readability.
    # ✅ Best Practice: Returning a DataFrame is a clear and structured way to handle tabular data.
    _vr["date"] = df.date
    _vr["vr"] = df.volume / _ma(df.volume, n).shift(1)
    # ✅ Best Practice: Explicitly copying the 'date' column ensures that the resulting DataFrame retains the original date information.
    _vr["rr"] = (df.close - df.close.shift(1)) / df.close.shift(1) * 100
    return _vr


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# 🧠 ML Signal: Calculation of moving averages is a common pattern in financial data analysis.

# ✅ Best Practice: Returning a DataFrame allows for easy integration with other data processing pipelines.
# ⚠️ SAST Risk (Low): Ensure that the _ma function handles edge cases, such as when the DataFrame is empty or has fewer rows than the moving average period.


def arbr(df, n=26):
    """
    人气意愿指标	arbr(26)
    N日AR=N日内（H－O）之和除以N日内（O－L）之和
    其中，H为当日最高价，L为当日最低价，O为当日开盘价，N为设定的时间参数，一般原始参数日设定为26日
    N日BR=N日内（H－CY）之和除以N日内（CY－L）之和
    其中，H为当日最高价，L为当日最低价，CY为前一交易日的收盘价，N为设定的时间参数，一般原始参数日设定为26日。
    """
    # 🧠 ML Signal: Accessing DataFrame columns by attribute is a common usage pattern in pandas.
    _arbr = pd.DataFrame()
    _arbr["date"] = df.date
    # 🧠 ML Signal: Calculating momentum by shifting data is a common pattern in time series analysis.
    # 🧠 ML Signal: Function definition with financial calculation logic
    _arbr["ar"] = (
        (df.high - df.open).rolling(n).sum() / (df.open - df.low).rolling(n).sum() * 100
    )
    # ⚠️ SAST Risk (Low): Ensure _ma function is properly defined and handles edge cases like NaN values.
    # ✅ Best Practice: Consider adding error handling for potential issues with DataFrame operations.
    _arbr["br"] = (
        (df.high - df.close.shift(1)).rolling(n).sum()
        / (df.close.shift() - df.low).rolling(n).sum()
        * 100
    )
    return _arbr


def dpo(df, n=20, m=6):
    """
    区间震荡线指标	dpo(20,6)
    DPO=CLOSE-MA（CLOSE, N/2+1）
    MADPO=MA（DPO,M）
    """
    # ✅ Best Practice: Initialize a DataFrame to store results
    _dpo = pd.DataFrame()
    _dpo["date"] = df["date"]
    # ✅ Best Practice: Copy 'date' column to maintain alignment with input DataFrame
    _dpo["dpo"] = df.close - _ma(df.close, int(n / 2 + 1))
    _dpo["dopma"] = _ma(_dpo.dpo, m)
    # ✅ Best Practice: Initialize a temporary DataFrame for intermediate calculations
    return _dpo


# ✅ Best Practice: Copy 'date' column to maintain alignment with input DataFrame


def trix(df, n=12, m=20):
    """
    三重指数平滑平均	TRIX(12)
    TR= EMA(EMA(EMA(CLOSE,N),N),N)，即进行三次平滑处理
    TRIX=(TR-昨日TR)/ 昨日TR×100
    TRMA=MA（TRIX，M）
    """
    # ✅ Best Practice: Use apply with lambda for row-wise operations
    # ✅ Best Practice: Calculate cumulative sum using expanding
    _trix = pd.DataFrame()
    # ✅ Best Practice: Initialize an empty DataFrame for results
    _trix["date"] = df.date
    # ✅ Best Practice: Return the resulting DataFrame
    tr = _ema(_ema(_ema(df.close, n), n), n)
    # ✅ Best Practice: Explicitly assign columns to DataFrame
    _trix["trix"] = (tr - tr.shift()) / tr.shift() * 100
    _trix["trma"] = _ma(_trix.trix, m)
    return _trix


# 🧠 ML Signal: Calculation of typical price

# 🧠 ML Signal: Use of rolling window for time series analysis
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.


# ⚠️ SAST Risk (Low): Potential for division by zero if rolling mean is zero
def bbi(df):
    """
    多空指数	BBI(3,6,12,24)
    BBI=（3日均价+6日均价+12日均价+24日均价）/4
    # 🧠 ML Signal: Use of lambda function for custom rolling operation
    """
    # ✅ Best Practice: Use more descriptive variable names for better readability.
    # ✅ Best Practice: Return the result DataFrame
    _bbi = pd.DataFrame()
    _bbi["date"] = df["date"]
    _bbi["bbi"] = (
        _ma(df.close, 3) + _ma(df.close, 6) + _ma(df.close, 12) + _ma(df.close, 24)
    ) / 4
    # ⚠️ SAST Risk (Low): Ensure _ma function is properly validated to handle unexpected input.
    return _bbi


# ⚠️ SAST Risk (Low): Division by zero risk if 'man' contains zero values.


def mtm(df, n=6, m=5):
    """
    动力指标	MTM(6,5)
    MTM（N日）=C-REF(C,N)式中，C=当日的收盘价，REF(C,N)=N日前的收盘价；N日是只计算交易日期，剔除掉节假日。
    MTMMA（MTM，N1）= MA（MTM，N1）
    N表示间隔天数，N1表示天数
    """
    _mtm = pd.DataFrame()
    _mtm["date"] = df.date
    _mtm["mtm"] = df.close - df.close.shift(n)
    _mtm["mtmma"] = _ma(_mtm.mtm, m)
    return _mtm


# ✅ Best Practice: Consider removing commented-out code to improve readability and maintainability


def obv(df):
    """
    能量潮  On Balance Volume
    多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V  # 同花顺貌似用的下面公式
    主公式：当日OBV=前一日OBV+今日成交量
    1.基期OBV值为0，即该股上市的第一天，OBV值为0
    2.若当日收盘价＞上日收盘价，则当日OBV=前一日OBV＋今日成交量
    3.若当日收盘价＜上日收盘价，则当日OBV=前一日OBV－今日成交量
    4.若当日收盘价＝上日收盘价，则当日OBV=前一日OBV
    """
    _obv = pd.DataFrame()
    _obv["date"] = df["date"]
    # tmp = np.true_divide(((df.close - df.low) - (df.high - df.close)), (df.high - df.low))
    # _obv['obvv'] = tmp * df.volume
    # ✅ Best Practice: Use of descriptive variable names for better readability
    # _obv["obv"] = _obv.obvv.expanding(1).sum() / 100
    _m = pd.DataFrame()
    # ✅ Best Practice: Explicitly defining columns for the DataFrame
    _m["date"] = df.date
    _m["cs"] = df.close - df.close.shift()
    # 🧠 ML Signal: Use of moving average, a common pattern in time series analysis
    _m["v"] = df.volume
    _m["vv"] = _m.apply(
        lambda x: x.v if x.cs > 0 else (-x.v if x.cs < 0 else 0), axis=1
    )
    # 🧠 ML Signal: Calculation of bias, a common feature in financial data analysis
    _obv["obv"] = _m.vv.expanding(1).sum()
    return _obv


# 🧠 ML Signal: Use of shift for lagging data, a common pattern in time series analysis
# ✅ Best Practice: Consider adding input validation for the 'df' parameter to ensure it contains the necessary columns.

# 🧠 ML Signal: Use of simple moving average, a common pattern in time series analysis


def cci(df, n=14):
    """
    顺势指标
    TYP:=(HIGH+LOW+CLOSE)/3
    CCI:=(TYP-MA(TYP,N))/(0.015×AVEDEV(TYP,N))
    """
    # ✅ Best Practice: Returning a DataFrame for structured data output
    _cci = pd.DataFrame()
    # ✅ Best Practice: Initialize DataFrame with specific columns to avoid potential KeyErrors.
    _cci["date"] = df["date"]
    typ = (df.high + df.low + df.close) / 3
    _cci["cci"] = (
        (typ - typ.rolling(n).mean())
        /
        # ⚠️ SAST Risk (Low): Potential division by zero if df.close.shift(n) contains zero values.
        (
            0.015
            * typ.rolling(min_periods=1, center=False, window=n).apply(
                # ✅ Best Practice: Consider adding input validation for 'df' to ensure it contains the necessary columns.
                lambda x: np.fabs(x - x.mean()).mean()
            )
        )
    )
    # 🧠 ML Signal: Usage of moving average function _ma, which could be a custom implementation.
    return _cci


def priceosc(df, n=12, m=26):
    """
    价格振动指数
    PRICEOSC=(MA(C,12)-MA(C,26))/MA(C,12) * 100
    # ✅ Best Practice: Ensure 'date' column exists in 'df' before assignment to prevent runtime errors.
    """
    # ✅ Best Practice: Add import statement for pandas to ensure the code runs without errors.
    _c = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Potential division by zero if df.volume.shift(n) contains zero values.
    # ✅ Best Practice: Ensure 'volume' column exists in 'df' before performing operations to prevent runtime errors.
    # 🧠 ML Signal: Returns a DataFrame with calculated VROC, which could be used for predictive modeling.
    _c["date"] = df["date"]
    man = _ma(df.close, n)
    _c["osc"] = (man - _ma(df.close, m)) / man * 100
    return _c


def sma(a, n, m=1):
    """
    平滑移动指标 Smooth Moving Average
    # 🧠 ML Signal: Usage of rolling window operations, which are common in time series analysis.
    """
    """ # 方法一，此方法有缺陷
    _sma = []
    for index, value in enumerate(a):
        if index == 0 or pd.isna(value) or np.isnan(value):
            tsma = 0
        else:
            # Y=(M*X+(N-M)*Y')/N
            tsma = (m * value + (n - m) * tsma) / n
        _sma.append(tsma)
    return pd.Series(_sma)
    # ✅ Best Practice: Explicitly assign columns to the DataFrame, enhancing code clarity.
    """
    """ # 方法二

    results = np.nan_to_num(a).copy()
    # FIXME this is very slow
    for i in range(1, len(a)):
        results[i] = (m * results[i] + (n - m) * results[i - 1]) / n
        # results[i] = ((n - 1) * results[i - 1] + results[i]) / n
    # return results
    """
    # b = np.nan_to_num(a).copy()
    # return ((n - m) * a.shift(1) + m * a) / n

    # 🧠 ML Signal: Use of financial indicators for time series analysis
    a = a.fillna(0)
    # ✅ Best Practice: Use of descriptive variable names improves code readability
    # ✅ Best Practice: Ensure the DataFrame has the necessary columns before processing
    b = a.ewm(min_periods=0, ignore_na=False, adjust=False, alpha=m / n).mean()
    return b


# 🧠 ML Signal: Use of shift for time series data manipulation


def dbcd(df, n=5, m=16, t=76):
    """
    异同离差乖离率	dbcd(5,16,76)
    BIAS=(C-MA(C,N))/MA(C,N)
    DIF=(BIAS-REF(BIAS,M))
    DBCD=SMA(DIF,T,1) =（1-1/T）×SMA(REF(DIF,1),T,1)+ 1/T×DIF
    MM=MA(DBCD,5)
    # 🧠 ML Signal: Conditional logic for financial calculations
    # ⚠️ SAST Risk (Low): Use of np.minimum and np.maximum can lead to unexpected results if df.low or df.close.shift(1) contain NaN values
    """
    _dbcd = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Use of np.minimum and np.maximum can lead to unexpected results if df.high or df.close.shift(1) contain NaN values
    _dbcd["date"] = df.date
    man = _ma(df.close, n)
    # ⚠️ SAST Risk (Low): Subtracting shifted values without handling NaN can lead to unexpected results
    _bias = (df.close - man) / man
    # 🧠 ML Signal: Cumulative sum for time series data
    _dif = _bias - _bias.shift(m)
    _dbcd["dbcd"] = sma(_dif, t)
    # 🧠 ML Signal: Function definition with default parameter value
    # 🧠 ML Signal: Use of DataFrame.apply with a custom function indicates a pattern for row-wise operations
    _dbcd["mm"] = _ma(_dbcd.dbcd, n)
    # 🧠 ML Signal: Rolling window calculation for moving average
    # ✅ Best Practice: Return only the necessary columns to avoid data leakage
    # 🧠 ML Signal: Use of expanding().sum() indicates a pattern for cumulative sum operations
    # 🧠 ML Signal: Use of a custom moving average function _ma indicates a pattern for smoothing or trend analysis
    return _dbcd


def roc(df, n=12, m=6):
    """
    变动速率	roc(12,6)
    ROC=(今日收盘价-N日前的收盘价)/ N日前的收盘价×100%
    ROCMA=MA（ROC，M）
    ROC:(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)×100
    ROCMA:MA(ROC,M)
    """
    # ✅ Best Practice: Explicitly assign columns to DataFrame
    _roc = pd.DataFrame()
    _roc["date"] = df["date"]
    # ✅ Best Practice: Initialize an empty DataFrame for intermediate calculations
    _roc["roc"] = (df.close - df.close.shift(n)) / df.close.shift(n) * 100
    _roc["rocma"] = _ma(_roc.roc, m)
    # ✅ Best Practice: Calculate mean across specific columns for clarity
    return _roc


# ✅ Best Practice: Use descriptive column names for clarity


def vroc(df, n=12):
    """
    量变动速率
    VROC=(当日成交量-N日前的成交量)/ N日前的成交量×100%
    """
    _vroc = pd.DataFrame()
    _vroc["date"] = df["date"]
    # ✅ Best Practice: Use lambda for concise conditional logic
    # ✅ Best Practice: Use rolling window for moving calculations
    _vroc["vroc"] = (df.volume - df.volume.shift(n)) / df.volume.shift(n) * 100
    # ✅ Best Practice: Use descriptive variable names for better readability
    return _vroc


# ✅ Best Practice: Calculate final metric using clear mathematical operations


# ✅ Best Practice: Return the result DataFrame
# 🧠 ML Signal: Calculation of a financial indicator, useful for predictive models
def cr(df, n=26):
    """能量指标
    CR=∑（H-PM）/∑（PM-L）×100
    PM:上一交易日中价（(最高、最低、收盘价的均值)
    H：当天最高价
    L：当天最低价
    """
    # ✅ Best Practice: Initialize an empty DataFrame to store results, improving code organization.
    _cr = pd.DataFrame()
    _cr["date"] = df.date
    # ✅ Best Practice: Explicitly assign columns to the DataFrame for clarity and maintainability.
    # pm = ((df['high'] + df['low'] + df['close']) / 3).shift(1)
    pm = (df[["high", "low", "close"]]).mean(axis=1).shift(1)
    # ⚠️ SAST Risk (Low): Potential division by zero if df.high equals df.low.
    _cr["cr"] = (df.high - pm).rolling(n).sum() / (pm - df.low).rolling(n).sum() * 100
    # 🧠 ML Signal: Function definition for financial calculations
    # 🧠 ML Signal: Uses rolling window calculations, common in time series analysis.
    return _cr


# ✅ Best Practice: Use of helper function _ma for moving average calculation improves modularity.
# ✅ Best Practice: Return the DataFrame for further use or analysis.


def psy(df, n=12):
    """
    心理指标	PSY(12)
    PSY=N日内上涨天数/N×100
    PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N×100
    MAPSY=PSY的M日简单移动平均
    """
    # ✅ Best Practice: Initialize a new DataFrame for results
    _psy = pd.DataFrame()
    _psy["date"] = df.date
    # ✅ Best Practice: Explicitly assign columns to the new DataFrame
    p = df.close - df.close.shift()
    p[p <= 0] = np.nan
    # ⚠️ SAST Risk (Low): Potential misuse of DataFrame.shift() without checking for NaN values
    _psy["psy"] = p.rolling(n).count() / n * 100
    # ✅ Best Practice: Use descriptive column names for clarity
    return _psy


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# ⚠️ SAST Risk (Low): Potential misuse of DataFrame.shift() without checking for NaN values


# ⚠️ SAST Risk (Low): Potential misuse of DataFrame.shift() without checking for NaN values
def wad(df, n=30):
    """
    威廉聚散指标	WAD(30)
    TRL=昨日收盘价与今日最低价中价格最低者；TRH=昨日收盘价与今日最高价中价格最高者
    如果今日的收盘价>昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRL
    如果今日的收盘价<昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRH
    如果今日的收盘价=昨日的收盘价，则今日的A/D=0
    WAD=今日的A/D+昨日的WAD；MAWAD=WAD的M日简单移动平均
    # 🧠 ML Signal: Using 'date' as a key column suggests time series data, which is common in financial datasets.
    """

    def dmd(x):
        # 🧠 ML Signal: Rolling mean calculations are often used in time series analysis and financial indicators.
        # 🧠 ML Signal: Function definition with default parameter, indicating a common pattern for ML models to learn from
        if x.c > 0:
            # 🧠 ML Signal: Rolling mean calculations are often used in time series analysis and financial indicators.
            # ✅ Best Practice: Consider handling potential NaN values resulting from rolling mean calculations.
            y = x.close - x.trl
        elif x.c < 0:
            y = x.close - x.trh
        else:
            y = 0
        return y

    _wad = pd.DataFrame()
    _wad["date"] = df["date"]
    _ad = pd.DataFrame()
    _ad["trl"] = np.minimum(df.low, df.close.shift(1))
    _ad["trh"] = np.maximum(df.high, df.close.shift(1))
    _ad["c"] = df.close - df.close.shift()
    # ✅ Best Practice: Using a DataFrame to store results, which is efficient for handling tabular data
    _ad["close"] = df.close
    _ad["ad"] = _ad.apply(dmd, axis=1)
    # ✅ Best Practice: Explicitly assigning columns to the DataFrame for clarity
    _wad["wad"] = _ad.ad.expanding(1).sum()
    _wad["mawad"] = _ma(_wad.wad, n)
    # 🧠 ML Signal: Calculation of typical price, a common feature in financial data analysis
    return _wad


# 🧠 ML Signal: Rolling window operation, a common pattern in time series analysis


def mfi(df, n=14):
    """
    资金流向指标	mfi(14)
    MF＝TYP×成交量；TYP:当日中价（(最高、最低、收盘价的均值)
    如果当日TYP>昨日TYP，则将当日的MF值视为当日PMF值。而当日NMF值＝0
    如果当日TYP<=昨日TYP，则将当日的MF值视为当日NMF值。而当日PMF值=0
    MR=∑PMF/∑NMF
    MFI＝100-（100÷(1＋MR)）
    """
    # ✅ Best Practice: Using descriptive column names for clarity
    _mfi = pd.DataFrame()
    # ✅ Best Practice: Initialize an empty DataFrame with a clear purpose.
    _mfi["date"] = df.date
    # ✅ Best Practice: Using descriptive column names for clarity
    _m = pd.DataFrame()
    # ✅ Best Practice: Ensure 'df' has a 'date' column before accessing it.
    _m["typ"] = df[["high", "low", "close"]].mean(axis=1)
    # ✅ Best Practice: Using descriptive column names for clarity
    _m["mf"] = _m.typ * df.volume
    # ✅ Best Practice: Ensure 'df' has a 'volume' column before accessing it.
    # 🧠 ML Signal: Usage of moving average calculation on volume data.
    # ✅ Best Practice: Returning a DataFrame, which is a common practice for functions processing tabular data
    # ✅ Best Practice: Return a DataFrame with clear column names for better readability.
    _m["typ_shift"] = _m.typ - _m.typ.shift(1)
    _m["pmf"] = _m.apply(lambda x: x.mf if x.typ_shift > 0 else 0, axis=1)
    _m["nmf"] = _m.apply(lambda x: x.mf if x.typ_shift <= 0 else 0, axis=1)
    # _mfi['mfi'] = 100 - (100 / (1 + _m.pmf.rolling(n).sum() / _m.nmf.rolling(n).sum()))
    _m["mr"] = _m.pmf.rolling(n).sum() / _m.nmf.rolling(n).sum()
    _mfi["mfi"] = (
        100 * _m.mr / (1 + _m.mr)
    )  # 同花顺自己给出的公式和实际用的公式不一样，真操蛋，浪费两个小时时间
    return _mfi


# ✅ Best Practice: Initialize a DataFrame to store results, improving code organization and readability


# 🧠 ML Signal: Using 'date' as a key column suggests time-series data processing
def pvt(df):
    """
    pvt	量价趋势指标	pvt
    如果设x=(今日收盘价—昨日收盘价)/昨日收盘价×当日成交量，
    那么当日PVT指标值则为从第一个交易日起每日X值的累加。
    # 🧠 ML Signal: Calculation of 'macd' as a difference of 'diff' and 'dea' is a common financial analysis pattern
    """
    _pvt = pd.DataFrame()
    _pvt["date"] = df.date

    # ✅ Best Practice: Return the DataFrame to allow further processing or analysis
    x = (df.close - df.close.shift(1)) / df.close.shift(1) * df.volume
    # ✅ Best Practice: Use more descriptive variable names for better readability.
    _pvt["pvt"] = x.expanding(1).sum()
    return _pvt


# 🧠 ML Signal: Usage of moving average calculations, which are common in financial data analysis.
# ✅ Best Practice: Consider adding input validation for 'df' to ensure it contains 'amount' and 'close' columns


# ⚠️ SAST Risk (Low): Ensure _ma function is properly validated to handle edge cases like empty data.
def wvad(df, n=24, m=6):
    """# 算法是对的，同花顺计算wvad用的n=6
    威廉变异离散量	wvad(24,6)
    WVAD=N1日的∑ {(当日收盘价－当日开盘价)/(当日最高价－当日最低价)×成交量}
    MAWVAD=MA（WVAD，N2）
    # ✅ Best Practice: Initialize DataFrame with specific columns to avoid potential KeyErrors
    """
    _wvad = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Assumes 'amount' and 'close' columns exist in 'df', which may lead to KeyError
    _wvad["date"] = df.date
    # _wvad['wvad'] = (np.true_divide((df.close - df.open), (df.high - df.low)) * df.volume).rolling(n).sum()
    # 🧠 ML Signal: Usage of moving average function '_ma' indicates time series analysis
    _wvad["wvad"] = (
        (np.true_divide((df.close - df.open), (df.high - df.low)) * df.volume)
        .rolling(n)
        .sum()
    )
    _wvad["mawvad"] = _ma(_wvad.wvad, m)
    return _wvad


# ✅ Best Practice: Use a more descriptive variable name than '_vstd' for clarity.


def cdp(df):
    """
    逆势操作	cdp
    CDP=(最高价+最低价+收盘价)/3  # 同花顺实际用的(H+L+2*c)/4
    AH=CDP+(前日最高价-前日最低价)
    NH=CDP×2-最低价
    NL=CDP×2-最高价
    AL=CDP-(前日最高价-前日最低价)
    """
    _cdp = pd.DataFrame()
    _cdp["date"] = df.date
    # _cdp['cdp'] = (df.high + df.low + df.close * 2).shift(1) / 4
    _cdp["cdp"] = df[["high", "low", "close", "close"]].shift().mean(axis=1)
    _cdp["ah"] = _cdp.cdp + (df.high.shift(1) - df.low.shift())
    _cdp["al"] = _cdp.cdp - (df.high.shift(1) - df.low.shift())
    _cdp["nh"] = _cdp.cdp * 2 - df.low.shift(1)
    _cdp["nl"] = _cdp.cdp * 2 - df.high.shift(1)
    return _cdp


# ✅ Best Practice: Use of pandas DataFrame for structured data manipulation


def env(df, n=14):
    """
    ENV指标	ENV(14)
    Upper=MA(CLOSE，N)×1.06
    LOWER= MA(CLOSE，N)×0.94
    """
    # ✅ Best Practice: Use of lambda functions for concise operations
    _env = pd.DataFrame()
    _env["date"] = df.date
    _env["up"] = df.close.rolling(n).mean() * 1.06
    # ✅ Best Practice: Use of rolling window for time series analysis
    _env["low"] = df.close.rolling(n).mean() * 0.94
    return _env


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.

# ✅ Best Practice: Use of lambda functions for concise operations


def mike(df, n=12):
    """
    麦克指标	mike(12)
    初始价（TYP）=（当日最高价＋当日最低价＋当日收盘价）/3
    HV=N日内区间最高价
    LV=N日内区间最低价
    初级压力线（WR）=TYP×2-LV
    中级压力线（MR）=TYP+HV-LV
    强力压力线（SR）=2×HV-LV
    初级支撑线（WS）=TYP×2-HV
    中级支撑线（MS）=TYP-HV+LV
    强力支撑线（SS）=2×LV-HV
    """
    _mike = pd.DataFrame()
    _mike["date"] = df.date
    typ = df[["high", "low", "close"]].mean(axis=1)
    # ✅ Best Practice: Use more descriptive variable names for better readability.
    hv = df.high.rolling(n).max()
    lv = df.low.rolling(n).min()
    _mike["wr"] = typ * 2 - lv
    # 🧠 ML Signal: Usage of time series data operations, such as shift, can be a signal for financial data analysis.
    _mike["mr"] = typ + hv - lv
    _mike["sr"] = 2 * hv - lv
    # 🧠 ML Signal: Custom implementation of moving average (sma) can indicate specific domain logic.
    _mike["ws"] = typ * 2 - hv
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _mike["ms"] = typ - hv + lv
    # 🧠 ML Signal: Use of custom moving average function (_ma) can indicate specific domain logic.
    # 🧠 ML Signal: Repeated use of sma function suggests a pattern in data smoothing or trend analysis.
    _mike["ss"] = 2 * lv - hv
    return _mike


def vma(df, n=5):
    """
    量简单移动平均	VMA(5)	VMA=MA(volume,N)
    VOLUME表示成交量；N表示天数
    # 🧠 ML Signal: Accessing DataFrame columns, indicating a pattern of data processing.
    """
    _vma = pd.DataFrame()
    # 🧠 ML Signal: Calculation involving shifting data, a common pattern in time series analysis.
    _vma["date"] = df.date
    # 🧠 ML Signal: Usage of a custom function 'sma', indicating a pattern of applying statistical methods.
    # ✅ Best Practice: Returning a DataFrame, which is a common practice for functions processing tabular data.
    _vma["vma"] = _ma(df.volume, n)
    return _vma


def vmacd(df, qn=12, sn=26, m=9):
    """
    量指数平滑异同平均	vmacd(12,26,9)
    今日EMA（N）=2/（N+1）×今日成交量+(N-1)/（N+1）×昨日EMA（N）
    DIFF= EMA（N1）- EMA（N2）
    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
    MACD（BAR）=2×（DIF-DEA）
    # 🧠 ML Signal: Calculating rate of change is a common pattern in financial analysis.
    """
    _vmacd = pd.DataFrame()
    # ⚠️ SAST Risk (Low): Ensure 'sma' function is defined and handles edge cases like NaN values.
    _vmacd["date"] = df.date
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _vmacd["diff"] = _ema(df.volume, qn) - _ema(df.volume, sn)
    # ⚠️ SAST Risk (Low): Ensure '_ma' function is defined and handles edge cases like NaN values.
    _vmacd["dea"] = _ema(_vmacd["diff"], m)  # TODO: 不能用_vmacd.diff, 不知道为什么
    _vmacd["macd"] = _vmacd["diff"] - _vmacd["dea"]
    return _vmacd


def vosc(df, n=12, m=26):
    """
    成交量震荡	vosc(12,26)
    VOSC=（MA（VOLUME,SHORT）- MA（VOLUME,LONG））/MA（VOLUME,SHORT）×100
    """
    _c = pd.DataFrame()
    _c["date"] = df["date"]
    # ✅ Best Practice: Check if 'n' is within a valid range to prevent potential errors.
    _c["osc"] = (_ma(df.volume, n) - _ma(df.volume, m)) / _ma(df.volume, n) * 100
    return _c


# 🧠 ML Signal: Use of lambda function for element-wise operations on DataFrame.


def tapi(df, n=6):
    """# TODO: 由于get_k_data返回数据中没有amount，可以用get_h_data中amount，算法是正确的
    加权指数成交值	tapi(6)
    TAPI=每日成交总值/当日加权指数=a/PI；A表示每日的成交金额，PI表示当天的股价指数即指收盘价
    """
    # ✅ Best Practice: Use of copy to avoid modifying the original dataframe
    _tapi = pd.DataFrame()
    # _tapi['date'] = df.date
    # ✅ Best Practice: Setting index for efficient data manipulation
    _tapi["tapi"] = df.amount / df.close
    _tapi["matapi"] = _ma(_tapi.tapi, n)
    # ✅ Best Practice: Setting index for efficient data manipulation
    return _tapi


# ✅ Best Practice: Initializing a DataFrame with a specific index


def vstd(df, n=10):
    """
    成交量标准差	vstd(10)
    VSTD=STD（Volume,N）=[∑（Volume-MA(Volume，N)）^2/N]^0.5
    """
    # 🧠 ML Signal: Use of lambda for conditional logic, a common pattern in data processing
    _vstd = pd.DataFrame()
    _vstd["date"] = df.date
    _vstd["vstd"] = df.volume.rolling(n).std(ddof=1)
    return _vstd


# 🧠 ML Signal: Rolling window calculation, often used in time series analysis
# ⚠️ SAST Risk (Medium): Using `ts.get_k_data` without input validation can lead to potential data integrity issues.

# ✅ Best Practice: Dropping unnecessary columns to save memory


# ✅ Best Practice: Use of `copy()` to avoid modifying the original DataFrame.
def adtm(df, n=23, m=8):
    """
    动态买卖气指标	adtm(23,8)
    如果开盘价≤昨日开盘价，DTM=0
    如果开盘价＞昨日开盘价，DTM=(最高价-开盘价)和(开盘价-昨日开盘价)的较大值
    如果开盘价≥昨日开盘价，DBM=0
    如果开盘价＜昨日开盘价，DBM=(开盘价-最低价)
    STM=DTM在N日内的和
    SBM=DBM在N日内的和
    如果STM > SBM,ADTM=(STM-SBM)/STM
    如果STM < SBM , ADTM = (STM-SBM)/SBM
    如果STM = SBM,ADTM=0
    ADTMMA=MA(ADTM,M)
    # ✅ Best Practice: Using `apply` with a lambda for row-wise operations.
    """
    # ✅ Best Practice: Assigning NaN to irrelevant data points for clarity.
    _adtm = pd.DataFrame()
    _adtm["date"] = df.date
    _m = pd.DataFrame()
    _m["cc"] = df.open - df.open.shift(1)
    # ⚠️ SAST Risk (Medium): Use of external library 'ts' without import statement
    # 🧠 ML Signal: Use of rolling window calculations, a common pattern in time series analysis.
    _m["ho"] = df.high - df.open
    _m["ol"] = df.open - df.low
    # ✅ Best Practice: Use of copy to avoid modifying the original DataFrame
    # ✅ Best Practice: Dropping intermediate columns to clean up the DataFrame.
    _m["dtm"] = _m.apply(lambda x: max(x.ho, x.cc) if x.cc > 0 else 0, axis=1)
    _m["dbm"] = _m.apply(lambda x: x.ol if x.cc < 0 else 0, axis=1)
    # ✅ Best Practice: Resetting index to return a clean DataFrame.
    # ✅ Best Practice: Setting 'date' as index for easier time-based operations
    _m["stm"] = _m.dtm.rolling(n).sum()
    _m["sbm"] = _m.dbm.rolling(n).sum()
    # ✅ Best Practice: Setting 'date' as index for easier time-based operations
    _m["ss"] = _m.stm - _m.sbm
    _adtm["adtm"] = _m.apply(
        lambda x: x.ss / x.stm if x.ss > 0 else (x.ss / x.sbm if x.ss < 0 else 0),
        axis=1,
    )
    # ✅ Best Practice: Initializing DataFrame with index for alignment
    _adtm["adtmma"] = _ma(_adtm.adtm, m)
    return _adtm


# 🧠 ML Signal: Calculation of difference between close and open prices


# 🧠 ML Signal: Calculation of difference between close and open prices
def mi(df, n=12):
    """
    动量指标	mi(12)
    A=CLOSE-REF(CLOSE,N)
    MI=SMA(A,N,1)
    """
    _mi = pd.DataFrame()
    _mi["date"] = df.date
    _mi["mi"] = sma(df.close - df.close.shift(n), n)
    return _mi


# ✅ Best Practice: Dropping intermediate calculation columns to save memory
# 🧠 ML Signal: Rolling window calculation for time series analysis

# ✅ Best Practice: Use more descriptive variable names for better readability.


# ✅ Best Practice: Resetting index to return a clean DataFrame
def micd(df, n=3, m=10, k=20):
    """
    异同离差动力指数	micd(3,10,20)
    MI=CLOSE-ref(CLOSE,1)AMI=SMA(MI,N1,1)
    DIF=MA(ref(AMI,1),N2)-MA(ref(AMI,1),N3)
    MICD=SMA(DIF,10,1)
    """
    _micd = pd.DataFrame()
    # 🧠 ML Signal: Usage of rolling window operations, which are common in time series analysis.
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _micd["date"] = df.date
    # 🧠 ML Signal: Calculation of moving averages, a common pattern in financial data analysis.
    mi = df.close - df.close.shift(1)
    ami = sma(mi, n)
    dif = _ma(ami.shift(1), m) - _ma(ami.shift(1), k)
    _micd["micd"] = sma(dif, m)
    return _micd


# ✅ Best Practice: Use more descriptive variable names for better readability.


def rc(df, n=50):
    """
    变化率指数	rc(50)
    RC=收盘价/REF（收盘价，N）×100
    ARC=EMA（REF（RC，1），N，1）
    """
    _rc = pd.DataFrame()
    _rc["date"] = df.date
    _rc["rc"] = df.close / df.close.shift(n) * 100
    _rc["arc"] = sma(_rc.rc.shift(1), n)
    return _rc


# ✅ Best Practice: Use more descriptive variable names for better readability.


def rccd(df, n=59, m=21, k=28):
    """# TODO: 计算结果错误和同花顺不同，检查不出来为什么
    异同离差变化率指数 rate of change convergence divergence	rccd(59,21,28)
    RC=收盘价/REF（收盘价，N）×100%
    ARC=EMA(REF(RC,1),N,1)
    DIF=MA(ref(ARC,1),N1)-MA MA(ref(ARC,1),N2)
    RCCD=SMA(DIF,N,1)
    """
    _rccd = pd.DataFrame()
    _rccd["date"] = df.date
    # ✅ Best Practice: Initialize a new DataFrame for storing results
    rc = df.close / df.close.shift(n) * 100
    arc = sma(rc.shift(), n)
    # ✅ Best Practice: Explicitly assign columns to the DataFrame
    dif = _ma(arc.shift(), m) - _ma(arc.shift(), k)
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    _rccd["rccd"] = sma(dif, n)
    # 🧠 ML Signal: Use of rolling window operations on time series data
    # ⚠️ SAST Risk (Low): Assumes 'df' has 'close' and 'date' columns without validation
    return _rccd


def srmi(df, n=9):
    """
    SRMIMI修正指标	srmi(9)
    如果收盘价>N日前的收盘价，SRMI就等于（收盘价-N日前的收盘价）/收盘价
    如果收盘价<N日前的收盘价，SRMI就等于（收盘价-N日签的收盘价）/N日前的收盘价
    如果收盘价=N日前的收盘价，SRMI就等于0
    # ✅ Best Practice: Consider adding input validation for the 'df' parameter to ensure it contains the expected columns.
    """
    # ⚠️ SAST Risk (Low): Check for division by zero when p is zero to prevent runtime errors.
    _srmi = pd.DataFrame()
    _srmi["date"] = df.date
    _m = pd.DataFrame()
    _m["close"] = df.close
    # ✅ Best Practice: Initialize DataFrame with specific columns to avoid potential KeyError.
    _m["cp"] = df.close.shift(n)
    _m["cs"] = df.close - df.close.shift(n)
    _srmi["srmi"] = _m.apply(
        lambda x: x.cs / x.close if x.cs > 0 else (x.cs / x.cp if x.cs < 0 else 0),
        axis=1,
    )
    # ⚠️ SAST Risk (Low): Potential for KeyError if 'close' column is missing in 'df'.
    return _srmi


def dptb(df, n=7):
    """
    大盘同步指标	dptb(7)
    DPTB=（统计N天中个股收盘价>开盘价，且指数收盘价>开盘价的天数或者个股收盘价<开盘价，且指数收盘价<开盘价）/N
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index("date", inplace=True)  # 可能出现停盘等情况，所以将date设为index
    ind.set_index("date", inplace=True)
    # ✅ Best Practice: Include import statements for used libraries (e.g., pandas, itertools)
    _dptb = pd.DataFrame(index=df.date)
    q = ind.close - ind.open
    _dptb["p"] = sd.close - sd.open
    _dptb["q"] = q
    _dptb["m"] = _dptb.apply(
        lambda x: 1 if (x.p > 0 and x.q > 0) or (x.p < 0 and x.q < 0) else np.nan,
        axis=1,
    )
    _dptb["jdrs"] = _dptb.m.rolling(n).count() / n
    _dptb.drop(columns=["p", "q", "m"], inplace=True)
    # 🧠 ML Signal: Usage of DataFrame operations to calculate differences
    _dptb.reset_index(inplace=True)
    return _dptb


# 🧠 ML Signal: Use of list to accumulate results
def jdqs(df, n=20):
    """
    阶段强势指标	jdqs(20)
    JDQS=（统计N天中个股收盘价>开盘价，且指数收盘价<开盘价的天数）/（统计N天中指数收盘价<开盘价的天数）
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index("date", inplace=True)  # 可能出现停盘等情况，所以将date设为index
    ind.set_index("date", inplace=True)
    _jdrs = pd.DataFrame(index=df.date)
    # ✅ Best Practice: Use of join with set_index for merging dataframes on a specific column
    q = ind.close - ind.open
    _jdrs["p"] = sd.close - sd.open
    _jdrs["q"] = q
    _jdrs["m"] = _jdrs.apply(lambda x: 1 if (x.p > 0 and x.q < 0) else np.nan, axis=1)
    # ✅ Best Practice: Importing libraries within the main guard to avoid unnecessary imports
    # 🧠 ML Signal: Fetching stock data using tushare API, indicating financial data analysis
    # ⚠️ SAST Risk (High): Calling an undefined function 'rccd' will raise a NameError
    q[q > 0] = np.nan
    _jdrs["t"] = q
    _jdrs["jdrs"] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=["p", "q", "m", "t"], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs


def jdrs(df, n=20):
    """
    阶段弱势指标	jdrs(20)
    JDRS=（统计N天中个股收盘价<开盘价，且指数收盘价>开盘价的天数）/（统计N天中指数收盘价>开盘价的天数）
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index("date", inplace=True)
    ind.set_index("date", inplace=True)
    _jdrs = pd.DataFrame(index=df.date)
    q = ind.close - ind.open
    _jdrs["p"] = sd.close - sd.open
    _jdrs["q"] = q
    _jdrs["m"] = _jdrs.apply(lambda x: 1 if (x.p < 0 and x.q > 0) else np.nan, axis=1)
    q[q < 0] = np.nan
    _jdrs["t"] = q
    _jdrs["jdrs"] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=["p", "q", "m", "t"], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs


def zdzb(df, n=125, m=5, k=20):
    """
    筑底指标	zdzb(125,5,20)
    A=（统计N1日内收盘价>=前收盘价的天数）/（统计N1日内收盘价<前收盘价的天数）
    B=MA（A,N2）
    D=MA（A，N3）
    """
    _zdzb = pd.DataFrame()
    _zdzb["date"] = df.date
    p = df.close - df.close.shift(1)
    q = p.copy()
    p[p < 0] = np.nan
    q[q >= 0] = np.nan
    _zdzb["a"] = p.rolling(n).count() / q.rolling(n).count()
    _zdzb["b"] = _zdzb.a.rolling(m).mean()
    _zdzb["d"] = _zdzb.a.rolling(k).mean()
    return _zdzb


def atr(df, n=14):
    """
    真实波幅	atr(14)
    TR:MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
    ATR:MA(TR,N)
    """
    _atr = pd.DataFrame()
    _atr["date"] = df.date
    # _atr['tr'] = np.maximum(df.high - df.low, (df.close.shift(1) - df.low).abs())
    # _atr['tr'] = np.maximum.reduce([df.high - df.low, (df.close.shift(1) - df.high).abs(), (df.close.shift(1) - df.low).abs()])
    _atr["tr"] = np.vstack(
        [
            df.high - df.low,
            (df.close.shift(1) - df.high).abs(),
            (df.close.shift(1) - df.low).abs(),
        ]
    ).max(axis=0)
    _atr["atr"] = _atr.tr.rolling(n).mean()
    return _atr


def mass(df, n=9, m=25):
    """
    梅丝线	mass(9,25)
    AHL=MA(（H-L）,N1)
    BHL= MA（AHL，N1）
    MASS=SUM（AHL/BHL，N2）
    H：表示最高价；L：表示最低价
    """
    _mass = pd.DataFrame()
    _mass["date"] = df.date
    ahl = _ma((df.high - df.low), n)
    bhl = _ma(ahl, n)
    _mass["mass"] = (ahl / bhl).rolling(m).sum()
    return _mass


def vhf(df, n=28):
    """
    纵横指标	vhf(28)
    VHF=（N日内最大收盘价与N日内最小收盘价之前的差）/（N日收盘价与前收盘价差的绝对值之和）
    """
    _vhf = pd.DataFrame()
    _vhf["date"] = df.date
    _vhf["vhf"] = (df.close.rolling(n).max() - df.close.rolling(n).min()) / (
        df.close - df.close.shift(1)
    ).abs().rolling(n).sum()
    return _vhf


def cvlt(df, n=10):
    """
    佳庆离散指标	cvlt(10)
    cvlt=（最高价与最低价的差的指数移动平均-前N日的最高价与最低价的差的指数移动平均）/前N日的最高价与最低价的差的指数移动平均
    """
    _cvlt = pd.DataFrame()
    _cvlt["date"] = df.date
    p = _ema(df.high.shift(n) - df.low.shift(n), n)
    _cvlt["cvlt"] = (_ema(df.high - df.low, n) - p) / p * 100
    return _cvlt


def up_n(df):
    """
    连涨天数	up_n	连续上涨天数，当天收盘价大于开盘价即为上涨一天 # 同花顺实际结果用收盘价-前一天收盘价
    """
    _up = pd.DataFrame()
    _up["date"] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 1
    p[p < 0] = 0
    m = []
    for k, g in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    # _up['p'] = p
    _up["up"] = m
    return _up


def down_n(df):
    """
    连跌天数	down_n	连续下跌天数，当天收盘价小于开盘价即为下跌一天
    """
    _down = pd.DataFrame()
    _down["date"] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 0
    p[p < 0] = 1
    m = []
    for k, g in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    _down["down"] = m
    return _down


def join_frame(d1, d2, column="date"):
    # 将两个DataFrame 按照datetime合并
    return d1.join(d2.set_index(column), on=column)


if __name__ == "__main__":
    import tushare as ts

    # data = ts.get_k_data("000063", start="2017-05-01")
    data = ts.get_k_data("601138", start="2017-05-01")
    # print(data)
    # maf = ma(data, n=[5, 10, 20])
    # 将均线合并到data中
    # print(join_frame(data, maf))

    # data = pd.DataFrame({"close": [1,2,3,4,5,6,7,8,9,0]})
    # print(ma(data))
    # mdf = md(data)
    # print(md(data, n=26))
    # print(join_frame(data, mdf))
    # emaf = ema(data)
    # print(ema(data, 5))
    # print(join_frame(data, emaf))
    # print(macd(data))
    # print(kdj(data))
    # print(vrsi(data, 6))
    # print(boll(data))
    # print(bbiboll(data))
    # print(wr(data))
    # print(bias(data))
    # print(asi(data))
    # print(vr_rate(data))
    # print(vr(data))
    # print(arbr(data))
    # print(dpo(data))
    # print(trix(data))
    # print(bbi(data))
    # print(ts.top_list(date="2019-01-17"))
    # print(mtm(data))
    # print(obv(data))
    # print(cci(data))
    # print(priceosc(data))
    # print(dbcd(data))
    # print(roc(data))
    # print(vroc(data))
    # print(cr(data))
    # print(psy(data))
    # print(wad(data))
    # print(mfi(data))
    # print(pvt(data))
    # print(wvad(data))
    # print(cdp(data))
    # print(env(data))
    # print(mike(data))
    # print(vr(data))
    # print(vma(data))
    # print(vmacd(data))
    # print(vosc(data))
    # print(tapi(data))
    # print(vstd(data))
    # print(adtm(data))
    # print(mi(data))
    # print(micd(data))
    # print(rc(data))
    print(rccd(data))
    # print(srmi(data))
    # print(dptb(data))
    # print(jdqs(data))
    # pd.set_option('display.max_rows', 1000)
    # print(jdrs(data))
    # print(join_frame(data, jdrs(data)))
    # print(data)
    # print(zdzb(data))
    # print(atr(data))
    # print(mass(data))
    # print(vhf(data))
    # print(cvlt(data))
    # print(up_n(data))
    # print(down_n(data))
