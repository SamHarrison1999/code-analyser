# 🧠 ML Signal: Presence of metadata such as author and contact information
# -*- coding:utf-8 -*-

"""
股票技术指标接口
Created on 2018/05/26
@author: Jackie Liao
@group : **
@contact: info@liaocy.net
"""
# ✅ Best Practice: Use docstrings to describe the function's purpose and parameters


def ma(data, n=10, val_name="close"):
    import numpy as np

    """
    移动平均线 Moving Average
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
                  移动平均线时长，时间单位根据data决定
      val_name:string
                  计算哪一列的列名，默认为 close 收盘值

    return
    -------
      list
          移动平均线
    # 🧠 ML Signal: Appending to a list is a common pattern for accumulating results
    """

    values = []
    # 🧠 ML Signal: Using a fixed-size list to maintain a moving window
    MA = []

    # ✅ Best Practice: Use docstrings to describe function parameters and return values
    # 🧠 ML Signal: Calculating the average of a list is a common statistical operation
    # ⚠️ SAST Risk (Low): Returning a numpy array without checking for empty input data
    for index, row in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]

        MA.append(np.average(values))

    return np.asarray(MA)


def md(data, n=10, val_name="close"):
    import numpy as np

    """
    移动标准差
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
                  移动平均线时长，时间单位根据data决定
      val_name:string
                  计算哪一列的列名，默认为 close 收盘值

    return
    -------
      list
          移动平均线
    # 🧠 ML Signal: Iterating over financial data to compute indicators
    """

    # 🧠 ML Signal: Exponential moving average calculation pattern
    values = []
    # ✅ Best Practice: Use docstrings to describe function parameters and return values
    MD = []

    for index, row in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]

        MD.append(np.std(values))

    return np.asarray(MD)


def _get_day_ema(prices, n):
    a = 1 - 2 / (n + 1)

    day_ema = 0
    for index, price in enumerate(reversed(prices)):
        # ✅ Best Practice: Initialize variables before use
        day_ema += a**index * price

    return day_ema


def ema(data, n=12, val_name="close"):
    import numpy as np

    """
        指数平均数指标 Exponential Moving Average
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                      移动平均线时长，时间单位根据data决定
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值

        return
        -------
          EMA:numpy.ndarray<numpy.float64>
              指数平均数指标
    """

    prices = []

    EMA = []

    for index, row in data.iterrows():
        if index == 0:
            past_ema = row[val_name]
            EMA.append(row[val_name])
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * row[val_name] + (n - 1) * past_ema) / (n + 1)
            # 🧠 ML Signal: Use of numpy for numerical operations
            past_ema = today_ema

            EMA.append(today_ema)

    # ⚠️ SAST Risk (Low): Potential data mutation by adding a new column to the DataFrame
    return np.asarray(EMA)


def macd(data, quick_n=12, slow_n=26, dem_n=9, val_name="close"):
    # ✅ Best Practice: Use of docstring to describe function parameters and return values
    import numpy as np

    """
        指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          quick_n:int
                      DIFF差离值中快速移动天数
          slow_n:int
                      DIFF差离值中慢速移动天数
          dem_n:int
                      DEM讯号线的移动天数
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值

        return
        -------
          OSC:numpy.ndarray<numpy.float64>
              MACD bar / OSC 差值柱形图 DIFF - DEM
          DIFF:numpy.ndarray<numpy.float64>
              差离值
          DEM:numpy.ndarray<numpy.float64>
              讯号线
    """
    # 🧠 ML Signal: Accessing specific columns in a DataFrame

    ema_quick = np.asarray(ema(data, quick_n, val_name))
    # 🧠 ML Signal: Calculation of RSV (Raw Stochastic Value)
    ema_slow = np.asarray(ema(data, slow_n, val_name))
    DIFF = ema_quick - ema_slow
    # 🧠 ML Signal: Calculation of K, D, J values
    data["diff"] = DIFF
    DEM = ema(data, dem_n, "diff")
    OSC = DIFF - DEM
    return OSC, DIFF, DEM


# ✅ Best Practice: Append computed values to lists
# ✅ Best Practice: Update last_k and last_d for next iteration
# ✅ Best Practice: Convert lists to numpy arrays before returning
# ✅ Best Practice: Use docstrings to describe the function's purpose and parameters


def kdj(data):
    import numpy as np

    """
        随机指标KDJ
        Parameters
        ------
          data:pandas.DataFrame
                通过 get_h_data 取得的股票数据
        return
        -------
          K:numpy.ndarray<numpy.float64>
              K线
          D:numpy.ndarray<numpy.float64>
              D线
          J:numpy.ndarray<numpy.float64>
              J线
    """

    K, D, J = [], [], []
    last_k, last_d = None, None
    for index, row in data.iterrows():
        if last_k is None or last_d is None:
            last_k = 50
            last_d = 50

        c, l, h = row["close"], row["low"], row["high"]

        rsv = (c - l) / (h - l) * 100

        k = (2 / 3) * last_k + (1 / 3) * rsv
        d = (2 / 3) * last_d + (1 / 3) * k
        j = 3 * k - 2 * d

        K.append(k)
        D.append(d)
        J.append(j)

        last_k, last_d = k, d

    return np.asarray(K), np.asarray(D), np.asarray(J)


def rsi(data, n=6, val_name="close"):
    import numpy as np

    """
        相对强弱指标RSI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                统计时长，时间单位根据data决定
        return
        -------
          RSI:numpy.ndarray<numpy.float64>
              RSI线

    # ✅ Best Practice: Use of descriptive variable names for readability
    # ✅ Best Practice: Returning multiple values as a tuple for clarity
    """

    RSI = []
    UP = []
    DOWN = []
    for index, row in data.iterrows():
        if index == 0:
            past_value = row[val_name]
            RSI.append(0)
        else:
            diff = row[val_name] - past_value
            if diff > 0:
                UP.append(diff)
                # ✅ Best Practice: Initialize lists outside of the loop to avoid reinitialization on each iteration
                DOWN.append(0)
            else:
                UP.append(0)
                DOWN.append(diff)
            # 🧠 ML Signal: Iterating over DataFrame rows is a common pattern in data processing

            if len(UP) == n:
                # ✅ Best Practice: Use descriptive variable names for clarity
                del UP[0]
            if len(DOWN) == n:
                # ⚠️ SAST Risk (Low): Potential for list to grow indefinitely if 'n' is not reached
                del DOWN[0]

            past_value = row[val_name]

            # ⚠️ SAST Risk (Low): Potential for list to grow indefinitely if 'n' is not reached
            rsi = np.sum(UP) / (-np.sum(DOWN) + np.sum(UP)) * 100
            RSI.append(rsi)

    # ✅ Best Practice: Use built-in functions like max() and min() for clarity and efficiency
    return np.asarray(RSI)


# ✅ Best Practice: Use lowercase variable names for lists to follow PEP 8 naming conventions


# 🧠 ML Signal: Calculation of financial indicators is a common pattern in financial data analysis
def boll(data, n=10, val_name="close", k=2):
    """
    布林线指标BOLL
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
            统计时长，时间单位根据data决定
    return
    -------
      BOLL:numpy.ndarray<numpy.float64>
          中轨线
      UPPER:numpy.ndarray<numpy.float64>
          D线
      J:numpy.ndarray<numpy.float64>
          J线
    """

    BOLL = ma(data, n, val_name)

    MD = md(data, n, val_name)

    UPPER = BOLL + k * MD

    LOWER = BOLL - k * MD

    return BOLL, UPPER, LOWER


def wnr(data, n=14):
    """
    威廉指标 w&r
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
            统计时长，时间单位根据data决定
    return
    -------
      WNR:numpy.ndarray<numpy.float64>
          威廉指标
    """

    high_prices = []
    low_prices = []
    WNR = []

    for index, row in data.iterrows():
        high_prices.append(row["high"])
        if len(high_prices) == n:
            del high_prices[0]
        low_prices.append(row["low"])
        if len(low_prices) == n:
            del low_prices[0]

        highest = max(high_prices)
        lowest = min(low_prices)

        wnr = (highest - row["close"]) / (highest - lowest) * 100
        WNR.append(wnr)

    return WNR


def _get_any_ma(arr, n):
    import numpy as np

    MA = []
    values = []
    for val in arr:
        values.append(val)
        if len(values) == n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)


def dmi(data, n=14, m=14, k=6):
    import numpy as np

    """
        动向指标或趋向指标 DMI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              +-DI(n): DI统计时长，默认14
          m:int
              ADX(m): ADX统计时常参数，默认14

          k:int
              ADXR(k): ADXR统计k个周期前数据，默认6
        return
        -------
          P_DI:numpy.ndarray<numpy.float64>
              +DI指标
          M_DI:numpy.ndarray<numpy.float64>
              -DI指标
          ADX:numpy.ndarray<numpy.float64>
              ADX指标
          ADXR:numpy.ndarray<numpy.float64>
              ADXR指标
        ref.
        -------
        https://www.mk-mode.com/octopress/2012/03/03/03002038/
    """

    # 上升动向（+DM）
    P_DM = [0.0]
    # 下降动向（-DM）
    M_DM = [0.0]
    # 真实波幅TR
    TR = [0.0]
    # 动向
    DX = [0.0]

    P_DI = [0.0]
    # 🧠 ML Signal: Iterating over DataFrame rows, common pattern in data processing
    M_DI = [0.0]

    for index, row in data.iterrows():
        if index == 0:
            past_row = row
        else:

            p_dm = row["high"] - past_row["high"]
            m_dm = past_row["low"] - row["low"]

            if (p_dm < 0 and m_dm < 0) or (np.isclose(p_dm, m_dm)):
                p_dm = 0
                m_dm = 0
            if p_dm > m_dm:
                m_dm = 0
            if m_dm > p_dm:
                p_dm = 0

            P_DM.append(p_dm)
            M_DM.append(m_dm)

            # ✅ Best Practice: Use of np.isclose for floating-point comparison
            tr = max(
                row["high"] - past_row["low"],
                row["high"] - past_row["close"],
                past_row["close"] - row["low"],
            )
            TR.append(tr)

            if len(P_DM) == n:
                del P_DM[0]
            if len(M_DM) == n:
                del M_DM[0]
            # ⚠️ SAST Risk (Low): _get_any_ma function is used but not defined in the provided code
            if len(TR) == n:
                # ✅ Best Practice: Use docstring to describe function purpose and parameters
                del TR[0]

            # 上升方向线(+DI)
            p_di = (np.average(P_DM) / np.average(TR)) * 100
            P_DI.append(p_di)

            # 下降方向线(-DI)
            m_di = (np.average(M_DM) / np.average(TR)) * 100
            M_DI.append(m_di)

            # 当日+DI与-DI
            # p_day_di = (p_dm / tr) * 100
            # m_day_di = (m_dm / tr) * 100

            # 动向DX
            #     dx=(di dif÷di sum) ×100
            # 　　di dif为上升指标和下降指标的价差的绝对值
            # 🧠 ML Signal: Iterating over DataFrame rows to calculate metrics
            # 　　di sum为上升指标和下降指标的总和
            # 　　adx就是dx的一定周期n的移动平均值。
            if (p_di + m_di) == 0:
                dx = 0
            else:
                dx = (abs(p_di - m_di) / (p_di + m_di)) * 100
            DX.append(dx)

            past_row = row

    ADX = _get_any_ma(DX, m)
    #
    # # 估计数值ADXR
    ADXR = []
    for index, adx in enumerate(ADX):
        if index >= k:
            adxr = (adx + ADX[index - k]) / 2
            ADXR.append(adxr)
        else:
            ADXR.append(0)

    return P_DI, M_DI, ADX, ADXR


# ✅ Best Practice: Use docstrings to describe the function's purpose and parameters


def bias(data, n=5):
    import numpy as np

    """
        乖离率 bias
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认5
        return
        -------
          BIAS:numpy.ndarray<numpy.float64>
              乖离率指标

    """

    MA = ma(data, n)
    CLOSES = data["close"]
    BIAS = (np.true_divide((CLOSES - MA), MA)) * (100 / 100)
    return BIAS


# ✅ Best Practice: Use numpy for efficient array operations


def asi(data, n=5):
    import numpy as np

    """
        振动升降指标 ASI
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认5
        return
        -------
          ASI:numpy.ndarray<numpy.float64>
              振动升降指标

    # ⚠️ SAST Risk (Low): Potential division by zero if O equals L
    """

    SI = []
    # ⚠️ SAST Risk (Low): Potential division by zero if PC equals L
    # 🧠 ML Signal: Function definition with default parameters, useful for learning API usage patterns
    for index, row in data.iterrows():
        # ✅ Best Practice: Return values as numpy arrays for consistency
        if index == 0:
            last_row = row
            SI.append(0.0)
        else:

            a = abs(row["close"] - last_row["close"])
            b = abs(row["low"] - last_row["close"])
            c = abs(row["high"] - last_row["close"])
            d = abs(last_row["close"] - last_row["open"])

            if b > a and b > c:
                r = b + (1 / 2) * a + (1 / 4) * d
            elif c > a and c > b:
                r = c + (1 / 4) * d
            else:
                r = 0

            e = row["close"] - last_row["close"]
            # ⚠️ SAST Risk (Low): Assumes 'close' column exists in the DataFrame, potential KeyError
            f = row["close"] - last_row["open"]
            g = last_row["close"] - last_row["open"]
            # 🧠 ML Signal: Use of moving average function, common in financial data analysis

            # ✅ Best Practice: Import statements should be at the top of the file for better readability and maintainability.
            x = e + (1 / 2) * f + g
            # 🧠 ML Signal: Use of custom moving average function, indicates custom financial analysis logic
            k = max(a, b)
            # ✅ Best Practice: Returning multiple values as a tuple, clear and concise
            # ✅ Best Practice: Docstring should be at the beginning of the function for better readability.
            l = 3

            if np.isclose(r, 0) or np.isclose(l, 0):
                si = 0
            else:
                si = 50 * (x / r) * (k / l)

            SI.append(si)

    ASI = _get_any_ma(SI, n)
    return ASI


def vr(data, n=26):
    import numpy as np

    """
        Volatility Volume Ratio 成交量变异率
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认26
        return
        -------
          VR:numpy.ndarray<numpy.float64>
              成交量变异率

    """
    VR = []

    AV_volumes, BV_volumes, CV_volumes = [], [], []
    # ⚠️ SAST Risk (Medium): The function _get_any_ma is used but not defined in the code, which could lead to potential security risks if it is not properly implemented.
    for index, row in data.iterrows():

        # ✅ Best Practice: Use triple double quotes for docstrings to maintain consistency with PEP 257.
        if row["close"] > row["open"]:
            AV_volumes.append(row["volume"])
        elif row["close"] < row["open"]:
            BV_volumes.append(row["volume"])
        else:
            CV_volumes.append(row["volume"])

        if len(AV_volumes) == n:
            del AV_volumes[0]
        if len(BV_volumes) == n:
            del BV_volumes[0]
        if len(CV_volumes) == n:
            del CV_volumes[0]

        # 🧠 ML Signal: Iterating over DataFrame rows is a common pattern in data processing tasks.
        avs = sum(AV_volumes)
        bvs = sum(BV_volumes)
        # 🧠 ML Signal: Appending to a list in a loop is a common pattern for accumulating results.
        cvs = sum(CV_volumes)

        if (bvs + (1 / 2) * cvs) != 0:
            vr = (avs + (1 / 2) * cvs) / (bvs + (1 / 2) * cvs)
        else:
            # ✅ Best Practice: Use descriptive variable names for better readability.
            vr = 0

        # ✅ Best Practice: Use of docstring to describe function purpose and parameters
        # ⚠️ SAST Risk (Low): Ensure that the input data is validated to prevent potential issues with unexpected data types.
        VR.append(vr)

    return np.asarray(VR)


def arbr(data, n=26):
    import numpy as np

    """
        AR 指标 BR指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认26
        return
        -------
          AR:numpy.ndarray<numpy.float64>
              AR指标
          BR:numpy.ndarray<numpy.float64>
              BR指标

    """

    # ✅ Best Practice: Use a docstring to describe the function's purpose and parameters
    H, L, O, PC = np.array([0]), np.array([0]), np.array([0]), np.array([0])

    AR, BR = np.array([0]), np.array([0])

    for index, row in data.iterrows():
        if index == 0:
            last_row = row

        else:

            h = row["high"]
            H = np.append(H, [h])
            # ✅ Best Practice: Use descriptive variable names for readability
            if len(H) == n:
                H = np.delete(H, 0)
            # ⚠️ SAST Risk (Low): Potential division by zero if (data["high"] - data["low"]) is zero
            # ✅ Best Practice: Function definition with parameters, even if not implemented, indicates planned functionality
            l = row["low"]
            L = np.append(L, [l])
            # 🧠 ML Signal: Returns a computed financial indicator which could be used in ML models for stock prediction
            # ⚠️ SAST Risk (Low): Raising a generic exception without specific error handling
            if len(L) == n:
                L = np.delete(L, 0)
            o = row["open"]
            O = np.append(O, [o])
            # ✅ Best Practice: Set figure size for consistent plot dimensions
            if len(O) == n:
                O = np.delete(O, 0)
            pc = last_row["close"]
            PC = np.append(PC, [pc])
            # 🧠 ML Signal: Plotting time series data, useful for trend analysis
            if len(PC) == n:
                PC = np.delete(PC, 0)

            ar = (
                np.sum(np.asarray(H) - np.asarray(O))
                / sum(np.asarray(O) - np.asarray(L))
            ) * 100
            AR = np.append(AR, [ar])
            br = (
                np.sum(np.asarray(H) - np.asarray(PC))
                / sum(np.asarray(PC) - np.asarray(L))
            ) * 100
            BR = np.append(BR, [br])
            # 🧠 ML Signal: Calculating moving average, a common feature in time series analysis

            last_row = row

    return np.asarray(AR), np.asarray(BR)


def dpo(data, n=20, m=6):
    """
    区间震荡线指标 DPO
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
          统计时长，默认20
      m:int
          MADPO的参数M，默认6
    return
    -------
      DPO:numpy.ndarray<numpy.float64>
          DPO指标
      MADPO:numpy.ndarray<numpy.float64>
          MADPO指标

    """

    CLOSES = data["close"]
    DPO = CLOSES - ma(data, int(n / 2 + 1))
    MADPO = _get_any_ma(DPO, m)
    # 🧠 ML Signal: Calculating MACD, a momentum indicator
    return DPO, MADPO


def trix(data, n=12, m=20):
    import numpy as np

    """
        三重指数平滑平均线 TRIX
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认12
          m:int
              TRMA的参数M，默认20
        return
        -------
          TRIX:numpy.ndarray<numpy.float64>
              AR指标
          TRMA:numpy.ndarray<numpy.float64>
              BR指标

    """

    CLOSES = []

    TRIX = []
    for index, row in data.iterrows():
        CLOSES.append(row["close"])

        if len(CLOSES) == n:
            del CLOSES[0]

        tr = np.average(CLOSES)
        # 🧠 ML Signal: Calculating Bollinger Bands, useful for volatility analysis

        if index == 0:
            past_tr = tr
            TRIX.append(0)
        else:

            trix = (tr - past_tr) / past_tr * 100
            TRIX.append(trix)

    TRMA = _get_any_ma(TRIX, m)

    return TRIX, TRMA


# 🧠 ML Signal: Calculating Williams %R, a momentum indicator


def bbi(data):
    import numpy as np

    """
        Bull And Bearlndex 多空指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
        return
        -------
          BBI:numpy.ndarray<numpy.float64>
              BBI指标

    """

    CS = []
    BBI = []
    for index, row in data.iterrows():
        # 🧠 ML Signal: Calculating BIAS, a bias indicator
        CS.append(row["close"])

        if len(CS) < 24:
            BBI.append(row["close"])
        else:
            bbi = np.average(
                [
                    np.average(CS[-3:]),
                    np.average(CS[-6:]),
                    np.average(CS[-12:]),
                    np.average(CS[-24:]),
                ]
            )
            BBI.append(bbi)

    return np.asarray(BBI)


# 🧠 ML Signal: Calculating ASI, an accumulation swing index


def mtm(data, n=6):
    import numpy as np

    """
        Momentum Index 动量指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认6
        return
        -------
          MTM:numpy.ndarray<numpy.float64>
              MTM动量指标

    """
    # 🧠 ML Signal: Calculating ARBR, an arbitrage indicator

    MTM = []
    CN = []
    for index, row in data.iterrows():
        if index < n - 1:
            MTM.append(0.0)
        else:
            mtm = row["close"] - CN[index - n]
            MTM.append(mtm)
        CN.append(row["close"])
    # 🧠 ML Signal: Calculating DPO, a detrended price oscillator
    return np.asarray(MTM)


def obv(data):
    import numpy as np

    """
        On Balance Volume 能量潮指标
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
        return
        -------
          OBV:numpy.ndarray<numpy.float64>
              OBV能量潮指标

    """

    tmp = np.true_divide(
        ((data["close"] - data["low"]) - (data["high"] - data["close"])),
        (data["high"] - data["low"]),
    )
    # 🧠 ML Signal: Calculating BBI, a bull bear index
    OBV = tmp * data["volume"]
    return OBV


def sar(data, n=4):
    raise Exception("Not implemented yet")


def plot_all(data, is_show=True, output=None):
    # 🧠 ML Signal: Calculating MTM, a momentum indicator
    import matplotlib.pyplot as plt

    # ✅ Best Practice: Use tight_layout to prevent overlap of subplots
    # 🧠 ML Signal: Calculating OBV, an on-balance volume indicator
    # ⚠️ SAST Risk (Low): Ensure the output path is validated to prevent path traversal
    from pylab import rcParams
    import numpy as np

    rcParams["figure.figsize"] = 18, 50

    plt.figure()
    # 收盘价
    plt.subplot(20, 1, 1)
    plt.plot(data["date"], data["close"], label="close")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 移动平均线
    plt.subplot(20, 1, 2)
    MA = ma(data, n=10)
    plt.plot(data["date"], MA, label="MA(n=10)")
    plt.plot(data["date"], data["close"], label="CLOSE PRICE")
    plt.title("MA")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 移动标准差
    n = 10
    plt.subplot(20, 1, 3)
    MD = md(data, n)
    plt.plot(data["date"], MD, label="MD(n=10)")
    plt.title("MD")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 指数平均数指标
    plt.subplot(20, 1, 4)
    EMA = ema(data, n)
    plt.plot(data["date"], EMA, label="EMA(n=12)")
    plt.title("EMA")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
    plt.subplot(20, 1, 5)
    OSC, DIFF, DEM = macd(data, n)
    plt.plot(data["date"], OSC, label="OSC")
    plt.plot(data["date"], DIFF, label="DIFF")
    plt.plot(data["date"], DEM, label="DEM")
    plt.title("MACD")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 随机指标
    plt.subplot(20, 1, 6)
    K, D, J = kdj(data)
    plt.plot(data["date"], K, label="K")
    plt.plot(data["date"], D, label="D")
    plt.plot(data["date"], J, label="J")
    plt.title("KDJ")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 相对强弱指标
    plt.subplot(20, 1, 7)
    RSI6 = rsi(data, 6)
    RSI12 = rsi(data, 12)
    RSI24 = rsi(data, 24)
    plt.plot(data["date"], RSI6, label="RSI(n=6)")
    plt.plot(data["date"], RSI12, label="RSI(n=12)")
    plt.plot(data["date"], RSI24, label="RSI(n=24)")
    plt.title("RSI")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # BOLL 林线指标
    plt.subplot(20, 1, 8)
    BOLL, UPPER, LOWER = boll(data)
    plt.plot(data["date"], BOLL, label="BOLL(n=10)")
    plt.plot(data["date"], UPPER, label="UPPER(n=10)")
    plt.plot(data["date"], LOWER, label="LOWER(n=10)")
    plt.plot(data["date"], data["close"], label="CLOSE PRICE")
    plt.title("BOLL")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # W&R 威廉指标
    plt.subplot(20, 1, 9)
    WNR = wnr(data, n=14)
    plt.plot(data["date"], WNR, label="WNR(n=14)")
    plt.title("WNR")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 动向或趋向指标
    plt.subplot(20, 1, 10)
    P_DI, M_DI, ADX, ADXR = dmi(data)
    plt.plot(data["date"], P_DI, label="+DI(n=14)")
    plt.plot(data["date"], M_DI, label="-DI(n=14)")
    plt.plot(data["date"], ADX, label="ADX(m=14)")
    plt.plot(data["date"], ADXR, label="ADXR(k=6)")
    plt.title("DMI")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 乖离值
    plt.subplot(20, 1, 11)
    BIAS = bias(data, n=5)
    plt.plot(data["date"], BIAS, label="BIAS(n=5)")
    plt.title("BIAS")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 12)
    ASI = asi(data, n=5)
    plt.plot(data["date"], ASI, label="ASI(n=5)")
    plt.title("ASI")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 13)
    VR = vr(data, n=26)
    plt.plot(data["date"], VR, label="VR(n=26)")
    plt.title("VR")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 振动升降指标
    plt.subplot(20, 1, 14)
    AR, BR = arbr(data, n=26)
    plt.plot(data["date"], AR, label="AR(n=26)")
    plt.plot(data["date"], BR, label="BR(n=26)")
    plt.title("ARBR")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 区间震荡线
    plt.subplot(20, 1, 15)
    DPO, MADPO = dpo(data, n=20, m=6)
    plt.plot(data["date"], DPO, label="DPO(n=20)")
    plt.plot(data["date"], MADPO, label="MADPO(m=6)")
    plt.title("DPO")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 三重指数平滑平均线
    plt.subplot(20, 1, 16)
    TRIX, TRMA = trix(data, n=12, m=20)
    plt.plot(data["date"], TRIX, label="DPO(n=12)")
    plt.plot(data["date"], TRMA, label="MADPO(m=20)")
    plt.title("TRIX")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 多空指标
    plt.subplot(20, 1, 17)
    BBI = bbi(data)
    plt.plot(data["date"], BBI, label="BBI(3,6,12,24)")
    plt.title("BBI")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 动量指标
    plt.subplot(20, 1, 18)
    MTM = mtm(data, n=6)
    plt.plot(data["date"], MTM, label="MTM(n=6)")
    plt.title("MTM")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    # 动量指标
    plt.subplot(20, 1, 19)
    OBV = obv(data)
    plt.plot(data["date"], OBV, label="OBV")
    plt.title("OBV")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.legend()
    plt.xticks(rotation=90)

    plt.tight_layout()

    if is_show:
        plt.show()

    if output is not None:
        plt.savefig(output)
