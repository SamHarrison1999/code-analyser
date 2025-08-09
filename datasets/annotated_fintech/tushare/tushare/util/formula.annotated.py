#!/usr/bin/python
# -*- coding: utf-8 -*-
# ⚠️ SAST Risk (Low): Function does not validate input types, which could lead to runtime errors if DF is not a DataFrame or N is not an integer.

# ✅ Best Practice: Consider adding a docstring to describe the function's purpose, parameters, and return value.
import numpy as np
# ✅ Best Practice: Use descriptive parameter names for better readability, e.g., `data_frame` instead of `DF` and `span` instead of `N`.
# ⚠️ SAST Risk (Low): Function does not validate input types, which could lead to runtime errors if DF is not a DataFrame or N is not an integer.
import pandas as pd
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# 🧠 ML Signal: Usage of Exponential Moving Average (EMA) calculation, which is common in financial data analysis.

# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
# 🧠 ML Signal: Use of rolling mean function indicates time series data processing, which is common in financial or sensor data analysis.

def EMA(DF, N):
    # ✅ Best Practice: Use method chaining for better readability and to avoid intermediate variables.
    # ⚠️ SAST Risk (Low): Ensure DF is a pandas DataFrame or Series to avoid unexpected behavior.
    return pd.Series.ewm(DF, span=N, min_periods=N - 1, adjust=True).mean()

# 🧠 ML Signal: Usage of fillna indicates handling of missing data.

def MA(DF, N):
    # ✅ Best Practice: Use descriptive variable names instead of single letters like 'z'.
    return pd.Series.rolling(DF, N).mean()

# ⚠️ SAST Risk (Low): Ensure numpy is imported as np to avoid NameError.

def SMA(DF, N, M):
    DF = DF.fillna(0)
    # 🧠 ML Signal: Iterative calculation pattern for moving averages.
    z = len(DF)
    var = np.zeros(z)
    var[0] = DF[0]
    # ✅ Best Practice: Use descriptive variable names for better readability
    for i in range(1, z):
        var[i] = (DF[i] * M + var[i - 1] * (N - M)) / N
    # ⚠️ SAST Risk (Low): Ensure that the functions MAX, ABS, REF, and MA handle edge cases and invalid inputs
    for i in range(z):
        # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
        DF[i] = var[i]
    return DF
# ⚠️ SAST Risk (Low): Function does not validate input types, which may lead to runtime errors if DF is not a DataFrame or N is not an integer.
# ⚠️ SAST Risk (Low): Assumes DF is a DataFrame and N is a valid integer, lacks input validation

# ✅ Best Practice: Consider adding input validation to ensure DF is a DataFrame and N is a positive integer.
# ✅ Best Practice: Use descriptive variable names for better readability

# 🧠 ML Signal: Use of rolling window operations, which are common in time series analysis.
# 🧠 ML Signal: Function definition with parameters suggests a pattern for function usage
def ATR(DF, N):
    # ✅ Best Practice: Add a docstring to describe the function's purpose, parameters, and return value for better readability and maintainability.
    # ✅ Best Practice: Function name should be lowercase to follow PEP 8 naming conventions
    C = DF['close']
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
    # ✅ Best Practice: Function name should be lowercase to follow PEP 8 naming conventions
    H = DF['high']
    # ⚠️ SAST Risk (Low): Assumes DF is a DataFrame or Series without validation
    L = DF['low']
    # ✅ Best Practice: Function names should be lowercase to follow PEP 8 naming conventions
    # ⚠️ SAST Risk (Low): Assumes N is a valid integer without validation
    # 🧠 ML Signal: Use of built-in abs function to calculate absolute value
    TR1 = MAX(MAX((H - L), ABS(REF(C, 1) - H)), ABS(REF(C, 1) - L))
    atr = MA(TR1, N)
    # ⚠️ SAST Risk (Low): IF is not a built-in Python function, potential misuse or typo
    return atr
# ✅ Best Practice: Function names should be lowercase according to PEP 8.


# ✅ Best Practice: Variable names should be lowercase according to PEP 8.
def HHV(DF, N):
    # ⚠️ SAST Risk (High): Use of undefined function 'IF', which could lead to execution of arbitrary code if 'IF' is user-defined.
    # ⚠️ SAST Risk (Low): Function name 'IF' is too generic and may conflict with built-in or other library functions.
    return pd.Series.rolling(DF, N).max()

# 🧠 ML Signal: Return statements indicate the output of a function, useful for understanding function behavior.
# ⚠️ SAST Risk (Low): Use of np.where without input validation may lead to unexpected behavior if inputs are not as expected.

def LLV(DF, N):
    # ✅ Best Practice: Consider using enumerate for better readability and to avoid potential index errors.
    return pd.Series.rolling(DF, N).min()
# 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns

# ⚠️ SAST Risk (Low): Directly modifying input list V1 can lead to unintended side effects outside the function.

# 🧠 ML Signal: Use of DataFrame method 'diff', common in data manipulation tasks
def SUM(DF, N):
    return pd.Series.rolling(DF, N).sum()
# ✅ Best Practice: Reassigning 'var' to improve readability and maintainability
# 🧠 ML Signal: Function definition with parameters indicating a pattern for statistical computation

# ✅ Best Practice: Function name should be more descriptive, e.g., calculate_standard_deviation

# 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
# 🧠 ML Signal: Return statement, useful for understanding function output patterns
# ✅ Best Practice: Parameter names should be more descriptive, e.g., dataframe and window_size
def ABS(DF):
    # ⚠️ SAST Risk (Low): Assumes DF is a DataFrame, which may not be validated
    return abs(DF)
# 🧠 ML Signal: Calling a function with specific arguments, useful for learning function call patterns
# ⚠️ SAST Risk (Low): Assumes N is a valid integer for window size, which may not be validated

# ✅ Best Practice: Import statements for required libraries (e.g., pandas) are missing

# 🧠 ML Signal: Calling a function with specific arguments, useful for learning function call patterns
def MAX(A, B):
    var = IF(A > B, A, B)
    # 🧠 ML Signal: Arithmetic operation on variables, useful for learning data manipulation patterns
    return var

# 🧠 ML Signal: Calling a function with specific arguments, useful for learning function call patterns

# 🧠 ML Signal: Function definition for a financial indicator, useful for feature extraction in ML models
def MIN(A, B):
    # 🧠 ML Signal: Arithmetic operation on variables, useful for learning data manipulation patterns
    var = IF(A < B, A, B)
    # 🧠 ML Signal: Usage of DataFrame columns, common in data preprocessing for ML
    return var
# 🧠 ML Signal: Dictionary creation, useful for learning data structure usage patterns


# ⚠️ SAST Risk (Low): Assumes 'pd' is imported and is pandas, could raise NameError if not
def IF(COND, V1, V2):
    # ⚠️ SAST Risk (Low): Potential division by zero if HHV(H, N) equals LLV(L, N)
    var = np.where(COND, V1, V2)
    # ✅ Best Practice: Explicit return of a variable, improves readability
    for i in range(len(var)):
        # 🧠 ML Signal: Calculation of moving averages, often used in time series analysis
        V1[i] = var[i]
    return V1

# 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns

# ✅ Best Practice: Use descriptive variable names for clarity
def REF(DF, N):
    # 🧠 ML Signal: Accessing DataFrame column, common operation in data processing
    var = DF.diff(N)
    # 🧠 ML Signal: Conversion to DataFrame, a common step in data preparation for ML
    var = DF - var
    # 🧠 ML Signal: Calculation involving moving average, common in financial data analysis
    return var

# 🧠 ML Signal: Exponential moving average calculation, common in time series analysis

# 🧠 ML Signal: Function definition with multiple parameters, indicating a pattern for ML model input
def STD(DF, N):
    # ✅ Best Practice: Use of descriptive dictionary keys for clarity
    return pd.Series.rolling(DF, N).std()
# 🧠 ML Signal: Accessing DataFrame column, common pattern in data processing

# ✅ Best Practice: Converting dictionary to DataFrame for structured data handling

# 🧠 ML Signal: Use of moving average function, common in financial data analysis
def MACD(DF, FAST, SLOW, MID):
    # 🧠 ML Signal: Returning a DataFrame, common pattern in data processing functions
    EMAFAST = EMA(DF, FAST)
    # 🧠 ML Signal: Function definition with multiple parameters, indicating a complex operation
    # ✅ Best Practice: Use of descriptive variable names for dictionary keys
    EMASLOW = EMA(DF, SLOW)
    DIFF = EMAFAST - EMASLOW
    # ✅ Best Practice: Creating a DataFrame from a dictionary for structured data handling
    # 🧠 ML Signal: Usage of a custom function BBI, indicating a specific calculation pattern
    DEA = EMA(DIFF, MID)
    MACD = (DIFF - DEA) * 2
    # 🧠 ML Signal: Returning a DataFrame, common pattern in data processing functions
    # 🧠 ML Signal: Calculation of upper band using a multiplier and standard deviation
    DICT = {'DIFF': DIFF, 'DEA': DEA, 'MACD': MACD}
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Calculation of lower band using a multiplier and standard deviation
    return VAR
# 🧠 ML Signal: Function definition with multiple parameters, indicating a pattern for complex calculations

# ✅ Best Practice: Use of a dictionary to organize related data

# 🧠 ML Signal: Accessing a specific column from a DataFrame, common in data processing tasks
def KDJ(DF, N, M1, M2):
    # ✅ Best Practice: Conversion of dictionary to DataFrame for structured data handling
    C = DF['close']
    # 🧠 ML Signal: Use of Exponential Moving Average (EMA), a common pattern in financial data analysis
    H = DF['high']
    # ✅ Best Practice: Explicit return of the DataFrame for clarity
    L = DF['low']
    # 🧠 ML Signal: Repeated pattern of EMA calculations with different parameters
    RSV = (C - LLV(L, N)) / (HHV(H, N) - LLV(L, N)) * 100
    K = SMA(RSV, M1, 1)
    D = SMA(K, M2, 1)
    J = 3 * K - 2 * D
    DICT = {'KDJ_K': K, 'KDJ_D': D, 'KDJ_J': J}
    VAR = pd.DataFrame(DICT)
    # ✅ Best Practice: Using a dictionary to organize related variables
    # 🧠 ML Signal: Function definition with financial indicator calculation
    return VAR

# 🧠 ML Signal: Accessing 'close' column from DataFrame

# ✅ Best Practice: Converting a dictionary to a DataFrame for structured data handling
def OSC(DF, N, M):  # 变动速率线
    # 🧠 ML Signal: Calculation of moving average
    C = DF['close']
    # ✅ Best Practice: Returning a DataFrame, which is a common practice for data processing functions
    OS = (C - MA(C, N)) * 100
    # 🧠 ML Signal: Calculation of upper Bollinger Band
    MAOSC = EMA(OS, M)
    DICT = {'OSC': OS, 'MAOSC': MAOSC}
    # 🧠 ML Signal: Calculation of lower Bollinger Band
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    VAR = pd.DataFrame(DICT)
    return VAR
# 🧠 ML Signal: Accessing DataFrame column, common operation in data processing
# ✅ Best Practice: Use of descriptive dictionary keys for clarity


# ✅ Best Practice: Conversion of dictionary to DataFrame for structured data handling
# ⚠️ SAST Risk (Low): Potential division by zero if REF(C, N) returns zero
def BBI(DF, N1, N2, N3, N4):  # 多空指标
    C = DF['close']
    # ✅ Best Practice: Returning a DataFrame for consistency in data handling
    # 🧠 ML Signal: Calculation of moving average, common in time series analysis
    bbi = (MA(C, N1) + MA(C, N2) + MA(C, N3) + MA(C, N4)) / 4
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    DICT = {'BBI': bbi}
    # 🧠 ML Signal: Dictionary creation, useful for learning data structuring patterns
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Accessing DataFrame column, common operation in data processing
    return VAR
# 🧠 ML Signal: DataFrame creation from dictionary, common in data manipulation

# 🧠 ML Signal: Subtraction operation on series, useful for learning arithmetic operations on data

# ✅ Best Practice: Explicit return of the DataFrame, improves readability
def BBIBOLL(DF, N1, N2, N3, N4, N, M):  # 多空布林线
    # 🧠 ML Signal: Function call pattern, useful for learning how functions are used
    bbiboll = BBI(DF, N1, N2, N3, N4)
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    UPER = bbiboll + M * STD(bbiboll, N)
    # 🧠 ML Signal: Dictionary creation, useful for learning data structuring patterns
    DOWN = bbiboll - M * STD(bbiboll, N)
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    DICT = {'BBIBOLL': bbiboll, 'UPER': UPER, 'DOWN': DOWN}
    # 🧠 ML Signal: DataFrame creation from dictionary, common in data manipulation
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    return VAR
# ✅ Best Practice: Explicit return of a variable, improves readability

# 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing

def PBX(DF, N1, N2, N3, N4, N5, N6):  # 瀑布线
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    C = DF['close']
    PBX1 = (EMA(C, N1) + EMA(C, 2 * N1) + EMA(C, 4 * N1)) / 3
    # 🧠 ML Signal: Calculation of typical price, a common financial metric
    PBX2 = (EMA(C, N2) + EMA(C, 2 * N2) + EMA(C, 4 * N2)) / 3
    PBX3 = (EMA(C, N3) + EMA(C, 2 * N3) + EMA(C, 4 * N3)) / 3
    # ⚠️ SAST Risk (Low): Use of undefined functions SUM, IF, REF, potential NameError
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    PBX4 = (EMA(C, N4) + EMA(C, 2 * N4) + EMA(C, 4 * N4)) / 3
    PBX5 = (EMA(C, N5) + EMA(C, 2 * N5) + EMA(C, 4 * N5)) / 3
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    PBX6 = (EMA(C, N6) + EMA(C, 2 * N6) + EMA(C, 4 * N6)) / 3
    # 🧠 ML Signal: Calculation of MFI, a common financial indicator
    DICT = {'PBX1': PBX1, 'PBX2': PBX2, 'PBX3': PBX3,
            # 🧠 ML Signal: Function call with specific parameters, useful for learning function usage patterns
            'PBX4': PBX4, 'PBX5': PBX5, 'PBX6': PBX6}
    # ✅ Best Practice: Use of descriptive variable names for dictionary keys
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Function call with specific parameters, useful for learning function usage patterns
    return VAR
# ⚠️ SAST Risk (Low): Use of undefined module 'pd', potential NameError

# ⚠️ SAST Risk (Low): Division operation, potential for division by zero if HIGHV equals LOWV

# 🧠 ML Signal: Function call with specific parameters, useful for learning function usage patterns
# 🧠 ML Signal: Returning a DataFrame, common in data processing functions
def BOLL(DF, N):  # 布林线
    # 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns
    C = DF['close']
    # 🧠 ML Signal: Function call with specific parameters, useful for learning function usage patterns
    boll = MA(C, N)
    # 🧠 ML Signal: Accessing DataFrame columns, common pattern in data manipulation
    UB = boll + 2 * STD(C, N)
    # 🧠 ML Signal: Function call with specific parameters, useful for learning function usage patterns
    LB = boll - 2 * STD(C, N)
    # 🧠 ML Signal: Accessing DataFrame columns, common pattern in data manipulation
    DICT = {'BOLL': boll, 'UB': UB, 'LB': LB}
    # ✅ Best Practice: Use of descriptive dictionary keys for clarity
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Accessing DataFrame columns, common pattern in data manipulation
    return VAR
# 🧠 ML Signal: Creating a DataFrame from a dictionary, common data manipulation pattern

# ⚠️ SAST Risk (Low): Potential division by zero if HHV(HIGH, N) equals LLV(LOW, N)

# 🧠 ML Signal: Returning a DataFrame, common pattern in data processing functions
# 🧠 ML Signal: Function definition with financial data processing
def ROC(DF, N, M):  # 变动率指标
    # ⚠️ SAST Risk (Low): Potential division by zero if HHV(HIGH, N1) equals LLV(LOW, N1)
    C = DF['close']
    # 🧠 ML Signal: Accessing 'close' column from DataFrame
    roc = 100 * (C - REF(C, N)) / REF(C, N)
    # ✅ Best Practice: Use lowercase variable names for consistency with Python naming conventions
    MAROC = MA(roc, M)
    # 🧠 ML Signal: Calculation of BIAS1 using moving average
    DICT = {'ROC': roc, 'MAROC': MAROC}
    # ✅ Best Practice: Use lowercase variable names for consistency with Python naming conventions
    VAR = pd.DataFrame(DICT)
    # 🧠 ML Signal: Calculation of BIAS2 using moving average
    return VAR
# 🧠 ML Signal: Returning a DataFrame, common pattern in data processing functions

# 🧠 ML Signal: Calculation of BIAS3 using moving average
# 🧠 ML Signal: Function definition with financial indicators can be used to train models for stock market predictions

def MTM(DF, N, M):  # 动量线
    # ✅ Best Practice: Use of descriptive dictionary keys
    # 🧠 ML Signal: Usage of DataFrame column 'close' indicates reliance on historical price data
    C = DF['close']
    mtm = C - REF(C, N)
    # ✅ Best Practice: Conversion of dictionary to DataFrame for structured data handling
    # ✅ Best Practice: Use descriptive variable names for better readability
    MTMMA = MA(mtm, M)
    DICT = {'MTM': mtm, 'MTMMA': MTMMA}
    # ✅ Best Practice: Returning a DataFrame for further analysis or processing
    # ✅ Best Practice: Use descriptive variable names for better readability
    VAR = pd.DataFrame(DICT)
    return VAR
# ✅ Best Practice: Use descriptive variable names for better readability

# 🧠 ML Signal: Function definition with parameters, useful for learning function usage patterns

# ✅ Best Practice: Use descriptive variable names for better readability
def MFI(DF, N):  # 资金指标
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    C = DF['close']
    # ✅ Best Practice: Use descriptive variable names for better readability
    H = DF['high']
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    # 🧠 ML Signal: Conversion to DataFrame suggests data preparation for further analysis or modeling
    L = DF['low']
    VOL = DF['vol']
    # 🧠 ML Signal: Accessing DataFrame columns, common operation in data processing
    TYP = (C + H + L) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / \
        # ⚠️ SAST Risk (Low): Potential issue if 'REF' or 'MAX' are not defined or imported
        SUM(IF(TYP < REF(TYP, 1), TYP * VOL, 0), N)
    mfi = 100 - (100 / (1 + V1))
    DICT = {'MFI': mfi}
    # ⚠️ SAST Risk (Low): Potential issue if 'REF' or 'MAX' are not defined or imported
    VAR = pd.DataFrame(DICT)
    return VAR
# ⚠️ SAST Risk (Low): Potential issue if 'SUM' is not defined or imported


# ⚠️ SAST Risk (Low): Potential issue if 'SUM' is not defined or imported
# 🧠 ML Signal: Function definition with multiple parameters, indicating a complex operation
def SKDJ(DF, N, M):
    CLOSE = DF['close']
    # ⚠️ SAST Risk (Low): Potential issue if 'IF' is not defined or imported
    # 🧠 ML Signal: Accessing DataFrame columns, common in data processing tasks
    LOWV = LLV(DF['low'], N)
    # 🧠 ML Signal: Accessing DataFrame columns, common in data processing tasks
    HIGHV = HHV(DF['high'], N)
    RSV = EMA((CLOSE - LOWV) / (HIGHV - LOWV) * 100, M)
    # ⚠️ SAST Risk (Low): Potential issue if 'MA' is not defined or imported
    # ⚠️ SAST Risk (Low): Use of undefined function IF, potential for NameError
    K = EMA(RSV, M)
    D = MA(K, M)
    # ✅ Best Practice: Use descriptive variable names for better readability
    DICT = {'SKDJ_K': K, 'SKDJ_D': D}
    # ⚠️ SAST Risk (Low): Use of undefined function REF, potential for NameError
    VAR = pd.DataFrame(DICT)
    # ⚠️ SAST Risk (Low): Ensure 'pd' (pandas) is imported before use
    return VAR
# ⚠️ SAST Risk (Low): Use of undefined function IF, potential for NameError

# 🧠 ML Signal: Returning a DataFrame, common pattern in data processing functions

# ⚠️ SAST Risk (Low): Use of undefined function REF, potential for NameError
def WR(DF, N, N1):  # 威廉指标
    HIGH = DF['high']
    # 🧠 ML Signal: Use of dictionary to store multiple related values
    # ⚠️ SAST Risk (Low): Use of undefined function SUM, potential for NameError
    # 🧠 ML Signal: Calculation of a difference, common in financial indicators
    # ✅ Best Practice: Explicit return of a variable, improves readability
    LOW = DF['low']
    CLOSE = DF['close']
    WR1 = 100 * (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N))
    WR2 = 100 * (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1))
    DICT = {'WR1': WR1, 'WR2': WR2}
    VAR = pd.DataFrame(DICT)
    return VAR


def BIAS(DF, N1, N2, N3):  # 乖离率
    CLOSE = DF['close']
    BIAS1 = (CLOSE - MA(CLOSE, N1)) / MA(CLOSE, N1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, N2)) / MA(CLOSE, N2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, N3)) / MA(CLOSE, N3) * 100
    DICT = {'BIAS1': BIAS1, 'BIAS2': BIAS2, 'BIAS3': BIAS3}
    VAR = pd.DataFrame(DICT)
    return VAR


def RSI(DF, N1, N2, N3):  # 相对强弱指标RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
    CLOSE = DF['close']
    LC = REF(CLOSE, 1)
    RSI1 = SMA(MAX(CLOSE - LC, 0), N1, 1) / SMA(ABS(CLOSE - LC), N1, 1) * 100
    RSI2 = SMA(MAX(CLOSE - LC, 0), N2, 1) / SMA(ABS(CLOSE - LC), N2, 1) * 100
    RSI3 = SMA(MAX(CLOSE - LC, 0), N3, 1) / SMA(ABS(CLOSE - LC), N3, 1) * 100
    DICT = {'RSI1': RSI1, 'RSI2': RSI2, 'RSI3': RSI3}
    VAR = pd.DataFrame(DICT)
    return VAR


def ADTM(DF, N, M):  # 动态买卖气指标
    HIGH = DF['high']
    LOW = DF['low']
    OPEN = DF['open']
    DTM = IF(OPEN <= REF(OPEN, 1), 0, MAX(
        (HIGH - OPEN), (OPEN - REF(OPEN, 1))))
    DBM = IF(OPEN >= REF(OPEN, 1), 0, MAX((OPEN - LOW), (OPEN - REF(OPEN, 1))))
    STM = SUM(DTM, N)
    SBM = SUM(DBM, N)
    ADTM1 = IF(STM > SBM, (STM - SBM) / STM,
               IF(STM == SBM, 0, (STM - SBM) / SBM))
    MAADTM = MA(ADTM1, M)
    DICT = {'ADTM': ADTM1, 'MAADTM': MAADTM}
    VAR = pd.DataFrame(DICT)
    return VAR


def DDI(DF, N, N1, M, M1):  # 方向标准离差指数
    H = DF['high']
    L = DF['low']
    DMZ = IF((H + L) <= (REF(H, 1) + REF(L, 1)), 0,
             MAX(ABS(H - REF(H, 1)), ABS(L - REF(L, 1))))
    DMF = IF((H + L) >= (REF(H, 1) + REF(L, 1)), 0,
             MAX(ABS(H - REF(H, 1)), ABS(L - REF(L, 1))))
    DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))
    DIF = SUM(DMF, N) / (SUM(DMF, N) + SUM(DMZ, N))
    ddi = DIZ - DIF
    ADDI = SMA(ddi, N1, M)
    AD = MA(ADDI, M1)
    DICT = {'DDI': ddi, 'ADDI': ADDI, 'AD': AD}
    VAR = pd.DataFrame(DICT)
    return VAR