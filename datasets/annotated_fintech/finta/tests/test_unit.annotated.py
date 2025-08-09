import pytest
import os
import pandas as pd
# ✅ Best Practice: Importing specific modules or classes can improve code readability and maintainability.
from pandas.core import series
from finta import TA
# ✅ Best Practice: Importing specific functions or classes from a library can improve code readability and maintainability.
# ⚠️ SAST Risk (Medium): os.path.abspath(__file__) can expose sensitive file path information.


def rootdir():

    # ⚠️ SAST Risk (Medium): os.path.join with user-controlled input can lead to directory traversal vulnerabilities.
    return os.path.dirname(os.path.abspath(__file__))

# 🧠 ML Signal: Usage of TA.SMA function with specific parameters

# 🧠 ML Signal: Reading CSV files is a common operation in data processing tasks.
data_file = os.path.join(rootdir(), "data/bittrex_btc-usdt.csv")
# ⚠️ SAST Risk (Low): pd.read_csv can be exploited if the CSV file contains malicious content.
# ⚠️ SAST Risk (Low): Assert statements are used for testing, which may be disabled in production

# ✅ Best Practice: Specify 'index_col' and 'parse_dates' for better data handling and performance.
ohlc = pd.read_csv(data_file, index_col="date", parse_dates=True)
# ⚠️ SAST Risk (Low): Assert statements are used for testing, which may be disabled in production

# 🧠 ML Signal: Testing function for TA.SMM, useful for learning test patterns

def test_sma():
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    """test TA.ma"""
    # ✅ Best Practice: Use of round() to ensure precision in floating-point operations

    ma = TA.SMA(ohlc, 14).round(decimals=8)
    # 🧠 ML Signal: Testing function for TA.SSMA, useful for learning test patterns
    # ⚠️ SAST Risk (Low): If 'ma' is not a series, this will raise an AssertionError

    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    assert ma.values[-1] == 6922.33922063
# 🧠 ML Signal: Use of assert to validate expected outcomes in tests
# ✅ Best Practice: Use of round() for consistent decimal precision
# 🧠 ML Signal: Function definition for testing, useful for identifying test patterns


# ⚠️ SAST Risk (Low): Assertion without error message can make debugging harder
def test_smm():
    # 🧠 ML Signal: Usage of TA.EMA with specific parameters, indicating a pattern for EMA calculation
    """test TA.SMM"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable

    # ⚠️ SAST Risk (Low): Lack of exception handling for potential errors in TA.EMA or round
    ma = TA.SMM(ohlc).round(decimals=8)
    # 🧠 ML Signal: Type checking with isinstance, useful for understanding expected data types

    # 🧠 ML Signal: Function name follows a common pattern for test functions
    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    assert ma.values[-1] == 6490.0
# ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated and sanitized before use
# 🧠 ML Signal: Assertion with specific value, indicating expected output for test case

# ✅ Best Practice: Use of 'round' for consistent decimal precision

def test_ssma():
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported
    # 🧠 ML Signal: Use of TA.TEMA indicates a pattern of using technical analysis for financial data
    """test TA.SSMA"""

    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable
    ma = TA.SSMA(ohlc).round(decimals=8)
    # 🧠 ML Signal: Rounding to a specific number of decimals is a common data preprocessing step

    # ✅ Best Practice: Ensure that the rounding is necessary for the precision required
    assert isinstance(ma, series.Series)
    # 🧠 ML Signal: Function name follows a common pattern for test functions
    assert ma.values[-1] == 6907.53759817
# ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported properly

# 🧠 ML Signal: Type checking with 'isinstance' is a common pattern for ensuring data integrity
# ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues

# 🧠 ML Signal: Use of method chaining with round, common in data processing
def test_ema():
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    """test TA.EMA"""
    # 🧠 ML Signal: Function name follows a common pattern for test functions
    # 🧠 ML Signal: Use of assert statements indicates a pattern of test-driven development
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the expected type to prevent type errors

    # 🧠 ML Signal: Use of isinstance for type checking
    ma = TA.EMA(ohlc, 50).round(decimals=8)
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Use of method chaining with round, common in data processing
    assert isinstance(ma, series.Series)
    # 🧠 ML Signal: Use of assert for validation in test functions
    assert ma.values[-1] == 7606.84391951
# 🧠 ML Signal: Testing function for TA.VAMA, useful for learning test patterns
# ✅ Best Practice: Use of isinstance to check type, ensures correct type handling


# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
# ⚠️ SAST Risk (Low): Assumes 'ohlc' is defined and valid, potential for NameError
def test_dema():
    # ✅ Best Practice: Use of method chaining for concise code
    # 🧠 ML Signal: Use of assert for validation, common in test functions
    """test TA.DEMA"""

    # ⚠️ SAST Risk (Low): Assumes 'series' is defined and valid, potential for NameError
    # 🧠 ML Signal: Function call to TA.ER with ohlc as input
    ma = TA.DEMA(ohlc).round(decimals=8)
    # ✅ Best Practice: Use of assert to validate the type of 'ma'
    # ✅ Best Practice: Rounding to a specific number of decimals for consistency

    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point values can lead to precision issues
    # ✅ Best Practice: Using isinstance to check the type of the variable
    assert ma.values[-1] == 6323.41923994
# ✅ Best Practice: Use of assert to validate the final value of 'ma'

# 🧠 ML Signal: Testing function for TA.KAMA, useful for learning test patterns
# 🧠 ML Signal: Assertion to check value range, indicating expected behavior

def test_tema():
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    """test TA.TEMA"""
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency

    ma = TA.TEMA(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported
    # 🧠 ML Signal: Use of TA.ZLEMA indicates a pattern of using technical analysis for financial data

    assert isinstance(ma, series.Series)
    # 🧠 ML Signal: Use of assert to validate expected output, useful for learning expected behavior
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    assert ma.values[-1] == 6307.48151844
# ⚠️ SAST Risk (Low): Hardcoded value in test, ensure it matches expected output
# 🧠 ML Signal: Rounding to a specific number of decimals is a common data preprocessing step


# 🧠 ML Signal: Use of TA.WMA indicates a pattern of using technical analysis for financial data
# ⚠️ SAST Risk (Low): Use of assert statements for validation; may be disabled in optimized mode
def test_trima():
    # 🧠 ML Signal: Checking the type of 'ma' suggests a pattern of ensuring data structure integrity
    """test TA.TRIMA"""
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Use of isinstance to check type, common pattern in testing
    ma = TA.TRIMA(ohlc).round(decimals=8)
    # 🧠 ML Signal: Asserting specific values indicates a pattern of using known benchmarks for validation

    # 🧠 ML Signal: Use of assert to validate expected output, common in test functions
    # 🧠 ML Signal: Function call to TA.HMA with ohlc as input, indicating usage of a specific technical analysis function
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7464.85307304
# ⚠️ SAST Risk (Low): Lack of exception handling for TA.HMA call could lead to unhandled exceptions

# ✅ Best Practice: Use isinstance to ensure ma is of the expected type, improving code robustness

def test_trix():
    # 🧠 ML Signal: Function name follows a common test naming pattern
    # 🧠 ML Signal: Asserting the last value of ma, indicating expected output behavior
    """test TA.TRIX"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

    # ⚠️ SAST Risk (Low): Potential risk if 'TA.EVWMA' or 'ohlc' are not validated or sanitized
    ma = TA.TRIX(ohlc).round(decimals=8)
    # ✅ Best Practice: Use of 'round' for consistent decimal precision
    # ✅ Best Practice: Include a docstring to describe the purpose of the test function

    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Assertion without error message can make debugging harder
    assert ma.values[-1] == -0.5498364
# 🧠 ML Signal: Usage of TA.VWAP indicates a pattern of using technical analysis functions

# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

# ⚠️ SAST Risk (Low): Ensure that 'ohlc' is properly validated and sanitized before use
def test_vama():
    # 🧠 ML Signal: Checking the type of 'ma' suggests a pattern of type validation
    """test TA.VAMA"""
    # 🧠 ML Signal: Testing function for TA.SMMA, useful for learning test patterns

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    ma = TA.VAMA(ohlc, 20).round(decimals=8)
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized

    # 🧠 ML Signal: Use of method chaining with round, common in data processing
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6991.57791258
# ⚠️ SAST Risk (Low): Assumes 'series.Series' is a valid type, could raise an exception if not
# 🧠 ML Signal: Usage of TA.FRAMA function indicates a pattern for financial analysis

# 🧠 ML Signal: Assertion for type checking, useful for learning type validation

# ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
def test_er():
    # ✅ Best Practice: Use of round() for consistent decimal precision
    # 🧠 ML Signal: Function definition for testing a specific feature or component
    # ⚠️ SAST Risk (Low): Hardcoded value in assertion, may lead to false positives if data changes
    """test TA.ER"""
    # 🧠 ML Signal: Assertion for value checking, useful for learning value validation

    # ⚠️ SAST Risk (Low): Assertion without error handling could lead to unhandled exceptions
    er = TA.ER(ohlc).round(decimals=8)
    # 🧠 ML Signal: Instantiation and usage of a specific method from a library

    # ⚠️ SAST Risk (Low): Hardcoded value in assertion may lead to brittle tests
    assert isinstance(er, series.Series)
    # ⚠️ SAST Risk (Low): Lack of exception handling for potential errors in method calls
    assert -100 < er.values[-1] < 100
# ✅ Best Practice: Use of isinstance to ensure the correct type of the object


# ✅ Best Practice: Use of isinstance to ensure the correct type of the object
def test_kama():
    # 🧠 ML Signal: Usage of TA.PPO function with specific rounding
    """test TA.KAMA"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

    # 🧠 ML Signal: Checking type of "PPO" key in ppo dictionary
    ma = TA.KAMA(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

    # 🧠 ML Signal: Checking type of "SIGNAL" key in ppo dictionary
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6742.11786502
# 🧠 ML Signal: Checking type of "HISTO" key in ppo dictionary


# 🧠 ML Signal: Verifying specific value in "PPO" series
def test_zlema():
    # 🧠 ML Signal: Testing function for VW_MACD, useful for learning test patterns
    """test TA.ZLEMA"""
    # 🧠 ML Signal: Verifying specific value in "SIGNAL" series

    # ⚠️ SAST Risk (Low): Assumes 'ohlc' is defined and valid, potential for NameError
    ma = TA.ZLEMA(ohlc).round(decimals=8)
    # ✅ Best Practice: Use of round() for consistent decimal precision
    # 🧠 ML Signal: Verifying specific value in "HISTO" series

    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Assumes 'series.Series' is the correct type, potential for TypeError
    # ✅ Best Practice: Include a docstring to describe the purpose of the test function
    assert ma.values[-1] == 6462.46183365

# ⚠️ SAST Risk (Low): Assumes 'series.Series' is the correct type, potential for TypeError

# 🧠 ML Signal: Usage of a specific method from a library (TA.EV_MACD) indicates a pattern in how this library is used
def test_wma():
    # ⚠️ SAST Risk (Low): Hardcoded value for test, may cause false negatives if data changes
    """test TA.WMA"""
    # ⚠️ SAST Risk (Low): Ensure that 'ohlc' is validated and sanitized before use to prevent potential data integrity issues

    # ✅ Best Practice: Use of assert to validate the type of the result
    # ⚠️ SAST Risk (Low): Hardcoded value for test, may cause false negatives if data changes
    ma = TA.WMA(ohlc).round(decimals=8)

    # ✅ Best Practice: Use of assert to validate the type of the result
    assert isinstance(ma, series.Series)
    # 🧠 ML Signal: Use of a method from a library (TA) to perform a calculation
    assert ma.values[-1] == 6474.47003078
# ✅ Best Practice: Use of assert to validate the expected value of the result

# ⚠️ SAST Risk (Low): Lack of error handling for TA.MOM method call

# ✅ Best Practice: Use of assert to validate the expected value of the result
# 🧠 ML Signal: Type checking with isinstance, indicating expected data type
def test_hma():
    """test TA.HMA"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    # 🧠 ML Signal: Use of assert to validate the output of a function
    ma = TA.HMA(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent potential misuse or errors

    # ✅ Best Practice: Use of 'round' for precision control in floating-point operations
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6186.93727146
# ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported properly
# 🧠 ML Signal: Testing function for TA.VBM, useful for learning test patterns


# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
# ✅ Best Practice: Checking the type of the result to ensure correct function output
def test_evwma():
    """test TA.EVWMA"""
    # 🧠 ML Signal: Use of assert to validate expected output, common in test-driven development

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Use of a method from a specific library (TA) for RSI calculation
    evwma = TA.EVWMA(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Assumes 'ohlc' is defined and valid, potential for NameError if not
    assert isinstance(evwma, series.Series)
    # ✅ Best Practice: Use of assert to validate the type of 'rsi'
    assert evwma.values[-1] == 7445.46084062

# 🧠 ML Signal: Function name follows a common test naming pattern
# ✅ Best Practice: Use of assert to validate the range of 'rsi' values

def test_vwap():
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    """test TA.VWAP"""
    # 🧠 ML Signal: Use of method chaining with round

    ma = TA.VWAP(ohlc).round(decimals=8)
    # 🧠 ML Signal: Use of a method from a library (TA.DYMI) indicates a pattern of library usage
    # ⚠️ SAST Risk (Low): Assumes 'rsi' is always a 'series.Series' without error handling

    # 🧠 ML Signal: Use of isinstance for type checking
    # ✅ Best Practice: Rounding to a specific number of decimals for consistency in test results
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7976.51743477
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
# ⚠️ SAST Risk (Low): Lack of exception handling may cause the test to fail ungracefully if TA.DYMI or rounding fails

# 🧠 ML Signal: Use of assert for validation in tests
# 🧠 ML Signal: Checking the type of a variable is a common pattern in testing

# 🧠 ML Signal: Testing function for TA.TR, useful for learning test patterns
def test_smma():
    # 🧠 ML Signal: Asserting specific values is a common pattern in testing to ensure correctness
    """test TA.SMMA"""
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues

    # ✅ Best Practice: Use of round() for consistent decimal precision
    ma = TA.SMMA(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct expected type
    # 🧠 ML Signal: Function name follows a common pattern for test functions, useful for identifying test cases.
    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 8020.2742957
# ⚠️ SAST Risk (Low): Hardcoded value in test, ensure it matches expected output
# ⚠️ SAST Risk (Low): Ensure that 'ohlc' is validated and sanitized before use to prevent data integrity issues.

# ✅ Best Practice: Rounding to a fixed number of decimals improves consistency in test results.

def test_frama():
    # 🧠 ML Signal: Function name follows a common test naming pattern
    # ⚠️ SAST Risk (Low): Type checking with 'isinstance' is generally safe but ensure 'series.Series' is the expected type.
    """test TA.FRAMA"""

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues; consider using a tolerance.
    # ⚠️ SAST Risk (Low): Assumes 'TA.SAR' and 'ohlc' are safe and correctly implemented
    ma = TA.FRAMA(ohlc).round(decimals=8)
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency

    assert isinstance(ma, series.Series)
    # ⚠️ SAST Risk (Low): Assumes 'sar' is a valid series object
    # 🧠 ML Signal: Usage of TA.PSAR indicates a pattern of using technical analysis for financial data.
    assert ma.values[-1] == 6574.14605454

# ⚠️ SAST Risk (Low): Hardcoded value in test may lead to false positives/negatives
# ⚠️ SAST Risk (Low): Ensure that 'ohlc' is validated and sanitized to prevent potential data integrity issues.

# ✅ Best Practice: Use of assert to validate expected outcome
# 🧠 ML Signal: Checking the type of 'sar.psar' suggests a pattern of verifying data structures.
def test_macd():
    """test TA.MACD"""
    # 🧠 ML Signal: Use of TA.BBANDS indicates testing of a technical analysis function, which is common in financial applications.
    # 🧠 ML Signal: Asserting specific values indicates a pattern of expected output validation.

    macd = TA.MACD(ohlc).round(decimals=8)
    # ✅ Best Practice: Checking the type of the result ensures that the function returns the expected data structure.

    assert isinstance(macd["MACD"], series.Series)
    # ✅ Best Practice: Checking the type of the result ensures that the function returns the expected data structure.
    assert isinstance(macd["SIGNAL"], series.Series)

    # ✅ Best Practice: Checking the type of the result ensures that the function returns the expected data structure.
    assert macd["MACD"].values[-1] == -419.21923359
    assert macd["SIGNAL"].values[-1] == -372.39851312
# 🧠 ML Signal: Use of hardcoded expected values in assertions can indicate a test for specific known outputs.

# 🧠 ML Signal: Use of TA.MOBO function indicates a pattern of using technical analysis for financial data.

# 🧠 ML Signal: Use of hardcoded expected values in assertions can indicate a test for specific known outputs.
def test_ppo():
    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    """test TA.PPO"""
    # 🧠 ML Signal: Use of hardcoded expected values in assertions can indicate a test for specific known outputs.

    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    ppo = TA.PPO(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    assert isinstance(ppo["PPO"], series.Series)
    assert isinstance(ppo["SIGNAL"], series.Series)
    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    assert isinstance(ppo["HISTO"], series.Series)
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    assert ppo["PPO"].values[-1] == -5.85551658
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency
    assert ppo["SIGNAL"].values[-1] == -5.05947256
    # ⚠️ SAST Risk (Low): Use of assert statements for testing can be disabled with optimization flags.
    assert ppo["HISTO"].values[-1] == -0.79604402
# ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and not a typo or undefined

# 🧠 ML Signal: Function name follows a common test naming pattern

# ✅ Best Practice: Asserting value range to ensure expected output
def test_vw_macd():
    # ⚠️ SAST Risk (Low): Assumes 'TA.PERCENT_B' and 'ohlc' are safe and correctly implemented
    """test TA.VW_MACD"""
    # 🧠 ML Signal: Use of method chaining with round

    macd = TA.VW_MACD(ohlc).round(decimals=8)
    # 🧠 ML Signal: Usage of TA.ZLEMA indicates a pattern of using technical analysis indicators
    # ⚠️ SAST Risk (Low): Assumes 'bb' is a valid series object

    # 🧠 ML Signal: Use of isinstance for type checking
    assert isinstance(macd["MACD"], series.Series)
    # 🧠 ML Signal: Usage of TA.KC indicates a pattern of using technical analysis indicators
    assert isinstance(macd["SIGNAL"], series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers

    # 🧠 ML Signal: Use of assert for validation
    # ✅ Best Practice: Checking the type of the result ensures that the function returns the expected data structure
    assert macd["MACD"].values[-1] == -535.21281201
    assert macd["SIGNAL"].values[-1] == -511.64584818
# ✅ Best Practice: Checking the type of the result ensures that the function returns the expected data structure


# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
# 🧠 ML Signal: Usage of a method from the TA module, indicating a pattern for technical analysis.
def test_ev_macd():
    """test TA.EV_MACD"""
    # 🧠 ML Signal: Checking the type of the result, indicating a pattern of expected data structure.
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

    macd = TA.EV_MACD(ohlc).round(decimals=8)
    # 🧠 ML Signal: Checking the type of the result, indicating a pattern of expected data structure.

    assert isinstance(macd["MACD"], series.Series)
    # 🧠 ML Signal: Checking the type of the result, indicating a pattern of expected data structure.
    assert isinstance(macd["SIGNAL"], series.Series)

    # 🧠 ML Signal: Asserting specific values, indicating a pattern of expected output.
    assert macd["MACD"].values[-1] == -786.70979566
    # 🧠 ML Signal: Use of TA.DMI indicates a pattern of using technical analysis for financial data
    assert macd["SIGNAL"].values[-1] == -708.68194345
# 🧠 ML Signal: Asserting specific values, indicating a pattern of expected output.

# ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized

# 🧠 ML Signal: Asserting specific values, indicating a pattern of expected output.
def test_mom():
    # ✅ Best Practice: Checking type ensures that the expected data structure is returned
    """test TA.MOM"""

    # ✅ Best Practice: Checking type ensures that the expected data structure is returned
    mom = TA.MOM(ohlc).round(decimals=8)
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    assert isinstance(mom, series.Series)
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    assert mom.values[-1] == -1215.54681371
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
# 🧠 ML Signal: Use of method chaining with round function


# 🧠 ML Signal: Function name follows a common pattern for test functions
# ⚠️ SAST Risk (Low): Assumes 'adx' is always a 'series.Series' without error handling
def test_roc():
    # 🧠 ML Signal: Use of isinstance to check object type
    """test TA.ROC"""
    # ✅ Best Practice: Use of isinstance to check the type of 'st'

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    roc = TA.ROC(ohlc).round(decimals=8)
    # 🧠 ML Signal: Use of assert statement for validation
    # ⚠️ SAST Risk (Low): Assumes 'st.values' is a list-like object with at least one element

    # 🧠 ML Signal: Testing function for TA.STOCHD, useful for learning test patterns
    assert isinstance(roc, series.Series)
    assert roc.values[-1] == -16.0491877
# ✅ Best Practice: Checking the type of the result to ensure correct function behavior

def test_vbm():
    # ✅ Best Practice: Asserting value range to ensure expected output
    """test TA.VBM"""
    # 🧠 ML Signal: Function name follows a common pattern for test functions, useful for identifying test cases.

    vbm = TA.VBM(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Assumes 'TA.STOCHRSI' and 'ohlc' are defined and valid, potential for NameError if not.

    # ✅ Best Practice: Rounding to a fixed number of decimals improves consistency in floating-point operations.
    assert isinstance(vbm, series.Series)
    assert vbm.values[-1] == -27.57038694
# ⚠️ SAST Risk (Low): Assumes 'series.Series' is the correct type, potential for AssertionError if not.
# 🧠 ML Signal: Use of TA.WILLIAMS indicates a pattern of financial technical analysis

def test_rsi():
    # ⚠️ SAST Risk (Low): Assumes 'st.values' is a valid list-like object, potential for IndexError if empty.
    # ✅ Best Practice: Checking the type of the result ensures the function returns the expected data structure
    """test TA.RSI"""
    # ✅ Best Practice: Asserting value range ensures the output is within expected bounds.

    # ⚠️ SAST Risk (Low): Directly accessing the last element of a series without checking if it's empty could lead to an IndexError
    rsi = TA.RSI(ohlc).round(decimals=8)
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    assert isinstance(rsi, series.Series)
    # ⚠️ SAST Risk (Low): Assumes 'TA.UO' and 'ohlc' are safe and correctly implemented
    assert -100 < rsi.values[-1] < 100
# ✅ Best Practice: Rounding to a fixed number of decimals for consistency


# ⚠️ SAST Risk (Low): Assumes 'uo' is a valid object and 'series.Series' is the correct type
# 🧠 ML Signal: Testing function for TA.AO, useful for learning test patterns
def test_ift_rsi():
    """test TA.IFT_RSI"""
    # ⚠️ SAST Risk (Low): Assumes 'uo.values' is a valid list and has at least one element
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues

    # ✅ Best Practice: Asserting value range to ensure expected behavior
    # ✅ Best Practice: Use of round() for consistent precision in floating-point operations
    rsi = TA.IFT_RSI(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported
    # 🧠 ML Signal: Function name follows a common test naming pattern
    assert isinstance(rsi, series.Series)
    assert rsi.values[-1] == 0.62803976
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable
# ⚠️ SAST Risk (Low): Assumes TA.MI and ohlc are safe and correctly implemented

# 🧠 ML Signal: Use of assert to validate expected outcomes in tests
# ✅ Best Practice: Rounding to a fixed number of decimals for consistency

def test_dymi():
    # ⚠️ SAST Risk (Low): Assumes mi is a valid series object
    # 🧠 ML Signal: Function name follows a common test naming pattern
    """test TA.DYMI"""
    # ✅ Best Practice: Using isinstance to ensure mi is of the expected type

    # ⚠️ SAST Risk (Low): Assumes 'TA.BOP' and 'ohlc' are safe and correctly implemented
    dymi = TA.DYMI(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Hardcoded value in assertion may lead to brittle tests
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency

    # ✅ Best Practice: Asserting the last value to ensure correct calculation
    assert isinstance(dymi, series.Series)
    # ⚠️ SAST Risk (Low): Assumes 'bop' is a valid series object
    # 🧠 ML Signal: Usage of a specific method from a library (TA.VORTEX) indicates a pattern in how this library is used.
    assert dymi.values[-1] == 32.4897564

# ⚠️ SAST Risk (Low): Hardcoded value in test may lead to false positives/negatives
# ⚠️ SAST Risk (Low): Direct use of assert statements for testing can be disabled with the -O and -OO flags in Python, potentially skipping these checks.

# ✅ Best Practice: Use of assertions to validate expected outcomes
# 🧠 ML Signal: Checking the type of an object is a common pattern in testing to ensure correct data types.
def test_tr():
    """test TA.TR"""
    # ⚠️ SAST Risk (Low): Direct use of assert statements for testing can be disabled with the -O and -OO flags in Python, potentially skipping these checks.

    # 🧠 ML Signal: Checking the type of an object is a common pattern in testing to ensure correct data types.
    tr = TA.TR(ohlc).round(decimals=8)
    # 🧠 ML Signal: Testing function for TA.KST, useful for learning test patterns

    # ⚠️ SAST Risk (Low): Direct use of assert statements for testing can be disabled with the -O and -OO flags in Python, potentially skipping these checks.
    assert isinstance(tr, series.Series)
    # 🧠 ML Signal: Asserting specific values in a test indicates expected behavior or output.
    # ⚠️ SAST Risk (Low): Assumes 'ohlc' is defined and valid, potential NameError if not
    assert tr.values[-1] == 113.4
# ✅ Best Practice: Use of round() for consistent decimal precision

# ⚠️ SAST Risk (Low): Direct use of assert statements for testing can be disabled with the -O and -OO flags in Python, potentially skipping these checks.

# 🧠 ML Signal: Asserting specific values in a test indicates expected behavior or output.
# ⚠️ SAST Risk (Low): Assumes 'series.Series' is the correct type, potential AttributeError if not
def test_atr():
    """test TA.ATR"""
    # ⚠️ SAST Risk (Low): Assumes 'series.Series' is the correct type, potential AttributeError if not
    # 🧠 ML Signal: Function name follows a common test naming pattern

    tr = TA.ATR(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers, potential precision issues
    # ⚠️ SAST Risk (Low): Assumes 'TA.TSI' and 'ohlc' are safe and correctly implemented

    assert isinstance(tr, series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers, potential precision issues
    # ⚠️ SAST Risk (Low): Assumes 'tsi["TSI"]' is a valid key and 'series.Series' is the correct type
    assert tr.values[-1] == 328.56890383

# ⚠️ SAST Risk (Low): Assumes 'tsi["signal"]' is a valid key and 'series.Series' is the correct type

# 🧠 ML Signal: Function name follows a common test naming pattern
def test_sar():
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point values can lead to precision issues
    """test TA.SAR"""
    # ⚠️ SAST Risk (Low): Assumes TA.TP and ohlc are defined and valid

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point values can lead to precision issues
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency
    # 🧠 ML Signal: Function definition for testing, indicating a test pattern
    sar = TA.SAR(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Assumes 'series' module is imported and 'Series' is a valid class
    # 🧠 ML Signal: Instantiation of a class from a module, indicating usage pattern
    assert isinstance(sar, series.Series)
    assert sar.values[-1] == 7127.15087821
# ⚠️ SAST Risk (Low): Lack of exception handling for potential errors in class instantiation
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues

# 🧠 ML Signal: Use of assert to validate expected outcomes in tests
# ✅ Best Practice: Use of isinstance to check the type of a variable

# 🧠 ML Signal: Function name follows a common pattern for test functions, useful for identifying test cases.
def test_psar():
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    """test TA.PSAR"""
    # ⚠️ SAST Risk (Low): Assumes `TA.CHAIKIN` and `ohlc` are defined and valid, potential for NameError if not.

    # 🧠 ML Signal: Use of method chaining, common in data processing libraries.
    # ✅ Best Practice: Function docstring provides a brief description of the test
    sar = TA.PSAR(ohlc).round(decimals=8)
    # ✅ Best Practice: Rounding to a fixed number of decimals for consistency in test results.

    assert isinstance(sar.psar, series.Series)
    # 🧠 ML Signal: Usage of TA.MFI function indicates a pattern for financial analysis
    # ⚠️ SAST Risk (Low): Assumes `series.Series` is defined, potential for NameError if not.
    assert sar.psar.values[-1] == 7113.5666702
# ✅ Best Practice: Rounding the result to a specific number of decimals for consistency
# 🧠 ML Signal: Type checking with `isinstance`, common pattern for ensuring correct data types.


# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
# ✅ Best Practice: Asserting the type of the result ensures expected behavior
def test_bbands():
    # 🧠 ML Signal: Use of assert statements, common in test functions to validate outcomes.
    # 🧠 ML Signal: Function name follows a common pattern for test functions
    """test TA.BBANDS"""
    # ⚠️ SAST Risk (Low): Assumes mfi.values is not empty, which could raise an IndexError

    # ✅ Best Practice: Asserting value range to ensure the result is within expected bounds
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    bb = TA.BBANDS(ohlc).round(decimals=8)
    # ✅ Best Practice: Use of method chaining for concise code

    assert isinstance(bb["BB_UPPER"], series.Series)
    # ⚠️ SAST Risk (Low): Lack of error handling for type checking
    # ✅ Best Practice: Use of descriptive variable names improves code readability
    assert isinstance(bb["BB_MIDDLE"], series.Series)
    assert isinstance(bb["BB_LOWER"], series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Checking the type of a variable can indicate expected data structures

    assert bb["BB_UPPER"].values[-1] == 8212.7979228
    # 🧠 ML Signal: Asserting specific values can indicate expected outcomes or invariants
    assert bb["BB_MIDDLE"].values[-1] == 7110.55082354
    # 🧠 ML Signal: Function name follows a common test naming pattern
    assert bb["BB_LOWER"].values[-1] == 6008.30372428

# ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated and sanitized before use
def test_mobo():
    # 🧠 ML Signal: Function definition with a specific naming pattern indicating a test function
    # 🧠 ML Signal: Use of TA.VZO indicates a pattern of technical analysis
    """test TA.mobo"""
    
    # ✅ Best Practice: Use of isinstance for type checking
    mbb = TA.MOBO(ohlc).round(decimals=8)
    # 🧠 ML Signal: Instantiation of a class from a module, indicating usage pattern

    # ✅ Best Practice: Asserting value range for expected output
    assert isinstance(mbb["BB_UPPER"], series.Series)
    # ⚠️ SAST Risk (Low): Lack of exception handling for the instantiation
    assert isinstance(mbb["BB_MIDDLE"], series.Series)
    # 🧠 ML Signal: Use of isinstance to check the type of an object
    assert isinstance(mbb["BB_LOWER"], series.Series)
    # 🧠 ML Signal: Testing function for TA.EFI, useful for learning test patterns

    # ⚠️ SAST Risk (Low): Lack of exception handling for the assertion
    assert mbb["BB_UPPER"].values[-1] == 6919.48336631
    # 🧠 ML Signal: Use of assert to validate a condition
    # ⚠️ SAST Risk (Low): Assumes TA.EFI and ohlc are defined and valid
    assert mbb["BB_MIDDLE"].values[-1] == 6633.75040888
    assert mbb["BB_LOWER"].values[-1] == 6348.01745146
# ⚠️ SAST Risk (Low): Lack of exception handling for the assertion
# ⚠️ SAST Risk (Low): Assumes efi is a series.Series object


# ⚠️ SAST Risk (Low): Assumes efi.values has enough elements and valid indices
def test_bbwidth():
    """test TA.BBWIDTH"""
    # ⚠️ SAST Risk (Low): Assumes efi.values has enough elements and valid indices
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    bb = TA.BBWIDTH(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Potential risk if `ohlc` is not validated or sanitized
    # ⚠️ SAST Risk (Low): Assumes efi.values has enough elements and valid indices

    # 🧠 ML Signal: Use of method chaining with `round` on the result of `TA.CFI`
    assert isinstance(bb, series.Series)
    # ⚠️ SAST Risk (Low): Assumes efi.values has enough elements and valid indices
    assert 0 < bb.values[-1] < 1
# ⚠️ SAST Risk (Low): Assumes `cfi` is always a `series.Series`, which may not be the case if `TA.CFI` changes
# 🧠 ML Signal: Function name follows a common pattern for test functions

# 🧠 ML Signal: Use of `assert` to validate the type of `cfi`

# ⚠️ SAST Risk (Low): Assumes 'TA.EBBP' and 'ohlc' are defined and safe
def test_percentb():
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # ✅ Best Practice: Use of 'round' for consistent decimal precision
    """test TA.PERCENT_B"""
    # 🧠 ML Signal: Use of `assert` to validate the last value of `cfi`

    # ⚠️ SAST Risk (Low): Assumes 'eb' has the expected structure with "Bull." key
    bb = TA.PERCENT_B(ohlc).round(decimals=8)
    # ✅ Best Practice: Use of 'isinstance' for type checking
    # 🧠 ML Signal: Testing function for TA.EMV, useful for learning test patterns

    assert isinstance(bb, series.Series)
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues
    # ⚠️ SAST Risk (Low): Assumes 'eb' has the expected structure with "Bear." key
    assert bb.values[-1] == 0.18695874
# ✅ Best Practice: Use of 'isinstance' for type checking
# ✅ Best Practice: Use descriptive variable names for clarity


# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable
# ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported
# 🧠 ML Signal: Function name follows a common pattern for test functions
def test_kc():
    # ✅ Best Practice: Use of assert for validation in test functions
    """test TA.KC"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues

    # 🧠 ML Signal: Assertion for expected output, useful for learning expected behavior
    # ✅ Best Practice: Use of 'round' for consistent decimal precision
    ma = TA.ZLEMA(ohlc, 20).round(decimals=8)
    kc = TA.KC(ohlc, MA=ma).round(decimals=8)
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the expected type to prevent runtime errors
    # 🧠 ML Signal: Usage of a method from a library (TA.BASP) which could indicate a pattern in financial data analysis.

    assert isinstance(kc["KC_UPPER"], series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # ⚠️ SAST Risk (Low): Potential risk if 'TA.BASP' or 'ohlc' are not properly validated or sanitized.
    assert isinstance(kc["KC_LOWER"], series.Series)

    # ⚠️ SAST Risk (Low): Potential risk if 'basp["Buy."]' is not properly validated or sanitized.
    assert kc["KC_UPPER"].values[-1] == 7014.74943624
    assert kc["KC_LOWER"].values[-1] == 5546.71157518
# ⚠️ SAST Risk (Low): Potential risk if 'basp["Sell."]' is not properly validated or sanitized.

# 🧠 ML Signal: Usage of TA.BASPN with rounding indicates a pattern for data preprocessing

# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
def test_do():
    # ⚠️ SAST Risk (Low): Lack of error handling for TA.BASPN function call
    """test TA.DO"""
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
    # 🧠 ML Signal: Checking type of 'Buy.' and 'Sell.' suggests expected data structure

    do = TA.DO(ohlc).round(decimals=8)

    # 🧠 ML Signal: Validation of specific values indicates expected output for test case
    assert isinstance(do["UPPER"], series.Series)
    # 🧠 ML Signal: Testing function for TA.CMO, useful for learning test patterns
    assert isinstance(do["MIDDLE"], series.Series)
    assert isinstance(do["LOWER"], series.Series)
    # ⚠️ SAST Risk (Low): Assumes ohlc is defined and valid, potential NameError if not

    # 🧠 ML Signal: Use of TA.CMO function, indicating usage of technical analysis library
    assert do["UPPER"].values[-1] == 7770.0
    assert do["MIDDLE"].values[-1] == 7010.0005000000001
    # 🧠 ML Signal: Use of TA.CHANDELIER indicates a pattern of using technical analysis indicators.
    # ⚠️ SAST Risk (Low): Assumes series is defined and valid, potential NameError if not
    assert do["LOWER"].values[-1] == 6250.0010000000002
# 🧠 ML Signal: Checking type of cmo, useful for learning type validation patterns

# ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized before use.

# 🧠 ML Signal: Validating range of cmo values, useful for learning validation patterns
# 🧠 ML Signal: Use of isinstance to check types is a common pattern for type validation.
def test_dmi():
    # ⚠️ SAST Risk (Low): Assumes cmo.values is a valid list, potential IndexError if empty
    """test TA.DMI"""
    # 🧠 ML Signal: Use of isinstance to check types is a common pattern for type validation.

    dmi = TA.DMI(ohlc).round(decimals=8)
    # 🧠 ML Signal: Use of assert to validate expected values is a common testing pattern.
    # 🧠 ML Signal: Function name follows a common pattern for test functions

    assert isinstance(dmi["DI+"], series.Series)
    # 🧠 ML Signal: Use of assert to validate expected values is a common testing pattern.
    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
    assert isinstance(dmi["DI-"], series.Series)
    # 🧠 ML Signal: Use of method chaining with round
    # 🧠 ML Signal: Use of pytest for testing indicates a pattern of test-driven development

    # ✅ Best Practice: Use of round for consistent decimal precision
    assert dmi["DI+"].values[-1] == 7.07135289
    # ✅ Best Practice: Using pytest.raises to assert exceptions is a good practice for testing error handling
    assert dmi["DI-"].values[-1] == 28.62895818
# 🧠 ML Signal: Function definition with a specific naming pattern indicating a test function
# 🧠 ML Signal: Use of isinstance to check object type

# ⚠️ SAST Risk (Low): Instantiating an object without checking input validation may lead to unexpected behavior

# 🧠 ML Signal: Use of assert to validate expected outcome
def test_adx():
    # ⚠️ SAST Risk (Low): Hardcoded value in test may lead to brittle tests
    # 🧠 ML Signal: Instantiation of a class with method chaining
    """test TA.ADX"""

    # ⚠️ SAST Risk (Low): Lack of error handling for potential exceptions during method calls
    adx = TA.ADX(ohlc).round(decimals=8)
    # 🧠 ML Signal: Type checking using isinstance

    assert isinstance(adx, series.Series)
    # 🧠 ML Signal: Type checking using isinstance
    assert adx.values[-1] == 46.43950615
# 🧠 ML Signal: Function name follows a common pattern for test functions

# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers

# ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized
def test_stoch():
    # 🧠 ML Signal: Use of method chaining with round, common in data processing
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers
    """test TA.STOCH"""

    # 🧠 ML Signal: Usage of TA.ICHIMOKU with specific parameters could indicate a pattern in financial data analysis.
    # ✅ Best Practice: Using isinstance to check the type of 'fish'
    st = TA.STOCH(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # ✅ Best Practice: Using isinstance to check the type of an object ensures that the object behaves as expected.
    assert isinstance(st, series.Series)
    # 🧠 ML Signal: Use of assert to validate expected outcomes in tests
    assert 0 < st.values[-1] < 100
# ✅ Best Practice: Using isinstance to check the type of an object ensures that the object behaves as expected.


# ✅ Best Practice: Using isinstance to check the type of an object ensures that the object behaves as expected.
def test_stochd():
    """test TA.STOCHD"""
    # ✅ Best Practice: Using isinstance to check the type of an object ensures that the object behaves as expected.

    # 🧠 ML Signal: Function definition for testing, useful for identifying test patterns
    st = TA.STOCHD(ohlc).round(decimals=8)
    # 🧠 ML Signal: Asserting specific values in the output could indicate expected patterns or thresholds in the data.

    assert isinstance(st, series.Series)
    # 🧠 ML Signal: Asserting specific values in the output could indicate expected patterns or thresholds in the data.
    # 🧠 ML Signal: Instantiation of TA.APZ with rounding, useful for understanding usage patterns
    assert 0 < st.values[-1] < 100

# 🧠 ML Signal: Asserting specific values in the output could indicate expected patterns or thresholds in the data.
# ⚠️ SAST Risk (Low): Lack of error handling for potential exceptions in isinstance checks

def test_stochrsi():
    # 🧠 ML Signal: Asserting specific values in the output could indicate expected patterns or thresholds in the data.
    # ⚠️ SAST Risk (Low): Lack of error handling for potential exceptions in isinstance checks
    """test TA.STOCRSI"""
    # 🧠 ML Signal: Function name follows a common test naming pattern

    # ⚠️ SAST Risk (Low): Lack of error handling for potential exceptions in equality check
    st = TA.STOCHRSI(ohlc).round(decimals=8)
    # 🧠 ML Signal: Use of assert to validate the type of the result

    # ✅ Best Practice: Check if the result is an instance of the expected type
    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100
# 🧠 ML Signal: Use of assert to validate the behavior of the result
# 🧠 ML Signal: Testing function for VPT calculation, useful for learning test patterns

# ✅ Best Practice: Ensure the last value in the series is as expected

# ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues
def test_williams():
    # ✅ Best Practice: Use of round() for consistent decimal precision
    """test TA.WILLIAMS"""

    # 🧠 ML Signal: Function name follows a common test naming pattern
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type and imported
    w = TA.WILLIAMS(ohlc).round(decimals=8)

    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can be unreliable
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated to prevent unexpected data issues
    assert isinstance(w, series.Series)
    # 🧠 ML Signal: Use of TA.FVE indicates a pattern of using technical analysis functions
    assert -100 < w.values[-1] < 0

# ✅ Best Practice: Asserting the type ensures the function returns the expected data structure
# 🧠 ML Signal: Function name follows a common test naming pattern

def test_uo():
    # ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated before use to prevent potential data issues
    # ✅ Best Practice: Asserting value range helps ensure the function's output is within expected bounds
    """test TA.UO"""
    # ✅ Best Practice: Use of method chaining for concise code

    uo = TA.UO(ohlc).round(decimals=8)
    # ⚠️ SAST Risk (Low): Ensure 'series.Series' is the correct type to prevent assertion errors
    # 🧠 ML Signal: Use of TA.PIVOT function indicates a pattern of financial data analysis.

    assert isinstance(uo, series.Series)
    # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues
    # 🧠 ML Signal: Checking type of pivot["pivot"] suggests a pattern of ensuring data structure integrity.
    assert 0 < uo.values[-1] < 100

# 🧠 ML Signal: Checking type of pivot["s1"] suggests a pattern of ensuring data structure integrity.

def test_ao():
    # 🧠 ML Signal: Checking type of pivot["s2"] suggests a pattern of ensuring data structure integrity.
    """test TA.AO"""

    # 🧠 ML Signal: Checking type of pivot["s3"] suggests a pattern of ensuring data structure integrity.
    ao = TA.AO(ohlc).round(decimals=8)

    # 🧠 ML Signal: Checking type of pivot["r1"] suggests a pattern of ensuring data structure integrity.
    assert isinstance(ao, series.Series)
    assert ao.values[-1] == -957.63459033
# 🧠 ML Signal: Checking type of pivot["r2"] suggests a pattern of ensuring data structure integrity.


# 🧠 ML Signal: Checking type of pivot["r3"] suggests a pattern of ensuring data structure integrity.
def test_mi():
    """test TA.MI"""
    # 🧠 ML Signal: Asserting specific value of pivot["pivot"] indicates a pattern of expected output validation.

    mi = TA.MI(ohlc).round(decimals=8)
    # 🧠 ML Signal: Asserting specific value of pivot["s1"] indicates a pattern of expected output validation.

    # 🧠 ML Signal: Testing function for TA.PIVOT_FIB, useful for learning test patterns
    assert isinstance(mi, series.Series)
    # 🧠 ML Signal: Asserting specific value of pivot["s2"] indicates a pattern of expected output validation.
    assert mi.values[-1] == 23.92808696
# ✅ Best Practice: Using isinstance to check the type of pivot["pivot"]

# 🧠 ML Signal: Asserting specific value of pivot["s3"] indicates a pattern of expected output validation.

# ✅ Best Practice: Using isinstance to check the type of pivot["s1"]
def test_bop():
    # 🧠 ML Signal: Asserting specific value of pivot["s4"] indicates a pattern of expected output validation.
    """test TA.BOP"""
    # ✅ Best Practice: Using isinstance to check the type of pivot["s2"]

    # 🧠 ML Signal: Asserting specific value of pivot["r1"] indicates a pattern of expected output validation.
    bop = TA.BOP(ohlc).round(decimals=8)
    # ✅ Best Practice: Using isinstance to check the type of pivot["s3"]

    # 🧠 ML Signal: Asserting specific value of pivot["r2"] indicates a pattern of expected output validation.
    assert isinstance(bop, series.Series)
    # ✅ Best Practice: Using isinstance to check the type of pivot["r1"]
    assert bop.values[-1] == 0.03045138
# 🧠 ML Signal: Asserting specific value of pivot["r3"] indicates a pattern of expected output validation.

# ✅ Best Practice: Using isinstance to check the type of pivot["r2"]

# 🧠 ML Signal: Asserting specific value of pivot["r4"] indicates a pattern of expected output validation.
def test_vortex():
    # ✅ Best Practice: Using isinstance to check the type of pivot["r3"]
    """test TA.VORTEX"""

    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    v = TA.VORTEX(ohlc).round(decimals=8)

    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    assert isinstance(v["VIp"], series.Series)
    # 🧠 ML Signal: Use of a method from a specific library (TA) indicates a pattern of financial data analysis.
    assert isinstance(v["VIm"], series.Series)
    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns

    # ⚠️ SAST Risk (Low): Potential risk if 'ohlc' is not validated or sanitized before use.
    assert v["VIp"].values[-1] == 0.76856105
    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    # 🧠 ML Signal: Use of 'round' method indicates a pattern of precision control in numerical computations.
    assert v["VIm"].values[-1] == 1.27305188

# 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
# ⚠️ SAST Risk (Low): Assumes 'msd' is always a 'series.Series', which might not be the case if input changes.
# 🧠 ML Signal: Function definition with a specific naming pattern indicating a test function

# 🧠 ML Signal: Use of 'assert' indicates a pattern of test-driven development or validation.
def test_kst():
    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    # ⚠️ SAST Risk (Low): Assumes TA.STC and ohlc are defined and valid, potential for NameError
    """test TA.KST"""
    # ⚠️ SAST Risk (Low): Hardcoded value in test can lead to false positives/negatives if data changes.
    # ✅ Best Practice: Chaining method calls for concise code

    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    # ✅ Best Practice: Consider using a tolerance range for floating-point comparison.
    kst = TA.KST(ohlc).round(decimals=8)
    # 🧠 ML Signal: Testing function for TA.EVSTC, useful for learning test patterns
    # ⚠️ SAST Risk (Low): Assumes series.Series is the correct type, potential for AssertionError

    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    # ✅ Best Practice: Use of isinstance for type checking
    assert isinstance(kst["KST"], series.Series)
    # ⚠️ SAST Risk (Low): Assumes ohlc is defined and valid, potential for NameError
    assert isinstance(kst["signal"], series.Series)
    # 🧠 ML Signal: Asserting specific values, useful for learning expected output patterns
    # ✅ Best Practice: Use of round() for consistent decimal precision
    # ⚠️ SAST Risk (Low): Assumes stc.values is a valid sequence, potential for IndexError
    # 🧠 ML Signal: Function definition with a specific naming pattern indicating a test function

    # ✅ Best Practice: Asserting value range for expected output
    assert kst["KST"].values[-1] == -157.42229442
    # ⚠️ SAST Risk (Low): Assumes series is imported and valid, potential for NameError
    assert kst["signal"].values[-1] == -132.10367593
# ✅ Best Practice: Use of isinstance() to ensure correct type
# 🧠 ML Signal: Calling a method from a module with a specific pattern


# ✅ Best Practice: Boundary check to ensure values are within expected range
# ⚠️ SAST Risk (Low): Lack of error handling for the function call
def test_tsi():
    # 🧠 ML Signal: Type checking using isinstance
    """test TA.TSI"""

    # 🧠 ML Signal: Type checking using isinstance
    tsi = TA.TSI(ohlc).round(decimals=8)
    # 🧠 ML Signal: Function name follows a common test naming pattern, useful for identifying test functions.

    # ⚠️ SAST Risk (Low): Direct access to list elements without bounds checking
    assert isinstance(tsi["TSI"], series.Series)
    # ⚠️ SAST Risk (Low): Assumes 'TA.VC' and 'ohlc' are safe and correctly implemented.
    assert isinstance(tsi["signal"], series.Series)
    # ⚠️ SAST Risk (Low): Direct access to list elements without bounds checking
    # ✅ Best Practice: Rounding to a fixed number of decimals improves consistency in floating-point operations.

    # ✅ Best Practice: Include a docstring to describe the purpose of the test function
    assert tsi["TSI"].values[-1] == -32.12837201
    # ⚠️ SAST Risk (Low): Assumes 'vc' contains the expected keys and types.
    assert tsi["signal"].values[-1] == -26.94173827

# 🧠 ML Signal: Usage of a specific TA function with parameters
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.

# ✅ Best Practice: Use of assert to validate the type of the result
# ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
# ⚠️ SAST Risk (Low): Ensure 'ohlc' is validated and sanitized before use
def test_tp():
    """test TA.TP"""

    tp = TA.TP(ohlc).round(decimals=8)

    assert isinstance(tp, series.Series)
    assert tp.values[-1] == 6429.01772876


def test_adl():

    adl = TA.ADL(ohlc).round(decimals=8)

    assert isinstance(adl, series.Series)
    assert adl.values[-1] == 303320.96403697


def test_chaikin():
    """test TA.CHAIKIN"""

    c = TA.CHAIKIN(ohlc).round(decimals=8)

    assert isinstance(c, series.Series)
    assert c.values[-1] == -378.66969549


def test_mfi():
    """test TA.MFI"""

    mfi = TA.MFI(ohlc).round(decimals=8)

    assert isinstance(mfi, series.Series)
    assert 0 < mfi.values[-1] < 100


def test_obv():
    """test TA.OBV"""

    o = TA.OBV(ohlc).round(decimals=8)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -6726.6904375


def test_wobv():
    """test TA.OBV"""

    o = TA.WOBV(ohlc).round(decimals=8)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -85332065.01331231


def test_vzo():
    """test TA.VZO"""

    vzo = TA.VZO(ohlc)

    assert isinstance(vzo, series.Series)
    assert -85 < vzo.values[-1] < 85


def test_pzo():
    """test TA.PZO"""

    pzo = TA.PZO(ohlc)

    assert isinstance(pzo, series.Series)
    assert -85 < pzo.values[-1] < 85


def test_efi():
    """test TA.EFI"""

    efi = TA.EFI(ohlc)

    assert isinstance(efi, series.Series)
    assert efi.values[1] > 0
    assert efi.values[2] > 0

    assert efi.values[-2] < 0
    assert efi.values[-1] < 0


def test_cfi():
    """test TA.CFI"""

    cfi = TA.CFI(ohlc).round(decimals=8)

    assert isinstance(cfi, series.Series)
    assert cfi.values[-1] == -84856289.556287795


def test_ebbp():
    """test TA.EBBP"""

    eb = TA.EBBP(ohlc).round(decimals=8)

    assert isinstance(eb["Bull."], series.Series)
    assert isinstance(eb["Bear."], series.Series)
    assert eb["Bull."].values[-1] == -285.40231904


def test_emv():
    """test TA.EMV"""

    emv = TA.EMV(ohlc).round(decimals=1)

    assert isinstance(emv, series.Series)
    assert emv.values[-1] == -26103140.8
                             
def test_cci():
    """test TA.CCI"""

    cci = TA.CCI(ohlc).round(decimals=8)

    assert isinstance(cci, series.Series)
    assert cci.values[-1] == -91.76341956


def test_basp():
    """test TA.BASP"""

    basp = TA.BASP(ohlc).round(decimals=8)

    assert isinstance(basp["Buy."], series.Series)
    assert isinstance(basp["Sell."], series.Series)

    assert basp["Buy."].values[-1] == 0.06691681
    assert basp["Sell."].values[-1] == 0.0914869


def test_baspn():
    """test TA.BASPN"""

    basp = TA.BASPN(ohlc).round(decimals=8)

    assert isinstance(basp["Buy."], series.Series)
    assert isinstance(basp["Sell."], series.Series)

    assert basp["Buy."].values[-1] == 0.56374213
    assert basp["Sell."].values[-1] == 0.74103021


def test_cmo():
    """test TA.CMO"""

    cmo = TA.CMO(ohlc)

    assert isinstance(cmo, series.Series)
    assert -100 < cmo.values[-1] < 100


def test_chandelier():
    """test TA.CHANDELIER"""

    chan = TA.CHANDELIER(ohlc).round(decimals=8)

    assert isinstance(chan["Long."], series.Series)
    assert isinstance(chan["Short."], series.Series)

    assert chan["Long."].values[-1] == 6801.59276465
    assert chan["Short."].values[-1] == 7091.40723535


def test_qstick():
    """test TA.QSTICK"""

    q = TA.QSTICK(ohlc).round(decimals=8)

    assert isinstance(q, series.Series)
    assert q.values[-1] == 0.24665616


def test_tmf():

    with pytest.raises(NotImplementedError):
        tmf = TA.TMF(ohlc)


def test_wto():
    """test TA.WTO"""

    wto = TA.WTO(ohlc).round(decimals=8)

    assert isinstance(wto["WT1."], series.Series)
    assert isinstance(wto["WT2."], series.Series)

    assert wto["WT1."].values[-1] == -60.29006991
    assert wto["WT2."].values[-1] == -61.84105024


def test_fish():
    """test TA.FISH"""

    fish = TA.FISH(ohlc).round(decimals=8)

    assert isinstance(fish, series.Series)
    assert fish.values[-1] == -2.29183153


def test_ichimoku():
    """test TA.ICHIMOKU"""

    ichi = TA.ICHIMOKU(ohlc, 10, 25).round(decimals=8)

    assert isinstance(ichi["TENKAN"], series.Series)
    assert isinstance(ichi["KIJUN"], series.Series)
    assert isinstance(ichi["SENKOU"], series.Series)
    assert isinstance(ichi["CHIKOU"], series.Series)

    assert ichi["TENKAN"].values[-1] == 6911.5 
    assert ichi["KIJUN"].values[-1] == 6946.5
    assert ichi["SENKOU"].values[-1] == 8243.0 
    assert ichi["CHIKOU"].values[-27] == 6420.45318629


def test_apz():
    """test TA.APZ"""

    apz = TA.APZ(ohlc).round(decimals=8)

    assert isinstance(apz["UPPER"], series.Series)
    assert isinstance(apz["LOWER"], series.Series)

    assert apz["UPPER"].values[-1] == 7193.97725794


def test_sqzmi():
    """test TA.SQZMI"""

    sqz = TA.SQZMI(ohlc)

    assert isinstance(sqz, series.Series)

    assert not sqz.values[-1]


def test_vpt():
    """test TA.VPT"""

    vpt = TA.VPT(ohlc).round(decimals=8)

    assert isinstance(vpt, series.Series)
    assert vpt.values[-1] == 94068.85032709


def test_fve():
    """test TA.FVE"""

    fve = TA.FVE(ohlc)

    assert isinstance(fve, series.Series)
    assert -100 < fve.values[-1] < 100


def test_vfi():
    """test TA.VFI"""

    vfi = TA.VFI(ohlc).round(decimals=8)

    assert isinstance(vfi, series.Series)
    assert vfi.values[-1] == -6.49159549


def test_pivot():
    """test TA.PIVOT"""

    pivot = TA.PIVOT(ohlc).round(decimals=8)

    assert isinstance(pivot["pivot"], series.Series)
    assert isinstance(pivot["s1"], series.Series)
    assert isinstance(pivot["s2"], series.Series)
    assert isinstance(pivot["s3"], series.Series)
    assert isinstance(pivot["r1"], series.Series)
    assert isinstance(pivot["r2"], series.Series)
    assert isinstance(pivot["r3"], series.Series)

    assert pivot["pivot"].values[-1] == 6467.40629761

    assert pivot["s1"].values[-1] == 6364.00470239
    assert pivot["s2"].values[-1] == 6311.00940479
    assert pivot["s3"].values[-1] == 6207.60780957
    assert pivot["s4"].values[-1] == 6104.20621436

    assert pivot["r1"].values[-1] == 6520.40159521
    assert pivot["r2"].values[-1] == 6623.80319043
    assert pivot["r3"].values[-1] == 6676.79848803
    assert pivot["r4"].values[-1] == 6729.79378564


def test_pivot_fib():
    """test TA.PIVOT_FIB"""

    pivot = TA.PIVOT_FIB(ohlc).round(decimals=8)

    assert isinstance(pivot["pivot"], series.Series)
    assert isinstance(pivot["s1"], series.Series)
    assert isinstance(pivot["s2"], series.Series)
    assert isinstance(pivot["s3"], series.Series)
    assert isinstance(pivot["r1"], series.Series)
    assert isinstance(pivot["r2"], series.Series)
    assert isinstance(pivot["r3"], series.Series)

    assert pivot["pivot"].values[-1] == 6467.40629761

    assert pivot["s1"].values[-1] == 6407.66268455
    assert pivot["s2"].values[-1] == 6370.75301784
    assert pivot["s3"].values[-1] == 6311.00940479
    assert pivot["s4"].values[-1] == 6251.26579173

    assert pivot["r1"].values[-1] == 6527.14991066
    assert pivot["r2"].values[-1] == 6564.05957737
    assert pivot["r3"].values[-1] == 6623.80319043
    assert pivot["r4"].values[-1] == 6683.54680348


def test_msd():
    """test TA.MSD"""

    msd = TA.MSD(ohlc).round(decimals=8)

    assert isinstance(msd, series.Series)
    assert msd.values[-1] == 542.25201592


def test_stc():
    """test TA.STC"""

    stc = TA.STC(ohlc).round(decimals=2)

    assert isinstance(stc, series.Series)
    assert 0 <= stc.values[-1] <= 100


def test_evstc():
    """test TA.EVSTC"""

    stc = TA.EVSTC(ohlc).round(decimals=2)

    assert isinstance(stc, series.Series)
    assert 0 <= stc.values[-1] <= 100


def test_williams_fractal():
    """test TA.WILLIAMS_FRACTAL"""

    fractals = TA.WILLIAMS_FRACTAL(ohlc)

    assert isinstance(fractals["BullishFractal"], series.Series)
    assert isinstance(fractals["BearishFractal"], series.Series)
    assert fractals.BearishFractal.values[-3] == 0
    assert fractals.BullishFractal.values[-3] == 0


def test_vc():
    """test TA.VC"""

    vc = TA.VC(ohlc).round(decimals=8)

    assert isinstance(vc["Value Chart Open"], series.Series)
    assert vc.values[-1][0] == 0.50469864
    assert vc.values[-1][-1] == -0.87573258

def test_sma():
    """test TA.WAVEPM"""

    wavepm = TA.WAVEPM(ohlc, 14, 100, "close").round(decimals=8)

    assert isinstance(wavepm, series.Series)
    assert wavepm.values[-1] == 0.83298565