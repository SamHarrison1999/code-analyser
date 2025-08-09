# Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
# Licensed under the MIT License.

# 🧠 ML Signal: Importing a library indicates usage patterns
from arctic.arctic import Arctic
# ✅ Best Practice: Inheriting from unittest.TestCase to create a test class
import qlib
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns
# 🧠 ML Signal: Importing unittest suggests testing practices
from qlib.data import D
import unittest


class TestClass(unittest.TestCase):
    """
    Useful commands
    - run all tests: pytest examples/orderbook_data/example.py
    - run a single test:  pytest -s --pdb --disable-warnings examples/orderbook_data/example.py::TestClass::test_basic01
    """
    # 🧠 ML Signal: Usage of qlib.init with specific parameters can indicate a pattern for initializing a data provider
    # ⚠️ SAST Risk (Low): Hardcoded provider URI may expose sensitive paths or configurations
    # ✅ Best Practice: Use constants or configuration files for repeated values like mem_cache_size_limit
    # 🧠 ML Signal: Custom expression provider configuration can indicate specific data processing needs

    def setUp(self):
        """
        Configure for arctic
        """
        provider_uri = "~/.qlib/qlib_data/yahoo_cn_1min"
        qlib.init(
            provider_uri=provider_uri,
            mem_cache_size_limit=1024**3 * 2,
            mem_cache_type="sizeof",
            kernels=1,
            expression_provider={"class": "LocalExpressionProvider", "kwargs": {"time2idx": False}},
            feature_provider={
                "class": "ArcticFeatureProvider",
                "module_path": "qlib.contrib.data.data",
                "kwargs": {"uri": "127.0.0.1"},
            },
            dataset_provider={
                # 🧠 ML Signal: Custom feature provider configuration can indicate specific feature extraction needs
                # 🧠 ML Signal: Custom dataset provider configuration can indicate specific dataset handling requirements
                "class": "LocalDatasetProvider",
                "kwargs": {
                    # 🧠 ML Signal: Function name suggests this is a test case, useful for identifying test patterns
                    "align_time": False,  # Order book is not fixed, so it can't be align to a shared fixed frequency calendar
                },
            },
        )
        # self.stocks_list = ["SH600519"]
        self.stocks_list = ["SZ000725"]

    # 🧠 ML Signal: Initialization of stocks_list with specific stock codes can indicate a pattern for stock selection
    def test_basic(self):
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        # NOTE: this data contains a lot of zeros in $askX and $bidX
        # 🧠 ML Signal: Use of a method named 'features' suggests feature extraction or data transformation
        df = D.features(
            self.stocks_list,
            # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
            fields=["$ask1", "$ask2", "$bid1", "$bid2"],
            # 🧠 ML Signal: Use of a method that extracts features, indicating data processing or transformation
            # 🧠 ML Signal: Function definition with a specific naming pattern, useful for identifying test functions
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic_without_time(self):
        # 🧠 ML Signal: Use of a specific resampling method, indicating time series data manipulation
        df = D.features(self.stocks_list, fields=["$ask1"], freq="ticks")
        print(df)
    # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
    # 🧠 ML Signal: Use of a method named 'features' suggests a pattern for feature extraction in data processing
    # 🧠 ML Signal: Use of 'self.stocks_list' indicates a pattern of using instance variables for data input

    def test_basic01(self):
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230",
            # 🧠 ML Signal: Use of 'fields' parameter suggests a pattern for selecting specific data attributes
            # 🧠 ML Signal: Use of 'freq' parameter indicates a pattern for specifying data granularity
            end_time="20210101",
        # 🧠 ML Signal: Use of 'start_time' and 'end_time' parameters suggests a pattern for time-bounded data queries
        )
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        # 🧠 ML Signal: Usage of a method that fetches features from a data source
        print(df)

    def test_basic02(self):
        df = D.features(
            self.stocks_list,
            fields=["$function_code"],
            freq="transaction",
            start_time="20201230",
            # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
            end_time="20210101",
        # 🧠 ML Signal: Expression pattern for financial data resampling
        # 🧠 ML Signal: Dynamic generation of expressions for data resampling
        )
        print(df)

    def test_basic03(self):
        df = D.features(
            self.stocks_list,
            # 🧠 ML Signal: Use of string formatting and list comprehension
            fields=["$function_code"],
            # ✅ Best Practice: Use of list comprehension for concise and efficient iteration
            freq="order",
            start_time="20201230",
            # ✅ Best Practice: Use of @staticmethod decorator for methods that do not access instance data
            # 🧠 ML Signal: String manipulation and dynamic variable naming pattern
            end_time="20210101",
        # ⚠️ SAST Risk (Low): Potential risk if 'name' or 'method' are derived from untrusted input
        )
        # ✅ Best Practice: Use of f-string for clearer and more efficient string formatting
        print(df)

    # 🧠 ML Signal: Iterating over a fixed range to generate expressions
    # Here are some popular expressions for high-frequency
    # 1) some shared expression
    # 🧠 ML Signal: Constructing dynamic column names based on loop variables
    expr_sum_buy_ask_1 = "(TResample($ask1, '1min', 'last') + TResample($bid1, '1min', 'last'))"
    total_volume = (
        # ⚠️ SAST Risk (Low): Potential risk if `self.stocks_list` or `exprs` contain untrusted data
        "TResample("
        + "+".join([f"${name}{i}" for i in range(1, 11) for name in ["asize", "bsize"]])
        # ✅ Best Practice: Assigning meaningful column names to the DataFrame
        # 🧠 ML Signal: Use of lambda functions for dynamic string generation
        + ", '1min', 'sum')"
    )

    # ⚠️ SAST Risk (Low): Printing DataFrame can expose sensitive data in logs
    # 🧠 ML Signal: Use of lambda functions for dynamic string generation
    @staticmethod
    def total_func(name, method):
        return "TResample(" + "+".join([f"${name}{i}" for i in range(1, 11)]) + ",'1min', '{}')".format(method)

    def test_exp_01(self):
        # ✅ Best Practice: Initialize lists before loops for better readability
        exprs = []
        names = []
        for name in ["asize", "bsize"]:
            for i in range(1, 11):
                # 🧠 ML Signal: Use of extend method to add multiple items to a list
                exprs.append(f"TResample(${name}{i}, '1min', 'mean') / ({self.total_volume})")
                names.append(f"v_{name}_{i}")
        # 🧠 ML Signal: Use of extend method to add multiple items to a list
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        # ⚠️ SAST Risk (Low): Potential risk if D.features is not properly validated or sanitized
        # 🧠 ML Signal: Use of lambda function for dynamic string formatting
        df.columns = names
        print(df)

    # ✅ Best Practice: Assign meaningful column names for better data frame readability
    # ✅ Best Practice: Consider using a named function instead of a lambda for better readability
    # 2) some often used papers;
    def test_exp_02(self):
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        spread_func = (
            lambda index: f"2 * TResample($ask{index} - $bid{index}, '1min', 'last') / {self.expr_sum_buy_ask_1}"
        # 🧠 ML Signal: Iterating over a range to generate expressions
        )
        mid_func = (
            # 🧠 ML Signal: Dynamic generation of variable names
            lambda index: f"2 * TResample(($ask{index} + $bid{index})/2, '1min', 'last') / {self.expr_sum_buy_ask_1}"
        )
        # ⚠️ SAST Risk (Low): Potential for index out of range if list is empty before extend

        exprs = []
        # ⚠️ SAST Risk (Low): Potential for index out of range if list is empty before extend
        names = []
        for i in range(1, 11):
            # 🧠 ML Signal: Use of external library function with dynamic parameters
            exprs.extend([spread_func(i), mid_func(i)])
            # 🧠 ML Signal: Iterating over a list of strings to generate expressions dynamically
            names.extend([f"p_spread_{i}", f"p_mid_{i}"])
        # 🧠 ML Signal: Assigning dynamic column names to a DataFrame
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        # 🧠 ML Signal: Appending dynamically generated names to a list
        df.columns = names
        # 🧠 ML Signal: Outputting DataFrame to console
        print(df)
    # 🧠 ML Signal: Using a method to generate features based on dynamic expressions

    def test_exp_03(self):
        # ⚠️ SAST Risk (Low): Printing DataFrame can expose sensitive data in logs
        # 🧠 ML Signal: Renaming DataFrame columns based on dynamically generated names
        expr3_func1 = (
            lambda name, index_left, index_right: f"2 * TResample(Abs(${name}{index_left} - ${name}{index_right}), '1min', 'last') / {self.expr_sum_buy_ask_1}"
        )
        for name in ["ask", "bid"]:
            # 🧠 ML Signal: Use of formatted strings to dynamically create expressions
            for i in range(1, 10):
                exprs = [expr3_func1(name, i + 1, i)]
                # 🧠 ML Signal: Use of a DataFrame to store and manipulate data
                names = [f"p_diff_{name}_{i}_{i+1}"]
        exprs.extend([expr3_func1("ask", 10, 1), expr3_func1("bid", 1, 10)])
        # ✅ Best Practice: Assigning meaningful column names to a DataFrame
        names.extend(["p_diff_ask_10_1", "p_diff_bid_1_10"])
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        # 🧠 ML Signal: Use of lambda function to dynamically generate expressions
        df.columns = names
        print(df)

    def test_exp_04(self):
        # 🧠 ML Signal: Initialization of lists to store expressions and names
        exprs = []
        names = []
        for name in ["asize", "bsize"]:
            # 🧠 ML Signal: Iterating over a fixed range and list of names
            exprs.append(f"(({ self.total_func(name, 'mean')}) / 10) / {self.total_volume}")
            names.append(f"v_avg_{name}")

        # 🧠 ML Signal: Appending formatted strings to a list
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_05(self):
        exprs = [
            f"2 * Sub({ self.total_func('ask', 'last')}, {self.total_func('bid', 'last')})/{self.expr_sum_buy_ask_1}",
            f"Sub({ self.total_func('asize', 'mean')}, {self.total_func('bsize', 'mean')})/{self.total_volume}",
        ]
        # ⚠️ SAST Risk (Low): Potential risk if D.features is not properly validated
        # 🧠 ML Signal: Function definition with parameters indicating a pattern for data processing
        names = ["p_accspread", "v_accspread"]

        # 🧠 ML Signal: Assigning new column names to a DataFrame
        # ⚠️ SAST Risk (Low): Potential injection risk if inputs are not validated or sanitized
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data
        # 🧠 ML Signal: Use of lambda functions for dynamic string generation
        # 🧠 ML Signal: Use of string formatting with dynamic inputs
        # ✅ Best Practice: Use f-string for clear and concise string interpolation
        df.columns = names
        print(df)

    # ⚠️ SAST Risk (Low): Potential for code injection if inputs are not sanitized
    #  (p|v)_diff_(ask|bid|asize|bsize)_(time_interval)
    def test_exp_06(self):
        # 🧠 ML Signal: Pattern of using lambda functions to generate expressions
        t = 3
        expr6_price_func = (
            # 🧠 ML Signal: Use of descriptive variable names for data columns
            lambda name, index, method: f'2 * (TResample(${name}{index}, "{t}s", "{method}") - Ref(TResample(${name}{index}, "{t}s", "{method}"), 1)) / {t}'
        )
        # 🧠 ML Signal: Use of a method to generate features from a list of expressions
        exprs = []
        # 🧠 ML Signal: Use of lambda function to dynamically create expressions
        # ✅ Best Practice: Assigning meaningful column names to a DataFrame
        names = []
        for i in range(1, 11):
            for name in ["bid", "ask"]:
                # 🧠 ML Signal: Use of print statements for debugging or output
                exprs.append(
                    # 🧠 ML Signal: Use of lists to store expressions and names
                    f"TResample({expr6_price_func(name, i, 'last')}, '1min', 'mean') / {self.expr_sum_buy_ask_1}"
                # 🧠 ML Signal: Use of dictionary for mapping or translation
                )
                names.append(f"p_diff_{name}{i}_{t}s")

        for i in range(1, 11):
            # 🧠 ML Signal: Appending dynamically generated expressions to a list
            for name in ["asize", "bsize"]:
                exprs.append(f"TResample({expr6_price_func(name, i, 'mean')}, '1min', 'mean') / {self.total_volume}")
                # 🧠 ML Signal: Appending dynamically generated names to a list
                names.append(f"v_diff_{name}{i}_{t}s")

        # ⚠️ SAST Risk (Low): Potential risk if D.features is not properly validated or sanitized
        # 🧠 ML Signal: Function definition with parameters indicating a pattern for processing or transforming data
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        # ⚠️ SAST Risk (Low): Potential for code injection if inputs are not properly sanitized
        # ✅ Best Practice: Assigning meaningful column names to DataFrame
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and behavior of the test function.
        print(df)
    # ⚠️ SAST Risk (Low): Printing DataFrame can expose sensitive data in logs
    # ✅ Best Practice: Use of f-string for readability and performance
    # 🧠 ML Signal: Use of f-string for dynamic string formatting
    # ✅ Best Practice: Use a more descriptive name for the lambda function to improve readability.

    # TODOs:
    # Following expressions may be implemented in the future
    # expr7_2 = lambda funccode, bsflag, time_interval: \
    #     "TResample(TRolling(TEq(@transaction.function_code,  {}) & TEq(@transaction.bs_flag ,{}), '{}s', 'sum') / \
    #     TRolling(@transaction.function_code, '{}s', 'count') , '1min', 'mean')".format(ord(funccode), bsflag,time_interval,time_interval)
    # create_dataset(7, "SH600000", [expr7_2("C")] + [expr7(funccode, ordercode) for funccode in ['B','S'] for ordercode in ['0','1']])
    # 🧠 ML Signal: Iterating over fixed sets of values for 'funccode' and 'ordercode'.
    # create_dataset(7,  ["SH600000"], [expr7_2("C", 48)] )

    @staticmethod
    # 🧠 ML Signal: Appending generated expressions to a list.
    def expr7_init(funccode, ordercode, time_interval):
        # NOTE: based on on order frequency (i.e. freq="order")
        # 🧠 ML Signal: Appending generated names to a list.
        return f"Rolling(Eq($function_code,  {ord(funccode)}) & Eq($order_kind ,{ord(ordercode)}), '{time_interval}s', 'sum') / Rolling($function_code, '{time_interval}s', 'count')"
    # 🧠 ML Signal: Use of lambda functions for dynamic string formatting
    # ⚠️ SAST Risk (Low): Ensure 'self.stocks_list' and 'exprs' are properly validated to prevent injection attacks.

    # (la|lb|ma|mb|ca|cb)_intensity_(time_interval)
    def test_exp_07_1(self):
        # ⚠️ SAST Risk (Low): Ensure 'names' list matches the number of columns in 'df' to avoid potential errors.
        # NOTE: based on transaction frequency (i.e. freq="transaction")
        # 🧠 ML Signal: Use of list to store expressions for further processing
        expr7_3 = (
            # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs.
            lambda funccode, code, time_interval: f"TResample(Rolling(Eq($function_code,  {ord(funccode)}) & {code}($ask_order, $bid_order) , '{time_interval}s', 'sum')   / Rolling($function_code, '{time_interval}s', 'count') , '1min', 'mean')"
        # 🧠 ML Signal: Use of list to store column names for DataFrame
        )

        # ⚠️ SAST Risk (Low): Potential risk if D.features is not properly validated or sanitized
        exprs = [expr7_3("C", "Gt", "3"), expr7_3("C", "Lt", "3")]
        # ✅ Best Practice: Explicitly setting DataFrame column names for clarity
        # 🧠 ML Signal: Use of f-strings for dynamic expression generation
        names = ["ca_intensity_3s", "cb_intensity_3s"]

        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        # ✅ Best Practice: Use of print for debugging or output verification
        print(df)
    # 🧠 ML Signal: Use of descriptive variable names for mapping expressions to names

    trans_dict = {"B": "a", "S": "b", "0": "l", "1": "m"}
    # 🧠 ML Signal: Use of a method to generate features from expressions

    # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the test_exp_09_order method.
    def test_exp_07_2(self):
        # ✅ Best Practice: Explicitly setting DataFrame column names for clarity
        # NOTE: based on on order frequency
        # ✅ Best Practice: Initialize lists outside of loops to avoid repeated initialization.
        expr7 = (
            # ⚠️ SAST Risk (Low): Potential exposure of sensitive data through print statements
            lambda funccode, ordercode, time_interval: f"TResample({self.expr7_init(funccode, ordercode, time_interval)}, '1min', 'mean')"
        )
        # 🧠 ML Signal: Iterating over combinations of codes suggests a pattern for generating expressions.

        exprs = []
        names = []
        # 🧠 ML Signal: Use of f-strings for dynamic expression generation.
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(expr7(funccode, ordercode, "3"))
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_intensity_3s")
        # 🧠 ML Signal: Dynamic naming based on dictionary lookups and string concatenation.
        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        # ⚠️ SAST Risk (Low): Ensure that self.stocks_list and exprs are validated to prevent injection or unexpected behavior.
        print(df)

    # 🧠 ML Signal: Use of string formatting to create expressions dynamically
    # ✅ Best Practice: Ensure that the length of names matches the number of columns in df to avoid potential errors.
    @staticmethod
    def expr7_3_init(funccode, code, time_interval):
        # NOTE: It depends on transaction frequency
        # ✅ Best Practice: Consider using logging instead of print for better control over output in production environments.
        return f"Rolling(Eq($function_code,  {ord(funccode)}) & {code}($ask_order, $bid_order) , '{time_interval}s', 'sum') / Rolling($function_code, '{time_interval}s', 'count')"
    # ⚠️ SAST Risk (Low): Potential for incorrect list comprehension inside append

    # (la|lb|ma|mb|ca|cb)_relative_intensity_(time_interval_small)_(time_interval_big)
    # 🧠 ML Signal: Use of a custom method to generate features
    def test_exp_08_1(self):
        expr8_1 = (
            # ⚠️ SAST Risk (Low): Potential mismatch between df columns and names list
            # ✅ Best Practice: Use logging instead of print for better control over output
            # ✅ Best Practice: Use of unittest framework for testing
            lambda funccode, ordercode, time_interval_short, time_interval_long: f"TResample(Gt({self.expr7_init(funccode, ordercode, time_interval_short)},{self.expr7_init(funccode, ordercode, time_interval_long)}), '1min', 'mean')"
        )

        exprs = []
        names = []
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(expr8_1(funccode, ordercode, "10", "900"))
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_relative_intensity_10s_900s")

        df = D.features(self.stocks_list, fields=exprs, freq="order")
        df.columns = names
        print(df)

    def test_exp_08_2(self):
        # NOTE: It depends on transaction frequency
        expr8_2 = (
            lambda funccode, ordercode, time_interval_short, time_interval_long: f"TResample(Gt({self.expr7_3_init(funccode, ordercode, time_interval_short)},{self.expr7_3_init(funccode, ordercode, time_interval_long)}), '1min', 'mean')"
        )

        exprs = [expr8_2("C", "Gt", "10", "900"), expr8_2("C", "Lt", "10", "900")]
        names = ["ca_relative_intensity_10s_900s", "cb_relative_intensity_10s_900s"]

        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    ## v9(la|lb|ma|mb|ca|cb)_diff_intensity_(time_interval1)_(time_interval2)
    # 1) calculating the original data
    # 2) Resample data to 3s and calculate the changing rate
    # 3) Resample data to 1min

    def test_exp_09_trans(self):
        exprs = [
            f'TResample(Div(Sub(TResample({self.expr7_3_init("C", "Gt", "3")}, "3s", "last"), Ref(TResample({self.expr7_3_init("C", "Gt", "3")}, "3s","last"), 1)), 3), "1min", "mean")',
            f'TResample(Div(Sub(TResample({self.expr7_3_init("C", "Lt", "3")}, "3s", "last"), Ref(TResample({self.expr7_3_init("C", "Lt", "3")}, "3s","last"), 1)), 3), "1min", "mean")',
        ]
        names = ["ca_diff_intensity_3s_3s", "cb_diff_intensity_3s_3s"]
        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    def test_exp_09_order(self):
        exprs = []
        names = []
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(
                    f'TResample(Div(Sub(TResample({self.expr7_init(funccode, ordercode, "3")}, "3s", "last"), Ref(TResample({self.expr7_init(funccode, ordercode, "3")},"3s", "last"), 1)), 3) ,"1min", "mean")'
                )
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_diff_intensity_3s_3s")
        df = D.features(self.stocks_list, fields=exprs, freq="order")
        df.columns = names
        print(df)

    def test_exp_10(self):
        exprs = []
        names = []
        for i in [5, 10, 30, 60]:
            exprs.append(
                f'TResample(Ref(TResample($ask1 + $bid1, "1s", "ffill"), {-i}) / TResample($ask1 + $bid1, "1s", "ffill") - 1, "1min", "mean" )'
            )
            names.append(f"lag_{i}_change_rate" for i in [5, 10, 30, 60])
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)


if __name__ == "__main__":
    unittest.main()