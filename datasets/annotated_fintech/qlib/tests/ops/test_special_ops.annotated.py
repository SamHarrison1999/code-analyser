import unittest
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns for those modules

from qlib.data import D
# 🧠 ML Signal: Importing specific classes from a library indicates usage patterns for those classes
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.ops import ChangeInstrument, Cov, Feature, Ref, Var
# 🧠 ML Signal: Importing specific classes from a library indicates usage patterns for those classes
from qlib.tests import TestOperatorData
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage

# 🧠 ML Signal: Use of a method named 'features' suggests a pattern for feature extraction in ML models
# 🧠 ML Signal: Importing specific modules from a library indicates usage patterns for those modules

class TestOperatorDataSetting(TestOperatorData):
    # 🧠 ML Signal: Use of a method named 'features' suggests a pattern for feature extraction in ML models
    def test_setting(self):
        # 🧠 ML Signal: Use of a method named 'features' suggests a pattern for feature extraction in ML models
        # All the query below passes
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close)"])

        # 🧠 ML Signal: Use of a method named 'features' suggests a pattern for feature extraction in ML models
        # get market return for "SH600519"
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"])
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"])
        # excess return
        # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
        df = D.features(
            ["SH600519"], ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"]
        # 🧠 ML Signal: Logging and printing are common patterns for debugging and monitoring
        )
        print(df)
    # 🧠 ML Signal: Function return values are often used for further processing or validation

    def test_case2(self):
        # 🧠 ML Signal: Function calls with specific parameters can indicate usage patterns
        def test_case(instruments, queries, note=None):
            if note:
                print(note)
            print(f"checking {instruments} with queries {queries}")
            df = D.features(instruments, queries)
            print(df)
            return df

        test_case(["SH600519"], ["ChangeInstrument('SH000300', $close)"], "get market index close")
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"],
            "get market index return with Feature",
        )
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get market index return with expression",
        # 🧠 ML Signal: String formatting and variable usage can indicate dynamic query generation
        )
        test_case(
            ["SH600519"],
            ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get excess return with expression with beta=1",
        )

        ret = "Feature('close') / Ref(Feature('close'), 1) - 1"
        benchmark = "SH000300"
        n_period = 252
        marketRet = f"ChangeInstrument('{benchmark}', Feature('close') / Ref(Feature('close'), 1) - 1)"
        marketVar = f"ChangeInstrument('{benchmark}', Var({marketRet}, {n_period}))"
        beta = f"Cov({ret}, {marketRet}, {n_period}) / {marketVar}"
        excess_return = f"{ret} - {beta}*({marketRet})"
        fields = [
            "Feature('close')",
            # 🧠 ML Signal: Function calls with specific parameters can indicate usage patterns
            f"ChangeInstrument('{benchmark}', Feature('close'))",
            ret,
            marketRet,
            # ⚠️ SAST Risk (Low): Potential for division by zero if Ref(Feature("close"), 1) is zero
            beta,
            # ⚠️ SAST Risk (Low): Potential for division by zero if Ref(Feature("close"), 1) is zero
            excess_return,
        ]
        test_case(["SH600519"], fields[5:], "get market beta and excess_return with estimated beta")

        instrument = "sh600519"
        ret = Feature("close") / Ref(Feature("close"), 1) - 1
        benchmark = "sh000300"
        n_period = 252
        # ⚠️ SAST Risk (Low): Potential for division by zero if marketVar is zero
        marketRet = ChangeInstrument(benchmark, Feature("close") / Ref(Feature("close"), 1) - 1)
        marketVar = ChangeInstrument(benchmark, Var(marketRet, n_period))
        beta = Cov(ret, marketRet, n_period) / marketVar
        fields = [
            Feature("close"),
            ChangeInstrument(benchmark, Feature("close")),
            # ✅ Best Practice: Use of dictionaries for configuration improves readability and maintainability
            # 🧠 ML Signal: Object instantiation with specific configurations can indicate usage patterns
            # 🧠 ML Signal: Method calls on objects can indicate common usage patterns
            # 🧠 ML Signal: Logging and printing are common patterns for debugging and monitoring
            # 🧠 ML Signal: Use of unittest framework indicates testing practices
            ret,
            marketRet,
            beta,
            ret - beta * marketRet,
        ]
        names = ["close", "marketClose", "ret", "marketRet", f"beta_{n_period}", "excess_return"]
        data_loader_config = {"feature": (fields, names)}
        data_loader = QlibDataLoader(config=data_loader_config)
        df = data_loader.load(instruments=[instrument])  # , start_time=start_time)
        print(df)

        # test_case(["sh600519"],fields,
        # "get market beta and excess_return with estimated beta")


if __name__ == "__main__":
    unittest.main()