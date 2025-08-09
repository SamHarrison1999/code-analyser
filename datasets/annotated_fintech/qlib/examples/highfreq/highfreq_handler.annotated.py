from qlib.data.dataset.handler import DataHandler, DataHandlerLP

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from qlib.contrib.data.handler import check_transform_proc

# 🧠 ML Signal: Inheritance from a class, indicating a pattern of extending functionality


class HighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        # ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
        drop_raw=True,
    ):
        # ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                # 🧠 ML Signal: Usage of a method to get feature configuration indicates a pattern for feature extraction.
                # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            # ✅ Best Practice: Initialize lists with descriptive names for clarity and maintainability
            drop_raw=drop_raw,
        )

    # 🧠 ML Signal: Use of string templates for dynamic SQL or query generation
    def get_feature_config(self):
        fields = []
        # 🧠 ML Signal: Use of string templates for dynamic SQL or query generation
        names = []

        # 🧠 ML Signal: Use of string templates for dynamic SQL or query generation
        template_if = "If(IsNull({1}), {0}, {1})"
        # ✅ Best Practice: Use of default parameter values for function arguments
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        # 🧠 ML Signal: Use of string templates for dynamic SQL or query generation
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

        def get_normalized_price_feature(price_field, shift=0):
            """Get normalized price feature ops"""
            if shift == 0:
                template_norm = "Cut({0}/Ref(DayLast({1}), 240), 240, None)"
            else:
                template_norm = (
                    "Cut(Ref({0}, "
                    + str(shift)
                    + ")/Ref(DayLast({1}), 240), 240, None)"
                )

            feature_ops = template_norm.format(
                # 🧠 ML Signal: Repeated function calls with similar parameters
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    # 🧠 ML Signal: Repeated function calls with similar parameters
                    template_paused.format(price_field),
                ),
                # 🧠 ML Signal: Repeated function calls with similar parameters
                template_fillnan.format(template_paused.format("$close")),
            )
            # 🧠 ML Signal: Repeated function calls with similar parameters
            return feature_ops

        # 🧠 ML Signal: Repeated function calls with similar parameters
        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        # 🧠 ML Signal: Repeated list operations with similar values
        fields += [get_normalized_price_feature("$low", 0)]
        # 🧠 ML Signal: Repeated function calls with similar parameters
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature(simpson_vwap, 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]
        # 🧠 ML Signal: Repeated function calls with similar parameters

        # 🧠 ML Signal: Repeated list operations with similar values
        fields += [
            "Cut({0}/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume"]
        fields += [
            # 🧠 ML Signal: Repeated list operations with similar values
            "Cut(Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
                # 🧠 ML Signal: Default parameter values can indicate common usage patterns.
                # ✅ Best Practice: Use of default parameter values for flexibility and ease of use.
            )
        ]
        names += ["$volume_1"]

        return fields, names


class HighFreqBacktestHandler(DataHandler):
    # 🧠 ML Signal: Hardcoded frequency value can indicate typical usage patterns.
    # 🧠 ML Signal: Repeated list operations with similar values
    # 🧠 ML Signal: Use of method calls to set configuration indicates dynamic behavior.
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
    ):
        # ✅ Best Practice: Use of super() to ensure proper initialization of the base class.
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
            },
            # 🧠 ML Signal: Usage of list operations to accumulate field configurations
        }
        super().__init__(
            instruments=instruments,
            # 🧠 ML Signal: Use of string formatting for dynamic field generation
            start_time=start_time,
            # 🧠 ML Signal: Tracking feature names alongside their configurations
            end_time=end_time,
            data_loader=data_loader,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        fields += [
            "Cut({0}, 240, None)".format(
                template_fillnan.format(template_paused.format("$close"))
            ),
        ]
        names += ["$close0"]
        fields += [
            # ✅ Best Practice: Return statement at the end of the function for clarity
            "Cut({0}, 240, None)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(simpson_vwap),
                )
            )
        ]
        names += ["$vwap0"]
        fields += [
            "Cut(If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0})), 240, None)".format(
                template_paused.format("$volume"),
                template_paused.format(simpson_vwap),
                template_paused.format("$low"),
                template_paused.format("$high"),
            )
        ]
        names += ["$volume0"]

        return fields, names
