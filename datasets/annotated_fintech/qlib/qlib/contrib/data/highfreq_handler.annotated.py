from qlib.data.dataset.handler import DataHandler, DataHandlerLP

# ✅ Best Practice: Grouping imports from the same module together improves readability.

from .handler import check_transform_proc

# ✅ Best Practice: Constants should be named using all uppercase letters with underscores.

EPSILON = 1e-4


class HighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        # ⚠️ SAST Risk (Low): Mutable default arguments like lists can lead to unexpected behavior.
        fit_start_time=None,
        fit_end_time=None,
        # ⚠️ SAST Risk (Low): Mutable default arguments like lists can lead to unexpected behavior.
        drop_raw=True,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                # 🧠 ML Signal: Usage of 'get_feature_config' suggests a pattern for feature configuration in ML.
                # 🧠 ML Signal: Use of 'super().__init__' indicates inheritance, common in ML model or data pipeline setup.
                "config": self.get_feature_config(),
                "swap_level": False,
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
            # ✅ Best Practice: Use of descriptive variable names improves code readability.
            drop_raw=drop_raw,
        )

    # ✅ Best Practice: Use of descriptive variable names improves code readability.

    # ✅ Best Practice: Use of default parameter values for flexibility
    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        # ⚠️ SAST Risk (Low): Potential for format string injection if inputs are not sanitized
        template_paused = "Select(Gt($paused_num, 1.001), {0})"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            else:
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"
            # 🧠 ML Signal: Repeated pattern of adding normalized price features

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(
                        template_if.format("$close", price_field),
                        template_fillnan.format("$close"),
                    )
                    # 🧠 ML Signal: Consistent naming pattern for features
                )
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature("$vwap", 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        # ⚠️ SAST Risk (Low): Potential for format string injection if inputs are not sanitized
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature("$vwap", 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        # calculate and fill nan with 0
        template_gzero = "If(Ge({0}, 0), {0}, 0)"
        fields += [
            template_gzero.format(
                # ⚠️ SAST Risk (Low): Potential for format string injection if inputs are not sanitized
                template_paused.format(
                    "If(IsNull({0}), 0, {0})".format(
                        "{0}/Ref(DayLast(Mean({0}, 7200)), 240)".format("$volume")
                    )
                )
            )
        ]
        names += ["$volume"]

        fields += [
            template_gzero.format(
                template_paused.format(
                    "If(IsNull({0}), 0, {0})".format(
                        "Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240)".format(
                            "$volume"
                        )
                    )
                )
            )
        ]
        names += ["$volume_1"]

        # ✅ Best Practice: Consider using immutable default arguments like None instead of mutable ones like lists
        return fields, names


# ⚠️ SAST Risk (Low): Potential issue with mutable default arguments (list)
class HighFreqGeneralHandler(DataHandlerLP):
    # ⚠️ SAST Risk (Low): Potential issue with mutable default arguments (list)
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        # 🧠 ML Signal: Usage of a method to get feature configuration
        # 🧠 ML Signal: Use of a superclass constructor with specific parameters
        drop_raw=True,
        day_length=240,
        freq="1min",
        columns=["$open", "$high", "$low", "$close", "$vwap"],
        inst_processors=None,
    ):
        self.day_length = day_length
        self.columns = columns

        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        # ✅ Best Practice: Use of descriptive variable names improves code readability.
        data_loader = {
            "class": "QlibDataLoader",
            # ✅ Best Practice: Use of f-string for string formatting is more readable and efficient.
            "kwargs": {
                # ✅ Best Practice: Use of f-strings for string formatting improves readability and performance.
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": freq,
                "inst_processors": inst_processors,
            },
            # ⚠️ SAST Risk (Low): Potential for format string injection if template_paused or template_if are user-controlled.
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            # 🧠 ML Signal: Iterating over self.columns suggests dynamic feature generation based on data columns.
            drop_raw=drop_raw,
        )

    # 🧠 ML Signal: Appending features to a list indicates feature engineering for ML models.

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"

        def get_normalized_price_feature(price_field, shift=0):
            # ⚠️ SAST Risk (Low): Use of string formatting with potential for injection if template_paused is user-controlled.
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = f"{{0}}/DayLast(Ref({{1}}, {self.day_length * 2}))"
            else:
                template_norm = (
                    "Ref({0}, "
                    + str(shift)
                    + f")/DayLast(Ref({{1}}, {self.day_length}))"
                )

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    # ⚠️ SAST Risk (Low): Use of string formatting with potential for injection if template_paused is user-controlled.
                    template_norm.format(
                        template_if.format("$close", price_field),
                        template_fillnan.format("$close"),
                    )
                )
                # ✅ Best Practice: Class should inherit from a base class to ensure consistent interface and behavior
            )
            return feature_ops

        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, 0))
            names.append(column_name)

        # 🧠 ML Signal: Default parameter values can indicate common usage patterns.
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use.
        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, self.day_length))
            names.append(column_name + "_1")

        # calculate and fill nan with 0
        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format(
                    # 🧠 ML Signal: Frequency settings can indicate common data processing intervals.
                    # 🧠 ML Signal: Configuration settings can indicate common feature extraction patterns.
                    f"{{0}}/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format(
                        "$volume"
                    )
                )
            )
        ]
        names += ["$volume"]

        # ✅ Best Practice: Use of super() to ensure proper inheritance and initialization.
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
        fields += [
            template_paused.format(
                # ✅ Best Practice: Initialize lists before using them to store values.
                "If(IsNull({0}), 0, {0})".format(
                    f"Ref({{0}}, {self.day_length})/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format(
                        "$volume"
                        # ✅ Best Practice: Use descriptive variable names for better readability.
                    )
                )
            )
        ]
        # 🧠 ML Signal: Usage of template strings for feature configuration.
        names += ["$volume_1"]
        # 🧠 ML Signal: Tracking feature names for later reference.

        return fields, names


class HighFreqBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
    ):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                # 🧠 ML Signal: Class definition for a backtest handler, useful for identifying patterns in financial data processing
                "config": self.get_feature_config(),
                # ✅ Best Practice: Return statements should be clear and consistent.
                "swap_level": False,
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
        )

    # ✅ Best Practice: Use of self to define instance variables for encapsulation and clarity.

    def get_feature_config(self):
        # ✅ Best Practice: Converting list to set for columns to ensure uniqueness and faster lookup.
        # 🧠 ML Signal: Use of a dictionary to configure a data loader, indicating a pattern for dynamic configuration.
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Gt($paused_num, 1.001), {0})"
        template_fillnan = "FFillNan({0})"
        fields += [
            template_fillnan.format(template_paused.format("$close")),
        ]
        # 🧠 ML Signal: Dynamic feature configuration retrieval, useful for model training.
        names += ["$close0"]

        fields += [
            template_paused.format(
                template_if.format(
                    template_fillnan.format("$close"),
                    # ✅ Best Practice: Use of super() to ensure proper inheritance and initialization of the parent class.
                    "$vwap",
                )
            )
            # 🧠 ML Signal: Checking for specific column names in self.columns indicates feature selection logic.
        ]
        names += ["$vwap0"]
        # ✅ Best Practice: Use of f-string for string formatting improves readability.

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
        names += ["$volume0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$factor"))]
        # 🧠 ML Signal: Appending to fields list suggests dynamic feature configuration.
        names += ["$factor0"]

        # 🧠 ML Signal: Appending to names list suggests dynamic feature naming.
        return fields, names


# 🧠 ML Signal: Checking for specific column names in self.columns indicates feature selection logic.
class HighFreqGeneralBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        # 🧠 ML Signal: Checking for specific column names in self.columns indicates feature selection logic.
        # ✅ Best Practice: Returning a tuple of fields and names improves function clarity.
        day_length=240,
        freq="1min",
        columns=["$close", "$vwap", "$volume"],
        inst_processors=None,
    ):
        self.day_length = day_length
        self.columns = set(columns)
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                # ⚠️ SAST Risk (Low): Mutable default arguments (lists) can lead to unexpected behavior
                "freq": freq,
                "inst_processors": inst_processors,
                # ⚠️ SAST Risk (Low): Mutable default arguments (lists) can lead to unexpected behavior
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
        )

    # 🧠 ML Signal: Use of data loader configuration for machine learning data processing
    def get_feature_config(self):
        fields = []
        names = []

        if "$close" in self.columns:
            template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"
            template_fillnan = "FFillNan({0})"
            template_if = "If(IsNull({1}), {0}, {1})"
            fields += [
                template_paused.format(template_fillnan.format("$close")),
            ]
            names += ["$close0"]
        # ✅ Best Practice: Use descriptive variable names for better readability and maintainability.

        if "$vwap" in self.columns:
            # ✅ Best Practice: Use descriptive variable names for better readability and maintainability.
            fields += [
                # ✅ Best Practice: Function name is descriptive and indicates its purpose
                template_paused.format(
                    template_if.format(template_fillnan.format("$close"), "$vwap")
                ),
                # ✅ Best Practice: Use descriptive variable names for better readability and maintainability.
            ]
            # ✅ Best Practice: Default parameter value for 'shift' is provided
            names += ["$vwap0"]

        # ✅ Best Practice: Readable string formatting for template
        if "$volume" in self.columns:
            fields += [
                template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))
            ]
            # ✅ Best Practice: Readable string formatting for template
            names += ["$volume0"]

        return fields, names


# ✅ Best Practice: Readable string formatting for template
# ⚠️ SAST Risk (Medium): Potential for format string injection if 'template_paused' or 'template_if' are user-controlled
class HighFreqOrderHandler(DataHandlerLP):
    def __init__(
        # ⚠️ SAST Risk (Medium): Potential for format string injection if 'template_fillnan' is user-controlled
        self,
        # ⚠️ SAST Risk (Medium): Potential for format string injection if 'template_if' is user-controlled
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        # ✅ Best Practice: Returns a value, making the function reusable
        # ⚠️ SAST Risk (Low): Potential use of undefined variable 'template_paused'
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        inst_processors=None,
        drop_raw=True,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            # 🧠 ML Signal: Usage of financial indicators for feature generation
            "kwargs": {
                "config": self.get_feature_config(),
                # 🧠 ML Signal: Usage of financial indicators for feature generation
                "swap_level": False,
                "freq": "1min",
                # 🧠 ML Signal: Usage of financial indicators for feature generation
                "inst_processors": inst_processors,
            },
            # 🧠 ML Signal: Usage of financial indicators for feature generation
        }
        super().__init__(
            # 🧠 ML Signal: Usage of financial indicators for feature generation
            instruments=instruments,
            start_time=start_time,
            # 🧠 ML Signal: Tracking feature names for financial data
            end_time=end_time,
            data_loader=data_loader,
            # 🧠 ML Signal: Usage of financial indicators for feature generation
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            # 🧠 ML Signal: Usage of financial indicators for feature generation
            drop_raw=drop_raw,
        )

    # 🧠 ML Signal: Usage of financial indicators for feature generation

    def get_feature_config(self):
        # 🧠 ML Signal: Usage of financial indicators for feature generation
        fields = []
        # 🧠 ML Signal: Usage of financial indicators for feature generation
        # 🧠 ML Signal: Tracking feature names for financial data
        # ⚠️ SAST Risk (Low): Potential for format string injection if template_paused is user-controlled
        # ✅ Best Practice: Ensure template_paused is sanitized or controlled
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_ifinf = "If(IsInf({1}), {0}, {1})"
        template_paused = "Select(Gt($paused_num, 1.001), {0})"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            # 🧠 ML Signal: Usage of financial indicators for feature generation
            # 🧠 ML Signal: Tracking feature names for financial data
            else:
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(
                        template_if.format("$close", price_field),
                        template_fillnan.format("$close"),
                    )
                )
                # 🧠 ML Signal: Tracking feature names for financial data
                # ⚠️ SAST Risk (Low): Potential for format string injection if template_paused is user-controlled
                # ✅ Best Practice: Ensure template_paused is sanitized or controlled
            )
            return feature_ops

        def get_normalized_vwap_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = "{0}/DayLast(Ref({1}, 243))"
            else:
                # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
                template_norm = "Ref({0}, " + str(shift) + ")/DayLast(Ref({1}, 243))"

            # 🧠 ML Signal: Naming conventions for features could be used to infer feature types
            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
            feature_ops = template_paused.format(
                template_fillnan.format(
                    # 🧠 ML Signal: Naming conventions for features could be used to infer feature types
                    template_norm.format(
                        template_if.format(
                            "$close", template_ifinf.format("$close", price_field)
                        ),
                        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
                        template_fillnan.format("$close"),
                    )
                    # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
                )
            )
            # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
            return feature_ops

        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which this does.
        fields += [get_normalized_price_feature("$low", 0)]
        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_vwap_price_feature("$vwap", 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        # ✅ Best Practice: Use of a dictionary to store configuration settings improves code readability and maintainability.
        # 🧠 ML Signal: Use of a method to get configuration suggests dynamic or customizable behavior.
        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        # 🧠 ML Signal: Naming conventions for features could be used to infer feature types
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_vwap_price_feature("$vwap", 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [get_normalized_price_feature("$bid", 0)]
        fields += [get_normalized_price_feature("$ask", 0)]
        names += ["$bid", "$ask"]
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns

        fields += [get_normalized_price_feature("$bid", 240)]
        fields += [get_normalized_price_feature("$ask", 240)]
        names += ["$bid_1", "$ask_1"]

        # calculate and fill nan with 0
        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function

        # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
        def get_volume_feature(volume_field, shift=0):
            # ✅ Best Practice: Initialize lists before using them
            template_gzero = "If(Ge({0}, 0), {0}, 0)"
            # 🧠 ML Signal: Usage of specific volume fields and shifts could indicate feature engineering patterns
            if shift == 0:
                feature_ops = template_gzero.format(
                    # 🧠 ML Signal: Naming conventions for features could be used to infer feature types
                    # ✅ Best Practice: Use descriptive variable names for better readability
                    template_paused.format(
                        "If(IsInf({0}), 0, {0})".format(
                            "If(IsNull({0}), 0, {0})".format(
                                "{0}/Ref(DayLast(Mean({0}, 7200)), 240)".format(
                                    volume_field
                                )
                                # 🧠 ML Signal: Usage of list operations to accumulate feature configurations
                            )
                            # 🧠 ML Signal: Usage of list operations to accumulate feature names
                        )
                    )
                )
            else:
                feature_ops = template_gzero.format(
                    template_paused.format(
                        "If(IsInf({0}), 0, {0})".format(
                            "If(IsNull({0}), 0, {0})".format(
                                f"Ref({{0}}, {shift})/Ref(DayLast(Mean({{0}}, 7200)), 240)".format(
                                    volume_field
                                )
                            )
                        )
                    )
                )
            return feature_ops

        fields += [get_volume_feature("$volume", 0)]
        names += ["$volume"]

        fields += [get_volume_feature("$volume", 240)]
        names += ["$volume_1"]

        fields += [get_volume_feature("$bidV", 0)]
        fields += [get_volume_feature("$bidV1", 0)]
        fields += [get_volume_feature("$bidV3", 0)]
        fields += [get_volume_feature("$bidV5", 0)]
        fields += [get_volume_feature("$askV", 0)]
        fields += [get_volume_feature("$askV1", 0)]
        fields += [get_volume_feature("$askV3", 0)]
        fields += [get_volume_feature("$askV5", 0)]
        names += [
            "$bidV",
            "$bidV1",
            "$bidV3",
            "$bidV5",
            "$askV",
            "$askV1",
            "$askV3",
            "$askV5",
        ]

        # ✅ Best Practice: Return statement should be at the end of the function
        fields += [get_volume_feature("$bidV", 240)]
        fields += [get_volume_feature("$bidV1", 240)]
        fields += [get_volume_feature("$bidV3", 240)]
        fields += [get_volume_feature("$bidV5", 240)]
        fields += [get_volume_feature("$askV", 240)]
        fields += [get_volume_feature("$askV1", 240)]
        fields += [get_volume_feature("$askV3", 240)]
        fields += [get_volume_feature("$askV5", 240)]
        names += [
            "$bidV_1",
            "$bidV1_1",
            "$bidV3_1",
            "$bidV5_1",
            "$askV_1",
            "$askV1_1",
            "$askV3_1",
            "$askV5_1",
        ]

        return fields, names


class HighFreqBacktestOrderHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
    ):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Gt($paused_num, 1.001), {0})"
        template_fillnan = "FFillNan({0})"
        fields += [
            template_fillnan.format(template_paused.format("$close")),
        ]
        names += ["$close0"]

        fields += [
            template_paused.format(
                template_if.format(
                    template_fillnan.format("$close"),
                    "$vwap",
                )
            )
        ]
        names += ["$vwap0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
        names += ["$volume0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$bid"))]
        names += ["$bid0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$bidV"))]
        names += ["$bidV0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$ask"))]
        names += ["$ask0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$askV"))]
        names += ["$askV0"]

        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format("($bid + $ask) / 2")
            )
        ]
        names += ["$median0"]

        fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$factor"))]
        names += ["$factor0"]

        fields += [
            template_paused.format("If(IsNull({0}), 0, {0})".format("$downlimitmarket"))
        ]
        names += ["$downlimitmarket0"]

        fields += [
            template_paused.format("If(IsNull({0}), 0, {0})".format("$uplimitmarket"))
        ]
        names += ["$uplimitmarket0"]

        fields += [
            template_paused.format("If(IsNull({0}), 0, {0})".format("$highmarket"))
        ]
        names += ["$highmarket0"]

        fields += [
            template_paused.format("If(IsNull({0}), 0, {0})".format("$lowmarket"))
        ]
        names += ["$lowmarket0"]

        return fields, names
