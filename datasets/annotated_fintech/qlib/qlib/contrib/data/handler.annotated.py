# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Grouping imports from the same module together improves readability.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
# ✅ Best Practice: Importing specific functions or classes is preferred over importing the entire module.
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
# ✅ Best Practice: Using 'as' to alias imports can help avoid naming conflicts and improve clarity.
from ...data.dataset import processor as processor_module
from inspect import getfullargspec
# ✅ Best Practice: Importing specific functions or classes is preferred over importing the entire module.
# 🧠 ML Signal: Iterating over a list of processors, which may indicate a pattern of applying transformations or preprocessing steps.


# ⚠️ SAST Risk (Low): Dynamic class instantiation can lead to security risks if not properly controlled.
def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    # 🧠 ML Signal: Using reflection to get function arguments, which can indicate dynamic behavior in ML pipelines.
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            # ⚠️ SAST Risk (Low): Assertion statements can be disabled in production, potentially leading to unexpected behavior.
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    # 🧠 ML Signal: Inheritance from a class, indicating use of object-oriented programming
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        # ✅ Best Practice: Validate or sanitize inputs to prevent unexpected behavior or errors.
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        # ✅ Best Practice: Validate or sanitize inputs to prevent unexpected behavior or errors.
        # 🧠 ML Signal: Usage of a dictionary to configure a data loader, indicating a pattern for dynamic configuration.
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            # 🧠 ML Signal: Dynamic feature configuration, useful for model training or inference.
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        # ✅ Best Practice: Use of super() to ensure proper initialization of the base class.
        # 🧠 ML Signal: Function returning a configuration for labels, likely used in ML model training or evaluation
        }

        # 🧠 ML Signal: Returning a list of expressions and labels, indicating a pattern for feature-label mapping
        super().__init__(
            instruments=instruments,
            # 🧠 ML Signal: Function returning a configuration, possibly for ML model labeling
            start_time=start_time,
            end_time=end_time,
            # ✅ Best Practice: Use descriptive variable names for readability
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


# ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
class Alpha158(DataHandlerLP):
    def __init__(
        # 🧠 ML Signal: Usage of dynamic configuration for labels can indicate model training or evaluation.
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            # 🧠 ML Signal: Usage of a configuration dictionary for feature settings
            start_time=start_time,
            # 🧠 ML Signal: Function returning a configuration, possibly for labeling data in ML tasks
            end_time=end_time,
            data_loader=data_loader,
            # 🧠 ML Signal: Returning a list of expressions and labels, indicating a pattern for data labeling
            # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
            infer_processors=infer_processors,
            # ✅ Best Practice: Method should have a docstring explaining its purpose and return values
            learn_processors=learn_processors,
            # 🧠 ML Signal: Returns a configuration that could be used for labeling data in ML models
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]