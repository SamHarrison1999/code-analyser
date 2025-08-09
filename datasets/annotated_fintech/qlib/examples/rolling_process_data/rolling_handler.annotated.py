from qlib.data.dataset.handler import DataHandlerLP
# 🧠 ML Signal: Importing specific classes from a library indicates usage patterns and dependencies
from qlib.data.dataset.loader import DataLoaderDH
# ✅ Best Practice: Grouping imports from the same library together improves readability
from qlib.contrib.data.handler import check_transform_proc
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage

# ✅ Best Practice: Grouping imports from the same library together improves readability
# 🧠 ML Signal: Importing specific classes from a library indicates usage patterns and dependencies

class RollingDataHandler(DataHandlerLP):
    def __init__(
        self,
        start_time=None,
        end_time=None,
        # ✅ Best Practice: Grouping imports from the same library together improves readability
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        # ⚠️ SAST Risk (Low): Using mutable default arguments (lists and dicts) can lead to unexpected behavior.
        data_loader_kwargs={},
    ):
        # 🧠 ML Signal: Usage of a dictionary to configure a data loader, indicating a pattern for dynamic configuration.
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
            "class": "DataLoaderDH",
            "kwargs": {**data_loader_kwargs},
        }

        super().__init__(
            instruments=None,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )