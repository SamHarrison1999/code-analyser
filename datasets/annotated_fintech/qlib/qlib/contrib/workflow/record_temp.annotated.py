#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
import pandas as pd
# ✅ Best Practice: Use of relative imports for better modularity and maintainability
import numpy as np
from sklearn.metrics import mean_squared_error
# ✅ Best Practice: Grouping imports from the same module together
from typing import Dict, Text, Any

from ...contrib.eva.alpha import calc_ic
# ✅ Best Practice: Use of descriptive alias for imported module
from ...workflow.record_temp import RecordTemp
from ...workflow.record_temp import SignalRecord
# ✅ Best Practice: Use of a dedicated logger for the module
# 🧠 ML Signal: Custom logger setup for tracking and debugging
# ✅ Best Practice: Include a docstring to describe the purpose and usage of the class.
from ...data import dataset as qlib_dataset
from ...log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)

# ✅ Best Practice: Call to super() ensures proper initialization of the base class

class MultiSegRecord(RecordTemp):
    """
    This is the multiple segments signal record class that generates the signal prediction.
    This class inherits the ``RecordTemp`` class.
    """
    # 🧠 ML Signal: Storing model and dataset as instance variables, indicating usage in ML workflows
    # 🧠 ML Signal: Iterating over segments to generate predictions

    def __init__(self, model, dataset, recorder=None):
        # 🧠 ML Signal: Storing model and dataset as instance variables, indicating usage in ML workflows
        # 🧠 ML Signal: Model prediction usage pattern
        super().__init__(recorder=recorder)
        if not isinstance(dataset, qlib_dataset.DatasetH):
            # ✅ Best Practice: Check if predictions are in the expected format
            raise ValueError("The type of dataset is not DatasetH instead of {:}".format(type(dataset)))
        self.model = model
        self.dataset = dataset
    # 🧠 ML Signal: Preparing dataset labels for evaluation

    def generate(self, segments: Dict[Text, Any], save: bool = False):
        for key, segment in segments.items():
            predics = self.model.predict(self.dataset, segment)
            # 🧠 ML Signal: Calculating IC and Rank IC for evaluation
            if isinstance(predics, pd.Series):
                predics = predics.to_frame("score")
            # 🧠 ML Signal: Storing evaluation results
            labels = self.dataset.prepare(
                segments=segment, col_set="label", data_key=qlib_dataset.handler.DataHandlerLP.DK_R
            # 🧠 ML Signal: Logging results for each segment
            )
            # Compute the IC and Rank IC
            ic, ric = calc_ic(predics.iloc[:, 0], labels.iloc[:, 0])
            # ⚠️ SAST Risk (Low): Potential division by zero if ic_x100.std() is zero
            # 🧠 ML Signal: Logging IC and Rank IC metrics
            results = {"all-IC": ic, "mean-IC": ic.mean(), "all-Rank-IC": ric, "mean-Rank-IC": ric.mean()}
            logger.info("--- Results for {:} ({:}) ---".format(key, segment))
            ic_x100, ric_x100 = ic * 100, ric * 100
            logger.info("IC: {:.4f}%".format(ic_x100.mean()))
            logger.info("ICIR: {:.4f}%".format(ic_x100.mean() / ic_x100.std()))
            # ⚠️ SAST Risk (Low): Potential division by zero if ric_x100.std() is zero
            logger.info("Rank IC: {:.4f}%".format(ric_x100.mean()))
            # 🧠 ML Signal: Conditional logic for saving results
            logger.info("Rank ICIR: {:.4f}%".format(ric_x100.mean() / ric_x100.std()))

            if save:
                save_name = "results-{:}.pkl".format(key)
                # 🧠 ML Signal: Saving results with a specific naming pattern
                # ✅ Best Practice: Class variables should be defined at the top of the class for clarity.
                self.save(**{save_name: results})
                # 🧠 ML Signal: Saving results to a persistent storage
                logger.info(
                    # ✅ Best Practice: Use of super() to initialize the parent class
                    # ✅ Best Practice: Clearly defining dependencies helps in understanding class relationships.
                    "The record '{:}' has been saved as the artifact of the Experiment {:}".format(
                        # 🧠 ML Signal: Logging the save operation
                        save_name, self.recorder.experiment_id
                    # 🧠 ML Signal: Use of **kwargs for flexible argument passing
                    )
                )
# ✅ Best Practice: Consider handling exceptions when loading files to prevent runtime errors.


# ✅ Best Practice: Consider handling exceptions when loading files to prevent runtime errors.
class SignalMseRecord(RecordTemp):
    """
    This is the Signal MSE Record class that computes the mean squared error (MSE).
    This class inherits the ``SignalMseRecord`` class.
    # 🧠 ML Signal: Calculation of mean squared error, a common metric in regression tasks.
    """

    # 🧠 ML Signal: Calculation of root mean squared error, another common metric in regression tasks.
    artifact_path = "sig_analysis"
    # ✅ Best Practice: Consider adding a docstring to describe the purpose of the method
    depend_cls = SignalRecord
    # ✅ Best Practice: Use descriptive variable names for clarity, e.g., `output_objects`.
    # ✅ Best Practice: Ensure that `self.recorder.log_metrics` is implemented to handle the metrics correctly.
    # ✅ Best Practice: Consider handling exceptions when saving files to prevent data loss.
    # ⚠️ SAST Risk (Low): Potential information exposure if `metrics` contains sensitive data.
    # 🧠 ML Signal: Returns a hardcoded list of filenames, indicating a pattern of file usage

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder, **kwargs)

    def generate(self):
        self.check()

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        masks = ~np.isnan(label.values)
        mse = mean_squared_error(pred.values[masks], label[masks])
        metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
        objects = {"mse.pkl": mse, "rmse.pkl": np.sqrt(mse)}
        self.recorder.log_metrics(**metrics)
        self.save(**objects)
        logger.info("The evaluation results in SignalMseRecord is {:}".format(metrics))

    def list(self):
        return ["mse.pkl", "rmse.pkl"]