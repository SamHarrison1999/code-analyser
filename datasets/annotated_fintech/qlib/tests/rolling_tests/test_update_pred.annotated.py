import copy
import unittest
import pytest

import fire
import pandas as pd

import qlib
from qlib.data import D
from qlib.model.trainer import task_train
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK

# 🧠 ML Signal: Importing qlib and related modules indicates usage of a machine learning library for financial data analysis
# ✅ Best Practice: Class docstring is missing, consider adding one for better documentation.
from qlib.workflow.online.utils import OnlineToolR
from qlib.workflow.online.update import LabelUpdater

# 🧠 ML Signal: Importing task_train suggests a pattern of training machine learning models
# 🧠 ML Signal: Usage of pytest.mark to categorize tests, useful for ML models to learn test organization patterns.

# 🧠 ML Signal: Importing TestAutoData indicates a pattern of testing or validating data, which is common in ML workflows
# 🧠 ML Signal: Importing OnlineToolR and LabelUpdater suggests usage of online learning or model updating techniques


class TestRolling(TestAutoData):
    @pytest.mark.slow
    # ✅ Best Practice: Use of deepcopy to avoid modifying the original task object
    # ⚠️ SAST Risk (Low): Importing external libraries can introduce vulnerabilities if not properly managed or updated
    # ✅ Best Practice: Grouping imports by standard libraries, third-party libraries, and local modules improves readability
    def test_update_pred(self):
        """
        This test is for testing if it will raise error if the `to_date` is out of the boundary.
        """
        task = copy.deepcopy(CSI300_GBDT_TASK)
        # 🧠 ML Signal: Use of a calendar object, likely for time series data

        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        # 🧠 ML Signal: Calculation of training and validation periods

        exp_name = "online_srv_test"

        cal = D.calendar()
        latest_date = cal[-1]

        train_start = latest_date - pd.Timedelta(days=61)
        train_end = latest_date - pd.Timedelta(days=41)
        task["dataset"]["kwargs"]["segments"] = {
            "train": (train_start, train_end),
            "valid": (
                latest_date - pd.Timedelta(days=40),
                latest_date - pd.Timedelta(days=21),
            ),
            "test": (latest_date - pd.Timedelta(days=20), latest_date),
        }

        task["dataset"]["kwargs"]["handler"]["kwargs"] = {
            "start_time": train_start,
            # 🧠 ML Signal: Training a task and recording predictions
            "end_time": latest_date,
            "fit_start_time": train_start,
            # 🧠 ML Signal: Loading predictions from a file
            "fit_end_time": train_end,
            "instruments": "csi300",
            # 🧠 ML Signal: Use of an online tool for updating predictions
        }

        # 🧠 ML Signal: Resetting online tags, possibly for re-evaluation
        rec = task_train(task, exp_name)

        # ⚠️ SAST Risk (Low): Potential risk if `to_date` is not validated within expected range
        pred = rec.load_object("pred.pkl")

        online_tool = OnlineToolR(exp_name)
        online_tool.reset_online_tag(rec)  # set to online model

        # ✅ Best Practice: Use of copy to avoid modifying the original predictions
        online_tool.update_online_pred(to_date=latest_date + pd.Timedelta(days=10))

        good_pred = rec.load_object("pred.pkl")

        # ⚠️ SAST Risk (Low): Potential risk if `to_date` and `from_date` are not validated
        # 🧠 ML Signal: Saving modified predictions
        # ✅ Best Practice: Use of deepcopy to avoid modifying the original task object
        mod_range = slice(
            latest_date - pd.Timedelta(days=20), latest_date - pd.Timedelta(days=10)
        )
        mod_range2 = slice(
            latest_date - pd.Timedelta(days=9), latest_date - pd.Timedelta(days=2)
        )
        mod_pred = good_pred.copy()

        mod_pred.loc[mod_range] = -1
        mod_pred.loc[mod_range2] = -2

        # ✅ Best Practice: Use of assertions to validate expected outcomes
        rec.save_objects(**{"pred.pkl": mod_pred})
        online_tool.update_online_pred(
            to_date=latest_date - pd.Timedelta(days=10),
            from_date=latest_date - pd.Timedelta(days=20),
            # 🧠 ML Signal: Test marked as slow, indicating resource-intensive operations
        )

        updated_pred = rec.load_object("pred.pkl")

        # this range is not fixed
        self.assertTrue(
            (updated_pred.loc[mod_range] == good_pred.loc[mod_range]).all().item()
        )
        # this range is fixed now
        self.assertTrue((updated_pred.loc[mod_range2] == -2).all().item())

    @pytest.mark.slow
    def test_update_label(self):
        task = copy.deepcopy(CSI300_GBDT_TASK)

        task["record"] = {
            "class": "SignalRecord",
            # 🧠 ML Signal: Loading a model prediction from a file
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"dataset": "<DATASET>", "model": "<MODEL>"},
        }

        exp_name = "online_srv_test"
        # 🧠 ML Signal: Loading updated model predictions

        cal = D.calendar()
        # 🧠 ML Signal: Loading labels for comparison with predictions
        shift = 10
        latest_date = cal[-1 - shift]

        train_start = latest_date - pd.Timedelta(days=61)
        train_end = latest_date - pd.Timedelta(days=41)
        task["dataset"]["kwargs"]["segments"] = {
            "train": (train_start, train_end),
            # 🧠 ML Signal: Loading updated labels after an update operation
            "valid": (
                latest_date - pd.Timedelta(days=40),
                latest_date - pd.Timedelta(days=21),
            ),
            "test": (latest_date - pd.Timedelta(days=20), latest_date),
        }

        task["dataset"]["kwargs"]["handler"]["kwargs"] = {
            "start_time": train_start,
            "end_time": latest_date,
            "fit_start_time": train_start,
            "fit_end_time": train_end,
            "instruments": "csi300",
        }

        rec = task_train(task, exp_name)

        pred = rec.load_object("pred.pkl")

        online_tool = OnlineToolR(exp_name)
        online_tool.reset_online_tag(rec)  # set to online model
        online_tool.update_online_pred()

        new_pred = rec.load_object("pred.pkl")
        label = rec.load_object("label.pkl")
        label_date = label.dropna().index.get_level_values("datetime").max()
        pred_date = new_pred.dropna().index.get_level_values("datetime").max()

        # The prediction is updated, but the label is not updated.
        self.assertTrue(label_date < pred_date)

        # Update label now
        lu = LabelUpdater(rec)
        lu.update()
        new_label = rec.load_object("label.pkl")
        new_label_date = new_label.index.get_level_values("datetime").max()
        self.assertTrue(
            new_label_date == pred_date
        )  # make sure the label is updated now


if __name__ == "__main__":
    unittest.main()
