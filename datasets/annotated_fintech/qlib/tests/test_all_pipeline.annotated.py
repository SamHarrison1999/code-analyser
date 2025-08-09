# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import shutil
import unittest
import pytest
from pathlib import Path

import qlib
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from qlib.config import C
from qlib.utils import init_instance_by_config, flatten_dict
# ✅ Best Practice: Consider adding type hints for the return values for better readability and maintainability.
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.tests import TestAutoData
from qlib.tests.config import CSI300_GBDT_TASK, CSI300_BENCH


def train(uri_path: str = None):
    """train model

    Returns
    -------
        pred_score: pandas.DataFrame
            predict scores
        performance: dict
            model performance
    # ⚠️ SAST Risk (Low): Use of undefined variable 'R' can lead to runtime errors.
    """

    # ⚠️ SAST Risk (Low): Potential misuse of 'R.start' if 'R' is not properly defined or imported.
    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    # 🧠 ML Signal: Logging parameters is a common practice in ML experiments.
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    # To test __repr__
    # 🧠 ML Signal: Model fitting is a key step in ML training processes.
    print(dataset)
    print(R)
    # 🧠 ML Signal: Saving trained models is a common pattern in ML workflows.

    # start exp
    # 🧠 ML Signal: Retrieving a recorder is indicative of experiment tracking in ML.
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        # prediction
        # 🧠 ML Signal: SignalRecord usage suggests a pattern for handling ML model signals.
        recorder = R.get_recorder()
        # 🧠 ML Signal: Loading prediction scores is a common pattern in ML workflows.
        # To test __repr__
        print(recorder)
        # To test get_local_dir
        print(recorder.get_local_dir())
        rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        # 🧠 ML Signal: SigAnaRecord usage suggests a pattern for analyzing ML model signals.
        # 🧠 ML Signal: Loading 'ic' and 'ric' indicates a pattern for evaluating ML model performance.
        # 🧠 ML Signal: Usage of a function to get a URI, indicating a pattern of resource or experiment tracking
        pred_score = sr.load("pred.pkl")

        # 🧠 ML Signal: Hardcoded URI for a temporary experiment, indicating a pattern of local testing or development
        # calculate ic and ric
        # 🧠 ML Signal: Retrieving URI path is indicative of experiment tracking in ML.
        sar = SigAnaRecord(recorder)
        # 🧠 ML Signal: Starting an experiment with a specific name and URI, indicating a pattern of experiment management
        sar.generate()
        # ✅ Best Practice: Consider returning a named tuple or a dataclass for better readability and maintainability.
        ic = sar.load("ic.pkl")
        # 🧠 ML Signal: Logging parameters, indicating a pattern of tracking experiment configurations
        ric = sar.load("ric.pkl")

        # 🧠 ML Signal: Checking the current URI after starting an experiment, indicating a pattern of validation or verification
        # 🧠 ML Signal: Returning boolean checks and a URI, indicating a pattern of result validation and resource tracking
        uri_path = R.get_uri()
    return pred_score, {"ic": ic, "ric": ric}, rid, uri_path


def fake_experiment():
    """A fake experiment workflow to test uri

    Returns
    -------
        pass_or_not_for_default_uri: bool
        pass_or_not_for_current_uri: bool
        temporary_exp_dir: str
    # ✅ Best Practice: Use of context manager for resource management
    """

    # 🧠 ML Signal: Loading a recorder by ID, indicating a pattern of experiment tracking
    # start exp
    default_uri = R.get_uri()
    # 🧠 ML Signal: Initialization of dataset instance, indicating a pattern of data handling
    # 🧠 ML Signal: Loading a trained model, indicating a pattern of model usage
    current_uri = "file:./temp-test-exp-mag"
    with R.start(experiment_name="fake_workflow_for_expm", uri=current_uri):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        current_uri_to_check = R.get_uri()
    default_uri_to_check = R.get_uri()
    return default_uri == default_uri_to_check, current_uri == current_uri_to_check, current_uri


def backtest_analysis(pred, rid, uri_path: str = None):
    """backtest and analysis

    Parameters
    ----------
    rid : str
        the id of the recorder to be used in this function
    uri_path: str
        mlflow uri path

    Returns
    -------
    analysis : pandas.DataFrame
        the analysis result

    """
    with R.uri_context(uri=uri_path):
        recorder = R.get_recorder(experiment_name="workflow", recorder_id=rid)

    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model = recorder.load_object("trained_model")

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                # 🧠 ML Signal: Use of PortAnaRecord for analysis, indicating a pattern of financial analysis
                "generate_portfolio_metrics": True,
            },
        # 🧠 ML Signal: Generating analysis, indicating a pattern of result computation
        },
        "strategy": {
            # ⚠️ SAST Risk (Low): Loading data from a file without validation
            "class": "TopkDropoutStrategy",
            # ✅ Best Practice: Printing the analysis dataframe for debugging or logging
            # ✅ Best Practice: Use of Path and resolve() ensures the path is absolute and platform-independent
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                # ⚠️ SAST Risk (Medium): Potential directory traversal if URI_PATH is not properly validated
                "topk": 50,
                "n_drop": 5,
            # 🧠 ML Signal: Usage of pytest marker to categorize tests
            },
        # 🧠 ML Signal: Testing function for a training process
        },
        "backtest": {
            # ✅ Best Practice: Use of assertGreaterEqual for clear error messages
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            # ✅ Best Practice: Use of assertGreaterEqual for clear error messages
            "account": 100000000,
            # 🧠 ML Signal: Usage of backtest_analysis function indicates a pattern for financial model evaluation
            "benchmark": CSI300_BENCH,
            # 🧠 ML Signal: Use of pytest marker to categorize slow tests
            # 🧠 ML Signal: Asserting on financial metrics like 'excess_return_with_cost' and 'annualized_return' is a common pattern in financial ML models
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            # ✅ Best Practice: Checking for NaN values ensures data integrity in test assertions
            },
        },
    # 🧠 ML Signal: Use of assertTrue indicates a pattern for testing boolean conditions
    # 🧠 ML Signal: Usage of pytest.mark.slow indicates a pattern for categorizing test execution time
    }
    # backtest
    # 🧠 ML Signal: Use of assertTrue indicates a pattern for testing boolean conditions
    par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
    # 🧠 ML Signal: Use of unittest framework for testing
    par.generate()
    # ⚠️ SAST Risk (Medium): Use of shutil.rmtree can delete files or directories, ensure uri_path is validated
    analysis_df = par.load("port_analysis_1day.pkl")
    # 🧠 ML Signal: Creation of a test suite
    print(analysis_df)
    return analysis_df
# 🧠 ML Signal: Adding specific test cases to the test suite


# 🧠 ML Signal: Adding specific test cases to the test suite
class TestAllFlow(TestAutoData):
    REPORT_NORMAL = None
    # 🧠 ML Signal: Adding specific test cases to the test suite
    # 🧠 ML Signal: Use of unittest framework for running tests
    # ✅ Best Practice: Standard Python idiom for making a script executable
    # 🧠 ML Signal: Execution of the test suite
    POSITIONS = None
    RID = None
    URI_PATH = "file:" + str(Path(__file__).parent.joinpath("test_all_flow_mlruns").resolve())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.URI_PATH.lstrip("file:"))

    @pytest.mark.slow
    def test_0_train(self):
        TestAllFlow.PRED_SCORE, ic_ric, TestAllFlow.RID, uri_path = train(self.URI_PATH)
        self.assertGreaterEqual(ic_ric["ic"].all(), 0, "train failed")
        self.assertGreaterEqual(ic_ric["ric"].all(), 0, "train failed")

    @pytest.mark.slow
    def test_1_backtest(self):
        analyze_df = backtest_analysis(TestAllFlow.PRED_SCORE, TestAllFlow.RID, self.URI_PATH)
        self.assertGreaterEqual(
            analyze_df.loc(axis=0)["excess_return_with_cost", "annualized_return"].values[0],
            0.05,
            "backtest failed",
        )
        self.assertTrue(not analyze_df.isna().any().any(), "backtest failed")

    @pytest.mark.slow
    def test_2_expmanager(self):
        pass_default, pass_current, uri_path = fake_experiment()
        self.assertTrue(pass_default, msg="default uri is incorrect")
        self.assertTrue(pass_current, msg="current uri is incorrect")
        shutil.rmtree(str(Path(uri_path.strip("file:")).resolve()))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_train"))
    _suite.addTest(TestAllFlow("test_1_backtest"))
    _suite.addTest(TestAllFlow("test_2_expmanager"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())