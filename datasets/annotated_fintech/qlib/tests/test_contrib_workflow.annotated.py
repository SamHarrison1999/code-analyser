# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.workflow.record_temp import SignalRecord
import shutil
import unittest
import pytest
from pathlib import Path

from qlib.contrib.workflow import MultiSegRecord, SignalMseRecord
# 🧠 ML Signal: Configuration of dataset with specific time segments
# 🧠 ML Signal: Definition of a task configuration for a machine learning model
# 🧠 ML Signal: Usage of a predefined model configuration
# 🧠 ML Signal: Specification of training data time range
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.tests import TestAutoData
from qlib.tests.config import GBDT_MODEL, get_dataset_config, CSI300_MARKET


CSI300_GBDT_TASK = {
    "model": GBDT_MODEL,
    "dataset": get_dataset_config(
        train=("2020-05-01", "2020-06-01"),
        valid=("2020-06-01", "2020-07-01"),
        test=("2020-07-01", "2020-08-01"),
        handler_kwargs={
            "start_time": "2020-05-01",
            "end_time": "2020-08-01",
            # 🧠 ML Signal: Specification of validation data time range
            # 🧠 ML Signal: Additional dataset handler configuration
            # 🧠 ML Signal: Function definition for training a model, indicating a common ML operation
            "fit_start_time": "<dataset.kwargs.segments.train.0>",
            "fit_end_time": "<dataset.kwargs.segments.train.1>",
            # 🧠 ML Signal: Initialization of a model instance, common in ML workflows
            "instruments": CSI300_MARKET,
        },
    # 🧠 ML Signal: Specification of market instruments for the dataset
    # 🧠 ML Signal: Initialization of a dataset instance, common in ML workflows
    ),
}
# ⚠️ SAST Risk (Low): Potential risk if uri_path is not validated or sanitized


# 🧠 ML Signal: Logging parameters, common in ML experiment tracking
def train_multiseg(uri_path: str = None):
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    # 🧠 ML Signal: Model fitting, a core ML operation
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    # 🧠 ML Signal: Function signature indicates a training process with a model and dataset
    with R.start(experiment_name="workflow", uri=uri_path):
        # 🧠 ML Signal: Retrieving a recorder, indicating experiment tracking
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        # 🧠 ML Signal: Model initialization from a configuration, common in ML workflows
        model.fit(dataset)
        # 🧠 ML Signal: Creating a record for model and dataset, common in ML workflows
        recorder = R.get_recorder()
        # 🧠 ML Signal: Dataset initialization from a configuration, common in ML workflows
        sr = MultiSegRecord(model, dataset, recorder)
        # 🧠 ML Signal: Generating records for validation and testing, common in ML workflows
        sr.generate(dict(valid="valid", test="test"), True)
        # 🧠 ML Signal: Use of experiment tracking with a start method
        uri = R.get_uri()
    # 🧠 ML Signal: Retrieving URI, indicating experiment tracking
    return uri
# 🧠 ML Signal: Logging parameters for experiment tracking

# 🧠 ML Signal: Returning URI, indicating the end of an ML experiment

# 🧠 ML Signal: Model training process
def train_mse(uri_path: str = None):
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    # 🧠 ML Signal: Recorder retrieval for tracking experiment metrics
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    # 🧠 ML Signal: Signal recording for model and dataset, indicating a custom tracking mechanism
    # ✅ Best Practice: Constants should be defined at the class level for easy configuration and readability.
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        # ⚠️ SAST Risk (Medium): Potential directory traversal if URI_PATH is not properly validated
        # 🧠 ML Signal: Specific signal recording for MSE, indicating evaluation metric tracking
        recorder = R.get_recorder()
        SignalRecord(recorder=recorder, model=model, dataset=dataset).generate()
        # 🧠 ML Signal: Usage of pytest mark to categorize tests
        # 🧠 ML Signal: Function definition with a test prefix, indicating a test case
        sr = SignalMseRecord(recorder)
        # 🧠 ML Signal: Retrieval of URI, likely for tracking or referencing the experiment
        # 🧠 ML Signal: Usage of a function call within a test case
        sr.generate()
        uri = R.get_uri()
    # 🧠 ML Signal: Function name suggests this is a test case for mean squared error, indicating a pattern for testing ML models
    # 🧠 ML Signal: Returning URI, indicating the end of the training and tracking process
    return uri
# 🧠 ML Signal: Usage of pytest marker for slow tests

# 🧠 ML Signal: Usage of a function that likely trains a model and returns a URI path, indicating a pattern in ML workflows

# ✅ Best Practice: Use of unittest.TestSuite for organizing test cases
class TestAllFlow(TestAutoData):
    URI_PATH = "file:" + str(Path(__file__).parent.joinpath("test_contrib_mlruns").resolve())
    # 🧠 ML Signal: Adding specific test cases to a test suite

    @classmethod
    # 🧠 ML Signal: Adding specific test cases to a test suite
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.URI_PATH.lstrip("file:"))
    # 🧠 ML Signal: Execution of a test suite using a test runner
    # ✅ Best Practice: Standard Python idiom for making a script executable
    # ✅ Best Practice: Use of unittest.TextTestRunner for running tests

    @pytest.mark.slow
    def test_0_multiseg(self):
        uri_path = train_multiseg(self.URI_PATH)

    @pytest.mark.slow
    def test_1_mse(self):
        uri_path = train_mse(self.URI_PATH)


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_multiseg"))
    _suite.addTest(TestAllFlow("test_1_mse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())