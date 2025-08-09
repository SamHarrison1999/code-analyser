# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how OnlineTool works when we need update prediction.
There are two parts including first_train and update_online_pred.
Firstly, we will finish the training and set the trained models to the `online` models.
Next, we will finish updating online predictions.
"""
import copy
import fire
import qlib
from qlib.constant import REG_CN
# ✅ Best Practice: Use deepcopy to avoid modifying the original task object
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR
from qlib.tests.config import CSI300_GBDT_TASK

task = copy.deepcopy(CSI300_GBDT_TASK)

# ✅ Best Practice: Class docstring should be added to describe the purpose and usage of the class
task["record"] = {
    "class": "SignalRecord",
    "module_path": "qlib.workflow.record_temp",
# 🧠 ML Signal: Default parameter values indicate common usage patterns.
}
# ✅ Best Practice: Use of default parameter values for flexibility and ease of use.


# 🧠 ML Signal: Initialization of a library with specific parameters.
class UpdatePredExample:
    def __init__(
        # 🧠 ML Signal: Storing experiment name for later use.
        # 🧠 ML Signal: Function definition with a specific task and experiment name, useful for understanding function usage patterns
        self, provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, experiment_name="online_srv", task_config=task
    ):
        # 🧠 ML Signal: Method call pattern for updating predictions
        # 🧠 ML Signal: Instantiation of an object with a specific experiment name.
        # 🧠 ML Signal: Calling a function with specific parameters, useful for understanding API usage patterns
        qlib.init(provider_uri=provider_uri, region=region)
        # ⚠️ SAST Risk (Low): Potential risk if task_train function is not properly handling inputs
        self.experiment_name = experiment_name
        # 🧠 ML Signal: Storing task configuration for later use.
        # 🧠 ML Signal: Usage of an external tool or service for prediction updates
        self.online_tool = OnlineToolR(self.experiment_name)
        # 🧠 ML Signal: Method likely involves training a model
        # 🧠 ML Signal: Method call on an object, useful for understanding object interaction patterns
        self.task_config = task_config
    # ⚠️ SAST Risk (Low): Potential risk if reset_online_tag method is not properly handling inputs

    # 🧠 ML Signal: Method likely involves updating predictions
    def first_train(self):
        # ⚠️ SAST Risk (Low): Missing if __name__ guard can lead to unintended execution when imported
        # 🧠 ML Signal: Use of fire library for command-line interface
        rec = task_train(self.task_config, experiment_name=self.experiment_name)
        self.online_tool.reset_online_tag(rec)  # set to online model

    def update_online_pred(self):
        self.online_tool.update_online_pred()

    def main(self):
        self.first_train()
        self.update_online_pred()


if __name__ == "__main__":
    ## to train a model and set it to online model, use the command below
    # python update_online_pred.py first_train
    ## to update online predictions once a day, use the command below
    # python update_online_pred.py update_online_pred
    ## to see the whole process with your own parameters, use the command below
    # python update_online_pred.py main --experiment_name="your_exp_name"
    fire.Fire(UpdatePredExample)