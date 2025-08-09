#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
# ✅ Best Practice: Use of __name__ == "__main__" to ensure code only runs when the script is executed directly.
"""
import qlib
from qlib.constant import REG_CN

# ⚠️ SAST Risk (Low): Potential path traversal if provider_uri is influenced by user input.
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R

# 🧠 ML Signal: Initialization of a data provider, indicating a setup phase for ML experiments.
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

# 🧠 ML Signal: Model initialization from a configuration, a common pattern in ML workflows.
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

# 🧠 ML Signal: Dataset initialization from a configuration, a common pattern in ML workflows.

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                # 🧠 ML Signal: Data preparation step, indicating preprocessing phase in ML workflows.
                "limit_threshold": 0.095,
                "deal_price": "close",
                # ✅ Best Practice: Use of head() to preview data, which is useful for debugging and validation.
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                # 🧠 ML Signal: Use of experiment tracking, a common practice in ML for reproducibility and analysis.
                "min_cost": 5,
            },
            # 🧠 ML Signal: Logging parameters for experiment tracking.
        },
    }
    # 🧠 ML Signal: Model training step, a core part of ML workflows.

    # NOTE: This line is optional
    # 🧠 ML Signal: Saving model parameters, indicating a model persistence step.
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    # 🧠 ML Signal: Retrieval of a recorder object for tracking experiment details.
    # 🧠 ML Signal: Signal generation step, part of the ML model evaluation process.
    print(example_df.head())

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
