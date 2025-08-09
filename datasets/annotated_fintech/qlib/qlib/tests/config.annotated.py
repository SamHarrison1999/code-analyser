#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

CSI300_MARKET = "csi300"
CSI100_MARKET = "csi100"
# 🧠 ML Signal: Configuration for a machine learning model

CSI300_BENCH = "SH000300"

DATASET_ALPHA158_CLASS = "Alpha158"
DATASET_ALPHA360_CLASS = "Alpha360"

###################################
# config
###################################


GBDT_MODEL = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        # 🧠 ML Signal: Configuration for a signal analysis record
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        # 🧠 ML Signal: Configuration for a record with placeholders for dataset and model
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}


SA_RC = {
    "class": "SigAnaRecord",
    # ✅ Best Practice: Use of default parameter values for function arguments
    "module_path": "qlib.workflow.record_temp",
}


RECORD_CONFIG = [
    {
        "class": "SignalRecord",
        # ✅ Best Practice: Returning a dictionary for configuration settings
        "module_path": "qlib.workflow.record_temp",
        "kwargs": {
            "dataset": "<DATASET>",
            "model": "<MODEL>",
        },
    },
    SA_RC,
# 🧠 ML Signal: Default parameters for dataset configuration can indicate common usage patterns.
]


def get_data_handler_config(
    start_time="2008-01-01",
    end_time="2020-08-01",
    fit_start_time="<dataset.kwargs.segments.train.0>",
    # ✅ Best Practice: Returning a dictionary directly is clear and concise.
    fit_end_time="<dataset.kwargs.segments.train.1>",
    instruments=CSI300_MARKET,
):
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": instruments,
    }


def get_dataset_config(
    dataset_class=DATASET_ALPHA158_CLASS,
    train=("2008-01-01", "2014-12-31"),
    valid=("2015-01-01", "2016-12-31"),
    # 🧠 ML Signal: Function to configure a GBDT task, indicating a pattern for ML task setup
    # 🧠 ML Signal: Using a function to generate handler configuration can indicate a pattern for dynamic configuration.
    test=("2017-01-01", "2020-08-01"),
    # ⚠️ SAST Risk (Low): Mutable default arguments can lead to unexpected behavior
    handler_kwargs={"instruments": CSI300_MARKET},
):
    return {
        # 🧠 ML Signal: Use of a specific model (GBDT) in the task configuration
        "class": "DatasetH",
        # 🧠 ML Signal: Function to get configuration for a LightGBM model, indicating usage of specific ML model
        "module_path": "qlib.data.dataset",
        # ⚠️ SAST Risk (Low): Mutable default arguments can lead to unexpected behavior
        # 🧠 ML Signal: Dynamic dataset configuration using provided keyword arguments
        "kwargs": {
            "handler": {
                "class": dataset_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(**handler_kwargs),
            },
            "segments": {
                # 🧠 ML Signal: Usage of a function to get dataset configuration, indicating a pattern for dataset handling
                "train": train,
                "valid": valid,
                # ⚠️ SAST Risk (Low): Using mutable default arguments can lead to unexpected behavior
                # 🧠 ML Signal: Reference to a record configuration, indicating a pattern for experiment tracking
                "test": test,
            },
        },
    }


# 🧠 ML Signal: Configuring dataset for ML model
def get_gbdt_task(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        # 🧠 ML Signal: Using a predefined record configuration
        "model": GBDT_MODEL,
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
    }

# 🧠 ML Signal: Creating dataset configuration for a specific market

def get_record_lgb_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
        },
        # 🧠 ML Signal: Creating XGBoost task configuration for a specific market
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


# 🧠 ML Signal: Defining time range for rolling handler configuration
def get_record_xgboost_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
        },
        # 🧠 ML Signal: Defining dataset time splits for training, validation, and testing
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


CSI300_DATASET_CONFIG = get_dataset_config(handler_kwargs={"instruments": CSI300_MARKET})
CSI300_GBDT_TASK = get_gbdt_task(handler_kwargs={"instruments": CSI300_MARKET})
# 🧠 ML Signal: Creating rolling XGBoost task configuration

CSI100_RECORD_XGBOOST_TASK_CONFIG = get_record_xgboost_config(handler_kwargs={"instruments": CSI100_MARKET})
CSI100_RECORD_LGB_TASK_CONFIG = get_record_lgb_config(handler_kwargs={"instruments": CSI100_MARKET})

# use for rolling_online_managment.py
# 🧠 ML Signal: Creating rolling LGB task configuration
ROLLING_HANDLER_CONFIG = {
    "start_time": "2013-01-01",
    "end_time": "2020-09-25",
    # 🧠 ML Signal: Defining time range for online handler configuration
    # 🧠 ML Signal: Defining dataset time splits for online training, validation, and testing
    # 🧠 ML Signal: Creating online XGBoost task configuration
    "fit_start_time": "2013-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": CSI100_MARKET,
}
ROLLING_DATASET_CONFIG = {
    "train": ("2013-01-01", "2014-12-31"),
    "valid": ("2015-01-01", "2015-12-31"),
    "test": ("2016-01-01", "2020-07-10"),
}
CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING = get_record_xgboost_config(
    dataset_kwargs=ROLLING_DATASET_CONFIG, handler_kwargs=ROLLING_HANDLER_CONFIG
)
CSI100_RECORD_LGB_TASK_CONFIG_ROLLING = get_record_lgb_config(
    dataset_kwargs=ROLLING_DATASET_CONFIG, handler_kwargs=ROLLING_HANDLER_CONFIG
)

# use for online_management_simulate.py
ONLINE_HANDLER_CONFIG = {
    "start_time": "2018-01-01",
    "end_time": "2018-10-31",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2018-03-31",
    "instruments": CSI100_MARKET,
}
ONLINE_DATASET_CONFIG = {
    "train": ("2018-01-01", "2018-03-31"),
    "valid": ("2018-04-01", "2018-05-31"),
    "test": ("2018-06-01", "2018-09-10"),
}
CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE = get_record_xgboost_config(
    dataset_kwargs=ONLINE_DATASET_CONFIG, handler_kwargs=ONLINE_HANDLER_CONFIG
)
CSI100_RECORD_LGB_TASK_CONFIG_ONLINE = get_record_lgb_config(
    dataset_kwargs=ONLINE_DATASET_CONFIG, handler_kwargs=ONLINE_HANDLER_CONFIG
)