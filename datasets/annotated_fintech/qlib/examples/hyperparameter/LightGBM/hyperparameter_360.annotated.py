import qlib
import optuna
# 🧠 ML Signal: Importing specific constants and functions from a library indicates usage patterns.
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import get_dataset_config, CSI300_MARKET, DATASET_ALPHA360_CLASS

DATASET_CONFIG = get_dataset_config(market=CSI300_MARKET, dataset_class=DATASET_ALPHA360_CLASS)
# 🧠 ML Signal: Use of hyperparameter optimization with Optuna
# 🧠 ML Signal: Hyperparameter tuning for LightGBM model
# 🧠 ML Signal: Usage of a function to get a configuration object is a common pattern in ML pipelines.
# ✅ Best Practice: Constants should be in uppercase to indicate they are not meant to be changed.


def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }
    # 🧠 ML Signal: Model initialization and fitting

    evals_result = dict()
    model = init_instance_by_config(task["model"])
    # 🧠 ML Signal: Use of validation results for optimization
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


# ⚠️ SAST Risk (Low): Hardcoded file path, consider making it configurable
# 🧠 ML Signal: Data preparation step
# 🧠 ML Signal: Initialization of Qlib with specific data provider
# 🧠 ML Signal: Dataset initialization
# ⚠️ SAST Risk (Low): Use of SQLite for storage, consider security implications
# 🧠 ML Signal: Optimization process with parallel jobs
if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    dataset = init_instance_by_config(DATASET_CONFIG)

    study = optuna.Study(study_name="LGBM_360", storage="sqlite:///db.sqlite3")
    study.optimize(objective, n_jobs=6)