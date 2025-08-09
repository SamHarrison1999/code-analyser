# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing machine learning models indicates usage of ML for classification or regression tasks
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import make_pipeline
# 🧠 ML Signal: Using pipelines suggests a pattern of combining preprocessing and modeling steps
from sklearn.preprocessing import StandardScaler

# 🧠 ML Signal: StandardScaler is commonly used for feature scaling in ML pipelines
from zvt.contract import AdjustType
from zvt.ml import MaStockMLMachine
# ⚠️ SAST Risk (Low): Importing from external libraries can introduce security risks if not properly vetted

# 🧠 ML Signal: Function definition for testing a specific machine learning model
start_timestamp = "2015-01-01"
# ⚠️ SAST Risk (Low): Importing from external libraries can introduce security risks if not properly vetted
# ✅ Best Practice: Using ISO 8601 format for dates improves readability and consistency
# 🧠 ML Signal: Instantiation of a custom machine learning machine with specific parameters
end_timestamp = "2019-01-01"
predict_start_timestamp = "2018-06-01"


def test_sgd_classification():
    machine = MaStockMLMachine(
        data_provider="joinquant",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        # ✅ Best Practice: Using ISO 8601 format for dates improves readability and consistency
        predict_start_timestamp=predict_start_timestamp,
        entity_ids=["stock_sz_000001"],
        # 🧠 ML Signal: Creation of a machine learning pipeline with data scaling and classification
        label_method="behavior_cls",
        adjust_type=AdjustType.qfq,
    # 🧠 ML Signal: Function definition for testing a machine learning model
    # 🧠 ML Signal: Training the machine learning model
    )
    # 🧠 ML Signal: Instantiation of a custom machine learning class
    # 🧠 ML Signal: Making predictions with the trained model
    # 🧠 ML Signal: Visualizing the prediction results
    # 🧠 ML Signal: Use of a specific data provider for machine learning
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    machine.train(model=clf)
    machine.predict()
    machine.draw_result(entity_id="stock_sz_000001")


def test_sgd_regressor():
    machine = MaStockMLMachine(
        data_provider="joinquant",
        # 🧠 ML Signal: Use of specific timestamps for training data
        start_timestamp=start_timestamp,
        # 🧠 ML Signal: Use of specific entity IDs for training
        end_timestamp=end_timestamp,
        predict_start_timestamp=predict_start_timestamp,
        # 🧠 ML Signal: Creation of a machine learning pipeline with preprocessing and model
        # 🧠 ML Signal: Training the machine learning model
        # 🧠 ML Signal: Making predictions with the trained model
        # 🧠 ML Signal: Visualization of prediction results
        # 🧠 ML Signal: Use of a specific label method
        # 🧠 ML Signal: Use of a specific adjustment type
        entity_ids=["stock_sz_000001"],
        label_method="raw",
        adjust_type=AdjustType.qfq,
    )
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    machine.train(model=reg)
    machine.predict()
    machine.draw_result(entity_id="stock_sz_000001")