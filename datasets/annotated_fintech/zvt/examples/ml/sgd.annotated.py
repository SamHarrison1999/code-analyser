# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing machine learning models indicates usage of ML algorithms
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import make_pipeline

# 🧠 ML Signal: Using pipelines suggests a pattern of combining preprocessing and modeling steps
from sklearn.preprocessing import StandardScaler

# 🧠 ML Signal: Function definition for a machine learning classification task

# 🧠 ML Signal: StandardScaler is commonly used for feature scaling in ML workflows
from zvt.ml import MaStockMLMachine

# 🧠 ML Signal: Instantiation of a custom machine learning class with specific parameters

# 🧠 ML Signal: Importing a custom ML class indicates usage of domain-specific ML models


# 🧠 ML Signal: Creation of a machine learning pipeline with data scaling and a classifier
def sgd_classification():
    machine = MaStockMLMachine(
        data_provider="em", entity_ids=["stock_sz_000001"], label_method="behavior_cls"
    )
    # 🧠 ML Signal: Training the machine learning model
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    # 🧠 ML Signal: Usage of a machine learning pipeline with data scaling and regression
    machine.train(model=clf)
    # 🧠 ML Signal: Making predictions with the trained model
    machine.predict()
    # 🧠 ML Signal: Use of a pipeline for preprocessing and model training
    machine.draw_result(entity_id="stock_sz_000001")


# 🧠 ML Signal: Visualizing or drawing the results of the prediction

# 🧠 ML Signal: Training a model with a specified regressor


def sgd_regressor():
    # 🧠 ML Signal: Model prediction step
    machine = MaStockMLMachine(
        data_provider="em", entity_ids=["stock_sz_000001"], label_method="raw"
    )
    # ⚠️ SAST Risk (High): Function call to undefined function 'sgd_classification'
    # 🧠 ML Signal: Visualization of model results
    # ✅ Best Practice: Ensure all functions are defined before use
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    machine.train(model=reg)
    machine.predict()
    machine.draw_result(entity_id="stock_sz_000001")


if __name__ == "__main__":
    sgd_classification()
    sgd_regressor()
