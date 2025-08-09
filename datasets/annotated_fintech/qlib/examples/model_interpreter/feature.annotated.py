#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
from qlib.constant import REG_CN

# ⚠️ SAST Risk (Low): Potential path traversal if `provider_uri` is influenced by user input
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
# 🧠 ML Signal: Initialization of a data provider for ML model training
from qlib.tests.config import CSI300_GBDT_TASK

# 🧠 ML Signal: Model configuration and initialization

if __name__ == "__main__":
    # 🧠 ML Signal: Dataset configuration and initialization
    # use default data
    # 🧠 ML Signal: Model training process
    # ✅ Best Practice: Use logging instead of print for better control over output
    # 🧠 ML Signal: Feature importance extraction from the model
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()
    print("feature importance:")
    print(feature_importance)