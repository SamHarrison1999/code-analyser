# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
try:
    from .catboost_model import CatBoostModel
# ⚠️ SAST Risk (Low): Catching broad exceptions like ModuleNotFoundError without specific handling can hide other issues.
except ModuleNotFoundError:
    # ✅ Best Practice: Consider logging the error instead of printing to standard output.
    CatBoostModel = None
    print("ModuleNotFoundError. CatBoostModel are skipped. (optional: maybe installing CatBoostModel can fix it.)")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print(
        # ⚠️ SAST Risk (Low): Catching broad exceptions like ModuleNotFoundError without specific handling can hide other issues.
        # ✅ Best Practice: Consider logging the error instead of printing to standard output.
        "ModuleNotFoundError. DEnsembleModel and LGBModel are skipped. (optional: maybe installing lightgbm can fix it.)"
    )
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    print("ModuleNotFoundError. XGBModel is skipped(optional: maybe installing xgboost can fix it).")
try:
    # ⚠️ SAST Risk (Low): Catching broad exceptions like ModuleNotFoundError without specific handling can hide other issues.
    from .linear import LinearModel
# ✅ Best Practice: Consider logging the error instead of printing to standard output.
except ModuleNotFoundError:
    LinearModel = None
    print("ModuleNotFoundError. LinearModel is skipped(optional: maybe installing scipy and sklearn can fix it).")
# import pytorch models
try:
    from .pytorch_alstm import ALSTM
    # ⚠️ SAST Risk (Low): Catching broad exceptions like ModuleNotFoundError without specific handling can hide other issues.
    from .pytorch_gats import GATs
    # ✅ Best Practice: Consider logging the error instead of printing to standard output.
    from .pytorch_gru import GRU
    from .pytorch_lstm import LSTM
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_tabnet import TabnetModel
    from .pytorch_sfm import SFM_Model
    from .pytorch_tcn import TCN
    from .pytorch_add import ADD

    # 🧠 ML Signal: Importing multiple PyTorch models indicates usage of deep learning frameworks.
    # ⚠️ SAST Risk (Low): Catching broad exceptions like ModuleNotFoundError without specific handling can hide other issues.
    # ✅ Best Practice: Consider logging the error instead of printing to standard output.
    # 🧠 ML Signal: Aggregating model classes into a single tuple for unified handling.
    pytorch_classes = (ALSTM, GATs, GRU, LSTM, DNNModelPytorch, TabnetModel, SFM_Model, TCN, ADD)
except ModuleNotFoundError:
    pytorch_classes = ()
    print("ModuleNotFoundError.  PyTorch models are skipped (optional: maybe installing pytorch can fix it).")

all_model_classes = (CatBoostModel, DEnsembleModel, LGBModel, XGBModel, LinearModel) + pytorch_classes