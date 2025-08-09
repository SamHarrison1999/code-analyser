# 🧠 ML Signal: Usage of hyperopt's hp.choice for hyperparameter optimization
# Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Defining a search space for hyperparameters using discrete choices
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa
# 🧠 ML Signal: Defining a search space for hyperparameters using discrete choices
# 🧠 ML Signal: Usage of hyperopt's hp.choice for hyperparameter optimization

from hyperopt import hp


TopkAmountStrategySpace = {
    "topk": hp.choice("topk", [30, 35, 40]),
    "buffer_margin": hp.choice("buffer_margin", [200, 250, 300]),
}

QLibDataLabelSpace = {
    "labels": hp.choice(
        "labels",
        [["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["Ref($close, -5)/$close - 1"]],
    )
}
