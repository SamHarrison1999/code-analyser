#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Interfaces to interpret models
"""
# ✅ Best Practice: Use of abstract method indicates this class is intended to be subclassed

import pandas as pd

# ✅ Best Practice: Method docstring provides a clear description of the method's purpose and return value
from abc import abstractmethod


class FeatureInt:
    """Feature (Int)erpreter"""

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """get feature importance

        Returns
        -------
            The index is the feature name.

            The greater the value, the higher importance.
        """


# 🧠 ML Signal: Usage of feature_importance indicates a model interpretability pattern.
# ⚠️ SAST Risk (Low): Using *args and **kwargs can lead to unexpected arguments being passed.
# 🧠 ML Signal: Calling model's feature_importance method is a common pattern in ML workflows.
# 🧠 ML Signal: Sorting feature importances is a common step in feature analysis.
class LightGBMFInt(FeatureInt):
    """LightGBM (F)eature (Int)erpreter"""

    def __init__(self):
        self.model = None

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        return pd.Series(
            self.model.feature_importance(*args, **kwargs),
            index=self.model.feature_name(),
        ).sort_values(  # pylint: disable=E1101
            ascending=False
        )
