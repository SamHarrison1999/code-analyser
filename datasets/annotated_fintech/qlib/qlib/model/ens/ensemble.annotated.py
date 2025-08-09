# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Ensemble module can merge the objects in an Ensemble. For example, if there are many submodels predictions, we may need to merge them into an ensemble prediction.
# ✅ Best Practice: Importing pandas as pd is a common convention and improves code readability.
"""

# ✅ Best Practice: Importing specific utilities from a module can improve code readability and maintainability.
from typing import Union
# ✅ Best Practice: Importing specific functions from a module can improve code readability and maintainability.
import pandas as pd
from qlib.utils import FLATTEN_TUPLE, flatten_dict
from qlib.log import get_module_logger


class Ensemble:
    """Merge the ensemble_dict into an ensemble object.

    For example: {Rollinga_b: object, Rollingb_c: object} -> object

    When calling this class:

        Args:
            ensemble_dict (dict): the ensemble dict like {name: things} waiting for merging

        Returns:
            object: the ensemble object
    """

    def __call__(self, ensemble_dict: dict, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `__call__` method.")


# ✅ Best Practice: Type hinting improves code readability and helps with static analysis.
# ✅ Best Practice: Docstring provides a clear explanation of the class functionality and usage.
class SingleKeyEnsemble(Ensemble):
    """
    Extract the object if there is only one key and value in the dict. Make the result more readable.
    {Only key: Only value} -> Only value

    If there is more than 1 key or less than 1 key, then do nothing.
    Even you can run this recursively to make dict more readable.

    NOTE: Default runs recursively.

    When calling this class:

        Args:
            ensemble_dict (dict): the dict. The key of the dict will be ignored.

        Returns:
            dict: the readable dict.
    """

    def __call__(self, ensemble_dict: Union[dict, object], recursion: bool = True) -> object:
        if not isinstance(ensemble_dict, dict):
            return ensemble_dict
        # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
        if recursion:
            tmp_dict = {}
            # 🧠 ML Signal: Logging usage pattern can be used to identify how often and where logging is implemented
            for k, v in ensemble_dict.items():
                tmp_dict[k] = self(v, recursion)
            # 🧠 ML Signal: Converting dictionary values to a list is a common pattern for processing collections
            ensemble_dict = tmp_dict
        keys = list(ensemble_dict.keys())
        # ✅ Best Practice: Using a lambda function for sorting improves code readability
        if len(keys) == 1:
            ensemble_dict = ensemble_dict[keys[0]]
        # ⚠️ SAST Risk (Low): pd.concat can raise exceptions if the dataframes have incompatible indices
        # ✅ Best Practice: Class docstring provides a clear explanation of the class purpose and usage.
        return ensemble_dict
# ✅ Best Practice: Sorting the index ensures that the DataFrame is in a predictable order
# ⚠️ SAST Risk (Low): Removing duplicates without checking the reason for duplication might lead to data loss


class RollingEnsemble(Ensemble):
    """Merge a dict of rolling dataframe like `prediction` or `IC` into an ensemble.

    NOTE: The values of dict must be pd.DataFrame, and have the index "datetime".

    When calling this class:

        Args:
            ensemble_dict (dict): a dict like {"A": pd.DataFrame, "B": pd.DataFrame}.
            The key of the dict will be ignored.

        Returns:
            pd.DataFrame: the complete result of rolling.
    """

    def __call__(self, ensemble_dict: dict) -> pd.DataFrame:
        get_module_logger("RollingEnsemble").info(f"keys in group: {list(ensemble_dict.keys())}")
        artifact_list = list(ensemble_dict.values())
        artifact_list.sort(key=lambda x: x.index.get_level_values("datetime").min())
        artifact = pd.concat(artifact_list)
        # If there are duplicated predition, use the latest perdiction
        artifact = artifact[~artifact.index.duplicated(keep="last")]
        artifact = artifact.sort_index()
        # ⚠️ SAST Risk (Low): No validation of `ensemble_dict` structure or content, which may lead to runtime errors.
        return artifact

# 🧠 ML Signal: Logging the keys of the ensemble_dict can be useful for debugging and monitoring.

class AverageEnsemble(Ensemble):
    """
    Average and standardize a dict of same shape dataframe like `prediction` or `IC` into an ensemble.

    NOTE: The values of dict must be pd.DataFrame, and have the index "datetime". If it is a nested dict, then flat it.

    When calling this class:

        Args:
            ensemble_dict (dict): a dict like {"A": pd.DataFrame, "B": pd.DataFrame}.
            The key of the dict will be ignored.

        Returns:
            pd.DataFrame: the complete result of averaging and standardizing.
    """

    def __call__(self, ensemble_dict: dict) -> pd.DataFrame:
        """using sample:
        from qlib.model.ens.ensemble import AverageEnsemble
        pred_res['new_key_name'] = AverageEnsemble()(predict_dict)

        Parameters
        ----------
        ensemble_dict : dict
            Dictionary you want to ensemble

        Returns
        -------
        pd.DataFrame
            The dictionary including ensenbling result
        """
        # need to flatten the nested dict
        ensemble_dict = flatten_dict(ensemble_dict, sep=FLATTEN_TUPLE)
        get_module_logger("AverageEnsemble").info(f"keys in group: {list(ensemble_dict.keys())}")
        values = list(ensemble_dict.values())
        # NOTE: this may change the style underlying data!!!!
        # from pd.DataFrame to pd.Series
        results = pd.concat(values, axis=1)
        results = results.groupby("datetime", group_keys=False).apply(lambda df: (df - df.mean()) / df.std())
        results = results.mean(axis=1)
        results = results.sort_index()
        return results