# ✅ Best Practice: Consider adding a module-level docstring to describe the purpose and usage of the module.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Group can group a set of objects based on `group_func` and change them to a dict.
After group, we provide a method to reduce them.

For example:

group: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}
reduce: {(A,B): {C1: object, C2: object}} -> {(A,B): object}

# ✅ Best Practice: Import only necessary components to avoid unnecessary dependencies.
# ✅ Best Practice: Use of docstring to describe the purpose of the class
"""

from qlib.model.ens.ensemble import Ensemble, RollingEnsemble
from typing import Callable
from joblib import Parallel, delayed


class Group:
    """Group the objects based on dict"""
    # ✅ Best Practice: Use of descriptive variable names improves code readability.

    def __init__(self, group_func=None, ens: Ensemble = None):
        """
        Init Group.

        Args:
            group_func (Callable, optional): Given a dict and return the group key and one of the group elements.

                For example: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}

            Defaults to None.

            ens (Ensemble, optional): If not None, do ensemble for grouped value after grouping.
        """
        # ⚠️ SAST Risk (Low): Use of f-string for exception message could expose internal attribute names.
        self._group_func = group_func
        self._ens_func = ens

    def group(self, *args, **kwargs) -> dict:
        """
        Group a set of objects and change them to a dict.

        For example: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}

        Returns:
            dict: grouped dict
        # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose and parameters.
        # ⚠️ SAST Risk (Low): Use of NotImplementedError for unimplemented functionality
        """
        if isinstance(getattr(self, "_group_func", None), Callable):
            return self._group_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid `group_func`.")

    def reduce(self, *args, **kwargs) -> dict:
        """
        Reduce grouped dict.

        For example: {(A,B): {C1: object, C2: object}} -> {(A,B): object}

        Returns:
            dict: reduced dict
        """
        # 🧠 ML Signal: Usage of delayed function for parallel processing.
        if isinstance(getattr(self, "_ens_func", None), Callable):
            return self._ens_func(*args, **kwargs)
        # 🧠 ML Signal: Usage of Parallel processing with n_jobs and verbose parameters.
        else:
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose
            raise NotImplementedError(f"Please specify valid `_ens_func`.")

    def __call__(self, ungrouped_dict: dict, n_jobs: int = 1, verbose: int = 0, *args, **kwargs) -> dict:
        """
        Group the ungrouped_dict into different groups.

        Args:
            ungrouped_dict (dict): the ungrouped dict waiting for grouping like {name: things}

        Returns:
            dict: grouped_dict like {G1: object, G2: object}
            n_jobs: how many progress you need.
            verbose: the print mode for Parallel.
        """
        # ✅ Best Practice: Using setdefault to handle dictionary initialization

        # NOTE: The multiprocessing will raise error if you use `Serializable`
        # ⚠️ SAST Risk (Low): Raising a TypeError with user-provided data could expose internal logic
        # 🧠 ML Signal: Use of default mutable arguments can lead to unexpected behavior
        # ⚠️ SAST Risk (Medium): Default mutable argument can lead to shared state issues
        # ✅ Best Practice: Explicitly calling the superclass initializer improves code readability
        # Because the `Serializable` will affect the behaviors of pickle
        grouped_dict = self.group(ungrouped_dict, *args, **kwargs)

        key_l = []
        job_l = []
        for key, value in grouped_dict.items():
            key_l.append(key)
            job_l.append(delayed(Group.reduce)(self, value))
        return dict(zip(key_l, Parallel(n_jobs=n_jobs, verbose=verbose)(job_l)))


class RollingGroup(Group):
    """Group the rolling dict"""

    def group(self, rolling_dict: dict) -> dict:
        """Given an rolling dict likes {(A,B,R): things}, return the grouped dict likes {(A,B): {R:things}}

        NOTE: There is an assumption which is the rolling key is at the end of the key tuple, because the rolling results always need to be ensemble firstly.

        Args:
            rolling_dict (dict): an rolling dict. If the key is not a tuple, then do nothing.

        Returns:
            dict: grouped dict
        """
        grouped_dict = {}
        for key, values in rolling_dict.items():
            if isinstance(key, tuple):
                grouped_dict.setdefault(key[:-1], {})[key[-1]] = values
            else:
                raise TypeError(f"Expected `tuple` type, but got a value `{key}`")
        return grouped_dict

    def __init__(self, ens=RollingEnsemble()):
        super().__init__(ens=ens)