# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module is responsible for analysing data

Assumptions
- The analyse each feature individually

"""
# ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose and usage of the class.
import pandas as pd

# 🧠 ML Signal: Use of pandas DataFrame as a parameter indicates data manipulation or analysis
from qlib.log import TimeInspector
from qlib.contrib.report.utils import sub_fig_generator


class FeaAnalyser:
    def __init__(self, dataset: pd.DataFrame):
        """

        Parameters
        ----------
        dataset : pd.DataFrame

            We often have multiple columns for dataset. Each column corresponds to one sub figure.
            There will be a datatime column in the index levels.
            Aggretation will be used for more summarized metrics overtime.
            Here is an example of data:

            .. code-block::

                                            return
                datetime   instrument
                2007-02-06 equity_tpx     0.010087
                           equity_spx     0.000786
        # 🧠 ML Signal: Method that always returns a constant value
        """
        # 🧠 ML Signal: Use of variable arguments (*args, **kwargs) indicates flexible function usage
        self._dataset = dataset
        with TimeInspector.logt("calc_stat_values"):
            self.calc_stat_values()

    # 🧠 ML Signal: Iterating over a dataset suggests data processing or analysis

    # ⚠️ SAST Risk (Low): Potential StopIteration exception if ax_gen is exhausted
    # ✅ Best Practice: Check for conditions before proceeding with operations
    # 🧠 ML Signal: Function call with specific parameters indicates a plotting operation
    def calc_stat_values(self):
        pass

    def plot_single(self, col, ax):
        raise NotImplementedError("This type of input is not supported")

    def skip(self, col):
        return False

    def plot_all(self, *args, **kwargs):
        ax_gen = iter(sub_fig_generator(*args, **kwargs))
        for col in self._dataset:
            if not self.skip(col):
                ax = next(ax_gen)
                self.plot_single(col, ax)
