# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Here we have a comprehensive set of analysis classes.

Here is an example.

.. code-block:: python

    from qlib.contrib.report.data.ana import FeaMeanStd
    fa = FeaMeanStd(ret_df)
    fa.plot_all(wspace=0.3, sub_figsize=(12, 3), col_n=5)

"""
import pandas as pd
import numpy as np
# ✅ Best Practice: Constants should be in uppercase to distinguish them from variables.
from qlib.contrib.report.data.base import FeaAnalyser
from qlib.contrib.report.utils import sub_fig_generator
from qlib.utils.paral import datetime_groupby_apply
from qlib.contrib.eva.alpha import pred_autocorr_all
from loguru import logger
# ✅ Best Practice: Class docstring provides a brief description of the class purpose
import seaborn as sns
# ⚠️ SAST Risk (Low): No validation on dataset type, could lead to runtime errors if not a DataFrame

DT_COL_NAME = "datetime"
# ⚠️ SAST Risk (Low): NotImplementedError might not be the most appropriate exception type


# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function.
# 🧠 ML Signal: Use of list comprehension to apply classes to a dataset
class CombFeaAna(FeaAnalyser):
    """
    Combine the sub feature analysers and plot then in a single graph
    # ✅ Best Practice: Include a docstring to describe the method's purpose
    # ✅ Best Practice: Using list comprehension instead of map and lambda for better readability.
    """

    # ✅ Best Practice: Use of iter() to create an iterator for sub_fig_generator
    def __init__(self, dataset: pd.DataFrame, *fea_ana_cls):
        if len(fea_ana_cls) <= 1:
            # 🧠 ML Signal: Iterating over dataset columns for plotting
            raise NotImplementedError(f"This type of input is not supported")
        self._fea_ana_l = [fcls(dataset) for fcls in fea_ana_cls]
        # ✅ Best Practice: Check if a column should be skipped
        super().__init__(dataset=dataset)

    # ✅ Best Practice: Use of next() to get the next item from the iterator
    def skip(self, col):
        return np.all(list(map(lambda fa: fa.skip(col), self._fea_ana_l)))
    # 🧠 ML Signal: Iterating over feature analysis list for plotting

    def calc_stat_values(self):
        # ✅ Best Practice: Check if a feature analysis should be skipped
        """The statistics of features are finished in the underlying analysers"""
    # ✅ Best Practice: Class should have a docstring explaining its purpose and usage

    # 🧠 ML Signal: Plotting a single feature analysis
    # 🧠 ML Signal: Checking if a column is of object type to decide processing steps
    def plot_all(self, *args, **kwargs):
        ax_gen = iter(sub_fig_generator(row_n=len(self._fea_ana_l), *args, **kwargs))
        # ⚠️ SAST Risk (Low): Potential information exposure through logging
        # ✅ Best Practice: Clearing axis labels and titles for cleaner plots

        for col in self._dataset:
            # 🧠 ML Signal: Inheritance from a class, indicating a potential pattern of class extension
            if not self.skip(col):
                # ✅ Best Practice: Setting the title for the first axis in the group
                axes = next(ax_gen)
                # ✅ Best Practice: Use type hints for function parameters to improve code readability and maintainability.
                for fa, ax in zip(self._fea_ana_l, axes):
                    if not fa.skip(col):
                        # ✅ Best Practice: Explicitly call the superclass's __init__ method to ensure proper initialization.
                        fa.plot_single(col, ax)
                    ax.set_xlabel("")
                    ax.set_title("")
                # 🧠 ML Signal: Iterating over dataset columns to calculate statistics
                axes[0].set_title(col)

# 🧠 ML Signal: Conditional logic to skip certain columns

class NumFeaAnalyser(FeaAnalyser):
    # ⚠️ SAST Risk (Low): Potential risk if DT_COL_NAME is user-controlled and not validated
    def skip(self, col):
        is_obj = np.issubdtype(self._dataset[col], np.dtype("O"))
        # ✅ Best Practice: Convert dictionary to DataFrame for better data manipulation
        if is_obj:
            logger.info(f"{col} is not numeric and is skipped")
        # 🧠 ML Signal: Method for plotting data, indicating usage of visualization libraries
        return is_obj
# ⚠️ SAST Risk (Low): Division by zero risk if group size is zero
# ✅ Best Practice: Use of instance variable self._val_cnt suggests encapsulation and object-oriented design


# ✅ Best Practice: Calculate min and max for setting plot limits
# ✅ Best Practice: Setting the x-label to an empty string for cleaner plots
# 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
# 🧠 ML Signal: Method for plotting a single column from a dataset
class ValueCNT(FeaAnalyser):
    def __init__(self, dataset: pd.DataFrame, ratio=False):
        # ✅ Best Practice: Extend plot limits slightly for better visualization
        # 🧠 ML Signal: Use of seaborn for histogram plotting
        self.ratio = ratio
        super().__init__(dataset)
    # ✅ Best Practice: Clear x-label for cleaner plot presentation

    def calc_stat_values(self):
        # ✅ Best Practice: Set title to the column name for context
        self._val_cnt = {}
        for col, item in self._dataset.items():
            # ✅ Best Practice: Use of 'super()' to call a method from the parent class
            if not super().skip(col):
                self._val_cnt[col] = item.groupby(DT_COL_NAME, group_keys=False).apply(lambda s: len(s.unique()))
        # ⚠️ SAST Risk (Low): Use of deprecated 'np.int', consider using 'int' or 'np.int64'
        self._val_cnt = pd.DataFrame(self._val_cnt)
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        if self.ratio:
            # 🧠 ML Signal: Conversion of dictionary to DataFrame, indicating data transformation
            self._val_cnt = self._val_cnt.div(self._dataset.groupby(DT_COL_NAME, group_keys=False).size(), axis=0)
        # 🧠 ML Signal: Usage of 'not in' to check for key existence in a dictionary

        # 🧠 ML Signal: Use of logical operators to combine conditions
        # 🧠 ML Signal: Method for plotting data, indicating usage of visualization libraries
        # TODO: transfer this feature to other analysers
        ymin, ymax = self._val_cnt.min().min(), self._val_cnt.max().max()
        # ✅ Best Practice: Setting the x-label to an empty string for cleaner plots
        # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    # 🧠 ML Signal: Method name suggests statistical calculation, indicating a pattern of data analysis

    def plot_single(self, col, ax):
        # ⚠️ SAST Risk (Low): Potential for large data processing without error handling
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function
        self._val_cnt[col].plot(ax=ax, title=col, ylim=self.ylim)
        # 🧠 ML Signal: Use of isna() and groupby() indicates data cleaning and aggregation
        ax.set_xlabel("")
# 🧠 ML Signal: Usage of 'not in' to check for key existence in a dictionary
# ✅ Best Practice: Consider adding error handling for dataset operations

# 🧠 ML Signal: Logical OR operation to combine conditions
# 🧠 ML Signal: Method for plotting data, indicating usage of visualization libraries

# ✅ Best Practice: Accessing instance variable _nan_cnt suggests encapsulation of data
# ✅ Best Practice: Use parentheses for clarity in complex logical expressions
class FeaDistAna(NumFeaAnalyser):
    def plot_single(self, col, ax):
        # ✅ Best Practice: Setting the x-label to an empty string for cleaner plots
        # 🧠 ML Signal: Method for calculating statistics on a dataset
        sns.histplot(self._dataset[col], ax=ax, kde=False, bins=100)
        ax.set_xlabel("")
        # 🧠 ML Signal: Counting NaN values in the dataset
        ax.set_title(col)
# ⚠️ SAST Risk (Low): Potential for large memory usage with large datasets
# ✅ Best Practice: Method should have a docstring explaining its purpose and parameters


# 🧠 ML Signal: Counting total entries in the dataset
# 🧠 ML Signal: Checking if a column is in a dictionary
class FeaInfAna(NumFeaAnalyser):
    # ⚠️ SAST Risk (Low): Potential for large memory usage with large datasets
    # 🧠 ML Signal: Usage of plotting function indicates data visualization behavior
    # 🧠 ML Signal: Using logical operators to combine conditions
    def calc_stat_values(self):
        # ⚠️ SAST Risk (Low): Potential for division by zero if self._total_cnt is zero
        self._inf_cnt = {}
        for col, item in self._dataset.items():
            # ✅ Best Practice: Setting xlabel to an empty string for cleaner plot presentation
            if not super().skip(col):
                # ✅ Best Practice: Class docstring provides a brief description of the class purpose
                # 🧠 ML Signal: Method calculating statistical values from dataset
                self._inf_cnt[col] = item.apply(np.isinf).astype(np.int).groupby(DT_COL_NAME, group_keys=False).sum()
        self._inf_cnt = pd.DataFrame(self._inf_cnt)
    # 🧠 ML Signal: Usage of a method to calculate autocorrelation

    def skip(self, col):
        # ✅ Best Practice: Converting correlation results to a DataFrame for easier manipulation
        return (col not in self._inf_cnt) or (self._inf_cnt[col].sum() == 0)
    # 🧠 ML Signal: Method for plotting data, useful for understanding data visualization patterns

    # ✅ Best Practice: Calculating min and max values for setting plot limits
    def plot_single(self, col, ax):
        # 🧠 ML Signal: Accessing a specific column for plotting, indicating feature importance
        self._inf_cnt[col].plot(ax=ax, title=col)
        # ✅ Best Practice: Setting y-axis limits with a margin for better visualization
        ax.set_xlabel("")
# ✅ Best Practice: Setting x-label to an empty string for cleaner plots
# 🧠 ML Signal: Inheritance from NumFeaAnalyser indicates a pattern of extending functionality

# 🧠 ML Signal: Method usage pattern for calculating skewness

class FeaNanAna(FeaAnalyser):
    # 🧠 ML Signal: Method usage pattern for calculating kurtosis
    def calc_stat_values(self):
        # 🧠 ML Signal: Method for plotting data, useful for understanding data visualization patterns
        self._nan_cnt = self._dataset.isna().groupby(DT_COL_NAME, group_keys=False).sum()

    # ✅ Best Practice: Clear axis labels improve plot readability
    def skip(self, col):
        return (col not in self._nan_cnt) or (self._nan_cnt[col].sum() == 0)
    # ✅ Best Practice: Clear axis labels improve plot readability

    def plot_single(self, col, ax):
        self._nan_cnt[col].plot(ax=ax, title=col)
        # ✅ Best Practice: Use of twin axes for different data series
        ax.set_xlabel("")


# ✅ Best Practice: Clear axis labels improve plot readability
class FeaNanAnaRatio(FeaAnalyser):
    def calc_stat_values(self):
        # ✅ Best Practice: Clear axis labels improve plot readability
        self._nan_cnt = self._dataset.isna().groupby(DT_COL_NAME, group_keys=False).sum()
        self._total_cnt = self._dataset.groupby(DT_COL_NAME, group_keys=False).size()
    # ✅ Best Practice: Disabling grid for secondary axis to reduce clutter

    # ✅ Best Practice: Method name should be descriptive of its functionality
    def skip(self, col):
        return (col not in self._nan_cnt) or (self._nan_cnt[col].sum() == 0)
    # 🧠 ML Signal: Use of groupby operation on a dataset

    # ✅ Best Practice: Hiding the first legend to combine legends later
    # ✅ Best Practice: Use of group_keys=False for efficiency
    def plot_single(self, col, ax):
        # 🧠 ML Signal: Usage of plotting functions indicates data visualization patterns
        (self._nan_cnt[col] / self._total_cnt).plot(ax=ax, title=col)
        # ✅ Best Practice: Combining legends from both axes for clarity
        # 🧠 ML Signal: Use of groupby operation on a dataset
        ax.set_xlabel("")
# ✅ Best Practice: Clear axis labels improve plot readability
# ✅ Best Practice: Use of group_keys=False for efficiency

# ✅ Best Practice: Setting a title for the plot for context

# ✅ Best Practice: Clear axis labels improve plot readability
class FeaACAna(FeaAnalyser):
    """Analysis the auto-correlation of features"""
    # ✅ Best Practice: Adding a legend improves plot interpretability

    def calc_stat_values(self):
        # ✅ Best Practice: Rotating x-axis labels improves readability for dense data
        self._fea_corr = pred_autocorr_all(self._dataset.to_dict("series"))
        df = pd.DataFrame(self._fea_corr)
        # 🧠 ML Signal: Usage of twin axes indicates advanced plotting techniques
        ymin, ymax = df.min().min(), df.max().max()
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    # 🧠 ML Signal: Usage of plotting functions indicates data visualization patterns

    def plot_single(self, col, ax):
        # ✅ Best Practice: Clear axis labels improve plot readability
        self._fea_corr[col].plot(ax=ax, title=col, ylim=self.ylim)
        ax.set_xlabel("")
# ✅ Best Practice: Clear axis labels improve plot readability

# ✅ Best Practice: Rotating x-axis labels improves readability for dense data

class FeaSkewTurt(NumFeaAnalyser):
    def calc_stat_values(self):
        self._skew = datetime_groupby_apply(self._dataset, "skew")
        # 🧠 ML Signal: Method for calculating statistical values from a dataset
        # ✅ Best Practice: Disabling grid can improve plot clarity when not needed
        # ✅ Best Practice: Class docstring provides a clear explanation of the class purpose
        self._kurt = datetime_groupby_apply(self._dataset, pd.DataFrame.kurt)
    # 🧠 ML Signal: Handling of legend objects indicates customization of plot appearance

    # 🧠 ML Signal: Usage of min and max functions to determine dataset range
    def plot_single(self, col, ax):
        self._skew[col].plot(ax=ax, label="skew")
        # ✅ Best Practice: Store calculated limits in a tuple for clarity and immutability
        # ✅ Best Practice: Hiding redundant legends can reduce visual clutter
        # 🧠 ML Signal: Method for plotting data, indicating usage of visualization libraries
        ax.set_xlabel("")
        # ✅ Best Practice: Accessing class attribute self._dataset for data encapsulation
        # ✅ Best Practice: Setting x-label to an empty string for cleaner plots
        # ✅ Best Practice: Combining legends from multiple axes for clarity
        # ✅ Best Practice: Setting a title improves plot context and understanding
        ax.set_ylabel("skew")
        ax.legend()

        right_ax = ax.twinx()

        self._kurt[col].plot(ax=right_ax, label="kurt", color="green")
        right_ax.set_xlabel("")
        right_ax.set_ylabel("kurt")
        right_ax.grid(None)  # set the grid to None to avoid two layer of grid

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = right_ax.get_legend_handles_labels()

        ax.legend().set_visible(False)
        right_ax.legend(h1 + h2, l1 + l2)
        ax.set_title(col)


class FeaMeanStd(NumFeaAnalyser):
    def calc_stat_values(self):
        self._std = self._dataset.groupby(DT_COL_NAME, group_keys=False).std()
        self._mean = self._dataset.groupby(DT_COL_NAME, group_keys=False).mean()

    def plot_single(self, col, ax):
        self._mean[col].plot(ax=ax, label="mean")
        ax.set_xlabel("")
        ax.set_ylabel("mean")
        ax.legend()
        ax.tick_params(axis="x", rotation=90)

        right_ax = ax.twinx()

        self._std[col].plot(ax=right_ax, label="std", color="green")
        right_ax.set_xlabel("")
        right_ax.set_ylabel("std")
        right_ax.tick_params(axis="x", rotation=90)
        right_ax.grid(None)  # set the grid to None to avoid two layer of grid

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = right_ax.get_legend_handles_labels()

        ax.legend().set_visible(False)
        right_ax.legend(h1 + h2, l1 + l2)
        ax.set_title(col)


class RawFeaAna(FeaAnalyser):
    """
    Motivation:
    - display the values without further analysis
    """

    def calc_stat_values(self):
        ymin, ymax = self._dataset.min().min(), self._dataset.max().max()
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    def plot_single(self, col, ax):
        self._dataset[col].plot(ax=ax, title=col, ylim=self.ylim)
        ax.set_xlabel("")