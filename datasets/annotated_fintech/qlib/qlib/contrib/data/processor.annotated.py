import numpy as np

# ✅ Best Practice: Importing specific classes or functions improves code readability and maintainability.

from ...log import TimeInspector

# ✅ Best Practice: Class docstring provides a description of the class and its purpose
# ✅ Best Practice: Importing specific classes or functions improves code readability and maintainability.
from ...data.dataset.processor import Processor, get_group_columns

# ✅ Best Practice: Method docstring provides a description of the method and its purpose


class ConfigSectionProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    # 🧠 ML Signal: Use of kwargs to configure object behavior

    def __init__(self, fields_group=None, **kwargs):
        # 🧠 ML Signal: Use of kwargs to configure object behavior
        super().__init__()
        # Options
        # 🧠 ML Signal: Use of kwargs to configure object behavior
        self.fillna_feature = kwargs.get("fillna_feature", True)
        # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be used as a function
        self.fillna_label = kwargs.get("fillna_label", True)
        # 🧠 ML Signal: Use of kwargs to configure object behavior
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        # 🧠 ML Signal: Passing a DataFrame to a method suggests data manipulation or transformation
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        # 🧠 ML Signal: Use of kwargs to configure object behavior
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the _transform method.
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        # ✅ Best Practice: Initialize instance variables in the constructor
        # 🧠 ML Signal: Normalizing data by subtracting the mean and dividing by the standard deviation
        self.fields_group = None

    # 🧠 ML Signal: Normalizing data by dividing by the standard deviation
    def __call__(self, df):
        return self._transform(df)

    # ⚠️ SAST Risk (Low): Potential for division by zero if x.std() is zero

    def _transform(self, df):
        # 🧠 ML Signal: Clipping outliers to a specified range
        def _label_norm(x):
            # 🧠 ML Signal: Normalization of features is a common preprocessing step in ML pipelines.
            x = x - x.mean()  # copy
            x /= x.std()
            # 🧠 ML Signal: Filling NaN values with a specified value
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            # ⚠️ SAST Risk (Low): In-place modification of data can lead to unexpected side effects.
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        # ⚠️ SAST Risk (Low): In-place modification of data can lead to unexpected side effects.

        def _feature_norm(x):
            # ⚠️ SAST Risk (Low): In-place modification of data can lead to unexpected side effects.
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                # ⚠️ SAST Risk (Low): In-place modification of data can lead to unexpected side effects.
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                # 🧠 ML Signal: Time tracking for performance monitoring is useful in ML pipelines.
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                # 🧠 ML Signal: Selecting specific columns for processing is a common pattern in data preprocessing.
                x.fillna(0, inplace=True)
            # ✅ Best Practice: Creating a copy of the DataFrame to avoid modifying the original data.
            return x

        TimeInspector.set_time_mark()

        # ✅ Best Practice: Dropping unnecessary levels in a MultiIndex for cleaner data.
        # Copy the focus part and change it to single level
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        # 🧠 ML Signal: Label normalization is a common preprocessing step in ML pipelines.
        # 🧠 ML Signal: Feature transformation using power functions is common in ML preprocessing.
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        # Label
        cols = df_focus.columns[df_focus.columns.str.contains("^LABEL")]
        df_focus[cols] = (
            df_focus[cols]
            .groupby(level="datetime", group_keys=False)
            .apply(_label_norm)
        )

        # Features
        cols = df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: x**0.25)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: x**0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VWAP",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            # 🧠 ML Signal: Pattern matching for column selection is a common preprocessing step.
            "VSUMP",
            # 🧠 ML Signal: Log transformation is a common technique to handle skewed data.
            "VSUMN",
            "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df_focus.columns[
            df_focus.columns.str.contains(pat)
            & (~df_focus.columns.isin(["HIGH0", "LOW0"]))
        ]
        df_focus[cols] = (
            df_focus[cols]
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )
        # 🧠 ML Signal: Handling missing values is a common preprocessing step in ML pipelines.

        # 🧠 ML Signal: Feature transformation using power functions is common in ML preprocessing.
        cols = df_focus.columns[
            df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")
        ]
        df_focus[cols] = (
            df_focus[cols]
            .apply(np.log)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^RSQR")]
        df_focus[cols] = (
            df_focus[cols]
            .fillna(0)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )
        # ✅ Best Practice: Assigning processed values back to the original DataFrame.
        # 🧠 ML Signal: Feature transformation using power functions is common in ML preprocessing.
        # 🧠 ML Signal: Exponential transformation is a common technique in data preprocessing.
        # 🧠 ML Signal: Log1p transformation is a common technique to handle skewed data.
        # 🧠 ML Signal: Logging time taken for operations is useful for performance monitoring.

        cols = df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (x - 1) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (1 - x) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(np.exp)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^WVMA")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(np.log1p)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        df[selected_cols] = df_focus.values

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df
