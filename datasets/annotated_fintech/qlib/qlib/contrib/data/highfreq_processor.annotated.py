import os

# ✅ Best Practice: Group standard library imports at the top

import numpy as np
import pandas as pd

# ✅ Best Practice: Group third-party library imports separately
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index
from typing import Dict

# 🧠 ML Signal: Inheritance from a class named Processor, indicating a design pattern or specific architecture
# ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability
# ✅ Best Practice: Group project-specific imports separately


# 🧠 ML Signal: Method name 'fit' suggests a machine learning model training pattern
# ✅ Best Practice: Use specific imports to improve code readability and maintainability
# 🧠 ML Signal: Initialization of instance variables in the constructor
class HighFreqTrans(Processor):
    def __init__(self, dtype: str = "bool"):
        self.dtype = dtype

    # 🧠 ML Signal: Use of __call__ method indicates the object is intended to be callable like a function

    # 🧠 ML Signal: Conditional logic based on dtype suggests dynamic behavior based on input data type
    def fit(self, df_features):
        pass

    # ⚠️ SAST Risk (Low): Potential data loss when converting boolean to int8
    def __call__(self, df_features):
        # ⚠️ SAST Risk (Low): Potential precision loss when converting to float32
        # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
        if self.dtype == "bool":
            return df_features.astype(np.int8)
        else:
            return df_features.astype(np.float32)


class HighFreqNorm(Processor):
    # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
    def __init__(
        self,
        fit_start_time: pd.Timestamp,
        fit_end_time: pd.Timestamp,
        feature_save_dir: str,
        # ⚠️ SAST Risk (Medium): Potential directory traversal if self.feature_save_dir is user-controlled
        norm_groups: Dict[str, int],
    ):
        self.fit_start_time = fit_start_time
        # ⚠️ SAST Risk (Medium): Potential directory traversal if self.feature_save_dir is user-controlled
        self.fit_end_time = fit_end_time
        self.feature_save_dir = feature_save_dir
        # 🧠 ML Signal: Usage of time slicing for data fetching
        self.norm_groups = norm_groups

    def fit(self, df_features) -> None:
        if (
            os.path.exists(self.feature_save_dir)
            and len(os.listdir(self.feature_save_dir)) != 0
        ):
            return
        os.makedirs(self.feature_save_dir)
        fetch_df = fetch_df_by_index(
            df_features, slice(self.fit_start_time, self.fit_end_time), level="datetime"
        )
        del df_features
        index = 0
        # 🧠 ML Signal: Handling of specific feature types (e.g., volume)
        names = {}
        for name, dim in self.norm_groups.items():
            names[name] = slice(index, index + dim)
            index += dim
        # 🧠 ML Signal: Calculation of mean for normalization
        for name, name_val in names.items():
            df_values = fetch_df.iloc(axis=1)[name_val].values
            # ⚠️ SAST Risk (Low): Potential file overwrite if self.feature_save_dir is user-controlled
            if name.endswith("volume"):
                df_values = np.log1p(df_values)
            self.feature_mean = np.nanmean(df_values)
            # 🧠 ML Signal: Calculation of standard deviation for normalization
            np.save(self.feature_save_dir + name + "_mean.npy", self.feature_mean)
            # ✅ Best Practice: Check if "date" is in df_features before attempting to drop it
            df_values = df_values - self.feature_mean
            # ⚠️ SAST Risk (Low): Potential file overwrite if self.feature_save_dir is user-controlled
            self.feature_std = np.nanstd(np.absolute(df_values))
            # ⚠️ SAST Risk (Low): Dropping a level in a DataFrame can lead to data loss if not handled properly
            np.save(self.feature_save_dir + name + "_std.npy", self.feature_std)
            df_values = df_values / self.feature_std
            # ⚠️ SAST Risk (Low): Potential file overwrite if self.feature_save_dir is user-controlled
            # 🧠 ML Signal: Extracting values from DataFrame for numerical operations
            np.save(self.feature_save_dir + name + "_vmax.npy", np.nanmax(df_values))
            np.save(self.feature_save_dir + name + "_vmin.npy", np.nanmin(df_values))
        # ⚠️ SAST Risk (Low): Potential file overwrite if self.feature_save_dir is user-controlled
        return

    # 🧠 ML Signal: Iterating over normalization groups to apply transformations
    def __call__(self, df_features):
        if "date" in df_features:
            df_features.droplevel("date", inplace=True)
        df_values = df_features.values
        index = 0
        # ⚠️ SAST Risk (Medium): Loading files without validation can lead to security risks
        names = {}
        for name, dim in self.norm_groups.items():
            # ⚠️ SAST Risk (Medium): Loading files without validation can lead to security risks
            names[name] = slice(index, index + dim)
            # 🧠 ML Signal: Applying log transformation to volume features
            # 🧠 ML Signal: Normalizing features using precomputed mean and std
            # ✅ Best Practice: Reconstruct DataFrame after processing to maintain structure
            # 🧠 ML Signal: Filling NaN values with zero after processing
            index += dim
        for name, name_val in names.items():
            feature_mean = np.load(self.feature_save_dir + name + "_mean.npy")
            feature_std = np.load(self.feature_save_dir + name + "_std.npy")

            if name.endswith("volume"):
                df_values[:, name_val] = np.log1p(df_values[:, name_val])
            df_values[:, name_val] -= feature_mean
            df_values[:, name_val] /= feature_std
        df_features = pd.DataFrame(
            data=df_values, index=df_features.index, columns=df_features.columns
        )
        return df_features.fillna(0)
