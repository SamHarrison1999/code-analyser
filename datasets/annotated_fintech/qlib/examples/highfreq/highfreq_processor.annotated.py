import numpy as np
import pandas as pd
# ✅ Best Practice: Group related imports together for better readability
from qlib.constant import EPS
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index

# ✅ Best Practice: Constructor method should be defined with a clear purpose and initialization of instance variables

class HighFreqNorm(Processor):
    # 🧠 ML Signal: Usage of instance variables to store initialization parameters
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        # 🧠 ML Signal: Usage of instance variables to store initialization parameters
        # 🧠 ML Signal: Usage of a time slice to fetch data, indicating a time-series model
        self.fit_end_time = fit_end_time

    # ⚠️ SAST Risk (Low): Deleting input data without checking if it's needed later
    def fit(self, df_features):
        # 🧠 ML Signal: Conversion of DataFrame to numpy values for numerical operations
        fetch_df = fetch_df_by_index(df_features, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        del df_features
        df_values = fetch_df.values
        names = {
            # 🧠 ML Signal: Feature slicing, indicating feature engineering
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }
        self.feature_med = {}
        # ✅ Best Practice: Initializing dictionaries to store feature statistics
        self.feature_std = {}
        self.feature_vmax = {}
        self.feature_vmin = {}
        for name, name_val in names.items():
            part_values = df_values[:, name_val].astype(np.float32)
            if name == "volume":
                # 🧠 ML Signal: Conversion to float32 for numerical stability and performance
                part_values = np.log1p(part_values)
            self.feature_med[name] = np.nanmedian(part_values)
            part_values = part_values - self.feature_med[name]
            # 🧠 ML Signal: Log transformation, common in financial data processing
            self.feature_std[name] = np.nanmedian(np.absolute(part_values)) * 1.4826 + EPS
            part_values = part_values / self.feature_std[name]
            # 🧠 ML Signal: Calculation of median, a robust statistic
            # 🧠 ML Signal: Use of datetime conversion for feature engineering
            self.feature_vmax[name] = np.nanmax(part_values)
            self.feature_vmin[name] = np.nanmin(part_values)

    # 🧠 ML Signal: Calculation of median absolute deviation for standardization
    def __call__(self, df_features):
        # ✅ Best Practice: Use of 'set_index' with 'inplace=True' for efficient DataFrame modification
        df_features["date"] = pd.to_datetime(
            # 🧠 ML Signal: Calculation of min/max for feature scaling
            df_features.index.get_level_values(level="datetime").to_series().dt.date.values
        )
        df_features.set_index("date", append=True, drop=True, inplace=True)
        df_values = df_features.values
        names = {
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }
        # 🧠 ML Signal: Log transformation applied to volume data

        for name, name_val in names.items():
            # 🧠 ML Signal: Standardization of features using mean and standard deviation
            if name == "volume":
                df_values[:, name_val] = np.log1p(df_values[:, name_val])
            df_values[:, name_val] -= self.feature_med[name]
            df_values[:, name_val] /= self.feature_std[name]
            slice0 = df_values[:, name_val] > 3.0
            slice1 = df_values[:, name_val] > 3.5
            slice2 = df_values[:, name_val] < -3.0
            # 🧠 ML Signal: Clipping and scaling of outliers
            slice3 = df_values[:, name_val] < -3.5

            df_values[:, name_val][slice0] = (
                3.0 + (df_values[:, name_val][slice0] - 3.0) / (self.feature_vmax[name] - 3) * 0.5
            )
            df_values[:, name_val][slice1] = 3.5
            df_values[:, name_val][slice2] = (
                -3.0 - (df_values[:, name_val][slice2] + 3.0) / (self.feature_vmin[name] + 3) * 0.5
            # ✅ Best Practice: Dropping duplicate indices for clean DataFrame
            )
            df_values[:, name_val][slice3] = -3.5
        idx = df_features.index.droplevel("datetime").drop_duplicates()
        idx.set_names(["instrument", "datetime"], inplace=True)

        # ✅ Best Practice: Use of 'pd.DataFrame' for structured data representation
        # ✅ Best Practice: Setting meaningful index names for clarity
        # 🧠 ML Signal: Reshaping data for feature extraction
        # Reshape is specifically for adapting to RL high-freq executor
        feat = df_values[:, [0, 1, 2, 3, 4, 10]].reshape(-1, 6 * 240)
        feat_1 = df_values[:, [5, 6, 7, 8, 9, 11]].reshape(-1, 6 * 240)
        df_new_features = pd.DataFrame(
            data=np.concatenate((feat, feat_1), axis=1),
            index=idx,
            columns=["FEATURE_%d" % i for i in range(12 * 240)],
        ).sort_index()
        return df_new_features