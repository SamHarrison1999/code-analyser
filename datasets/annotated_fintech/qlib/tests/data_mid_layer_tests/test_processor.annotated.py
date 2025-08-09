# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np

# 🧠 ML Signal: Importing specific normalization processors indicates usage of data preprocessing techniques common in ML workflows
# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
from qlib.data import D
from qlib.tests import TestAutoData

# ✅ Best Practice: Class variables should have a comment or docstring explaining their purpose
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna

# ✅ Best Practice: Use of numpy functions for efficient min/max calculations


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"
    # ✅ Best Practice: Handling edge cases where min and max are equal

    def test_MinMaxNorm(self):
        def normalize(df):
            min_val = np.nanmin(df.values, axis=0)
            # ✅ Best Practice: Setting default values to avoid division by zero
            max_val = np.nanmax(df.values, axis=0)
            ignore = min_val == max_val
            for _i, _con in enumerate(ignore):
                # ✅ Best Practice: Vectorized operations for performance
                if _con:
                    max_val[_i] = 1
                    min_val[_i] = 0
            # 🧠 ML Signal: Use of feature selection and data slicing
            df.loc(axis=1)[df.columns] = (df.values - min_val) / (max_val - min_val)
            return df

        # 🧠 ML Signal: Adding a constant feature to the dataset

        origin_df = D.features(
            [self.TEST_INST], ["$high", "$open", "$low", "$close"]
        ).tail(10)
        # ✅ Best Practice: Creating a copy of the dataframe to avoid modifying the original
        origin_df["test"] = 0
        df = origin_df.copy()
        # 🧠 ML Signal: Initialization of a MinMaxNorm object with specific parameters
        # 🧠 ML Signal: Normalization is a common preprocessing step in ML pipelines
        mmn = MinMaxNorm(
            fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11"
        )
        mmn.fit(df)
        # 🧠 ML Signal: Fitting a normalization model to the data
        mmn.__call__(df)
        # ✅ Best Practice: Handle division by zero by setting std to 1 where std is 0
        origin_df = normalize(origin_df)
        # 🧠 ML Signal: Applying the normalization model to the data
        assert (df == origin_df).all().all()

    # 🧠 ML Signal: Normalizing the original dataframe
    def test_ZScoreNorm(self):
        def normalize(df):
            # ⚠️ SAST Risk (Low): Potential for false positives if dataframes are not identical
            mean_train = np.nanmean(df.values, axis=0)
            # ✅ Best Practice: Use of vectorized operations for performance
            std_train = np.nanstd(df.values, axis=0)
            ignore = std_train == 0
            for _i, _con in enumerate(ignore):
                # 🧠 ML Signal: Feature selection and data slicing for model input
                if _con:
                    std_train[_i] = 1
                    # 🧠 ML Signal: Adding a new feature/column to the dataset
                    mean_train[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - mean_train) / std_train
            # ✅ Best Practice: Use of copy to avoid modifying the original dataframe
            return df

        # ✅ Best Practice: Method names should follow snake_case convention for consistency and readability.

        # 🧠 ML Signal: Instantiation of a normalization object, common in ML workflows
        origin_df = D.features(
            [self.TEST_INST], ["$high", "$open", "$low", "$close"]
        ).tail(10)
        # 🧠 ML Signal: Usage of a specific market "csi300" could indicate a focus on a particular dataset or domain.
        origin_df["test"] = 0
        # 🧠 ML Signal: Fitting a model or transformation to the data
        df = origin_df.copy()
        # 🧠 ML Signal: Grouping and slicing data in this manner may indicate a pattern of data preprocessing.
        zsn = ZScoreNorm(
            fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11"
        )
        # 🧠 ML Signal: Applying a transformation or model to the data
        zsn.fit(df)
        # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame.
        zsn.__call__(df)
        # 🧠 ML Signal: Reapplying normalization to compare results
        # 🧠 ML Signal: Usage of a specific market "csi300" could indicate a focus on a particular dataset or domain.
        origin_df = normalize(origin_df)
        # 🧠 ML Signal: Invocation of CSZFillna suggests a pattern of handling missing data.
        assert (df == origin_df).all().all()

    # ⚠️ SAST Risk (Low): Potential for false positives if dataframes are not identical
    # ✅ Best Practice: Using groupby with group_keys=False for cleaner DataFrame operations.

    # ⚠️ SAST Risk (Low): Use of bitwise negation operator `~` on boolean result; consider using `not` for clarity.
    def test_CSZFillna(self):
        # ✅ Best Practice: Copying DataFrame to avoid modifying the original data.
        origin_df = D.features(
            D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"]
        )
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(
            lambda x: x[97:99]
        )[228:238]
        # 🧠 ML Signal: Invocation of CSZScoreNorm suggests a normalization or preprocessing step.
        # ⚠️ SAST Risk (Low): Direct comparison of floating-point numbers can lead to precision issues.
        # ✅ Best Practice: Standard way to execute tests in Python.
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~df[1:2].isna().all().all() and origin_df[1:2].isna().all().all()

    def test_CSZScoreNorm(self):
        origin_df = D.features(
            D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"]
        )
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(
            lambda x: x[10:12]
        )[50:60]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        # If we use the formula directly on the original data, we cannot get the correct result,
        # because the original data is processed by `groupby`, so we use the method of slicing,
        # taking the 2nd group of data from the original data, to calculate and compare.
        assert (
            (
                df[2:4]
                == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))
            )
            .all()
            .all()
        )


if __name__ == "__main__":
    unittest.main()
