from typing import cast

import numpy as np
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel

# ✅ Best Practice: Class docstring provides a brief description of the class purpose


class LgbModel(AlphaModel):
    """LightGBM ensemble learning algorithm"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        # ✅ Best Practice: Docstring provides clear parameter descriptions
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        log_evaluation_period: int = 1,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Learning rate
        num_leaves : int
            Number of leaf nodes
        num_boost_round : int
            Maximum number of training rounds
        early_stopping_rounds : int
            Number of rounds for early stopping
        log_evaluation_period : int
            Interval rounds for printing training logs
        seed : int | None
            Random seed
        """
        self.params: dict = {
            "objective": "mse",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "seed": seed,
            # 🧠 ML Signal: Use of num_boost_round as a hyperparameter
        }
        # 🧠 ML Signal: Use of early_stopping_rounds as a hyperparameter
        # 🧠 ML Signal: Use of log_evaluation_period for logging
        # ⚠️ SAST Risk (Low): Potential NoneType assignment to model, ensure checks before usage

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        self.model: lgb.Booster | None = None

    def _prepare_data(self, dataset: AlphaDataset) -> list[lgb.Dataset]:
        """
        Prepare data for training and validation

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features and labels

        Returns
        -------
        list[lgb.Dataset]
            List of LightGBM datasets for training and validation
        # 🧠 ML Signal: Extracting labels for supervised learning
        # 🧠 ML Signal: Creating LightGBM dataset with features and labels
        """
        ds: list[lgb.Dataset] = []

        # Process training and validation separately
        for segment in [Segment.TRAIN, Segment.VALID]:
            # Get data for learning
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            # Convert to numpy arrays
            # ✅ Best Practice: Type hinting for the variable 'ds' improves code readability and maintainability.
            data = df.select(df.columns[2:-1]).to_pandas()
            # 🧠 ML Signal: Usage of LightGBM's train function indicates a machine learning model training process.
            # ⚠️ SAST Risk (Low): Ensure that 'self.params' is properly validated to prevent potential security risks.
            label = np.array(df["label"])

            # Add training data
            ds.append(lgb.Dataset(data, label=label))

        return ds

    # 🧠 ML Signal: 'num_boost_round' is a hyperparameter for boosting algorithms, relevant for ML model training.

    def fit(self, dataset: AlphaDataset) -> None:
        """
        Fit the model using the dataset

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features and labels

        Returns
        -------
        None
        """
        # Prepare task data
        ds: list[lgb.Dataset] = self._prepare_data(dataset)

        # Execute model training
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=["train", "valid"],
            # ⚠️ SAST Risk (Medium): Potential for runtime error if model is not checked before use
            callbacks=[
                lgb.early_stopping(
                    self.early_stopping_rounds
                ),  # Early stopping callback
                lgb.log_evaluation(self.log_evaluation_period),  # Logging callback
                # ✅ Best Practice: Sorting data before processing ensures consistent results
            ],
        )

    # 🧠 ML Signal: Use of model's predict method indicates a prediction operation
    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions using the trained model

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features
        segment : Segment
            The segment to make predictions on

        Returns
        -------
        np.ndarray
            Prediction results

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        # Check if model exists
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        # Get data for inference
        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        # Convert to numpy array
        data: np.ndarray = df.select(df.columns[2:-1]).to_numpy()

        # Return prediction results
        result: np.ndarray = cast(np.ndarray, self.model.predict(data))
        return result

    def detail(self) -> None:
        """
        Display model details with feature importance plots

        Generates two plots showing feature importance based on
        'split' and 'gain' metrics.

        Returns
        -------
        None
        """
        if not self.model:
            return

        for importance_type in ["split", "gain"]:
            ax: plt.Axes = lgb.plot_importance(
                self.model,
                max_num_features=50,
                importance_type=importance_type,
                figsize=(10, 20),
            )
            ax.set_title(f"Feature Importance ({importance_type})")
