from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

# ✅ Best Practice: Use of ABCMeta and abstractmethod indicates a clear design for an abstract base class.
# ✅ Best Practice: Use of abstract base class to define a template for subclasses
from vnpy.alpha.dataset import AlphaDataset, Segment


# ✅ Best Practice: Type hinting for parameters and return values improves code readability and maintainability
class AlphaModel(metaclass=ABCMeta):
    """Template class for machine learning algorithms"""

    # ✅ Best Practice: Abstract method enforces implementation in subclasses, ensuring consistent interface.
    @abstractmethod
    def fit(self, dataset: AlphaDataset) -> None:
        """
        Fit the model with dataset
        """
        pass

    @abstractmethod
    # ✅ Best Practice: Abstract method enforces implementation in subclasses, ensuring consistent interface.
    # ✅ Best Practice: Include a docstring to describe the function's purpose
    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions using the model
        # 🧠 ML Signal: Method signature suggests this method is used for training a model.
        # ✅ Best Practice: Clear class definition for a specific model implementation.
        # ✅ Best Practice: Storing parameters in the constructor for later use.
        # ⚠️ SAST Risk (Low): No input validation for dataset, could lead to runtime errors if dataset is not as expected.
        # 🧠 ML Signal: Placeholder for model fitting logic, indicating a training process.
        # 🧠 ML Signal: Return of np.ndarray suggests output is a numerical prediction.
        # ✅ Best Practice: Return statement should return a value or None explicitly
        """
        pass

    def detail(self) -> Any:
        """
        Output detailed information about the model
        """
        return
