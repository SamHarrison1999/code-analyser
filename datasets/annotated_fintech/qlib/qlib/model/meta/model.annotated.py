# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Use of relative imports can improve module portability within a package

import abc
from typing import List

from .dataset import MetaTaskDataset


class MetaModel(metaclass=abc.ABCMeta):
    """
    The meta-model guiding the model learning.

    The word `Guiding` can be categorized into two types based on the stage of model learning
    - The definition of learning tasks:  Please refer to docs of `MetaTaskModel`
    - Controlling the learning process of models: Please refer to the docs of `MetaGuideModel`
    """
    # ✅ Best Practice: Use of abstractmethod enforces implementation in subclasses, ensuring consistent interface
    # 🧠 ML Signal: Use of *args and **kwargs indicates flexibility in input parameters

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        The training process of the meta-model.
        """

    @abc.abstractmethod
    # ✅ Best Practice: Class docstring provides a clear explanation of the class purpose and behavior.
    def inference(self, *args, **kwargs) -> object:
        """
        The inference process of the meta-model.

        Returns
        -------
        object:
            Some information to guide the model learning
        """
# ✅ Best Practice: Use docstrings to describe the purpose and behavior of methods

# ⚠️ SAST Risk (Low): Using NotImplementedError without implementation can lead to runtime errors if not handled
# ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and return type.

class MetaTaskModel(MetaModel):
    """
    This type of meta-model deals with base task definitions. The meta-model creates tasks for training new base forecasting models after it is trained. `prepare_tasks` directly modifies the task definitions.
    """

    def fit(self, meta_dataset: MetaTaskDataset):
        """
        The MetaTaskModel is expected to get prepared MetaTask from meta_dataset.
        And then it will learn knowledge from the meta tasks
        # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called.
        """
        raise NotImplementedError(f"Please implement the `fit` method")

    def inference(self, meta_dataset: MetaTaskDataset) -> List[dict]:
        """
        MetaTaskModel will make inference on the meta_dataset
        The MetaTaskModel is expected to get prepared MetaTask from meta_dataset.
        Then it will create modified task with Qlib format which can be executed by Qlib trainer.

        Returns
        -------
        List[dict]:
            A list of modified task definitions.

        """
        raise NotImplementedError(f"Please implement the `inference` method")


class MetaGuideModel(MetaModel):
    """
    This type of meta-model aims to guide the training process of the base model. The meta-model interacts with the base forecasting models during their training process.
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        pass