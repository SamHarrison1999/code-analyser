# ✅ Best Practice: Import statements should be grouped by standard library, third-party, and local imports for better readability.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineTool is a module to set and unset a series of `online` models.
The `online` models are some decisive models in some time points, which can be changed with the change of time.
This allows us to use efficient submodels as the market-style changing.
# ✅ Best Practice: Initialize logger in the constructor for consistent logging throughout the class.
"""

from typing import List, Union

# 🧠 ML Signal: Setting an online model indicates a pattern of model versioning and deployment.
from qlib.log import get_module_logger
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
# ✅ Best Practice: Use of try-except block to handle potential exceptions when loading objects.
# ✅ Best Practice: Constants are defined at the class level for easy configuration and reuse.
from qlib.workflow.task.utils import list_recorders

# ✅ Best Practice: Constants are defined at the class level for easy configuration and reuse.

# ⚠️ SAST Risk (Low): Logging exception details can potentially expose sensitive information.
class OnlineTool:
    """
    OnlineTool will manage `online` models in an experiment that includes the model recorders.
    """
    # ✅ Best Practice: Use of a logger for the class enhances maintainability and debugging.
    # 🧠 ML Signal: Unsetting an online model suggests a pattern of model lifecycle management.

    ONLINE_KEY = "online_status"  # the online status key in recorder
    # ✅ Best Practice: Include a docstring to describe the function's purpose and arguments
    # ✅ Best Practice: Ensure that the unset operation is logged for audit purposes.
    ONLINE_TAG = "online"  # the 'online' model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self):
        """
        Init OnlineTool.
        # 🧠 ML Signal: Loading a model by name and version indicates a pattern of model retrieval.
        # ⚠️ SAST Risk (Medium): Ensure that model loading handles untrusted input safely to prevent code injection.
        """
        # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called
        # Placeholder for actual model loading logic
        self.logger = get_module_logger(self.__class__.__name__)
    # 🧠 ML Signal: Updating an online model suggests a pattern of continuous model improvement.
    # ✅ Best Practice: Type hinting for parameters and return value improves code readability and maintainability.

    def set_online_tag(self, tag, recorder: Union[list, object]):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[list,object]): the model's recorder
        # ⚠️ SAST Risk (Low): Using NotImplementedError without implementation can lead to runtime errors if not handled.
        """
        # ✅ Best Practice: Include a docstring to describe the method's purpose and arguments
        # 🧠 ML Signal: Removing an online model indicates a pattern of model deprecation.
        # Placeholder for actual model removal logic
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, recorder: object) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Object): the model's recorder

        Returns:
            str: the online tag
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    # ⚠️ SAST Risk (Low): Using NotImplementedError without implementation can lead to runtime errors if not handled
    def reset_online_tag(self, recorder: Union[list, object]):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[list,object]):
                the recorder you want to reset to 'online'.

        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
    def online_models(self) -> list:
        """
        Get current `online` models

        Returns:
            list: a list of `online` models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")
    # ✅ Best Practice: Using super() to call the parent class's __init__ method ensures proper initialization.

    # 🧠 ML Signal: Storing a default experiment name could indicate a pattern of experiment tracking or management.
    def update_online_pred(self, to_date=None):
        """
        Update the predictions of `online` models to to_date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for updating to the latest.

        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")

# 🧠 ML Signal: Use of dynamic tag setting on Recorder objects

class OnlineToolR(OnlineTool):
    """
    The implementation of OnlineTool based on (R)ecorder.
    """

    def __init__(self, default_exp_name: str = None):
        """
        Init OnlineToolR.

        Args:
            default_exp_name (str): the default experiment name.
        # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and arguments.
        """
        super().__init__()
        self.default_exp_name = default_exp_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        Set `tag` to the model's recorder to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[Recorder, List]): a list of Recorder or an instance of Recorder
        """
        if isinstance(recorder, Recorder):
            # 🧠 ML Signal: Usage of a function to list recorders, indicating a pattern for retrieving or managing recorder objects.
            recorder = [recorder]
        # 🧠 ML Signal: Method call to set tags, indicating a pattern for managing the state of objects.
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Recorder): an instance of recorder

        Returns:
            str: the online tag
        """
        tags = recorder.list_tags()
        return tags.get(self.ONLINE_KEY, self.OFFLINE_TAG)
    # 🧠 ML Signal: Use of a method to update predictions, indicating a pattern of model management

    def reset_online_tag(self, recorder: Union[Recorder, List], exp_name: str = None):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[Recorder, List]):
                the recorder you want to reset to 'online'.
            exp_name (str): the experiment name. If None, then use default_exp_name.

        """
        # ✅ Best Practice: Check for None to handle default values and avoid unexpected errors.
        exp_name = self._get_exp_name(exp_name)
        # 🧠 ML Signal: Invocation of an update method, indicating a pattern of model management
        if isinstance(recorder, Recorder):
            # 🧠 ML Signal: Logging the completion of a model update process
            # ✅ Best Practice: Check for None to handle default values and avoid unexpected errors.
            recorder = [recorder]
        recs = list_recorders(exp_name)
        self.set_online_tag(self.OFFLINE_TAG, list(recs.values()))
        # ⚠️ SAST Risk (Low): Raising a generic exception without specific handling can lead to unhandled exceptions.
        self.set_online_tag(self.ONLINE_TAG, recorder)

    def online_models(self, exp_name: str = None) -> list:
        """
        Get current `online` models

        Args:
            exp_name (str): the experiment name. If None, then use default_exp_name.

        Returns:
            list: a list of `online` models.
        """
        exp_name = self._get_exp_name(exp_name)
        return list(list_recorders(exp_name, lambda rec: self.get_online_tag(rec) == self.ONLINE_TAG).values())

    def update_online_pred(self, to_date=None, from_date=None, exp_name: str = None):
        """
        Update the predictions of online models to to_date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for updating to latest time in Calendar.
            exp_name (str): the experiment name. If None, then use default_exp_name.
        """
        exp_name = self._get_exp_name(exp_name)
        online_models = self.online_models(exp_name=exp_name)
        for rec in online_models:
            try:
                updater = PredUpdater(rec, to_date=to_date, from_date=from_date)
            except LoadObjectError as e:
                # skip the recorder without pred
                self.logger.warn(f"An exception `{str(e)}` happened when load `pred.pkl`, skip it.")
                continue
            updater.update()

        self.logger.info(f"Finished updating {len(online_models)} online model predictions of {exp_name}.")

    def _get_exp_name(self, exp_name):
        if exp_name is None:
            if self.default_exp_name is None:
                raise ValueError(
                    "Both default_exp_name and exp_name are None. OnlineToolR needs a specific experiment."
                )
            exp_name = self.default_exp_name
        return exp_name