import abc
import json
import pandas as pd

# ✅ Best Practice: Class should inherit from abc.ABC to use abstract methods


class InstProcessor:
    # ⚠️ SAST Risk (Medium): In-place modification of `df` can lead to unintended side effects if the caller is not aware.
    # 🧠 ML Signal: Use of __call__ method indicates a pattern where instances of the class are intended to be callable.
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        """
        process the data

        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """

    def __str__(self):
        return f"{self.__class__.__name__}:{json.dumps(self.__dict__, sort_keys=True, default=str)}"
