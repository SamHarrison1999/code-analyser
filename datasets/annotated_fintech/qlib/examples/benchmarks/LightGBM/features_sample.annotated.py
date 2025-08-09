import datetime
import pandas as pd

# ✅ Best Practice: Import only necessary components to reduce memory usage and improve readability

from qlib.data.inst_processor import InstProcessor

# ✅ Best Practice: Class docstring provides a clear description of the class functionality

# ✅ Best Practice: Use of type annotations for function parameters improves code readability and maintainability


class Resample1minProcessor(InstProcessor):
    # 🧠 ML Signal: Storing parameters as instance attributes is a common pattern
    """This processor tries to resample the data. It will reasmple the data from 1min freq to day freq by selecting a specific miniute"""

    # 🧠 ML Signal: Storing parameters as instance attributes is a common pattern
    # 🧠 ML Signal: Use of pandas DataFrame, common in data processing tasks
    def __init__(self, hour: int, minute: int, **kwargs):
        # ✅ Best Practice: Type hint for df parameter improves code readability and maintainability
        self.hour = hour
        self.minute = minute

    # 🧠 ML Signal: Filtering DataFrame based on specific time, common in time-series data processing
    # ⚠️ SAST Risk (Low): Potential for timezone-related issues when converting to datetime
    # 🧠 ML Signal: Normalizing datetime index, typical in time-series data preprocessing

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index.time == datetime.time(self.hour, self.minute)]
        df.index = df.index.normalize()
        return df
