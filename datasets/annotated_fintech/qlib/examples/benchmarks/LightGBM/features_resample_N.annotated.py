#  Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing pandas, a common library for data manipulation, indicating data processing tasks
#  Licensed under the MIT License.

# 🧠 ML Signal: Importing InstProcessor, suggesting usage of qlib for financial data processing
# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
import pandas as pd

# ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
# 🧠 ML Signal: Importing resam_calendar, indicating potential use of calendar resampling in data processing
from qlib.data.inst_processor import InstProcessor
from qlib.utils.resam import resam_calendar

# 🧠 ML Signal: Storing method parameters as instance variables is a common pattern.
# ✅ Best Practice: Ensure the DataFrame index is in datetime format for time series operations


# ⚠️ SAST Risk (Low): Potential risk if resam_calendar is not properly validated or sanitized
class ResampleNProcessor(InstProcessor):
    # ✅ Best Practice: Use resample and reindex for consistent time series data manipulation
    # 🧠 ML Signal: Returns a DataFrame, indicating a transformation or processing step
    def __init__(self, target_frq: str, **kwargs):
        self.target_frq = target_frq

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        df.index = pd.to_datetime(df.index)
        res_index = resam_calendar(df.index, "1min", self.target_frq)
        df = df.resample(self.target_frq).last().reindex(res_index)
        return df
