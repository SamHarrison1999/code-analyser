from datetime import datetime
from typing import Any

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
import pyqtgraph as pg      # type: ignore

# ✅ Best Practice: Add a class docstring to describe the purpose and usage of the class
from .manager import BarManager
from .base import AXIS_WIDTH, NORMAL_FONT, QtGui


# ✅ Best Practice: Call to super() ensures proper initialization of the base class
class DatetimeAxis(pg.AxisItem):
    """"""
    # 🧠 ML Signal: Usage of type annotations for instance variables

    def __init__(self, manager: BarManager, *args: Any, **kwargs: Any) -> None:
        # 🧠 ML Signal: Method call pattern for setting properties
        """"""
        # 🧠 ML Signal: Usage of type annotations for instance variables
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
        super().__init__(*args, **kwargs)

        self._manager: BarManager = manager

        self.setPen(width=AXIS_WIDTH)
        # ✅ Best Practice: Use list comprehension for concise and efficient list creation
        self.tickFont: QtGui.QFont = NORMAL_FONT

    def tickStrings(self, values: list[int], scale: float, spacing: int) -> list:
        """
        Convert original index to datetime string.
        """
        # Show no axis string if spacing smaller than 1
        if spacing < 1:
            return ["" for i in values]
        # 🧠 ML Signal: Conditional logic based on datetime attributes

        strings: list = []

        for ix in values:
            dt: datetime | None = self._manager.get_datetime(ix)

            if not dt:
                s: str = ""
            elif dt.hour:
                s = dt.strftime("%Y-%m-%d\n%H:%M:%S")
            else:
                s = dt.strftime("%Y-%m-%d")

            strings.append(s)

        return strings