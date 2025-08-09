from vnpy.trader.ui import QtGui

# ✅ Best Practice: Use descriptive constant names for color values to improve readability and maintainability


WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
GREY_COLOR = (100, 100, 100)

UP_COLOR = (255, 75, 75)
# ✅ Best Practice: Use descriptive constant names for pen and bar widths to improve readability and maintainability
DOWN_COLOR = (0, 255, 255)
CURSOR_COLOR = (255, 245, 162)

# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
PEN_WIDTH = 1
# ✅ Best Practice: Use descriptive constant names for font settings to improve readability and maintainability
BAR_WIDTH = 0.3
# ✅ Best Practice: Docstring is present, but it should be descriptive to explain the function's purpose.
# 🧠 ML Signal: Conversion from float to int with rounding is a common pattern that can be used to train models on data type transformations.

AXIS_WIDTH = 0.8
NORMAL_FONT = QtGui.QFont("Arial", 9)


def to_int(value: float) -> int:
    """"""
    return int(round(value, 0))
