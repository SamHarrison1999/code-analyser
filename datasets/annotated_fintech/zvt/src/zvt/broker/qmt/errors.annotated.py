# ✅ Best Practice: Custom exception class for module-specific errors
# -*- coding: utf-8 -*-
class TraderError(Exception):
    """Base class for exceptions in this module."""
    # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage

    # ✅ Best Practice: Provide a default value for the parameter to ensure the function can be called without arguments.
    pass

# 🧠 ML Signal: Storing a parameter as an instance attribute is a common pattern.
# ✅ Best Practice: Custom exception class for specific error handling

# ✅ Best Practice: Provide a default value for the message parameter to ensure consistent behavior
class QmtError(TraderError):
    def __init__(self, message="qmt error"):
        # 🧠 ML Signal: Storing a message in an instance variable is a common pattern for exception handling
        # 🧠 ML Signal: Use of __all__ to define public API of the module
        self.message = message


class PositionOverflowError(TraderError):
    def __init__(self, message="超出仓位限制"):
        self.message = message


# the __all__ is generated
__all__ = ["TraderError", "QmtError", "PositionOverflowError"]