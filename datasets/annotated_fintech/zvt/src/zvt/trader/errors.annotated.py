# ✅ Best Practice: Custom exception class for module-specific errors
# -*- coding: utf-8 -*-
class TraderError(Exception):
    """Base class for exceptions in this module."""
    # ✅ Best Practice: Define custom exception classes for specific error handling

    # ✅ Best Practice: Default parameter values improve function usability and flexibility
    pass

# ✅ Best Practice: Custom exception class for specific error handling
# 🧠 ML Signal: Storing error messages in instance variables is a common pattern

# ✅ Best Practice: Use of default parameter value for flexibility
class InvalidOrderParamError(TraderError):
    def __init__(self, message="invalid order param"):
        # 🧠 ML Signal: Storing a parameter value in an instance variable
        self.message = message
# ✅ Best Practice: Define a custom exception class for specific error handling

# ✅ Best Practice: Use of default parameter value for flexibility and ease of use

# ✅ Best Practice: Custom exception class for specific error handling
class NotEnoughMoneyError(TraderError):
    # 🧠 ML Signal: Storing a parameter as an instance attribute, common pattern in class initializers
    # ✅ Best Practice: Provide a default value for the parameter to ensure the function can be called without arguments.
    def __init__(self, message="not enough money"):
        self.message = message
# 🧠 ML Signal: Storing a parameter value in an instance variable is a common pattern.

# ✅ Best Practice: Default argument values should be immutable to avoid unexpected behavior.

class NotEnoughPositionError(TraderError):
    # ✅ Best Practice: Storing the message in an instance variable for later use.
    # ✅ Best Practice: Using __all__ to define the public API of the module.
    def __init__(self, message="not enough position"):
        self.message = message


class InvalidOrderError(TraderError):
    def __init__(self, message="invalid order"):
        self.message = message


class WrongKdataError(TraderError):
    def __init__(self, message="wrong kdata"):
        self.message = message


# the __all__ is generated
__all__ = [
    "TraderError",
    "InvalidOrderParamError",
    "NotEnoughMoneyError",
    "NotEnoughPositionError",
    "InvalidOrderError",
    "WrongKdataError",
]