# ✅ Best Practice: Custom exception class for specific error handling
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Class docstring provides a clear description of the class purpose


# Base exception class
# ✅ Best Practice: Include a docstring to describe the purpose of the class
class QlibException(Exception):
    # ✅ Best Practice: Custom exception class for specific error handling
    pass


class RecorderInitializationError(QlibException):
    """Error type for re-initialization when starting an experiment"""


class LoadObjectError(QlibException):
    """Error type for Recorder when can not load object"""


class ExpAlreadyExistError(Exception):
    """Experiment already exists"""
