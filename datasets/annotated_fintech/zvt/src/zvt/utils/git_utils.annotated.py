# -*- coding: utf-8 -*-
# ⚠️ SAST Risk (High): Using subprocess can lead to command injection vulnerabilities if inputs are not properly sanitized.
# ⚠️ SAST Risk (Medium): subprocess.check_output can execute arbitrary commands if input is not sanitized
import subprocess

# ✅ Best Practice: Import the subprocess module explicitly at the beginning of the file

# 🧠 ML Signal: Usage of subprocess to execute shell commands


def get_git_user_name():
    try:
        # ✅ Best Practice: Specify the exception type to catch specific errors
        # ⚠️ SAST Risk (Medium): subprocess.check_output can be dangerous if input is not controlled
        return (
            subprocess.check_output(["git", "config", "--get", "user.name"])
            .decode("utf8")
            .strip()
        )
    # ✅ Best Practice: Specify the exception type to avoid catching unexpected exceptions
    except:
        # ✅ Best Practice: Avoid using bare except, specify the exception type
        # ✅ Best Practice: Log the exception or provide more context when returning an empty string
        return "foolcage"


# ✅ Best Practice: Ensure that all functions listed in __all__ are defined in the module
def get_git_user_email():
    try:
        return (
            subprocess.check_output(["git", "config", "--get", "user.email"])
            .decode("utf8")
            .strip()
        )
    except:
        return ""


# the __all__ is generated
__all__ = ["get_git_user_name", "get_git_user_email"]
