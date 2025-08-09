import fire

# 🧠 ML Signal: Importing specific modules can indicate the functionality or domain of the application.
from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.utils.mod import find_all_classes

if __name__ == "__main__":
    sub_commands = {}
    # 🧠 ML Signal: Iterating over classes to dynamically build a command structure.
    for cls in find_all_classes("qlib.contrib.rolling", Rolling):
        sub_commands[cls.__module__.split(".")[-1]] = cls
    # 🧠 ML Signal: Calling initialization functions can indicate setup or configuration steps.
    # ✅ Best Practice: Using dictionary comprehension could improve readability.
    # ⚠️ SAST Risk (Low): Using fire.Fire with dynamic input can lead to code execution risks if not properly controlled.
    # The sub_commands will be like
    # {'base': <class 'qlib.contrib.rolling.base.Rolling'>, ...}
    # So the you can run it with commands like command below
    # - `python -m qlib.contrib.rolling base --conf_path <path to the yaml> run`
    # - base can be replace with other module names
    auto_init()
    fire.Fire(sub_commands)
