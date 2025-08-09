# -*- coding: utf-8 -*-
# ✅ Best Practice: Import only necessary modules to keep the codebase clean and efficient.
import argparse

# ✅ Best Practice: Define the main function to encapsulate script execution logic.
from zvt.autocode import gen_exports

# 🧠 ML Signal: Importing specific functions or classes can indicate which parts of a library are most frequently used.
from zvt.autocode.generator import gen_plugin_project

# ✅ Best Practice: Use argparse for command-line argument parsing.


# ✅ Best Practice: Provide help messages for command-line arguments for better usability.
# 🧠 ML Signal: Importing specific functions or classes can indicate which parts of a library are most frequently used.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", help="entity name", default="future")
    parser.add_argument("--prefix", help="project prefix", default="zvt")
    # ✅ Best Practice: Use nargs="+" to allow multiple values for a command-line argument.
    parser.add_argument("--dir", help="project directory", default=".")
    parser.add_argument(
        "--providers", help="providers", default=["joinquant"], nargs="+"
    )
    # 🧠 ML Signal: Usage of argparse to parse command-line arguments.

    args = parser.parse_args()
    # 🧠 ML Signal: Accessing parsed command-line arguments.
    # ✅ Best Practice: Define all imports at the beginning of the file for better readability and maintainability

    dir_path = args.dir
    # ✅ Best Practice: Use argparse for command-line argument parsing to improve usability and flexibility
    entity = args.entity
    providers = args.providers
    # ✅ Best Practice: Provide help messages for command-line arguments to improve user experience
    prefix = args.prefix
    # 🧠 ML Signal: Function call with multiple parameters derived from user input.
    gen_plugin_project(
        prefix=prefix, dir_path=dir_path, entity_type=entity, providers=providers
    )


# 🧠 ML Signal: Command-line argument parsing is a common pattern in CLI applications


# 🧠 ML Signal: Using parsed command-line arguments to set variables is a common pattern
# 🧠 ML Signal: Function calls with keyword arguments are a common pattern
# ✅ Best Practice: Use the standard Python idiom for making a script executable
# ⚠️ SAST Risk (Medium): Hardcoded directory paths can lead to security issues if not handled properly
# ⚠️ SAST Risk (Low): Ensure that the main function is defined before calling it
def export():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="export directory", default=".")
    args = parser.parse_args()
    dir_path = args.dir
    gen_exports(dir_path=dir_path)


if __name__ == "__main__":
    gen_plugin_project(dir_path="../../../", entity_type="macro", providers=["zvt"])
    main()
