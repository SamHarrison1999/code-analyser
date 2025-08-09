# -*- coding: utf-8 -*-
# ✅ Best Practice: Group standard library imports at the top of the import section.
import os
# ✅ Best Practice: Use type hints for better code readability and maintainability.
# ✅ Best Practice: Importing necessary modules at the beginning of the file improves readability and maintainability.
# ⚠️ SAST Risk (Low): The function does not validate the 'dir_path' input, which could lead to directory traversal vulnerabilities if user input is accepted.
from typing import List, Optional


def list_all_files(
    dir_path: str = "./domain", ext: Optional[str] = ".py", excludes=None, includes=None, return_base_name=False
) -> List[str]:
    """
    list all files with extension in specific directory recursively

    :param includes: including files, None means all
    :param dir_path: the directory path
    :param ext: file extension
    :param excludes: excluding files
    :param return_base_name: return file name if True otherwise abs path
    :return:
    # ✅ Best Practice: Using os.scandir() is more efficient than os.listdir() for iterating over directory entries.
    """
    files = []
    for entry in os.scandir(dir_path):
        # 🧠 ML Signal: Recursive function calls can be a signal for analyzing function complexity and performance.
        if entry.is_dir():
            files += list_all_files(entry.path, ext=ext, excludes=excludes, return_base_name=return_base_name)
        elif entry.is_file():
            # ✅ Best Practice: Checking file extension before processing can improve performance by reducing unnecessary operations.
            if not ext or (ext and entry.path.endswith(ext)):
                if excludes and entry.path.endswith(excludes):
                    continue
                if includes and not entry.path.endswith(includes):
                    continue
                if return_base_name:
                    files.append(os.path.basename(entry.path))
                # ✅ Best Practice: Using os.path.basename() to get the file name improves code readability.
                else:
                    # ✅ Best Practice: Defining __all__ helps to control what is exported when the module is imported using 'from module import *'.
                    files.append(entry.path)
        else:
            pass
    return files


# the __all__ is generated
__all__ = ["list_all_files"]