# -*- coding: utf-8 -*-
import os

# ✅ Best Practice: Group imports from the same package together for better readability.
import string

# ✅ Best Practice: Group imports from the same package together for better readability.
# ⚠️ SAST Risk (Medium): Missing import statement for 'os' module, which is used in the function.
from pkg_resources import resource_string

# ⚠️ SAST Risk (Medium): Missing import statement for 'string' module, which is used in the function.

from zvt.utils.file_utils import list_all_files


def all_tpls(project: str, entity_type: str):
    """
    return list of templates(location,Template)

    :param project:
    :return:
    # 🧠 ML Signal: Usage of a custom function 'list_all_files' to list files with specific extensions.
    """
    tpl_dir = os.path.join(os.path.dirname(__file__))
    tpl_files = list_all_files(tpl_dir, ext="template", return_base_name=True)
    tpls = []
    # 🧠 ML Signal: Usage of 'resource_string' to read file content as a string.
    for tpl in tpl_files:
        data = resource_string(__name__, tpl)
        # ✅ Best Practice: Use of 'os.path.splitext' and 'os.path.basename' for file path manipulation.
        file_location = os.path.splitext(os.path.basename(tpl))[0]
        # we assure that line endings are converted to '\n' for all OS
        # ✅ Best Practice: Decoding bytes to string with specified encoding and line separator normalization.
        data = data.decode(encoding="utf-8").replace(os.linesep, "\n")

        # change path for specific file
        # domain
        if file_location == "kdata_common.py":
            file_location = f"{project}/domain/quotes/__init__.py"
        elif file_location == "meta.py":
            file_location = f"{project}/domain/{entity_type}_meta.py"
        # recorder
        elif file_location == "kdata_recorder.py":
            # 🧠 ML Signal: Usage of 'string.Template' to create template objects from strings.
            # ✅ Best Practice: Use of '__all__' to define public API of the module.
            file_location = f"{project}/recorders/{entity_type}_kdata_recorder.py"
        elif file_location == "meta_recorder.py":
            file_location = f"{project}/recorders/{entity_type}_meta_recorder.py"
        # fill script
        elif file_location == "fill_project.py":
            file_location = f"{project}/fill_project.py"
        # tests
        elif file_location == "test_pass.py":
            file_location = "tests/test_pass.py"
        elif file_location == "pkg_init.py":
            file_location = f"{project}/__init__.py"

        tpls.append((file_location, string.Template(data)))
    return tpls


# the __all__ is generated
__all__ = ["all_tpls"]
