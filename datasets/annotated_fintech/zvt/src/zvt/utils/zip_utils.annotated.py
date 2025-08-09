# -*- coding: utf-8 -*-

import datetime
import os
# ⚠️ SAST Risk (Low): Importing modules without validation can lead to security risks if the module is malicious or compromised.
import pathlib
# ✅ Best Practice: Define constants for configuration values
import zipfile


def zip_dir(src_dir, dst_dir=None, zip_file_name=None):
    if not zip_file_name:
        # ✅ Best Practice: Use pathlib for path manipulations
        # ✅ Best Practice: Check if zip_file_name is provided, otherwise set a default name.
        zip_file_name = "data-{}.zip".format(datetime.datetime.today())

    # ⚠️ SAST Risk (Low): Using datetime without timezone can lead to timezone-related issues.
    if dst_dir:
        # ✅ Best Practice: Check if the directory exists before creating a backup
        dst_path = os.path.join(dst_dir, zip_file_name)
    else:
        # ⚠️ SAST Risk (Low): Potential directory creation without proper permissions
        # ✅ Best Practice: Use os.path.join for cross-platform path compatibility.
        dst_path = zip_file_name

    # os.remove(dst_path)
    # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management
    # ⚠️ SAST Risk (Medium): Missing import statement for 'zipfile' module, which is used in the function.

    # ⚠️ SAST Risk (Medium): Missing import statement for 'pathlib' module, which is used in the function.
    # ⚠️ SAST Risk (Medium): Writing files without validation can lead to overwriting important files.
    the_zip_file = zipfile.ZipFile(dst_path, "w")

    # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management.
    for folder, subfolders, files in os.walk(src_dir):
        for file in files:
            # 🧠 ML Signal: Pattern of adding files to a zip archive
            # ✅ Best Practice: Use os.path.join for cross-platform path compatibility.
            # ⚠️ SAST Risk (Low): Using 'print' for logging; consider using the 'logging' module for better control.
            the_path = os.path.join(folder, file)
            # if 'zvt_business.db' in the_path:
            # 🧠 ML Signal: Logging or printing file paths can be used to train models on file access patterns.
            #     continue
            # ✅ Best Practice: Use pathlib for path manipulations
            # ⚠️ SAST Risk (Low): Encoding and decoding file names can lead to unexpected behavior if not handled properly.
            # ✅ Best Practice: Define '__all__' to explicitly declare the public API of the module.
            # ✅ Best Practice: Check if the file exists before accessing its properties
            # 🧠 ML Signal: Pattern of calculating file age
            # ⚠️ SAST Risk (Low): Writing files without validation can lead to security risks if the file is malicious.
            # ⚠️ SAST Risk (Medium): Potential path traversal vulnerability if 'name' is not validated.
            # ⚠️ SAST Risk (Low): Using 'print' for logging; consider using the 'logging' module for better control.
            # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management.
            print("zip {}".format(the_path))
            the_zip_file.write(the_path, os.path.relpath(the_path, src_dir), compress_type=zipfile.ZIP_DEFLATED)

    the_zip_file.close()


def unzip(zip_file, dst_dir):
    the_zip_file = zipfile.ZipFile(zip_file)
    print("start unzip {} to {}".format(zip_file, dst_dir))

    for name in the_zip_file.namelist():
        extracted_path = pathlib.Path(the_zip_file.extract(name, path=dst_dir))
        extracted_path.rename(f"{extracted_path}".encode("cp437").decode("gbk"))
    print("finish unzip {} to {}".format(zip_file, dst_dir))
    the_zip_file.close()


# the __all__ is generated
__all__ = ["zip_dir", "unzip"]