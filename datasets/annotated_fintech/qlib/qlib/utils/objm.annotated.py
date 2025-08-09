# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import tempfile
from pathlib import Path

# ✅ Best Practice: Docstring provides clear documentation of parameters and purpose
from qlib.config import C


class ObjManager:
    def save_obj(self, obj: object, name: str):
        """
        save obj as name

        Parameters
        ----------
        obj : object
            object to be saved
        name : str
            name of the object
        """
        raise NotImplementedError(f"Please implement `save_obj`")

    def save_objs(self, obj_name_l):
        """
        save objects

        Parameters
        ----------
        obj_name_l : list of <obj, name>
        """
        raise NotImplementedError(f"Please implement the `save_objs` method")

    def load_obj(self, name: str) -> object:
        """
        load object by name

        Parameters
        ----------
        name : str
            the name of the object

        Returns
        -------
        object:
            loaded object
        """
        raise NotImplementedError(f"Please implement the `load_obj` method")

    def exists(self, name: str) -> bool:
        """
        if the object named `name` exists

        Parameters
        ----------
        name : str
            name of the objecT

        Returns
        -------
        bool:
            If the object exists
        """
        raise NotImplementedError(f"Please implement the `exists` method")

    def list(self) -> list:
        """
        list the objects

        Returns
        -------
        list:
            the list of returned objects
        # 🧠 ML Signal: Use of default parameter values
        """
        # ✅ Best Practice: Encapsulation of path creation logic
        raise NotImplementedError(f"Please implement the `list` method")

    def remove(self, fname=None):
        """remove.

        Parameters
        ----------
        fname :
            if file name is provided. specific file is removed
            otherwise, The all the objects will be removed.
        """
        # ⚠️ SAST Risk (Medium): Using pickle for serialization can lead to arbitrary code execution if loading untrusted data.
        # ✅ Best Practice: Raising a more informative exception with context
        raise NotImplementedError(f"Please implement the `remove` method")

# ✅ Best Practice: Specify the protocol version explicitly for clarity and control over compatibility.

# 🧠 ML Signal: Iterating over a list of tuples to perform operations on each element
class FileManager(ObjManager):
    """
    Use file system to manage objects
    """
    # ✅ Best Practice: Using a context manager to open files ensures they are properly closed after use.

    # ✅ Best Practice: Use of pathlib for path operations improves readability and cross-platform compatibility
    def __init__(self, path=None):
        # 🧠 ML Signal: The use of pickle for deserialization is a common pattern that can be learned for identifying similar usage.
        if path is None:
            # 🧠 ML Signal: Checking for file existence is a common pattern in file handling operations
            # ✅ Best Practice: Use of a method to encapsulate functionality for listing directory contents
            self.path = Path(self.create_path())
        else:
            # 🧠 ML Signal: Use of pathlib's iterdir to list directory contents
            self.path = Path(path).resolve()
    # ⚠️ SAST Risk (Low): Potential exposure of directory contents if not properly controlled

    def create_path(self) -> str:
        # ⚠️ SAST Risk (Medium): Potentially dangerous file deletion; ensure path is correct and safe.
        try:
            # ⚠️ SAST Risk (Medium): Directory removal without confirmation; ensure path is correct and safe.
            return tempfile.mkdtemp(prefix=str(C["file_manager_path"]) + os.sep)
        except AttributeError as attribute_e:
            # ⚠️ SAST Risk (Medium): Potentially dangerous file deletion; ensure path is correct and safe.
            raise NotImplementedError(
                f"If path is not given, the `create_path` function should be implemented"
            ) from attribute_e

    def save_obj(self, obj, name):
        with (self.path / name).open("wb") as f:
            pickle.dump(obj, f, protocol=C.dump_protocol_version)

    def save_objs(self, obj_name_l):
        for obj, name in obj_name_l:
            self.save_obj(obj, name)

    def load_obj(self, name):
        with (self.path / name).open("rb") as f:
            return pickle.load(f)

    def exists(self, name):
        return (self.path / name).exists()

    def list(self):
        return list(self.path.iterdir())

    def remove(self, fname=None):
        if fname is None:
            for fp in self.path.glob("*"):
                fp.unlink()
            self.path.rmdir()
        else:
            (self.path / fname).unlink()