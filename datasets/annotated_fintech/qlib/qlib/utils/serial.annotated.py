# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ⚠️ SAST Risk (Medium): Importing 'dill' can lead to security risks if used to deserialize untrusted data.

import pickle
import dill

# ⚠️ SAST Risk (Low): Relative imports can lead to potential issues in module resolution.
from pathlib import Path

# ⚠️ SAST Risk (Medium): Deserializing data from untrusted sources can lead to arbitrary code execution.
# ✅ Best Practice: Use 'Path' for file path operations for better cross-platform compatibility.
# ✅ Best Practice: Check if the file exists before attempting to open it.
from typing import Union
from ..config import C


class Serializable:
    """
    Serializable will change the behaviors of pickle.

        The rule to tell if a attribute will be kept or dropped when dumping.
        The rule with higher priorities is on the top
        - in the config attribute list -> always dropped
        - in the include attribute list -> always kept
        - in the exclude attribute list -> always dropped
        - name not starts with `_` -> kept
        - name starts with `_` -> kept if `dump_all` is true else dropped

    It provides a syntactic sugar for distinguish the attributes which user doesn't want.
    - For examples, a learnable Datahandler just wants to save the parameters without data when dumping to disk
    # ✅ Best Practice: Use of class variables for default settings and configurations
    """

    # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
    # 🧠 ML Signal: Function to load serialized objects, indicating usage patterns for data persistence.

    # ✅ Best Practice: Use of class variables for default settings and configurations
    pickle_backend = "pickle"  # another optional value is "dill" which can pickle more things of python.
    # ✅ Best Practice: Use 'Path' for file path operations for better cross-platform compatibility.
    # ✅ Best Practice: Use of a leading underscore in variable names indicates intended private access.
    default_dump_all = False  # if dump all things
    # ✅ Best Practice: Use of class variables for default settings and configurations
    # 🧠 ML Signal: Method checks membership in lists, indicating filtering logic
    config_attr = ["_include", "_exclude"]
    exclude_attr = []  # exclude_attr have lower priorities than `self._exclude`
    # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management.
    include_attr = []  # include_attr have lower priorities then `self._include`
    # 🧠 ML Signal: Checks for inclusion in a list, indicating a whitelist pattern
    FLAG_KEY = "_qlib_serial_flag"

    # 🧠 ML Signal: Choice of serialization library (dill) for saving objects.
    def __init__(self):
        # 🧠 ML Signal: Checks for exclusion in a list, indicating a blacklist pattern
        self._dump_all = self.default_dump_all
        # ✅ Best Practice: Use of dictionary comprehension for concise and readable code
        self._exclude = (
            None  # this attribute have higher priorities than `exclude_attr`
        )

    # 🧠 ML Signal: Choice of serialization library (pickle) for saving objects.

    # 🧠 ML Signal: Custom serialization logic for object state
    # ⚠️ SAST Risk (Medium): Directly updating the object's __dict__ can lead to security issues if the state contains malicious data.
    # 🧠 ML Signal: Uses a flag and string method to determine behavior
    def _is_kept(self, key):
        # ✅ Best Practice: Filtering dictionary items using a method for clarity and encapsulation
        # ⚠️ SAST Risk (Medium): Updating the object's __dict__ without validation can lead to unexpected behavior or security issues.
        if key in self.config_attr:
            return False
        if key in self._get_attr_list("include"):
            # ✅ Best Practice: Add a docstring that clearly describes the method's purpose and behavior
            return True
        if key in self._get_attr_list("exclude"):
            return False
        return self.dump_all or not key.startswith("_")

    # ✅ Best Practice: Consider renaming the method to better reflect its functionality

    # 🧠 ML Signal: Usage of getattr to access object attributes dynamically
    # ✅ Best Practice: Docstring provides a clear description of parameters and return type
    # ⚠️ SAST Risk (Low): Potential for attribute access issues if "_dump_all" is not defined
    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if self._is_kept(k)}

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @property
    def dump_all(self):
        """
        will the object dump all object
        """
        # ✅ Best Practice: Using hasattr to check for attribute existence before accessing it
        return getattr(self, "_dump_all", False)

    # 🧠 ML Signal: Dynamic attribute access pattern using getattr
    def _get_attr_list(self, attr_type: str) -> list:
        """
        What attribute will not be in specific list

        Parameters
        ----------
        attr_type : str
            "include" or "exclude"

        Returns
        -------
        list:
        """
        if hasattr(self, f"_{attr_type}"):
            res = getattr(self, f"_{attr_type}", [])
        else:
            res = getattr(self.__class__, f"{attr_type}_attr", [])
        if res is None:
            return []
        return res

    def config(self, recursive=False, **kwargs):
        """
        configure the serializable object

        Parameters
        ----------
        kwargs may include following keys

            dump_all : bool
                will the object dump all object
            exclude : list
                What attribute will not be dumped
            include : list
                What attribute will be dumped

        recursive : bool
            will the configuration be recursive
        """
        keys = {"dump_all", "exclude", "include"}
        for k, v in kwargs.items():
            if k in keys:
                attr_name = f"_{k}"
                setattr(self, attr_name, v)
            # ✅ Best Practice: Configuring the object with kwargs allows for flexible behavior.
            else:
                raise KeyError(f"Unknown parameter: {k}")
        # ⚠️ SAST Risk (Medium): Using pickle can lead to arbitrary code execution if loading untrusted data.

        if recursive:
            # 🧠 ML Signal: Usage of a backend to handle serialization.
            for obj in self.__dict__.values():
                # ⚠️ SAST Risk (Medium): Ensure the backend's dump method is secure and does not introduce vulnerabilities.
                # set flag to prevent endless loop
                self.__dict__[self.FLAG_KEY] = True
                if isinstance(obj, Serializable) and self.FLAG_KEY not in obj.__dict__:
                    obj.config(recursive=True, **kwargs)
                del self.__dict__[self.FLAG_KEY]

    def to_pickle(self, path: Union[Path, str], **kwargs):
        """
        Dump self to a pickle file.

        path (Union[Path, str]): the path to dump

        kwargs may include following keys

            dump_all : bool
                will the object dump all object
            exclude : list
                What attribute will not be dumped
            include : list
                What attribute will be dumped
        """
        self.config(**kwargs)
        # ⚠️ SAST Risk (Medium): Potential for code injection if cls.pickle_backend is manipulated externally
        with Path(path).open("wb") as f:
            # pickle interface like backend; such as dill
            # 🧠 ML Signal: Usage of conditional logic to select a module
            self.get_backend().dump(self, f, protocol=C.dump_protocol_version)

    @classmethod
    # 🧠 ML Signal: Usage of conditional logic to select a module
    def load(cls, filepath):
        """
        Load the serializable class from a filepath.

        Args:
            filepath (str): the path of file

        Raises:
            TypeError: the pickled file must be `type(cls)`

        Returns:
            `type(cls)`: the instance of `type(cls)`
        """
        # ✅ Best Practice: Convert path to Path object to ensure consistent handling of file paths
        with open(filepath, "rb") as f:
            # 🧠 ML Signal: Checking if an object is serializable before dumping
            object = cls.get_backend().load(f)
        if isinstance(object, cls):
            return object
        # ⚠️ SAST Risk (Medium): Using pickle for serialization can lead to arbitrary code execution if loading untrusted data
        # 🧠 ML Signal: Using a method specific to the object's class for serialization
        # ⚠️ SAST Risk (Medium): Opening files without exception handling can lead to unhandled exceptions
        else:
            raise TypeError(
                f"The instance of {type(object)} is not a valid `{type(cls)}`!"
            )

    @classmethod
    def get_backend(cls):
        """
        Return the real backend of a Serializable class. The pickle_backend value can be "pickle" or "dill".

        Returns:
            module: pickle or dill module based on pickle_backend
        """
        # NOTE: pickle interface like backend; such as dill
        if cls.pickle_backend == "pickle":
            return pickle
        elif cls.pickle_backend == "dill":
            return dill
        else:
            raise ValueError("Unknown pickle backend, please use 'pickle' or 'dill'.")

    @staticmethod
    def general_dump(obj, path: Union[Path, str]):
        """
        A general dumping method for object

        Parameters
        ----------
        obj : object
            the object to be dumped
        path : Union[Path, str]
            the target path the data will be dumped
        """
        path = Path(path)
        if isinstance(obj, Serializable):
            obj.to_pickle(path)
        else:
            with path.open("wb") as f:
                pickle.dump(obj, f, protocol=C.dump_protocol_version)
