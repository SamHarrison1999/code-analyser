# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from typing import Optional
import mlflow
import shutil
import pickle
import tempfile
import subprocess
import platform
from pathlib import Path
from datetime import datetime
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.

from qlib.utils.serial import Serializable
# 🧠 ML Signal: Logging is often used in ML workflows for tracking experiments and debugging.
from qlib.utils.exceptions import LoadObjectError
from qlib.utils.paral import AsyncCaller
# 🧠 ML Signal: Modifying MLflow's configuration can indicate custom experiment tracking behavior.

# ⚠️ SAST Risk (Low): Changing default MLflow settings might lead to unexpected behavior if not documented.
from ..log import TimeInspector, get_module_logger
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository

logger = get_module_logger("workflow")
# mlflow limits the length of log_param to 500, but this caused errors when using qrun, so we extended the mlflow limit.
# 🧠 ML Signal: Use of constants for status values indicates a pattern for state management
mlflow.utils.validation.MAX_PARAM_VAL_LENGTH = 1000

# 🧠 ML Signal: Use of constants for status values indicates a pattern for state management

class Recorder:
    """
    This is the `Recorder` class for logging the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The status of the recorder can be SCHEDULED, RUNNING, FINISHED, FAILED.
    """

    # ✅ Best Practice: Use class constants for status values to avoid magic strings.
    # ✅ Best Practice: Use __repr__ for unambiguous string representation of objects
    # status type
    STATUS_S = "SCHEDULED"
    # ✅ Best Practice: Use format method for string formatting for better readability
    # ✅ Best Practice: Implementing __str__ for user-friendly string representation of objects
    STATUS_R = "RUNNING"
    STATUS_FI = "FINISHED"
    # 🧠 ML Signal: Usage of __str__ method to convert object to string
    STATUS_FA = "FAILED"
    # ✅ Best Practice: Using a unique identifier for hashing ensures consistent and reliable hash values.

    def __init__(self, experiment_id, name):
        # ✅ Best Practice: Use of a method to encapsulate logic for creating a dictionary representation of the object
        self.id = None
        self.name = name
        # ✅ Best Practice: Initializing a dictionary using dict() for clarity
        self.experiment_id = experiment_id
        self.start_time = None
        # 🧠 ML Signal: Storing class name in the output dictionary
        self.end_time = None
        self.status = Recorder.STATUS_S
    # 🧠 ML Signal: Storing object attributes in a dictionary for serialization or logging

    def __repr__(self):
        return "{name}(info={info})".format(name=self.__class__.__name__, info=self.info)

    # ✅ Best Practice: Method name should be descriptive of its action
    def __str__(self):
        return str(self.info)
    # ✅ Best Practice: Use of 'self' indicates this is an instance method

    # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and parameters.
    # ✅ Best Practice: Returning a dictionary for easy conversion to JSON or other formats
    # 🧠 ML Signal: Setting an instance variable
    def __hash__(self) -> int:
        return hash(self.info["id"])

    @property
    def info(self):
        output = dict()
        output["class"] = "Recorder"
        output["id"] = self.id
        output["name"] = self.name
        output["experiment_id"] = self.experiment_id
        output["start_time"] = self.start_time
        output["end_time"] = self.end_time
        # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called.
        output["status"] = self.status
        # ✅ Best Practice: Docstring provides clear documentation of parameters and return value
        return output

    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        """
        Save objects such as prediction file or model checkpoints to the artifact URI. User
        can save object through keywords arguments (name:value).

        Please refer to the docs of qlib.workflow:R.save_objects

        Parameters
        ----------
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path=None : str
            the relative path for the artifact to be stored in the URI.
        """
        raise NotImplementedError(f"Please implement the `save_objects` method.")
    # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called.

    def load_object(self, name):
        """
        Load objects such as prediction file or model checkpoints.

        Parameters
        ----------
        name : str
            name of the file to be loaded.

        Returns
        -------
        The saved object.
        """
        # ⚠️ SAST Risk (Low): NotImplementedError should be handled to avoid runtime exceptions
        raise NotImplementedError(f"Please implement the `load_object` method.")
    # ✅ Best Practice: Use of docstring to describe the function and its parameters

    def start_run(self):
        """
        Start running or resuming the Recorder. The return value can be used as a context manager within a `with` block;
        otherwise, you must call end_run() to terminate the current run. (See `ActiveRun` class in mlflow)

        Returns
        -------
        An active running object (e.g. mlflow.ActiveRun object).
        # ✅ Best Practice: Docstring provides clear explanation of parameters and functionality
        """
        raise NotImplementedError(f"Please implement the `start_run` method.")

    def end_run(self):
        """
        End an active Recorder.
        """
        raise NotImplementedError(f"Please implement the `end_run` method.")

    def log_params(self, **kwargs):
        """
        Log a batch of params for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as parameters.
        """
        raise NotImplementedError(f"Please implement the `log_params` method.")
    # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called

    # 🧠 ML Signal: Method signature with variable arguments indicates flexible input handling
    def log_metrics(self, step=None, **kwargs):
        """
        Log multiple metrics for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as metrics.
        # ⚠️ SAST Risk (Low): Method not implemented, potential for misuse if called unexpectedly
        """
        # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
        # ✅ Best Practice: Use of NotImplementedError to indicate an unimplemented method
        raise NotImplementedError(f"Please implement the `log_metrics` method.")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file or directory as an artifact of the currently active run.

        Parameters
        ----------
        local_path : str
            Path to the file to write.
        artifact_path : Optional[str]
            If provided, the directory in ``artifact_uri`` to write to.
        # ✅ Best Practice: Type hints are used for function parameters and return type.
        """
        raise NotImplementedError(f"Please implement the `log_metrics` method.")

    def set_tags(self, **kwargs):
        """
        Log a batch of tags for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as tags.
        """
        raise NotImplementedError(f"Please implement the `set_tags` method.")

    def delete_tags(self, *keys):
        """
        Delete some tags from a run.

        Parameters
        ----------
        keys : series of strs of the keys
            all the name of the tag to be deleted.
        """
        raise NotImplementedError(f"Please implement the `delete_tags` method.")

    def list_artifacts(self, artifact_path: str = None):
        """
        List all the artifacts of a recorder.

        Parameters
        ----------
        artifact_path : str
            the relative path for the artifact to be stored in the URI.

        Returns
        -------
        A list of artifacts information (name, path, etc.) that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        Parameters
        ----------
        path : str
            Relative source path to the desired artifact.
        dst_path : Optional[str]
            Absolute path of the local filesystem destination directory to which to
            download the specified artifacts. This directory must already exist.
            If unspecified, the artifacts will either be downloaded to a new
            uniquely-named directory on the local filesystem.

        Returns
        -------
        str
            Local path of desired artifact.
        """
        # 🧠 ML Signal: Usage of mlflow.tracking.MlflowClient indicates interaction with MLflow tracking server.
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")

    def list_metrics(self):
        """
        List all the metrics of a recorder.

        Returns
        -------
        A dictionary of metrics that being stored.
        """
        # ✅ Best Practice: Conversion of timestamps to human-readable format improves readability.
        raise NotImplementedError(f"Please implement the `list_metrics` method.")

    def list_params(self):
        """
        List all the params of a recorder.

        Returns
        -------
        A dictionary of params that being stored.
        # 🧠 ML Signal: Usage of class name in string formatting
        """
        # ✅ Best Practice: Calculating space length for better formatted output
        # ✅ Best Practice: Using str.format for string formatting
        raise NotImplementedError(f"Please implement the `list_params` method.")

    def list_tags(self):
        """
        List all the tags of a recorder.

        Returns
        -------
        A dictionary of tags that being stored.
        """
        # 🧠 ML Signal: Usage of dictionary access pattern with a specific key.
        # ✅ Best Practice: Use of type hinting for method parameters and return type
        raise NotImplementedError(f"Please implement the `list_tags` method.")
# ⚠️ SAST Risk (Low): Assumes 'id' key exists in self.info, which may raise a KeyError if not present.

# ✅ Best Practice: Use of isinstance to check object type

# 🧠 ML Signal: Equality comparison based on a specific attribute
class MLflowRecorder(Recorder):
    """
    Use mlflow to implement a Recorder.

    Due to the fact that mlflow will only log artifact from a file or directory, we decide to
    use file manager to help maintain the objects in the project.

    Instead of using mlflow directly, we use another interface wrapping mlflow to log experiments.
    Though it takes extra efforts, but it brings users benefits due to following reasons.
    - It will be more convenient to change the experiment logging backend without changing any code in upper level
    - We can provide more convenience to automatically do some extra things and make interface easier. For examples:
        - Automatically logging the uncommitted code
        - Automatically logging part of environment variables
        - User can control several different runs by just creating different Recorder (in mlflow, you always have to switch artifact_uri and pass in run ids frequently)
    """
    # ✅ Best Practice: Use Path from pathlib for path manipulations

    def __init__(self, experiment_id, uri, name=None, mlflow_run=None):
        super(MLflowRecorder, self).__init__(experiment_id, name)
        # ✅ Best Practice: Use Path from pathlib for path manipulations
        self._uri = uri
        self._artifact_uri = None
        # ✅ Best Practice: Convert Path object to string after resolving
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        # construct from mlflow run
        # ✅ Best Practice: Check if the directory exists before returning
        if mlflow_run is not None:
            assert isinstance(mlflow_run, mlflow.entities.run.Run), "Please input with a MLflow Run object."
            self.name = mlflow_run.data.tags["mlflow.runName"]
            self.id = mlflow_run.info.run_id
            # ⚠️ SAST Risk (Low): Raising a generic RuntimeError without specific error handling
            # 🧠 ML Signal: Setting a tracking URI is a common pattern in ML experiment tracking
            self.status = mlflow_run.info.status
            self.start_time = (
                # 🧠 ML Signal: Starting a run is a key action in ML experiment management
                datetime.fromtimestamp(float(mlflow_run.info.start_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                # ⚠️ SAST Risk (Low): Raising a ValueError without specific error handling
                if mlflow_run.info.start_time is not None
                # 🧠 ML Signal: Storing run ID for future reference is a common pattern
                else None
            )
            # 🧠 ML Signal: Storing artifact URI for accessing experiment outputs
            self.end_time = (
                datetime.fromtimestamp(float(mlflow_run.info.end_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                # 🧠 ML Signal: Recording start time is useful for tracking experiment duration
                if mlflow_run.info.end_time is not None
                else None
            # 🧠 ML Signal: Setting status to running is a common state management pattern
            )
            self._artifact_uri = mlflow_run.info.artifact_uri
        self.async_log = None
    # ✅ Best Practice: Using logging for information output instead of print

    # 🧠 ML Signal: Asynchronous logging can be used to improve performance
    def __repr__(self):
        # 🧠 ML Signal: Logging uncommitted code helps in experiment reproducibility
        name = self.__class__.__name__
        space_length = len(name) + 1
        return "{name}(info={info},\n{space}uri={uri},\n{space}artifact_uri={artifact_uri},\n{space}client={client})".format(
            name=name,
            # 🧠 ML Signal: Iterating over a list of tuples to execute commands and log results
            # 🧠 ML Signal: Logging command-line arguments is useful for experiment tracking
            # 🧠 ML Signal: Logging environment variables can be useful for experiment context
            space=" " * space_length,
            info=self.info,
            uri=self.uri,
            artifact_uri=self.artifact_uri,
            client=self.client,
        # 🧠 ML Signal: Returning the run object allows further interaction with the run
        )

    # ⚠️ SAST Risk (High): Use of shell=True with subprocess.check_output can lead to shell injection vulnerabilities
    def __hash__(self) -> int:
        return hash(self.info["id"])
    # 🧠 ML Signal: Logging output of subprocess command to a client

    def __eq__(self, o: object) -> bool:
        # ✅ Best Practice: Logging exceptions provides insight into failures
        # 🧠 ML Signal: Use of assert for input validation
        if isinstance(o, MLflowRecorder):
            return self.info["id"] == o.info["id"]
        return False

    @property
    def uri(self):
        return self._uri
    # ✅ Best Practice: Use of datetime for timestamping

    @property
    def artifact_uri(self):
        return self._artifact_uri

    # ✅ Best Practice: Use of context manager for resource management
    def get_local_dir(self):
        """
        This function will return the directory path of this recorder.
        # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled with optimized execution
        """
        # 🧠 ML Signal: Interaction with mlflow for experiment tracking
        if self.artifact_uri is not None:
            if platform.system() == "Windows":
                local_dir_path = Path(self.artifact_uri.lstrip("file:").lstrip("/")).parent
            # ✅ Best Practice: Check if path is a directory to decide logging method
            else:
                local_dir_path = Path(self.artifact_uri.lstrip("file:")).parent
            local_dir_path = str(local_dir_path.resolve())
            if os.path.isdir(local_dir_path):
                return local_dir_path
            else:
                # ✅ Best Practice: Use of tempfile for temporary directory creation
                raise RuntimeError("This recorder is not saved in the local file system.")

        else:
            raise ValueError(
                # 🧠 ML Signal: Usage of Serializable.general_dump for data serialization
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            # ⚠️ SAST Risk (Low): Potential risk if temp_dir is not properly managed
            )

    def start_run(self):
        # set the tracking uri
        mlflow.set_tracking_uri(self.uri)
        # start the run
        run = mlflow.start_run(self.id, self.experiment_id, self.name)
        # save the run id and artifact_uri
        self.id = run.info.run_id
        self._artifact_uri = run.info.artifact_uri
        # ⚠️ SAST Risk (Low): Use of assert statement for runtime checks, which can be disabled with optimization flags.
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status = Recorder.STATUS_R
        logger.info(f"Recorder {self.id} starts running under Experiment {self.experiment_id} ...")

        # 🧠 ML Signal: Downloading artifacts, which is a common pattern in ML workflows.
        # NOTE: making logging async.
        # - This may cause delay when uploading results
        # ⚠️ SAST Risk (Medium): Unpickling data can lead to arbitrary code execution if the source is untrusted.
        # - The logging time may not be accurate
        self.async_log = AsyncCaller()

        # TODO: currently, this is only supported in MLflowRecorder.
        # Maybe we can make this feature more general.
        # ✅ Best Practice: Raising a custom error with context for better error handling.
        self._log_uncommitted_code()

        self.log_params(**{"cmd-sys.argv": " ".join(sys.argv)})  # log the command to produce current experiment
        # ✅ Best Practice: Cleaning up resources in a finally block to ensure execution.
        self.log_params(
            # 🧠 ML Signal: Use of **kwargs indicates dynamic parameter handling
            **{k: v for k, v in os.environ.items() if k.startswith("_QLIB_")}
        # 🧠 ML Signal: Iterating over dictionary items
        # ⚠️ SAST Risk (Low): Potential risk of deleting unintended files if path is not correctly managed.
        )  # Log necessary environment variables
        return run
    # ✅ Best Practice: Descriptive variable names improve readability
    # ✅ Best Practice: Use of **kwargs allows for flexible function arguments

    # 🧠 ML Signal: Use of decorators to handle asynchronous operations, common in ML logging.
    def _log_uncommitted_code(self):
        """
        Mlflow only log the commit id of the current repo. But usually, user will have a lot of uncommitted changes.
        So this tries to automatically to log them all.
        """
        # ✅ Best Practice: Use of **kwargs allows for flexible function arguments
        # TODO: the sub-directories maybe git repos.
        # So it will be better if we can walk the sub-directories and log the uncommitted changes.
        # 🧠 ML Signal: Iterating over dictionary items is a common pattern
        # 🧠 ML Signal: Use of asynchronous decorator indicates a pattern of non-blocking operations
        for cmd, fname in [
            # ✅ Best Practice: Decorator pattern used for adding asynchronous behavior, enhancing code modularity
            ("git diff", "code_diff.txt"),
            # ⚠️ SAST Risk (Low): Potential risk if self.client.set_tag does not handle inputs safely
            # ✅ Best Practice: Iterating over variable arguments allows for flexible input handling.
            ("git status", "code_status.txt"),
            ("git diff --cached", "code_cached.txt"),
        # 🧠 ML Signal: Usage of a client object to perform operations, indicating a possible API interaction pattern.
        ]:
            # ⚠️ SAST Risk (Medium): Potential for improper handling of exceptions if the delete_tag method fails.
            # ✅ Best Practice: Check for None to ensure the attribute is set before use
            try:
                out = subprocess.check_output(cmd, shell=True)
                self.client.log_text(self.id, out.decode(), fname)  # this behaves same as above
            # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific error handling
            except subprocess.CalledProcessError:
                logger.info(f"Fail to log the uncommitted code of $CWD({os.getcwd()}) when run {cmd}.")

    def end_run(self, status: str = Recorder.STATUS_S):
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode, leading to potential issues.
        assert status in [
            # ✅ Best Practice: Consider using exception handling instead of assert for better error management.
            Recorder.STATUS_S,
            Recorder.STATUS_R,
            # 🧠 ML Signal: Usage of a client to list artifacts, indicating interaction with an external system or service.
            # 🧠 ML Signal: Method signature with type hints indicates expected input and output types
            Recorder.STATUS_FI,
            # ✅ Best Practice: Use of Optional for dst_path indicates that the parameter is not required
            Recorder.STATUS_FA,
        # 🧠 ML Signal: List comprehension used to extract paths from artifacts, indicating data transformation.
        # 🧠 ML Signal: Method accessing client to retrieve run data
        ], f"The status type {status} is not supported."
        # 🧠 ML Signal: Delegating functionality to another method (self.client.download_artifacts)
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # ⚠️ SAST Risk (Low): Potential for improper handling of file paths leading to path traversal
        # 🧠 ML Signal: Client method call to get run by ID
        if self.status != Recorder.STATUS_S:
            # 🧠 ML Signal: Method accessing client to retrieve run data
            self.status = status
        # 🧠 ML Signal: Accessing metrics from run data
        if self.async_log is not None:
            # 🧠 ML Signal: Accessing external client to get run information
            # Waiting Queue should go before mlflow.end_run. Otherwise mlflow will raise error
            # 🧠 ML Signal: Method accessing a client's run data
            with TimeInspector.logt("waiting `async_log`"):
                # 🧠 ML Signal: Returning parameters from run data
                self.async_log.wait()
        # 🧠 ML Signal: Fetching a run object using a client
        # 🧠 ML Signal: Accessing tags from run data
        self.async_log = None
        mlflow.end_run(status)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        if local_path is not None:
            path = Path(local_path)
            if path.is_dir():
                self.client.log_artifacts(self.id, local_path, artifact_path)
            else:
                self.client.log_artifact(self.id, local_path, artifact_path)
        else:
            temp_dir = Path(tempfile.mkdtemp()).resolve()
            for name, data in kwargs.items():
                path = temp_dir / name
                Serializable.general_dump(data, path)
                self.client.log_artifact(self.id, temp_dir / name, artifact_path)
            shutil.rmtree(temp_dir)

    def load_object(self, name, unpickler=pickle.Unpickler):
        """
        Load object such as prediction file or model checkpoint in mlflow.

        Args:
            name (str): the object name

            unpickler: Supporting using custom unpickler

        Raises:
            LoadObjectError: if raise some exceptions when load the object

        Returns:
            object: the saved object in mlflow.
        """
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."

        path = None
        try:
            path = self.client.download_artifacts(self.id, name)
            with Path(path).open("rb") as f:
                data = unpickler(f).load()
            return data
        except Exception as e:
            raise LoadObjectError(str(e)) from e
        finally:
            ar = self.client._tracking_client._get_artifact_repo(self.id)
            if isinstance(ar, AzureBlobArtifactRepository) and path is not None:
                # for saving disk space
                # For safety, only remove redundant file for specific ArtifactRepository
                shutil.rmtree(Path(path).absolute().parent)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_params(self, **kwargs):
        for name, data in kwargs.items():
            self.client.log_param(self.id, name, data)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metrics(self, step=None, **kwargs):
        for name, data in kwargs.items():
            self.client.log_metric(self.id, name, data, step=step)

    def log_artifact(self, local_path, artifact_path: Optional[str] = None):
        self.client.log_artifact(self.id, local_path=local_path, artifact_path=artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def set_tags(self, **kwargs):
        for name, data in kwargs.items():
            self.client.set_tag(self.id, name, data)

    def delete_tags(self, *keys):
        for key in keys:
            self.client.delete_tag(self.id, key)

    def get_artifact_uri(self):
        if self.artifact_uri is not None:
            return self.artifact_uri
        else:
            raise ValueError(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def list_artifacts(self, artifact_path=None):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        artifacts = self.client.list_artifacts(self.id, artifact_path)
        return [art.path for art in artifacts]

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        return self.client.download_artifacts(self.id, path, dst_path)

    def list_metrics(self):
        run = self.client.get_run(self.id)
        return run.data.metrics

    def list_params(self):
        run = self.client.get_run(self.id)
        return run.data.params

    def list_tags(self):
        run = self.client.get_run(self.id)
        return run.data.tags