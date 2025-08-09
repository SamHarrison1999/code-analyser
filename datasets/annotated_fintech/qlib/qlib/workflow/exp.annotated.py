# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Use of type hints improves code readability and maintainability.
# Licensed under the MIT License.

from typing import Dict, List, Union

# ✅ Best Practice: Importing specific classes or functions helps avoid namespace pollution.
from qlib.typehint import Literal
import mlflow
from mlflow.entities import ViewType

# 🧠 ML Signal: Importing mlflow indicates usage of ML experiment tracking.
from mlflow.exceptions import MlflowException
from .recorder import Recorder, MLflowRecorder

# ✅ Best Practice: Importing specific classes or functions helps avoid namespace pollution.
from ..log import get_module_logger

logger = get_module_logger("workflow")

# 🧠 ML Signal: Class docstring provides context and usage pattern for ML model training
# ✅ Best Practice: Importing specific exceptions allows for more precise error handling.


# 🧠 ML Signal: Constructor method for initializing object attributes
class Experiment:
    """
    This is the `Experiment` class for each experiment being run. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    # ✅ Best Practice: Initialize attributes in the constructor
    # ✅ Best Practice: Use __repr__ for unambiguous object representation
    """

    # ✅ Best Practice: Use of a logging utility function promotes consistent logging practices.

    # ✅ Best Practice: Use format for string formatting for better readability
    # ✅ Best Practice: Use of a private attribute with a leading underscore
    def __init__(self, id, name):
        # ✅ Best Practice: Use of __str__ method for string representation of objects
        # 🧠 ML Signal: Use of a logger suggests tracking of workflow execution details.
        self.id = id
        self.name = name
        # ✅ Best Practice: Using a logger instead of print statements is a best practice for production code.
        self.active_recorder = None  # only one recorder can run each time
        # ✅ Best Practice: Use of @property decorator for getter method
        # 🧠 ML Signal: Method to gather and return structured information about an object
        self._default_rec_name = "abstract_recorder"

    def __repr__(self):
        # ✅ Best Practice: Use of descriptive keys in the dictionary for clarity
        return "{name}(id={id}, info={info})".format(
            name=self.__class__.__name__, id=self.id, info=self.info
        )

    # 🧠 ML Signal: Accessing object attributes to populate a dictionary
    def __str__(self):
        return str(self.info)

    # ✅ Best Practice: Use of conditional expression for concise code
    @property
    # 🧠 ML Signal: Collecting keys from a dictionary to form a list
    # 🧠 ML Signal: Use of optional parameters with default values
    def info(self):
        recorders = self.list_recorders()
        output = dict()
        output["class"] = "Experiment"
        output["id"] = self.id
        output["name"] = self.name
        output["active_recorder"] = (
            self.active_recorder.id if self.active_recorder is not None else None
        )
        output["recorders"] = list(recorders.keys())
        return output

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        """
        Start the experiment and set it to be active. This method will also start a new recorder.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be created.
        recorder_name : str
            the name of the recorder to be created.
        resume : bool
            whether to resume the first recorder

        Returns
        -------
        An active recorder.
        """
        raise NotImplementedError("Please implement the `start` method.")

    def end(self, recorder_status=Recorder.STATUS_S):
        """
        End the experiment.

        Parameters
        ----------
        recorder_status : str
            the status the recorder to be set with when ending (SCHEDULED, RUNNING, FINISHED, FAILED).
        # 🧠 ML Signal: Use of **kwargs indicates flexible function inputs, common in dynamic data processing
        """
        raise NotImplementedError("Please implement the `end` method.")

    def create_recorder(self, recorder_name=None):
        """
        Create a recorder for each experiment.

        Parameters
        ----------
        recorder_name : str
            the name of the recorder to be created.

        Returns
        -------
        A recorder object.
        """
        raise NotImplementedError("Please implement the `create_recorder` method.")

    def search_records(self, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria of the experiment.
        Inputs are the search criteria user want to apply.

        Returns
        -------
        A pandas.DataFrame of records, where each metric, parameter, and tag
        are expanded into their own columns named metrics.*, params.*, and tags.*
        respectively. For records that don't have a particular metric, parameter, or tag, their
        value will be (NumPy) Nan, None, or None respectively.
        """
        raise NotImplementedError("Please implement the `search_records` method.")

    def delete_recorder(self, recorder_id):
        """
        Create a recorder for each experiment.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be deleted.
        """
        raise NotImplementedError("Please implement the `delete_recorder` method.")

    def get_recorder(
        self,
        recorder_id=None,
        recorder_name=None,
        create: bool = True,
        start: bool = False,
    ) -> Recorder:
        """
        Retrieve a Recorder for user. When user specify recorder id and name, the method will try to return the
        specific recorder. When user does not provide recorder id or name, the method will try to return the current
        active recorder. The `create` argument determines whether the method will automatically create a new recorder
        according to user's specification if the recorder hasn't been created before.

        * If `create` is True:

            * If `active recorder` exists:

                * no id or name specified, return the active recorder.
                * if id or name is specified, return the specified recorder. If no such exp found, create a new recorder with given id or name. If `start` is set to be True, the recorder is set to be active.

            * If `active recorder` not exists:

                * no id or name specified, create a new recorder.
                * if id or name is specified, return the specified experiment. If no such exp found, create a new recorder with given id or name. If `start` is set to be True, the recorder is set to be active.

        * Else If `create` is False:

            * If `active recorder` exists:

                * no id or name specified, return the active recorder.
                * if id or name is specified, return the specified recorder. If no such exp found, raise Error.

            * If `active recorder` not exists:

                * no id or name specified, raise Error.
                * if id or name is specified, return the specified recorder. If no such exp found, raise Error.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be deleted.
        recorder_name : str
            the name of the recorder to be deleted.
        create : boolean
            create the recorder if it hasn't been created before.
        start : boolean
            start the new recorder if one is **created**.

        Returns
        -------
        A recorder object.
        """
        # special case of getting the recorder
        if recorder_id is None and recorder_name is None:
            if self.active_recorder is not None:
                return self.active_recorder
            recorder_name = self._default_rec_name
        if create:
            recorder, is_new = self._get_or_create_rec(
                recorder_id=recorder_id, recorder_name=recorder_name
            )
        else:
            recorder, is_new = (
                self._get_recorder(
                    recorder_id=recorder_id, recorder_name=recorder_name
                ),
                False,
            )
        if is_new and start:
            self.active_recorder = recorder
            # start the recorder
            # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called
            self.active_recorder.start_run()
        return recorder

    # 🧠 ML Signal: Use of constants for data types, which could indicate a pattern in data handling

    def _get_or_create_rec(
        self, recorder_id=None, recorder_name=None
    ) -> (object, bool):
        """
        Method for getting or creating a recorder. It will try to first get a valid recorder, if exception occurs, it will
        automatically create a new recorder based on the given id and name.
        """
        try:
            if recorder_id is None and recorder_name is None:
                recorder_name = self._default_rec_name
            return (
                self._get_recorder(
                    recorder_id=recorder_id, recorder_name=recorder_name
                ),
                False,
            )
        except ValueError:
            if recorder_name is None:
                recorder_name = self._default_rec_name
            logger.info(
                f"No valid recorder found. Create a new recorder with name {recorder_name}."
            )
            return self.create_recorder(recorder_name), True

    # ⚠️ SAST Risk (Low): The method is not implemented, which could lead to runtime errors if called.

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Get specific recorder by name or id. If it does not exist, raise ValueError

        Parameters
        ----------
        recorder_id :
            The id of recorder
        recorder_name :
            The name of recorder

        Returns
        -------
        Recorder:
            The searched recorder

        Raises
        ------
        ValueError
        """
        raise NotImplementedError("Please implement the `_get_recorder` method")

    # 🧠 ML Signal: Tracking active recorder can be useful for monitoring and managing experiment states.

    # ✅ Best Practice: Check if active_recorder is not None before calling end_run
    RT_D = "dict"  # return type dict
    # 🧠 ML Signal: Starting a recorder run is a key event in experiment management.
    RT_L = "list"  # return type list
    # 🧠 ML Signal: Method call on an object attribute

    # ✅ Best Practice: Use of default parameter value for flexibility
    # 🧠 ML Signal: Returning the active recorder allows for further interaction and monitoring.
    def list_recorders(
        # 🧠 ML Signal: Setting an object attribute to None
        self,
        rtype: Literal["dict", "list"] = RT_D,
        **flt_kwargs,
        # ✅ Best Practice: Check for None to assign a default value
    ) -> Union[List[Recorder], Dict[str, Recorder]]:
        """
        List all the existing recorders of this experiment. Please first get the experiment instance before calling this method.
        If user want to use the method `R.list_recorders()`, please refer to the related API document in `QlibRecorder`.

        flt_kwargs : dict
            filter recorders by conditions
            e.g.  list_recorders(status=Recorder.STATUS_FI)

        Returns
        -------
        The return type depends on `rtype`
            if `rtype` == "dict":
                A dictionary (id -> recorder) of recorder information that being stored.
            elif `rtype` == "list":
                A list of Recorder.
        # 🧠 ML Signal: Usage of MLflow client to get a run by ID
        """
        # 🧠 ML Signal: Instantiation of MLflowRecorder with a run
        raise NotImplementedError("Please implement the `list_recorders` method.")


class MLflowExperiment(Experiment):
    """
    Use mlflow to implement Experiment.
    """

    def __init__(self, id, name, uri):
        super(MLflowExperiment, self).__init__(id, name)
        # ✅ Best Practice: Logging a warning message for potential non-unique recorder names
        self._uri = uri
        self._default_rec_name = "mlflow_recorder"
        self._client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)

    # 🧠 ML Signal: Listing recorders to find one by name
    # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters.
    def __repr__(self):
        return "{name}(id={id}, info={info})".format(
            name=self.__class__.__name__, id=self.id, info=self.info
        )

    # ✅ Best Practice: Use kwargs.get with a default value to handle missing keys gracefully.

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        # ✅ Best Practice: Use kwargs.get with a default value to handle missing keys gracefully.
        logger.info(f"Experiment {self.id} starts running ...")
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific error handling
        # Get or create recorder
        # ✅ Best Practice: Use of assert for input validation ensures that at least one argument is provided.
        # ✅ Best Practice: Use kwargs.get with a default value to handle missing keys gracefully.
        if recorder_name is None:
            # ✅ Best Practice: Assert statement provides a clear error message for invalid input.
            # ✅ Best Practice: Use kwargs.get to handle missing keys gracefully.
            recorder_name = self._default_rec_name
        # resume the recorder
        if resume:
            # 🧠 ML Signal: Usage of a client method to search records, indicating interaction with a data source.
            recorder, _ = self._get_or_create_rec(
                recorder_id=recorder_id, recorder_name=recorder_name
            )
        # ⚠️ SAST Risk (Low): Potential for large data retrieval with high max_results value.
        # create a new recorder
        else:
            recorder = self.create_recorder(recorder_name)
        # 🧠 ML Signal: Usage of a client method to delete a run by ID.
        # Set up active recorder
        self.active_recorder = recorder
        # Start the recorder
        # 🧠 ML Signal: Retrieving a recorder by name before deletion.
        self.active_recorder.start_run()

        # 🧠 ML Signal: Usage of a client method to delete a run by recorder object.
        return self.active_recorder

    # ⚠️ SAST Risk (Low): Catching broad exceptions can mask other issues.
    def end(self, recorder_status=Recorder.STATUS_S):
        if self.active_recorder is not None:
            self.active_recorder.end_run(recorder_status)
            self.active_recorder = None

    # ✅ Best Practice: Constants should be defined in uppercase to distinguish them from variables.
    def create_recorder(self, recorder_name=None):
        if recorder_name is None:
            # ✅ Best Practice: Docstring provides clear documentation for the function and its parameters
            recorder_name = self._default_rec_name
        recorder = MLflowRecorder(self.id, self._uri, recorder_name)

        return recorder

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Method for getting or creating a recorder. It will try to first get a valid recorder, if exception occurs, it will
        raise errors.

        Quoting docs of search_runs from MLflow
        > The default ordering is to sort by start_time DESC, then run_id.
        """
        # 🧠 ML Signal: Usage of search_runs method indicates interaction with MLflow for experiment tracking
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input at least one of recorder id or name before retrieving recorder."
        if recorder_id is not None:
            try:
                run = self._client.get_run(recorder_id)
                recorder = MLflowRecorder(self.id, self._uri, mlflow_run=run)
                return recorder
            # 🧠 ML Signal: Instantiation of MLflowRecorder suggests a pattern of wrapping MLflow runs
            except MlflowException as mlflow_exp:
                raise ValueError(
                    "No valid recorder has been found, please make sure the input recorder id is correct."
                ) from mlflow_exp
        elif recorder_name is not None:
            # ✅ Best Practice: Using dict and zip to create a dictionary from two lists
            logger.warning(
                f"Please make sure the recorder name {recorder_name} is unique, we will only return the latest recorder if there exist several matched the given name."
                # ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported types is safe but could be more informative
            )
            recorders = self.list_recorders()
            for rid in recorders:
                if recorders[rid].name == recorder_name:
                    return recorders[rid]
            raise ValueError(
                "No valid recorder has been found, please make sure the input recorder name is correct."
            )

    def search_records(self, **kwargs):
        filter_string = (
            "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        )
        run_view_type = (
            1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        )
        max_results = (
            100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        )
        order_by = kwargs.get("order_by")

        return self._client.search_runs(
            [self.id], filter_string, run_view_type, max_results, order_by
        )

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input a valid recorder id or name before deleting."
        try:
            if recorder_id is not None:
                self._client.delete_run(recorder_id)
            else:
                recorder = self._get_recorder(recorder_name=recorder_name)
                self._client.delete_run(recorder.id)
        except MlflowException as e:
            raise ValueError(
                f"Error: {e}. Something went wrong when deleting recorder. Please check if the name/id of the recorder is correct."
            ) from e

    UNLIMITED = 50000  # FIXME: Mlflow can only list 50000 records at most!!!!!!!

    def list_recorders(
        self,
        rtype: Literal["dict", "list"] = Experiment.RT_D,
        max_results: int = UNLIMITED,
        status: Union[str, None] = None,
        filter_string: str = "",
    ):
        """
        Quoting docs of search_runs
        > The default ordering is to sort by start_time DESC, then run_id.

        Parameters
        ----------
        max_results : int
            the number limitation of the results'
        status : str
            the criteria based on status to filter results.
            `None` indicates no filtering.
        filter_string : str
            mlflow supported filter string like 'params."my_param"="a" and tags."my_tag"="b"', use this will help to reduce too much run number.
        """
        runs = self._client.search_runs(
            self.id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results,
            filter_string=filter_string,
        )
        rids = []
        recorders = []
        for i, n in enumerate(runs):
            recorder = MLflowRecorder(self.id, self._uri, mlflow_run=n)
            if status is None or recorder.status == status:
                rids.append(n.info.run_id)
                recorders.append(recorder)

        if rtype == Experiment.RT_D:
            return dict(zip(rids, recorders))
        elif rtype == Experiment.RT_L:
            return recorders
        else:
            raise NotImplementedError("This type of input is not supported")
