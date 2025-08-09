# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
The Trainer will train a list of tasks and return a list of model recorders.
There are two steps in each Trainer including ``train`` (make model recorder) and ``end_train`` (modify model recorder).

This is a concept called ``DelayTrainer``, which can be used in online simulating for parallel training.
In ``DelayTrainer``, the first step is only to save some necessary info to model recorders, and the second step which will be finished in the end can do some concurrent and time-consuming operations such as model fitting.

``Qlib`` offer two kinds of Trainer, ``TrainerR`` is the simplest way and ``TrainerRM`` is based on TaskManager to help manager tasks lifecycle automatically.
"""

import socket
from typing import Callable, List, Optional

# ✅ Best Practice: Using a logger instead of print statements for logging is a good practice.

from tqdm.auto import tqdm

from qlib.config import C
from qlib.data.dataset import Dataset
from qlib.data.dataset.weight import Reweighter
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import (
    # ⚠️ SAST Risk (Medium): Using subprocess calls can lead to security risks if inputs are not properly sanitized.
    auto_filter_kwargs,
    # 🧠 ML Signal: Function definition with a single responsibility for logging task information
    fill_placeholder,
    flatten_dict,
    # 🧠 ML Signal: Usage of a logging function with dynamic parameters
    init_instance_by_config,
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information if task_config contains sensitive data
)

# 🧠 ML Signal: Function definition with a dictionary parameter, common in ML configurations
from qlib.utils.paral import call_in_subproc

# 🧠 ML Signal: Usage of a function to save objects with dynamic parameters
from qlib.workflow import R

# ⚠️ SAST Risk (Low): Potential exposure of sensitive information if task_config contains sensitive data
# 🧠 ML Signal: Usage of a recorder, often used for logging or tracking experiments
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.manage import TaskManager, run_task

# 🧠 ML Signal: Initialization of model instance from configuration
# 🧠 ML Signal: Usage of a function to set tags with dynamic parameters

# ⚠️ SAST Risk (Low): Potential exposure of sensitive information if hostname is sensitive
# ✅ Best Practice: Type hinting for better readability and maintainability


def _log_task_info(task_config: dict):
    # 🧠 ML Signal: Initialization of dataset instance from configuration
    R.log_params(**flatten_dict(task_config))
    # ✅ Best Practice: Type hinting for better readability and maintainability
    R.save_objects(**{"task": task_config})  # keep the original format and datatype
    R.set_tags(**{"hostname": socket.gethostname()})


# 🧠 ML Signal: Optional configuration parameter for reweighter


# 🧠 ML Signal: Dynamic function call with auto-filtering of keyword arguments
def _exe_task(task_config: dict):
    rec = R.get_recorder()
    # 🧠 ML Signal: Saving model parameters, common in ML workflows
    # model & dataset initialization
    # 🧠 ML Signal: Saving dataset state, common in ML workflows
    # 🧠 ML Signal: Dataset configuration, often used in data preprocessing
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(
        task_config["dataset"], accept_types=Dataset
    )
    reweighter: Reweighter = task_config.get("reweighter", None)
    # model training
    auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
    R.save_objects(**{"params.pkl": model})
    # ✅ Best Practice: Use of a dictionary for placeholder values improves code readability
    # this dataset is saved for online inference. So the concrete data should not be dumped
    # 🧠 ML Signal: Placeholder filling in configuration, common in templated configurations
    dataset.config(dump_all=False, recursive=True)
    # 🧠 ML Signal: Handling of record configurations, common in experiment tracking
    # ✅ Best Practice: Ensures records is always a list, simplifying subsequent processing
    R.save_objects(**{"dataset": dataset})
    # fill placehorder
    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    # generate records: prediction, backtest, and analysis
    records = task_config.get("record", [])
    if isinstance(records, dict):  # prevent only one dict
        records = [records]
    for record in records:
        # 🧠 ML Signal: Iterating over records to initialize and generate them
        # 🧠 ML Signal: Dynamic initialization of record instances
        # ✅ Best Practice: Using a context manager ensures that resources are properly managed and released.
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # 🧠 ML Signal: Logging task configuration can be useful for tracking and reproducing experiments.
        # to make defining the tasking easier
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
        r = init_instance_by_config(
            # 🧠 ML Signal: Record generation, often used in logging or tracking
            # 🧠 ML Signal: Returning a recorder object is a common pattern in ML for tracking experiments.
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def begin_task_train(
    task_config: dict, experiment_name: str, recorder_name: str = None
) -> Recorder:
    """
    Begin task training to start a recorder and save the task config.

    Args:
        task_config (dict): the config of a task
        experiment_name (str): the name of experiment
        recorder_name (str): the given name will be the recorder name. None for using rid.

    Returns:
        Recorder: the model recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        return R.get_recorder()


def end_task_train(rec: Recorder, experiment_name: str) -> Recorder:
    """
    Finish task training with real model fitting and saving.

    Args:
        rec (Recorder): the recorder will be resumed
        experiment_name (str): the name of experiment

    Returns:
        Recorder: the model recorder
    """
    with R.start(
        experiment_name=experiment_name, recorder_id=rec.info["id"], resume=True
    ):
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        task_config = R.load_object("task")
        _exe_task(task_config)
    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    return rec


def task_train(
    task_config: dict, experiment_name: str, recorder_name: str = None
) -> Recorder:
    """
    Task based training, will be divided into two steps.

    Parameters
    ----------
    task_config : dict
        The config of a task.
    experiment_name: str
        The name of experiment
    recorder_name: str
        The name of recorder

    Returns
    ----------
    Recorder: The instance of the recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        _exe_task(task_config)
        # 🧠 ML Signal: Returning the models list directly may indicate a pattern where models are processed or modified in place.
        return R.get_recorder()


class Trainer:
    """
    The trainer can train a list of models.
    There are Trainer and DelayTrainer, which can be distinguished by when it will finish real training.
    """

    # 🧠 ML Signal: Use of *args and **kwargs indicates a flexible function signature

    # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and return value
    # 🧠 ML Signal: Chaining method calls is a common pattern in functional programming
    def __init__(self):
        self.delay = False

    def train(self, tasks: list, *args, **kwargs) -> list:
        """
        Given a list of task definitions, begin training, and return the models.

        For Trainer, it finishes real training in this method.
        For DelayTrainer, it only does some preparation in this method.

        Args:
            tasks: a list of tasks

        Returns:
            list: a list of models
        """
        raise NotImplementedError("Please implement the `train` method.")

    # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which is expected for abstract methods but should be implemented in subclasses
    # 🧠 ML Signal: Class definition for a trainer, indicating a pattern for training models
    def end_train(self, models: list, *args, **kwargs) -> list:
        """
        Given a list of models, finished something at the end of training if you need.
        The models may be Recorder, txt file, database, and so on.

        For Trainer, it does some finishing touches in this method.
        For DelayTrainer, it finishes real training in this method.

        Args:
            models: a list of models

        Returns:
            list: a list of models
        """
        # do nothing if you finished all work in `train` method
        return models

    def is_delay(self) -> bool:
        """
        If Trainer will delay finishing `end_train`.

        Returns:
            bool: if DelayTrainer
        # ✅ Best Practice: Always call the superclass's __init__ method to ensure proper initialization.
        """
        return self.delay

    # 🧠 ML Signal: Storing experiment name, which could be used for tracking or logging experiments.

    def __call__(self, *args, **kwargs) -> list:
        # 🧠 ML Signal: Storing default record name, which could be used for logging or saving results.
        return self.end_train(self.train(*args, **kwargs))

    # 🧠 ML Signal: Storing a training function, indicating a customizable training process.

    def has_worker(self) -> bool:
        """
        Some trainer has backend worker to support parallel training
        This method can tell if the worker is enabled.

        Returns
        -------
        bool:
            if the worker is enabled

        """
        return False

    # ⚠️ SAST Risk (Low): Potential type confusion if tasks is not a list or dict

    def worker(self):
        """
        start the worker

        Raises
        ------
        NotImplementedError:
            If the worker is not supported
        """
        raise NotImplementedError("Please implement the `worker` method")


# 🧠 ML Signal: Iterating over tasks with a progress bar


class TrainerR(Trainer):
    """
    Trainer based on (R)ecorder.
    It will train a list of tasks and return a list of model recorders in a linear way.

    Assumption: models were defined by `task` and the results will be saved to `Recorder`.
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"
    # ⚠️ SAST Risk (Low): Potential type confusion if models is not a list or Recorder

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        # ⚠️ SAST Risk (Low): Assumes all elements in models are Recorder instances
        train_func: Callable = task_train,
        # 🧠 ML Signal: Iterating over a list of objects to set a common attribute
        call_in_subproc: bool = False,
        # 🧠 ML Signal: Returning the same list that was passed as an argument
        default_rec_name: Optional[str] = None,
    ):
        """
        Init TrainerR.

        Args:
            experiment_name (str, optional): the default name of experiment.
            train_func (Callable, optional): default training method. Defaults to `task_train`.
            call_in_subproc (bool): call the process in subprocess to force memory release
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.default_rec_name = default_rec_name
        self.train_func = train_func
        self._call_in_subproc = call_in_subproc

    # ✅ Best Practice: Calling `super().__init__()` ensures proper initialization of the base class.

    def train(
        # ✅ Best Practice: Storing `end_train_func` as an instance variable allows for flexible method overriding.
        # 🧠 ML Signal: The `delay` attribute could be used to control or signal asynchronous behavior in training.
        self,
        tasks: list,
        train_func: Optional[Callable] = None,
        experiment_name: Optional[str] = None,
        **kwargs,
    ) -> List[Recorder]:
        """
        Given a list of `tasks` and return a list of trained Recorder. The order can be guaranteed.

        Args:
            tasks (list): a list of definitions based on `task` dict
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        # ✅ Best Practice: Use default end_train_func if none is provided
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            # ✅ Best Practice: Use default experiment_name if none is provided
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            # ✅ Best Practice: Skip processing for models already marked as ended
            experiment_name = self.experiment_name
        recs = []
        # ✅ Best Practice: Class docstring provides a clear description of the class functionality and assumptions
        for task in tqdm(tasks, desc="train tasks"):
            # 🧠 ML Signal: Usage of a custom training function with additional parameters
            # 🧠 ML Signal: Setting a status tag to indicate the end of training
            if self._call_in_subproc:
                get_module_logger("TrainerR").info(
                    "running models in sub process (for forcing release memroy)."
                )
                train_func = call_in_subproc(train_func, C)
            rec = train_func(
                task, experiment_name, recorder_name=self.default_rec_name, **kwargs
            )
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            recs.append(rec)
        # ✅ Best Practice: Constants are defined with clear and descriptive names
        return recs

    def end_train(self, models: list, **kwargs) -> List[Recorder]:
        """
        Set STATUS_END tag to the recorders.

        Args:
            models (list): a list of trained recorders.

        Returns:
            List[Recorder]: the same list as the param.
        """
        if isinstance(models, Recorder):
            models = [models]
        for rec in models:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class DelayTrainerR(TrainerR):
    """
    A delayed implementation based on TrainerR, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.
    """

    # 🧠 ML Signal: Storing experiment name, which could be used for tracking or logging ML experiments.

    def __init__(
        # 🧠 ML Signal: Storing task pool name, which could be used for managing or categorizing tasks.
        self,
        experiment_name: str = None,
        train_func=begin_task_train,
        end_train_func=end_task_train,
        **kwargs,
    ):
        """
        Init TrainerRM.

        Args:
            experiment_name (str): the default name of experiment.
            train_func (Callable, optional): default train method. Defaults to `begin_task_train`.
            end_train_func (Callable, optional): default end_train method. Defaults to `end_task_train`.
        """
        super().__init__(experiment_name, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True

    def end_train(
        self, models, end_train_func=None, experiment_name: str = None, **kwargs
    ) -> List[Recorder]:
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finish real data loading and model fitting.

        Args:
            models (list): a list of Recorder, the tasks have been saved to them
            end_train_func (Callable, optional): the end_train method which needs at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for end_train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(models, Recorder):
            models = [models]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        for rec in models:
            if rec.list_tags()[self.STATUS_KEY] == self.STATUS_END:
                continue
            end_train_func(rec, experiment_name, **kwargs)
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class TrainerRM(Trainer):
    """
    Trainer based on (R)ecorder and Task(M)anager.
    It can train a list of tasks and return a list of model recorders in a multiprocessing way.

    Assumption: `task` will be saved to TaskManager and `task` will be fetched and trained from TaskManager
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    # This tag is the _id in TaskManager to distinguish tasks.
    TM_ID = "_id in TaskManager"

    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        train_func=task_train,
        skip_run_task: bool = False,
        default_rec_name: Optional[str] = None,
    ):
        """
        Init TrainerR.

        Args:
            experiment_name (str): the default name of experiment.
            task_pool (str): task pool name in TaskManager. None for use same name as experiment_name.
            train_func (Callable, optional): default training method. Defaults to `task_train`.
            skip_run_task (bool):
                If skip_run_task == True:
                Only run_task in the worker. Otherwise skip run_task.
        """

        super().__init__()
        self.experiment_name = experiment_name
        self.task_pool = task_pool
        self.train_func = train_func
        self.skip_run_task = skip_run_task
        self.default_rec_name = default_rec_name

    def train(
        # ✅ Best Practice: Use of default values for function parameters to allow flexibility in function calls.
        self,
        tasks: list,
        train_func: Callable = None,
        # ✅ Best Practice: Use of default values for function parameters to allow flexibility in function calls.
        experiment_name: str = None,
        before_status: str = TaskManager.STATUS_WAITING,
        after_status: str = TaskManager.STATUS_DONE,
        default_rec_name: Optional[str] = None,
        # ✅ Best Practice: Fallback to a default value if task_pool is None, ensuring task_pool is always set.
        **kwargs,
        # ✅ Best Practice: Method signature includes type hinting for return type
    ) -> List[Recorder]:
        """
        Given a list of `tasks` and return a list of trained Recorder. The order can be guaranteed.

        This method defaults to a single process, but TaskManager offered a great way to parallel training.
        Users can customize their train_func to realize multiple processes or even multiple machines.

        Args:
            tasks (list): a list of definitions based on `task` dict
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
            before_status (str): the tasks in before_status will be fetched and trained. Can be STATUS_WAITING, STATUS_PART_DONE.
            after_status (str): the tasks after trained will become after_status. Can be STATUS_WAITING, STATUS_PART_DONE.
            kwargs: the params for train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        if default_rec_name is None:
            # ✅ Best Practice: Call to superclass's __init__ ensures proper initialization of inherited attributes.
            default_rec_name = self.default_rec_name
        task_pool = self.task_pool
        # ✅ Best Practice: Storing end_train_func as an instance variable for later use.
        if task_pool is None:
            task_pool = experiment_name
        # 🧠 ML Signal: Setting a flag that might be used to control behavior in ML training.
        tm = TaskManager(task_pool=task_pool)
        # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and parameters
        # 🧠 ML Signal: Conditional logic based on skip_run_task could influence task execution flow.
        _id_list = tm.create_task(tasks)  # all tasks will be saved to MongoDB
        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=before_status,
                after_status=after_status,
                recorder_name=default_rec_name,
                # ⚠️ SAST Risk (Low): Implicit conversion of dict to list may lead to unexpected behavior if dict is not intended
                **kwargs,
            )

        # ✅ Best Practice: Early return for empty task list improves readability
        if not self.is_delay():
            tm.wait(query=query)

        recs = []
        for _id in _id_list:
            rec = tm.re_query(_id)["res"]
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            rec.set_tags(**{self.TM_ID: _id})
            recs.append(rec)
        return recs

    # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose and parameters
    def end_train(self, recs: list, **kwargs) -> List[Recorder]:
        """
        Set STATUS_END tag to the recorders.

        Args:
            recs (list): a list of trained recorders.

        Returns:
            List[Recorder]: the same list as the param.
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        for rec in recs:
            # ⚠️ SAST Risk (Low): Potential type confusion if `recs` is not a list or Recorder
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    # 🧠 ML Signal: Default function assignment pattern
    def worker(
        self,
        train_func: Callable = None,
        # 🧠 ML Signal: Default value assignment pattern
        experiment_name: str = None,
    ):
        """
        The multiprocessing method for `train`. It can share a same task_pool with `train` and can run in other progress or other machines.

        Args:
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
        """
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        # ⚠️ SAST Risk (Medium): Potential security risk if `end_train_func` or `kwargs` are user-controlled
        if task_pool is None:
            task_pool = experiment_name
        run_task(train_func, task_pool=task_pool, experiment_name=experiment_name)

    def has_worker(self) -> bool:
        return True


class DelayTrainerRM(TrainerRM):
    """
    A delayed implementation based on TrainerRM, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.

    """

    # ✅ Best Practice: Use of default values for function parameters enhances flexibility and usability.
    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        # ✅ Best Practice: Use of default values for function parameters enhances flexibility and usability.
        # 🧠 ML Signal: Function call with parameters that could be used to track task execution patterns.
        train_func=begin_task_train,
        end_train_func=end_task_train,
        skip_run_task: bool = False,
        **kwargs,
    ):
        """
        Init DelayTrainerRM.

        Args:
            experiment_name (str): the default name of experiment.
            task_pool (str): task pool name in TaskManager. None for use same name as experiment_name.
            train_func (Callable, optional): default train method. Defaults to `begin_task_train`.
            end_train_func (Callable, optional): default end_train method. Defaults to `end_task_train`.
            skip_run_task (bool):
                If skip_run_task == True:
                Only run_task in the worker. Otherwise skip run_task.
                E.g. Starting trainer on a CPU VM and then waiting tasks to be finished on GPU VMs.
        """
        super().__init__(experiment_name, task_pool, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True
        self.skip_run_task = skip_run_task

    def train(
        self, tasks: list, train_func=None, experiment_name: str = None, **kwargs
    ) -> List[Recorder]:
        """
        Same as `train` of TrainerRM, after_status will be STATUS_PART_DONE.

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `tasks` and `experiment_name`. Defaults to None for using self.train_func.
            experiment_name (str): the experiment name, None for use default name.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        _skip_run_task = self.skip_run_task
        self.skip_run_task = False  # The task preparation can't be skipped
        res = super().train(
            tasks,
            train_func=train_func,
            experiment_name=experiment_name,
            after_status=TaskManager.STATUS_PART_DONE,
            **kwargs,
        )
        self.skip_run_task = _skip_run_task
        return res

    def end_train(
        self, recs, end_train_func=None, experiment_name: str = None, **kwargs
    ) -> List[Recorder]:
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finish real data loading and model fitting.

        Args:
            recs (list): a list of Recorder, the tasks have been saved to them.
            end_train_func (Callable, optional): the end_train method which need at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for end_train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        _id_list = []
        for rec in recs:
            _id_list.append(rec.list_tags()[self.TM_ID])

        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                end_train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=TaskManager.STATUS_PART_DONE,
                **kwargs,
            )

        TaskManager(task_pool=task_pool).wait(query=query)

        for rec in recs:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    def worker(self, end_train_func=None, experiment_name: str = None):
        """
        The multiprocessing method for `end_train`. It can share a same task_pool with `end_train` and can run in other progress or other machines.

        Args:
            end_train_func (Callable, optional): the end_train method which need at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
        """
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        run_task(
            end_train_func,
            task_pool=task_pool,
            experiment_name=experiment_name,
            before_status=TaskManager.STATUS_PART_DONE,
        )

    def has_worker(self) -> bool:
        return True
