# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the TaskManager class.
# 🧠 ML Signal: Usage of MongoDB for storing tasks can be a feature for ML models.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TaskManager can fetch unused tasks automatically and manage the lifecycle of a set of tasks with error handling.
These features can run tasks concurrently and ensure every task will be used only once.
Task Manager will store all tasks in `MongoDB <https://www.mongodb.com/>`_.
Users **MUST** finished the configuration of `MongoDB <https://www.mongodb.com/>`_ when using this module.

A task in TaskManager consists of 3 parts
- tasks description: the desc will define the task
- tasks status: the status of the task
- tasks result: A user can get the task with the task description and task result.
"""
import concurrent
import pickle
import time
from contextlib import contextmanager
from typing import Callable, List
# ⚠️ SAST Risk (Low): Potential risk of inserting unvalidated data into MongoDB.

import fire
# 🧠 ML Signal: Tracking task addition can be useful for ML models to understand task creation patterns.
import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId
# 🧠 ML Signal: Class definition with constants indicating task statuses
# ⚠️ SAST Risk (Low): Catching specific exceptions helps in understanding failure points.
from pymongo.errors import InvalidDocument
# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the method.
# ✅ Best Practice: Constants for task statuses improve code readability and maintainability
# ✅ Best Practice: Docstring provides detailed usage and assumptions, aiding understanding
# ⚠️ SAST Risk (Medium): Use of pickle for serialization can lead to arbitrary code execution if data is untrusted
# ⚠️ SAST Risk (Low): Fetching tasks without filtering might lead to processing unwanted tasks.
# 🧠 ML Signal: Fetching tasks can be a feature for ML models to understand task processing patterns.
# ⚠️ SAST Risk (Low): Updating tasks without validation might lead to inconsistent data states.
from qlib import auto_init, get_module_logger
from tqdm.cli import tqdm

from .utils import get_mongodb
from ...config import C


class TaskManager:
    """
    TaskManager

    Here is what will a task looks like when it created by TaskManager

    .. code-block:: python

        {
            'def': pickle serialized task definition.  using pickle will make it easier
            'filter': json-like data. This is for filtering the tasks.
            'status': 'waiting' | 'running' | 'done'
            'res': pickle serialized task result,
        }

    The tasks manager assumes that you will only update the tasks you fetched.
    The mongo fetch one and update will make it date updating secure.

    This class can be used as a tool from commandline. Here are several examples.
    You can view the help of manage module with the following commands:
    python -m qlib.workflow.task.manage -h # show manual of manage module CLI
    python -m qlib.workflow.task.manage wait -h # show manual of the wait command of manage

    .. code-block:: shell

        python -m qlib.workflow.task.manage -t <pool_name> wait
        python -m qlib.workflow.task.manage -t <pool_name> task_stat


    .. note::

        Assumption: the data in MongoDB was encoded and the data out of MongoDB was decoded

    Here are four status which are:

        STATUS_WAITING: waiting for training

        STATUS_RUNNING: training

        STATUS_PART_DONE: finished some step and waiting for next step

        STATUS_DONE: all work done
    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    # ⚠️ SAST Risk (Medium): Potential exposure of database structure through collection names
    STATUS_DONE = "done"
    # 🧠 ML Signal: Iterating over a list of prefixes to modify dictionary keys
    STATUS_PART_DONE = "part_done"

    # 🧠 ML Signal: Iterating over dictionary keys to find matches
    ENCODE_FIELDS_PREFIX = ["def", "res"]

    # 🧠 ML Signal: Checking if a string starts with a specific prefix
    def __init__(self, task_pool: str):
        """
        Init Task Manager, remember to make the statement of MongoDB url and database name firstly.
        A TaskManager instance serves a specific task pool.
        The static method of this module serves the whole MongoDB.

        Parameters
        ----------
        task_pool: str
            the name of Collection in MongoDB
        """
        self.task_pool: pymongo.collection.Collection = getattr(get_mongodb(), task_pool)
        self.logger = get_module_logger(self.__class__.__name__)
        self.logger.info(f"task_pool:{task_pool}")
    # ✅ Best Practice: Use of self.ENCODE_FIELDS_PREFIX suggests this is a class method, which is a good practice for encapsulation.

    @staticmethod
    # ✅ Best Practice: Using list(task.keys()) to avoid RuntimeError due to dictionary size change during iteration.
    def list() -> list:
        """
        List the all collection(task_pool) of the db.

        Returns:
            list
        """
        return get_mongodb().list_collection_names()

    def _encode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(pickle.dumps(task[k], protocol=C.dump_protocol_version))
        # ✅ Best Practice: Check if "_id" is in query to avoid unnecessary processing.
        return task

    # ✅ Best Practice: Check if query["_id"] is a dictionary to handle different query structures.
    def _decode_task(self, task):
        """
        _decode_task is Serialization tool.
        Mongodb needs JSON, so it needs to convert Python objects into JSON objects through pickle

        Parameters
        ----------
        task : dict
            task information

        Returns
        -------
        dict
            JSON required by mongodb
        # 🧠 ML Signal: Use of a query with ObjectId, indicating a pattern of database operations
        """
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                # 🧠 ML Signal: Use of replace_one, indicating a pattern of database update operations
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task
    # ⚠️ SAST Risk (Low): Potential data integrity issue if task["filter"] is not expected to be a string
    # 🧠 ML Signal: Retry logic after exception, indicating a pattern of error handling

    def _dict_to_str(self, flt):
        return {k: str(v) for k, v in flt.items()}

    def _decode_query(self, query):
        """
        If the query includes any `_id`, then it needs `ObjectId` to decode.
        For example, when using TrainerRM, it needs query `{"_id": {"$in": _id_list}}`. Then we need to `ObjectId` every `_id` in `_id_list`.

        Args:
            query (dict): query dict. Defaults to {}.

        Returns:
            dict: the query after decoding.
        # ✅ Best Practice: Ensure the function returns a consistent type, as documented.
        """
        if "_id" in query:
            if isinstance(query["_id"], dict):
                for key in query["_id"]:
                    query["_id"][key] = [ObjectId(i) for i in query["_id"][key]]
            else:
                query["_id"] = ObjectId(query["_id"])
        return query

    def replace_task(self, task, new_task):
        """
        Use a new task to replace a old one

        Args:
            task: old task
            new_task: new task
        """
        new_task = self._encode_task(new_task)
        # 🧠 ML Signal: Usage of a method to insert tasks into a database
        query = {"_id": ObjectId(task["_id"])}
        try:
            self.task_pool.replace_one(query, new_task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            self.task_pool.replace_one(query, new_task)

    def insert_task(self, task):
        """
        Insert a task.

        Args:
            task: the task waiting for insert

        Returns:
            pymongo.results.InsertOneResult
        """
        try:
            insert_result = self.task_pool.insert_one(task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            insert_result = self.task_pool.insert_one(task)
        return insert_result

    # ⚠️ SAST Risk (Medium): Potentially unsafe conversion of a dictionary to a string
    def insert_task_def(self, task_def):
        """
        Insert a task to task_pool

        Parameters
        ----------
        task_def: dict
            the task definition

        Returns
        -------
        pymongo.results.InsertOneResult
        # ✅ Best Practice: Use logging instead of print for better control over output
        """
        task = self._encode_task(
            {
                # 🧠 ML Signal: Conditional logic based on a flag (print_nt)
                "def": task_def,
                "filter": task_def,  # FIXME: catch the raised error
                "status": self.STATUS_WAITING,
            # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior.
            }
        )
        insert_result = self.insert_task(task)
        return insert_result

    def create_task(self, task_def_l, dry_run=False, print_nt=False) -> List[str]:
        """
        If the tasks in task_def_l are new, then insert new tasks into the task_pool, and record inserted_id.
        If a task is not new, then just query its _id.

        Parameters
        ----------
        task_def_l: list
            a list of task
        dry_run: bool
            if insert those new tasks to task pool
        print_nt: bool
            if print new task

        Returns
        -------
        List[str]
            a list of the _id of task_def_l
        """
        new_tasks = []
        _id_list = []
        for t in task_def_l:
            try:
                r = self.task_pool.find_one({"filter": t})
            except InvalidDocument:
                r = self.task_pool.find_one({"filter": self._dict_to_str(t)})
            # 🧠 ML Signal: Calls a method to fetch a task, indicating a pattern of task retrieval.
            # When r is none, it indicates that r s a new task
            if r is None:
                new_tasks.append(t)
                # 🧠 ML Signal: Use of yield indicates a generator pattern, which can be a signal for asynchronous processing.
                if not dry_run:
                    insert_result = self.insert_task_def(t)
                    _id_list.append(insert_result.inserted_id)
                # ✅ Best Practice: Checking if task is not None before proceeding ensures robustness.
                else:
                    _id_list.append(None)
            # 🧠 ML Signal: Logging information before returning a task, useful for tracking task lifecycle.
            else:
                # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior if modified.
                _id_list.append(self._decode_task(r)["_id"])
        # 🧠 ML Signal: Calls a method to return a task, indicating a pattern of task management.

        # ✅ Best Practice: Using a context manager to handle resources safely.
        self.logger.info(f"Total Tasks: {len(task_def_l)}, New Tasks: {len(new_tasks)}")
        # 🧠 ML Signal: Logging after returning a task, useful for tracking task lifecycle.

        if print_nt:  # print new task
            # ✅ Best Practice: Re-raising the exception ensures that the error is not silently ignored.
            for t in new_tasks:
                # 🧠 ML Signal: Use of generator pattern to yield tasks one by one.
                # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior if modified.
                print(t)

        if dry_run:
            return []

        return _id_list

    def fetch_task(self, query={}, status=STATUS_WAITING) -> dict:
        """
        Use query to fetch tasks.

        Args:
            query (dict, optional): query dict. Defaults to {}.
            status (str, optional): [description]. Defaults to STATUS_WAITING.

        Returns:
            dict: a task(document in collection) after decoding
        """
        # 🧠 ML Signal: Iterating over a database cursor is a common pattern in data retrieval tasks.
        # 🧠 ML Signal: Decoding tasks after retrieval could indicate a pattern of data post-processing.
        query = query.copy()
        query = self._decode_query(query)
        query.update({"status": status})
        task = self.task_pool.find_one_and_update(
            query, {"$set": {"status": self.STATUS_RUNNING}}, sort=[("priority", pymongo.DESCENDING)]
        )
        # null will be at the top after sorting when using ASCENDING, so the larger the number higher, the higher the priority
        # ⚠️ SAST Risk (Medium): Potential for NoSQL injection if _id is not properly validated
        if task is None:
            return None
        # 🧠 ML Signal: Pattern of decoding a task after retrieval from a database
        task["status"] = self.STATUS_RUNNING
        return self._decode_task(task)

    @contextmanager
    def safe_fetch_task(self, query={}, status=STATUS_WAITING):
        """
        Fetch task from task_pool using query with contextmanager

        Parameters
        ----------
        query: dict
            the dict of query

        Returns
        -------
        dict: a task(document in collection) after decoding
        """
        task = self.fetch_task(query=query, status=status)
        try:
            yield task
        except (Exception, KeyboardInterrupt):  # KeyboardInterrupt is not a subclass of Exception
            # ✅ Best Practice: Check for None to avoid overwriting valid status values
            if task is not None:
                self.logger.info("Returning task before raising error")
                self.return_task(task, status=status)  # return task as the original status
                self.logger.info("Task returned")
            # ⚠️ SAST Risk (Low): Potential risk if task["_id"] is not validated or sanitized
            raise
    # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior.

    def task_fetcher_iter(self, query={}):
        while True:
            with self.safe_fetch_task(query=query) as task:
                if task is None:
                    break
                yield task

    def query(self, query={}, decode=True):
        """
        Query task in collection.
        This function may raise exception `pymongo.errors.CursorNotFound: cursor id not found` if it takes too long to iterate the generator

        python -m qlib.workflow.task.manage -t <your task pool> query '{"_id": "615498be837d0053acbc5d58"}'

        Parameters
        ----------
        query: dict
            the dict of query
        decode: bool

        Returns
        -------
        dict: a task(document in collection) after decoding
        """
        # 🧠 ML Signal: Querying a database or data source is a common pattern in data processing.
        # ✅ Best Practice: Use of a default mutable argument (dictionary) can lead to unexpected behavior.
        query = query.copy()
        # 🧠 ML Signal: Iterating over a collection to aggregate or count items is a common pattern.
        query = self._decode_query(query)
        for t in self.task_pool.find(query):
            yield self._decode_task(t)

    def re_query(self, _id) -> dict:
        """
        Use _id to query task.

        Args:
            _id (str): _id of a document

        Returns:
            dict: a task(document in collection) after decoding
        # ⚠️ SAST Risk (Low): Printing database operation results can expose sensitive information in logs.
        """
        t = self.task_pool.find_one({"_id": ObjectId(_id)})
        return self._decode_task(t)

    def commit_task_res(self, task, res, status=STATUS_DONE):
        """
        Commit the result to task['res'].

        Args:
            task ([type]): [description]
            res (object): the result you want to save
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE. Defaults to STATUS_DONE.
        # ⚠️ SAST Risk (Medium): Potential risk of NoSQL injection if task["_id"] is not properly validated
        # ✅ Best Practice: Use of a private method to encapsulate functionality
        """
        # 🧠 ML Signal: Use of dictionary get method with default values
        # 🧠 ML Signal: Usage of MongoDB update pattern
        # A workaround to use the class attribute.
        if status is None:
            status = TaskManager.STATUS_DONE
        self.task_pool.update_one(
            {"_id": task["_id"]},
            # ✅ Best Practice: Use of a private method indicates encapsulation and controlled access
            {"$set": {"status": status, "res": Binary(pickle.dumps(res, protocol=C.dump_protocol_version))}},
        )
    # 🧠 ML Signal: Use of sum() function to aggregate values

    # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior if modified.
    def return_task(self, task, status=STATUS_WAITING):
        """
        Return a task to status. Always using in error handling.

        Args:
            task ([type]): [description]
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE. Defaults to STATUS_WAITING.
        """
        if status is None:
            status = TaskManager.STATUS_WAITING
        update_dict = {"$set": {"status": status}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    # 🧠 ML Signal: Usage of tqdm for progress tracking can be a signal for training models on task completion times.
    def remove(self, query={}):
        """
        Remove the task using query

        Parameters
        ----------
        query: dict
            the dict of query

        # 🧠 ML Signal: Use of f-string for string formatting
        """
        query = query.copy()
        query = self._decode_query(query)
        self.task_pool.delete_many(query)

    def task_stat(self, query={}) -> dict:
        """
        Count the tasks in every status.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        Returns:
            dict
        """
        query = query.copy()
        query = self._decode_query(query)
        tasks = self.query(query=query, decode=False)
        status_stat = {}
        for t in tasks:
            status_stat[t["status"]] = status_stat.get(t["status"], 0) + 1
        return status_stat

    def reset_waiting(self, query={}):
        """
        Reset all running task into waiting status. Can be used when some running task exit unexpected.

        Args:
            query (dict, optional): the query dict. Defaults to {}.
        """
        query = query.copy()
        # default query
        if "status" not in query:
            query["status"] = self.STATUS_RUNNING
        return self.reset_status(query=query, status=self.STATUS_WAITING)
    # 🧠 ML Signal: Use of TaskManager to manage task states

    def reset_status(self, query, status):
        query = query.copy()
        query = self._decode_query(query)
        # 🧠 ML Signal: Use of context manager for safe task fetching
        print(self.task_pool.update_many(query, {"$set": {"status": status}}))

    def prioritize(self, task, priority: int):
        """
        Set priority for task

        Parameters
        ----------
        task : dict
            The task query from the database
        priority : int
            the target priority
        """
        update_dict = {"$set": {"priority": priority}}
        # ⚠️ SAST Risk (Medium): Use of ProcessPoolExecutor can lead to resource exhaustion if not managed properly
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def _get_undone_n(self, task_stat):
        return (
            # 🧠 ML Signal: Use of fire for command-line interface
            # 🧠 ML Signal: Committing task results to TaskManager
            # 🧠 ML Signal: Initialization of the application
            task_stat.get(self.STATUS_WAITING, 0)
            + task_stat.get(self.STATUS_RUNNING, 0)
            + task_stat.get(self.STATUS_PART_DONE, 0)
        )

    def _get_total(self, task_stat):
        return sum(task_stat.values())

    def wait(self, query={}):
        """
        When multiprocessing, the main progress may fetch nothing from TaskManager because there are still some running tasks.
        So main progress should wait until all tasks are trained well by other progress or machines.

        Args:
            query (dict, optional): the query dict. Defaults to {}.
        """
        task_stat = self.task_stat(query)
        total = self._get_total(task_stat)
        last_undone_n = self._get_undone_n(task_stat)
        if last_undone_n == 0:
            return
        self.logger.warning(f"Waiting for {last_undone_n} undone tasks. Please make sure they are running.")
        with tqdm(total=total, initial=total - last_undone_n) as pbar:
            while True:
                time.sleep(10)
                undone_n = self._get_undone_n(self.task_stat(query))
                pbar.update(last_undone_n - undone_n)
                last_undone_n = undone_n
                if undone_n == 0:
                    break

    def __str__(self):
        return f"TaskManager({self.task_pool})"


def run_task(
    task_func: Callable,
    task_pool: str,
    query: dict = {},
    force_release: bool = False,
    before_status: str = TaskManager.STATUS_WAITING,
    after_status: str = TaskManager.STATUS_DONE,
    **kwargs,
):
    r"""
    While the task pool is not empty (has WAITING tasks), use task_func to fetch and run tasks in task_pool

    After running this method, here are 4 situations (before_status -> after_status):

        STATUS_WAITING -> STATUS_DONE: use task["def"] as `task_func` param, it means that the task has not been started

        STATUS_WAITING -> STATUS_PART_DONE: use task["def"] as `task_func` param

        STATUS_PART_DONE -> STATUS_PART_DONE: use task["res"] as `task_func` param, it means that the task has been started but not completed

        STATUS_PART_DONE -> STATUS_DONE: use task["res"] as `task_func` param

    Parameters
    ----------
    task_func : Callable
        def (task_def, \**kwargs) -> <res which will be committed>

        the function to run the task
    task_pool : str
        the name of the task pool (Collection in MongoDB)
    query: dict
        will use this dict to query task_pool when fetching task
    force_release : bool
        will the program force to release the resource
    before_status : str:
        the tasks in before_status will be fetched and trained. Can be STATUS_WAITING, STATUS_PART_DONE.
    after_status : str:
        the tasks after trained will become after_status. Can be STATUS_WAITING, STATUS_PART_DONE.
    kwargs
        the params for `task_func`
    """
    tm = TaskManager(task_pool)

    ever_run = False

    while True:
        with tm.safe_fetch_task(status=before_status, query=query) as task:
            if task is None:
                break
            get_module_logger("run_task").info(task["def"])
            # when fetching `WAITING` task, use task["def"] to train
            if before_status == TaskManager.STATUS_WAITING:
                param = task["def"]
            # when fetching `PART_DONE` task, use task["res"] to train because the middle result has been saved to task["res"]
            elif before_status == TaskManager.STATUS_PART_DONE:
                param = task["res"]
            else:
                raise ValueError("The fetched task must be `STATUS_WAITING` or `STATUS_PART_DONE`!")
            if force_release:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    res = executor.submit(task_func, param, **kwargs).result()
            else:
                res = task_func(param, **kwargs)
            tm.commit_task_res(task, res, status=after_status)
            ever_run = True

    return ever_run


if __name__ == "__main__":
    # This is for using it in cmd
    # E.g. : `python -m qlib.workflow.task.manage list`
    auto_init()
    fire.Fire(TaskManager)