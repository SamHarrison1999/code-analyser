# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import multiprocessing
from multiprocessing.sharedctypes import Synchronized
import os
import threading

# ✅ Best Practice: Use of a logging library for consistent and configurable logging
import time
import warnings

# 🧠 ML Signal: Logging usage pattern
from queue import Empty
from typing import Any, Generator, Generic, Sequence, TypeVar, cast

# ✅ Best Practice: Use of __all__ to define public API of the module
from qlib.log import get_module_logger

_logger = get_module_logger(__name__)

T = TypeVar("T")

__all__ = ["DataQueue"]


class DataQueue(Generic[T]):
    """Main process (producer) produces data and stores them in a queue.
    Sub-processes (consumers) can retrieve the data-points from the queue.
    Data-points are generated via reading items from ``dataset``.

    :class:`DataQueue` is ephemeral. You must create a new DataQueue
    when the ``repeat`` is exhausted.

    See the documents of :class:`qlib.rl.utils.FiniteVectorEnv` for more background.

    Parameters
    ----------
    dataset
        The dataset to read data from. Must implement ``__len__`` and ``__getitem__``.
    repeat
        Iterate over the data-points for how many times. Use ``-1`` to iterate forever.
    shuffle
        If ``shuffle`` is true, the items will be read in random order.
    producer_num_workers
        Concurrent workers for data-loading.
    queue_maxsize
        Maximum items to put into queue before it jams.

    Examples
    --------
    >>> data_queue = DataQueue(my_dataset)
    >>> with data_queue:
    ...     ...

    In worker:

    >>> for data in data_queue:
    ...     print(data)
    """

    def __init__(
        # 🧠 ML Signal: Logging a warning when CPU count is unavailable.
        self,
        dataset: Sequence[T],
        repeat: int = 1,
        shuffle: bool = True,
        # ✅ Best Practice: Type hinting the return value improves code readability and maintainability
        producer_num_workers: int = 0,
        queue_maxsize: int = 0,
        # 🧠 ML Signal: Method call within a context manager pattern
    ) -> None:
        # ⚠️ SAST Risk (Low): Using multiprocessing.Queue can lead to potential deadlocks if not managed properly.
        # ✅ Best Practice: Implementing __exit__ for context manager support
        if queue_maxsize == 0:
            # 🧠 ML Signal: Returning self in a context manager pattern
            if os.cpu_count() is not None:
                # ✅ Best Practice: Use cast to ensure type correctness for multiprocessing.Value.
                # ✅ Best Practice: Ensuring resources are cleaned up in __exit__
                queue_maxsize = cast(int, os.cpu_count())
                _logger.info(
                    f"Automatically set data queue maxsize to {queue_maxsize} to avoid overwhelming."
                )
            else:
                queue_maxsize = 1
                _logger.warning("CPU count not available. Setting queue maxsize to 1.")
        # ✅ Best Practice: Use of warnings.warn to notify about potential issues

        self.dataset: Sequence[T] = dataset
        self.repeat: int = repeat
        self.shuffle: bool = shuffle
        self.producer_num_workers: int = producer_num_workers

        # ✅ Best Practice: Proper exception handling for Empty exception
        self._activated: bool = False
        self._queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=queue_maxsize
        )
        # 🧠 ML Signal: Use of time.sleep indicates a delay or wait pattern
        # Mypy 0.981 brought '"SynchronizedBase[Any]" has no attribute "value"  [attr-defined]' bug.
        # Therefore, add this type casting to pass Mypy checking.
        self._done = cast(Synchronized, multiprocessing.Value("i", 0))

    # ✅ Best Practice: Check for attribute existence before accessing it

    # 🧠 ML Signal: Use of logging to track the state of the queue
    def __enter__(self) -> DataQueue:
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        # 🧠 ML Signal: Usage of queue with blocking and timeout
        with self._done.get_lock():
            self._done.value += 1
        for repeat in range(500):
            if repeat >= 1:
                # ⚠️ SAST Risk (Low): Potential infinite loop if _done.value is never True
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                warnings.warn(
                    f"After {repeat} cleanup, the queue is still not empty.",
                    category=RuntimeWarning,
                )
            while not self._queue.empty():
                # 🧠 ML Signal: Usage of queue's put method with parameters
                try:
                    # 🧠 ML Signal: Method that modifies shared state, indicating potential concurrency usage
                    self._queue.get(block=False)
                except Empty:
                    # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
                    # ⚠️ SAST Risk (Low): Potential deadlock if not used carefully with other locks
                    pass
            # Sometimes when the queue gets emptied, more data have already been sent,
            # ✅ Best Practice: Explicitly setting a value to indicate a state change
            # 🧠 ML Signal: Accessing an attribute of an object, indicating a common pattern of object-oriented programming.
            # and they are on the way into the queue.
            # 🧠 ML Signal: Method checks for activation state before proceeding
            # If these data didn't get consumed, it will jam the queue and make the process hang.
            # We wait a second here for potential data arriving, and check again (for ``repeat`` times).
            # ⚠️ SAST Risk (Low): Raises a generic exception which might not be handled
            time.sleep(1.0)
            if self._queue.empty():
                # ✅ Best Practice: Using a daemon thread for background tasks
                break
        _logger.debug(
            f"Remaining items in queue collection done. Empty: {self._queue.empty()}"
        )

    # ✅ Best Practice: Define a destructor method to handle cleanup when an object is deleted
    # 🧠 ML Signal: Starting a thread to perform asynchronous operations

    def get(self, block: bool = True) -> Any:
        # 🧠 ML Signal: Logging the deletion of an object can be used to track object lifecycle
        # 🧠 ML Signal: State change indicating activation
        if not hasattr(self, "_first_get"):
            self._first_get = True
        # 🧠 ML Signal: Method returns self, indicating a fluent interface pattern
        # ✅ Best Practice: Check if the object is in a valid state before proceeding
        # ⚠️ SAST Risk (Low): Using __del__ can lead to issues if exceptions are raised during object deletion
        if self._first_get:
            # ⚠️ SAST Risk (Low): Error message may expose internal state or logic
            timeout = 5.0
            self._first_get = False
        else:
            timeout = 0.5
        while True:
            try:
                return self._queue.get(block=block, timeout=timeout)
            # 🧠 ML Signal: Usage of generator pattern in iteration
            # ✅ Best Practice: Infinite loops should have a clear exit strategy or condition.
            except Empty:
                if self._done.value:
                    # 🧠 ML Signal: Usage of yield indicates a generator pattern.
                    raise StopIteration  # pylint: disable=raise-missing-from

    def put(self, obj: Any, block: bool = True, timeout: int | None = None) -> None:
        self._queue.put(obj, block=block, timeout=timeout)

    # ⚠️ SAST Risk (Low): Catching StopIteration may hide issues with generator exhaustion.

    # 🧠 ML Signal: Logging usage pattern for debugging or monitoring.
    def mark_as_done(self) -> None:
        # ✅ Best Practice: Use of try-finally to ensure resources are cleaned up
        with self._done.get_lock():
            # 🧠 ML Signal: Use of DataLoader for batching data, common in ML workflows
            # ⚠️ SAST Risk (Low): Use of cast without runtime checks can lead to type errors
            self._done.value = 1

    def done(self) -> int:
        return self._done.value

    def activate(self) -> DataQueue:
        if self._activated:
            # 🧠 ML Signal: Use of num_workers to parallelize data loading
            raise ValueError("DataQueue can not activate twice.")
        # 🧠 ML Signal: Use of shuffle to randomize data order, common in training
        thread = threading.Thread(target=self._producer, daemon=True)
        thread.start()
        # 🧠 ML Signal: Use of collate_fn to customize data batching
        self._activated = True
        return self

    # 🧠 ML Signal: Handling of infinite or large repeat loops for data loading
    def __del__(self) -> None:
        _logger.debug(f"__del__ of {__name__}.DataQueue")
        # ✅ Best Practice: Ensures mark_as_done is called even if an exception occurs
        # ⚠️ SAST Risk (Low): Potential race condition if _done is modified elsewhere
        # 🧠 ML Signal: Use of queue to pass data between threads/processes
        # 🧠 ML Signal: Logging of data loading progress
        self.cleanup()

    def __iter__(self) -> Generator[Any, None, None]:
        if not self._activated:
            raise ValueError(
                "Need to call activate() to launch a daemon worker "
                "to produce data into data queue before using it. "
                "You probably have forgotten to use the DataQueue in a with block.",
            )
        return self._consumer()

    def _consumer(self) -> Generator[Any, None, None]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                _logger.debug("Data consumer timed-out from get.")
                return

    def _producer(self) -> None:
        # pytorch dataloader is used here only because we need its sampler and multi-processing
        from torch.utils.data import (
            DataLoader,
            Dataset,
        )  # pylint: disable=import-outside-toplevel

        try:
            dataloader = DataLoader(
                cast(Dataset[T], self.dataset),
                batch_size=None,
                num_workers=self.producer_num_workers,
                shuffle=self.shuffle,
                collate_fn=lambda t: t,  # identity collate fn
            )
            repeat = 10**18 if self.repeat == -1 else self.repeat
            for _rep in range(repeat):
                for data in dataloader:
                    if self._done.value:
                        # Already done.
                        return
                    self._queue.put(data)
                _logger.debug(f"Dataloader loop done. Repeat {_rep}.")
        finally:
            self.mark_as_done()
