# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import multiprocessing
import time
# ✅ Best Practice: Import only necessary components from a module to improve code readability and maintainability.

import numpy as np
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
import pandas as pd
# ✅ Best Practice: Use of constructor to initialize object attributes

from torch.utils.data import Dataset, DataLoader
# ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations.
from qlib.rl.utils.data_queue import DataQueue

# 🧠 ML Signal: Generates a DataFrame with random data, which could indicate synthetic data generation.
# ✅ Best Practice: Implementing __len__ allows objects to be used with len() function

# ✅ Best Practice: Consider parameterizing the range and size for better flexibility and testing.
class DummyDataset(Dataset):
    # 🧠 ML Signal: Function definition with parameters indicating a worker pattern
    # 🧠 ML Signal: Returning an attribute in __len__ suggests the object has a length property
    def __init__(self, length):
        self.length = length
    # 🧠 ML Signal: Iterating over a dataloader, common in data processing tasks

    # ✅ Best Practice: Use of a private function name to indicate internal use
    def __getitem__(self, index):
        # 🧠 ML Signal: Collecting data length, indicative of data size tracking
        assert 0 <= index < self.length
        return pd.DataFrame(np.random.randint(0, 100, size=(index + 1, 4)), columns=list("ABCD"))
    # 🧠 ML Signal: Iterating over a queue to convert it to a list

    def __len__(self):
        # ⚠️ SAST Risk (Low): Potential for blocking if queue.get() is used without a timeout in a multithreaded context
        # 🧠 ML Signal: Use of PyTorch DataLoader for batching data
        return self.length
# ✅ Best Practice: Use of a test function to validate functionality


# 🧠 ML Signal: Use of a custom dataset for testing
def _worker(dataloader, collector):
    # for i in range(3):
    # ⚠️ SAST Risk (Low): batch_size=None may lead to unexpected behavior
    for i, data in enumerate(dataloader):
        # 🧠 ML Signal: Function definition for testing, useful for identifying test patterns
        # 🧠 ML Signal: Use of DataLoader with specific parameters
        collector.put(len(data))

# 🧠 ML Signal: Use of multiprocessing for parallel data processing
# 🧠 ML Signal: Instantiation of a dataset, common in data processing tasks

def _queue_to_list(queue):
    # 🧠 ML Signal: Custom worker function for data processing
    # 🧠 ML Signal: Use of context manager, indicates resource management pattern
    result = []
    while not queue.empty():
        # ✅ Best Practice: Use of assertions to verify test outcomes
        # 🧠 ML Signal: Use of multiprocessing queue, indicates parallel processing pattern
        result.append(queue.get())
    return result


# 🧠 ML Signal: Creation of multiple processes, common in parallel execution
def test_pytorch_dataloader():
    dataset = DummyDataset(100)
    # ⚠️ SAST Risk (Low): Starting a process without error handling
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    # 🧠 ML Signal: Custom function definition with specific behavior
    queue = multiprocessing.Queue()
    _worker(dataloader, queue)
    # ⚠️ SAST Risk (Low): Joining a process without timeout, may lead to hanging
    # 🧠 ML Signal: Instantiation of a dataset object
    assert len(set(_queue_to_list(queue))) == 100

# ✅ Best Practice: Use of assert for validation in tests
# 🧠 ML Signal: Usage of a context manager with specific parameters

def test_multiprocess_shared_dataloader():
    # 🧠 ML Signal: Usage of time.sleep to introduce a delay
    dataset = DummyDataset(100)
    with DataQueue(dataset, producer_num_workers=1) as data_queue:
        # ⚠️ SAST Risk (Low): Raising a generic exception without handling
        queue = multiprocessing.Queue()
        processes = []
        # 🧠 ML Signal: Creation of a multiprocessing process with a target function
        # 🧠 ML Signal: Usage of DataQueue with repeat=-1 indicates an infinite loop pattern
        for _ in range(3):
            processes.append(multiprocessing.Process(target=_worker, args=(data_queue, queue)))
            # 🧠 ML Signal: Starting a multiprocessing process
            # 🧠 ML Signal: Usage of time.sleep to simulate delay or wait
            processes[-1].start()
        for p in processes:
            # ⚠️ SAST Risk (Low): Raising a generic exception without handling
            # 🧠 ML Signal: Joining a multiprocessing process to wait for its completion
            p.join()
        assert len(set(_queue_to_list(queue))) == 100
# 🧠 ML Signal: Usage of multiprocessing to run a function in a separate process

# ✅ Best Practice: Joining a process to ensure it completes before moving on
# ⚠️ SAST Risk (Low): Function call without definition in the provided code
# ✅ Best Practice: Explicitly starting a process
# ✅ Best Practice: Using the main guard to ensure code is only executed when the script is run directly

def test_exit_on_crash_finite():
    def _exit_finite():
        dataset = DummyDataset(100)

        with DataQueue(dataset, producer_num_workers=4) as data_queue:
            time.sleep(3)
            raise ValueError

        # https://stackoverflow.com/questions/34506638/how-to-register-atexit-function-in-pythons-multiprocessing-subprocess

    process = multiprocessing.Process(target=_exit_finite)
    process.start()
    process.join()


def test_exit_on_crash_infinite():
    def _exit_infinite():
        dataset = DummyDataset(100)
        with DataQueue(dataset, repeat=-1, queue_maxsize=100) as data_queue:
            time.sleep(3)
            raise ValueError

    process = multiprocessing.Process(target=_exit_infinite)
    process.start()
    process.join()


if __name__ == "__main__":
    test_multiprocess_shared_dataloader()