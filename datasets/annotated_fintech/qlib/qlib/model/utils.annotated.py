# Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing Dataset from torch.utils.data indicates usage of PyTorch for machine learning tasks
# ✅ Best Practice: Inheriting from Dataset indicates this class is intended for data handling, which is a clear design choice.
# Licensed under the MIT License.

# 🧠 ML Signal: Use of *args to accept a variable number of arguments
from torch.utils.data import Dataset
# ✅ Best Practice: Use of __getitem__ method to allow object indexing

# ✅ Best Practice: Storing datasets in an instance variable for later use

# ✅ Best Practice: Implementing __len__ allows the object to be used with len()
# 🧠 ML Signal: Iterating over a collection of datasets
class ConcatDataset(Dataset):
    # ✅ Best Practice: Use of tuple comprehension for concise and efficient tuple creation
    def __init__(self, *datasets):
        # 🧠 ML Signal: Usage of min() function to determine the length
        self.datasets = datasets
    # 🧠 ML Signal: Iterating over self.datasets to calculate length
    # ✅ Best Practice: Initialize class attributes in the constructor for clarity and maintainability
    # ✅ Best Practice: Constructor should initialize all attributes

    def __getitem__(self, i):
        # ✅ Best Practice: Type hinting for the parameter improves code readability and maintainability
        # 🧠 ML Signal: Usage of class constructor to initialize with data
        # 🧠 ML Signal: Storing a parameter as an instance attribute
        return tuple(d[i] for d in self.datasets)

    # 🧠 ML Signal: Usage of __getitem__ suggests this class might be used like a list or dictionary
    # ✅ Best Practice: Implementing __len__ allows the object to be used with len(), improving usability.
    def __len__(self):
        # ✅ Best Practice: Use of a method to encapsulate functionality for sampling
        # ⚠️ SAST Risk (Medium): No input validation for 'n', could lead to unexpected behavior
        # 🧠 ML Signal: Sampling pattern from a dataset
        # ✅ Best Practice: Importing modules at the top of the file is preferred
        # ✅ Best Practice: Method to get the size of the data, improving encapsulation
        # 🧠 ML Signal: Access pattern to determine the size of a dataset
        # 🧠 ML Signal: Usage of len() on custom objects can indicate object size or count properties.
        return min(len(d) for d in self.datasets)


class IndexSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, i: int):
        return self.sampler[i], i

    def __len__(self):
        return len(self.sampler)