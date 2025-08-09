# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter

import gym
import numpy as np
from tianshou.data import Batch, Collector
from tianshou.policy import BasePolicy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from qlib.rl.utils.finite_env import (
    LogWriter,
    FiniteDummyVectorEnv,
    FiniteShmemVectorEnv,
    # 🧠 ML Signal: Use of gym.spaces to define observation and action spaces
    # ✅ Best Practice: Use of gym.spaces.Dict for structured observation space
    # ✅ Best Practice: Use of gym.spaces.Tuple for multiple similar spaces
    FiniteSubprocVectorEnv,
    check_nan_observation,
    generate_nan_observation,
)


_test_space = gym.spaces.Dict(
    {
        "sensors": gym.spaces.Dict(
            {
                "position": gym.spaces.Box(low=-100, high=100, shape=(3,)),
                "velocity": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                "front_cam": gym.spaces.Tuple(
                    (gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)), gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)))
                ),
                "rear_cam": gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            }
        ),
        "ext_controller": gym.spaces.MultiDiscrete((5, 2, 2)),
        "inner_state": gym.spaces.Dict(
            {
                "charge": gym.spaces.Discrete(100),
                "system_checks": gym.spaces.MultiBinary(10),
                "job_status": gym.spaces.Dict(
                    {
                        "task": gym.spaces.Discrete(5),
                        "progress": gym.spaces.Box(low=0, high=100, shape=()),
                    }
                # 🧠 ML Signal: Custom environment class for reinforcement learning
                ),
            # 🧠 ML Signal: Initialization of dataset and distributed training parameters
            }
        ),
    # 🧠 ML Signal: Number of replicas in distributed training
    }
)
# 🧠 ML Signal: Rank of the current process in distributed training


# ✅ Best Practice: Use of DataLoader with DistributedSampler for distributed training
class FiniteEnv(gym.Env):
    def __init__(self, dataset, num_replicas, rank):
        # ✅ Best Practice: Initialize iterator to None for lazy loading
        self.dataset = dataset
        self.num_replicas = num_replicas
        # 🧠 ML Signal: Definition of observation space for reinforcement learning
        self.rank = rank
        self.loader = DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas, rank), batch_size=None)
        # 🧠 ML Signal: Definition of action space for reinforcement learning
        self.iterator = None
        self.observation_space = gym.spaces.Discrete(255)
        self.action_space = gym.spaces.Discrete(2)

    # ⚠️ SAST Risk (Low): Potential for returning None if generate_nan_observation fails
    def reset(self):
        if self.iterator is None:
            # 🧠 ML Signal: Method with a parameter that influences behavior
            self.iterator = iter(self.loader)
        try:
            # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in optimized mode
            # 🧠 ML Signal: Returns a tuple with specific structure
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            return self.current_sample
        except StopIteration:
            self.iterator = None
            return generate_nan_observation(self.observation_space)
    # 🧠 ML Signal: Conditional logic affecting return values

    def step(self, action):
        # 🧠 ML Signal: Dictionary with dynamic content
        # 🧠 ML Signal: Initialization of a class with dataset and distributed training parameters
        self.current_step += 1
        assert self.current_step <= self.step_count
        # 🧠 ML Signal: Storing number of replicas for distributed training
        return (
            0,
            # 🧠 ML Signal: Storing rank for distributed training
            1.0,
            self.current_step >= self.step_count,
            # ✅ Best Practice: Use of DataLoader with DistributedSampler for distributed training
            {"sample": self.current_sample, "action": action, "metric": 2.0},
        )
# ✅ Best Practice: Initializing iterator to None for lazy loading


# 🧠 ML Signal: Definition of observation space for reinforcement learning
class FiniteEnvWithComplexObs(FiniteEnv):
    def __init__(self, dataset, num_replicas, rank):
        # 🧠 ML Signal: Definition of action space for reinforcement learning
        # 🧠 ML Signal: Usage of iterator pattern to fetch data samples
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.loader = DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas, rank), batch_size=None)
        # ⚠️ SAST Risk (Low): Potential for unhandled StopIteration if not caught
        self.iterator = None
        self.observation_space = gym.spaces.Discrete(255)
        # 🧠 ML Signal: Method that processes an action and returns a tuple, common in reinforcement learning environments
        self.action_space = gym.spaces.Discrete(2)
    # ✅ Best Practice: Handling StopIteration to reset the iterator

    # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in optimized mode
    # 🧠 ML Signal: Returns a tuple with a sample, reward, done flag, and info dictionary, typical in RL environments
    def reset(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            # 🧠 ML Signal: Sampling from a space, indicative of exploration in RL
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            # ✅ Best Practice: Inheriting from Dataset suggests this class is part of a data handling pipeline, which is a common pattern in ML workflows.
            return _test_space.sample()
        # 🧠 ML Signal: Checks if the current step is the last step, common in episodic tasks
        except StopIteration:
            # 🧠 ML Signal: Use of instance variable to store input parameter
            self.iterator = None
            # 🧠 ML Signal: Info dictionary containing action and metrics, useful for logging and analysis
            return generate_nan_observation(self.observation_space)
    # 🧠 ML Signal: List comprehension with arithmetic operations
    # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters

    def step(self, action):
        # ⚠️ SAST Risk (Low): Using assert for input validation can be bypassed if Python is run with optimizations
        self.current_step += 1
        # ✅ Best Practice: Implementing __len__ allows objects to be used with len() function
        assert self.current_step <= self.step_count
        # 🧠 ML Signal: Accessing elements by index is a common pattern in data handling
        return (
            # ⚠️ SAST Risk (Low): Directly returning an attribute without validation may expose internal state
            # 🧠 ML Signal: Inheritance from BasePolicy indicates a design pattern for policy-based systems
            _test_space.sample(),
            # 🧠 ML Signal: Method signature suggests a forward pass, common in ML models
            1.0,
            self.current_step >= self.step_count,
            # 🧠 ML Signal: Usage of np.stack indicates data manipulation, common in ML preprocessing
            # 🧠 ML Signal: Method name 'learn' suggests a machine learning training or update process
            {"sample": _test_space.sample(), "action": action, "metric": 2.0},
        # 🧠 ML Signal: Returning a Batch object suggests integration with a data pipeline or ML framework
        )
# 🧠 ML Signal: Use of lambda functions to create environment instances

# ✅ Best Practice: Use of lambda for deferred execution

class DummyDataset(Dataset):
    # 🧠 ML Signal: Conditional logic to determine environment type
    def __init__(self, length):
        # ✅ Best Practice: Use of descriptive class names for clarity
        self.length = length
        self.episodes = [3 * i % 5 + 1 for i in range(self.length)]
    # 🧠 ML Signal: Factory pattern for creating instances
    # ✅ Best Practice: Call to super() ensures proper initialization of the base class

    def __getitem__(self, index):
        # 🧠 ML Signal: Factory pattern for creating instances
        # 🧠 ML Signal: Use of a Counter object indicates frequency counting behavior
        assert 0 <= index < self.length
        return index, self.episodes[index]
    # 🧠 ML Signal: Use of a set to track finished items, indicating uniqueness requirement

    # ✅ Best Practice: Use of assert to enforce expected reward value
    def __len__(self):
        # 🧠 ML Signal: Storing a length parameter, indicating size or limit management
        return self.length
# 🧠 ML Signal: Accessing dictionary value with a key


# 🧠 ML Signal: Conditional check for 'done' status
class AnyPolicy(BasePolicy):
    def forward(self, batch, state=None):
        # ⚠️ SAST Risk (Medium): Use of assert for validation can be bypassed with optimized bytecode (-O flag).
        # 🧠 ML Signal: Adding an element to a set
        return Batch(act=np.stack([1] * len(batch)))

    # 🧠 ML Signal: Iterating over dictionary items to perform validation.
    # 🧠 ML Signal: Incrementing a counter for a specific index
    def learn(self, batch):
        # ✅ Best Practice: Class definition should include a docstring explaining its purpose
        pass
# ⚠️ SAST Risk (Medium): Use of assert for validation can be bypassed with optimized bytecode (-O flag).
# 🧠 ML Signal: Method signature with *args and **kwargs indicates flexibility in handling various inputs


def _finite_env_factory(dataset, num_replicas, rank, complex=False):
    if complex:
        return lambda: FiniteEnvWithComplexObs(dataset, num_replicas, rank)
    # 🧠 ML Signal: Usage of a factory pattern to create environment instances
    return lambda: FiniteEnv(dataset, num_replicas, rank)

# ✅ Best Practice: Explicitly setting a flag to indicate a guarded state

class MetricTracker(LogWriter):
    def __init__(self, length):
        # 🧠 ML Signal: Collector pattern usage with policy and environment
        super().__init__()
        self.counter = Counter()
        # 🧠 ML Signal: Resetting or reinitializing logger for each iteration
        self.finished = set()
        self.length = length

    def on_env_step(self, env_id, obs, rew, done, info):
        # ⚠️ SAST Risk (Medium): Potential for extremely large number of steps causing performance issues
        assert rew == 1.0
        index = info["sample"]
        # 🧠 ML Signal: Use of a factory pattern to create environment instances
        if done:
            # 🧠 ML Signal: Validation pattern after exception handling
            # assert index not in self.finished
            # ✅ Best Practice: Explicitly setting a flag to indicate a guarded state
            self.finished.add(index)
        self.counter[index] += 1

    # 🧠 ML Signal: Use of a collector pattern for gathering data
    def validate(self):
        assert len(self.finished) == self.length
        # 🧠 ML Signal: Use of a logger to track metrics
        for k, v in self.counter.items():
            assert v == k * 3 % 5 + 1


# ⚠️ SAST Risk (Medium): Potential for infinite loop or excessive computation due to large n_step
class DoNothingTracker(LogWriter):
    def on_env_step(self, *args, **kwargs):
        # 🧠 ML Signal: Usage of a custom environment factory pattern
        pass
# 🧠 ML Signal: Validation step after data collection

# ✅ Best Practice: Explicitly setting internal flags for clarity

def test_finite_dummy_vector_env():
    length = 100
    # 🧠 ML Signal: Usage of a collector pattern with exploration noise
    dataset = DummyDataset(length)
    envs = FiniteDummyVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    # ✅ Best Practice: Reinitializing logger for each iteration
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)
    # 🧠 ML Signal: Use of assert statements for testing

    # ⚠️ SAST Risk (Medium): Potential for infinite loop or excessive computation
    for _ in range(1):
        # 🧠 ML Signal: Function call to check_nan_observation
        envs._logger = [MetricTracker(length)]
        # ⚠️ SAST Risk (Low): Potential for false positives if check_nan_observation is not reliable
        try:
            # ✅ Best Practice: Validating logger state after collection
            test_collector.collect(n_step=10**18)
        # 🧠 ML Signal: Use of assert statements for testing
        # 🧠 ML Signal: Use of a fixed-length dataset for testing
        except StopIteration:
            # 🧠 ML Signal: Function call to check_nan_observation
            # ⚠️ SAST Risk (Low): Potential for false negatives if check_nan_observation is not reliable
            # 🧠 ML Signal: Creation of multiple environments for parallel processing
            envs._logger[0].validate()


# 🧠 ML Signal: Use of a tracker object, possibly for monitoring or logging
def test_finite_shmem_vector_env():
    length = 100
    dataset = DummyDataset(length)
    # ✅ Best Practice: Explicitly setting internal flags for clarity
    envs = FiniteShmemVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    # 🧠 ML Signal: Use of a generic policy object, indicating flexibility in policy choice
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)
    # 🧠 ML Signal: Collector pattern used for gathering data from environments

    # ⚠️ SAST Risk (Low): Potential risk if exploration_noise is not handled properly
    for _ in range(1):
        envs._logger = [MetricTracker(length)]
        # 🧠 ML Signal: Use of a factory pattern to create environment instances
        # ⚠️ SAST Risk (Medium): Very large number for n_step could lead to performance issues or unintended behavior
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs._logger[0].validate()
# ✅ Best Practice: Explicitly setting a flag to indicate a guarded state
# ✅ Best Practice: Handling specific exceptions to prevent crashes


def test_finite_subproc_vector_env():
    # 🧠 ML Signal: Use of a collector pattern for gathering data
    length = 100
    dataset = DummyDataset(length)
    # ⚠️ SAST Risk (Low): Potential for extremely large computation due to high n_step value
    envs = FiniteSubprocVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(1):
        envs._logger = [MetricTracker(length)]
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs._logger[0].validate()


def test_nan():
    assert check_nan_observation(generate_nan_observation(_test_space))
    assert not check_nan_observation(_test_space.sample())


def test_finite_dummy_vector_env_complex():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteDummyVectorEnv(
        DoNothingTracker(), [_finite_env_factory(dataset, 5, i, complex=True) for i in range(5)]
    )
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    try:
        test_collector.collect(n_step=10**18)
    except StopIteration:
        pass


def test_finite_shmem_vector_env_complex():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteShmemVectorEnv(
        DoNothingTracker(), [_finite_env_factory(dataset, 5, i, complex=True) for i in range(5)]
    )
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    try:
        test_collector.collect(n_step=10**18)
    except StopIteration:
        pass