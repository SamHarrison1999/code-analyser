# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations.
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Imports are organized and specific, improving readability and maintainability.

from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Tuple, TypeVar

# ⚠️ SAST Risk (Low): Importing from external libraries can introduce security risks if the library is compromised.
from qlib.typehint import final

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper
# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports and improve performance during runtime.
# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability.

SimulatorState = TypeVar("SimulatorState")
# ✅ Best Practice: Using @final indicates that the method should not be overridden, which can prevent errors in subclassing.
# ✅ Best Practice: Use of __call__ method allows instances of the class to be called as functions, improving readability and usability.

# ✅ Best Practice: TypeVar is used for generic programming, enhancing code flexibility and reusability.


# 🧠 ML Signal: The method returns a float, indicating it might be used for numerical computations or evaluations.
# ✅ Best Practice: Method docstring provides clarity on the method's purpose.
class Reward(Generic[SimulatorState]):
    """
    Reward calculation component that takes a single argument: state of simulator. Returns a real number: reward.

    Subclass should implement ``reward(simulator_state)`` to implement their own reward calculation recipe.
    # ⚠️ SAST Risk (Low): The use of assert for runtime checks can be disabled with optimization flags, potentially leading to unexpected behavior.
    """

    # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
    # 🧠 ML Signal: Logging scalar values is a common pattern in ML for tracking metrics and model performance.
    env: Optional[EnvWrapper] = None
    # ✅ Best Practice: Use of type hints for function parameters and return type

    @final
    # ✅ Best Practice: Storing input parameters as instance variables
    def __call__(self, simulator_state: SimulatorState) -> float:
        # ✅ Best Practice: Initialize variables before use
        return self.reward(simulator_state)

    # 🧠 ML Signal: Iterating over a dictionary of functions and weights
    def reward(self, simulator_state: SimulatorState) -> float:
        """Implement this method for your own reward."""
        # 🧠 ML Signal: Function call pattern with dynamic function execution
        raise NotImplementedError("Implement reward calculation recipe in `reward()`.")

    # 🧠 ML Signal: Logging pattern for tracking function outputs
    # ✅ Best Practice: Use of descriptive variable names
    # ✅ Best Practice: Explicit return of the computed result

    def log(self, name: str, value: Any) -> None:
        assert self.env is not None
        self.env.logger.add_scalar(name, value)


class RewardCombination(Reward):
    """Combination of multiple reward."""

    def __init__(self, rewards: Dict[str, Tuple[Reward, float]]) -> None:
        self.rewards = rewards

    def reward(self, simulator_state: Any) -> float:
        total_reward = 0.0
        for name, (reward_fn, weight) in self.rewards.items():
            rew = reward_fn(simulator_state) * weight
            total_reward += rew
            self.log(name, rew)
        return total_reward


# TODO:
# reward_factory is disabled for now

# _RegistryConfigReward = RegistryConfig[REWARDS]


# @configclass
# class _WeightedRewardConfig:
#     weight: float
#     reward: _RegistryConfigReward


# RewardConfig = Union[_RegistryConfigReward, Dict[str, Union[_RegistryConfigReward, _WeightedRewardConfig]]]


# def reward_factory(reward_config: RewardConfig) -> Reward:
#     """
#     Use this factory to instantiate the reward from config.
#     Simply using ``reward_config.build()`` might not work because reward can have complex combinations.
#     """
#     if isinstance(reward_config, dict):
#         # as reward combination
#         rewards = {}
#         for name, rew in reward_config.items():
#             if not isinstance(rew, _WeightedRewardConfig):
#                 # default weight is 1.
#                 rew = _WeightedRewardConfig(weight=1., rew=rew)
#             # no recursive build in this step
#             rewards[name] = (rew.reward.build(), rew.weight)
#         return RewardCombination(rewards)
#     else:
#         # single reward
#         return reward_config.build()
