# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Importing annotations from __future__ for forward compatibility with type hints
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, OrderedDict, Tuple, cast

import gym
import numpy as np
import torch
import torch.nn as nn
# ✅ Best Practice: Using __all__ to define public API of the module
# ✅ Best Practice: Class docstring provides a clear description of the class purpose.
from gym.spaces import Discrete
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy, PPOPolicy, DQNPolicy

from qlib.rl.trainer.trainer import Trainer
# ✅ Best Practice: Explicitly defining the types of parameters for better readability and maintainability

# 🧠 ML Signal: Method signature with a batch parameter, indicating a potential machine learning training step
__all__ = ["AllOne", "PPO", "DQN"]

# ✅ Best Practice: Function signature is clear and includes type hints for parameters and return type
# ✅ Best Practice: Returning an empty dictionary as a default implementation

# baselines #


class NonLearnablePolicy(BasePolicy):
    """Tianshou's BasePolicy with empty ``learn`` and ``process_fn``.

    This could be moved outside in future.
    # ✅ Best Practice: Docstring explains the method's behavior and its use case.
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
    # ✅ Best Practice: Explicitly calling the superclass initializer ensures proper initialization of inherited attributes.

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        # 🧠 ML Signal: Storing initialization parameters as instance attributes is a common pattern in ML models.
        return {}

    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        indices: np.ndarray,
    # 🧠 ML Signal: The function signature suggests a pattern for processing batches, which is common in ML models.
    # 🧠 ML Signal: Definition of a class for a neural network model, common in ML applications
    ) -> Batch:
        return Batch({})
# 🧠 ML Signal: Use of np.full indicates a pattern of filling arrays, which is common in data preprocessing or model predictions.
# ✅ Best Practice: Explicitly define the constructor with type annotations for better readability and maintainability.

# ✅ Best Practice: Returning a Batch object maintains consistency with the expected output type.

# ✅ Best Practice: Store the extractor as an instance variable for later use.
class AllOne(NonLearnablePolicy):
    """Forward returns a batch full of 1.

    Useful when implementing some baselines (e.g., TWAP).
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, fill_value: float | int = 1.0) -> None:
        # ✅ Best Practice: Use of a feature extractor function to process input data
        super().__init__(obs_space, action_space)

        # ✅ Best Practice: Clear separation of feature extraction and output layer processing
        self.fill_value = fill_value
    # 🧠 ML Signal: Definition of a class for a neural network module, indicating a pattern for model architecture

    # ✅ Best Practice: Returning a tuple for consistency and potential future expansion
    def forward(
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        self,
        batch: Batch,
        # 🧠 ML Signal: Storing a neural network module as an instance variable
        state: dict | Batch | np.ndarray = None,
        # ⚠️ SAST Risk (Low): Directly using extractor.output_dim without validation could lead to runtime errors if the attribute is missing or incorrect
        # ✅ Best Practice: Using cast to ensure type compatibility for extractor.output_dim
        # ✅ Best Practice: Consider adding type hints for the return type of the function
        **kwargs: Any,
    ) -> Batch:
        return Batch(act=np.full(len(batch), self.fill_value), state=state)


# ppo #
# ⚠️ SAST Risk (Low): Using a mutable default value for 'info' can lead to unexpected behavior


# 🧠 ML Signal: Usage of a feature extractor indicates a pattern for processing input data
# 🧠 ML Signal: Class definition with docstring provides context and usage patterns for ML models
class PPOActor(nn.Module):
    # 🧠 ML Signal: Returning a squeezed tensor suggests a pattern of reducing dimensions
    def __init__(self, extractor: nn.Module, action_dim: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(cast(int, extractor.output_dim), action_dim), nn.Softmax(dim=-1))

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        info: dict = {},
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        out = self.layer_out(feature)
        return out, state


class PPOCritic(nn.Module):
    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(cast(int, extractor.output_dim), 1)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        # ⚠️ SAST Risk (Low): Use of assert for type checking can be bypassed if Python is run with optimizations
        info: dict = {},
    ) -> torch.Tensor:
        # 🧠 ML Signal: Instantiation of PPOActor with network and action_space.n
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        # 🧠 ML Signal: Instantiation of PPOCritic with network
        return self.value_out(feature).squeeze(dim=-1)


class PPO(PPOPolicy):
    """A wrapper of tianshou PPOPolicy.

    Differences:

    - Auto-create actor and critic network. Supports discrete action space only.
    - Dedup common parameters between actor network and critic network
      (not sure whether this is included in latest tianshou or not).
    - Support a ``weight_file`` that supports loading checkpoint.
    - Some parameters' default values are different from original.
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        eps_clip: float = 0.3,
        # ✅ Best Practice: Check if weight_file is not None before calling set_weight
        value_clip: bool = True,
        vf_coef: float = 1.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        # 🧠 ML Signal: Loading weights from a file into the model
        # ⚠️ SAST Risk (Low): DQNModel is assigned to PPOActor, which may cause confusion if used interchangeably
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)
        actor = PPOActor(network, action_space.n)
        critic = PPOCritic(network)
        optimizer = torch.optim.Adam(
            chain_dedup(actor.parameters(), critic.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        super().__init__(
            actor,
            critic,
            optimizer,
            torch.distributions.Categorical,
            # ⚠️ SAST Risk (Low): Lack of input validation for parameters like lr, weight_decay, discount_factor
            discount_factor=discount_factor,
            max_grad_norm=max_grad_norm,
            # 🧠 ML Signal: Use of DQNModel indicates reinforcement learning pattern
            # 🧠 ML Signal: Use of Adam optimizer is common in training neural networks
            reward_normalization=reward_normalization,
            eps_clip=eps_clip,
            value_clip=value_clip,
            vf_coef=vf_coef,
            gae_lambda=gae_lambda,
            # ✅ Best Practice: Calling superclass constructor ensures proper initialization
            max_batchsize=max_batch_size,
            deterministic_eval=deterministic_eval,
            observation_space=obs_space,
            action_space=action_space,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


DQNModel = PPOActor  # Reuse PPOActor.


# 🧠 ML Signal: Function to determine the device of a neural network module
class DQN(DQNPolicy):
    """A wrapper of tianshou DQNPolicy.

    Differences:

    - Auto-create model network. Supports discrete action space only.
    - Support a ``weight_file`` that supports loading checkpoint.
    """

    def __init__(
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide other issues
        self,
        network: nn.Module,
        # ✅ Best Practice: Use descriptive variable names for clarity
        # ✅ Best Practice: Using a set for 'seen' ensures O(1) average time complexity for lookups.
        obs_space: gym.Space,
        action_space: gym.Space,
        # 🧠 ML Signal: Retrying to load weights after modifying keys
        lr: float,
        weight_decay: float = 0.0,
        # ✅ Best Practice: Checking membership in a set is efficient and prevents duplicates.
        discount_factor: float = 0.99,
        # ✅ Best Practice: Adding to a set is efficient and maintains uniqueness.
        # 🧠 ML Signal: Yielding values in a generator function indicates a streaming or lazy evaluation pattern.
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)

        model = DQNModel(network, action_space.n)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(
            model,
            optimizer,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


# utilities: these should be put in a separate (common) file. #


def auto_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")  # fallback to cpu


def set_weight(policy: nn.Module, loaded_weight: OrderedDict) -> None:
    try:
        policy.load_state_dict(loaded_weight)
    except RuntimeError:
        # try again by loading the converted weight
        # https://github.com/thu-ml/tianshou/issues/468
        for k in list(loaded_weight):
            loaded_weight["_actor_critic." + k] = loaded_weight[k]
        policy.load_state_dict(loaded_weight)


def chain_dedup(*iterables: Iterable) -> Generator[Any, None, None]:
    seen = set()
    for iterable in iterables:
        for i in iterable:
            if i not in seen:
                seen.add(i)
                yield i