import os
import random
import sys
from pathlib import Path

import pytest

import torch
import torch.nn as nn
from gym import spaces
from tianshou.policy import PPOPolicy

from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
# 🧠 ML Signal: Conditional test skipping based on Python version
from qlib.rl.simulator import Simulator
from qlib.rl.reward import Reward
from qlib.rl.trainer import Trainer, TrainingVessel, EarlyStopping, Checkpoint
# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability

pytestmark = pytest.mark.skipif(sys.version_info < (3, 8), reason="Pickle styled data only supports Python >= 3.8")
# 🧠 ML Signal: Captures the action taken, which can be used to understand decision patterns


# 🧠 ML Signal: Captures whether the action was correct, useful for training models on success rates
class ZeroSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        # ⚠️ SAST Risk (Low): Use of random.choice can lead to non-deterministic behavior, which might be undesirable in some contexts
        self.action = self.correct = 0
    # ✅ Best Practice: Method name 'get_state' suggests it returns an object's state, which is clear and descriptive.

    # 🧠 ML Signal: Logs the accuracy, which can be used to track performance over time
    # ✅ Best Practice: Consider checking if 'self.env.logger' and 'add_scalar' are defined to avoid potential AttributeError
    # 🧠 ML Signal: Returning a dictionary is a common pattern for encapsulating multiple related values.
    def step(self, action):
        self.action = action
        self.correct = action == 0
        # 🧠 ML Signal: Multiplying by 100 suggests conversion to a percentage, a common data transformation.
        self._done = random.choice([False, True])
        # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability.
        if self._done:
            # 🧠 ML Signal: Including 'action' in the state suggests it's an important attribute for the object's behavior.
            self.env.logger.add_scalar("acc", self.correct * 100)
    # ✅ Best Practice: Class should inherit from a base class to ensure consistent interface
    # 🧠 ML Signal: Method returning a boolean value, indicating a status or completion flag.

    # 🧠 ML Signal: Use of observation_space suggests reinforcement learning or similar ML context
    def get_state(self):
        return {
            "acc": self.correct * 100,
            "action": self.action,
        }

    # 🧠 ML Signal: Discrete space indicates categorical or limited set of values
    def done(self) -> bool:
        # 🧠 ML Signal: Function returns input directly, indicating a possible identity function
        return self._done
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.


# ✅ Best Practice: Class attributes should be documented to explain their purpose.
# ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
class NoopStateInterpreter(StateInterpreter):
    observation_space = spaces.Dict(
        # 🧠 ML Signal: Directly returning a parameter could indicate a pass-through or identity function
        {
            "acc": spaces.Discrete(200),
            # 🧠 ML Signal: Use of environment status to determine reward logic
            "action": spaces.Discrete(2),
        }
    # 🧠 ML Signal: Reward calculation based on simulator state
    )
    # 🧠 ML Signal: Custom neural network class definition

    # ✅ Best Practice: Explicit return of default value for clarity
    def interpret(self, simulator_state):
        # ✅ Best Practice: Call the superclass's __init__ method to ensure proper initialization
        return simulator_state

# 🧠 ML Signal: Usage of nn.Linear indicates a neural network layer, common in ML models

class NoopActionInterpreter(ActionInterpreter):
    # 🧠 ML Signal: return_state flag suggests optional return of internal state, a pattern in RNNs
    # 🧠 ML Signal: Use of forward method suggests this is a neural network model
    action_space = spaces.Discrete(2)
    # 🧠 ML Signal: Use of obs as input indicates processing of observations, common in RL or similar tasks

    # ⚠️ SAST Risk (Low): Use of torch.randn without a fixed seed can lead to non-deterministic behavior
    def interpret(self, simulator_state, action):
        return action
# ✅ Best Practice: Consider using a fixed seed for reproducibility

# 🧠 ML Signal: Function defining a PPO policy, useful for training RL models

# 🧠 ML Signal: Conditional return of state suggests model may be used in stateful contexts
class AccReward(Reward):
    # 🧠 ML Signal: Instantiation of a policy network with specific parameters
    def reward(self, simulator_state):
        # 🧠 ML Signal: Instantiation of a policy network without parameters
        # ⚠️ SAST Risk (Low): Softmax without numerical stability checks can lead to overflow issues
        # 🧠 ML Signal: Creation of a PPO policy with actor, critic, optimizer, and distribution
        if self.env.status["done"]:
            return simulator_state["acc"] / 100
        return 0.0


class PolicyNet(nn.Module):
    def __init__(self, out_features=1, return_state=False):
        # ⚠️ SAST Risk (Low): Potential risk if parameters are not properly validated
        super().__init__()
        # 🧠 ML Signal: Function definition for testing a trainer, useful for understanding test patterns
        self.fc = nn.Linear(32, out_features)
        # 🧠 ML Signal: Use of Categorical distribution for action selection
        self.return_state = return_state
    # 🧠 ML Signal: Logging configuration setup, useful for understanding logging practices

    # ⚠️ SAST Risk (Low): Potential risk if action space is not properly defined
    def forward(self, obs, state=None, **kwargs):
        # 🧠 ML Signal: Instantiation of a Trainer object, useful for understanding object creation patterns
        # 🧠 ML Signal: Use of lambda for simulator function, useful for understanding functional programming usage
        # ✅ Best Practice: Return statement for the created policy
        # 🧠 ML Signal: Policy creation, useful for understanding policy patterns in ML
        # 🧠 ML Signal: TrainingVessel instantiation, useful for understanding how training environments are set up
        res = self.fc(torch.randn(obs["acc"].shape[0], 32))
        if self.return_state:
            return nn.functional.softmax(res, dim=-1), state
        else:
            return res


def _ppo_policy():
    actor = PolicyNet(2, True)
    critic = PolicyNet()
    policy = PPOPolicy(
        actor,
        # 🧠 ML Signal: Policy assignment, useful for understanding policy usage
        # 🧠 ML Signal: State interpreter setup, useful for understanding state handling
        critic,
        torch.optim.Adam(tuple(actor.parameters()) + tuple(critic.parameters())),
        # 🧠 ML Signal: Initial states for training, useful for understanding data initialization
        torch.distributions.Categorical,
        action_space=NoopActionInterpreter().action_space,
    # 🧠 ML Signal: Initial states for validation, useful for understanding data initialization
    )
    return policy
# 🧠 ML Signal: Initial states for testing, useful for understanding data initialization

# 🧠 ML Signal: Function definition for testing a trainer, useful for identifying test patterns

# 🧠 ML Signal: Reward setup, useful for understanding reward mechanisms
def test_trainer():
    # 🧠 ML Signal: Logging configuration setup, useful for understanding logging practices
    set_log_with_config(C.logging_config)
    # 🧠 ML Signal: Episode configuration, useful for understanding training iteration setup
    trainer = Trainer(max_iters=10, finite_env_type="subproc")
    # 🧠 ML Signal: Fitting the trainer, useful for understanding training execution
    # 🧠 ML Signal: Assertions for testing, useful for understanding test validation patterns
    # 🧠 ML Signal: Training vessel setup, useful for understanding training environment configuration
    # 🧠 ML Signal: Update configuration, useful for understanding training update patterns
    # 🧠 ML Signal: Trainer initialization with specific parameters, useful for model training patterns
    # 🧠 ML Signal: Policy creation, useful for understanding policy usage in training
    # 🧠 ML Signal: Simulator function setup, useful for understanding simulator initialization
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        # 🧠 ML Signal: Assertions for testing, useful for understanding test validation patterns
        # 🧠 ML Signal: State interpreter setup, useful for understanding state processing
        # 🧠 ML Signal: Policy assignment, useful for understanding policy integration
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    # 🧠 ML Signal: Testing the trainer, useful for understanding test execution
    # 🧠 ML Signal: Initial states for training, useful for understanding data initialization
    trainer.fit(vessel)
    # 🧠 ML Signal: Logging configuration is set, indicating the importance of logging in ML workflows.
    assert trainer.current_iter == 10
    # 🧠 ML Signal: Assertions for testing, useful for understanding test validation patterns
    # 🧠 ML Signal: Trainer initialization with specific parameters, common in ML model training.
    # 🧠 ML Signal: Initial states for validation, useful for understanding validation setup
    assert trainer.current_episode == 5000
    assert abs(trainer.metrics["acc"] - trainer.metrics["reward"] * 100) < 1e-4
    assert trainer.metrics["acc"] > 80
    trainer.test(vessel)
    assert trainer.metrics["acc"] > 60

# 🧠 ML Signal: Episode configuration, useful for understanding training iteration setup
# 🧠 ML Signal: Use of EarlyStopping callback, a common pattern in ML to prevent overfitting.
# 🧠 ML Signal: Reward function setup, useful for understanding reward mechanisms

# 🧠 ML Signal: Policy creation, indicating reinforcement learning context.
# 🧠 ML Signal: Use of lambda for simulator function, indicating dynamic environment setup.
# 🧠 ML Signal: Update parameters, useful for understanding training update strategies
# 🧠 ML Signal: Trainer fitting process, useful for understanding model training execution
# 🧠 ML Signal: Assertion for test validation, useful for understanding test verification practices
# 🧠 ML Signal: TrainingVessel setup with various interpreters and initial states, typical in RL environments.
def test_trainer_fast_dev_run():
    set_log_with_config(C.logging_config)
    trainer = Trainer(max_iters=2, fast_dev_run=2, finite_env_type="shmem")
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        update_kwargs=dict(repeat=10, batch_size=64),
    # 🧠 ML Signal: Logging configuration is set, indicating a pattern of tracking and monitoring during training.
    )
    trainer.fit(vessel)
    # 🧠 ML Signal: Fitting the trainer with the vessel, a key step in model training.
    # ✅ Best Practice: Using Path for file paths improves cross-platform compatibility.
    assert trainer.current_episode == 4

# 🧠 ML Signal: Trainer is initialized with specific parameters, indicating a pattern of model training setup.
# 🧠 ML Signal: Use of lambda for simulator function, indicating a pattern of dynamic function definition.
# ⚠️ SAST Risk (Low): Assertion without error message; consider adding a message for clarity.
# 🧠 ML Signal: Policy setup for training, indicating a pattern of reinforcement learning model configuration.
# 🧠 ML Signal: TrainingVessel is configured, indicating a pattern of environment and policy setup for training.

def test_trainer_earlystop():
    # TODO this is just sanity check.
    # need to see the logs to check whether it works.
    set_log_with_config(C.logging_config)
    trainer = Trainer(
        max_iters=10,
        val_every_n_iters=1,
        finite_env_type="dummy",
        callbacks=[EarlyStopping("val/reward", restore_best_weights=True)],
    )
    policy = _ppo_policy()
    # 🧠 ML Signal: Initial states for training, validation, and testing, indicating a pattern of dataset partitioning.

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        # 🧠 ML Signal: Episode and update configuration, indicating a pattern of training loop setup.
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        # 🧠 ML Signal: Assertions on trainer state, indicating a pattern of verifying training progress.
        # 🧠 ML Signal: Fitting the trainer with the vessel, indicating a pattern of executing the training process.
        # ⚠️ SAST Risk (Low): Potential issue if the output directory is not writable or does not exist.
        # ⚠️ SAST Risk (Low): Use of os.readlink can be risky if the symlink is manipulated.
        # ⚠️ SAST Risk (Medium): Loading state from a file can be risky if the file is tampered with.
        # 🧠 ML Signal: Resuming training from a checkpoint, indicating a pattern of checkpoint management.
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)
    assert trainer.metrics["val/acc"] > 30
    assert trainer.current_iter == 2  # second iteration


def test_trainer_checkpoint():
    set_log_with_config(C.logging_config)
    output_dir = Path(__file__).parent / ".output"
    trainer = Trainer(max_iters=2, finite_env_type="dummy", callbacks=[Checkpoint(output_dir, every_n_iters=1)])
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=100,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)

    assert (output_dir / "001.pth").exists()
    assert (output_dir / "002.pth").exists()
    assert os.readlink(output_dir / "latest.pth") == str(output_dir / "002.pth")

    trainer.load_state_dict(torch.load(output_dir / "001.pth", weights_only=False))
    assert trainer.current_iter == 1
    assert trainer.current_episode == 100

    # Reload the checkpoint at first iteration
    trainer.fit(vessel, ckpt_path=output_dir / "001.pth")