# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Tuple,
)

import gym
from gym import Space

from qlib.rl.aux_info import AuxiliaryInfoCollector
from qlib.rl.interpreter import (
    ActionInterpreter,
    ObsType,
    PolicyActType,
    StateInterpreter,
)

# ✅ Best Practice: Use of __all__ to define public API of the module
from qlib.rl.reward import Reward

# ✅ Best Practice: Use of TypedDict for type hinting improves code readability and maintainability
from qlib.rl.simulator import ActType, InitialStateType, Simulator, StateType
from qlib.typehint import TypedDict

# ✅ Best Practice: Use of a constant for a missing seed iterator value
from .finite_env import generate_nan_observation

# ✅ Best Practice: Type hinting for dictionary keys and values enhances code clarity
from .log import LogCollector, LogLevel

__all__ = ["InfoDict", "EnvWrapperStatus", "EnvWrapper"]
# ✅ Best Practice: Use of TypedDict for defining a structured dictionary type
# ✅ Best Practice: Type hinting with Dict[str, Any] provides flexibility while maintaining some type safety

# ✅ Best Practice: Docstring provides clear explanation of the class purpose and field semantics
# ✅ Best Practice: Type hinting for cur_step improves code readability and maintainability
# in this case, there won't be any seed for simulator
SEED_INTERATOR_MISSING = "_missing_"


class InfoDict(TypedDict):
    """The type of dict that is used in the 4th return value of ``env.step()``."""

    # ✅ Best Practice: Type hinting for obs_history improves code readability and maintainability

    aux_info: dict
    """Any information depends on auxiliary info collector."""
    log: Dict[str, Any]
    """Collected by LogCollector."""


class EnvWrapperStatus(TypedDict):
    """
    This is the status data structure used in EnvWrapper.
    The fields here are in the semantics of RL.
    For example, ``obs`` means the observation fed into policy.
    ``action`` means the raw action returned by policy.
    """

    cur_step: int
    done: bool
    initial_state: Optional[Any]
    obs_history: list
    action_history: list
    reward_history: list


class EnvWrapper(
    gym.Env[ObsType, PolicyActType],
    Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType],
):
    """Qlib-based RL environment, subclassing ``gym.Env``.
    A wrapper of components, including simulator, state-interpreter, action-interpreter, reward.

    This is what the framework of simulator - interpreter - policy looks like in RL training.
    All the components other than policy needs to be assembled into a single object called "environment".
    The "environment" are replicated into multiple workers, and (at least in tianshou's implementation),
    one single policy (agent) plays against a batch of environments.

    Parameters
    ----------
    simulator_fn
        A callable that is the simulator factory.
        When ``seed_iterator`` is present, the factory should take one argument,
        that is the seed (aka initial state).
        Otherwise, it should take zero argument.
    state_interpreter
        State-observation converter.
    action_interpreter
        Policy-simulator action converter.
    seed_iterator
        An iterable of seed. With the help of :class:`qlib.rl.utils.DataQueue`,
        environment workers in different processes can share one ``seed_iterator``.
    reward_fn
        A callable that accepts the StateType and returns a float (at least in single-agent case).
    aux_info_collector
        Collect auxiliary information. Could be useful in MARL.
    logger
        Log collector that collects the logs. The collected logs are sent back to main process,
        via the return value of ``env.step()``.

    Attributes
    ----------
    status : EnvWrapperStatus
        Status indicator. All terms are in *RL language*.
        It can be used if users care about data on the RL side.
        Can be none when no trajectory is available.
    """

    simulator: Simulator[InitialStateType, StateType, ActType]
    # ✅ Best Practice: Using a constant for missing values improves code readability and maintainability.
    seed_iterator: str | Iterator[InitialStateType] | None

    def __init__(
        # 🧠 ML Signal: Converting an iterable to an iterator is a common pattern in data processing.
        self,
        simulator_fn: Callable[..., Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        # ✅ Best Practice: Using a default logger if none is provided ensures logging is always available.
        # 🧠 ML Signal: Method returning a property-like value, indicating a pattern of encapsulation
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        seed_iterator: Optional[Iterable[InitialStateType]],
        # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability
        reward_fn: Reward | None = None,
        # ✅ Best Practice: Use of @property decorator for getter method, enhancing readability and maintainability
        # ⚠️ SAST Risk (Low): Using `cast` can lead to runtime errors if the type is incorrect.
        aux_info_collector: AuxiliaryInfoCollector[StateType, Any] | None = None,
        # 🧠 ML Signal: Accessing properties of an object, indicating a pattern of object-oriented design
        logger: LogCollector | None = None,
    ) -> None:
        # Assign weak reference to wrapper.
        #
        # Use weak reference here, because:
        # 1. Logically, the other components should be able to live without an env_wrapper.
        #    For example, they might live in a strategy_wrapper in future.
        #    Therefore injecting a "hard" attribute called "env" is not appropripate.
        # 2. When the environment gets destroyed, it gets destoryed.
        # 🧠 ML Signal: Usage of a global constant to check a condition
        #    We don't want it to silently live inside some interpreters.
        # 3. Avoid circular reference.
        # 🧠 ML Signal: Dynamic instantiation of a simulator object
        # 4. When the components get serialized, we can throw away the env without any burden.
        #    (though this part is not implemented yet)
        for obj in [
            state_interpreter,
            action_interpreter,
            reward_fn,
            aux_info_collector,
        ]:
            if obj is not None:
                # 🧠 ML Signal: Use of iterator to fetch initial state
                # 🧠 ML Signal: Passing initial state to simulator function
                # ✅ Best Practice: Use of a structured data type for status management
                obj.env = weakref.proxy(self)  # type: ignore

        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter

        if seed_iterator is None:
            # In this case, there won't be any seed for simulator
            # We can't set it to None because None actually means something else.
            # If `seed_iterator` is None, it means that it's exhausted.
            self.seed_iterator = SEED_INTERATOR_MISSING
        else:
            # ⚠️ SAST Risk (Low): Use of weak references can lead to unexpected behavior if not handled properly
            self.seed_iterator = iter(seed_iterator)
        self.reward_fn = reward_fn

        self.aux_info_collector = aux_info_collector
        # 🧠 ML Signal: Interpretation of simulator state to observation
        self.logger: LogCollector = logger or LogCollector()
        # 🧠 ML Signal: Tracking observation history
        self.status: EnvWrapperStatus = cast(EnvWrapperStatus, None)

    @property
    # ⚠️ SAST Risk (Medium): Potential for NoneType dereference if seed_iterator is None
    def action_space(self) -> Space:
        return self.action_interpreter.action_space

    # 🧠 ML Signal: Handling of exhausted iterator

    # ✅ Best Practice: Resetting logger at the start of the step to ensure clean logging for each step
    @property
    # ⚠️ SAST Risk (Low): Returning NaN can lead to issues if not handled properly downstream
    def observation_space(self) -> Space:
        # 🧠 ML Signal: Tracking action history can be used for behavioral analysis
        return self.state_interpreter.observation_space

    # 🧠 ML Signal: Interpreting actions can be used to understand decision-making processes
    def reset(self, **kwargs: Any) -> ObsType:
        """
        Try to get a state from state queue, and init the simulator with this state.
        If the queue is exhausted, generate an invalid (nan) observation.
        """
        # 🧠 ML Signal: Checking for completion of an episode

        try:
            if self.seed_iterator is None:
                raise RuntimeError(
                    "You can trying to get a state from a dead environment wrapper."
                )
            # 🧠 ML Signal: Interpreting state to observation can be used for state representation learning

            # TODO: simulator/observation might need seed to prefetch something
            # 🧠 ML Signal: Tracking observation history can be used for sequence modeling
            # as only seed has the ability to do the work beforehands

            # ✅ Best Practice: Checking for None before calling a function to avoid errors
            # NOTE: though logger is reset here, logs in this function won't work,
            # because we can't send them outside.
            # See https://github.com/thu-ml/tianshou/issues/605
            self.logger.reset()

            # 🧠 ML Signal: Tracking reward history can be used for reward prediction models
            if self.seed_iterator is SEED_INTERATOR_MISSING:
                # no initial state
                # ✅ Best Practice: Checking for None before calling a function to avoid errors
                initial_state = None
                # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
                self.simulator = cast(Callable[[], Simulator], self.simulator_fn)()
            # 🧠 ML Signal: Logging steps per episode can be used for performance analysis
            # 🧠 ML Signal: Logging reward can be used for reward analysis
            # 🧠 ML Signal: Logging observations and actions for debugging and analysis
            # 🧠 ML Signal: Collecting auxiliary information for additional insights
            # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation may lead to runtime errors if the method is called.
            else:
                initial_state = next(
                    cast(Iterator[InitialStateType], self.seed_iterator)
                )
                self.simulator = self.simulator_fn(initial_state)

            self.status = EnvWrapperStatus(
                cur_step=0,
                done=False,
                initial_state=initial_state,
                obs_history=[],
                action_history=[],
                reward_history=[],
            )

            self.simulator.env = cast(EnvWrapper, weakref.proxy(self))

            sim_state = self.simulator.get_state()
            obs = self.state_interpreter(sim_state)

            self.status["obs_history"].append(obs)

            return obs

        except StopIteration:
            # The environment should be recycled because it's in a dead state.
            self.seed_iterator = None
            return generate_nan_observation(self.observation_space)

    def step(
        self, policy_action: PolicyActType, **kwargs: Any
    ) -> Tuple[ObsType, float, bool, InfoDict]:
        """Environment step.

        See the code along with comments to get a sequence of things happening here.
        """

        if self.seed_iterator is None:
            raise RuntimeError(
                "State queue is already exhausted, but the environment is still receiving action."
            )

        # Clear the logged information from last step
        self.logger.reset()

        # Action is what we have got from policy
        self.status["action_history"].append(policy_action)
        action = self.action_interpreter(self.simulator.get_state(), policy_action)

        # This update must be after action interpreter and before simulator.
        self.status["cur_step"] += 1

        # Use the converted action of update the simulator
        self.simulator.step(action)

        # Update "done" first, as this status might be used by reward_fn later
        done = self.simulator.done()
        self.status["done"] = done

        # Get state and calculate observation
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status["obs_history"].append(obs)

        # Reward and extra info
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            # No reward. Treated as 0.
            rew = 0.0
        self.status["reward_history"].append(rew)

        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}

        # Final logging stuff: RL-specific logs
        if done:
            self.logger.add_scalar("steps_per_episode", self.status["cur_step"])
        self.logger.add_scalar("reward", rew)
        self.logger.add_any("obs", obs, loglevel=LogLevel.DEBUG)
        self.logger.add_any("policy_act", policy_action, loglevel=LogLevel.DEBUG)

        info_dict = InfoDict(log=self.logger.logs(), aux_info=aux_info)
        return obs, rew, done, info_dict

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented in EnvWrapper.")
