from typing import (Any, Collection, Iterable, List, Literal, Optional,
                    Sequence, Union)

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType
from mpcrl.core.exploration import ExplorationStrategy, NoExploration

from rldmpc.mpc.mpc_mld import MpcMld


class MldAgent(Agent):
    """A pwa agent who uses an mld controller."""

    def __init__(
        self,
        mpc: MpcMld,
    ) -> None:
        """Constructor is overriden and the super class' instructor is not called as
        this agent uses an mpc that does not inheret from the MPC baseclass."""
        self._exploration: ExplorationStrategy = NoExploration()  # to keep compatable
        self.mpc = mpc
        self.x_pred = None  # stores most recent predicted state after being solved

    def evaluate(
        self,
        env: Env,
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] = None,
    ):
        """Evaluates the agent in a given environment. Overriding the function of Agent
        to use the mld_mpc instead.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to test the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically; by default, `True`.
        seed : None, int or sequence of ints, optional
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode)."""
        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            self.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode)

            while not (truncated or terminated):
                # changed origonal agents evaluate here to use the mld mpc
                action = self.get_control(state)
                action = cs.DM(action)

                state, r, truncated, terminated, _ = env.step(action)
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns

    def get_control(self, state):
        u, x = self.mpc.solve_mpc(state)
        self.x_pred = x
        return u

    def set_cost(self, Q_x, Q_u, x_goal: np.ndarray = None, u_goal: np.ndarray = None):
        """Set cost of the agents mpc-MIP as sum_k x(k)' * Q_x * x(k) + u(k)' * Q_u * u(k).
        Restricted to quadratic in the states and control.
        If x_goal or u_goal passed the cost uses (x-x_goal) and (u_goal)"""

        self.mpc.set_cost(Q_x, Q_u, x_goal, u_goal)
