import contextlib
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import casadi as cs
from csnlp.wrappers.wrapper import Nlp
import gymnasium as gym
import matplotlib.pyplot as plt

# import networkx as netx
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.util.math import quad_form
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.schedulers import ExponentialScheduler
from scipy.linalg import block_diag
from rldmpc.agents.agent_coordinator import LstdQLearningAgentCoordinator
from rldmpc.core.admm import g_map
from rldmpc.utils.discretisation import zero_order_hold, forward_euler
from model import (
    get_cent_model,
    get_model_dims,
    get_learnable_pars_init_list,
    get_learnable_dynamics,
)

import pickle
import datetime

CENTRALISED = False

n, nx_l, nu_l = get_model_dims()
u_lim = 0.5
load_val = 0

class PowerSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Discrete time network of four power system areas connected with tie lines."""

    A, B, A_load = get_cent_model()  # Get centralised model

    # stage cost params
    Q_l = np.diag((5, 0, 0, 5))
    Q = block_diag(*([Q_l] * n))
    R_l = 1
    R = block_diag(*([R_l] * n))

    load = np.array([[0], [load_val], [0], [0]])  # load ref points
    step_counter = 0
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        super().reset(seed=seed, options=options)
        self.x = np.zeros((n * nx_l, 1))
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes stage cost L(s,a)"""
        return state.T @ self.Q @ state + action.T @ self.R @ action

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        action = action.full()
        x_new = self.A @ self.x + self.B @ action + self.A_load @ self.load

        if self.step_counter == 5:
            d = np.zeros((n*nx_l, 1))
            d[1*nx_l, :] = 0.025
            x_new += d
        self.step_counter += 1

        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = 20
    discount_factor = 1

    # initialise learnable parameters vals

    learnable_pars_init = {}
    learnable_pars_init_list = get_learnable_pars_init_list()
    for i in range(n):
        for name, val in learnable_pars_init_list[i].items():
            if not (
                (name == "T_tie") and (i == 0)
            ):  # first agent has no tie line to control - therefore param fixed to 0
                learnable_pars_init[f"{name}_{i}"] = val

    fixed_pars_init = {
        "load": np.array([[0], [load_val], [0], [0]]),
        "T_tie_0": np.array(
            [0]
        ),  # first agent has no tie line to control - therefore param fixed to 0
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init params

        D_list = [self.parameter(f"D_{i}", (1,)) for i in range(n)]
        R_f_list = [self.parameter(f"R_f_{i}", (1,)) for i in range(n)]
        M_a_list = [self.parameter(f"M_a_{i}", (1,)) for i in range(n)]
        T_CH_list = [self.parameter(f"T_CH_{i}", (1,)) for i in range(n)]
        T_G_list = [self.parameter(f"T_G_{i}", (1,)) for i in range(n)]
        T_tie_list = [self.parameter(f"T_tie_{i}", (1,)) for i in range(n)]

        A, B, A_load = get_learnable_dynamics(
            D_list, R_f_list, M_a_list, T_CH_list, T_G_list, T_tie_list
        )

        load = self.parameter("load", (n, 1))
        # mpc vars

        x, _ = self.state("x", n * nx_l)
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)
        s, _, _ = self.variable("s", (n * nx_l, N), lb=0)

        # TODO add state constraints

        # dynamics
        self.set_dynamics(
            lambda x, u: A @ x + B @ u + A_load @ load, n_in=2, n_out=1
        )

        # objective

        # TODO make objective learnable
        Q_l = np.diag((5, 0, 0, 5))
        Q = block_diag(*([Q_l] * n))
        R_l = 1
        R = block_diag(*([R_l] * n))

        self.minimize(
            sum(
                (gamma**k) * (x[:, k].T @ Q @ x[:, k] + u[:, k].T @ R @ u[:, k])
                for k in range(N)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 1000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


# centralised
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)

env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(8e1)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            discount_factor=mpc.discount_factor,
            update_strategy=1,
            learning_rate=ExponentialScheduler(4e-2, factor=0.99),
            hessian_type="approx",
            record_td_errors=True,
            exploration=None,
            experience=None,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

agent.evaluate(env=env, episodes=1, seed=1)

# extract data
if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0].squeeze()
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)
TD = np.squeeze(agent.td_errors)
time = np.arange(R.size)
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD, "o", markersize=1)
axs[1].semilogy(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

idx = 1  # index of agent to plot
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(X)
axs[1].plot(U)

plt.show()
