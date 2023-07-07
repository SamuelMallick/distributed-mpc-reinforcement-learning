import contextlib
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import casadi as cs
from csnlp.wrappers.wrapper import Nlp
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import pandas as pd

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
from model_Hycon2 import (
    get_cent_model,
    get_model_dims,
    get_pars_init_list,
    get_learnable_dynamics,
    get_P_tie_init,
)

import pickle
import datetime

np.random.seed(1)

CENTRALISED = True

n, nx_l, nu_l, Adj = get_model_dims()  # Adj is adjacency matrix
u_lim = 0.5


class PowerSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Discrete time network of four power system areas connected with tie lines."""

    def set_points(self, load_val):
        # set points for the system
        x_o_val = np.array(
            [
                [
                    0,
                    0,
                    load_val[0, :].item(),
                    load_val[0, :].item(),
                    0,
                    0,
                    load_val[1, :].item(),
                    load_val[1, :].item(),
                    0,
                    0,
                    load_val[2, :].item(),
                    load_val[2, :].item(),
                    0,
                    0,
                    load_val[3, :].item(),
                    load_val[3, :].item(),
                ]
            ]
        ).T
        u_o_val = load_val.copy()
        return x_o_val, u_o_val

    A, B, L = get_cent_model()  # Get centralised model

    # stage cost params
    Q_l = np.diag((500, 0.01, 0.01, 10))
    Q = block_diag(*([Q_l] * n))
    R_l = 10
    R = block_diag(*([R_l] * n))

    load = np.array([[0], [0], [0], [0]])  # load ref points
    x_o = np.zeros((n * nx_l, 1))
    u_o = np.zeros((n * nu_l, 1))


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.zeros((n * nx_l, 1))
        self.load = np.random.uniform(-0.25, 0.25, (n, 1))
        self.x_o, self.u_o = self.set_points(self.load)
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes stage cost L(s,a)"""
        return (state - self.x_o).T @ self.Q @ (state - self.x_o) + (
            action - self.u_o
        ).T @ self.R @ (action - self.u_o)

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        action = action.full()
        x_new = self.A @ self.x + self.B @ action + self.L @ self.load

        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = 15
    discount_factor = 0.9

    # define which params are learnable
    to_learn = []
    to_learn = [f"H_{i}" for i in range(n)]
    #to_learn = to_learn + [f"R_{i}" for i in range(n)]
    to_learn = to_learn + [f"D_{i}" for i in range(n)]
    #to_learn = to_learn + [f"T_t_{i}" for i in range(n)]
    #to_learn = to_learn + [f"T_g_{i}" for i in range(n)]

    # initialise parameters vals

    learnable_pars_init = {}
    fixed_pars_init = {
        "load": np.zeros((n, 1)),
        "x_o": np.zeros((n * nx_l, 1)),
        "u_o": np.zeros((n * nu_l, 1)),
    }

    # model params
    pars_init_list = get_pars_init_list()
    for i in range(n):
        for name, val in pars_init_list[i].items():
            if f"{name}_{i}" in to_learn:
                learnable_pars_init[f"{name}_{i}"] = val
            else:
                fixed_pars_init[f"{name}_{i}"] = val

    # coupling params
    P_tie_init = get_P_tie_init()
    for i in range(n):
        for j in range(n):
            if Adj[i, j] == 1:
                learnable_pars_init[f"P_tie_{i}_{j}"] = P_tie_init[i, j]

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init params

        H_list = [self.parameter(f"H_{i}", (1,)) for i in range(n)]
        R__list = [self.parameter(f"R_{i}", (1,)) for i in range(n)]
        D_list = [self.parameter(f"D_{i}", (1,)) for i in range(n)]
        T_t_list = [self.parameter(f"T_t_{i}", (1,)) for i in range(n)]
        T_g_list = [self.parameter(f"T_g_{i}", (1,)) for i in range(n)]
        P_tie_list_list = []
        for i in range(n):
            P_tie_list_list.append([])
            for j in range(n):
                if Adj[i, j] == 1:
                    P_tie_list_list[i].append(self.parameter(f"P_tie_{i}_{j}", (1,)))
                else:
                    P_tie_list_list[i].append(0)

        A, B, L = get_learnable_dynamics(
            H_list, R__list, D_list, T_t_list, T_g_list, P_tie_list_list
        )

        load = self.parameter("load", (n, 1))
        x_o = self.parameter("x_o", (n * nx_l, 1))
        u_o = self.parameter("u_o", (n * nu_l, 1))

        # mpc vars

        x, _ = self.state("x", n * nx_l)
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)
        s, _, _ = self.variable("s", (n * nx_l, N), lb=0)

        # TODO add state constraints

        # trivial terminal constraint
        self.constraint("X_f", x[:, [N]] - x_o, "==", 0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + L @ load, n_in=2, n_out=1)

        # objective

        # TODO make objective learnable
        Q_l = np.diag((500, 0.01, 0.01, 10))
        Q = block_diag(*([Q_l] * n))
        R_l = 10
        R = block_diag(*([R_l] * n))

        self.minimize(
            sum(
                (gamma**k)
                * (
                    (x[:, k] - x_o).T @ Q @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ R @ (u[:, k] - u_o)
                )
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


# override the learning agent to check for new load values each iter
class LoadedLstdQLearningAgent(LstdQLearningAgent):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o

        return super().on_timestep_end(env, episode, timestep)
    
    def on_episode_start(self, env, episode: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o
        
        return super().on_episode_start(env, episode)


# centralised
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)
ep_len = int(5e1)
env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(ep_len)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LoadedLstdQLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            discount_factor=mpc.discount_factor,
            update_strategy=1,
            learning_rate=ExponentialScheduler(1e-1, factor=1),
            hessian_type="none",
            record_td_errors=True,
            exploration=None,
            experience=None,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

num_eps = 100
agent.train(env=env, episodes=num_eps, seed=1)

# extract data
if len(env.observations) > 0:
    X = np.hstack([env.observations[i].squeeze().T for i in range(num_eps)]).T
    U = np.hstack([env.actions[i].squeeze().T for i in range(num_eps)]).T
    R = np.hstack([env.rewards[i].squeeze().T for i in range(num_eps)]).T
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)
TD = np.squeeze(agent.td_errors)
TD_eps = [sum(TD[ep_len*i:ep_len*(i+1)]) for i in range(num_eps)]
R_eps = [sum(R[ep_len*i:ep_len*(i+1)]) for i in range(num_eps)]
param_list = [
    np.asarray(agent.updates_history[f"P_tie_{i}_{j}"])
    for j in range(n)
    for i in range(n)
    if Adj[i, j] != 0
]
param_list = param_list + [
    np.asarray(agent.updates_history[name]) for name in mpc.to_learn
]
time = np.arange(R.size)
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD, "o", markersize=1)
axs[1].semilogy(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD_eps, "o", markersize=1)
axs[1].semilogy(R_eps, "o", markersize=1)

idx = 1  # index of agent to plot
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(X)
axs[1].plot(U)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for param in param_list:
    axs.plot(param)
plt.show()
