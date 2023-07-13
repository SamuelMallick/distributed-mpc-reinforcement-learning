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
theta_lim = 0.1
w = 100 * np.ones((n, 1))  # penalty on state viols


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

    load_noise_bnd = 1e-1  # uniform noise bound on load noise

    phi_weight = 0  # weight given to power transfer term in stage cost
    P_tie_list = get_P_tie_init()  # true power transfer coefficients

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.zeros((n * nx_l, 1))
        #self.load = np.random.uniform(-0.15, 0.15, (n, 1))
        self.load = np.array([[0, 0.6, 0.3, -0.1]]).T
        self.x_o, self.u_o = self.set_points(self.load)
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes stage cost L(s,a)"""
        return (
            (state - self.x_o).T @ self.Q @ (state - self.x_o)
            + (action - self.u_o).T @ self.R @ (action - self.u_o)
            + self.phi_weight
            * (  # power transfer term
                sum(
                    np.abs(self.P_tie_list[i, j] * (state[i*nx_l]- state[j*nx_l]))
                    for j in range(n)
                    for i in range(n)
                    if Adj[i, j] == 1
                )
            )
            # pulling out thetas via slice [0, 4, 8, 12]
            + w.T @ np.maximum(0, -np.ones((n, 1)) * theta_lim - state[[0, 4, 8, 12]])
            + w.T @ np.maximum(0, state[[0, 4, 8, 12]] - np.ones((n, 1)) * theta_lim)
        )

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        action = action.full()

        load_noise = np.random.uniform(
            0, self.load_noise_bnd, (n, 1)
        )

        x_new = self.A @ self.x + self.B @ action + self.L @ (self.load + load_noise)

        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}


class MPCAdmm(Mpc[cs.SX]):
    """MPC for agent inner prob in ADMM."""

    rho = 0.5

    horizon = 15
    discount_factor = 0.9

    # define learnable parameters

    to_learn = []
    to_learn = to_learn + ["H"]
    to_learn = to_learn + ["D"]
    to_learn = to_learn + ["T_t"]
    to_learn = to_learn + ["T_g"]
    to_learn = to_learn + ["theta_lb_"]
    to_learn = to_learn + ["theta_ub_"]
    to_learn = to_learn + ["V0"]
    to_learn = to_learn + ["b"]
    to_learn = to_learn + ["f_x"]
    to_learn = to_learn + ["f_u"]
    to_learn = to_learn + ["Q_x"]
    to_learn = to_learn + ["Q_u"]

    def __init__(self, num_neighbours, my_index, P_tie_init) -> None:
        """Instantiate inner MPC for admm. My index is used to pick out own state from the grouped coupling states. It should be passed in via the mapping G (G[i].index(i))"""

        # add coupling to learn
        for i in range(num_neighbours):
            to_learn = to_learn + ["P_tie_{i}"]

        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init param vals

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
                if name in to_learn:
                    learnable_pars_init[name] = val
                else:
                    fixed_pars_init[name] = val


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = 15
    discount_factor = 0.9

    b_scaling = 0.1    # scale the learnable model offset to prevent instability

    # define which params are learnable
    to_learn = []
    # to_learn = [f"H_{i}" for i in range(n)]
    # to_learn = to_learn + [f"R_{i}" for i in range(n)]
    # to_learn = to_learn + [f"D_{i}" for i in range(n)]
    # to_learn = to_learn + [f"T_t_{i}" for i in range(n)]
    # to_learn = to_learn + [f"T_g_{i}" for i in range(n)]
    to_learn = to_learn + [f"theta_lb_{i}" for i in range(n)]
    to_learn = to_learn + [f"theta_ub_{i}" for i in range(n)]
    to_learn = to_learn + [f"V0_{i}" for i in range(n)]
    to_learn = to_learn + [f"b_{i}" for i in range(n)]
    to_learn = to_learn + [f"f_x_{i}" for i in range(n)]
    to_learn = to_learn + [f"f_u_{i}" for i in range(n)]
    to_learn = to_learn + [f"Q_x_{i}" for i in range(n)]
    to_learn = to_learn + [f"Q_u_{i}" for i in range(n)]
    to_learn = to_learn + [
        f"P_tie_{i}_{j}" for j in range(n) for i in range(n) if Adj[i, j] == 1
    ]

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
                if f"P_tie_{i}_{j}" in to_learn:
                    learnable_pars_init[f"P_tie_{i}_{j}"] = P_tie_init[i, j]
                else:
                    fixed_pars_init[f"P_tie_{i}_{j}"] = P_tie_init[i, j]

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

        theta_lb = [self.parameter(f"theta_lb_{i}", (1,)) for i in range(n)]
        theta_ub = [self.parameter(f"theta_ub_{i}", (1,)) for i in range(n)]

        V0 = [self.parameter(f"V0_{i}", (1,)) for i in range(n)]
        b = [self.parameter(f"b_{i}", (nx_l,)) for i in range(n)]
        f_x = [self.parameter(f"f_x_{i}", (nx_l,)) for i in range(n)]
        f_u = [self.parameter(f"f_u_{i}", (nu_l,)) for i in range(n)]
        Q_x = [self.parameter(f"Q_x_{i}", (nx_l, nx_l)) for i in range(n)]
        Q_u = [self.parameter(f"Q_u_{i}", (nu_l, nu_l)) for i in range(n)]

        # mpc vars

        x, _ = self.state("x", n * nx_l)
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)
        s, _, _ = self.variable(
            "s", (n, N), lb=0,
        )  # n in first dim as only cnstr on theta

        # state constraints

        for i in range(n):  # only a constraint on theta
            for k in range(1, N):
                self.constraint(
                    f"theta_lb_{i}_{k}",
                    -theta_lim + theta_lb[i] - s[i, k],
                    "<=",
                    x[i * nx_l, k],
                )
                self.constraint(
                    f"theta_ub_{i}_{k}",
                    x[i * nx_l, k],
                    "<=",
                    theta_lim + theta_ub[i] + s[i, k],
                )

        # trivial terminal constraint

        #self.constraint("X_f", x[:, [N]] - x_o, "==", 0)

        # dynamics

        b_full = cs.SX()
        for i in range(n):
            b_full = cs.vertcat(b_full, b[i])
        self.set_dynamics(
            lambda x, u: A @ x + B @ u + L @ load + self.b_scaling * b_full, n_in=2, n_out=1
        )

        # objective

        Q_x_full = cs.diagcat(*Q_x)
        Q_u_full = cs.diagcat(*Q_u)

        f_x_full = cs.SX()
        f_u_full = cs.SX()
        for i in range(n):
            f_x_full = cs.vertcat(f_x_full, f_x[i])
            f_u_full = cs.vertcat(f_u_full, f_u[i])

        self.minimize(
            sum(V0)
            + sum(
                f_x_full.T @ x[:, k]
                + f_u_full.T @ u[:, k]
                + (gamma**k)
                * (
                    (x[:, k] - x_o).T @ Q_x_full @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ Q_u_full @ (u[:, k] - u_o)
                    + w.T @ s[:, [k]]
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
ep_len = int(20e0)
env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(ep_len)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LoadedLstdQLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            discount_factor=mpc.discount_factor,
            update_strategy=ep_len,
            learning_rate=ExponentialScheduler(1e-5, factor=1),
            hessian_type="none",
            record_td_errors=True,
            exploration=#None,
            EpsilonGreedyExploration(  # None,  # None,  # None,
                epsilon=ExponentialScheduler(
                    0.5, factor=0.99
                ), 
                strength=0.1 * (2*u_lim),
                seed=1,
            ),
            experience=#None,
            ExperienceReplay(
                maxlen=ep_len, sample_size=0.5, include_latest=0.1, seed=1
            ),
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

num_eps = 50
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
TD_eps = [sum(np.abs(TD[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
R_eps = [sum(np.abs(R[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
param_list = []
param_list = param_list + [
    np.asarray(agent.updates_history[name]) for name in mpc.to_learn
]
time = np.arange(R.size)
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD, "o", markersize=1)
axs[1].plot(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD_eps, "o", markersize=1)
axs[1].semilogy(R_eps, "o", markersize=1)

_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    axs[0].plot(X[:, i * nx_l])
    axs[0].axhline(theta_lim, color="r")
    axs[0].axhline(-theta_lim, color="r")
    axs[1].plot(X[:, i * (nx_l) + 1])
    axs[2].plot(X[:, i * (nx_l) + 2])
    axs[3].plot(X[:, i * (nx_l) + 3])
axs[4].plot(U)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for param in param_list:
    if len(param.shape) <= 2:  # TODO dont skip plotting Q
        axs.plot(param.squeeze())
plt.show()
