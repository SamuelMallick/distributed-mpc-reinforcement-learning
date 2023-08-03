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
    get_learnable_dynamics_local,
    get_P_tie
)

import pickle
import datetime

np.random.seed(1)

CENTRALISED = False
LEARN = True
SCENARIO_1 = True
SCENARIO_2 = False

STORE_DATA = True
PLOT = False

n, nx_l, nu_l, Adj, ts = get_model_dims()  # Adj is adjacency matrix
u_lim = np.array([[0.2], [0.1], [0.3], [0.1]])
theta_lim = 0.1
w = 500 * np.ones((n, 1))  # penalty on state viols
w_l = 500  # local penalty on state viols
b_scaling = 0.1  # scale the learnable model offset to prevent instability

prediction_length = 5  # length of prediction horizon

# distributed stuff
G = g_map(Adj)
eps = 0.25  # must be less than 0.5 as max neighborhood cardinality is 2
D_in = np.array(
    [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
)  # Hard coded D_in matrix from Adj
L = D_in - Adj  # graph laplacian
P = np.eye(n) - eps * L  # consensus matrix


class PowerSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Discrete time network of four power system areas connected with tie lines."""

    A, B, L = get_cent_model(discrete=False)  # Get continuous centralised model

    # stage cost params
    # Q_x_l = np.diag((0, 1, 0, 0))
    Q_x_l = np.diag((500, 0.01, 0.01, 10))
    Q_x = block_diag(*([Q_x_l] * n))
    Q_u_l = 0
    Q_u = block_diag(*([Q_u_l] * n))

    load = np.array([[0], [0], [0], [0]])  # load ref points
    x_o = np.zeros((n * nx_l, 1))
    u_o = np.zeros((n * nu_l, 1))

    load_noise_bnd = 1e-1  # uniform noise bound on load noise

    phi_weight = 0.5  # weight given to power transfer term in stage cost
    P_tie_list = get_P_tie_init()  # true power transfer coefficients

    step_counter = 1

    def __init__(self) -> None:
        super().__init__()
        x = cs.SX.sym("x", self.A.shape[1])
        u = cs.SX.sym("u", self.B.shape[1])
        l = cs.SX.sym("l", self.L.shape[1])
        p = cs.vertcat(u, l)
        x_new = self.A @ x + self.B @ u + self.L @ l
        ode = {"x": x, "p": p, "ode": x_new}
        self.integrator = cs.integrator(
            "env_integrator",
            "cvodes",
            ode,
            0,
            ts,
            {"abstol": 1e-8, "reltol": 1e-8},
        )

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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.zeros((n * nx_l, 1))
        # self.load = np.random.uniform(-0.15, 0.15, (n, 1))
        #self.load = np.array([[0, 0.6, 0.3, -0.1]]).T
        # self.load = np.array([[0, 0.2, 0.3, -0.1]]).T
        self.load = np.zeros((n, 1))
        self.x_o, self.u_o = self.set_points(self.load)
        self.step_counter = 1
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes stage cost L(s,a)"""
        return (
            (state - self.x_o).T @ self.Q_x @ (state - self.x_o)
            + (action - self.u_o).T @ self.Q_u @ (action - self.u_o)
            + self.phi_weight
            * ts * (  # power transfer term
                sum(
                    np.abs(self.P_tie_list[i, j] * (state[i * nx_l] - state[j * nx_l]))
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
        r = float(self.get_stage_cost(self.x, action))

        if SCENARIO_1:  # Change load according t oscenario
            if self.step_counter == 5:
                self.load = np.array([[0.15, 0, 0, 0]]).T
                self.x_o, self.u_o = self.set_points(self.load)
            elif self.step_counter == 15:
                self.load = np.array([[0.15, -0.15, 0, 0]]).T
                self.x_o, self.u_o = self.set_points(self.load)
            elif self.step_counter == 20:
                self.load = np.array([[0.15, -0.15, 0.12, 0]]).T
                self.x_o, self.u_o = self.set_points(self.load)
            elif self.step_counter == 40:
                self.load = np.array([[0.15, -0.15, -0.12, 0.28]]).T
                self.x_o, self.u_o = self.set_points(self.load)

        load_noise = np.random.uniform(
            -self.load_noise_bnd, self.load_noise_bnd, (n, 1)
        )
        l = self.load + load_noise
        x_new = self.integrator(x0=self.x, p=cs.vertcat(action, l))["xf"]
        self.x = x_new
        self.step_counter += 1
        return x_new, r, False, False, {}


class MPCAdmm(Mpc[cs.SX]):
    """MPC for agent inner prob in ADMM."""

    rho = 50

    horizon = prediction_length
    discount_factor = 0.9

    # define learnable parameters

    to_learn = []
    # to_learn = to_learn + ["H"]
    # to_learn = to_learn + ["D"]
    # to_learn = to_learn + ["T_t"]
    # to_learn = to_learn + ["T_g"]
    to_learn = to_learn + ["theta_lb"]
    to_learn = to_learn + ["theta_ub"]
    to_learn = to_learn + ["V0"]
    to_learn = to_learn + ["b"]
    to_learn = to_learn + ["f_x"]
    to_learn = to_learn + ["f_u"]
    to_learn = to_learn + ["Q_x"]
    to_learn = to_learn + ["Q_u"]

    def __init__(self, num_neighbours, my_index, pars_init, P_tie_init, u_lim) -> None:
        """Instantiate inner MPC for admm. My index is used to pick out own state from the grouped coupling states. It should be passed in via the mapping G (G[i].index(i))"""

        # add coupling to learn
        for i in range(num_neighbours):
            self.to_learn = self.to_learn + [f"P_tie_{i}"]

        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init param vals

        self.learnable_pars_init = {}
        self.fixed_pars_init = {
            "load": np.zeros((n, 1)),
            "x_o": np.zeros((n * nx_l, 1)),
            "u_o": np.zeros((n * nu_l, 1)),
            "y": np.zeros(
                (nx_l * (num_neighbours + 1), N)
            ),  # lagrange multipliers ADMM
            "z": np.zeros((nx_l * (num_neighbours + 1), N)),  # global consensus vars
        }

        # model params
        for name, val in pars_init.items():
            if name in self.to_learn:
                self.learnable_pars_init[name] = val
            else:
                self.fixed_pars_init[name] = val
        for i in range(num_neighbours):
            if f"P_tie_{i}" in self.to_learn:
                self.learnable_pars_init[f"P_tie_{i}"] = P_tie_init[i]
            else:
                self.fixed_pars_init[f"P_tie_{i}"] = P_tie_init[i]

        # create the params

        H = self.parameter("H", (1,))
        R = self.parameter("R", (1,))
        D = self.parameter("D", (1,))
        T_t = self.parameter("T_t", (1,))
        T_g = self.parameter("T_g", (1,))
        P_tie_list = []
        for i in range(num_neighbours):
            P_tie_list.append(self.parameter(f"P_tie_{i}", (1,)))

        theta_lb = self.parameter(f"theta_lb", (1,))
        theta_ub = self.parameter(f"theta_ub", (1,))

        V0 = self.parameter(f"V0", (1,))
        b = self.parameter(f"b", (nx_l,))
        f_x = self.parameter(f"f_x", (nx_l,))
        f_u = self.parameter(f"f_u", (nu_l,))
        Q_x = self.parameter(f"Q_x", (nx_l, nx_l))
        Q_u = self.parameter(f"Q_u", (nu_l, nu_l))

        y = self.parameter("y", (nx_l * (num_neighbours + 1), N))
        z = self.parameter("z", (nx_l * (num_neighbours + 1), N))

        load = self.parameter("load", (1, 1))
        x_o = self.parameter("x_o", (nx_l, 1))
        u_o = self.parameter("u_o", (nu_l, 1))

        A, B, L, A_c_list = get_learnable_dynamics_local(H, R, D, T_t, T_g, P_tie_list)

        x, _ = self.state("x", nx_l)  # local state
        x_c, _, _ = self.variable("x_c", (nx_l * (num_neighbours), N))  # coupling

        # adding them together as the full decision var
        x_cat = cs.vertcat(
            x_c[: (my_index * nx_l), :], x[:, :-1], x_c[(my_index * nx_l) :, :]
        )
        u, _ = self.action(
            "u",
            nu_l,
            lb=-u_lim,
            ub=u_lim,
        )
        s, _, _ = self.variable("s", (1, N), lb=0)  # dim 1 as only theta has bound

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        # dynamics - added manually due to coupling

        for k in range(N):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                "dynam_" + str(k),
                A @ x[:, [k]] + B @ u[:, [k]] + L @ load + coup + b_scaling * b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints

        self.constraint(f"theta_lb", -theta_lim + theta_lb - s, "<=", x[0, 1:])
        self.constraint(f"theta_ub", x[0, 1:], "<=", theta_lim + theta_ub + s)

        # objective
        self.minimize(
            V0
            + sum(
                f_x.T @ x[:, k]
                + f_u.T @ u[:, k]
                + (gamma**k)
                * (
                    (x[:, k] - x_o).T @ Q_x @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ Q_u @ (u[:, k] - u_o)
                    + w_l * s[:, [k]]
                )
                for k in range(N)
            )
            + sum((y[:, [k]].T @ (x_cat[:, [k]] - z[:, [k]])) for k in range(N))
            + sum(
                ((self.rho / 2) * cs.norm_2(x_cat[:, [k]] - z[:, [k]]) ** 2)
                for k in range(N)
            )
        )

        self.x_dim = (
            x_cat.shape
        )  # assigning it to class so that the dimension can be retreived later by admm procedure
        self.nx_l = nx_l
        self.nu_l = nu_l

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
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = prediction_length
    discount_factor = 0.9

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
            "s",
            (n, N),
            lb=0,
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

        # self.constraint("X_f", x[:, [N]] - x_o, "==", 0)

        # dynamics

        b_full = cs.SX()
        for i in range(n):
            b_full = cs.vertcat(b_full, b[i])
        self.set_dynamics(
            lambda x, u: A @ x + B @ u + L @ load + b_scaling * b_full, n_in=2, n_out=1
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
            + (gamma**N) * (x[:, N] - x_o).T @ Q_x_full @ (x[:, N] - x_o)
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
class LoadedLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o

        if not self.centralised_flag:
            for i in range(n):
                self.agents[i].fixed_parameters["load"] = env.load[i]
                self.agents[i].fixed_parameters["x_o"] = env.x_o[
                    nx_l * i : nx_l * (i + 1)
                ]
                self.agents[i].fixed_parameters["u_o"] = env.u_o[i]

        else:
            return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env, episode: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o

        if not self.centralised_flag:
            for i in range(n):
                self.agents[i].fixed_parameters["load"] = env.load[i]
                self.agents[i].fixed_parameters["x_o"] = env.x_o[
                    nx_l * i : nx_l * (i + 1)
                ]
                self.agents[i].fixed_parameters["u_o"] = env.u_o[i]

        return super().on_episode_start(env, episode)


# decentralised
P_tie_init = get_P_tie_init()
# distributed mpc and params
mpc_dist_list: list[Mpc] = []
learnable_dist_parameters_list: list[LearnableParametersDict] = []
fixed_dist_parameters_list: list = []

pars_init_list = get_pars_init_list()
for i in range(n):
    mpc_dist_list.append(
        MPCAdmm(
            num_neighbours=len(G[i]) - 1,
            my_index=G[i].index(i),
            pars_init=pars_init_list[i],
            P_tie_init=[P_tie_init[i, j] for j in range(n) if Adj[i, j] != 0],
            u_lim=u_lim[i]
        )
    )
    learnable_dist_parameters_list.append(
        LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name, val.shape, val, sym=mpc_dist_list[i].parameters[name]
                )
                for name, val in mpc_dist_list[i].learnable_pars_init.items()
            )
        )
    )
    fixed_dist_parameters_list.append(mpc_dist_list[i].fixed_pars_init)

# centralised
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)
ep_len = int(100e0)
env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(ep_len)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LoadedLstdQLearningAgentCoordinator(
            rho=MPCAdmm.rho,
            n=n,
            G=G,
            P=P,
            centralised_flag=CENTRALISED,
            centralised_debug=False,
            mpc_cent=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            mpc_dist_list=mpc_dist_list,
            learnable_dist_parameters_list=learnable_dist_parameters_list,
            fixed_dist_parameters_list=fixed_dist_parameters_list,
            discount_factor=mpc.discount_factor,
            update_strategy=ep_len,
            learning_rate=ExponentialScheduler(1e-6, factor=1),  # 5e-6
            hessian_type="none",
            record_td_errors=True,
            exploration=#None,
            EpsilonGreedyExploration( 
                epsilon=ExponentialScheduler(0.5, factor=0.8),
                strength=0.1 * (2 * 0.2),
                seed=1,
            ),
            experience=ExperienceReplay(  # None,
                maxlen=3*ep_len, sample_size=int(1.5*ep_len), include_latest=ep_len, seed=1
            ),  # None,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

num_eps = 500
if LEARN:
    agent.train(env=env, episodes=num_eps, seed=1)
else:
    agent.evaluate(env=env, episodes=num_eps, seed=1)

# extract data
if len(env.observations) > 0:
    X = np.hstack([env.observations[i].squeeze().T for i in range(num_eps)]).T
    U = np.hstack([env.actions[i].squeeze().T for i in range(num_eps)]).T
    R = np.hstack([env.rewards[i].squeeze().T for i in range(num_eps)]).T
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)
TD = np.squeeze(agent.td_errors) if CENTRALISED else agent.agents[0].td_errors
TD_eps = [sum((TD[ep_len * i : ep_len * (i + 1)]))/ep_len for i in range(num_eps)]
R_eps = [sum((R[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
param_dict = {}
if CENTRALISED:
    for name in mpc.to_learn:
        param_dict[name] = np.asarray(agent.updates_history[name])
else:
    for i in range(n):
        for name in mpc_dist_list[i].to_learn:
            param_dict[name + '_' + str(i)] = np.asarray(agent.agents[i].updates_history[name])

time = np.arange(R.size)

if STORE_DATA:
    with open(
        "data/power_C_"
        + str(CENTRALISED)
        + datetime.datetime.now().strftime("%d%H%M%S%f")
        + str(".pkl"),
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)

if PLOT:
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD, "o", markersize=1)
    axs[1].plot(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD_eps, "o", markersize=1)
    axs[1].semilogy(R_eps, "o", markersize=1)

    _, axs = plt.subplots(6, 1, constrained_layout=True, sharex=True)
    P_tie = get_P_tie()
    for i in range(n):
        axs[0].plot(X[:, i * nx_l])
        axs[0].axhline(theta_lim, color="r")
        axs[0].axhline(-theta_lim, color="r")
        axs[1].plot(X[:, i * (nx_l) + 1])
        axs[2].plot(X[:, i * (nx_l) + 2])
        axs[3].plot(X[:, i * (nx_l) + 3])
        for j in range(n):
            if P_tie[i, j] != 0:
                axs[5].plot(P_tie[i, j]*(X[:, i * nx_l] - X[:, j * nx_l]))
    axs[4].plot(U)

    if LEARN:
        _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
        for name in param_dict:
            if len(param_dict[name].shape) <= 2:  # TODO dont skip plotting Q
                axs.plot(param_dict[name].squeeze())

    plt.show()
