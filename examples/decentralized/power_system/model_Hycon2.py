from typing import Any, Dict, List, Optional, Tuple, Union

import casadi as cs
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

np.random.seed(1)

# Model from
# Hycon2 benchmark paper 2012 S. Riverso, G. Ferrari-Tracate

# real parameters of the power system - each is a list containing value for each of the four areas

n = 4
nx_l = 4
nu_l = 1

Adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

# Lists include 5 entries as up to 5 agents can be used
H_list = [12.0, 10, 8, 8, 10]
R_list = [0.05, 0.0625, 0.08, 0.08, 0.05]
D_list = [0.7, 0.9, 0.9, 0.7, 0.86]
T_t_list = [0.65, 0.4, 0.3, 0.6, 0.8]
T_g_list = [0.1, 0.1, 0.1, 0.1, 0.15]
P_tie = np.array(
    [
        [0, 4, 0, 0, 0],
        [4, 0, 2, 0, 3],
        [0, 2, 0, 2, 0],
        [0, 0, 2, 0, 3],
        [0, 3, 0, 3, 0],
    ]
)  # entri (i,j) represent P val between areas i and j
# P_tie = np.array(
#    [
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#    ]
# )
ts = 1  # time-step

# construct real dynamics - subscript l is for local components


def dynamics_from_parameters(
    H: list[float],
    R: list[float],
    D: list[float],
    T_t: list[float],
    T_g: list[float],
    P_tie: np.ndarray,
    ts: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_l = [
        np.array(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie[i, :n]) / (2 * H[i]),
                    -D[i] / (2 * H[i]),
                    1 / (2 * H[i]),
                    0,
                ],
                [0, 0, -1 / T_t[i], 1 / T_t[i]],
                [0, -1 / (R[i] * T_g[i]), 0, -1 / T_g[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [0], [1 / T_g[i]]]) for i in range(n)]
    L_l = [np.array([[0], [-1 / (2 * H[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            np.array(
                [
                    [0, 0, 0, 0],
                    [P_tie[i, j] / (2 * H[i]), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    # global
    A = np.vstack(
        (
            np.hstack((A_l[0], A_c[0][1], A_c[0][2], A_c[0][3])),
            np.hstack((A_c[1][0], A_l[1], A_c[1][2], A_c[1][3])),
            np.hstack((A_c[2][0], A_c[2][1], A_l[2], A_c[2][3])),
            np.hstack((A_c[3][0], A_c[3][1], A_c[3][2], A_l[3])),
        )
    )
    B = np.vstack(
        (
            np.hstack((B_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), B_l[1], np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), B_l[2], np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), B_l[3])),
        )
    )
    L = np.vstack(
        (
            np.hstack((L_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), L_l[1], np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), L_l[2], np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), L_l[3])),
        )
    )

    # disrctised
    A_d, B_d = forward_euler(A, B, ts)
    L_d = ts * L
    # A_d, B_d = zero_order_hold(A, B, ts)
    # L_d = np.linalg.inv(A)@(np.exp(A*ts) - np.eye(A.shape[0]))@L

    return A_d, B_d, L_d


def learnable_dynamics_from_parameters(
    H: list[cs.SX],
    R: list[cs.SX],
    D: list[cs.SX],
    T_t: list[cs.SX],
    T_g: list[cs.SX],
    P_tie_list_list: list[list[cs.SX]],
    ts: float,
):
    A_l = [
        np.array(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie_list_list[i]) / (2 * H[i]),
                    -D[i] / (2 * H[i]),
                    1 / (2 * H[i]),
                    0,
                ],
                [0, 0, -1 / T_t[i], 1 / T_t[i]],
                [0, -1 / (R[i] * T_g[i]), 0, -1 / T_g[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [0], [1 / T_g[i]]]) for i in range(n)]
    L_l = [np.array([[0], [-1 / (2 * H[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            np.array(
                [
                    [0, 0, 0, 0],
                    [P_tie_list_list[i][j] / (2 * H[i]), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    # global
    A = cs.vertcat(
        cs.horzcat(A_l[0], A_c[0][1], A_c[0][2], A_c[0][3]),
        cs.horzcat(A_c[1][0], A_l[1], A_c[1][2], A_c[1][3]),
        cs.horzcat(A_c[2][0], A_c[2][1], A_l[2], A_c[2][3]),
        cs.horzcat(A_c[3][0], A_c[3][1], A_c[3][2], A_l[3]),
    )
    B = cs.vertcat(
        cs.horzcat(B_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), B_l[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), B_l[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), B_l[3]),
    )
    L = cs.vertcat(
        cs.horzcat(L_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), L_l[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), L_l[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), L_l[3]),
    )

    # disrctised
    A_d, B_d = forward_euler(A, B, ts)
    L_d = ts * L

    return A_d, B_d, L_d


def get_cent_model() -> Tuple[np.ndarray, np.ndarray]:
    return dynamics_from_parameters(
        H_list, R_list, D_list, T_t_list, T_g_list, P_tie, ts
    )


def get_model_dims() -> Tuple[int, int, int]:
    return n, nx_l, nu_l, Adj


# initial guesses for each learnable parameter for each agent
pars_init = [
    {
        "H": (H_list[i] + np.random.normal(0, 1)) * np.ones((1,)),
        "R": (R_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "D": (D_list[i] + np.random.normal(0, 2)) * np.ones((1,)),
        "T_t": (T_t_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "T_g": (T_g_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
    }
    for i in range(n)
]


def get_pars_init_list() -> list[Dict]:
    return pars_init


def get_P_tie_init() -> np.ndarray:
    mean = 0
    dev = 2
    P_tie_init = P_tie.copy()
    for i in range(n):
        for j in range(n):
            if P_tie_init[i, j] != 0:
                P_tie_init[i, j] += np.random.normal(mean, dev)
    return P_tie_init


def get_learnable_dynamics(
    H_list: list[cs.SX],
    R_list: list[cs.SX],
    D_list: list[cs.SX],
    T_t_list: list[cs.SX],
    T_g_list: list[cs.SX],
    P_tie_list_list: list[list[cs.SX]],
):
    A, B, L = learnable_dynamics_from_parameters(
        H_list, R_list, D_list, T_t_list, T_g_list, P_tie_list_list, ts
    )
    return A, B, L
