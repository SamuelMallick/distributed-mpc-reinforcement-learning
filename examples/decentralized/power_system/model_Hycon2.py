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
from rldmpc.utils.discretisation import zero_order_hold, forward_euler, tustin

np.random.seed(1)

# Model from
# Hycon2 benchmark paper 2012 S. Riverso, G. Ferrari-Tracate

# real parameters of the power system - each is a list containing values for each of the four areas

n = 4
nx_l = 4
nu_l = 1

Adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

H_list = [12.0, 10, 8, 8]
R_list = [0.05, 0.0625, 0.08, 0.08]
D_list = [0.7, 0.9, 0.9, 0.7]
T_t_list = [0.65, 0.4, 0.3, 0.6]
T_g_list = [0.1, 0.1, 0.1, 0.1]
P_tie = np.array(
    [
        [0, 4, 0, 0],
        [4, 0, 2, 0],
        [0, 2, 0, 2],
        [0, 0, 2, 0],
    ]
)  # entri (i,j) represent P val between areas i and j
ts = 1  # time-step for discretisation

def get_P_tie():
    return P_tie


def get_model_details() -> Tuple[int, int, int, np.ndarray, float]:
    return n, nx_l, nu_l, Adj, ts

def get_cent_model(discrete: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get A, B and L matrices for the centralised system Ax + Bu + Ld. If discrete the continuous dynamics are discretised using ZOH."""
    A_l = [
        np.array(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie[i, :n]) / (2 * H_list[i]),
                    -D_list[i] / (2 * H_list[i]),
                    1 / (2 * H_list[i]),
                    0,
                ],
                [0, 0, -1 / T_t_list[i], 1 / T_t_list[i]],
                [0, -1 / (R_list[i] * T_g_list[i]), 0, -1 / T_g_list[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [0], [1 / T_g_list[i]]]) for i in range(n)]
    L_l = [np.array([[0], [-1 / (2 * H_list[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            np.array(
                [
                    [0, 0, 0, 0],
                    [P_tie[i, j] / (2 * H_list[i]), 0, 0, 0],
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
    if not discrete:
        return A, B, L
    else:
        B_comb = np.hstack((B, L))
        A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
        B_d = B_d_comb[:, :n]
        L_d = B_d_comb[:, n:]
        return A_d, B_d, L_d


# initial guesses for each learnable parameter for each agent - except for P_tie
pars_init = [
    {
        "H": (H_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "R": (R_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "D": (D_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "T_t": (T_t_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "T_g": (T_g_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "theta_lb": 0 * np.ones((1,)),
        "theta_ub": 0 * np.ones((1,)),
        "V0": 0 * np.ones((1,)),
        "b": 0 * np.ones((nx_l,)),
        "f_x": 0 * np.ones((nx_l, 1)),
        "f_u": 0 * np.ones((nu_l, 1)),
        "Q_x": np.diag((500, 0.1, 0.1, 10)),
        "Q_u": 10 * np.ones((1,)),
    }
    for i in range(n)
]

def get_pars_init_list() -> list[Dict]:
    """Get initial guesses for learnable parameters (exluding P_tie)."""
    return pars_init


def get_P_tie_init() -> np.ndarray:
    """Get initial guesses for learnable P_tie values."""
    norm_lim = 2
    P_tie_init = P_tie.copy()
    for i in range(n):
        for j in range(n):
            if P_tie_init[i, j] != 0:
                P_tie_init[i, j] += np.random.uniform(-norm_lim, norm_lim)
    return P_tie_init

def get_learnable_dynamics(
    H_list: list[cs.SX],
    R_list: list[cs.SX],
    D_list: list[cs.SX],
    T_t_list: list[cs.SX],
    T_g_list: list[cs.SX],
    P_tie_list_list: list[list[cs.SX]],
):
    """Get symbolic A, B and L matrices for the centralised system Ax + Bu + Ld. Always discretised."""
    A_l = [
        np.array(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie_list_list[i]) / (2 * H_list[i]),
                    -D_list[i] / (2 * H_list[i]),
                    1 / (2 * H_list[i]),
                    0,
                ],
                [0, 0, -1 / T_t_list[i], 1 / T_t_list[i]],
                [0, -1 / (R_list[i] * T_g_list[i]), 0, -1 / T_g_list[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [0], [1 / T_g_list[i]]]) for i in range(n)]
    L_l = [np.array([[0], [-1 / (2 * H_list[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            np.array(
                [
                    [0, 0, 0, 0],
                    [P_tie_list_list[i][j] / (2 * H_list[i]), 0, 0, 0],
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

    # discretise
    B_comb = cs.horzcat(B, L)
    # A_d, B_d_comb = forward_euler(A, B_comb, ts)
    A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
    B_d = B_d_comb[:, :n]
    L_d = B_d_comb[:, n:]
    return A_d, B_d, L_d

def get_learnable_dynamics_local(H, R, D, T_t, T_g, P_tie_list):
    """Get symbolic matrices A_i, B_i, L_i and A_ij for an agent. Always discretised."""
    A = cs.blockcat(
        [
            [0, 1, 0, 0],
            [
                -sum(P_tie_list) / (2 * H),
                -D / (2 * H),
                1 / (2 * H),
                0,
            ],
            [0, 0, -1 / T_t, 1 / T_t],
            [0, -1 / (R * T_g), 0, -1 / T_g],
        ]
    )
    B = cs.blockcat([[0], [0], [0], [1 / T_g]])
    L = cs.blockcat([[0], [-1 / (2 * H)], [0], [0]])

    A_c_list = []
    for i in range(len(P_tie_list)):
        A_c_list.append(
            cs.blockcat( 
                [
                    [0, 0, 0, 0],
                    [P_tie_list[i] / (2 * H), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )

    B_comb = cs.horzcat(B, L, *A_c_list)
    A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
    B_d = B_d_comb[:, :nu_l]
    L_d = B_d_comb[:, nu_l:2*nu_l]
    A_d_c_list = []
    for i in range(len(P_tie_list)):
        A_d_c_list.append(
            B_d_comb[:, 2*nu_l+i*nx_l:2*nu_l+(i+1)*nx_l]
        )
    return A_d, B_d, L_d, A_d_c_list
    