# Modifying the simple 2D system from David Mayne's
# 2003 paper on PWa systems, adapted to by a network
from csnlp import Nlp
import gymnasium as gym
import numpy as np
import casadi as cs
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.linalg import block_diag

from rldmpc.mpc.mpc_admm import MpcAdmm

n = 3  # number of agents
nx_l = 2
nu_l = 1
S = [np.array([[1, 0]]), np.array([[-1, 0]])]
R = [np.zeros((1, nu_l)), np.zeros((1, nu_l))]
T = [np.array([[1]]), np.array([[-1]])]
A = [np.array([[1, 0.2], [0, 1]]), np.array([[0.5, 0.2], [0, 1]])]
B = [np.array([[0], [1]]), np.array([[0], [1]])]
c = [np.zeros((nx_l, 1)), np.array([[0.5], [0]])]

D = np.array([[-1, 1], [-3, -1], [0.2, 1], [-1, 0], [1, 0], [0, -1]])
E = np.array([[15], [25], [9], [6], [8], [10]])
F = np.array([[1], [-1]])
G = np.array([[1], [1]])

Adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # adjacency matrix
Ac = np.array([[0, 0], [0, 0.1]])  # coupling matrix between any agents

Q_x_l = np.eye(nx_l)
Q_u_l = 0.1 * np.eye(nu_l)
N = 5

Q_x = block_diag(*([Q_x_l] * n))
Q_u = block_diag(*([Q_u_l] * n))


class LtiSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A discrete time network of LTI systems."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.tile([-5, 9], self.n).reshape(nx_l * n, 1)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""
        return state.T @ Q_x @ state + action.T @ Q_u @ action

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""
        action = action.full()
        x_new = np.zeros((nx_l * n, 1))
        for i in range(n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            local_action = action[nu_l * i : nu_l * (i + 1), :]
            for j in range(len(S)):
                if all(S[i] @ local_state + R[i] @ local_action <= T[i]):
                    x_new[nx_l * i : nx_l * (i + 1), :] = (
                        A[i] @ local_state + B[i] @ local_action + c[i]
                    )
            coupling = np.zeros((nx_l, 1))
            for j in range(n):
                if Adj[i, j] == 1:
                    coupling += Ac @ self.x[nx_l * j : nx_l * (j + 1), :]
            x_new[nx_l * i : nx_l * (i + 1), :] += coupling
        r = self.get_stage_cost(self.x, action)
        self.x = x_new

        return x_new, r, False, False, {}


class SwitchingMpc(MpcAdmm):
    """An admm based mpc with constraints and dynamics at every time-step
    that switch."""

    rho = 0.5
    horizon = N

    def __init__(self, num_neighbours, my_index) -> None:
        """Instantiate inner MPC for admm.
        My index is used to pick out own state from the grouped coupling states.
        It should be passed in via the mapping G (G[i].index(i))"""

        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        self.nx_l = nx_l
        self.nu_l = nu_l

        # create the params for switching dynamics and contraints
        A_list = []
        B_list = []
        c_list = []
        S_list = []
        R_list = []
        T_list = []
        r = T[0].shape[0]  # number of conditions when constraining a region
        for k in range(N):
            A_list.append(self.parameter(f"A_{k}", (nx_l, nx_l)))
            B_list.append(self.parameter(f"B_{k}", (nx_l, nu_l)))
            c_list.append(self.parameter(f"c_{k}", (nx_l, 1)))
            S_list.append(self.parameter(f"S_{k}", (r, nx_l)))
            R_list.append(self.parameter(f"R_{k}", (r, nu_l)))
            T_list.append(self.parameter(f"T_{k}", (r, 1)))

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
        )

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        # dynamics and region constraints - added manually due to coupling
        for k in range(N):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += Ac @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A_list[k] @ x[:, [k]] + B_list[k] @ u[:, [k]] + c_list[k] + coup,
                "==",
                x[:, [k + 1]],
            )
            self.constraint(
                f"region_{k}",
                S_list[k] @ x[:, [k]] + R_list[k] @ u[:, [k]],
                "<=",
                T_list[k],
            )
            # also normal state and control constraints
            self.constraint(f"state_{k}", D @ x[:, [k]], "<=", E)
            self.constraint(f"control_{k}", F @ u[:, [k]], "<=", G)

        # objective
        self.set_local_cost(
            sum(
                x[:, [k]].T @ Q_x_l @ x[:, [k]] + u[:, [k]].T @ Q_u_l @ u[:, [k]]
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
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")
