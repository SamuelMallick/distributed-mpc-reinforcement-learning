# Modifying the simple 2D system from David Mayne's
# 2003 paper on PWa systems, adapted to by a network
from csnlp import Nlp
import gymnasium as gym
import numpy as np
import casadi as cs
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.linalg import block_diag
from mpcrl.wrappers.envs import MonitorEpisodes
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log, RecordUpdates
import logging
from rldmpc.agents.mld_agent import MldAgent
from rldmpc.agents.sqp_admm_coordinator import SqpAdmmCoordinator
from rldmpc.mpc.mpc_admm import MpcAdmm
from rldmpc.core.admm import g_map
from rldmpc.agents.g_admm_coordinator import GAdmmCoordinator
import matplotlib.pyplot as plt
from rldmpc.mpc.mpc_mld import MpcMld

from rldmpc.mpc.mpc_switching import MpcSwitching
from rldmpc.utils.pwa_models import cent_from_dist

SIM_TYPE = "sqp_admm"  # options: "mld", "g_admm", "sqp_admm"

# create system

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

# this part of the system is common to everyone as they share the same dynamics
system = {
    "S": S,
    "R": R,
    "T": T,
    "A": A,
    "B": B,
    "c": c,
    "D": D,
    "E": E,
    "F": F,
    "G": G,
}
systems = []  # list of systems, 1 for each agent
for i in range(n):
    systems.append(system.copy())
    # add the coupling part of the system
    Ac_i = []
    for j in range(n):
        if Adj[i, j] == 1:
            Ac_i.append(Ac)
    systems[i]["Ac"] = []
    for j in range(
        len(S)
    ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
        systems[i]["Ac"] = systems[i]["Ac"] + [Ac_i]

cent_system = cent_from_dist(systems, Adj)

Q_x_l = np.eye(nx_l)
Q_u_l = 0.1 * np.eye(nu_l)
N = 10

Q_x = block_diag(*([Q_x_l] * n))
Q_u = block_diag(*([Q_u_l] * n))

G_map = g_map(Adj)


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
        # self.x = np.tile([-5, 9], n).reshape(nx_l * n, 1)
        self.x = np.array([[-5, 9, 4, -2, 1, 3]]).T
        # self.x = np.array([[0, 0, 6, 0, 0, 1]]).T
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
                if all(S[j] @ local_state + R[j] @ local_action <= T[j]):
                    x_new[nx_l * i : nx_l * (i + 1), :] = (
                        A[j] @ local_state + B[j] @ local_action + c[j]
                    )
            coupling = np.zeros((nx_l, 1))
            for j in range(n):
                if Adj[i, j] == 1:
                    coupling += Ac @ self.x[nx_l * j : nx_l * (j + 1), :]
            x_new[nx_l * i : nx_l * (i + 1), :] += coupling
        r = self.get_stage_cost(self.x, action)
        self.x = x_new

        return x_new, r, False, False, {}


class LocalMpc(MpcSwitching):
    rho = 0.5
    horizon = N

    def __init__(self, num_neighbours, my_index) -> None:
        """Instantiate inner switching MPC for admm.
        My index is used to pick out own state from the grouped coupling states.
        It should be passed in via the mapping G (G[i].index(i))"""

        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        self.nx_l = nx_l
        self.nu_l = nu_l

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

        r = T[0].shape[0]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N):
            self.constraint(f"state_{k}", D @ x[:, [k]], "<=", E)
            self.constraint(f"control_{k}", F @ u[:, [k]], "<=", G)

        # objective
        self.set_local_cost(
            sum(
                x[:, [k]].T @ Q_x_l @ x[:, [k]] + u[:, [k]].T @ Q_u_l @ u[:, [k]]
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
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


# distributed mpcs and params
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(n):
    local_mpcs.append(
        LocalMpc(num_neighbours=len(G_map[i]) - 1, my_index=G_map[i].index(i))
    )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)

# mld mpc
mld_mpc = MpcMld(cent_system, N)
mld_mpc.set_cost(Q_x, Q_u)

# env
env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=int(20)))

if SIM_TYPE == "mld":
    agent = MldAgent(mld_mpc)
elif SIM_TYPE == "g_admm":
    # coordinator
    agent = Log(
        GAdmmCoordinator(
            local_mpcs,
            local_fixed_dist_parameters,
            systems,
            G_map,
            Adj,
            local_mpcs[0].rho,
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 200},
    )
elif SIM_TYPE == "sqp_admm":
    # coordinator
    agent = Log(
        SqpAdmmCoordinator(
            local_mpcs,
            local_fixed_dist_parameters,
            systems,
            G_map,
            Adj,
            local_mpcs[0].rho,
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 200},
    )

agent.evaluate(env=env, episodes=1, seed=1)


if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[:, [0, 2, 4]])
axs[1].plot(X[:, [1, 3, 5]])
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
