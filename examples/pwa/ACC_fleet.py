from csnlp import Nlp
import gymnasium as gym
from mpcrl import Agent
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
from rldmpc.systems.ACC import ACC
from rldmpc.utils.pwa_models import cent_from_dist

SIM_TYPE = "mld"  # options: "mld", "g_admm", "sqp_admm"

n = 1  # 1 car
acc = ACC()
nx_l = acc.nx_l
nu_l = acc.nu_l
pwa_system = acc.get_pwa_system()

N = 3
Q_x = np.eye(nx_l)
Q_u = np.eye(nu_l)
goal_pos = 1000

# the true non-linear dynamics of the car
def step_car_dynamics(x, u, ts):
    """Steps the car dynamics by ts seconds. x is state, u is control."""
    num_steps = 100
    DT = ts / num_steps
    for i in range(num_steps):
        f = np.array(
            [[x[1, 0]], [-(acc.c_fric * x[1, 0] ** 2) / (acc.mass) - acc.mu * acc.grav]]
        )
        B = np.array([[0], [-acc.get_traction_force(x[1, 0]) / acc.mass]])
        x = x + DT * (f + B * u)

    return x


class CarFleet(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A fleet of non-linear hybrid vehicles who track each other."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.array([[500], [5]])
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""
        return (state - np.array([[goal_pos], [0]])).T @ Q_x @ (
            state - np.array([[goal_pos], [0]])
        ) + action.T @ Q_u @ action

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)
        x_new = step_car_dynamics(self.x, action, acc.ts)
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

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)
        u, _ = self.action(
            "u",
            nu_l,
        )

        # objective
        self.set_local_cost(0)

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


# mld mpc
mld_mpc = MpcMld(pwa_system, N)
mld_mpc.set_cost(Q_x, Q_u, x_goal=np.array([[goal_pos], [0]]))

# test mpc
mpc = LocalMpc(1, 1)

# env
env = MonitorEpisodes(TimeLimit(CarFleet(), max_episode_steps=int(100)))
if SIM_TYPE == "mld":
    agent = MldAgent(mld_mpc)
elif SIM_TYPE == "g_admm":
    # coordinator
    agent = Agent(mpc, mpc.fixed_pars_init)
elif SIM_TYPE == "sqp_admm":
    # coordinator
    pass

agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[:, 0])
axs[1].plot(X[:, 1])
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
