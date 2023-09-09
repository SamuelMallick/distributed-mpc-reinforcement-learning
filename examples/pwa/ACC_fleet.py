from csnlp import Nlp
import gymnasium as gym
from gymnasium import Env
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

N = 10
Q_x = np.diag([100, 1])
Q_u = np.eye(nu_l)
ep_len = 100  # length of episode (sim len)

# generate trajectory of leader
leader_state = np.zeros((2, ep_len + N + 1))
leader_speed = 15
leader_initial_pos = 500
leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
for k in range(ep_len + N):
    leader_state[:, [k + 1]] = leader_state[:, [k]] + acc.ts * np.array(
        [[leader_speed], [0]]
    )

class CarFleet(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A fleet of non-linear hybrid vehicles who track each other."""

    step_counter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.array([[50], [5]])
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""
        return (state - leader_state[:, [self.step_counter]]).T @ Q_x @ (
            state - leader_state[:, [self.step_counter]]
        ) + action.T @ Q_u @ action

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)
        x_new = acc.step_car_dynamics(self.x, action, acc.ts)
        self.x = x_new

        self.step_counter += 1
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


class TrackingMldAgent(MldAgent):
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.set_cost(Q_x, Q_u, x_goal=[leader_state[:, [k]] for k in range(timestep, timestep+N)])

        return super().on_timestep_end(env, episode, timestep)


# mld mpc
mld_mpc = MpcMld(pwa_system, N)
# initialise the cost with the first tracking point
mld_mpc.set_cost(Q_x, Q_u, x_goal=[leader_state[:, [k]] for k in range(N)])

# test mpc
mpc = LocalMpc(1, 1)

# env
env = MonitorEpisodes(TimeLimit(CarFleet(), max_episode_steps=ep_len))
if SIM_TYPE == "mld":
    agent = TrackingMldAgent(mld_mpc)
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
axs[0].plot(leader_state[0, :], "--")
axs[1].plot(leader_state[1, :], "--")
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
