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
import gurobipy as gp
from rldmpc.mpc.mpc_switching import MpcSwitching
from rldmpc.systems.ACC import ACC
from rldmpc.utils.pwa_models import cent_from_dist

np.random.seed(0)

SIM_TYPE = "mld"  # options: "mld", "g_admm", "sqp_admm"

n = 2  # num cars
acc = ACC()
nx_l = acc.nx_l
nu_l = acc.nu_l
system = acc.get_pwa_system()

N = 5
Q_x_l = np.diag([1, 1])
Q_u_l = np.eye(nu_l)
sep = np.array([[50], [0]])  # desired seperation between vehicles states
d_safe = 0
ep_len = 300  # length of episode (sim len)

NO_OVERTAKING = False
COMFORT = True  # regulate acell to be within bounds

# construct centralised system
systems = []  # list of systems, 1 for each agent
for i in range(n):
    systems.append(system.copy())
    # no state coupling here
    Ac_i = []
    systems[i]["Ac"] = []
    for j in range(
        len(system["S"])
    ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
        systems[i]["Ac"] = systems[i]["Ac"] + [Ac_i]

cent_system = cent_from_dist(systems, np.zeros((n * nx_l, n * nx_l)))

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
        starting_positions = [
            400 * np.random.random() for i in range(n)
        ]  # starting positions between 0-50 m
        self.x = np.tile(np.array([[0], [5]]), (n, 1))
        for i in range(n):
            IC = max(starting_positions)  # order the agents by starting distance
            self.x[i * nx_l, :] = IC
            starting_positions.remove(IC)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""

        cost = 0
        for i in range(n):
            local_state = state[nx_l * i : nx_l * (i + 1), :]
            local_action = action[nu_l * i : nu_l * (i + 1), :]
            if i == 0:
                # first car tracks leader
                follow_state = leader_state[:, [self.step_counter]]
            else:
                # other cars follow the next car
                follow_state = state[nx_l * (i - 1) : nx_l * (i), :]

            cost += (local_state - follow_state - sep).T @ Q_x_l @ (
                local_state - follow_state - sep
            ) + local_action.T @ Q_u_l @ local_action
        return cost

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)
        x_new = acc.step_car_dynamics(self.x, action, n, acc.ts)
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


class MPCMldCent(MpcMld):
    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        self.mpc_model.setObjective(0, gp.GRB.MINIMIZE)

        # add extra constraints
        if COMFORT:
        # acceleration constraints
            for i in range(n):
                for k in range(N):
                    self.mpc_model.addConstr(
                        acc.a_dec * acc.ts
                        <= self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]],
                        name=f"dec_car_{i}_step{k}",
                    )
                    self.mpc_model.addConstr(
                        self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]]
                        <= acc.a_acc * acc.ts,
                        name=f"acc_car_{i}_step{k}",
                    )
        if NO_OVERTAKING:
            # safe distance behind follower vehicle
            leader_traj = np.zeros(
                (nx_l, N)
            )  # fake leader_traj, will get updates each time-step
            for i in range(n):
                local_state = self.x[nx_l * i : nx_l * (i + 1), :]
                if i == 0:
                    self.first_car_safe_constrs = []
                    # first car follows leader and therefore this constraint gets changed when the leader traj gets set
                    follow_state = leader_traj
                    for k in range(N):
                        self.first_car_safe_constrs.append(
                            self.mpc_model.addConstr(
                                local_state[0, [k]] <= follow_state[0, [k]] - d_safe,
                                name=f"safe_dis_car_{i}_step{k}",
                            )
                        )
                else:
                    # otherwise follow car infront (i-1)
                    follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                    for k in range(N):
                        self.mpc_model.addConstr(
                            local_state[0, [k]] <= follow_state[0, [k]] - d_safe,
                            name=f"safe_dis_car_{i}_step{k}",
                        )

    def set_leader_traj(self, leader_traj):
        obj = 0
        for i in range(n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            local_control = self.u[nu_l * i : nu_l * (i + 1), :]
            if i == 0:
                # first car follows leader
                follow_state = leader_traj
                if NO_OVERTAKING:
                    for k in range(N):
                        self.first_car_safe_constrs[k].RHS = (
                            follow_state[0, [k]] + d_safe
                        )
            else:
                # otherwise follow car infront (i-1)
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
            for k in range(N):
                obj += (local_state[:, k] - follow_state[:, k] - sep.T) @ Q_x_l @ (
                    local_state[:, [k]] - follow_state[:, [k]] - sep
                ) + local_control[:, k] @ Q_u_l @ local_control[:, [k]]

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)


class TrackingMldAgent(MldAgent):
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(leader_state[:, timestep : (timestep + N)])
        return super().on_timestep_end(env, episode, timestep)


# mld mpc
mld_mpc = MPCMldCent(cent_system, N)
# initialise the cost with the first tracking point
mld_mpc.set_leader_traj(leader_state[:, 0:N])

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
for i in range(n):
    axs[0].plot(X[:, nx_l * i])
    axs[1].plot(X[:, nx_l * i + 1])
axs[0].plot(leader_state[0, :], "--")
axs[1].plot(leader_state[1, :], "--")
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
