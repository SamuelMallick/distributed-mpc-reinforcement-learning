from csnlp import Nlp

from gymnasium import Env
import numpy as np
import casadi as cs
import numpy.typing as npt
from scipy.linalg import block_diag
from mpcrl.wrappers.envs import MonitorEpisodes
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
import logging
from ACC_env import CarFleet
from rldmpc.agents.decent_mld_coordinator import DecentMldCoordinator
from rldmpc.agents.mld_agent import MldAgent
from rldmpc.agents.sequential_mld_coordinator import SequentialMldCoordinator
from rldmpc.core.admm import g_map
from rldmpc.agents.g_admm_coordinator import GAdmmCoordinator
import matplotlib.pyplot as plt
from rldmpc.mpc.mpc_mld import MpcMld
import gurobipy as gp
from rldmpc.mpc.mpc_switching import MpcSwitching
from rldmpc.systems.ACC import ACC
from rldmpc.utils.pwa_models import cent_from_dist

np.random.seed(0)

SIM_TYPE = "decent_mld"  # options: "mld", "g_admm", "sqp_admm", "decent_mld", "seq_mld"

n = 4  # num cars
N = 10  # controller horizon
ep_len = 100  # length of episode (sim len)
Adj = np.zeros((n, n))  # adjacency matrix
for i in range(n):  # make it chain coupling
    if i == 0:
        Adj[i, i + 1] = 1
    elif i == n - 1:
        Adj[i, i - 1] = 1
    else:
        Adj[i, i + 1] = 1
        Adj[i, i - 1] = 1
G_map = g_map(Adj)

acc = ACC(ep_len, N)
nx_l = acc.nx_l
nu_l = acc.nu_l
system = acc.get_pwa_system()
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()


NO_OVERTAKING = False

# construct centralised system
# no state coupling here so all zeros
Ac = np.zeros((nx_l, nx_l))
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
        len(system["S"])
    ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
        systems[i]["Ac"] = systems[i]["Ac"] + [Ac_i]

cent_system = cent_from_dist(systems, Adj)


class LocalMpc(MpcSwitching):
    rho = 0.5
    horizon = N

    def __init__(self, num_neighbours, my_index, leader=False) -> None:
        """Instantiate inner switching MPC for admm for car fleet. If leader is true the cost uses the reference traj
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

        r = system["T"][0].shape[0]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N):
            self.constraint(f"state_{k}", system["D"] @ x[:, [k]], "<=", system["E"])
            self.constraint(f"control_{k}", system["F"] @ u[:, [k]], "<=", system["G"])

            # acceleration limits

            self.constraint(f"acc_{k}", x[1, [k+1]] - x[1, [k]] , "<=", acc.a_acc* acc.ts)
            self.constraint(f"de_acc_{k}", x[1, [k+1]] - x[1, [k]] , ">=", acc.a_dec* acc.ts)



        # objective
        if leader:
            self.leader_traj = []
            for k in range(N):
                self.leader_traj.append(self.parameter(f"x_ref_{k}", (nx_l, 1)))
                self.fixed_pars_init[f"x_ref_{k}"] = np.zeros((nx_l, 1))
            self.set_local_cost(
                sum(
                    (x[:, [k]] - self.leader_traj[k] - sep).T
                    @ Q_x_l
                    @ (x[:, [k]] - self.leader_traj[k] - sep)
                    + u[:, [k]].T @ Q_u_l @ u[:, [k]]
                    for k in range(N)
                )
            )
        else:
            # following the agent ahead - therefore the index of the local state copy to track
            # is always the FIRST one in the local copies x_c
            self.set_local_cost(
                sum(
                    (x[:, [k]] - x_c[0:nx_l, [k]] - sep).T
                    @ Q_x_l
                    @ (x[:, [k]] - x_c[0:nx_l, [k]] - sep)
                    + u[:, [k]].T @ Q_u_l @ u[:, [k]]
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


class TrackingGAdmmCoordinator(GAdmmCoordinator):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.set_leader_traj(leader_state[:, timestep : (timestep + N)])
        return super().on_timestep_end(env, episode, timestep)

    def set_leader_traj(self, leader_traj):
        for k in range(N):  # we assume first agent is leader!
            self.agents[0].fixed_parameters[f"x_ref_{k}"] = leader_traj[:, [k]]


class MPCMldCent(MpcMld):
    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        self.mpc_model.setObjective(0, gp.GRB.MINIMIZE)

        # add extra constraints
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
                            follow_state[0, [k]] - d_safe
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
    
    def on_episode_start(self, env: Env, episode: int) -> None:
        self.mpc.set_leader_traj(leader_state[:, 0:N])
        return super().on_episode_start(env, episode)
    
class LocalMpcMld(MpcMld):

    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        # add accel cnstrs
        for k in range(N):
            self.mpc_model.addConstr(self.x[1, [k+1]] - self.x[1, [k]] <= acc.a_acc*acc.ts, name = f"acc_{k}")
            self.mpc_model.addConstr(self.x[1, [k+1]] - self.x[1, [k]] >= acc.a_dec*acc.ts, name = f"dec_{k}")

class TrackingDecentMldCoordinator(DecentMldCoordinator):
    # current state of car to be tracked is observed and propogated forward
    # to be the prediction
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.observe_states(timestep)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.observe_states(timestep=0)
        return super().on_episode_start(env, episode)

    def observe_states(self, timestep):
        for i in range(n):
            predicted_pos = np.zeros((1, N))
            predicted_vel = np.zeros((1, N))
            if i == 0:  # lead car
                predicted_pos[:, [0]] = leader_state[0, [timestep]]
                predicted_vel[:, [0]] = leader_state[1, [timestep]]
            else:
                predicted_pos[:, [0]] = env.x[nx_l * (i - 1), :]
                predicted_vel[:, [0]] = env.x[nx_l * (i - 1) + 1, :]
            for k in range(N - 1):
                predicted_pos[:, [k + 1]] = (
                    predicted_pos[:, [k]] + acc.ts * predicted_vel[:, [k]]
                )
                predicted_vel[:, [k + 1]] = predicted_vel[:, [k]]

            x_goal = np.vstack([predicted_pos, predicted_vel]) + np.tile(sep, N)

            self.agents[i].set_cost(Q_x_l, Q_u_l, x_goal=x_goal)


class TrackingSequentialMldCoordinator(SequentialMldCoordinator):
    # here we only set the leader, because the solutions are communicated down the sequence to other agents
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        x_goal = leader_state[:, timestep : timestep + N] + np.tile(sep, N)
        self.agents[0].set_cost(Q_x_l, Q_u_l, x_goal)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        x_goal = leader_state[:, 0:N] + np.tile(sep, N)
        self.agents[0].set_cost(Q_x_l, Q_u_l, x_goal)
        return super().on_episode_start(env, episode)


# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n), max_episode_steps=ep_len))
if SIM_TYPE == "mld":
    # mld mpc
    mld_mpc = MPCMldCent(cent_system, N)
    # initialise the cost with the first tracking point
    mld_mpc.set_leader_traj(leader_state[:, 0:N])
    agent = TrackingMldAgent(mld_mpc)
elif SIM_TYPE == "g_admm":
    # distributed mpcs and params
    local_mpcs: list[LocalMpc] = []
    local_fixed_dist_parameters: list[dict] = []
    for i in range(n):
        if i == 0:
            local_mpcs.append(
                LocalMpc(
                    num_neighbours=len(G_map[i]) - 1,
                    my_index=G_map[i].index(i),
                    leader=True,
                )
            )
        else:
            local_mpcs.append(
                LocalMpc(num_neighbours=len(G_map[i]) - 1, my_index=G_map[i].index(i))
            )
        local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)
    # coordinator
    agent = Log(
        TrackingGAdmmCoordinator(
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
elif SIM_TYPE == "decent_mld":
    # coordinator
    local_mpcs: list[MpcMld] = []
    for i in range(n):
        # passing local system
        local_mpcs.append(LocalMpcMld(system, N))
    agent = TrackingDecentMldCoordinator(local_mpcs, nx_l, nu_l)
elif SIM_TYPE == "seq_mld":
    # coordinator
    local_mpcs: list[MpcMld] = []
    for i in range(n):
        # passing local system
        local_mpcs.append(LocalMpcMld(system, N))
    agent = TrackingSequentialMldCoordinator(local_mpcs, nx_l, nu_l, Q_x_l, Q_u_l, sep)


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
