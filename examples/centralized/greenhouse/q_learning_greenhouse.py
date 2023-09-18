import contextlib
import logging
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
from env import GreenhouseAgent, LettuceGreenHouse, GreenhouseSampleAgent

from model import (
    multi_sample_rk4_step,
    get_control_bounds,
    multi_sample_output,
    get_model_details,
    output_real,
    rk4_step_real,
    get_y_min,
    get_y_max
)
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

np.random.seed(69)

SYM_TYPE = "sample" # options: "nom", "sample"

nx, nu, nd, ts = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = [10, 1, 1]  # penalty on each control signal
c_y = 10e3  # reward on yield


class NominalMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    horizon = 6 * 4  # prediction horizon
    discount_factor = 1

    def __init__(self) -> None:
        N = self.horizon
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, slack)
        x, _ = self.state("x", nx, lb=[[0], [0], [-float("inf")], [0]])
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        d = self.disturbance("d", nd)

        # dynamics
        # self.set_dynamics(lambda x, u, d: x + ts * df(x, u, d), n_in=3, n_out=1)
        self.set_dynamics(lambda x, u, d: rk4_step_real(x, u, d), n_in=3, n_out=1)

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx, 1)) for k in range(N + 1)]
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

            # output constraints
            y_k = output_real(x[:, [k]])
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_list[k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_list[k])

        y_N = output_real(x[:, [N]])
        self.constraint(f"y_min_{N}", y_N, ">=", y_min_list[N])
        self.constraint(f"y_max_{N}", y_N, "<=", y_max_list[N])

        obj = 0
        for k in range(N):
            for j in range(nu):
                obj += c_u[j] * u[j, k]
        obj += -c_y * y_N[0]
        self.minimize(obj)

        # solver
        opts = {
            "expand": True,
            "show_eval_warnings": True,
            "warn_initial_bounds": True,
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
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class SampleBasedMpc(Mpc[cs.SX]):
    """Non-linear Sample Based Robust MPC for greenhouse control."""

    horizon = 6 * 4  # prediction horizon
    discount_factor = 1

    def __init__(self, Ns) -> None:
        N = self.horizon
        self.Ns = Ns
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, slack)
        # state needs to be done manually as we have one state per scenario
        x = self.nlp.variable(
            "x",
            (nx * Ns, self._prediction_horizon + 1),
            lb=cs.vertcat(*[[0], [0], [-float("inf")], [0]]*Ns)
        )[0]
        x0 = self.nlp.parameter("x_0", (nx, 1))
        self.nlp.constraint("x_0", x[:, 0], "==", cs.repmat(x0, Ns, 1))
        self._states["x"] = x
        self._initial_states["x_0"] = x0
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        d = self.disturbance("d", nd)

        # dynamics
        self.set_dynamics(
            lambda x, u, d: multi_sample_rk4_step(x, u, d, Ns), n_in=3, n_out=1
        )

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx * Ns, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx * Ns, 1)) for k in range(N + 1)]
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

            # output constraints
            y_k = multi_sample_output(x[:, [k]], Ns)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_list[k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_list[k])

        y_N = multi_sample_output(x[:, [N]], Ns)
        self.constraint(f"y_min_{N}", y_N, ">=", y_min_list[N])
        self.constraint(f"y_max_{N}", y_N, "<=", y_max_list[N])

        obj = 0
        for k in range(N):
            for j in range(nu):
                obj += Ns*c_u[j] * u[j, k]
        for i in range(Ns):
            y_N_i = y_N[nx * i : nx * (i + 1), :]
            obj += -c_y * y_N_i[0]
        self.minimize(obj)

        # solver
        opts = {
            "expand": True,
            "show_eval_warnings": True,
            "warn_initial_bounds": True,
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
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


if SYM_TYPE == "nom":
    mpc = NominalMpc()
    agent = Log(
        GreenhouseAgent(mpc, {}),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
    )
elif SYM_TYPE == "sample":
    sample_mpc = SampleBasedMpc(Ns=5)
    agent = Log(
        GreenhouseSampleAgent(sample_mpc, {}),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
    )

days = 2
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
env = MonitorEpisodes(TimeLimit(LettuceGreenHouse(), max_episode_steps=int(ep_len)))
agent.evaluate(env=env, episodes=1, seed=1, raises=False)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")

# generate output
y = np.asarray([output_real(X[k, :]) for k in range(X.shape[0])]).squeeze()

# get bounds
d = env.disturbance_profile
y_min = np.zeros((nx, ep_len))
y_max = np.zeros((nx, ep_len))
for t in range(ep_len):
    y_min[:, [t]] = get_y_min(d[:, [t]])
    y_max[:, [t]] = get_y_max(d[:, [t]])

_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(X[:, i])
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(y[:, i])
    axs[i].plot(y_min[i, :], color = 'black')
    if i != 0:
        axs[i].plot(y_max[i, :], color = 'r')
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for i in range(3):
    axs[i].plot(U[:, i])
    axs[i].axhline(u_min[i], color = 'black')
    axs[i].axhline(u_max[i], color = 'r')
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(env.disturbance_profile[i, : days * 24 * 4])
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
