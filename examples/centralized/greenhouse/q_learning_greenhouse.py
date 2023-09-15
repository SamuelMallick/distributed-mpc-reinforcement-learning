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
from model import (
    df,
    get_control_bounds,
    get_disturbance_profile,
    get_model_details,
    output,
)
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

nx, nu, nd, ts = get_model_details()
u_min, u_max, du_lim = get_control_bounds()
d = get_disturbance_profile()

c_u = [10, 1, 1]  # penalty on each control signal
c_y = 1e3  # reward on yield


class NominalMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    horizon = 10

    def __init__(self) -> None:
        N = self.horizon
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # variables (state, action, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)

        # dynamics TODO add disturbances
        self.set_dynamics(lambda x, u: df(x, u, np.zeros((nd, 1))), n_in=2, n_out=1)

        # other constraints
        for k in range(1, N):
            self.constraint(f"du_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)
        # TODO add in output constraints

        obj = 0
        for k in range(N):
            for j in range(nu):
                obj += c_u[j] * u[j, k]
        y_N = output(x[:, [N]])
        obj += -c_y * y_N[0]
        self.minimize(obj)

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
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")
