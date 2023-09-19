import logging
from typing import Any, Collection, List, Literal, Optional, Sequence, Union

import casadi as cs
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.core.exploration import ExplorationStrategy, NoExploration

from rldmpc.agents.mld_agent import MldAgent
from rldmpc.agents.pwa_agent import PwaAgent
from rldmpc.core.admm import AdmmCoordinator
from rldmpc.mpc.mpc_admm import MpcAdmm
from rldmpc.mpc.mpc_mld import MpcMld

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SequentialMldCoordinator(MldAgent):
    """A coordinatoof MLD agents that solve their local problems in a sequence, communicating the soltuions to the next agent in sequence."""

    def __init__(
        self,
        local_mpcs: List[MpcMld],
        nx_l: int,
        nu_l: int,
        Q_x_l: np.ndarray,
        Q_u_l: np.ndarray,
        sep: np.ndarray,
        d_safe: float,
        w: float,
        N: int,
    ) -> None:
        """Initialise the coordinator.

        Parameters
        ----------
        local_mpcs: List[MpcMld]
            List of local MLD based MPCs - one for each agent.
        nx_l: int
            Dimension of local state.
        nu_l: int
            Dimension of local control.
        Q_x_l: np.ndarray
            Quadratic penalty matrix for state tracking.
        Q_u_l: np.ndarray
            Quadratic penalty matrix for control effort.
        sep: np.ndarray
            Desired state seperation between tracked vehicles.
        d_safe: float
            Safe distance between vehicles.
        w: float
            Penalty on slack var s in cost.
        N: int
            Prediction horizon.
        """
        self._exploration: ExplorationStrategy = NoExploration()  # to keep compatable
        self.n = len(local_mpcs)
        self.nx_l = nx_l
        self.nu_l = nu_l
        self.Q_x_l = Q_x_l
        self.Q_u_l = Q_u_l
        self.sep = sep
        self.d_safe = d_safe
        self.w = w
        self.N = N
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

    def get_control(self, state):
        u = [None] * self.n
        for i in range(self.n):
            xl = state[self.nx_l * i : self.nx_l * (i + 1), :]
            if i != 0:
                x_pred_prev = self.agents[i - 1].x_pred
                x_goal = x_pred_prev + np.tile(self.sep, self.N + 1)
                for k in range(self.N + 1):
                    self.agents[i].mpc.safety_constraints[k].RHS = (
                        x_pred_prev[0, [k]] - self.d_safe
                    )

                # set cost of agent
                self.agents[i].set_cost(self.Q_x_l, self.Q_u_l, x_goal)
                obj = 0
                for k in range(self.N):
                    obj += (
                        (self.agents[i].mpc.x[:, k] - x_pred_prev[:, k] - self.sep.T)
                        @ self.Q_x_l
                        @ (
                            self.agents[i].mpc.x[:, [k]]
                            - x_pred_prev[:, [k]]
                            - self.sep
                        )
                        + self.agents[i].mpc.u[:, k]
                        @ self.Q_u_l
                        @ self.agents[i].mpc.u[:, [k]]
                        + self.w * self.agents[i].mpc.s[:, [k]]
                    )
                obj += (
                    self.agents[i].mpc.x[:, self.N]
                    - x_pred_prev[:, self.N]
                    - self.sep.T
                ) @ self.Q_x_l @ (
                    self.agents[i].mpc.x[:, [self.N]]
                    - x_pred_prev[:, [self.N]]
                    - self.sep
                ) + self.w * self.agents[
                    i
                ].mpc.s[
                    :, [self.N]
                ]
                self.agents[i].mpc.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

            u[i] = self.agents[i].get_control(xl)
        return np.vstack(u)
