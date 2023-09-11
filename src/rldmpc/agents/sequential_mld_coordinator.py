from typing import Any, Collection, List, Literal, Optional, Sequence, Union
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType, SymType
import numpy as np
import casadi as cs
import numpy.typing as npt
from rldmpc.agents.mld_agent import MldAgent
from rldmpc.agents.pwa_agent import PwaAgent
from rldmpc.core.admm import AdmmCoordinator
import logging
from rldmpc.mpc.mpc_admm import MpcAdmm
import matplotlib.pyplot as plt
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
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
        """
        self._exploration: ExplorationStrategy = NoExploration()  # to keep compatable
        self.n = len(local_mpcs)
        self.nx_l = nx_l
        self.nu_l = nu_l
        self.Q_x_l = Q_x_l
        self.Q_u_l = Q_u_l
        self.sep = sep
        self.d_safe = d_safe
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

    def get_control(self, state):
        u = [None] * self.n
        for i in range(self.n):
            xl = state[self.nx_l * i : self.nx_l * (i + 1), :]
            if i != 0:
                x_pred_prev = self.agents[i - 1].x_pred
                x_goal = x_pred_prev + np.tile(self.sep, x_pred_prev.shape[1])
                for k in range(x_pred_prev.shape[1] - 1):
                    self.agents[i].mpc.safety_constraints[k].RHS = (
                        x_pred_prev[0, [k]] - self.d_safe
                    )
                self.agents[i].set_cost(self.Q_x_l, self.Q_u_l, x_goal)
            u[i] = self.agents[i].get_control(xl)
        return np.vstack(u)
