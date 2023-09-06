from typing import Any, List, Literal, Optional, Sequence, Union
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType, SymType
import numpy as np
import numpy.typing as npt
from rldmpc.agents.pwa_agent import PwaAgent
from rldmpc.core.ADMM import AdmmCoordinator

from rldmpc.mpc.mpc_admm import MpcAdmm


class GAdmmCoordinator(Agent):
    """Coordinates the greedy ADMM algorithm for PWA agents"""

    admm_iters = 50

    def __init__(
        self,
        local_mpcs: list[MpcAdmm[SymType]],
        local_fixed_parameters: List[dict],
        system: dict,
        G: List[List[int]],
        rho: float,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        """Instantiates the coordinator, creating n PWA agents.

        Parameters
        ----------
        local_mpcs: list[MpcAdmm[SymType]]
            List of local MPCs for agents.
        local_fixed_parameters: List[dict]
            List of dictionaries for fixed parameters for agents.
        system : dict
            PWA model for each agent.
        G :  List[List[int]]
            Map of local to global vars in ADMM.
        rho: float
            Augmented lagrangian penalty term."""

        # to the super class we pass the first local mpc just to satisfy the constructor
        super().__init__(local_mpcs[0], local_fixed_parameters[0], warmstart, name)

        # construct the agents
        self.n = len(local_mpcs)
        self.agents: list[PwaAgent] = []
        for i in range(self.n):
            self.agents.append(
                PwaAgent(local_mpcs[i], local_fixed_parameters[i], system)
            )

        # create ADMM coordinator
        N = local_mpcs[0].horizon
        nx_l = local_mpcs[0].nx_l
        nu_l = local_mpcs[0].nu_l

        # coordinator of ADMM using 1 iteration as g_admm coordinator checks sequences every ADMM iter
        self.admm_coordinator = AdmmCoordinator(
            self.agents,
            G,
            N,
            nx_l,
            nu_l,
            rho,
            iters=1,
        )

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] = None,
    ):
        return super().evaluate(
            env, episodes, deterministic, seed, raises, env_reset_options
        )
