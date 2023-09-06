from typing import Any, List, Literal, Optional, Sequence, Union
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType, SymType
import numpy as np
import casadi as cs
import numpy.typing as npt
from rldmpc.agents.pwa_agent import PwaAgent
from rldmpc.core.admm import AdmmCoordinator

from rldmpc.mpc.mpc_admm import MpcAdmm


class GAdmmCoordinator(Agent):
    """Coordinates the greedy ADMM algorithm for PWA agents"""

    admm_iters = 50

    def __init__(
        self,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: List[dict],
        systems: List[dict],
        G: List[List[int]],
        Adj: np.ndarray,
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
        systems : dict
            PWA model for each agent.
        G :  List[List[int]]
            Map of local to global vars in ADMM.
        Adj: np.ndarray
            Adjacency matrix for agent coupling
        rho: float
            Augmented lagrangian penalty term."""

        # to the super class we pass the first local mpc just to satisfy the constructor
        # we copy it so the parameters don't double up etc.
        super().__init__(
            local_mpcs[0].copy(), local_fixed_parameters[0].copy(), warmstart, name
        )

        # construct the agents
        self.n = len(local_mpcs)
        self.agents: list[PwaAgent] = []
        for i in range(self.n):
            self.agents.append(
                PwaAgent(local_mpcs[i], local_fixed_parameters[i], systems[i])
            )

        # create ADMM coordinator
        self.N = local_mpcs[0].horizon
        self.Adj = Adj
        self.nx_l = local_mpcs[0].nx_l
        self.nu_l = local_mpcs[0].nu_l

        # coordinator of ADMM using 1 iteration as g_admm coordinator checks sequences every ADMM iter
        self.admm_coordinator = AdmmCoordinator(
            self.agents,
            G,
            self.N,
            self.nx_l,
            self.nu_l,
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
        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            for agent in self.agents:
                agent.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            for agent in self.agents:
                agent.on_episode_start(env, episode)

            while not (truncated or terminated):
                action, sol_list = self.g_admm_control(state, deterministic)
                for i in range(len(sol_list)):
                    if not sol_list[i].success:
                        self.agents[i].on_mpc_failure(
                            episode, timestep, sol_list[i].status, raises
                        )

                state, r, truncated, terminated, _ = env.step(action)
                for agent in self.agents:
                    agent.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                for agent in self.agents:
                    agent.on_timestep_end(env, episode, timestep)

            for agent in self.agents:
                agent.on_episode_end(env, episode, returns[episode])

        for agent in self.agents:
            agent.on_validation_end(env, returns)
        return returns

    def g_admm_control(self, state, deterministic):
        seqs = [[0] * self.N for i in range(self.n)]  # switching seqs for agents

        xc = [None]*self.n  

        # break global state into local pieces
        x = [state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)]

        # initial feasible control guess
        u = [-np.ones((self.nu_l, self.N)) for i in range(self.n)]

        # generate initial feasible coupling via dynamics rollout
        x_rout = self.dynamics_rollout(x, u)

        for iter in range(self.admm_iters):
            pass
            # generate local sequences and choose one  - CHOICE: this can be done with vars from local output of ADMM
            # which may not have converged to consensus - therefore adding exploration OR a cooperative
            # dynamics rollout as before the loop
            for i in range(self.n):
                if iter == 0:  # first iter we must used rolled out state
                    new_seqs = self.agents[i].eval_sequences(
                        x[i],
                        u[i],
                        [x_rout[j] for j in range(self.n) if self.Adj[i, j] == 1],
                    )
                    seqs[i] = new_seqs[0]   # use first by default for first iter
                else:
                    new_seqs = self.agents[i].eval_sequences(
                        x[i],
                        u[i],
                        xc[i],  # use local ADMM vars if not first iter
                    )

                    if seqs[i] in new_seqs:
                        new_seqs.remove(seqs[i])
                    if len(new_seqs) > 0:
                        seqs[i] = new_seqs[0]   # for now choosing arbritrarily first
                # set sequences
                self.agents[i].set_sequence(seqs[i])
            
            # perform ADMM step
            action_list, sol_list, error_flag = self.admm_coordinator.solve_admm(state)

            # extract the vars across the horizon from the ADMM sol for each agent
            for i in range(self.n):
                u[i] = np.asarray(sol_list[i].vals['u'])
                xc_out = np.asarray(sol_list[i].vals['x_c'])
                xc_temp = []
                for j in range(self.agents[i].num_neighbours):
                    xc_temp.append(xc_out[self.nx_l*j:self.nx_l*(j+1), :])
                xc[i] = xc_temp

        return cs.DM(action_list), sol_list

    def dynamics_rollout(self, x: List[np.ndarray], u: List[np.ndarray]):
        """For a given state and u, rollout the agents' dynamics step by step."""
        x_temp = [np.zeros((self.nx_l, self.N)) for i in range(self.n)]

        for i in range(self.n):
            x_temp[i][:, [0]] = x[i]  # add the first known states to the temp
        for k in range(1, self.N):
            for i in range(self.n):
                xc_temp = []
                for j in range(self.n):
                    if self.Adj[i, j] == 1:
                        xc_temp.append(x_temp[j][:, [k - 1]])
                x_temp[i][:, [k]] = self.agents[i].next_state(
                    x_temp[i][:, [k - 1]], u[i][:, [k]], xc_temp
                )
        return x_temp
