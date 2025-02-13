from typing import Any

import casadi as cs
import numpy as np
from csnlp import Solution
from gymnasium.spaces import Box
from mpcrl import Agent

from dmpcrl.mpc.mpc_admm import MpcAdmm


class AdmmCoordinator:
    """Class for coordinating the ADMM procedure of a network of agents"""

    @staticmethod
    def g_map(Adj: np.ndarray) -> list[list[int]]:
        """Construct the mapping from local to global variables from an adjacency matrix.
        The mapping reads G[i][j] = g, where g is the global index of the j-th local copy for agent i.

        Parameters
        ----------
        Adj: np.ndarray
            Adjacency matrix of the network.

        Returns
        -------
        list[list[int]]
            Mapping from local to global variables.
        """
        n = Adj.shape[0]
        G: list[list[int]] = []
        for i in range(n):
            G.append([])
            for j in range(n):
                if Adj[i, j] == 1 or i == j:
                    G[i].append(j)
        return G

    def __init__(
        self,
        agents: list[Agent],
        Adj: np.ndarray,
        N: int,
        nx_l: int,
        nu_l: int,
        rho: float,
        iters: int = 50,
    ) -> None:
        """Initialise the ADMM coordinator.

        Parameters
        ----------
        agents: List[Agent]
            A list of the agents involved in the ADMM procedure.
        Adj: np.ndarray
            Adjacency matrix of the network.
        N: int
            Length of prediction horizon.
        nx_l: int
            Local state dimension of agents.
        nu_l: int
            Local control dimension of agnets.
        rho: float
            Constant penalty term for augmented lagrangian.
        iters: int = 50
            Fixed number of ADMM iterations."""
        if not all(
            isinstance(agent.V, MpcAdmm) or agent.V.is_wrapped(MpcAdmm)
            for agent in agents
        ):
            raise ValueError(
                f"All agents must have ADMM-based MPCs. Received: {[type(agent.V) for agent in agents]}"
            )
        self.agents = agents
        self.n = len(agents)
        self.iters = iters
        self.G = self.g_map(Adj)
        self.nx_l = nx_l
        self.nu_l = nu_l
        self.rho = rho
        self.N = N

        # create auxillary vars for ADMM procedure
        self.y = [
            np.zeros((nx_l * len(self.G[i]), N + 1)) for i in range(self.n)
        ]  # multipliers
        self.augmented_x = [
            np.zeros((nx_l * len(self.G[i]), N + 1)) for i in range(self.n)
        ]  # augmented states
        self.z = np.zeros((self.n, nx_l, N + 1))  # global copies of local states

    def solve_admm(
        self,
        state: np.ndarray,
        action: cs.DM | None = None,
        deterministic: bool = True,
        action_space: Box | None = None,
    ) -> tuple[np.ndarray, list[Solution], dict[str, Any]]:
        """Solve the mpc problem for the network of agents using ADMM. If an
        action provided, the first action is constrained to be the provided action.

        Parameters
        ----------
        state: np.ndarray
            Global state of the network.
        action: np.ndarray | None = None
            Global action of the network. If None, the action is solved for.
        deterministic: bool = True
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed.
        action_space: Optional[Box]
            Only applicable if action=None. The action space of the environment. If provided, the action is clipped to
            the action space.

        Returns
        -------
        tuple[np.ndarray, list[Solution], dict[str, Any]
            A tuple containing the local actions, local solutions, and an info dictionary
        """
        u_iters = np.empty(
            (self.iters, self.n, self.nu_l, self.N)
        )  # store actions over iterations
        y_iters = [
            np.empty((self.iters, self.nx_l * len(self.G[i]), self.N + 1))
            for i in range(self.n)
        ]  # store y over iterations
        z_iters = np.empty(
            (self.iters, self.n, self.nx_l, self.N + 1)
        )  # store z over iterations
        augmented_x_iters = [
            np.empty((self.iters, self.nx_l * len(self.G[i]), self.N + 1))
            for i in range(self.n)
        ]  # augmented states over iterations
        f_iters = np.empty((self.iters, self.n))  # store local costs over iterations

        loc_actions = np.empty((self.n, self.nu_l))
        local_sols: list[Solution] = [None] * len(self.agents)
        x_l = np.split(state, self.n)  # split global state and action into local states
        u_l: list[np.ndarray] = np.split(action, self.n) if action is not None else []

        for iter in range(self.iters):
            # x update: solve local minimisations
            for i in range(len(self.agents)):
                # set parameters in augmented lagrangian
                self.agents[i].fixed_parameters["y"] = self.y[i]
                # G[i] is the agent indices relevant to agent i. Reshape stacks them in the local augmented state
                self.agents[i].fixed_parameters["z"] = self.z[self.G[i], :].reshape(
                    -1, self.N + 1
                )

                if action is None:
                    loc_actions[i], local_sols[i] = self.agents[i].state_value(
                        x_l[i], deterministic=deterministic, action_space=action_space
                    )
                else:
                    local_sols[i] = self.agents[i].action_value(x_l[i], u_l[i])
                if not local_sols[i].success:
                    # not raising an error on MPC failures
                    u_iters[iter, i] = np.nan
                    self.agents[i].on_mpc_failure(
                        episode=0, status=local_sols[i].status, raises=False, timestep=0
                    )
                    f_iters[iter, i] = np.inf
                else:
                    u_iters[iter, i] = local_sols[i].vals["u"]
                    f_iters[iter, i] = local_sols[i].f

                # construct solution to augmented state from local state and coupled states
                self.augmented_x[i] = cs.vertcat(
                    local_sols[i].vals["x_c"][: self.nx_l * self.G[i].index(i), :],
                    local_sols[i].vals["x"],
                    local_sols[i].vals["x_c"][self.nx_l * self.G[i].index(i) :, :],
                )
                augmented_x_iters[i][iter] = self.augmented_x[i]

            # z update: an averaging of all agents' optinions on each z
            for i in range(self.n):
                self.z[i] = np.mean(
                    np.stack(
                        [
                            self.augmented_x[j][
                                self.nx_l
                                * self.G[j].index(i) : self.nx_l
                                * (self.G[j].index(i) + 1),
                                :,
                            ]
                            for j in self.G[i]
                        ]
                    ),
                    axis=0,
                )
                z_iters[iter, i] = self.z[i]

            # y update: increment by the residual
            for i in range(self.n):
                self.y[i] = self.y[i] + self.rho * (
                    self.augmented_x[i] - self.z[self.G[i], :].reshape(-1, self.N + 1)
                )
                y_iters[i][iter] = self.y[i]

        return (
            loc_actions,
            local_sols,
            {
                "u_iters": u_iters,
                "y_iters": y_iters,
                "z_iters": z_iters,
                "augmented_x_iters": augmented_x_iters,
                "f_iters": f_iters,
            },
        )  # return actions and solutions from last ADMM iter and an info dict
