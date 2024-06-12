import casadi as cs
from csnlp import Solution
import numpy as np
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
        if not all(isinstance(agent.V, MpcAdmm) for agent in agents):
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
        self.z = np.zeros((self.n, nx_l, N + 1))    # global copies of local states

    # TODO check if numpy arrays or DM's passed, also check return types
    def solve_admm(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[list[np.ndarray], list[Solution]]:
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

        Returns
        -------
        tuple[list[np.ndarray], list[Solution]]
            A tuple containing the local actions, and local solutions
        """

        loc_action_list: list[np.ndarray] = [None] * len(self.agents)   # TODO type error on the None
        local_sol_list: list[Solution] = [None] * len(self.agents)
        x_l = np.split(state, self.n)  # split global state and action into local states
        u_l = (
            np.split(action, self.n) if action is not None else [None] * self.n
        )  # TODO type error on the None

        for _ in range(self.iters):
            # x update: solve local minimisations
            for i in range(len(self.agents)):
                # set parameters in augmented lagrangian
                self.agents[i].fixed_parameters["y"] = self.y[i]
                # G[i] is the agent indices relevant to agent i. Reshape stacks them in the local augmented state
                self.agents[i].fixed_parameters["z"] = self.z[self.G[i], :].reshape(
                    -1, self.N + 1
                )

                if action is None:
                    loc_action_list[i], local_sol_list[i] = self.agents[i].state_value(
                        x_l[i], deterministic=deterministic
                    )
                else:
                    local_sol_list[i] = self.agents[i].action_value(x_l[i], u_l[i])
                if not local_sol_list[i].success:
                    # not raising an error on MPC failures
                    self.agents[i].on_mpc_failure(episode=0, status=local_sol_list[i].status, raises=False) 

                # construct solution to augmented state from local state and coupled states
                self.augmented_x[i] = cs.vertcat(
                    local_sol_list[i].vals["x_c"][: self.nx_l * self.G[i].index(i), :],
                    local_sol_list[i].vals["x"],
                    local_sol_list[i].vals["x_c"][self.nx_l * self.G[i].index(i) :, :],
                )

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

            # y update: increment by the residual
            for i in range(self.n):
                self.y[i] = self.y[i] + self.rho * (
                    self.augmented_x[i] - self.z[self.G[i], :].reshape(-1, self.N + 1)
                )

        return (
            loc_action_list,
            local_sol_list,
        )  # return actions and solutions from last ADMM iter
