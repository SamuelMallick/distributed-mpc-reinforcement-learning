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

        # create auxillary vars for ADMM procedure
        self.y = np.zeros((self.n, nx_l, N + 1))
        # self.y_list: list[np.ndarray] = []  # dual vars
        # self.x_temp: list[np.ndarray] = []  # intermediate numerical values for x
        self.z = np.zeros((self.n, nx_l, N + 1))

        # for i in range(self.n):
        #     x_dim = nx_l * len(G[i])  # dimension of augmented state for agent i
        #     # self.y_list.append(np.zeros((x_dim, N + 1)))
        #     # self.x_temp_list.append(np.zeros((x_dim, N + 1)))

        # # generate slices of z for each agent, slices that will pull the relevant
        # # component of global variable out for the agents
        # z_slices: list[list[int]] = []
        # for i in range(self.n):
        #     z_slices.append([])
        #     for j in self.G[i]:
        #         z_slices[i] += list(np.arange(nx_l * j, nx_l * (j + 1)))
        # self.z_slices = z_slices

    # TODO check if numpy arrays or DM's passed, also check return types
    def solve_admm(
        self, state: np.ndarray, action: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[Solution], bool]:
        """Solve the mpc problem for the network of agents using ADMM. If an
        action provided, the first action is constrained to be the provided action.

        Parameters
        ----------
        state: np.ndarray
            Global state of the network.
        action: np.ndarray | None = None
            Global action of the network. If None, the action is solved for.

        Returns
        -------
        tuple[list[np.ndarray], list[Solution], bool]
            A tuple containing the local actions, local solutions and a flag for error.
        """

        loc_action_list = [None] * len(self.agents)
        local_sol_list = [None] * len(self.agents)

        for _ in range(self.iters):
            # x update: solve local minimisations
            for i in range(len(self.agents)):
                loc_state = state[
                    self.nx_l * i : self.nx_l * (i + 1), :
                ]  # get local state from global

                # set parameters in augmented lagrangian
                self.agents[i].fixed_parameters["y"] = self.y_list[i]
                self.agents[i].fixed_parameters["z"] = self.z[
                    self.z_slices[i], :
                ]  # use z slice to extract relevant z component

                if action is None:
                    loc_action_list[i], local_sol_list[i] = self.agents[i].state_value(
                        loc_state, deterministic
                    )
                else:
                    # get local action from global
                    loc_action = action[self.nu_l * i : self.nu_l * (i + 1), :]
                    local_sol_list[i] = self.agents[i].action_value(
                        loc_state, loc_action
                    )
                if not local_sol_list[i].success:  # if a local problem failed
                    return (
                        loc_action_list,
                        local_sol_list,
                        True,
                    )  # return with error flag as true

                # construct solution to augmented state from local state and coupled states
                idx = self.G[i].index(i) * self.nx_l
                self.x_temp_list[i] = cs.vertcat(
                    local_sol_list[i].vals["x_c"][:idx, :],
                    local_sol_list[i].vals["x"],
                    local_sol_list[i].vals["x_c"][idx:, :],
                )

            # z update: an averaging of all agents' optinions on each z

            for i in range(len(self.agents)):  # loop through each z
                count = 0
                sum = np.zeros((self.nx_l, self.z.shape[1]))
                for j in range(
                    len(self.agents)
                ):  # loop through agents who have opinion on this z
                    if i in self.G[j]:
                        count += 1
                        x_slice = slice(
                            self.nx_l * self.G[j].index(i),
                            self.nx_l * (self.G[j].index(i) + 1),
                        )
                        sum += self.x_temp_list[j][x_slice, :]
                self.z[self.nx_l * i : self.nx_l * (i + 1), :] = sum / count

            # y update: increment by the residual

            for i in range(len(self.agents)):
                self.y_list[i] = self.y_list[i] + self.rho * (
                    self.x_temp_list[i] - self.z[self.z_slices[i], :]
                )

        return (
            loc_action_list,
            local_sol_list,
            False,
        )  # return last solutions with error flag as false
