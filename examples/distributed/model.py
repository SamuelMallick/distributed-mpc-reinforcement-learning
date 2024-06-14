from typing import ClassVar

import casadi as cs
import numpy as np


class Model:
    """Class to store model information for the system."""

    n: ClassVar[int] = 3  # number of agents
    nx_l: ClassVar[int] = 2  # local state dimension
    nu_l: ClassVar[int] = 1  # local control dimension

    x_bnd_l: ClassVar[np.ndarray] = np.array(
        [[0, -1], [1, 1]]
    )  # local state bounds x_bnd[0] <= x <= x_bnd[1]
    u_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-1], [1]]
    )  # local control bounds u_bnd[0] <= u <= u_bnd[1]

    adj: ClassVar[np.ndarray] = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32
    )  # adjacency matrix of coupling in network

    A_l: ClassVar[np.ndarray] = np.array(
        [[0.9, 0.35], [0, 1.1]]
    )  # local state-space matrix A
    B_l: ClassVar[np.ndarray] = np.array(
        [[0.0813], [0.2]]
    )  # local state-space matrix B
    A_c_l: ClassVar[np.ndarray] = np.array(
        [[0, 0], [0, -0.1]]
    )  # local coupling matrix A_c

    A_l_innacurate: ClassVar[np.ndarray] = np.asarray(
        [[1, 0.25], [0, 1]]
    )  # inaccurate local state-space matrix A
    B_l_innacurate: ClassVar[np.ndarray] = np.asarray(
        [[0.0312], [0.25]]
    )  # inaccurate local state-space matrix B
    A_c_l_innacurate: ClassVar[np.ndarray] = np.array(
        [[0, 0], [0, 0]]
    )  # inaccurate local coupling matrix A_c

    def __init__(self):
        """Initializes the model."""
        self.A, self.B = self.centralized_dynamics_from_local(
            [self.A_l] * self.n,
            [self.B_l] * self.n,
            [[self.A_c_l for _ in range(np.sum(self.adj[i]))] for i in range(self.n)],
        )

    def centralized_dynamics_from_local(
        self,
        A_list: list[np.ndarray | cs.SX],
        B_list: list[np.ndarray | cs.SX],
        A_c_list: list[list[np.ndarray | cs.SX]],
    ) -> tuple[np.ndarray | cs.SX, np.ndarray | cs.SX]:
        """Creates centralized representation from a list of local dynamics matrices.

        Parameters
        ----------
        A_list : list[np.ndarray | cs.SX]
            List of local state-space matrices A.
        B_list : list[np.ndarray | cs.SX]
            List of local state-space matrices B.
        A_c_list : list[list[np.ndarray | cs.SX]]
            List of local coupling matrices A_c. A_c[i][j] is coupling
            effect of agent j on agent i.

        Returns
        -------
        tuple[np.ndarray | cs.SX, np.ndarray | cs.SX]
            Global state-space matrices A and B.
        """
        if any(len(A_c_list[i]) != np.sum(self.adj[i]) for i in range(self.n)):
            raise ValueError(
                "A_c_list must have the same length as the number of neighbors."
            )

        if isinstance(A_list[0], np.ndarray):
            row_func = lambda x: np.hstack(x)
            col_func = lambda x: np.vstack(x)
            zero_func = np.zeros
        else:
            row_func = lambda x: cs.horzcat(*x)
            col_func = lambda x: cs.vertcat(*x)
            zero_func = cs.SX.zeros
        A = col_func(  # global state-space matrix A
            [
                row_func(
                    [
                        (
                            A_list[i]
                            if i == j
                            else (
                                A_c_list[i].pop(0)
                                if self.adj[i, j] == 1
                                else zero_func((self.nx_l, self.nx_l))
                            )
                        )
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        B = col_func(
            [
                row_func(
                    [
                        B_list[i] if i == j else zero_func((self.nx_l, self.nu_l))
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        return A, B


m = Model()
