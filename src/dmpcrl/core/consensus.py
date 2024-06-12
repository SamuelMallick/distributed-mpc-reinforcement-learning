import numpy as np


class ConsensusCoordinator:
    """Class for coordinating consensus algorithms"""

    def __init__(self, Adj: np.ndarray, iters: int = 100) -> None:
        """Instantiate coordinator.

        Parameters
        ----------
        Adj: np.ndarray
            Adjacency matrix of network.
        iters:int = 100
            Number of iterations to run conensus for."""
        self.iters = iters
        self.P = self.gen_consensus_matrix(Adj)

    def average_consensus(self, x: np.ndarray) -> np.ndarray:
        """Run average consensus algorithm.

        Parameters
        ----------
        x: np.ndarray
            Initial state vector.

        Returns
        -------
        np.ndarray
            Final state vector after consensus algorithm."""
        for _ in range(self.iters):
            x = self.P @ x
        return x

    def gen_consensus_matrix(self, Adj: np.ndarray) -> np.ndarray:
        """Generate P matrix for consensus using graph laplacian approach.

        Parameters
        ----------
        Adj: np.ndarray
            Adjacency matrix of network.

        Returns
        -------
        np.ndarray
            Consensus matrix P."""
        max_N = max(sum(Adj))  # maximum neighbourhood cardinality
        eps = 0.5 * (1 / max_N)  # must be less than 1/max_N
        D_in = np.zeros(Adj.shape)
        for i in range(Adj.shape[0]):
            D_in[i, i] = sum(Adj)[i]
        L = D_in - Adj  # graph laplacian
        P = np.eye(Adj.shape[0]) - eps * L  # consensus matrix
        return P
