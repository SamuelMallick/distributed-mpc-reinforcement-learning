from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import networkx as netx
from scipy.linalg import block_diag

# First, create class for environment

class LtiNetwork(gym.Env):
    """A discrete time network of LTI systems."""

    n = 3   # number of agents
    nx_l = 2  # number of agent states
    nu_l = 1  # number of agent inputs
    nx = n*nx_l # number of states
    nu = n*nu_l # number of inputs 

    A_l = np.array([[0.9, 0.35], [0, 1.1]])   # agent state-space matrix A
    B_l = np.array([[0.0813], [0.2]]) # agent state-space matrix B
    A_c = np.array([[1, 1], [1, 1]])  # common coupling state-space matrix
    x_bnd_l = (np.asarray([[0], [-1]]), np.asarray([[1], [1]]))  # agent bounds of state
    a_bnd_l = (-1, 1)  # agent bounds of control input
    w = np.asarray([[1e2], [1e2]])  # agent penalty weight for bound violations
    e_bnd = (-1e-1, 0)  # agent uniform noise bounds

    # create the centralised representation of the netork.
    
    p = 0.5 # probability of edge connection in network
    G = netx.binomial_graph(n, p)
    while not netx.is_connected(G): # generate random graphs until finding a connected one
        print('randomly generated graph not connected. Trying again...')
        G = netx.binomial_graph(n, p)
    Adj = netx.adjacency_matrix(G).toarray()    # adjacency matrix representing coupling in network
    
    A = np.zeros((n*nx, n*nx))  # global state-space matrix A
    for i in range(n):
        for j in range(i,n):
            if i == j:
                A[nx_l*i:nx_l*(i+1), nx_l*i:nx_l*(i+1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l*i:nx_l*(i+1), nx_l*j:nx_l*(j+1)] = A_c   # coupling is birectional
                A[nx_l*j:nx_l*(j+1), nx_l*i:nx_l*(i+1)] = A_c
    B = block_diag(*[B_l]*n)    # global state-space matix B
        
    x_bnd = ()
    a_bnd = ()
    for i in range(n):
        x_bnd += x_bnd_l    # all agents have the limits (for now)
        a_bnd += a_bnd_l    # NOTE: I dont know if this is the correct for a_bnd?

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the states of the LTI network."""
        super().reset(seed=seed, options=options)
        self.x = self.np_random.random((self.n*self.nx))
        return self.x, {}
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Steps the network by applying the dynamics with the action as control."""
        return super().step(action)
    
if __name__ == "__main__":
    sys = LtiNetwork()
    sys.reset()