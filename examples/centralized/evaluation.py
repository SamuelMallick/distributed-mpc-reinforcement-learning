import contextlib
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import networkx as netx
from scipy.linalg import block_diag 

from rldmpc.mpc.linear_mpc import LinearMPC
from rldmpc.agents.agent import Agent

# graph structure for network

n = 3   # number of agents
nx_l = 2  # number of agent states
nu_l = 1  # number of agent inputs

p = 0.5 # probability of edge connection in network
G = netx.binomial_graph(n, p)
while not netx.is_connected(G): # generate random graphs until finding a connected one
    print('randomly generated graph not connected. Trying again...')
    G = netx.binomial_graph(n, p)
Adj = netx.adjacency_matrix(G).toarray()    # adjacency matrix representing coupling in network

def get_centralized_dynamics(
    n,
    nx_l,
    A_l,
    B_l,
    A_c,
):
    """Creates the centralized representation of the dynamics."""
    A = np.zeros((n * nx_l, n * nx_l))  # global state-space matrix A
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c
                A[nx_l * j : nx_l * (j + 1), nx_l * i : nx_l * (i + 1)] = A_c
    
    B = block_diag(*[B_l]*n)    # global state-space matix B
    return A, B

# First, create class for environment

class LtiNetwork(gym.Env):
    """A discrete time network of LTI systems."""

    def __init__(self):
        self.n = n   # number of agents

        nx = n*nx_l # number of states
        nu = n*nu_l # number of inputs 

        A_l = np.array([[0.9, 0.35], [0, 1.1]])   # agent state-space matrix A
        B_l = np.array([[0.0813], [0.2]]) # agent state-space matrix B
        A_c = np.array([[0, 0], [0, 0.5]])  # common coupling state-space matrix
        x_bnd = (np.asarray([[0], [-1]]), np.asarray([[1], [1]]))  # agent bounds of state
        a_bnd = (-1, 1)  # agent bounds of control input
        w = np.asarray([[1e2], [1e2]])  # agent penalty weight for bound violations
        e_bnd = (-1e-1, 0)  # agent uniform noise bounds

        A, B = get_centralized_dynamics(n, nx_l, A_l, B_l, A_c) 

        # costs and penalties

        Q_x = np.eye(n*nx_l)
        Q_u = np.eye(n*nu_l)

        # assign vars to class that are needed later

        self.nx = nx
        self.nx_l = nx_l
        self.Q_x = Q_x
        self.Q_u = Q_u
        self.x_bnd = x_bnd
        self.w = w
        self.A = A
        self.B = B
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the states of the LTI network."""
        super().reset(seed=seed, options=options)
        self.x = self.np_random.random((self.nx, 1))
        return self.x, {}
    
    def get_stage_cost(self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]) -> float:
        """Computes the real stage cost L(s, a)"""
        lb, ub = self.x_bnd
        cost = state.T @ self.Q_x @ state + action.T @ self.Q_u @ action
        for i in range(self.n):
            cost += self.w.T @ np.maximum(0, lb - state[self.nx_l*i:self.nx_l*(i + 1), :]) + \
            self.w.T @ np.maximum(0, state[self.nx_l*i:self.nx_l*(i + 1), :] - ub)
        return float(cost)  # casting to float from 2D array size 1x1
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Steps the network by applying the dynamics with the action as control."""
        x_new = self.A @ self.x + self.B @ action
        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}
    
if __name__ == "__main__":
    N = 10

    state_history = []
    # initial guess of dynamics

    A_l = np.array([[1.4, 0.6], [0.1, 1.3]])   # agent state-space matrix A
    B_l = np.array([[0.0813], [0.2]]) # agent state-space matrix B
    A_c = np.array([[0, 0], [0, 0.5]])  # common coupling state-space matrix
    A, B = get_centralized_dynamics(n, nx_l, A_l, B_l, A_c)

    env = LtiNetwork()  # create environment
    state, _ = env.reset()  # reset environment to get new initial state
    mpc = LinearMPC(n = 2*n, m = 1*n, N = N, A = A, B = B, Q_x=np.eye(n*nx_l), Q_u = np.eye(n*nu_l))  
    agent = Agent(mpc, state) 

    episode_length = 50 # number of steps in evaluation episode
    for t in range(episode_length):
        state_history.append(state)
        action = agent.get_action()
        state, cost, truncated, terminated, _ = env.step(action)
        agent.set_state(state)
    env.close()

    for i in range(n*nx_l):
        plt.plot([state_history[t][i, :] for t in range(episode_length)])
    plt.show()