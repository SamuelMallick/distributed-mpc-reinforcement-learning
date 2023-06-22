import numpy as np

from rldmpc.mpc.linear_mpc import LinearMPC


class Agent:
    """Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller."""

    def __init__(self, mpc, state) -> None:
        """Instantiates an agent with an MPC controller."""

        self.mpc = mpc
        self.state = state

    def set_state(self, new_state):
        """Sets the current state of the agent"""

        self.state = new_state

    def get_action(self):
        """Gets the action for the current state by solving the MPC."""

        return self.mpc.solve_mpc(self.state)
