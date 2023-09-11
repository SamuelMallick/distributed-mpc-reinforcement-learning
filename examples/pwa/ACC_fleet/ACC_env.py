import gymnasium as gym
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np
import casadi as cs

from rldmpc.systems.ACC import ACC


class CarFleet(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A fleet of non-linear hybrid vehicles who track each other."""

    step_counter = 0

    def __init__(self, acc: ACC, n: int) -> None:
        self.acc = acc
        self.nx_l = acc.nx_l
        self.nu_l = acc.nu_l
        self.nu_l = acc.nu_l
        self.Q_x_l = acc.Q_x_l
        self.Q_u_l = acc.Q_u_l
        self.sep = acc.sep
        self.leader_state = acc.get_leader_state()
        self.n = n
        super().__init__()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        starting_positions = [
            400 * np.random.random() for i in range(self.n)
        ]  # starting positions between 0-400 m
        starting_velocities = [
            33 * np.random.random() + 2 for i in range(self.n) 
        ]  # starting velocities between 2-35 ms-1
        self.x = np.tile(np.array([[0], [0]]), (self.n, 1))
        for i in range(self.n):
            init_pos = max(starting_positions)  # order the agents by starting distance
            self.x[i * self.nx_l, :] = init_pos
            self.x[i * self.nx_l + 1, :] = starting_velocities[i]
            starting_positions.remove(init_pos)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""

        cost = 0
        for i in range(self.n):
            local_state = state[self.nx_l * i : self.nx_l * (i + 1), :]
            local_action = action[self.nu_l * i : self.nu_l * (i + 1), :]
            if i == 0:
                # first car tracks leader
                follow_state = self.leader_state[:, [self.step_counter]]
            else:
                # other cars follow the next car
                follow_state = state[self.nx_l * (i - 1) : self.nx_l * (i), :]

            cost += (local_state - follow_state - self.sep).T @ self.Q_x_l @ (
                local_state - follow_state - self.sep
            ) + local_action.T @ self.Q_u_l @ local_action
        return cost

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)
        x_new = self.acc.step_car_dynamics_pwa(self.x, action, self.n, self.acc.ts)
        self.x = x_new

        self.step_counter += 1
        return x_new, r, False, False, {}
