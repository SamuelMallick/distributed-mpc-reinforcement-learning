from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from model import Model


class LtiSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A discrete time network of LTI systems."""

    noise_bnd = np.array([[-1e-1], [0]])  # uniform noise bounds for process noise on local systems
   
    def __init__(self, model: Model) -> None:
        """Initializes the environment.
        
        Parameters
        ----------
        model : Model
            The model of the system."""
        super().__init__()
        self.A, self.B = model.A, model.B
        self.n = model.n
        self.nx = model.n*model.nx_l
        self.nx_l = model.nx_l
        self.x_bnd = np.tile(model.x_bnd_l, self.n)
        self.w = np.tile([[1.2e2, 1.2e2]], (1, self.n))  # penalty weight for bound violations

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the environment.
        
        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.
        options : dict[str, Any], optional
            The options for the reset.
            
        Returns
        -------
        tuple[npt.NDArray[np.floating], dict[str, Any]]
            The initial state and an info dictionary."""
        super().reset(seed=seed, options=options)
        if options is not None and "x0" in options:
            self.x = options["x0"]
        else:
            self.x = np.tile([0, 0.15], self.n).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: np.ndarray, action: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
        """Returns the stage cost of the system for a given state and action.
        
        Parameters
        ----------
        state : np.ndarray
            The state of the system.
        action : np.ndarray
            The action of the system.
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
            
        Returns
        -------
        float
            The stage cost."""
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * np.square(action).sum()
            + self.w @ np.maximum(0, lb[:, np.newaxis] - state)
            + self.w @ np.maximum(0, state - ub[:, np.newaxis])
        )

    def get_dist_stage_cost(self, state: np.ndarray, action: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> list[float]:
        """Returns the distributed costs of the system for a given centralized state and action.
        
        Parameters
        ----------
        state : np.ndarray
            The centralized state of the system.
        action : np.ndarray
            The centralized action of the system.
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
            
        Returns
        -------
        list[float]
            The distributed costs."""
        x_l, u_l, lb_l, ub_l = np.split(state, self.n), np.split(action, self.n), np.split(lb, self.n), np.split(ub, self.n)    # break into local pieces
        return [self.get_stage_cost(x_l[i], u_l[i], lb_l[i], ub_l[i]) for i in range(self.n)]

    def step(
        self, action: cs.DM
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Takes a step in the environment.
        
        Parameters
        ----------
        action : cs.DM
            The action to take.
        
        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            The new state, the reward, truncated flag, terminated flag, and an info dictionary."""
        action = action.full()  # convert action from casadi DM to numpy array
        x_new = self.A @ self.x + self.B @ action
        noise = self.np_random.uniform(*self.noise_bnd).reshape(-1, 1)
        x_new[np.arange(0, self.nx, self.nx_l)] += noise    # apply noise only to first state dimension of each agent

        r = self.get_stage_cost(self.x, action, lb=self.x_bnd[0], ub=self.x_bnd[1])
        r_dist = self.get_dist_stage_cost(self.x, action, lb=self.x_bnd[0], ub=self.x_bnd[1])
        self.x = x_new
        return x_new, r, False, False, {"r_dist": r_dist}
