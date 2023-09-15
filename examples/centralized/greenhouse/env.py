from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from model import df, get_disturbance_profile, get_model_details

np.random.seed(1)

nx, nu, nd, ts = get_model_details()
d = get_disturbance_profile()


class LettuceGreenHouse(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Continuous time environment for a luttuce greenhouse."""

    step_counter = 0

    def __init__(self) -> None:
        super().__init__()

        # set-up continuous time integrator for dynamics simulation
        x = cs.SX.sym("x", (nx, 1))
        u = cs.SX.sym("u", (nu, 1))
        d = cs.SX.sym("d", (nd, 1))
        p = cs.vertcat(u, d)
        x_new = df(x, u, d)
        ode = {"x": x, "p": p, "ode": x_new}
        self.integrator = cs.integrator(
            "env_integrator",
            "cvodes",
            ode,
            0,
            ts,
            {"abstol": 1e-8, "reltol": 1e-8},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.array([[0.0035], [0.001], [15], [0.008]])
        self.step_counter = 0
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        return 0.0

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        r = float(self.get_stage_cost(self.x, action))
        x_new = self.integrator(x0=self.x, p=cs.vertcat(action, cs.DM.zeros(4, 1)))[
            "xf"
        ]
        # TODO add in model uncertainty in step
        self.x = x_new
        self.step_counter += 1
        return x_new, r, False, False, {}


env = LettuceGreenHouse()
env.reset()
env.step(cs.DM.zeros(3, 1))
