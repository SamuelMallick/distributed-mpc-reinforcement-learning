from typing import Literal, TypeVar

import casadi as cs
import numpy as np
from csnlp.wrappers import Mpc
from csnlp.wrappers.wrapper import Nlp

SymType = TypeVar("SymType", cs.SX, cs.MX)


class MpcAdmm(Mpc[cs.SX]):
    """The local MPC minimisation in an ADMM scheme."""

    def __init__(
        self,
        nlp: Nlp[SymType],
        prediction_horizon: int,
        control_horizon: int | None = None,
        input_spacing: int = 1,
        shooting: Literal["single", "multi"] = "multi",
    ) -> None:
        """Initialises the ADMM based MPC

        Parameters
        ----------
        nlp : Nlp
            NLP scheme to be wrapped
        prediction_horizon : int
            A positive integer for the prediction horizon of the MPC controller.
        control_horizon : int, optional
            A positive integer for the control horizon of the MPC controller. If not
            given, it is set equal to the control horizon.
        input_spacing : int, optional
            Spacing between independent input actions. This argument allows to reduce
            the number of free actions along the control horizon by allowing only the
            first action every `n` to be free, and the following `n-1` to be fixed equal
            to that action (where `n` is given by `input_spacing`). By default, no
            spacing is allowed, i.e., 1.
        shooting : 'single' or 'multi', optional
            Type of approach in the direct shooting for parametrizing the control
            trajectory. See [1, Section 8.5]. By default, direct shooting is used.

        Raises
        ------
        ValueError
            Raises if the shooting method is invalid; or if any of the horizons are
            invalid."""
        self._fixed_pars_init: dict = {}
        Mpc.__init__(
            self, nlp, prediction_horizon, control_horizon, input_spacing, shooting
        )
        self.N = prediction_horizon

    @property
    def fixed_pars_init(self) -> dict:
        return self._fixed_pars_init

    @fixed_pars_init.setter
    def fixed_pars_init(self, value):
        """Prevent setting the dictionary, as it must contain the init values for y and z"""
        raise ValueError(
            "Can't set the value of fixed_pars_init. You can only add too it."
        )

    def augmented_state(
        self, num_neighbours: int, my_index: int, size: int
    ) -> tuple[cs.SX, cs.SX]:
        """Generates the local state and variables for the copies of neighbour
        states over the prediction horizon

        Parameters
        ----------
        num_neighbours : int
            Number of neighbours of the agent
        my_index : int
            Index of the agents local state within its augmented state
        size : int
            Number of elements in local

        Returns
        -------
        Tuple[cs.SX, cs.SX]
            The local state and the neighbour states over the prediction horizon
        """

        x, _ = self.state("x", size)  # local state
        x_c, _, _ = self.variable(
            "x_c", (size * (num_neighbours), self.N + 1)
        )  # neighbor states

        # adding them together as the full decision var for augmented cost
        self.x_cat = cs.vertcat(
            x_c[: (my_index * size), :], x, x_c[(my_index * size) :, :]
        )

        # Parameters in augmented lagrangian
        self.y = self.parameter("y", (size * (num_neighbours + 1), self.N + 1))
        self.z = self.parameter("z", (size * (num_neighbours + 1), self.N + 1))
        self._fixed_pars_init["y"] = np.zeros((size * (num_neighbours + 1), self.N + 1))
        self._fixed_pars_init["z"] = np.zeros((size * (num_neighbours + 1), self.N + 1))

        return x, x_c

    def set_dynamics(
        self,
        F,
        n_in,
        n_out,
    ) -> None:
        raise RuntimeError(
            "For ADMM based MPC the dynamics must be set manually as constraints due to the coupling."
        )

    def set_local_cost(self, local_cost: cs.SX, rho: float = 0.5) -> None:
        """Sets the cost function for the ADMM based MPC. The augmented lagrangian
        terms are augmented to the local cost.

        Parameters
        ----------
        local_cost : cs.SX
            The local cost function.
        rho : float, optional
            The penalty term for the augmented lagrangian. Default is 0.5."""

        if not hasattr(self, "x_cat"):
            raise RuntimeError(
                "You must call augmented_state before setting local cost."
            )

        self.minimize(
            local_cost
            + sum(
                (self.y[:, [k]].T @ (self.x_cat[:, [k]] - self.z[:, [k]]))
                for k in range(self.N + 1)
            )
            + sum(
                ((rho / 2) * cs.norm_2(self.x_cat[:, [k]] - self.z[:, [k]]) ** 2)
                for k in range(self.N + 1)
            )
        )
