from typing import List, Literal, Optional

import casadi as cs
import numpy as np
from csnlp.wrappers.wrapper import Nlp

from rldmpc.mpc.mpc_admm import MpcAdmm


class MpcSwitching(MpcAdmm):
    """An admm based mpc for a pwa system. Dynamics and constrains switch every time step of the horizon.
    They are initialised as all zeros, to be set by the agent."""

    def set_dynamics(
        self,
        nx_l: int,
        nu_l: int,
        r: int,
        x: cs.SX,
        u: cs.SX,
        x_c_list: List[cs.SX],
    ) -> None:
        """Initialised the switching dynamics and constraints with the correct dimension and all zero values

        Parameters
        ----------
        nx_l: int
            Dimension of local state.
        nu_l: int
            Dimension of local control.
        r: int
            Number of rows in inequality that checks regions Sx + Ru <= T.
        x: cs.SX
            Local state symbolic var.
        u: cs.SX
            Local control symbolic var.
        x_c_list: List[cs.SX]
            List of coupled state symbolic vars."""

        num_neighbours = len(x_c_list)

        # create the params for switching dynamics and contraints
        A_list = []
        B_list = []
        c_list = []
        S_list = []
        R_list = []
        T_list = []
        Ac_list = []
        for k in range(self.horizon):
            A_list.append(self.parameter(f"A_{k}", (nx_l, nx_l)))
            B_list.append(self.parameter(f"B_{k}", (nx_l, nu_l)))
            c_list.append(self.parameter(f"c_{k}", (nx_l, 1)))
            S_list.append(self.parameter(f"S_{k}", (r, nx_l)))
            R_list.append(self.parameter(f"R_{k}", (r, nu_l)))
            T_list.append(self.parameter(f"T_{k}", (r, 1)))
            Ac_list.append([])
            for i in range(num_neighbours):
                Ac_list[k].append(self.parameter(f"Ac_{k}_{i}", (nx_l, nx_l)))

            # initialise as all zeros
            self.fixed_pars_init[f"A_{k}"] = np.zeros((nx_l, nx_l))
            self.fixed_pars_init[f"B_{k}"] = np.zeros((nx_l, nu_l))
            self.fixed_pars_init[f"c_{k}"] = np.zeros((nx_l, 1))
            self.fixed_pars_init[f"S_{k}"] = np.zeros((r, nx_l))
            self.fixed_pars_init[f"R_{k}"] = np.zeros((r, nu_l))
            self.fixed_pars_init[f"T_{k}"] = np.zeros((r, 1))
            for i in range(num_neighbours):
                self.fixed_pars_init[f"Ac_{k}_{i}"] = np.zeros((nx_l, nx_l))

        # dynamics and region constraints - added manually due to coupling
        for k in range(self.horizon):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += Ac_list[k][i] @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A_list[k] @ x[:, [k]] + B_list[k] @ u[:, [k]] + c_list[k] + coup,
                "==",
                x[:, [k + 1]],
            )
            self.constraint(
                f"region_{k}",
                S_list[k] @ x[:, [k]] + R_list[k] @ u[:, [k]],
                "<=",
                T_list[k],
            )
