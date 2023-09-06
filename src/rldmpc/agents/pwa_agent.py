from typing import Collection, List, Literal, Optional, Union
from csnlp import Nlp
from csnlp.wrappers import Mpc
import numpy as np
import casadi as cs
from mpcrl.agents.agent import ActType, Agent, ObsType, SymType


class PwaAgent(Agent[SymType]):
    """An agent who has knowledge of it's own PWA dynamics and can use this to do things such as
    identify PWA regions given state and control trajectories."""

    def __init__(
        self,
        mpc: Mpc,
        fixed_parameters: dict,
        pwa_system: dict,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        """Initialise the agent.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        pwa_system: dict
            Contains {S, R, T, A, B, c, D, E, F, G, [Ac_j]}, where each is a list of matrices defining dynamics.
            When the inequality S[i]x + R[i]u <= T[i] is true, the dynamics are x^+ = A[i]x + B[i]u + c[i] + sum_j Ac_j[i] x_j.
            State constraints are: Dx <= E.
            Control constraints are: Fu <= G.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies."""
        super().__init__(mpc, fixed_parameters, warmstart, name)
        self.S = pwa_system["S"]
        self.R = pwa_system["R"]
        self.T = pwa_system["T"]
        self.A = pwa_system["A"]
        self.B = pwa_system["B"]
        self.c = pwa_system["c"]
        self.D = pwa_system["D"]
        self.E = pwa_system["E"]
        self.F = pwa_system["F"]
        self.G = pwa_system["G"]
        self.Ac = pwa_system["Ac"]

        self.num_neighbours = len(self.Ac[0])

    def next_state(self, x: np.ndarray, u: np.ndarray, xc: List[np.ndarray] = None):
        """Increment the dynamics as x+ = A[i]x + B[i]u + c[i] + sum_j Ac[i]_j xc_j
        if S[i]x + R[i]u <= T.
        If the coupled states xc are not passed the coupling part of dynamics is ignored."""
        for i in range(len(self.S)):
            if all(self.S[i] @ x + self.R[i] @ u <= self.T[i]):
                if xc is None:
                    return self.A[i] @ x + self.B[i] @ u + self.c[i]
                else:
                    return (
                        self.A[i] @ x
                        + self.B[i] @ u
                        + self.c[i]
                        + sum(self.Ac[i][j] @ xc[j] for j in range(self.num_neighbours))
                    )

        raise RuntimeError("Didn't find PWA region for given state-control.")

    def eval_sequences(
        self, x0: np.ndarray, u: np.ndarray, xc: List[np.ndarray] = None
    ):
        """Evaluate all possible sqitching sequences of PWA dynamics by rolling out
        dynamics from state x, applying control u, and coupled states xc."""

        N = u.shape[1]  # horizon length
        s = [[0] * N]  # list of sequences, start with just zeros
        x = x0  # temp state

        for k in range(N):
            current_regions = self.identify_regions(x, u[:, [k]])
            for i in range(len(current_regions)):
                if i == 0:  # first one gets appended to all current sequences
                    for j in range(len(s)):
                        s[j][k] = current_regions[i]
                else:
                    # for other identified regions, they define new sequences
                    s_temp = []
                    for j in range(len(s)):
                        s_temp.append(s[j].copy())
                        s_temp[j][k] = current_regions[i]
                    s = s + s_temp
            # rollout dynamics by one
            if xc is None:
                x = self.next_state(x, u[:, [k]])
            else:
                x = self.next_state(
                    x, u[:, [k]], [xc[i][:, [k]] for i in range(self.num_neighbours)]
                )
        return s

    def identify_regions(self, x: np.ndarray, u: np.ndarray, eps: int = 0):
        """Generate the indices of the regions where Sx+Ru<=T + eps is true."""

        regions = []
        for i in range(len(self.S)):
            if all(self.S[i] @ x + self.R[i] @ u <= self.T[i] + eps):
                regions.append(i)
        return regions

    def set_sequence(self, s: List[int]):
        """Modify the parameters in the constraints of the ADMM Mpc to
        enforce the sequence s"""

        # TODO confirm that these parameters have been named correctly in the MPC
        #
        for i in range(len(s)):
            self.fixed_parameters[f"A_{i}"] = self.A[s[i]]
            self.fixed_parameters[f"B_{i}"] = self.B[s[i]]
            self.fixed_parameters[f"c_{i}"] = self.c[s[i]]
            self.fixed_parameters[f"S_{i}"] = self.S[s[i]]
            self.fixed_parameters[f"R_{i}"] = self.R[s[i]]
            self.fixed_parameters[f"T_{i}"] = self.T[s[i]]  
            for j in range(self.num_neighbours):
                self.fixed_parameters[f"Ac_{i}_{j}"] = self.Ac[s[i]][j]