from typing import List
import casadi as cs
import numpy as np
from csnlp.wrappers import Mpc
import gurobipy as gp
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MpcMld:
    """An MPC that converts a PWA mpc problem into a MIP."""

    def __init__(self, system: dict, N: int) -> None:
        """Instantiate the mld based mpc. In the constructor pwa system is converted
        to mld and the associated dynamics and constraints are created, along with states
        and control variables.

        Parameters
        ----------
        system: dict
            Dictionary containing the definition of the PWA system {S, R, T, A, B, c, D, E, F, G}.
             When S[i]x+R[x]u <= T[i] -> x+ = A[i]x + B[i]u + c[i].
             For MLD conversion the state and input must be constrained: Dx <= E, Fu <= G.
        N:int
            Prediction horizon length."""

        # extract values from system
        S = system["S"]
        R = system["R"]
        T = system["T"]
        A = system["A"]
        B = system["B"]
        c = system["c"]
        D = system["D"]
        E = system["E"]
        F = system["F"]
        G = system["G"]
        s = len(S)  # number of PWA regions
        n = A[0].shape[0]
        m = B[0].shape[1]

        # calculate the upper and lower bounds used in the MLD form

        M_st = [None] * s  # The upper bound for each region
        model_lin = gp.Model("linear model for mld set-up")
        model_lin.setParam("OutputFlag", 0)

        x = model_lin.addMVar((n, 1), lb=-float("inf"), ub=float("inf"), name="x_lin")
        u = model_lin.addMVar((m, 1), lb=-float("inf"), ub=float("inf"), name="u_lin")
        model_lin.addConstr(D @ x <= E, name="state constraints")
        model_lin.addConstr(F @ u <= G, name="control constraints")
        for i in range(s):
            obj = S[i] @ x + R[i] @ u - T[i]
            M_st[i] = np.zeros(obj.shape)
            for j in range(obj.shape[0]):
                model_lin.setObjective(obj[j, 0], gp.GRB.MAXIMIZE)
                model_lin.update()
                model_lin.optimize()
                M_st[i][j, 0] = model_lin.ObjVal
        logger.critical(
            "Solved linear model for PWA region bounds, M_star = " + str(M_st)
        )

        # bounds for state updates

        M_ub = np.zeros((n, 1))
        m_lb = np.zeros((n, 1))
        for j in range(n):
            M_regions = [None] * s
            m_regions = [None] * s
            for i in range(s):
                obj = A[i][j, :] @ x + B[i][j, :] @ u + c[i][j, :]
                model_lin.setObjective(obj, gp.GRB.MAXIMIZE)
                model_lin.update()
                model_lin.optimize()
                M_regions[i] = model_lin.ObjVal
                model_lin.setObjective(obj, gp.GRB.MINIMIZE)
                model_lin.update()
                model_lin.optimize()
                m_regions[i] = model_lin.ObjVal
            M_ub[j] = np.max(M_regions)
            m_lb[j] = np.min(m_regions)
        logger.critical(
            "Solved linear model for PWA state update bounds, M = "
            + str(M_ub.T)
            + "', m = "
            + str(m_lb.T)
            + "'"
        )

        # build mld model

        mpc_model = gp.Model("mld_mpc")
        mpc_model.setParam("OutputFlag", 1)
        # mpc_model.setParam("MIPStart", 1)  # using warm-starting from previous sol

        # Uncomment if you need to differentiate between infeasbile and unbounded
        mpc_model.setParam("DualReductions", 0)

        x = mpc_model.addMVar(
            (n, N + 1), lb=-float("inf"), ub=float("inf"), name="x"
        )  # state
        u = mpc_model.addMVar(
            (m, N), lb=-float("inf"), ub=float("inf"), name="u"
        )  # control

        # auxillary var z has 3 dimensions. (Region, state, time)
        z = mpc_model.addMVar((s, n, N), lb=-float("inf"), ub=float("inf"), name="z")
        # binary auxillary var
        delta = mpc_model.addMVar((s, N), vtype=gp.GRB.BINARY, name="delta")

        # constraint that only 1 delta can be active at each time step
        mpc_model.addConstrs(
            (gp.quicksum(delta[i, j] for i in range(s)) == 1 for j in range(N)),
            name="Delta sum constraints",
        )

        # constraints along predictions horizon for dynamics, state and control
        for k in range(N):
            # Set the branch priority of the binaries proportional to earlyness (earlier is more important)
            for i in range(s):
                # delta[i, k].setAttr('BranchPriority', s*N - s*k - i)
                # delta[i, k].setAttr('BranchPriority', N-k)
                pass

            # add state and input constraints to model, then binary and auxillary constraint, then dynamics constraints

            mpc_model.addConstr(D @ x[:, [k]] <= E, name="state constraints")
            mpc_model.addConstr(F @ u[:, [k]] <= G, name="control constraints")

            mpc_model.addConstrs(
                (
                    S[i] @ x[:, [k]] + R[i] @ u[:, [k]] - T[i]
                    <= M_st[i] * (1 - delta[i, [k]])
                    for i in range(s)
                ),
                name="Region constraints",
            )

            mpc_model.addConstrs(
                (z[i, :, k] <= M_ub @ (delta[i, [k]]) for i in range(s)),
                name="Z leq binary constraints",
            )

            mpc_model.addConstrs(
                (z[i, :, k] >= m_lb @ (delta[i, [k]]) for i in range(s)),
                name="Z geq binary constraints",
            )

            mpc_model.addConstrs(
                (
                    z[i, :, k].reshape(n, 1)
                    <= A[i] @ x[:, [k]]
                    + B[i] @ u[:, [k]]
                    + c[i]
                    - (m_lb * (1 - delta[i, [k]]))
                    for i in range(s)
                ),
                name="Z leq state constraints",
            )

            mpc_model.addConstrs(
                (
                    z[i, :, k].reshape(n, 1)
                    >= (A[i] @ x[:, [k]])
                    + (B[i] @ u[:, [k]])
                    + c[i]
                    - (M_ub * (1 - delta[i, [k]]))
                    for i in range(s)
                ),
                name="Z geq state constraints",
            )

            mpc_model.addConstr(
                x[:, [k + 1]]
                == gp.quicksum(z[i, :, k].reshape(n, 1) for i in range(s)),
                name="dynamics",
            )

        mpc_model.addConstr(
            D @ x[:, [N]] <= E, name="state constraints"
        )  # final state constraint

        # trivial terminal constraint condition x(N) = 0
        # mpc_model.addConstr(x[:, [N]] == np.zeros((n, 1)))

        # IC constraint - gets updated everytime solve_mpc is called
        self.IC = mpc_model.addConstr(x[:, [0]] == np.zeros((n, 1)), name="IC")

        # assign parts of model to be used by class later
        self.mpc_model = mpc_model
        self.x = x
        self.u = u
        self.z = z
        self.delta = delta
        self.n = n
        self.m = m
        self.N = N

        logger.critical("MLD MPC setup complete.")

    def set_cost(
        self, Q_x, Q_u, x_goal: np.ndarray = None, u_goal: np.ndarray = None
    ):
        """Set cost of the MIP as sum_k x(k)' * Q_x * x(k) + u(k)' * Q_u * u(k).
        Restricted to quadratic in the states and control.
        If x_goal or u_goal passed the cost uses (x-x_goal) and (u_goal)"""

        # construct zero goal points if not passed
        if x_goal is None:
            x_goal = np.zeros((self.x[:, [0]].shape[0], self.N))
        if u_goal is None:
            u_goal = np.zeros((self.u[:, [0]].shape[0], self.N))

        obj = 0
        for k in range(self.N):
            obj += (self.x[:, k] - x_goal[:, k].T) @ Q_x @ (self.x[:, [k]] - x_goal[:, [k]]) + (
                self.u[:, k] - u_goal[:, k].T
            ) @ Q_u @ (self.u[:, [k]] - u_goal[:, [k]])
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def solve_mpc(self, state):
        self.IC.RHS = state
        self.mpc_model.optimize()
        if self.mpc_model.Status == 2:  # check for successful solve
            u = self.u.X
            x = self.x.X
        else:
            u = np.zeros((self.m, self.N))
            x = np.zeros((self.n, self.N))
            logger.info("Infeasible")

        runtime = self.mpc_model.Runtime
        node_count = self.mpc_model.NodeCount
        return u[:, [0]], x
