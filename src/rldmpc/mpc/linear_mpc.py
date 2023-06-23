import cvxpy as cp
import numpy as np


class LinearMPC:
    def __init__(self, n: int, m, N, A, B, u_bnd, x_bnd, w, Q_x, Q_u) -> None:
        """Instantiates a linear MPC and sets up the optimisation problem in cvxpy."""

        # optimisation variables

        x = cp.Variable((n, N + 1))  # state
        u = cp.Variable((m, N))  # control
        s = cp.Variable((n, N + 1))  # slack vars

        # parameters that can be modified later

        x0 = cp.Parameter((n, 1))

        # constraints and obj function over horizon

        cost = 0
        constraints = [x[:, [0]] == x0]
        for k in range(N):
            # constraints += [s[:, [k]] == 0]  # uncomment to force hard state constraints

            cost += (
                cp.quad_form(x[:, [k]], Q_x)
                + cp.quad_form(u[:, [k]], Q_u)
                + w.T @ s[:, [k]]
            )  # stage cost + viol pnlty

            constraints += [x[:, [k + 1]] == A @ x[:, [k]] + B @ u[:, [k]]]  # dynamics
            constraints += [
                -u[:, [k]] >= u_bnd[0],
                u[:, [k]] <= u_bnd[1],
            ]  # elementwise box bounds on input
            constraints += [
                x_bnd[0] - s[:, [k]] <= x[:, [k]],
                x[:, [k]] <= x_bnd[1] + s[:, [k]],
            ]  # soft bounds on state
            constraints += [s[:, [k]] >= 0]  # positive slack constraint

        cost += cp.quad_form(x[:, [N]], Q_x)  # terminal cost
        cost += w.T @ s[:, [N]]  # terminal viol pnlty
        constraints += [
            x_bnd[0] - s[:, [N]] <= x[:, [N]],
            x[:, [N]] <= x_bnd[1] + s[:, [N]],
        ]  # soft terminal cnstrt
        constraints += [s[:, [N]] >= 0]  # terminal positive slack constraint
        objective = cp.Minimize(cost)

        # create opt problem instance

        self.problem = cp.Problem(objective, constraints)

        # assign vars to class that are needed later

        self.x0 = x0
        self.u = u
        self.x = x
        self.s = s
        self.default_u = np.zeros((m, N))  # control if infeasible

    def solve_mpc(self, x0):
        """Solve the mpc for the given IC x0"""

        self.x0.value = x0
        self.problem.solve(solver=cp.ECOS, verbose=False)

        # check feasibility

        if self.problem.status == "infeasible":
            print("Infeasible MPC. Controls assigned to zeros.")
            u_opt = self.default_u
        else:
            u_opt = self.u.value

        return u_opt[:, [0]]


if __name__ == "__main__":
    pass
    # mpc = LinearMPC(2, 1, 10, np.eye(2), np.eye(1), np.eye(2), np.eye(1))
    # print(mpc.solve_mpc(x0=np.ones((2, 1))))
    # print(mpc.solve_mpc(x0=2 * np.zeros((2, 1))))
