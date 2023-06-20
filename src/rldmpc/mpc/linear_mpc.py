import cvxpy as cp
import numpy as np

class LinearMPC():

    def __init__(self, n, m, N, A, B, Q_x, Q_u) -> None:
        """Instantiates a linear MPC and sets up the optimisation problem in cvxpy."""

        # optimisation variables

        x = cp.Variable((n, N+1))
        u = cp.Variable((m, N))

        # parameters that can be modified later

        x0 = cp.Parameter((n, 1))        

        # constraints and obj function over horizon

        cost = 0
        constraints = [x[:, [0]] == x0]
        for k in range(N):
            cost += cp.quad_form(x[:, [k]], Q_x) + cp.quad_form(u[:, [k]], Q_u)

            constraints += [x[:, [k+1]] == A @ x[:, [k]] + B @ u[:, [k]]]
        objective = cp.Minimize(cost)

        # create opt problem instance

        self.problem = cp.Problem(objective, constraints)

        # assign vars to class that are needed later

        self.x0 = x0
        self.u = u


    def solve_mpc(self, x0):
        """Solve the mpc for the given IC x0"""

        self.x0.value = x0
        self.problem.solve()
        u_opt = self.u.value
        
        return u_opt[:, [0]]


if __name__ == "__main__":
    mpc = LinearMPC(2, 1, 10, np.eye(2), np.eye(1), np.eye(2), np.eye(1))
    print(mpc.solve_mpc(x0=np.ones((2, 1))))
    print(mpc.solve_mpc(x0=2*np.zeros((2, 1))))
