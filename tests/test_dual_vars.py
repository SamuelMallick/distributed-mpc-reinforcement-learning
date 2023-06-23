import numpy as np
import cvxpy as cp

n = 2
n_l = 2

Q1 = np.array([[1, 2], [2, 4]])
Q2 = np.array([[1, 0.2], [0.2, 0.5]])
A1 = 10 * np.random.rand(1, n * n_l) - 5
A2 = 10 * np.random.rand(1, n * n_l) - 5
b1 = -5 * np.random.rand(1)
b2 = -5 * np.random.rand(1)

x1 = cp.Variable((n, 1))
x2 = cp.Variable((n, 1))

cost = cp.QuadForm(x1, Q1) + cp.QuadForm(x2, Q2)
objective = cp.Minimize(cost)

constraints = [A1 @ cp.vstack([x1, x2]) <= b1, A2 @ cp.vstack([x1, x2]) <= b2]

problem = cp.Problem(objective, constraints)

problem.solve()

print("x1 = " + str(x1.value))
print("x2 = " + str(x2.value))

print("d1 = " + str(constraints[0].dual_value))
print("d2 = " + str(constraints[1].dual_value))

# ADMM
