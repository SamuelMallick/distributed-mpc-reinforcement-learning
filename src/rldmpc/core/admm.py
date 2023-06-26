import numpy as np
import cvxpy as cp


def g_map(Adj: np.ndarray):
    """Construct the ADMM mapping from local to global variables from an adjacency matrix."""
    n = Adj.shape[0]  # number of agents
    G: list[list[int]] = []
    for i in range(n):
        G.append([])
        for j in range(n):
            if Adj[i, j] == 1 or i == j:
                G[i].append(j)
    return G


def admm_inner(
    x: cp.Variable,
    cost: cp.Expression,
    constraints: list[cp.Constraint],
    y: np.ndarray,
    z: np.ndarray,
    rho: float,
):
    """Solve the x-minimisation inner problem of ADMM for given duals (y) and global estimates (z)."""

    # modify cost expression with residual terms
    inner_cost = (
        cost
        + cp.sum([y[:, i].T @ x[:, i] for i in range(x.shape[1])])
        + (rho / 2) * cp.square(cp.norm(x - z, 2))
    )

    obj = cp.Minimize(inner_cost)
    inner_problem = cp.Problem(obj, constraints)

    inner_problem.solve()

    dual_vals = [constraints[i].dual_value for i in range(len(constraints))]
    return x.value, dual_vals


def admm(
    x_list: list[cp.Variable],
    cost_list: list[cp.Expression],
    constraint_list: list[list[cp.Constraint]],
    G: list[list[int]],
):
    """Calculates the ADMM solution to the the optimisation problem given."""

    iters = 100  # number of iterations for the algorithm
    rho = 0.8  # penalty co-efficient

    n = len(G)  # number of agents
    nx_l = x_list[0].shape[0]  # dimension of local vars

    # create auxillary vars: y - lagrange multipliers for consensus constraints, z - global vars, and temporary x vars - x_temp

    y_list: list[np.ndarray] = []
    x_temp_list: list[np.ndarray] = []  # stores the intermediate numerical values for x
    dual_temp_list: list[np.ndarray] = []  # dual vals of from inner probs
    for i in range(n):
        y_list.append(np.zeros(x_list[i].shape))
        x_temp_list.append(np.zeros(x_list[i].shape))
        dual_temp_list.append(
            np.zeros((constraint_list[i][0].shape[0], len(constraint_list[i])))
        )
    z = np.zeros((nx_l, n))

    for iter in range(iters):
        # x-update - TODO parallelise

        for i in range(n):
            x_temp_list[i], dual_temp_list[i] = admm_inner(
                x_list[i],
                cost_list[i],
                constraint_list[i],
                y_list[i],
                z[:, [j for j in G[i]]],
                rho,
            )

        # z-update -> essentially an averaging of all agents' optinions on each z

        for i in range(n):  # looping through each global var associated with each agent
            count = 0
            sum = np.zeros((nx_l, 1))
            for j in range(n):  # looping through agents again to get each's optinion
                if (
                    i in G[j]
                ):  # if agent j has an opinion on the value of agent i's local var
                    count += 1
                    sum += x_temp_list[j][:, [G[j].index(i)]]
            z[:, [i]] = sum / count  # average the opinions

        # y-update - TODO parallelise

        for i in range(n):
            z_temp = z[:, [j for j in G[i]]]  # global vars relevant for agent i
            y_list[i] = y_list[i] + rho * (x_temp_list[i] - z_temp)

    return [x_list[i].value for i in range(n)], dual_temp_list


if __name__ == "__main__":
    n = 3
    nx_l = 2  # dimension of optimisation var per agent
    Adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # adjacency graph of network
    G = g_map(Adj)

    # make local copies of variables that are coupled

    x_list: list[cp.Variable] = []
    for i in range(n):
        num_local_vars = len(G[i])
        x_list.append(cp.Variable((nx_l, num_local_vars)))

    x_glob = cp.Variable((nx_l, n), "glob")  # global var

    # make the local costs

    cost_list: list[cp.Expression] = []
    cost_glob = 0  # global cost
    for i in range(n):
        Q = np.random.rand(nx_l, nx_l)
        Q = Q @ Q.T
        f = 2 * np.random.rand(nx_l, 1) - 1
        local_var = x_list[i][
            :, G[i].index(i)
        ]  # in this example the local cost only depends on the local var. Coupling is in the constraints
        cost_list.append(cp.quad_form(local_var, Q) + f.T @ local_var)

        cost_glob += cp.quad_form(x_glob[:, i], Q) + f.T @ x_glob[:, i]

    obj_glob = cp.Minimize(cost_glob)

    # make the constraints for each agent - in this example a restriction on the sum of coupled vars

    constraint_list: list[list[cp.Constraint]] = []
    constraints_glob: list[cp.Constraint] = []
    for i in range(n):
        val = np.random.rand(1)
        constraint_list.append([cp.sum(x_list[i], axis=1) <= val])

        constraints_glob.append(cp.sum([x_glob[:, j] for j in G[i]]) <= val)

    problem_glob = cp.Problem(obj_glob, constraints_glob)
    problem_glob.solve()

    print("------------------------")
    print("Centralised solution")
    print("Value: " + str(obj_glob.value))
    print("x: " + str(x_glob.value))
    for i in range(len(constraints_glob)):
        print("d_" + str(i + 1) + ": " + str(constraints_glob[i].dual_value))
    print("------------------------")

    x_list_admm, dual_list_admm = admm(x_list, cost_list, constraint_list, G)
    print("------------------------")
    print("ADMM solution")
    x_admm_opt = np.array([x_list_admm[i][:, G[i].index(i)] for i in range(n)]).T
    print("x: " + str(x_admm_opt))
    for i in range(n):
        print("d_" + str(i + 1) + ": " + str(dual_list_admm[i]))
    print("------------------------")

    print("------------------------")
    print("Errors")
    print("x: " + str(np.linalg.norm(x_glob.value - x_admm_opt)))
    print(
        "d: "
        + str(
            np.sum(
                [
                    np.linalg.norm(
                        constraints_glob[i].dual_value - dual_list_admm[i][0]
                    )
                    for i in range(n)
                ]
            )
        )
    )
