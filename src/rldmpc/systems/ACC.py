import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from rldmpc.utils.discretisation import forward_euler

# adaptive cruise control system. Single agent system outlined in:
# Adaptive Cruise Control for a SMART Car: A Comparison Benchmark for MPC-PWA Control Methods - D. Corona and B. De Schutter 2008


class ACC:
    nx_l = 2  # dimension of local state
    nu_l = 1  # dimension of local control
    ts = 1  # time step size for discretisation

    # local costs
    Q_x_l = np.diag([1, 0.1])
    Q_u_l = 0.1 * np.eye(nu_l)

    sep = np.array([[-50], [0]])  # desired seperation between vehicles states
    d_safe = 0  # safe distance between cars

    mass = 800  # mass
    c_fric = 0.5  # viscous friction coefficient
    mu = 0.01  # coulomb friction coefficient
    grav = 9.8  # gravity accel
    w_min = 105  # min rot speed rad/s
    w_max = 630  # max rot speed rad/s

    x1_min = 0  # min pos
    x1_max = 3000  # max_pos
    x2_min = 2  # min velocity
    x2_max = 40  # max velocity
    u_max = 1  # max throttle/brake
    a_acc = 2.5  # comfort acc
    a_dec = -2  # comfort dec
    d_safe = 15  # safe pos

    # transmission rate for each of the 6 gears
    p = [14.203, 10.310, 7.407, 5.625, 4.083, 2.933]
    b = [4057, 2945, 2116, 1607, 1166, 838]  # max traction force for each gear
    vl = [3.94, 5.43, 7.56, 9.96, 13.70, 19.10]
    vh = [9.46, 13.04, 18.15, 23.90, 32.93, 45.84]
    Te_max = 80  # maximum engine torque - constant in the range 200 < w < 480

    # PWA approximation of friction c*x2^2 = c1*x2 if x2 <= x2_max/2, = c2*x2-d
    beta = (3 * c_fric * x2_max**2) / (16)
    alpha = x2_max / 2
    c1 = beta / alpha
    c2 = (c_fric * x2_max**2 - beta) / (x2_max - alpha)
    d = beta - alpha * ((c_fric * x2_max**2 - beta) / (x2_max - alpha))
    # d = 0.230769

    # PWA approximation of gear function b(j, x2)
    # first step - consider only regions constant with velocity, therefore we use values in list b
    # second step (approximation) - make it so we can express b(j) as b(j) = beta_0 + j beta_1 with minimum deviation from values in list b

    opt_mod = gp.Model("linear model ")
    opt_mod.setParam("OutputFlag", 0)
    beta_0 = opt_mod.addVar(lb=-float("inf"), ub=float("inf"), name="beta_0")
    beta_1 = opt_mod.addVar(lb=-float("inf"), ub=float("inf"), name="beta_1")
    obj = 0
    for i in range(len(b)):
        j = i + 1  # gears go from 1-6
        obj += (b[i] - (beta_0 + j * beta_1)) ** 2
    opt_mod.setObjective(obj, gp.GRB.MINIMIZE)
    opt_mod.optimize()
    beta_0 = beta_0.X
    beta_1 = beta_1.X

    # third step (approximation) - express v_0 + v_1j <= s_dot <= v_0 + v_1(j+1)
    # this gives us a one to one mapping between gear and velocity
    # these values effect how we skew the choice of velocity to gear mapping
    # they are taken from the paper
    gamma_l = 1
    gamma_h = 100
    v_0 = opt_mod.addVar(lb=-float("inf"), ub=float("inf"), name="v_0")
    v_1 = opt_mod.addVar(lb=-float("inf"), ub=float("inf"), name="v_1")
    obj = 0
    for i in range(len(b)):
        j = i + 1  # gears go from 1-6
        obj += (
            gamma_l * (vl[i] - (v_0 + j * v_1)) ** 2
            + gamma_h * (vh[i] - (v_0 + (j + 1) * v_1)) ** 2
        )
    opt_mod.addConstr(v_0 + v_1 >= x2_min)
    opt_mod.setObjective(obj, gp.GRB.MINIMIZE)
    opt_mod.optimize()
    v_0 = v_0.X
    v_1 = v_1.X

    # build PWA system
    s = 7  # 7 PWA regions
    r = 2  # number of rows in Sx + RU <= T conditions
    S = []
    R = []
    T = []
    A = []
    B = []
    c = []

    for i in range(s):
        S.append(np.array([[0, 1], [0, -1]]))
        R.append(np.zeros((r, 1)))

    # manually append the limits
    T.append(np.array([[v_0 + v_1 * (2)], [-x2_min]]))
    T.append(np.array([[v_0 + v_1 * (3)], [-(v_0 + v_1 * (2))]]))
    T.append(np.array([[alpha], [-(v_0 + v_1 * (3))]]))
    T.append(np.array([[v_0 + v_1 * (4)], [-alpha]]))
    T.append(np.array([[v_0 + v_1 * (5)], [-(v_0 + v_1 * (4))]]))
    T.append(np.array([[v_0 + v_1 * (6)], [-(v_0 + v_1 * (5))]]))
    T.append(np.array([[x2_max], [-(v_0 + v_1 * (6))]]))

    # manually append the A matrices - first three regions have c1 and last four have c2 for friction
    A.append(np.array([[0, 1], [0, -(c1) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c1) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c1) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c2) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c2) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c2) / (mass)]]))
    A.append(np.array([[0, 1], [0, -(c2) / (mass)]]))

    # manually append B matrices
    B.append(np.array([[0], [(beta_0 + 1 * beta_1) / (mass)]]))
    B.append(np.array([[0], [(beta_0 + 2 * beta_1) / (mass)]]))
    # third and fourth share same gear as the split is over the friction coeff
    B.append(np.array([[0], [(beta_0 + 3 * beta_1) / (mass)]]))
    B.append(np.array([[0], [(beta_0 + 3 * beta_1) / (mass)]]))
    B.append(np.array([[0], [(beta_0 + 4 * beta_1) / (mass)]]))
    B.append(np.array([[0], [(beta_0 + 5 * beta_1) / (mass)]]))
    B.append(np.array([[0], [(beta_0 + 6 * beta_1) / (mass)]]))

    # manually append c matrices - last four regions have offset d due to friction PWA
    c.append(np.array([[0], [-mu * grav]]))
    c.append(np.array([[0], [-mu * grav]]))
    c.append(np.array([[0], [-mu * grav]]))
    c.append(np.array([[0], [-mu * grav - d / mass]]))
    c.append(np.array([[0], [-mu * grav - d / mass]]))
    c.append(np.array([[0], [-mu * grav - d / mass]]))
    c.append(np.array([[0], [-mu * grav - d / mass]]))

    # discretise the dynamics
    Ad = []
    Bd = []
    cd = []
    for i in range(s):
        Ad_i, Bd_i, cd_i = forward_euler(A[i], B[i], ts, c[i])
        Ad.append(Ad_i)
        Bd.append(Bd_i)
        cd.append(cd_i)

    D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    E = np.array([[x1_max], [-x1_min], [x2_max], [-x2_min]])
    F = np.array([[1], [-1]])
    G = np.array([[u_max], [u_max]])

    pwa_system = {
        "S": S,
        "R": R,
        "T": T,
        "A": Ad,
        "B": Bd,
        "c": cd,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    }

    def __init__(self, ep_len, N):
        # generate trajectory of leader
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 15
        leader_initial_pos = 500
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(ep_len + N):
            leader_state[:, [k + 1]] = leader_state[:, [k]] + self.ts * np.array(
                [[leader_speed], [0]]
            )
        self.leader_state = leader_state

    def get_traction_force(self, v):
        """Get the corresponding constant traction force for speed v."""
        for i in range(len(self.b)):
            j = i + 1
            if self.v_0 + j * self.v_1 <= v and v <= self.v_0 + (j + 1) * self.v_1:
                return self.b[i]

        # if speed is outside of bounds, use the traction force for the closest region
        if v <= self.v_0 + (2) * self.v_1:
            return self.b[0]
        if self.v_0 + 6 * self.v_1 <= v:
            return self.b[-1]
        raise RuntimeError("Didn't find any traction force for the given speed!")

    def get_pwa_system(self):
        """Get to system dictionary."""
        return self.pwa_system

    # the true non-linear dynamics of the car
    def step_car_dynamics_nl(self, x, u, n, ts):
        """Steps the car dynamics for n cars with non-linear model by ts seconds. x is state, u is control."""
        num_steps = 10
        DT = ts / num_steps
        for t in range(num_steps):
            x_temp = np.zeros(x.shape)
            for i in range(n):
                x_l = x[self.nx_l * i : self.nx_l * (i + 1), :]  # get local state
                u_l = u[self.nu_l * i : self.nu_l * (i + 1), :]  # get local control
                f = np.array(
                    [
                        [x_l[1, 0]],
                        [
                            -(self.c_fric * x_l[1, 0] ** 2) / (self.mass)
                            - self.mu * self.grav
                        ],
                    ]
                )
                B = np.array([[0], [self.get_traction_force(x_l[1, 0]) / self.mass]])
                x_temp[self.nx_l * i : self.nx_l * (i + 1), :] = x_l + DT * (
                    f + B * u_l
                )
            x = x_temp

        return x

    def step_car_dynamics_pwa(self, x, u, n, ts):
        """Steps the car dynamics for n cars with pwa model by ts seconds. x is state, u is control."""
        x_temp = np.zeros(x.shape)
        for i in range(n):
            x_l = x[self.nx_l * i : self.nx_l * (i + 1), :]  # get local state
            u_l = u[self.nu_l * i : self.nu_l * (i + 1), :]  # get local control
            for j in range(len(self.pwa_system["S"])):
                if all(
                    self.pwa_system["S"][j] @ x_l
                    + self.pwa_system["R"][j] @ u_l
                    <= self.pwa_system["T"][j] + np.array([[0], [1e-4]])    # buffer is to have one of the as a strict inequality
                ):
                    x_pwa = (
                        self.pwa_system["A"][j] @ x_l
                        + self.pwa_system["B"][j] @ u_l
                        + self.pwa_system["c"][j]
                    )
                    break
            x_temp[self.nx_l * i : self.nx_l * (i + 1), :] = x_pwa

        return x_temp

    def get_leader_state(self):
        return self.leader_state


if __name__ == "__main__":
    # check validity of PWA approximation

    acc = ACC(0, 0)
    pwa_sys = acc.get_pwa_system()

    test_len = 100
    np.random.seed(0)
    x0 = np.array([[10], [5], [20], [6]])
    u = np.random.random((2, test_len))  # random control samples between 0 and 1
    x_nl = np.zeros((4, test_len))  # non-linear traj
    x_pwa = np.zeros((4, test_len))  # pwa traj

    # set IC
    x_nl[:, [0]] = x0
    x_pwa[:, [0]] = x0
    for t in range(test_len - 1):
        # step non linear
        x_nl[:, [t + 1]] = acc.step_car_dynamics_nl(x_nl[:, [t]], u[:, [t]], 2, acc.ts)
        x_pwa[:, [t + 1]] = acc.step_car_dynamics_pwa(x_nl[:, [t]], u[:, [t]], 2, acc.ts)

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(x_nl[0, :], "r")
    axs[0].plot(x_pwa[0, :], "b")
    axs[1].plot(x_nl[1, :], "r")
    axs[1].plot(x_pwa[1, :], "b")
    plt.show()
