# model of lettuce greenhouse from van Henten thesis (1994)
import numpy as np
import casadi as cs

# model parameters
nx = 4
nu = 3
nd = 4
ts = 60 * 15  # 15 minute time steps
time_steps_per_day = 24 * 4  # how many 15 minute incrementes there are in a day
days_to_grow = 40  # length of each episode, from planting to harvesting

# u bounds
u_min = np.zeros((3, 1))
u_max = np.array([[1.2], [7.5], [150]])
du_lim = 0.1 * u_max

# noise terms
mean = 0
sd = 0

# disturbance profile
d = np.load("examples/centralized/greenhouse/disturbances.npy")


def get_model_details():
    return nx, nu, nd, ts

def get_disturbance_profile(init_day: int):
    # an extra days worth added to the profile for the prediction horizon
    return d[
        :,
        init_day
        * time_steps_per_day : (init_day + days_to_grow + 1)
        * time_steps_per_day,
    ]

def get_control_bounds():
    return u_min, u_max, du_lim

def get_y_min(d):
    if d[0] < 10:
        return np.array([[0], [0], [10], [0]])
    else:
        return np.array([[0], [0], [15], [0]])


def get_y_max(d):
    if d[0] < 10:
        return np.array([[1e6], [1.6], [15], [70]])  # 1e6 replaces infinity
    else:
        return np.array([[1e6], [1.6], [20], [70]])


p = [
    0.544,
    2.65e-7,
    53,
    3.55e-9,
    5.11e-6,
    2.3e-4,
    6.29e-4,
    5.2e-5,
    4.1,
    4.87e-7,
    7.5e-6,
    8.31,
    273.15,
    101325,
    0.044,
    3e4,
    1290,
    6.1,
    0.2,
    4.1,
    0.0036,
    9348,
    8314,
    273.15,
    17.4,
    239,
    17.269,
    238.3,
]


def generate_perturbed_p():
    # Define the desired mean and covariance matrix
    mean = np.asarray(p)  # Mean vector is real vals
    cov_matrix = 0.05*np.eye(len(p))  # Covariance matrix

    # Generate samples from a uniform distribution over the unit hypercube
    num_samples = 1  # Number of samples
    uniform_samples = np.random.rand(num_samples, len(mean))

    # Transform the uniform samples to have the desired mean and covariance
    # Use the Cholesky decomposition of the covariance matrix
    cholesky_matrix = np.linalg.cholesky(cov_matrix)
    p_hat = np.dot(uniform_samples, cholesky_matrix.T) + mean

    return p_hat


# continuos time model
# TODO: make all sub functions take p as an input.
# Then when you want the real dynamics (rk4 OR df) you use a different function called df_real or rk4_real
# Then make a function multi_sample_dynamics(x_ext, u, d, Ns) which takes the extended state x = [x_1, x_2, ...], and for each sample computes
# the update x_1+ = rk4(x1, u, d, p_hat) for a NEW p_hat.
# however the same control and disturbances are used.

# sub-functions within dynamics
def psi(x, d):
    return p[3] * d[0] + (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7])


def phi_phot_c(x, d):
    return (
        (1 - cs.exp(-p[2] * x[0]))
        * (p[3] * d[0] * (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7]))
    ) / (psi(x, d))


def phi_vent_c(x, u, d):
    return (u[1] * 1e-3 + p[10]) * (x[1] - d[1])


def phi_vent_h(x, u, d):
    return (u[1] * 1e-3 + p[10]) * (x[3] - d[3])


def phi_trasnp_h(x):
    return (
        p[1]
        * (1 - cs.exp(-p[2] * x[0]))
        * (
            ((p[21]) / (p[22] * (x[2] + p[23])))
            * (np.exp((p[24] * x[2]) / (x[2] + p[25])))
            - x[3]
        )
    )


def df(x, u, d):
    """Continuous derivative of state d_dot = df(x, u, d)/dt"""
    dx1 = p[0] * phi_phot_c(x, d) - p[1] * x[0] * 2 ** (x[2] / 10 - 5 / 2)
    dx2 = (1 / p[8]) * (
        -phi_phot_c(x, d)
        + p[9] * x[0] * 2 ** (x[2] / 10 - 5 / 2)
        + u[0] * 1e-6
        - phi_vent_c(x, u, d)
    )
    dx3 = (1 / p[15]) * (
        u[2] - (p[16] * u[1] * 1e-3 + p[17]) * (x[2] - d[2]) + p[18] * d[0]
    )
    dx4 = (1 / p[19]) * (phi_trasnp_h(x) - phi_vent_h(x, u, d))
    return cs.vertcat(dx1, dx2, dx3, dx4)


def rk4_step(x, u, d):
    k1 = df(x, u, d)
    k2 = df(x + (ts / 2) * k1, u, d)
    k3 = df(x + (ts / 2) * k2, u, d)
    k4 = df(x + ts * k3, u, d)
    return x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def output(x):
    """Output function of state y = output(x)"""
    y1 = 1e3 * x[0]
    y2 = ((1e3 * p[11] * (x[2] + p[12])) / (p[13] * p[14])) * x[1]
    y3 = x[2]
    y4 = (
        (1e2 * p[11] * (x[2] + p[12])) / (11 * cs.exp((p[26] * x[2]) / (x[2] + p[27])))
    ) * x[3]

    # add noise to measurement
    noise = np.random.normal(mean, sd, (nx, 1))
    return cs.vertcat(y1, y2, y3, y4) + noise
