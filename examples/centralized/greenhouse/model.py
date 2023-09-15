# model of lettuce greenhouse from van Henten thesis (1994)
import numpy as np

# model parameters
nx = 4
nu = 3
nd = 4
ts = 60 * 15  # 15 minute time steps
u_min = np.zeros((3, 1))
u_max = np.array([[1.2], [7.5], [150]])
du_lim = 0.1 * u_max

# disturbance profile
d = np.load("examples/centralized/greenhouse/disturbances.npy")


def get_disturbance_profile():
    return d


def get_control_bounds():
    return u_min, u_max, du_lim


def get_y_min(d):
    if d[0] < 10:
        return np.array([[0], [0], [10], [0]])
    else:
        return np.array([[0], [0], [15], [0]])


def get_y_max(d):
    if d[0] < 10:
        return np.array([[float("inf")], [1.6], [15], [70]])
    else:
        return np.array([[float("inf")], [1.6], [20], [70]])


def get_model_details():
    return nx, nu, nd, ts


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

# continuos time model

# sub-functions within dynamics
def psi(x, d):
    return p[3] * d[0] + (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7])


def phi_phot_c(x, d):
    return (
        (1 - np.exp(-p[2] * x[0]))
        * (p[3] * d[0] * (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7]))
    ) / (psi(x, d))


def phi_vent_c(x, u, d):
    return (u[1] * 1e-3 + p[10]) * (x[1] - d[1])


def phi_vent_h(x, u, d):
    return (u[1] * 1e-3 + p[10]) * (x[3] - d[3])


def phi_trasnp_h(x):
    return (
        p[1]
        * (1 - np.exp(-p[2] * x[0]))
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
    return np.array([[dx1], [dx2], [dx3], [dx4]])


def output(x):
    """Output function of state y = output(x)"""
    y1 = 1e3 * x[0]
    y2 = ((1e3 * p[11] * (x[2] + p[12])) / (p[13] * p[14])) * x[1]
    y3 = x[2]
    y4 = (
        (1e2 * p[11] * (x[2] + p[12])) / (11 * np.exp((p[26] * x[2]) / (x[2] + p[27])))
    ) * x[3]

    # TODO add in measurement uncertainty in measurement of output
    return np.array([[y1], [y2], [y3], [y4]])
