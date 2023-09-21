from math import factorial
from typing import Tuple

import casadi as cs
import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete


def forward_euler(
    A: np.ndarray, B: np.ndarray, ts: float, c: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using forward euler method.
    If c also passed uses system x_dot = Ax + Bu + c"""
    Ad = np.eye(A.shape[0]) + ts * A
    Bd = ts * B
    if c is None:
        return Ad, Bd
    else:
        cd = ts * c
        return Ad, Bd, cd


def zero_order_hold(
    A: np.ndarray, B: np.ndarray, ts: float, N: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using ZOH"""
    n = A.shape[0]
    I = np.eye(n)
    if isinstance(A, np.ndarray):
        D = expm(ts * np.vstack((np.hstack([A, I]), np.zeros((n, 2 * n)))))
        Ad = D[:n, :n]
        Id = D[:n, n:]
        Bd = Id.dot(B)
    else:
        M = ts * cs.vertcat(cs.horzcat(A, I), np.zeros((n, 2 * n)))
        D = sum(cs.mpower(M, k) / factorial(k) for k in range(N))
        Ad = D[:n, :n]
        Id = D[:n, n:]
        Bd = Id @ B
    return Ad, Bd


def tustin(A: np.ndarray, B: np.ndarray, ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using ZOH"""
    C = np.eye(A.shape[0])
    D = np.array([[0]])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), ts, method="bilinear")
    return Ad, Bd
