from typing import Tuple
import numpy as np
from scipy.signal import cont2discrete
import casadi as cs


def forward_euler(
    A: np.ndarray, B: np.ndarray, ts: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using forward euler method"""
    Ad = np.eye(A.shape[0]) + ts * A
    Bd = ts * B
    return Ad, Bd


def zero_order_hold(
    A: np.ndarray, B: np.ndarray, ts: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using ZOH"""
    C = np.eye(A.shape[0])
    D = np.array([[0]])
    Ad = cs.expm(A)
    Bd = cs.inv(A)@(Ad - np.eye(A.shape[0]))@B
    return Ad, Bd

def tustin(
    A: np.ndarray, B: np.ndarray, ts: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using ZOH"""
    C = np.eye(A.shape[0])
    D = np.array([[0]])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), ts, method="bilinear")
    return Ad, Bd

if __name__ == "__main__":
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    print(f"Cont: A = {A}, B = {B}")
    Ad, Bd = forward_euler(A, B, 0.1)
    print(f"Euler: Ad = {Ad}, Bd = {Bd}")
    Ad, Bd = zero_order_hold(A, B, 0.1)
    print(f"ZOH: Ad = {Ad}, Bd = {Bd}")
