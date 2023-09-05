import casadi as cs
from csnlp.wrappers import Mpc


class MpcAdmm(Mpc[cs.SX]):
    """A wrapper for the MPC controller, allowing easy construction of the local MPC
      minimisation in an ADMM scheme."""
