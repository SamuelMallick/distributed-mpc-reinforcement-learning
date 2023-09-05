class ADMM:
    """Class for coordinating the ADMM procedure of a network of agents"""

    def __init__(self, iters, N, nx_l, nu_l) -> None:
        self.iters = iters  # fixed number of ADMM iterations
        self.N = N  # length of MPC horizon
        self.nx_l = nx_l  # dimension of local state
        self.nu_l = nu_l  # dimension of local control
