import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from model import Model

from dmpcrl.mpc.mpc_admm import MpcAdmm
from dmpcrl.utils.solver_options import SolverOptions


class LearnableMpc(Mpc[cs.SX]):
    """Abstract class for learnable MPC controllers. Implemented by centralized and distributed child classes"""

    discount_factor = 0.9

    def __init__(self, model: Model) -> None:
        """Initializes the learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model of the system.
        prediction_horizon : int
            The prediction horizon."""
        self.n = model.n
        self.nx_l, self.nu_l = model.nx_l, model.nu_l
        self.nx, self.nu = model.n * model.nx_l, model.n * model.nu_l
        self.x_bnd_l, self.u_bnd_l = model.x_bnd_l, model.u_bnd_l
        self.x_bnd, self.u_bnd = np.tile(model.x_bnd_l, model.n), np.tile(
            model.u_bnd_l, model.n
        )
        self.w_l = np.array(
            [[1.2e2, 1.2e2]]
        )  # penalty weight for constraint violations in cost
        self.w = np.tile(self.w_l, (1, self.n))
        self.adj = model.adj

        # standard learnable parameters dictionary for local agent
        self.learnable_pars_init_local = {
            "V0": np.zeros((1, 1)),
            "x_lb": np.reshape([0, 0], (-1, 1)),
            "x_ub": np.reshape([1, 0], (-1, 1)),
            "b": np.zeros(self.nx_l),
            "f": np.zeros(self.nx_l + self.nu_l),
        }


class CentralizedMpc(LearnableMpc):
    """A centralised learnable MPC controller."""

    def __init__(self, model: Model, prediction_horizon: int) -> None:
        """Initializes the centralized learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model of the system.
        prediction_horizon : int
            The prediction horizon."""
        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        Mpc.__init__(self, nlp, prediction_horizon)
        LearnableMpc.__init__(self, model)

        # renaming them for ease of use
        N = prediction_horizon
        gamma = self.discount_factor

        # create MPC parameters
        # dynamics paratmeters
        A_list = [
            self.parameter(f"A_{i}", (self.nx_l, self.nx_l)) for i in range(self.n)
        ]
        B_list = [
            self.parameter(f"B_{i}", (self.nx_l, self.nu_l)) for i in range(self.n)
        ]
        # if no coupling between i and j, A_c_list[i, j] = None, otherwise we add a parameterized matrix
        A_c_list = [
            [
                self.parameter(f"A_c_{i}_{j}", (self.nx_l, self.nx_l))
                for j in range(self.n)
                if self.adj[i, j]
            ]
            for i in range(self.n)
        ]
        b_list = [self.parameter(f"b_{i}", (self.nx_l, 1)) for i in range(self.n)]
        # cost parameters
        V0_list = [self.parameter(f"V0_{i}", (1,)) for i in range(self.n)]
        f_list = [
            self.parameter(f"f_{i}", (self.nx_l + self.nu_l, 1)) for i in range(self.n)
        ]
        # constraints parameters
        x_lb_list = [self.parameter(f"x_lb_{i}", (self.nx_l,)) for i in range(self.n)]
        x_ub_list = [self.parameter(f"x_ub_{i}", (self.nx_l,)) for i in range(self.n)]

        # initial values for learnable parameters
        A_l_inac, B_l_inac, A_c_l_inac = (
            model.A_l_innacurate,
            model.B_l_innacurate,
            model.A_c_l_innacurate,
        )

        self.learnable_pars_init = {
            f"{name}_{i}": val
            for name, val in self.learnable_pars_init_local.items()
            for i in range(self.n)
        }
        self.learnable_pars_init.update({f"A_{i}": A_l_inac for i in range(self.n)})
        self.learnable_pars_init.update({f"B_{i}": B_l_inac for i in range(self.n)})
        self.learnable_pars_init.update(
            {
                f"A_c_{i}_{j}": A_c_l_inac
                for i in range(self.n)
                for j in range(self.n)
                if self.adj[i, j]
            }
        )

        # concat some params for use in cost and constraint expressions
        V0 = cs.vcat(V0_list)
        x_lb = cs.vcat(x_lb_list)
        x_ub = cs.vcat(x_ub_list)
        b = cs.vcat(b_list)
        f = cs.vcat(f_list)

        # get centralized symbolic dynamics
        A, B = model.centralized_dynamics_from_local(A_list, B_list, A_c_list)

        # variables (state, action, slack)
        x, _ = self.state("x", self.nx)
        u, _ = self.action(
            "u",
            self.nu,
            lb=self.u_bnd[0].reshape(-1, 1),
            ub=self.u_bnd[1].reshape(-1, 1),
        )
        s, _, _ = self.variable("s", (self.nx, N), lb=0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + b, n_in=2, n_out=1)

        # other constraints
        self.constraint("x_lb", self.x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", self.x_bnd[1].reshape(-1, 1) + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            cs.sum1(V0)
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + self.w @ s)
            )
        )

        # solver
        solver = "qpoases"
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)


class LocalMpc(MpcAdmm, LearnableMpc):
    """Local learnable MPC."""

    def __init__(
        self,
        model: Model,
        prediction_horizon: int,
        num_neighbours: int,
        my_index: int,
        rho: float = 0.5,
    ) -> None:
        """Initializes the local learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model object containign system information.
        prediction_horizon : int
            The prediction horizon.
        num_neighbours : int
            The number of neighbours for the agent.
        my_index : int
            The index of the agent within its local augmented state.
        rho : float, optional
            The ADMM penalty parameter, by default 0.5.
        """
        N = prediction_horizon
        gamma = self.discount_factor
        self.rho = rho

        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        LearnableMpc.__init__(self, model)
        MpcAdmm.__init__(self, nlp=nlp, prediction_horizon=prediction_horizon)

        # MPC parameters
        V0 = self.parameter("V0", (1,))
        x_lb = self.parameter("x_lb", (self.nx_l,))
        x_ub = self.parameter("x_ub", (self.nx_l,))
        b = self.parameter("b", (self.nx_l, 1))
        f = self.parameter("f", (self.nx_l + self.nu_l, 1))
        A = self.parameter("A", (self.nx_l, self.nx_l))
        B = self.parameter("B", (self.nx_l, self.nu_l))
        A_c_list = [
            self.parameter(f"A_c_{i}", (self.nx_l, self.nx_l))
            for i in range(num_neighbours)
        ]

        # dictionary containing initial values for local learnable parameters
        self.learnable_pars_init = self.learnable_pars_init_local.copy()
        self.learnable_pars_init["A"] = model.A_l_innacurate
        self.learnable_pars_init["B"] = model.B_l_innacurate
        self.learnable_pars_init.update(
            {f"A_c_{i}": model.A_c_l_innacurate for i in range(num_neighbours)}
        )

        # variables (state+coupling, action, slack)
        x, x_c = self.augmented_state(num_neighbours, my_index, self.nx_l)
        u, _ = self.action(
            "u",
            self.nu_l,
            lb=self.u_bnd_l[0][0],
            ub=self.u_bnd_l[1][0],
        )
        s, _, _ = self.variable("s", (self.nx_l, N), lb=0)

        x_c_list = cs.vertsplit(
            x_c, np.arange(0, self.nx_l * num_neighbours + 1, self.nx_l)
        )  # store the bits of x that are couplings in a list for ease of access

        # dynamics - added manually due to coupling
        for k in range(N):
            coup = cs.SX.zeros(self.nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A @ x[:, [k]] + B @ u[:, [k]] + coup + b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints
        self.constraint(f"x_lb", self.x_bnd_l[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", self.x_bnd_l[1] + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + self.w_l @ s)
            )
        )

        # solver
        solver = "ipopt"
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)
