import contextlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt

# import networkx as netx
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.util.math import quad_form
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.schedulers import ExponentialScheduler
from rldmpc.agents.agent_coordinator import LstdQLearningAgentCoordinator
from rldmpc.core.admm import g_map

Adj = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32
)  # adjacency matrix of coupling in network
G = g_map(Adj)  # mapping from global var to local var indexes for ADMM

def get_centralized_dynamics(
    n: int,
    nx_l: int,
    A_l: Union[cs.DM, cs.SX],
    B_l: Union[cs.DM, cs.SX],
    A_c: npt.NDArray[np.floating],
) -> tuple[Union[cs.DM, cs.SX], Union[cs.DM, cs.SX]]:
    """Creates the centralized representation of the dynamics from the real dynamics."""
    A = cs.SX.zeros(n * nx_l, n * nx_l)  # global state-space matrix A
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c
                A[nx_l * j : nx_l * (j + 1), nx_l * i : nx_l * (i + 1)] = A_c
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    B = cs.diagcat(*(B_l for _ in range(n)))  # global state-space matix B
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B


def get_learnable_centralized_dynamics(
    n: int,
    nx_l: int,
    nu_l: int,
    A_list: List[Union[cs.DM, cs.SX]],
    B_list: List[Union[cs.DM, cs.SX]],
    A_c_list: List[List[npt.NDArray[np.floating]]],
    B_c_list: List[List[npt.NDArray[np.floating]]],
) -> tuple[Union[cs.DM, cs.SX], Union[cs.DM, cs.SX]]:
    """Creates the centralized representation of the dynamics from the learnable dynamics."""
    A = cs.SX.zeros(n * nx_l, n * nx_l)  # global state-space matrix A
    B = cs.SX.zeros(n * nx_l, n * nu_l)  # global state-space matix B
    for i in range(n):
        for j in range(n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_list[i]
                B[nx_l * i : nx_l * (i + 1), nu_l * i : nu_l * (i + 1)] = B_list[i]
            else:
                if Adj[i, j] == 1:
                    A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c_list[
                        i
                    ][j]
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B


class LtiSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A discrete time network of LTI systems."""

    n = 3  # number of agents
    nx_l = 2  # number of agent states
    nu_l = 1  # number of agent inputs

    A_l = np.array([[0.9, 0.35], [0, 1.1]])  # agent state-space matrix A
    B_l = np.array([[0.0813], [0.2]])  # agent state-space matrix B
    A_c = np.array([[0, 0], [0, -0.1]])  # common coupling state-space matrix
    A, B = get_centralized_dynamics(n, nx_l, A_l, B_l, A_c)
    nx = n * nx_l  # number of states
    nu = n * nu_l  # number of inputs

    w = np.tile([[1e2, 1e2]], (1, n))  # agent penalty weight for bound violations
    x_bnd = np.tile([[0, -1], [1, 1]], (1, n))
    a_bnd = np.tile([[-1], [1]], (1, n))
    e_bnd = np.tile([[-1e-1], [0]], (1, n))  # uniform noise bounds

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.tile([0, 0.15], self.n).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: npt.NDArray[np.floating], action: float) -> float:
        """Computes the stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * np.square(action).sum()
            + self.w @ np.maximum(0, lb[:, np.newaxis] - state)
            + self.w @ np.maximum(0, state - ub[:, np.newaxis])
        )

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""
        action = action.full()
        x_new = self.A @ self.x + self.B @ action

        noise = self.np_random.uniform(*self.e_bnd).reshape(-1, 1)
        x_new[np.arange(0, self.nx, self.nx_l)] += noise

        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = 10
    discount_factor = 0.9

    A_l_init = np.asarray([[1, 0.25], [0, 1]])
    B_l_init = np.asarray([[0.0312], [0.25]])
    A_c_l_init = np.array([[0, 0], [0, 0]])
    B_c_l_init = np.array([[0], [0]])
    A_init, B_init = get_centralized_dynamics(
        LtiSystem.n, LtiSystem.nx_l, A_l_init, B_l_init, A_c_l_init
    )

    learnable_pars_init = {
        "V0": np.zeros((LtiSystem.n, 1)),
        "x_lb": np.tile([0, 0], LtiSystem.n).reshape(-1, 1),
        "x_ub": np.tile([1, 0], LtiSystem.n).reshape(-1, 1),
        "b": np.zeros(LtiSystem.nx),
        "f": np.zeros(LtiSystem.nx + LtiSystem.nu),
    }

    # add the initial guesses of the learable parameters

    for i in range(LtiSystem.n):
        learnable_pars_init["A_" + str(i)] = A_l_init
        learnable_pars_init["B_" + str(i)] = B_l_init
        for j in range(LtiSystem.n):
            if i != j:
                if Adj[i][j] == 1:
                    learnable_pars_init["A_c_" + str(i) + "_" + str(j)] = A_c_l_init

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = LtiSystem.w
        nx, nu = LtiSystem.nx, LtiSystem.nu
        x_bnd, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        V0 = self.parameter("V0", (LtiSystem.n,))
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))

        # learn parameters with topology knowledge

        A_list = []
        B_list = []
        A_c_list: List[list] = []
        B_c_list: List[list] = []
        for i in range(LtiSystem.n):
            A_list.append(
                self.parameter("A_" + str(i), (LtiSystem.nx_l, LtiSystem.nx_l))
            )
            B_list.append(
                self.parameter("B_" + str(i), (LtiSystem.nx_l, LtiSystem.nu_l))
            )

            A_c_list.append([])
            B_c_list.append([])
            for j in range(LtiSystem.n):
                A_c_list[i].append(None)
                B_c_list[i].append(None)
                if i != j:
                    if Adj[i][j] == 1:
                        A_c_list[i][j] = self.parameter(
                            "A_c_" + str(i) + "_" + str(j),
                            (LtiSystem.nx_l, LtiSystem.nx_l),
                        )

        A, B = get_learnable_centralized_dynamics(
            LtiSystem.n,
            LtiSystem.nx_l,
            LtiSystem.nu_l,
            A_list,
            B_list,
            A_c_list,
            B_c_list,
        )

        # variables (state, action, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action(
            "u", nu, lb=a_bnd[0].reshape(-1, 1), ub=a_bnd[1].reshape(-1, 1)
        )
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + b, n_in=2, n_out=1)

        # other constraints
        self.constraint("x_lb", x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1].reshape(-1, 1) + x_ub + s)

        # objective
        S = cs.DM(
            dlqr(self.A_init, self.B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1]
        )
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            cs.sum1(V0)
            # + quad_form(S, x[:, -1])   # TODO I took this out for now cause we cant do it distributed
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class MPCAdmm(Mpc[cs.SX]):
    """MPC for agent inner prob in ADMM."""

    rho = 0.5

    horizon = 10
    discount_factor = 0.9

    A_init = np.asarray([[1, 0.25], [0, 1]])
    B_init = np.asarray([[0.0312], [0.25]])
    A_c_l_init = np.array([[0, 0], [0, 0]])

    # learnable pars no related to coupling

    def __init__(self, num_neighbours, my_index) -> None:
        """Instantiate inner MPC for admm. My index is used to pick out own state from the grouped coupling states. It should be passed in via the mapping G (G[i].index(i))"""
        N = self.horizon
        gamma = self.discount_factor
        w_full = LtiSystem.w
        nx_l, nu_l = LtiSystem.nx_l, LtiSystem.nu_l
        w = w_full[:, 0:nx_l].reshape(-1, 1)
        x_bnd_full, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        x_bnd = (
            x_bnd_full[0, 0:nx_l].reshape(-1, 1),
            x_bnd_full[1, 0:nx_l].reshape(-1, 1),
        )  # here assumed bounds are homogenous
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        #learnable pars

        self.learnable_pars_init = {
        "V0": np.zeros((1, 1)),
        "x_lb": np.array([0, 0]).reshape(-1, 1),
        "x_ub": np.array([1, 0]).reshape(-1, 1),
        "b": np.zeros(LtiSystem.nx_l),
        "f": np.zeros(LtiSystem.nx_l + LtiSystem.nu_l),
        "A": np.zeros((LtiSystem.nx_l, LtiSystem.nx_l)),
        "B": np.zeros((LtiSystem.nx_l, LtiSystem.nu_l)),
        }

        self.num_neighbours = num_neighbours
        for i in range(
            num_neighbours
        ):  # add a learnable param init for each neighbours coupling matrix
            self.learnable_pars_init["A_c_" + str(i)] = self.A_c_l_init

        # fixed pars

        self.fixed_pars_init = {
            "y": np.zeros(
                (nx_l * (num_neighbours + 1), N)
            ),  # lagrange multipliers for ADMM
            "z": np.zeros((nx_l * (num_neighbours + 1), N)),  # global consensus vars
        }

        # parameters

        V0 = self.parameter("V0", (1,))
        x_lb = self.parameter("x_lb", (nx_l,))
        x_ub = self.parameter("x_ub", (nx_l,))
        b = self.parameter("b", (nx_l, 1))
        f = self.parameter("f", (nx_l + nu_l, 1))
        A = self.parameter("A", (nx_l, nx_l))
        B = self.parameter("B", (nx_l, nu_l))
        A_c_list: list[np.ndarray] = []  # list of coupling matrices
        for i in range(num_neighbours):
            A_c_list.append(self.parameter("A_c_" + str(i), (nx_l, nx_l)))
        y = self.parameter("y", (nx_l * (num_neighbours + 1), N))
        z = self.parameter("z", (nx_l * (num_neighbours + 1), N))

        # variables (state+coupling, action, slack)

        my_slice = slice(
            nx_l * my_index, nx_l * (my_index + 1)
        )  # used to slice the local state out of x

        x, _, _ = self.variable(
            "x", (nx_l * (num_neighbours + 1), N + 1)
        )  # local state and coupling together
        # TODO here assumed action bounds are homogenous
        u, _ = self.action(
            "u",
            nu_l,
            lb=a_bnd[0][0],
            ub=a_bnd[1][0],
        )
        s, _, _ = self.variable("s", (nx_l, N), lb=0)

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours + 1):
            if i != my_index:
                x_c_list.append(x[nx_l * i : nx_l * (i + 1), :])

        # dynamics - added manually due to coupling

        for k in range(N):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                "dynam_" + str(k),
                x[my_slice, [k + 1]],
                "==",
                A @ x[my_slice, [k]] + B @ u[:, [k]] + coup + b,
            )

        # other constraints

        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[my_slice, 1:])
        self.constraint("x_ub", x[my_slice, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[my_slice, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sumsqr(x[my_slice, :-1]) + 0.5 * cs.sumsqr(u) + w.T @ s)
            )
            + cs.trace(y.T @ (x[:, :-1] - z))  # TODO is this doing the right thing?
            # + sum((y[:, k].T @ (x[:, k] - z[:, k])) for k in range(N))  # TODO: check the same as above
            + (self.rho / 2) * cs.sumsqr(x[:, :-1] - z)
        )

        # solver

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


# now, let's create the instances of such classes and start the training
# centralised mpc and params
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)

# distributed mpc and params
mpc_dist_list: list[Mpc] = []
learnable_dist_parameters_list: list[LearnableParametersDict] = []
fixed_dist_parameters_list: list = []
for i in range(LtiSystem.n):
    mpc_dist_list.append(MPCAdmm(num_neighbours=len(G[i])-1, my_index=G[i].index(i)))
    learnable_dist_parameters_list.append(
        LearnableParametersDict[cs.SX](
            (
                LearnableParameter(name, val.shape, val, sym=mpc_dist_list[i].parameters[name])
                for name, val in mpc_dist_list[i].learnable_pars_init.items()
            )
        )
    )
    fixed_dist_parameters_list.append(
        mpc_dist_list[i].fixed_pars_init
    )



env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=int(5e1)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgentCoordinator(
            n=LtiSystem.n,
            centralised_flag=False,
            mpc_cent=mpc,
            learnable_parameters=learnable_pars,
            mpc_dist_list=mpc_dist_list,
            learnable_dist_parameters_list = learnable_dist_parameters_list,
            fixed_dist_parameters_list=fixed_dist_parameters_list,
            discount_factor=mpc.discount_factor,
            update_strategy=50,
            learning_rate=ExponentialScheduler(4e-2, factor=0.99),
            hessian_type="approx",
            record_td_errors=True,
            exploration=EpsilonGreedyExploration(
                epsilon=ExponentialScheduler(0.5, factor=0.9),
                strength=0.2 * (LtiSystem.a_bnd[1, 0] - LtiSystem.a_bnd[0, 0]),
            ),
            experience=ExperienceReplay(
                maxlen=200, sample_size=0.5, include_latest=0.1, seed=0
            ),
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 200},
)

agent.train(env=env, episodes=1, seed=69)


# plot the results
X = env.observations[0].squeeze()
U = env.actions[0].squeeze()
R = env.rewards[0]
time = np.arange(R.size)
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(time, X[:-1, np.arange(0, env.nx, env.nx_l)])
axs[1].plot(time, X[:-1, np.arange(1, env.nx, env.nx_l)])
axs[2].plot(time, U)
for i in range(2):
    axs[0].axhline(env.x_bnd[i][0], color="r")
    axs[1].axhline(env.x_bnd[i][1], color="r")
    axs[2].axhline(env.a_bnd[i][0], color="r")
axs[0].set_ylabel("$s_1$")
axs[1].set_ylabel("$s_2$")
axs[2].set_ylabel("$a$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(agent.td_errors, "o", markersize=1)
axs[1].semilogy(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
updates = np.arange(len(agent.updates_history["b"]))
axs[0, 0].plot(updates, np.asarray(agent.updates_history["b"]))
axs[0, 1].plot(
    updates,
    np.concatenate(
        [
            np.squeeze(agent.updates_history[n])[:, np.arange(0, env.nx, env.nx_l)]
            for n in ("x_lb", "x_ub")
        ],
        -1,
    ),
)
axs[1, 0].plot(updates, np.asarray(agent.updates_history["f"]))
axs[1, 1].plot(updates, np.squeeze(agent.updates_history["V0"]))
axs[2, 0].plot(
    updates, np.asarray(agent.updates_history["A_0"]).reshape(updates.size, -1)
)
axs[2, 0].plot(
    updates, np.asarray(agent.updates_history["A_c_0_1"]).reshape(updates.size, -1)
)
axs[2, 1].plot(
    updates,
    np.asarray(agent.updates_history["B_0"]).reshape(updates.size, -1),
)
axs[0, 0].set_ylabel("$b$")
axs[0, 1].set_ylabel("$x_1$")
axs[1, 0].set_ylabel("$f$")
axs[1, 1].set_ylabel("$V_0$")
axs[2, 0].set_ylabel("$A$")
axs[2, 1].set_ylabel("$B$")

plt.show()
