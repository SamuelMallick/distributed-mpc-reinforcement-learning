from typing import Any, Literal
from warnings import warn

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc
from gymnasium import Env
from gymnasium.spaces import Box
from mpcrl import LstdQLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy, StepWiseExploration
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.update import UpdateStrategy
from mpcrl.core.warmstart import WarmStartStrategy
from mpcrl.optim import GradientDescent
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer
from mpcrl.util.seeding import RngType, mk_seed

from dmpcrl.core.admm import AdmmCoordinator
from dmpcrl.core.consensus import ConsensusCoordinator


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator agent to handle the communication and learning of a multi-agent q-learning system.

    The agent maintains and organises a group of learning agents, each with their own MPC and learnable parameters.
    The value functions and policy are evaluated distrubutedly via ADMM and consensus.
    """

    def __init__(
        self,
        agents: list[LstdQLearningAgent],
        N: int,
        nx: int,
        nu: int,
        adj: np.ndarray,
        rho: float,
        admm_iters: int,
        consensus_iters: int,
        centralized_mpc: Mpc,
        centralized_learnable_parameters: None | (LearnableParametersDict) = None,
        centralized_fixed_parameters: dict[str, npt.ArrayLike] | None = None,
        centralized_update_strategy: int | UpdateStrategy | None = None,
        centralized_discount_factor: float | None = None,
        centralized_optimizer: GradientBasedOptimizer | None = None,
        centralized_exploration: ExplorationStrategy | None = None,
        centralized_experience: None | int | ExperienceReplay = None,
        warmstart: (
            Literal["last", "last-successful"] | WarmStartStrategy
        ) = "last-successful",
        hessian_type: Literal["none", "approx", "full"] = "approx",
        record_td_errors: bool = False,
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: str | None = None,
        centralized_flag: bool = False,
        centralized_debug: bool = False,
    ) -> None:
        """Initialise the coordinator agent.

        Parameters
        ----------
        agents : list[LstdQLearningAgent]
            The list of learning agents to coordinate. These agents have been initialized
            with mpcs, learnable parameters, and other learning arguments.
        N : int
            The prediction horizon of the MPCs.
        nx : int
            The state dimension of the agents. Assumed to be the same for all.
        nu : int
            The control dimension of the agents. Assumed to be the same for all.
        Adj : np.ndarray
            Adjacency matrix for the network of agents. Adj[i, j] = 1 if agent j influences
            agent i, and 0 otherwise.
        rho : float
            The penalty parameter for the ADMM algorithm.
        admm_iters : int
            The number of iterations to run the ADMM algorithm for.
        consensus_iters : int
            The number of iterations to run the consensus algorithm for.
        centralized_mpc : Mpc
            The centralized MPC used either; to check the distributed MPCs at each timestep or
            to learn a centralized controller for the whole system. If this is irrelevant, pass
            a dummpy MPC.
        centralized_learnable_parameters : LearnableParametersDict[SymType], optional
            The learnable parameters of the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_fixed_parameters : dict[str, npt.ArrayLike], optional
            The fixed parameters of the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_update_strategy : int or UpdateStrategy, optional
            The update strategy for the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_discount_factor : float, optional
            The discount factor for the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_optimizer : GradientBasedOptimizer, optional
            The optimizer for the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_exploration : ExplorationStrategy, optional
            The exploration strategy for the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        centralized_experience : int or ExperienceReplay, optional
            The exerience replay memory for the centralized MPC. By default, `None`. See LstdQLearningAgent for
            more information.
        warmstart: "last" or "last-successful" or WarmStartStrategy, optional
            The warmstart strategy for the MPC's NLP. If `last-successful`, the last
            successful solution is used to warm start the solver for the next iteration.
            If `last`, the last solution is used, regardless of success or failure.
            Furthermoer, a `WarmStartStrategy` object can be passed to specify a
            strategy for generating multiple warmstart points for the NLP. This is
            useful to generate multiple initial conditions for very non-convex problems.
            Can only be used with an MPC that has an underlying multistart NLP problem
            (see `csnlp.MultistartNlp`).
        hessian_type : {'none', 'approx', 'full'}, optional
            The type of hessian to use in this (potentially) second-order algorithm.
            If 'none', no second order information is used. If `approx`, an easier
            approximation of it is used; otherwise, the full hessian is computed but
            this is much more expensive. This option must be in accordance with the
            choice of `optimizer`, that is, if the optimizer does not use second order
            information, then this option must be set to `none`.
        record_td_errors: bool, optional
            If `True`, the TD errors are recorded in the field `td_errors`, which
            otherwise is `None`. By default, does not record them.
        use_last_action_on_fail : bool, optional
            In case the MPC solver fails
             * if `False`, the action from the last solver's iteration is returned
               anyway (though suboptimal)
             * if `True`, the action from the last successful call to the MPC is
               returned instead (if the MPC has been solved at least once successfully).
            By default, `False`.
        remove_bounds_on_initial_action : bool, optional
            When `True`, the upper and lower bounds on the initial action are removed in
            the action-value function approximator Q(s,a) since the first action is
            constrained to be equal to the initial action. This is useful to avoid
            issues in the LICQ of the NLP. However, it can lead to numerical problems.
            By default, `False`.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.

        centralized_learnable_parameters : LearnableParametersDict[SymType], optional
            The learnable parameters of the centralized MPC. By default, `None`.
        centralized_fixed_parameters : dict[str, npt.ArrayLike], optional
            The fixed parameters of the centralized MPC. By default, `None`.
        centralized_flag : bool, optional
            If `True`, the agent acts as a centralized agent, using the centralized MPC
            to learn a centralized controller for the whole system. By default, `False`.
        centralized_debug : bool, optional
            If true the centralized MPC will be used to check the distributed MPCs at each timestep.
        """
        if hessian_type != "none" and not centralized_flag:
            raise ValueError(
                "Distributed learning does not support second order information."
            )
        self.n = len(agents)
        self.agents = agents
        self.N = N
        self.centralized_flag = centralized_flag
        self.centralized_debug = centralized_debug
        flag = centralized_flag or centralized_debug

        if not all(
            isinstance(agent.exploration, StepWiseExploration) for agent in agents
        ):
            raise ValueError(
                "All agents must have a StepWiseExploration object for distributed learning."
            )

        # coordinator is itself a learning agent
        super().__init__(
            centralized_mpc,
            centralized_update_strategy if flag else 1,
            centralized_discount_factor if flag else 1.0,
            centralized_optimizer if flag else GradientDescent(1),  # dummy optimizer
            centralized_learnable_parameters if flag else LearnableParametersDict(),
            centralized_fixed_parameters if flag else None,
            centralized_exploration if flag else None,
            centralized_experience if flag else None,
            warmstart,
            hessian_type,
            record_td_errors,
            use_last_action_on_fail,
            remove_bounds_on_initial_action,
            name,
        )

        # ADMM and consensus coordinator objects
        self.admm_coordinator = AdmmCoordinator(
            self.agents,
            adj,
            N=N,
            nx_l=nx,
            nu_l=nu,
            rho=rho,
            iters=admm_iters,
        )
        self.consensus_coordinator = ConsensusCoordinator(adj, consensus_iters)

    def train(
        self,
        env: Env,
        episodes: int,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Train the system on the environment. Overiding the parent class method to handle distributed learning.
        Calls callback hooks also for distributed agents.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to train in.
        episodes : int
            Number of training episodes.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            training episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns for each training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
        if hasattr(env, "action_space"):
            assert isinstance(env.action_space, Box), "Env action space must be a Box,"
        rng = np.random.default_rng(seed)
        if not self.centralized_flag:
            for agent in self.agents:
                agent.reset(rng)
            self._updates_enabled = False
        else:
            self._updates_enabled = True
        self.reset(rng)

        self._raises = raises
        returns = np.zeros(episodes, float)

        self.on_training_start(env)
        for agent in self.agents:
            agent.on_training_start(env)

        for episode in range(episodes):
            state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)

            self.on_episode_start(env, episode, state)
            for agent in self.agents:
                agent.on_episode_start(env, episode, state)
            r = self.train_one_episode(env, episode, state, raises)

            self.on_episode_end(env, episode, r)
            for agent in self.agents:
                agent.on_episode_end(env, episode, r)
            returns[episode] = r

        self.on_training_end(env, returns)
        for agent in self.agents:
            agent.on_training_end(env, returns)
        return returns

    def evaluate(
        self,
        env: Env,
        episodes: int,
        deterministic: bool = True,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Evaluates the agent in a given environment. Overiding the parent class method to handle distributed learning.
        Calls callback hooks also for distributed agents.

        Note: after solving `V(s)` for the current state `s`, the action is computed and
        passed to the environment as the concatenation of the first optimal action
        variables of the MPC (see `csnlp.Mpc.actions`).

        Parameters
        ----------
        env : Env
            A gym environment where to test the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically; by default, `True`.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode).

        Raises
        ------
            Raises if the MPC optimization solver fails and `warns_on_exception=False`.
        """
        if self.centralized_flag:
            return super().evaluate(
                env, episodes, deterministic, seed, raises, env_reset_options
            )

        rng = np.random.default_rng(seed)
        self.reset(rng)
        for agent in self.agents:
            agent.reset(rng)
            agent.unwrapped._updates_enabled = False
        returns = np.zeros(episodes)
        self._updates_enabled = False

        self.on_validation_start(env)
        for agent in self.agents:
            agent.on_validation_start(env)

        for episode in range(episodes):
            state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)
            truncated, terminated, timestep = False, False, 0

            self.on_episode_start(env, episode, state)
            for agent in self.agents:
                agent.on_episode_start(env, episode, state)

            while not (truncated or terminated):
                joint_action, sol_list = self.distributed_state_value(
                    state, deterministic=deterministic
                )

                state, r, truncated, terminated, _ = env.step(joint_action)

                self.on_env_step(env, episode, timestep)
                for agent in self.agents:
                    agent.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1

                self.on_timestep_end(env, episode, timestep)
                for agent in self.agents:
                    agent.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])
            for agent in self.agents:
                agent.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        for agent in self.agents:
            agent.on_validation_end(env, returns)
        return returns

    def train_one_episode(
        self,
        env: Env,
        episode: int,
        init_state: np.ndarray,
        raises: bool = True,
    ) -> float:
        """Trains the agents on a single episode. Overiding the parent class method to handle distributed learning.

        Parameters
        ----------
        env : Env
            The environment to train in.
        episode : int
            The current episode number.
        init_state : np.ndarray
            The initial state of the environment.
        raises : bool, optional
            Only valid for centralized training. If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised. For distributed training
            local errors do not raise exceptions."""
        if self.centralized_flag:
            return super().train_one_episode(env, episode, init_state, raises)

        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = init_state
        action_space = getattr(env, "action_space", None)

        # get first action
        joint_action, solV_list = self.distributed_state_value(
            state, deterministic=False, action_space=action_space
        )
        if self.centralized_debug:
            action, solV = self.state_value(
                state, False, action_space=action_space
            )  # get centralised result for comparrison
            if not solV.success:
                self.on_mpc_failure(episode, None, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ_list = self.distributed_action_value(state, joint_action)
            if self.centralized_debug:
                solQ = self.action_value(state, action)
                self.validate_distributed_solution(solQ, solQ_list)

            # step the system with action computed at the previous iteration
            new_state, cost, truncated, terminated, info_dict = env.step(joint_action)
            dist_costs = info_dict.get(
                "r_dist", None
            )  # get distributed costs if returned by env

            self.on_env_step(env, episode, timestep)
            for agent in self.agents:
                agent.on_env_step(env, episode, timestep)

            # compute V(s+) and store transition
            new_joint_action, solV_list = self.distributed_state_value(
                new_state, deterministic=False
            )
            if self.centralized_debug:
                _, solV = self.state_value(new_state, False, action_space=action_space)
                if not self._try_store_experience(cost, solQ, solV):
                    self.on_mpc_failure(
                        episode, timestep, f"{solQ.status}/{solV.status}", raises
                    )

            # consensus on distributed values
            V_f_vec = np.asarray([sol.f for sol in solV_list])
            Q_f_vec = np.asarray([sol.f for sol in solQ_list])
            V_f_con = self.consensus_coordinator.average_consensus(V_f_vec)
            Q_f_con = self.consensus_coordinator.average_consensus(Q_f_vec)
            if not np.allclose(V_f_con, V_f_con[0], atol=1e-04) or not np.allclose(
                Q_f_con, Q_f_con[0], atol=1e-04
            ):
                warn(
                    f"Consensus on value functions innacurate. Max difference in V: {np.max(np.abs(V_f_con - V_f_con[0]))}, Max difference in Q: {np.max(np.abs(Q_f_con - Q_f_con[0]))}"
                )

            if dist_costs is None:  # agents get direct access to centralized cost
                cost_con = np.full((self.n,), cost)
            else:  # agents use consensus to get the centralized cost from local costs
                cost_con = self.n * self.consensus_coordinator.average_consensus(
                    np.asarray(dist_costs)
                )
                if not np.allclose(cost_con, cost_con[0], atol=1e-04):
                    warn(
                        f"Consensus on costs innacurate. Max difference: {np.max(np.abs(cost_con - cost_con[0]))}"
                    )

            for i in range(self.n):  # store local experiences
                object.__setattr__(solV_list[i], "f", V_f_con[i])
                object.__setattr__(solQ_list[i], "f", Q_f_con[i])

                if not self.agents[i].unwrapped._try_store_experience(
                    cost_con[i], solQ_list[i], solV_list[i]
                ):
                    self.agents[i].on_mpc_failure(
                        episode,
                        timestep,
                        f"{solQ_list[i].status}/{solV_list[i].status}",
                        raises,
                    )
            # increase counters
            state = new_state
            joint_action = new_joint_action
            rewards += float(cost)
            timestep += 1

            self.on_timestep_end(env, episode, timestep)
            for agent in self.agents:
                agent.on_timestep_end(env, episode, timestep)
        return rewards

    def validate_distributed_solution(
        self,
        centralized_sol: Solution,
        distributed_sols: list[Solution],
        constraint_type: Literal["dynamics"] = "dynamics",
    ) -> None:
        """Compares the centralized and distributed solutions.

        Parameters
        ----------
        centralized_sol : Solution
            The centralised solution.
        distributed_sols : list[Solution]
            The distributed solutions.
        constraint_type : {'dynamics'}, optional
            The type of constraint to validate the dual variables for. By default,
            'dynamics'.

        """
        # validate the primal variables
        dx = centralized_sol.vals["x"] - np.vstack(
            [s.vals["x"] for s in distributed_sols]
        )
        du = centralized_sol.vals["u"] - np.vstack(
            [s.vals["u"] for s in distributed_sols]
        )
        if cs.mmax(cs.fabs(dx)) > 1e-04 or cs.mmax(cs.fabs(du)) > 1e-04:
            warn(
                f"Max difference in primal variables: dx={cs.mmax(cs.fabs(dx))}, du={cs.mmax(cs.fabs(du))}."
            )
        df = centralized_sol.f - sum([s.f for s in distributed_sols])
        if cs.fabs(df) > 1e-04:
            warn(
                f"Total error of {cs.fabs(df)} in distributed objective function values."
            )
        if constraint_type == "dynamics":
            cent_duals = centralized_sol.dual_vals["lam_g_dyn"]
            # reshape to match centralised duals
            dist_duals = np.asarray(
                [
                    [
                        distributed_sols[i].dual_vals[f"lam_g_dynam_{k}"]
                        for k in range(self.N)
                    ]
                    for i in range(self.n)
                ]
            )  # shape: (n, N, nx, 1)
            dist_duals = dist_duals.transpose(0, 2, 1, 3)  # shape: (n, nx, N, 1)
            dist_duals = dist_duals.reshape((-1, self.N), order="C")  # shape: (n*nx, N)
            dist_duals = dist_duals.reshape((-1,), order="F")  # shape: (n*nx*N,)
            dynam_duals_error = np.linalg.norm(cent_duals - dist_duals)
            if dynam_duals_error > 1e-04:
                warn(
                    f"Total error of {dynam_duals_error} in distributed dual variables for dynamics."
                )
        else:
            raise ValueError(
                f"Constraint type {constraint_type} not supported for dual variable validation."
            )

    def distributed_state_value(
        self,
        state: np.ndarray,
        deterministic=False,
        action_space: Box | None = None,
    ) -> tuple[cs.DM, list[Solution]]:
        """Computes the distributed state value function using ADMM.

        Parameters
        ----------
        state : cs.DM
            The centralized state for which to compute the value function.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`."""
        (
            local_actions,
            local_sols,
            info_dict,
        ) = self.admm_coordinator.solve_admm(
            state, deterministic=deterministic, action_space=action_space
        )
        return cs.DM(local_actions), local_sols

    def distributed_action_value(
        self, state: np.ndarray, action: cs.DM
    ) -> list[Solution]:
        """Computes the distributed action value function using ADMM.

        Parameters
        ----------
        state : cs.DM
            The centralized state for which to compute the value function.
        action : cs.DM
            The centralized action for which to compute the value function.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`."""
        (_, local_sols, info_dict) = self.admm_coordinator.solve_admm(
            state, action=action, deterministic=True
        )
        return local_sols
