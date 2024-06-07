from collections.abc import Collection
from copy import deepcopy
from typing import Any, Literal
from mpcrl.core.warmstart import WarmStartStrategy
import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import LstdQLearningAgent
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer
from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.lstd_q_learning import ExpType
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy, StepWiseExploration
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.update import UpdateStrategy
from mpcrl.wrappers.agents import RecordUpdates

from dmpcrl.core.admm import AdmmCoordinator, g_map
from dmpcrl.core.consensus import ConsensusCoordinator
from dmpcrl.mpc.mpc_admm import MpcAdmm
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Literal,
    Optional,
    SupportsFloat,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from scipy.linalg import cho_solve
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.util.math import cholesky_added_multiple_identities


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator agent to handle the communication and learning of a multi-agent q-learning system.

    The agent maintains and organises a group of learning agents, each with their own MPC and learnable parameters.
    The value functions and policy are evaluated distrubutedly via ADMM and consensus.
    """

    def __init__(
        self,
        distributed_mpcs: list[MpcAdmm],
        update_strategy: int | UpdateStrategy,
        discount_factor: float,
        optimizer: GradientBasedOptimizer,
        distributed_learnable_parameters: list[LearnableParametersDict[SymType]],
        N: int,
        nx: int,
        nu: int,
        Adj: np.ndarray,
        rho: float,
        admm_iters: int,
        consensus_iters: int,
        distributed_fixed_parameters: (
            None
            | list[dict[str, npt.ArrayLike]]
            | list[Collection[dict[str, npt.ArrayLike]]]
        ) = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Union[None, int, ExperienceReplay] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        hessian_type: Literal["none", "approx", "full"] = "approx",
        record_td_errors: bool = False,
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: Optional[str] = None,
        centralized_mpc: Optional[Mpc] = None,
        centralized_learnable_parameters: Optional[
            LearnableParametersDict[SymType]
        ] = None,
        centralized_fixed_parameters: Optional[dict[str, npt.ArrayLike]] = None,
        centralized_flag: bool = False,
        centralized_debug: bool = False,
    ) -> None:
        """Initialise the coordinator agent.

        Parameters
        ----------
        distributed_mpcs : list[MpcAdmm]
            List of MPCs for each agent.
        update_strategy : UpdateStrategy or int
            The strategy used to decide which frequency to update the mpc parameters
            with. If an `int` is passed, then the default strategy that updates every
            `n` env's steps is used (where `n` is the argument passed); otherwise, an
            instance of `UpdateStrategy` can be passed to specify these in more details.
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1].
        optimizer : GradientBasedOptimizer
            A gradient-based optimizer (e.g., `mpcrl.optim.GradientDescent`) to compute
            the updates of the learnable parameters, based on the current gradient-based
            RL algorithm.
        distributed_learnable_parameters : list[LearnableParametersDict[SymType]]
            A list of dicts containing the learnable parameters of the MPCs, together with
            their bounds and values. This dict is complementary with `fixed_parameters`,
            which contains the MPC parameters that are not learnt by the agents.
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
        distributed_fixed_parameters : None | list[dict[str, npt.ArrayLike]] | list[Collection[dict[str, npt.ArrayLike]]]
            A list of dicts (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the online MPC policy. By
            default `None`, in which case `NoExploration` is used. Should not be set
            when offpolicy learning, as the exploration should be taken care in the
            offpolicy data generation.
        experience : int or ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory with length 1 is created, i.e., it keeps only the latest memory
            transition.  If an integer `n` is passed, then a memory with the length `n`
            is created and with sample size `n`.
            In the case of LSTD Q-learning, each memory item consists of the action
            value function's gradient and hessian computed at each (succesful) env's
            step.
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
        centralized_mpc : Mpc, optional
            A centralized MPC to be used for the centralized agent, if the centralized learning is
            conducted in place of distributed learning. By default, `None`.
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
        self.n = len(distributed_mpcs)
        self.centralized_flag = centralized_flag
        self.centralized_debug = centralized_debug

        # coordinator is itself a learning agent
        super().__init__(
            centralized_mpc
            if centralized_flag
            else deepcopy(
                distributed_mpcs[0]
            ),  # if not centralized, use the first agent's mpc to satisfy the parent class
            update_strategy,
            discount_factor,
            optimizer,
            centralized_learnable_parameters
            if centralized_flag
            else deepcopy(
                distributed_learnable_parameters[0]
            ),  # if not centralized, use the first agent's learnable params to satisfy the parent class
            centralized_fixed_parameters
            if centralized_flag
            else deepcopy(
                distributed_fixed_parameters[0]
            ),  # if not centralized, use the first agent's fixed params to satisfy the parent class
            exploration,
            experience,
            warmstart,
            hessian_type,
            record_td_errors,
            use_last_action_on_fail,
            remove_bounds_on_initial_action,
            name,
        )

        if (
            not centralized_flag
        ):  # coordinates the distributed learning, rather than doing the learning itself
            self._updates_enabled = False  # turn off updates for the coordinator
            exploration_list = [None] * self.n
            if exploration is not None:
                for i in range(self.n):
                    new_exp = deepcopy(exploration).reset(
                        seed=i
                    )  # copying and reseting with new seed, avoiding identical exploration between agents
                    exploration_list[
                        i
                    ] = StepWiseExploration(  # convert to stepwise exploration such that exploration is not changed within ADMM iterations
                        new_exp, admm_iters, stepwise_decay=False
                    )
            self.agents = [
                RecordUpdates(
                    LstdQLearningAgent(
                        distributed_mpcs[i],
                        deepcopy(update_strategy),
                        discount_factor,
                        deepcopy(optimizer),
                        distributed_learnable_parameters[i],
                        distributed_fixed_parameters[i],
                        exploration_list[i],
                        deepcopy(experience),
                        warmstart,
                        hessian_type,
                        record_td_errors,
                        use_last_action_on_fail,
                        remove_bounds_on_initial_action,
                        f"{name}_{i}",
                    )
                )
                for i in range(self.n)
            ]
            # ADMM and consensus coordinator objects
            G = g_map(Adj)
            self.admm_coordinator = AdmmCoordinator(
                self.agents,
                G,
                N=N,
                nx_l=nx,
                nu_l=nu,
                rho=rho,
                iters=admm_iters,
            )
            self.consensus_coordinator = ConsensusCoordinator(Adj, consensus_iters)

    # need to override train method to manually reset all agent MPCs and NOT activate updates for centralised
    def train(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        seed=None,
        raises: bool = True,
        env_reset_options: dict[str, Any] | None = None,
    ):
        if not self.centralized_flag:
            for agent in self.agents:
                agent.reset(seed)
            self._updates_enabled = False
        else:
            self._updates_enabled = True

        self._raises = raises
        returns = np.zeros(episodes, float)
        self.on_training_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            self.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            self.on_episode_start(env, episode)
            returns[episode] = self.train_one_episode(env, episode, state, raises)
            self.on_episode_end(env, episode, returns[episode])

        self.on_training_end(env, returns)
        return returns

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed=None,
        raises: bool = True,
        env_reset_options: dict[str, Any] | None = None,
    ):
        if self.centralized_flag:
            return super().evaluate(env, episodes, seed, raises, env_reset_options)

        for agent in self.agents:
            agent.reset(seed)

        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            self.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode)

            while not (truncated or terminated):
                joint_action, sol_list, error_flag = self.distributed_state_value(
                    state, episode, raises, deterministic=deterministic
                )
                if error_flag:
                    return returns

                state, r, truncated, terminated, _ = env.step(joint_action)
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns

    def train_one_episode(
        self,
        env: Env[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
    ) -> float:
        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = init_state

        if self.centralized_flag:
            return super().train_one_episode(env, episode, init_state, raises)
        else:
            # solve for the first action
            action, solV = self.state_value(
                state, False
            )  # get centralised result for comparrison
            if not solV.success:
                self.on_mpc_failure(episode, None, solV.status, raises)

            joint_action, solV_list, error_flag = self.distributed_state_value(
                state, episode, raises, deterministic=False
            )

            iter_count = 0
            while not (truncated or terminated):
                iter_count += 1
                print(iter_count)
                # compute Q(s,a)
                solQ_list, error_flag = self.distributed_action_value(
                    state, joint_action, episode, raises
                )
                if error_flag:
                    # self.on_training_end(env, rewards)
                    return rewards

                # compare dual vars of centralised and distributed sols
                # dynamics
                if self.centralized_debug:
                    solQ = self.action_value(
                        state, joint_action
                    )  # centralized val from joint action
                    cent_duals_dyn = solQ.dual_vals["lam_g_dyn"]
                    dist_duals_list_dyn: list = []
                    cent_duals_list_dyn: list = []
                    for i in range(len(self.agents)):
                        dist_duals_list_dyn.append(np.zeros((self.nx_l, self.N)))
                        cent_duals_list_dyn.append(np.zeros((self.nx_l, self.N)))
                        for k in range(self.N):  # TODO get rid of hard coded horizon
                            dist_duals_list_dyn[i][:, [k]] = np.array(
                                solQ_list[i].dual_vals[f"lam_g_dynam_{k}"]
                            )
                            cent_duals_list_dyn[i][:, [k]] = np.array(
                                cent_duals_dyn[
                                    (
                                        self.nx_l * len(self.agents) * k + self.nx_l * i
                                    ) : (
                                        self.nx_l * len(self.agents) * k
                                        + self.nx_l * (i + 1)
                                    )
                                ]
                            )
                    dynam_duals_error = np.linalg.norm(
                        cent_duals_list_dyn[0] - dist_duals_list_dyn[0]
                    )
                    if dynam_duals_error > 1e-04:
                        # exit('Duals of dynamics werent accurate!')
                        print("Duals of dynamics werent accurate!")
                # step the system with action computed at the previous iteration
                new_state, cost, truncated, terminated, info_dict = env.step(
                    joint_action
                )
                if (
                    "r_dist" in info_dict.keys()
                ):  # get distributed costs from env dict if its there
                    dist_costs = info_dict["r_dist"]
                else:
                    dist_costs = None

                self.on_env_step(env, episode, timestep)  # step centralised
                for agent in self.agents:  # step distributed agents
                    agent.on_env_step(env, episode, timestep)

                # compute V(s+) and store transition
                if self.centralized_debug:
                    new_action, solV = self.state_value(new_state, False)  # centralised
                    if not self._try_store_experience(cost, solQ, solV):
                        self.on_mpc_failure(
                            episode, timestep, f"{solQ.status}/{solV.status}", raises
                        )

                new_joint_action, solV_list, error_flag = self.distributed_state_value(
                    new_state, episode, raises, deterministic=False
                )  # distributed
                if error_flag:
                    return rewards

                # calculate centralised costs from locals with consensus
                # V_f = sum((solV_list[i].f) for i in range(len(self.agents)))
                # Q_f = sum((solQ_list[i].f) for i in range(len(self.agents)))
                V_f_vec = np.array([solV_list[i].f for i in range(len(self.agents))])
                Q_f_vec = np.array([solQ_list[i].f for i in range(len(self.agents))])
                V_f = self.consensus_coordinator.average_consensus(V_f_vec)[0] * len(
                    self.agents
                )
                Q_f = self.consensus_coordinator.average_consensus(Q_f_vec)[0] * len(
                    self.agents
                )
                if dist_costs is not None:
                    av_cost = self.consensus_coordinator.average_consensus(
                        np.asarray(dist_costs)
                    )
                    cost_f = av_cost[0] * len(self.agents)
                else:
                    cost_f = cost
                for i in range(len(self.agents)):  # store experience for agents
                    object.__setattr__(
                        solV_list[i], "f", V_f
                    )  # overwrite the local costs with the global ones TODO make this nicer
                    object.__setattr__(solQ_list[i], "f", Q_f)

                    if not self.agents[i].unwrapped._try_store_experience(
                        cost_f, solQ_list[i], solV_list[i]
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

    def distributed_state_value(self, state, episode, raises, deterministic):
        # local_action_list, local_sol_list, error_flag = self.admm(
        #    state, episode, raises, deterministic=deterministic
        # )
        (
            local_action_list,
            local_sol_list,
            error_flag,
        ) = self.admm_coordinator.solve_admm(state, deterministic=deterministic)
        if not error_flag:
            return cs.DM(local_action_list), local_sol_list, error_flag
        else:
            return None, local_sol_list, error_flag

    def distributed_action_value(self, state, action, episode, raises):
        (
            local_action_list,
            local_sol_list,
            error_flag,
        ) = self.admm_coordinator.solve_admm(state, action=action)
        return local_sol_list, error_flag
