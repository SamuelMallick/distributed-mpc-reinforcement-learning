from copy import deepcopy
from typing import Any, Collection, Dict, Literal, Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import LstdQLearningAgent
from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.lstd_q_learning import ExpType
from mpcrl.agents.rl_learning_agent import LrType
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy, StepWiseExploration
# from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.wrappers.agents import RecordUpdates

from dmpcrl.core.admm import AdmmCoordinator
from dmpcrl.core.consensus import ConsensusCoordinator
from dmpcrl.mpc.mpc_admm import MpcAdmm


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator to handle the communication and learning of a multi-agent q-learning system."""

    admm_iters = 50
    consensus_iters = 100

    def __init__(
        self,
        mpc_cent: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        # learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],    # TODO find out why learning rate cannot import
        learning_rate,
        learnable_parameters: LearnableParametersDict[SymType],
        n: int,
        mpc_dist_list: list[MpcAdmm],
        learnable_dist_parameters_list: list[LearnableParametersDict[SymType]],
        fixed_dist_parameters_list: list,
        G: list[list[int]],
        Adj: np.ndarray,
        rho: float,
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        max_percentage_update: float = float("+inf"),
        warmstart: Literal["last", "last-successful"] = "last-successful",
        hessian_type: Literal["approx", "full"] = "approx",
        record_td_errors: bool = False,
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        centralised_flag: bool = False,
        centralised_debug: bool = False,
    ) -> None:
        """Instantiates the coordinator. If centralised_flag is true it acts as a LstdQLearningAgent."""
        # TODO parameter descriptions
        self.centralised_flag = centralised_flag
        self.centralised_debug = centralised_debug
        self.n = n

        super().__init__(  # use itself as a learning agent for error checking
            mpc_cent,
            update_strategy,
            discount_factor,
            learning_rate,
            learnable_parameters,
            fixed_parameters,
            exploration,
            experience,
            max_percentage_update,
            warmstart,
            hessian_type,
            record_td_errors,
            cho_maxiter,
            cho_solve_kwargs,
            name,
        )

        if not centralised_flag:  # act as a coordinator of learning agents
            self._updates_enabled = False  # Turn updates off for the centralised
            # configure copies of exploration with different rndom generators to avoid all agents exploring identicaly
            exploration_list = [None] * n
            if exploration is not None:
                for i in range(n):
                    new_exp = deepcopy(exploration)
                    new_exp.reset(i)  # reseting with new seed
                    exploration_list[i] = StepWiseExploration(
                        new_exp, self.admm_iters, stepwise_decay=False
                    )  # step wise to account for ADMM iters
            self.agents: list[LstdQLearningAgent] = []
            for i in range(n):
                self.agents.append(  # create agents here, passing the mpc, learnable, and fixed params from the lists
                    RecordUpdates(
                        LstdQLearningAgent(
                            mpc_dist_list[i],
                            deepcopy(update_strategy),
                            discount_factor,
                            deepcopy(learning_rate),
                            learnable_dist_parameters_list[i],
                            fixed_dist_parameters_list[i],
                            exploration_list[i],
                            deepcopy(experience),
                            max_percentage_update,
                            warmstart,
                            hessian_type,
                            record_td_errors,
                            cho_maxiter,
                            cho_solve_kwargs,  # TODO add copy
                            f"{name}_{i}",
                        )
                    )
                )

            # vars for admm procedures
            self.N = mpc_cent.horizon

            self.nx_l = mpc_dist_list[
                0
            ].nx_l  # used to seperate states into each agents
            self.nu_l = mpc_dist_list[
                0
            ].nu_l  # used to seperate control into each agents

            self.admm_coordinator = AdmmCoordinator(
                self.agents,
                G,
                N=mpc_cent.horizon,
                nx_l=self.nx_l,
                nu_l=self.nu_l,
                rho=rho,
                iters=self.admm_iters,
            )

            self.consensus_coordinator = ConsensusCoordinator(Adj, self.consensus_iters)

    # need to override train method to manually reset all agent MPCs and NOT activate updates for centralised
    def train(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        seed=None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ):
        if not self.centralised_flag:
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
        env_reset_options: Optional[Dict[str, Any]] = None,
    ):
        if self.centralised_flag:
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

        if self.centralised_flag:
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
                if self.centralised_debug:
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
                if self.centralised_debug:
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
