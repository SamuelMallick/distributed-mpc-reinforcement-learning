from copy import deepcopy
from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from typing import Any, Collection, Dict, Literal, Optional, Union
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import LstdQLearningAgent
from mpcrl.agents.agent import ActType, ObsType
import numpy.typing as npt
import numpy as np
import casadi as cs
from mpcrl.agents.lstd_q_learning import ExpType
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy, StepWiseExploration
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler

import matplotlib.pyplot as plt


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator to handle the communication and learning of a multi-agent q-learning system."""

    iters = 50  # number of iters in ADMM procedure

    def __init__(
        self,
        mpc_cent: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        learnable_parameters: LearnableParametersDict[SymType],
        n: int,
        mpc_dist_list: list[Mpc[SymType]],
        learnable_dist_parameters_list: list[LearnableParametersDict[SymType]],
        fixed_dist_parameters_list: list,
        G: list[list[int]],
        P: np.ndarray,
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
        self.G = G
        self.P = P
        self.rho = rho


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
            # configure copies of exploration with different rndom generators to avoid all agents exploring identicaly
            exploration_list = [None]*n
            if exploration is not None:
                for i in range(n):
                    new_exp = deepcopy(exploration)
                    #new_exp.reset(i)    # reseting with new seed
                    exploration_list[i] = StepWiseExploration(new_exp, self.iters)  # step wise to account for ADMM iters
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
            self.y_list: list[np.ndarray] = []  # dual vars
            self.x_temp_list: list[
                np.ndarray
            ] = []  # intermediate numerical values for x
            dual_temp_list: list[np.ndarray] = []  # dual vals of from inner probs
            for i in range(n):
                x_dim = mpc_dist_list[i].x_dim
                self.y_list.append(np.zeros((x_dim[0], x_dim[1])))
                self.x_temp_list.append(np.zeros(x_dim))
            self.z = np.zeros(
                (n * self.nx_l, x_dim[1])
            )  # all states of all agents stacked, along horizon

            # generate slices of z for each agent
            z_slices: list[list[int]] = []
            for i in range(n):
                z_slices.append([])
                for j in self.G[i]:
                    z_slices[i] += list(np.arange(self.nx_l * j, self.nx_l * (j + 1)))
            self.z_slices = z_slices

    # need to override train method to manually reset all agent MPCs
    def train(self, env, episodes, seed):
        if not self.centralised_flag:
            for agent in self.agents:
                agent.reset(seed)
        super().train(env, episodes, seed)

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed=None,
        raises: bool = True,
        env_reset_options: Optional[Dict[str, Any]] = None,
    ):
        if not self.centralised_flag:
            for agent in self.agents:
                agent.reset(seed)
        return super().evaluate(env, episodes, seed, raises, env_reset_options)

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
                state, episode, raises
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
                if "r_dist" in info_dict.keys(): # get distributed costs from env dict if its there
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
                    new_state, episode, raises
                )  # distributed
                if error_flag:
                    return rewards

                # calculate centralised costs from locals with consensus
                # V_f = sum((solV_list[i].f) for i in range(len(self.agents)))
                # Q_f = sum((solQ_list[i].f) for i in range(len(self.agents)))
                V_f_vec = np.array([solV_list[i].f for i in range(len(self.agents))])
                Q_f_vec = np.array([solQ_list[i].f for i in range(len(self.agents))])
                V_f = self.consensus(V_f_vec)[0] * len(self.agents)
                Q_f = self.consensus(Q_f_vec)[0] * len(self.agents)
                if dist_costs is not None:
                    cost_f = self.consensus(np.asarray(dist_costs))[0] * len(self.agents)
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
                if self.centralised_debug:
                    self.on_timestep_end(env, episode, timestep)
                for agent in self.agents:
                    agent.on_timestep_end(env, episode, timestep)
            return rewards

    def distributed_state_value(self, state, episode, raises):
        local_action_list, local_sol_list, error_flag = self.admm(
            state, episode, raises
        )
        if not error_flag:
            return cs.DM(local_action_list), local_sol_list, error_flag
        else:
            return None, local_sol_list, error_flag

    def distributed_action_value(self, state, action, episode, raises):
        local_action_list, local_sol_list, error_flag = self.admm(
            state, episode, raises, action=action
        )
        return local_sol_list, error_flag

    def admm(self, state, episode, raises, action=None):
        """Uses ADMM to solve distributed problem. If actions passed, solves state-action val."""
        loc_action_list = [None] * len(self.agents)
        local_sol_list = [None] * len(self.agents)

        plot_list = []

        for iter in range(self.iters):
            for i in range(len(self.agents)):  # x-update TODO parallelise
                loc_state = state[
                    self.nx_l * i : self.nx_l * (i + 1), :
                ]  # get local state from global

                # set fixed vars
                self.agents[i].fixed_parameters["y"] = self.y_list[i]
                self.agents[i].fixed_parameters["z"] = self.z[
                    self.z_slices[i], :
                ]  # only pass global vars that are relevant to agent

                if action is None:
                    loc_action_list[i], local_sol_list[i] = self.agents[i].state_value(
                        loc_state, False
                    )
                else:
                    loc_action = action[self.nu_l * i : self.nu_l * (i + 1), :]
                    local_sol_list[i] = self.agents[i].action_value(
                        loc_state, loc_action
                    )
                if not local_sol_list[i].success:
                    # self.agents[i].on_mpc_failure(
                    #    episode, None, local_sol_list[i].status, raises
                    # )
                    return (
                        loc_action_list,
                        local_sol_list,
                        True,
                    )  # return with error flag as true

                idx = self.G[i].index(i) * self.nx_l
                self.x_temp_list[i] = cs.vertcat(  # insert local state into place
                    local_sol_list[i].vals["x_c"][:idx, :],
                    local_sol_list[i].vals["x"][:, :-1],
                    local_sol_list[i].vals["x_c"][idx:, :],
                )
            # plot_list.append(loc_solV[0].vals["x"])
            # z update -> essentially an averaging of all agents' optinions on each z
            for i in range(len(self.agents)):  # loop through each agents associated z
                count = 0
                sum = np.zeros((self.nx_l, self.z.shape[1]))
                for j in range(
                    len(self.agents)
                ):  # loop through agents who have opinion on this z
                    if i in self.G[j]:
                        count += 1
                        x_slice = slice(
                            self.nx_l * self.G[j].index(i),
                            self.nx_l * (self.G[j].index(i) + 1),
                        )
                        sum += self.x_temp_list[j][x_slice, :]
                self.z[self.nx_l * i : self.nx_l * (i + 1), :] = sum / count

            plot_list.append(local_sol_list[2].vals["u"])
            #plot_list.append(self.x_temp_list[2] - self.z[self.z_slices[2], :])
            #plot_list.append(self.x_temp_list[0])
            #plot_list.append(self.z[self.z_slices[0], :])

            # y update TODO parallelise
            for i in range(len(self.agents)):
                self.y_list[i] = self.y_list[i] + self.rho * (
                    self.x_temp_list[i] - self.z[self.z_slices[i], :]
                )

        #plot_list = np.asarray(plot_list)
        #plt.plot(plot_list[:, :, 0])
        #plt.show()

        return (
            loc_action_list,
            local_sol_list,
            False,
        )  # return last solutions with error flag as false

    def consensus(self, x):
        """Runs the average consensus algorithm on the vector x"""
        iters = 200  # number of consensus iters
        for iter in range(iters):
            x = self.P @ x
        return x

    def reset_admm_params(self):
        """Reset all vars for admm to zero."""
        for i in range(len(self.agents)):
            self.y_list[i][:] = 0
            self.x_temp_list[i][:] = 0
        self.z[:] = 0
