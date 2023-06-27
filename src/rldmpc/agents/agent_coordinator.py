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
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy

import matplotlib.pyplot as plt


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator to handle the communication and learning of a multi-agent q-learning system."""

    iters = 100  # number of iters in ADMM procedure

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
    ) -> None:
        """Instantiates the coordinator. If centralised_flag is true it acts as a LstdQLearningAgent."""
        # TODO parameter descriptions
        self.centralised_flag = centralised_flag
        self.n = n
        self.G = G
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
            self.agents: list[LstdQLearningAgent] = []
            for i in range(n):
                self.agents.append(  # create agents here, passing the mpc, learnable, and fixed params from the lists
                    LstdQLearningAgent(
                        mpc_dist_list[i],
                        deepcopy(update_strategy),
                        discount_factor,
                        deepcopy(learning_rate),
                        learnable_dist_parameters_list[i],
                        fixed_dist_parameters_list[i],
                        None,
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

            # vars for admm procedures

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

            joint_action, solV_list = self.distributed_state_value(
                state, episode, raises
            )
            for i in range(len(self.agents)):
                if not solV_list[i].success:
                    self.agents[i].on_mpc_failure(episode, None, solV[i].status, raises)

            while not (truncated or terminated):
                # compute Q(s,a)
                solQ = self.action_value(
                    state, joint_action
                )  # centralized val from joint action
                solQ_list = self.distributed_action_value(
                    state, joint_action, episode, raises
                )

                # step the system with action computed at the previous iteration
                new_state, cost, truncated, terminated, _ = env.step(joint_action)
                self.on_env_step(env, episode, timestep)  # step centralised
                for agent in self.agents:  # step distributed agents
                    agent.on_env_step(env, episode, timestep)

                # compute V(s+) and store transition
                new_action, solV = self.state_value(new_state, False)  # centralised
                new_joint_action, solV_list = self.distributed_state_value(
                    new_state, episode, raises
                )  # distributed

                if not self._try_store_experience(cost, solQ, solV):
                    self.on_mpc_failure(
                        episode, timestep, f"{solQ.status}/{solV.status}", raises
                    )

                # calculate centralised cost from local TODO make this consensus
                V_f = sum((solV_list[i].f) for i in range(len(self.agents)))
                Q_f = sum((solQ_list[i].f) for i in range(len(self.agents)))
                for i in range(len(self.agents)):  # store experience for agents
                    object.__setattr__(solV_list[i], "f", V_f)   # overwrite the local costs with the global ones TODO make this nicer
                    object.__setattr__(solQ_list[i], "f", Q_f)

                    if not self.agents[i]._try_store_experience(
                        cost, solQ_list[i], solV_list[i]
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

    def distributed_state_value(self, state, episode, raises):
        local_action_list, local_sol_list = self.admm(state, episode, raises)
        return cs.DM(local_action_list), local_sol_list

    def distributed_action_value(self, state, action, episode, raises):
        local_action_list, local_sol_list = self.admm(
            state, episode, raises, action=action
        )
        return local_sol_list

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
                    self.agents[i].on_mpc_failure(
                        episode, None, local_sol_list[i].status, raises
                    )

                idx = self.G[i].index(i) * self.nx_l
                self.x_temp_list[i] = cs.vertcat(  # insert local state into place
                    local_sol_list[i].vals["x_c"][:idx, :],
                    local_sol_list[i].vals["x"][:, :-1],
                    local_sol_list[i].vals["x_c"][idx:, :],
                )
            # plot_list.append(loc_solV[0].vals["x"])
            # z update -> essentially an averaging of all agents' optinions on each z
            # TODO -> make distributed with consensus
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

            plot_list.append(local_sol_list[1].vals["u"])

            # y update TODO parallelise
            for i in range(len(self.agents)):
                self.y_list[i] = self.y_list[i] + self.rho * (
                    self.x_temp_list[i] - self.z[self.z_slices[i], :]
                )

        plot_list = np.asarray(plot_list)
        # plt.plot(plot_list[:, :, 2])
        # plt.show()

        return loc_action_list, local_sol_list  # return last solutions

    def reset_admm_params(self):
        """Reset all vars for admm to zero."""
        for i in range(len(self.agents)):
            self.y_list[i][:] = 0
            self.x_temp_list[i][:] = 0
        self.z[:] = 0
