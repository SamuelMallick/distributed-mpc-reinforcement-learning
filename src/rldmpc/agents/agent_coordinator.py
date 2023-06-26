from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from typing import Any, Collection, Dict, Literal, Optional, Union
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import LstdQLearningAgent
from mpcrl.agents.agent import ActType, ObsType
import numpy.typing as npt
from mpcrl.agents.lstd_q_learning import ExpType
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy


class LstdQLearningAgentCoordinator(LstdQLearningAgent):
    """Coordinator to handle the communication and learning of a multi-agent q-learning system."""

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
                self.agents.append( # create agents here, passing the mpc, learnable, and fixed params from the lists
                    LstdQLearningAgent(
                        mpc_dist_list[i],
                        update_strategy,
                        discount_factor,
                        learning_rate,
                        learnable_dist_parameters_list[i],
                        fixed_dist_parameters_list[i],
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
                )

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
            action, solV = self.state_value(state, False)   # get centralised result for comparrison
            
            return super().train_one_episode(env, episode, init_state, raises)

    def distributed_state_value(state):
        pass