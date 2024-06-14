import logging
import pickle

import casadi as cs
import numpy as np
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from learnable_mpc import CentralizedMpc, LearnableMpc, LocalMpc
from model import Model
from mpcrl import LearnableParameter, LearnableParametersDict, optim
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import AdmmCoordinator

save_data = True

centralized_flag = False
prediction_horizon = 10
rho = 0.5
model = Model()
G = AdmmCoordinator.g_map(model.adj)
# centralised mpc and params
centralized_mpc = CentralizedMpc(model, prediction_horizon)
centralized_learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=centralized_mpc.parameters[name])
        for name, val in centralized_mpc.learnable_pars_init.items()
    )
)

# distributed mpc and params
distributed_mpcs: list[LocalMpc] = [
    LocalMpc(
        model=model,
        prediction_horizon=prediction_horizon,
        num_neighbours=len(G[i]) - 1,
        my_index=G[i].index(i),
        rho=rho,
    )
    for i in range(Model.n)
]
distributed_learnable_parameters: list[LearnableParametersDict] = [
    LearnableParametersDict[cs.SX](
        (
            LearnableParameter(
                name, val.shape, val, sym=distributed_mpcs[i].parameters[name]
            )
            for name, val in distributed_mpcs[i].learnable_pars_init.items()
        )
    )
    for i in range(Model.n)
]
distributed_fixed_parameters: list = [
    distributed_mpcs[i].fixed_pars_init for i in range(Model.n)
]

env = MonitorEpisodes(TimeLimit(LtiSystem(model=model), max_episode_steps=int(2e3)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgentCoordinator(
            distributed_mpcs=distributed_mpcs,
            update_strategy=2,
            discount_factor=LearnableMpc.discount_factor,
            optimizer=optim.GradientDescent(
                learning_rate=ExponentialScheduler(6e-5, factor=0.9996)
            ),
            distributed_learnable_parameters=distributed_learnable_parameters,
            N=prediction_horizon,
            nx=2,
            nu=1,
            adj=model.adj,
            rho=rho,
            admm_iters=50,
            consensus_iters=100,
            distributed_fixed_parameters=distributed_fixed_parameters,
            # exploration=None,
            exploration=EpsilonGreedyExploration(
                epsilon=ExponentialScheduler(0.7, factor=0.99),
                strength=0.1 * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
                seed=1,
            ),
            experience=ExperienceReplay(
                maxlen=100, sample_size=15, include_latest=10, seed=1
            ),
            hessian_type="none",
            record_td_errors=True,
            centralized_mpc=centralized_mpc,
            centralized_learnable_parameters=centralized_learnable_pars,
            centralized_flag=centralized_flag,
            centralized_debug=False,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

agent.train(env=env, episodes=1, seed=1, raises=False)

# extract data
# from agent
TD = (
    agent.td_errors if centralized_flag else agent.agents[0].td_errors
)  # all smaller agents have global TD error
param_dict = {}
if centralized_flag:
    for name, val in agent.updates_history.items():
        param_dict[name] = np.asarray(val)
else:
    for i in range(Model.n):
        for name, val in agent.agents[i].updates_history.items():
            param_dict[f"{name}_{i}"] = np.asarray(val)
X = np.asarray(env.observations)
U = np.asarray(env.actions)
R = np.asarray(env.rewards)

if save_data:
    with open(
        f"C_{centralized_flag}.pkl",
        "wb",
    ) as file:
        pickle.dump({"TD": TD, "param_dict": param_dict, "X": X, "U": U, "R": R}, file)
