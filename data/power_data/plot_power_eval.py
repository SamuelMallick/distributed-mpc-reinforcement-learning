import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)

num_eps = 100
ep_len = 100

with open(
    "data/power_data/line_40/distributed_con_eval.pkl",
    "rb",
) as file:
    X_l = pickle.load(file)
    U_l = pickle.load(file)
    R_l = pickle.load(file)
    TD_l = pickle.load(file)
    param_list_l = pickle.load(file)

with open(
    "data/power_data/scenario/power_scenario_79.pkl",
    "rb",
) as file:
    X_s = pickle.load(file)
    U_s = pickle.load(file)
    R_s = pickle.load(file)
    TD_s = pickle.load(file)
    param_list_s = pickle.load(file)

with open(
    "data/power_data/nominal/centralised.pkl",
    "rb",
) as file:
    X_n = pickle.load(file)
    U_n = pickle.load(file)
    R_n = pickle.load(file)
    TD_n = pickle.load(file)
    param_list_n = pickle.load(file)

R_l_eps = [sum((R_l[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
R_s_eps = [sum((R_s[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
R_n_eps = [sum((R_n[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R_l_eps, "o", color="black", markersize=1.5)
axs.plot(R_s_eps, "o", color="blue", markersize=1.5)
axs.plot(R_n_eps, "o", color="green", markersize=1.5)
axs.axhline(sum(R_l_eps) / len(R_l_eps), linestyle="--", color="black", linewidth=1)
axs.axhline(sum(R_s_eps) / len(R_s_eps), linestyle="--", color="blue", linewidth=1)
axs.axhline(sum(R_n_eps) / len(R_n_eps), linestyle="--", color="green", linewidth=1)
axs.set_xlabel("episode")
axs.set_ylabel(r"$\sum L$")
axs.legend(["learned", "scenario", "nominal"])
plt.savefig("data/eval.svg", format="svg", dpi=300)
plt.show()
