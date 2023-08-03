import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rc("text", usetex=True)

num_eps = 300
ep_len = 100
nx_l = 4
n = 4
theta_lim = 0.1
u_lim = np.array([[0.2], [0.1], [0.3], [0.1]])
P_tie = np.array(
    [
        [0, 4, 0, 0],
        [4, 0, 2, 0],
        [0, 2, 0, 2],
        [0, 0, 2, 0],
    ]
)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

with open(
    "data/power_dist_300.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    param_list = pickle.load(file)

# plot the results
TD_eps = [sum((TD[ep_len * i : ep_len * (i + 1)]))/ep_len for i in range(num_eps)]
R_eps = [sum((R[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD_eps, "o", color="black", markersize=0.8)
axs[1].plot(R_eps, "o", color="black", markersize=0.8)
axs[0].set_ylabel(r"$\overline{\delta}$")
axs[1].set_ylabel(r"$\sum L$")
axs[1].set_xlabel(r"$ep$")

# first episode
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    for j in range(nx_l):
        axs[j].plot(X[:ep_len+1, i*nx_l + j], color=colors[i])
    axs[4].plot(U[:ep_len, i], color=colors[i])
    axs[4].axhline(u_lim[i], color=colors[i], linewidth=1, linestyle='--')
    axs[4].axhline(-u_lim[i], color=colors[i], linewidth=1, linestyle='--')

axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)


# last episode
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    for j in range(nx_l):
        axs[j].plot(X[-ep_len-1:, i*nx_l + j], color=colors[i])
    axs[4].plot(U[-ep_len:, i], color=colors[i])
    axs[4].axhline(u_lim[i], color=colors[i], linewidth=1, linestyle='--')
    axs[4].axhline(-u_lim[i], color=colors[i], linewidth=1, linestyle='--')

axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)

# tie line power flows
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
if True:
    for i in range(n):
        for j in range(n):
            if P_tie[i, j] != 0:
                # first ep
                axs[0].plot(P_tie[i, j]*(X[:ep_len+1, i * nx_l] - X[:ep_len+1, j * nx_l]))
                # second ep
                axs[1].plot(P_tie[i, j]*(X[-ep_len-1:, i * nx_l] - X[-ep_len-1:, j * nx_l]))
else:
    axs[0].plot(P_tie[2, 3]*(X[:ep_len+1, 2 * nx_l] - X[:ep_len+1, 3 * nx_l]), color='black')
    axs[0].plot(P_tie[3, 2]*(X[:ep_len+1, 3 * nx_l] - X[:ep_len+1, 2 * nx_l]), color='black')
    axs[0].plot(P_tie[2, 3]*(X[-ep_len-1:, 2 * nx_l] - X[-ep_len-1:, 3 * nx_l]), color='blue')
    axs[0].plot(P_tie[3, 2]*(X[-ep_len-1:, 3 * nx_l] - X[-ep_len-1:, 2 * nx_l]), color='blue')

    axs[1].plot(P_tie[1, 2]*(X[:ep_len+1, 1 * nx_l] - X[:ep_len+1, 2 * nx_l]), color='black')
    axs[1].plot(P_tie[2, 1]*(X[:ep_len+1, 2 * nx_l] - X[:ep_len+1, 1 * nx_l]), color='black')
    axs[1].plot(P_tie[1, 2]*(X[-ep_len-1:, 1 * nx_l] - X[-ep_len-1:, 2 * nx_l]), color='blue')
    axs[1].plot(P_tie[2, 1]*(X[-ep_len-1:, 2 * nx_l] - X[-ep_len-1:, 1 * nx_l]), color='blue')

# parameters
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for param in param_list:
    pass
    #if len(param.shape) <= 2:  # TODO dont skip plotting Q
        #axs.plot(param.squeeze())

plt.show()