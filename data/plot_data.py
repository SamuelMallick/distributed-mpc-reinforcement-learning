import pickle
import matplotlib.pyplot as plt
import numpy as np

nx = 6
nx_l = 2
x_bnd = np.array([[0, -1], [1, 1]])
a_bnd = np.array([[-1], [1]])
update_rate = 2

limit = 8500

with open("data/decentralised_diverging_example.pkl", "rb") as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    b = pickle.load(file)
    f = pickle.load(file)
    V0 = pickle.load(file)
    bounds = pickle.load(file)
    A = pickle.load(file)
    B = pickle.load(file)
    A_cs = pickle.load(file)

# plot the results
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[:limit, np.arange(0, nx, nx_l)])
axs[1].plot(X[:limit, np.arange(1, nx, nx_l)])
axs[2].plot(U[:limit])
for i in range(2):
    axs[0].axhline(x_bnd[i][0], color="r")
    axs[1].axhline(x_bnd[i][1], color="r")
    axs[2].axhline(a_bnd[i][0], color="r")
axs[0].set_ylabel("$s_1$")
axs[1].set_ylabel("$s_2$")
axs[2].set_ylabel("$a$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD[:limit], "o", markersize=1)
axs[1].semilogy(R[:limit], "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

# Plot parameters
_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
for b_i in b:
    axs[0, 0].plot(b_i[: (int(limit / update_rate))])
for bnd_i in bounds:
    axs[0, 1].plot(bnd_i[: (int(limit / update_rate))])
for f_i in f:
    axs[1, 0].plot(f_i[: (int(limit / update_rate))])
for V0_i in V0:
    axs[1, 1].plot(V0_i.squeeze()[: (int(limit / update_rate))])
for A_i in A:
    axs[2, 0].plot(A_i[: (int(limit / update_rate))])
for B_i in B:
    axs[2, 1].plot(B_i[: (int(limit / update_rate))])

axs[0, 0].set_ylabel("$b$")
axs[0, 1].set_ylabel("$x_1$")
axs[1, 0].set_ylabel("$f$")
axs[1, 1].set_ylabel("$V_0$")
axs[2, 0].set_ylabel("$A$")
axs[2, 1].set_ylabel("$B$")
plt.show()
