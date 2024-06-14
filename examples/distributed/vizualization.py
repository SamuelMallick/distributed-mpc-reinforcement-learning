import pickle

import matplotlib.pyplot as plt
import numpy as np
from model import Model

# file_name = "examples/distributed/centralized_data.pkl"
file_name = "C_false.pkl"
with open(
    file_name,
    "rb",
) as file:
    data = pickle.load(file)

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(data["TD"], "o", markersize=1)
axs[1].plot(data["R"].flatten(), "o", markersize=1)

_, axs = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=True)
X = data["X"].squeeze()
U = data["U"].squeeze()
X_l = np.split(X, 3, axis=1)
U_l = np.split(U, 3, axis=1)
x_bnd_l = Model.x_bnd_l
u_bnd_l = Model.u_bnd_l
for i in range(3):
    axs[0, i].plot(X_l[i][:, 0], markersize=1)
    axs[0, i].plot(x_bnd_l[0, 0] * np.ones_like(X_l[i][:, 0]), "--", color="r")
    axs[0, i].plot(x_bnd_l[1, 0] * np.ones_like(X_l[i][:, 0]), "--", color="r")
    axs[1, i].plot(X_l[i][:, 1], markersize=1)
    axs[1, i].plot(x_bnd_l[0, 1] * np.ones_like(X_l[i][:, 1]), "--", color="r")
    axs[1, i].plot(x_bnd_l[1, 1] * np.ones_like(X_l[i][:, 1]), "--", color="r")
    axs[2, i].plot(U_l[i], markersize=1)
    axs[2, i].plot(u_bnd_l[0, 0] * np.ones_like(U_l[i]), "--", color="r")
    axs[2, i].plot(u_bnd_l[1, 0] * np.ones_like(U_l[i]), "--", color="r")
plt.show()
