from typing import List
import numpy as np
from itertools import product
from scipy.linalg import block_diag


def cent_from_dist(d_systems: List[dict], Adj: np.ndarray):
    """Creates a centralised representation of the distributed PWA system.
    PWA dynamics represented as: x+ = A[i]x + B[i]u + c[i] if S[i]x + R[i]u <= T.
    With state and control constraints Dx <= E, Fu <= G.

    Parameters
    ----------
    d_systems: List[dict]
        List of systems, each for an agent. Systems of form {S, R, T, A, B, c, D, E, F, G, [Aj]}.
    Adj: np.ndarray
        Adjacency matrix for system.
    """

    n = len(d_systems)  # num sub-systems
    nx_l = d_systems[0]["A"][0].shape[0]  # local x dim
    s_l = len(d_systems[0]["S"])  # num switching regions for each sub
    s = s_l**n  # num switching regions for centralised sys

    # each entry of the list contains a possible permutation of the PWA region indexes
    lst = [list(i) for i in product([j for j in range(s_l)], repeat=n)]

    S = []
    R = []
    T = []
    A = []
    B = []
    c = []

    # loop through every permutation
    for idxs in lst:
        S_tmp = []
        R_tmp = []
        T_tmp = []
        A_tmp = []
        B_tmp = []
        c_tmp = []
        for i in range(n):  # add the relevant region for each agent
            S_tmp.append(d_systems[i]["S"][idxs[i]])
            R_tmp.append(d_systems[i]["R"][idxs[i]])
            T_tmp.append(d_systems[i]["T"][idxs[i]])
            A_tmp.append(d_systems[i]["A"][idxs[i]])
            B_tmp.append(d_systems[i]["B"][idxs[i]])
            c_tmp.append(d_systems[i]["c"][idxs[i]])

        A_diag = block_diag(*A_tmp)
        # add coupling to A
        for i in range(n):
            coupling_idx = 0  # keep track of which Ac_i_j corresponds to Adj[i, j]
            for j in range(n):
                if Adj[i, j] == 1:
                    A_diag[
                        nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)
                    ] = d_systems[i]["Ac"][idxs[i]][coupling_idx]
                    coupling_idx += 1

        S.append(block_diag(*S_tmp))
        R.append(block_diag(*R_tmp))
        T.append(np.vstack(T_tmp))
        A.append(A_diag)
        B.append(block_diag(*B_tmp))
        c.append(np.vstack(c_tmp))

    # state and control constraints which don't switch by region
    D = block_diag(*[d_systems[i]["D"] for i in range(n)])
    E = np.vstack([d_systems[i]["E"] for i in range(n)])
    F = block_diag(*[d_systems[i]["F"] for i in range(n)])
    G = np.vstack([d_systems[i]["G"] for i in range(n)])

    return {
        "S": S,
        "R": R,
        "T": T,
        "A": A,
        "B": B,
        "c": c,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    }
