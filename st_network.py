# %% Packages
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import time
import itertools
import random
import copy
import sys
from scipy.spatial import distance
import scipy.sparse as sp
import osmnx as ox
from mpl_toolkits.basemap import Basemap
from scipy.spatial import distance
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed, load, dump

# %% Information
# tau := zone-to-zone travel time numpy array
# tau2 := zone-to-zone travel time numpy array with diagonal elements equal to one
# ctr := a dictionary with key as zones and values be the set of neighbor zones for the key including the key itself
# S := Number of zones
# T := Number of time steps
# beta := time flexibility budget
# Rider and Driver Numpy array, for every row we have these columns:
# 0: user index
# 1: IGNORE!
# 2: earliest departure time
# 3: latest arrival time
# 4: origin
# 5: destination

# %%
def One2Many(Rider, Driver, tau, tau2, ctr, config):
    beta = config["beta"]
    S = config["S"]
    T = config["T"]
    Tmin = np.min(np.r_[Rider[:, 2], Driver[:, 2]])
    Tmax = np.max(np.r_[Rider[:, 3], Driver[:, 3]])
    # Time expanded network
    TN = nx.DiGraph()
    for t in range(Tmin, Tmax + 1):
        for i in range(S):
            for j in ctr[i]:
                if t + tau2[i, j] <= T:
                    TN.add_edge(
                        t * S + i, (t + tau2[i, j]) * S + j
                    )  # Every node in time-expaned network is defined as t_i*S+s_i which maps (t_i,s_i) into a unique n_i value
    R, O_r, D_r, ED_r, LA_r, SP_r, TW_r, T_r = gp.multidict(
        {
            r: [
                Rider[i, 4],
                Rider[i, 5],
                Rider[i, 2],
                Rider[i, 3],
                tau[Rider[i, 4], Rider[i, 5]],
                Rider[i, 3] - Rider[i, 2],
                np.arange(Rider[i, 2], Rider[i, 3] + 1),
            ]
            for i, r in enumerate(Rider[:, 0])
        }
    )

    V_r = {r: 5 for r in R}  # num of transfer limit

    # T_d should be for the first single tour for the each driver, not for the entire time horizon
    # TW_d should be the time window for the first single tour
    # ED_d, LA_d should be earliest departure/latest arrival time for the first tour
    D, O_d, D_d, ED_d, LA_d, SP_d, TW_d, T_d = gp.multidict(
        {
            d: [
                Driver[i, 4],
                Driver[i, 5],
                Driver[i, 2],
                Driver[i, 3],
                tau[Driver[i, 4], Driver[i, 5]],
                Driver[i, 3] - Driver[i, 2],
                np.arange(Driver[i, 2], Driver[i, 3] + 1),
            ]
            for i, d in enumerate(Driver[:, 0])
        }
    )
    d_p = len(Driver)  # ID of the dummy driver
    D.add(d_p)

    pick = {
        (r, d): np.maximum(ED_d[d] + tau[O_d[d], O_r[r]], ED_r[r])
        for r in R
        for d in D
        if d != d_p
    }  # ???
    drop = {
        (r, d): pick[r, d] + tau[O_r[r], D_r[r]] for r in R for d in D if d != d_p
    }  # ???
    finish = {
        (r, d): drop[r, d] + tau[D_r[r], D_d[d]] for r in R for d in D if d != d_p
    }  # ???
    RD = {
        (r, d) for r in R for d in D if drop[r, d] <= LA_r[r] if finish[r, d] <= LA_d[d]
    }
    RD = RD | set((r, d_p) for r in R)
    drop2 = {
        (r, d): np.minimum(LA_r[r], LA_d[d] - tau[D_r[r], D_d[d]])
        for (r, d) in RD
        if d != d_p
    }  # ???
    # ---------------------------------------------------------
    SN_r = {r: set((ED_r[r] + b) * S + O_r[r] for b in range(beta + 1)) for r in R}
    EN_r = {r: set((LA_r[r] - b) * S + D_r[r] for b in range(beta + 1)) for r in R}
    L_d, N_d, SN_d, EN_d, MN_d = {}, {}, {}, {}, {}
    L_rd, N_rd = {}, {}
    # %%
    for d in D:
        # reduced network
        for m in range(T // TW_d[d]):
            Gs = list(np.where(tau[O_d[d], :] + tau[:, D_d[d]] <= TW_d[d])[0])
            N_d[d] = set()
            MN_d[d] = set()
            for s in Gs:
                ind = np.where(
                    (ED_d[d] + tau[O_d[d], s] <= T_d[d])
                    & (T_d[d] + tau[s, D_d[d]] <= LA_d[d])
                )[0]
                if s == O_d[d] and m == 0:
                    SN_d[d] = T_d[d][ind] * S + s  # need to be the first origin node
                elif s == D_d[d] and m == T // TW_d[d] - 1:
                    EN_d[d] = (
                        T_d[d][ind] + m * TW_d[d]
                    ) * S + s  # need to be the last destination node
                else:
                    MN_d[d].update((T_d[d][ind] + m * TW_d[d]) * S + s)
                N_d[d].update((T_d[d][ind] + m * TW_d[d]) * S + s)  # end for loop

        TN_d = TN.subgraph(N_d[d])
        L_d[d] = set(TN_d.edges())
        SN_d[d] = T_d[d][0] * S + O_d[d]  # need to be the first origin node
        # %%
        for r in R:
            if (r, d) in RD:
                nodes = nx.dag.descendants(
                    TN_d, pick[r, d] * S + O_r[r]
                ) & nx.dag.ancestors(TN_d, drop2[r, d] * S + D_r[r])
                nodes.add(pick[r, d] * S + O_r[r])
                nodes.add(drop2[r, d] * S + D_r[r])
                TN_r = TN_d.subgraph(nodes)
                L_rd[r, d] = set(TN_r.edges())
                nodes = nodes - SN_r[r]
                N_rd[r, d] = nodes - EN_r[r]

    L_d[d_p] = set((t * S + s, (t + 1) * S + s) for t in range(T) for s in range(S))

    for r in R:
        L_rd[r, d_p] = set((n1, n2) for (n1, n2) in L_rd[r, d] for d in D) & L_d[d_p]

    # ----------------------------------------------------
    m = gp.Model("one2one")
    #     m.params.OutputFlag = 0
    # m.params.Threads = 2
    TL = 200
    ### Sets ###
    rl, dl = len(Rider), len(Driver)
    DL_d = set(
        (d, n1, n2)
        for key, val in L_d.items()
        for (d, (n1, n2)) in itertools.product([key], val)
        if d != d_p
    )
    RDL_rd = gp.tuplelist(
        (r, d, n1, n2)
        for key, val in L_rd.items()
        for ((r, d), (n1, n2)) in itertools.product([key], val)
    )
    DN_d = [(d, n) for d in D for n in MN_d[d] if d != d_p]
    D_r = {r: set(d for d in D if (r, d) in RD)}

    # RDN_rd = [(r, d, n) for (r, d) in RD for n in N_rd[r, d]]  # hold
    ### Variables ###
    x = m.addVars(DL_d, vtype=GRB.BINARY, name="x")
    y = m.addVars(RDL_rd, vtype=GRB.BINARY, name="y")
    u = m.addVars(RD, vtype=GRB.BINARY, name="u")
    z = m.addVars(R, vtype=GRB.BINARY, name="z")

    # fixed-route setting
    if config["FIXED_ROUTE"]:  # TO DO
        Route = {d: list((n1, n2)) for d in fixed_route_D}
        for d in Route.keys():
            for (n1, n2) in Route[d]:
                x[d, n1, n2].lb = 1
                x[d, n1, n2].ub = 1
        m.update()

    ### Objective ###
    m.setObjective(
        z.sum(), GRB.MAXIMIZE,
    )
    # z.prod(SP_r)
    # - sum(x[d, n1, n2] * (n2 // S - n1 // S) for (d, n1, n2) in DL_d)
    # + sum(SP_d[d] for d in D),
    #     GRB.MAXIMIZE,
    # )
    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in SN_d[d])
            - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in SN_d[d])
            == 1
            for d in D
            if d != d_p
        ),
        "source_d",
    )
    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in EN_d[d])
            - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in EN_d[d])
            == 1
            for d in D
            if d != d_p
        ),
        "dest_d",
    )
    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 == n)
            - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 == n)
            == 0
            for (d, n) in DN_d
            if d != d_p
        ),
        "trans_d",
    )
    m.addConstrs(
        (
            gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n1 in SN_r[r]
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n2 in SN_r[r]
            )
            == z[r]
            for r in R
        ),
        "source_r",
    )

    m.addConstrs(
        (
            gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n2 in EN_r[r]
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n1 in EN_r[r]
            )
            == z[r]
            for r in R
        ),
        "dest_r",
    )
    m.addConstrs(
        (
            gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n2 not in EN_r[r]
                if n2 not in SN_r[r]
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in D_r[r]
                for (n1, n2) in L_rd[r, d]
                if n1 not in EN_r[r]
                if n1 not in SN_r[r]
            )
            == 0
            for r in R
            # for (r, d, n) in RDN_rd
        ),
        "trans_r",
    )
    m.addConstrs(
        (y.sum("*", d, n1, n2) <= 3 * x[d, n1, n2] for (d, n1, n2) in DL_d), "capacity"
    )
    # m.addConstrs((u.sum(r, "*") == z[r] for r in R), "singled")

    m.addConstrs(
        (u[r, d] - y[r, d, n1, n2] >= 0 for (r, d, n1, n2) in RDL_rd if d != d_p),
        "feasibility1",
    )

    m.addConstrs(
        (u[r, d] - y.sum(r, d, "*", "*") <= 0 for (r, d) in RD), "feasibility2"
    )

    m.addConstrs((u.sum(r, "*") - 1 - V_r[r] <= 0 for r in R), "tranfer_limit")

    # fixed route constraint 1
    m.addConstrs(
        (
            x[d, n1, n2] - x[d, n1 - T0 * S, n2 - T0 * S] == 0
            for (d, n1, n2) in DL_d
            if n1 >= T0 * S
            if n2 >= T0 * S
        ),
        "fix_route1",
    )
    # fixed route constraint 2
    # m.addConstrs(
    #     (

    #     ),
    #     "fix_route2",
    # )

    m.params.TimeLimit = TL
    m.params.MIPGap = 0.05
    m.optimize()
    if m.status == GRB.TIME_LIMIT:
        X = {e for e in DL_d if x[e].x > 0.001}
        U = {e for e in RD if u[e].x > 0.001}
        Y = {e for e in RDL_rd if y[e].x > 0.001}
    elif m.status == GRB.OPTIMAL:
        X = {e for e in DL_d if x[e].x > 0.001}
        U = {e for e in RD if u[e].x > 0.001}
        Y = {e for e in RDL_rd if y[e].x > 0.001}
    elif m.status == GRB.INFEASIBLE:
        raise ValueError("Infeasible solution!")
    return (X, U, set(D), m.OBJ)


# # Mapping back
# t_i, s_i = np.divmod(n_i, S)

# # Extract routes
# Yiyang = {}
# for d in D:
#     Yiyang[d] = set()
#     for (n_1, n_2) in L_d[d]:
#         if np.round(x[d, n_1, n_2].x) == 1:
#             Yiyang.add(n_1)
#             Yiyang.add(n_2)
# sorted(list(Yiyang[d]))

# sum(z[r].x for r in R) / len(R)

