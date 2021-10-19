# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101
# %% Packages
import pandas as pd
import numpy as np
import networkx as nx
import os
import itertools
import scipy.sparse as sp
from scipy.spatial import distance
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

# from joblib import Parallel, delayed, load, dump
# from mpl_toolkits.basemap import Basemap
# import osmnx as ox

# %% Information
"""
This version only for valuation of proposed routes!

tau := zone-to-zone travel time numpy array
tau2 := zone-to-zone travel time numpy array with diagonal elements equal to one
ctr := a dictionary with key as zones and values be the set of neighbor zones for the key including the key itself
S := Number of zones
T := Number of time steps
beta := time flexibility budget
Rider and Driver Numpy array, for every row we have these columns:
0: user index
1: IGNORE!
2: earliest departure time
3: latest arrival time
4: origin
5: destination
6 (Rider only): tranfer limit
6 (Driver only, repeated tour only): one tour duration
"""


def Many2Many(Rider, Driver, tau, tau2, ctr, V, config, fixed_route_D=None):
    beta = config["beta"]
    S = config["S"]
    T = config["T"]
    Tmin = np.min(np.r_[Rider[:, 2], Driver[:, 2]])
    Tmax = np.max(np.r_[Rider[:, 3], Driver[:, 3]])
    VEH_CAP = config["VEH_CAP"]
    FIXED_ROUTE = config["FIXED_ROUTE"]
    REPEATED_TOUR = config["REPEATED_TOUR"]
    TIME_LIMIT = config["TIME_LIMIT"]
    MIP_GAP = config["MIP_GAP"]
    PENALIZE_RATIO = config["PENALIZE_RATIO"]

    # Time expanded network
    TN = nx.DiGraph()
    for t in range(Tmin, Tmax + 1):
        for i in V:
            for j in set(ctr[i]) & V:
                if t + tau2[i, j] <= T:
                    TN.add_edge(
                        t * S + i, (t + tau2[i, j]) * S + j
                    )  # Every node in time-expaned network is defined as t_i*S+s_i
                    # which maps (t_i,s_i) into a unique n_i value
    # TN = TN.subgraph({t * S + i for t in range(Tmin, Tmax + 1) for i in V})
    # ---------------------------------------------------------
    # Rider data
    R, O_r, D_r, ED_r, LA_r, SP_r, TW_r, T_r, V_r = gp.multidict(
        {
            r: [
                Rider[i, 4],
                Rider[i, 5],
                Rider[i, 2],
                Rider[i, 3],
                tau[Rider[i, 4], Rider[i, 5]],
                Rider[i, 3] - Rider[i, 2],
                np.arange(Rider[i, 2], Rider[i, 3] + 1),
                Rider[i, 6],
            ]
            for i, r in enumerate(Rider[:, 0])
            if Rider[i, 4] in V and Rider[i, 5] in V
        }
    )

    # Driver data

    # If repeated tour: T_d should be for the first single tour for the each driver, not for the entire time horizon
    #                   TW_d should be the time window for the first single tour
    #                   ED_d, LA_d should be earliest departure/latest arrival time for the first tour

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
    D += [
        (d_p),
    ]

    if REPEATED_TOUR:
        Duration_d = {d: Driver[i, 6] for i, d in enumerate(Driver[:, 0])}
        num_tour = {d: T // (Duration_d[d]) for d in D if d != d_p}
        # rider r 登上 driver d 最早班次
        RD_on = {(r, d): ED_r[r] // (Duration_d[d]) for r in R for d in D if d != d_p}
        # rider r 离开 driver d 的最晚班次
        RD_off = {(r, d): LA_r[r] // (Duration_d[d]) for r in R for d in D if d != d_p}

    # ---------------------------------------------------------
    if REPEATED_TOUR:
        # no dummy driver, assume all rider just want to get on the bus ASAP,
        # won't wait till next tour
        # rider r 被 driver d 接乘的最早时间
        pick = {
            (r, d): np.maximum(
                ED_d[d] + RD_on[r, d] * (Duration_d[d]) + tau[O_d[d], O_r[r]], ED_r[r]
            )
            for r in R
            for d in D
            if d != d_p
        }
    else:
        # rider r 被 driver d 接乘的最早时间
        pick = {
            (r, d): np.maximum(ED_d[d] + tau[O_d[d], O_r[r]], ED_r[r])
            for r in R
            for d in D
            if d != d_p
        }  # no dummy driver

    # 从最早接乘时间开始计算，Rider r 被 driver d 放下的最早时间
    drop = {
        (r, d): pick[r, d] + tau[O_r[r], D_r[r]] for r in R for d in D if d != d_p
    }  # no dummy driver, pickup time + shortest travel time of rider
    # 从放下的最早时间开始计算，driver d 到达其终点站的最早时间
    finish = {
        (r, d): drop[r, d] + tau[D_r[r], D_d[d]] for r in R for d in D if d != d_p
    }  # no dummy driver, latest return-to-depot time of driver

    if REPEATED_TOUR:
        RD = {
            (r, d)
            for r in R
            for d in D
            if d != d_p
            if drop[r, d] <= LA_r[r]  # 最早放下时间不大于r的最晚到达时间
            if finish[r, d]
            <= LA_d[d] + RD_on[r, d] * (Duration_d[d])  # 最早到站时间不大于该周期的最晚到终点站时间
            if RD_on[r, d] < num_tour[d]
        }

    else:
        RD = {
            (r, d)
            for r in R
            for d in D
            if d != d_p
            if drop[r, d] <= LA_r[r]
            if finish[r, d] <= LA_d[d]
        }

    RD = RD | set((r, d_p) for r in R)  # includes dummy driver

    # All feasible R-D match pairs which ensures both rider and driver
    # arrive their destinations in time
    if REPEATED_TOUR:
        drop2 = {
            (r, d): np.minimum(
                LA_r[r],
                LA_d[d] + RD_on[r, d] * (Duration_d[d]) - tau[D_r[r], D_d[d]],
            )
            for (r, d) in RD
            if d != d_p
        }

    else:
        drop2 = {
            (r, d): np.minimum(LA_r[r], LA_d[d] - tau[D_r[r], D_d[d]])
            for (r, d) in RD
            if d != d_p
        }  # no dummy driver, 必须在此之前drop rider才能使driver按时返回O_d

    # ---------------------------------------------------------
    # SN_r = {r: set((ED_r[r] + b) * S + O_r[r] for b in range(beta + 1)) for r in R}
    SN_r = {
        r: set((ED_r[r] + m) * S + O_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    # EN_r = {r: set((LA_r[r] - b) * S + D_r[r] for b in range(beta + 1)) for r in R}
    EN_r = {
        r: set((LA_r[r] - m) * S + D_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    L_d, N_d, SN_d, EN_d, MN_d = {}, {}, {}, {}, {}
    L_rd, N_rd = {}, {}

    # %% Reduced network
    if not REPEATED_TOUR:
        for d in D:
            if d != d_p:
                Gs = list(np.where(tau[O_d[d], :] + tau[:, D_d[d]] <= TW_d[d])[0])
                N_d[d] = set()
                MN_d[d] = set()

                for s in Gs:
                    ind = np.where(
                        (ED_d[d] + tau[O_d[d], s] <= T_d[d])
                        & (T_d[d] + tau[s, D_d[d]] <= LA_d[d])
                    )[0]

                    if O_d[d] != D_d[d]:
                        if s == O_d[d]:
                            SN_d[d] = T_d[d][ind] * S + s
                        elif s == D_d[d]:
                            EN_d[d] = T_d[d][ind] * S + s
                        else:
                            MN_d[d].update(T_d[d][ind] * S + s)
                    else:
                        if s == O_d[d]:
                            SN_d[d] = T_d[d][ind] * S + s
                            EN_d[d] = T_d[d][ind] * S + s
                        else:
                            MN_d[d].update(T_d[d][ind] * S + s)

                    N_d[d].update(T_d[d][ind] * S + s)

                TN_d = TN.subgraph(N_d[d])
                L_d[d] = set(TN_d.edges())

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

        # L_r_dp = dict()
        # for r in R:
        #     L_r_dp[r] = set()
        #     for n in N_r_temp[r]:
        #         for m in range(LA_r[r]):
        #             if n + (m + 1) * S <= max(N_r_temp[r]):
        #                 L_r_dp[r] = L_r_dp[r] | {(n + m * S, n + (m + 1) * S)}

        # Link set for the dummy driver
        L_d[d_p] = set((t * S + s, (t + 1) * S + s) for t in range(T) for s in range(S))

        # Link set for all the pairs of the dummy driver and the rider
        for r in R:
            # if r in L_r_dp:
            L_rd[r, d_p] = (
                set(
                    (n1, n2)
                    for (r_, d_) in L_rd.keys()
                    if r_ == r
                    for (n1, n2) in L_rd[(r_, d_)]
                )
                & L_d[d_p]
                # &
                # L_r_dp[r]
            )

        # Node set for each rider, including the waiting nodes, not SNs, and ENs
        RN_r = dict()
        for r in R:
            RN_r[r] = set()
            for d in D:
                if (r, d) in L_rd:
                    for (n1, n2) in L_rd[r, d]:
                        RN_r[r].update([n1, n2])

            RN_r[r] = RN_r[r] - SN_r[r]
            RN_r[r] = RN_r[r] - EN_r[r]

    else:
        for d in D:
            if d != d_p:
                N_d[d] = set()
                MN_d[d] = set()
                for m in range(num_tour[d]):
                    Gs = list(
                        np.where(tau[O_d[d], :] + tau[:, D_d[d]] <= TW_d[d])[0]
                    )  # list of reachable zones for each driver

                    for s in Gs:
                        ind = np.where(
                            (ED_d[d] + tau[O_d[d], s] <= T_d[d])
                            & (T_d[d] + tau[s, D_d[d]] <= LA_d[d])
                        )[
                            0
                        ]  # for each reachable zone, the feasible routing time range for each driver

                        if O_d[d] != D_d[d]:
                            if s == O_d[d] and m == 0:
                                SN_d[d] = (
                                    T_d[d][ind] * S + s
                                )  # need to be the first origin node

                            elif s == D_d[d] and m == T // (Duration_d[d]) - 1:
                                EN_d[d] = (
                                    T_d[d][ind] + m * (Duration_d[d])
                                ) * S + s  # need to be the last destination node

                            else:
                                MN_d[d].update(
                                    (T_d[d][ind] + m * (Duration_d[d])) * S + s
                                )
                        else:
                            if s == O_d[d] and m == 0:
                                SN_d[d] = (
                                    T_d[d][ind] * S + s
                                )  # need to be the first origin node
                                if LA_d[d] in T_d[d][ind]:
                                    SN_d[d] = SN_d[d][SN_d[d] != LA_d[d] * S + O_d[d]]

                                if num_tour[d] == 1:
                                    EN_d[d] = (
                                        T_d[d][ind] + m * (Duration_d[d])
                                    ) * S + s  # need to be the last destination node
                                    if ED_d[d] + m * (Duration_d[d]) in T_d[d][
                                        ind
                                    ] + m * (Duration_d[d]):
                                        EN_d[d] = EN_d[d][
                                            EN_d[d]
                                            != (ED_d[d] + m * (Duration_d[d])) * S
                                            + D_d[d]
                                        ]
                                # elif s == D_d[d]:
                            elif s == D_d[d] and m == T // (Duration_d[d]) - 1:
                                EN_d[d] = (
                                    T_d[d][ind] + m * (Duration_d[d])
                                ) * S + s  # need to be the last destination node
                                if ED_d[d] + m * (Duration_d[d]) in T_d[d][ind] + m * (
                                    Duration_d[d]
                                ):
                                    EN_d[d] = EN_d[d][
                                        EN_d[d]
                                        != (ED_d[d] + m * (Duration_d[d])) * S + D_d[d]
                                    ]

                            else:
                                MN_d[d].update(
                                    (T_d[d][ind] + m * (Duration_d[d])) * S + s
                                )

                        N_d[d].update(
                            (T_d[d][ind] + m * (Duration_d[d])) * S + s
                        )  # end for loop

                TN_d = TN.subgraph(N_d[d])
                L_d[d] = set(TN_d.edges())

                # SN_d[d] = (
                #     T_d[d][0] * S + O_d[d]
                # )  # need to be the first origin node (just be safe, seems redundant)
                # L_r_dp = dict()  # link set for dummy driver-rider pair

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

        # N_r_temp = dict()

        # for r in R:
        #     N_r_temp[r] = set()
        #     for d in D:
        #         if (r, d) in N_rd and d != d_p:
        #             N_r_temp[r] = N_r_temp[r] | N_rd[r, d]

        # L_r_dp = dict()
        # for r in R:
        #     L_r_dp[r] = set()
        #     for n in N_r_temp[r]:
        #         for m in range(LA_r[r]):
        #             if n + (m + 1) * S <= max(N_r_temp[r]):
        #                 L_r_dp[r] = L_r_dp[r] | {(n + m * S, n + (m + 1) * S)}

        # L_r_dp = {
        #     r: (n + m * S, n + (m + 1) * S)
        #     for r in R
        #     for n in N_r_temp[r]
        #     for m in range(LA_r[r] + 1)
        #     if n + (m + 1) * S <= max(N_r_temp[r])
        # }

        # Link set for the dummy driver
        L_d[d_p] = set((t * S + s, (t + 1) * S + s) for t in range(T) for s in range(S))

        # Link set for all the pairs of the dummy driver and the rider
        for r in R:
            # if r in L_r_dp:
            L_rd[r, d_p] = (
                set(
                    (n1, n2)
                    for (r_, d_) in L_rd.keys()
                    if r_ == r
                    for (n1, n2) in L_rd[(r_, d_)]
                )
                & L_d[d_p]
                # &
                # L_r_dp[r]
            )

        # Node set for each rider, including the waiting nodes, no SNs, nor ENs
        RN_r = dict()
        for r in R:
            RN_r[r] = set()
            for d in D:
                if (r, d) in L_rd:
                    for (n1, n2) in L_rd[r, d]:
                        RN_r[r].update([n1, n2])

            RN_r[r] = RN_r[r] - SN_r[r]
            RN_r[r] = RN_r[r] - EN_r[r]

    # ----------------------------------------------------
    m = gp.Model("many2many")
    # m.params.OutputFlag = 0
    # m.params.Threads = 2

    TL = TIME_LIMIT

    ### Sets ###
    rl, dl = len(Rider), len(Driver)

    DL_d = set(
        (d, n1, n2)
        for key, val in L_d.items()
        for (d, (n1, n2)) in itertools.product([key], val)
        if d != d_p
    )  # not include dummy driver

    RDL_rd = gp.tuplelist(
        (r, d, int(n1), int(n2))
        for key, val in L_rd.items()
        for ((r, d), (n1, n2)) in itertools.product([key], val)
    )  # include dummy driver

    DN_d = [(d, n) for d in D if d != d_p for n in MN_d[d]]  # not include dummy driver
    DR_r = {r: set(d for d in D if (r, d) in RD) for r in R}  # include dummy driver

    # fixed-route setting
    # fixed_route: dict with key as fixed route driver id,
    # value as array of time (first column) and stop (second column)
    if FIXED_ROUTE and fixed_route_D != None:
        FR_d = dict()
        for d in fixed_route_D.keys():
            for (t1, t2, s1, s2, *_) in fixed_route_D[d]:
                # for i in range(len(fixed_route_D[d])):
                for j in range(num_tour[d]):
                    FR_d.setdefault(d, list()).append(
                        (
                            (t1 + j * (Duration_d[d])) * S + s1,
                            (t2 + j * (Duration_d[d])) * S + s2,
                        )
                    )
        FRL_d = set(
            (d, n1, n2)
            for key, val in FR_d.items()
            for (d, (n1, n2)) in itertools.product([key], val)
            if d != d_p
        )

        DL_d.update(FRL_d)

    ### Variables ###
    x = m.addVars(DL_d, vtype=GRB.BINARY, name="x")
    y = m.addVars(RDL_rd, vtype=GRB.BINARY, name="y")
    u = m.addVars(RD, vtype=GRB.BINARY, name="u")
    z = m.addVars(R, vtype=GRB.BINARY, name="z")

    if FIXED_ROUTE and fixed_route_D != None:
        for (d, n1, n2) in FRL_d:
            x[d, n1, n2].lb = 1
            x[d, n1, n2].ub = 1
        # for (d, n1, n2) in DL_d:
        #     if (d, n1, n2) not in FRL_d:
        #         x[d, n1, n2].lb = 0
        #         x[d, n1, n2].ub = 0
        m.update()

    ### Objective ###
    if not PENALIZE_RATIO:
        m.setObjective(
            z.sum(),
            GRB.MAXIMIZE,
        )
    else:
        LAMBDA = config["LAMBDA"]
        m.setObjective(
            sum(
                z[r]
                - LAMBDA
                * 1
                / SP_r[r]
                * sum(
                    (n2 // S - n1 // S) * y[rr, d, n1, n2]
                    for (rr, d, n1, n2) in RDL_rd
                    if rr == r
                )
                for r in R
            ),
            GRB.MAXIMIZE,
        )
    # m.setObjective(
    # z.prod(SP_r)
    # - sum(x[d, n1, n2] * (n2 // S - n1 // S) for (d, n1, n2) in DL_d)
    # + sum(SP_d[d] for d in D),
    #     GRB.MAXIMIZE,
    # )

    ### Constraints ###
    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in SN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in SN_d[d])
            == 1
            for d in D
            if d != d_p
        ),
        "source_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in EN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in EN_d[d])
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
            for (d, n) in DN_d  # DN_d not include dummy driver
            if d != d_p  # (just be safe, seemly redundant)
        ),
        "trans_d",
    )

    m.addConstrs(
        (
            gp.quicksum(
                y[r, d, n1, n2]
                for d in DR_r[r]  # include dummy driver
                if (r, d) in L_rd
                for (n1, n2) in L_rd[r, d]  # include dummy driver
                if n1 in SN_r[r]
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in DR_r[r]  # include dummy driver
                if (r, d) in L_rd
                for (n1, n2) in L_rd[r, d]  # include dummy driver
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
                for d in DR_r[r]  # include dummy driver
                if (r, d) in L_rd
                for (n1, n2) in L_rd[r, d]  # include dummy driver
                if n2 in EN_r[r]
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in DR_r[r]
                if (r, d) in L_rd
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
                for d in DR_r[r]  # include dummy driver
                if (r, d) in L_rd
                for (n1, n2) in L_rd[r, d]  # include dummy driver
                # if n2 not in EN_r[r]
                # if n2 not in SN_r[r]
                if n1 == n
            )
            - gp.quicksum(
                y[r, d, n1, n2]
                for d in DR_r[r]  # include dummy driver
                if (r, d) in L_rd
                for (n1, n2) in L_rd[r, d]  # include dummy driver
                # if n1 not in EN_r[r]
                # if n1 not in SN_r[r]
                if n2 == n
            )
            == 0
            for r in R
            for n in RN_r[r]
            # for (r, d, n) in RDN_rd
        ),
        "trans_r",
    )

    m.addConstrs(
        (y.sum("*", d, n1, n2) <= VEH_CAP * x[d, n1, n2] for (d, n1, n2) in DL_d),
        "capacity",
    )  # not include dummy driver

    m.addConstrs(
        (u[r, d] - y[r, d, n1, n2] >= 0 for (r, d, n1, n2) in RDL_rd if d != d_p),
        "feasibility1",
    )  # RDL_rd does not have dummy driver, redundant

    m.addConstrs(
        (u[r, d] - y.sum(r, d, "*", "*") <= 0 for (r, d) in RD if d != d_p),
        "feasibility2",
    )  # RD includes dummy driver, should exclude d_p

    m.addConstrs((u.sum(r, "*") - 1 - V_r[r] <= 0 for r in R), "tranfer_limit")

    # repeated tour constraint
    if REPEATED_TOUR:
        m.addConstrs(
            (
                x[d, n1, n2] - x[d, n1 - (Duration_d[d]) * S, n2 - (Duration_d[d]) * S]
                == 0
                for (d, n1, n2) in DL_d  # not include dummy driver
                # if n1 >= (Duration_d[d]) * S
                # if n2 >= (Duration_d[d]) * S
                if (d, n1 - (Duration_d[d]) * S, n2 - (Duration_d[d]) * S) in DL_d
            ),
            "fix_route1",
        )

    m.params.Method = -1
    m.params.TimeLimit = TL
    m.params.MIPGap = MIP_GAP
    m.params.OutputFlag = 1
    m.optimize()

    if m.status == GRB.TIME_LIMIT:
        X = {e for e in DL_d if x[e].x > 0.001}
        U = {e for e in RD if u[e].x > 0.001}
        Y = {e for e in RDL_rd if y[e].x > 0.001}
        R_match = {r for r in R if z[r].x > 0.001}
        mr = sum(z[r].x for r in R) / len(Rider)

        Route_D = dict()
        for d in D:
            if d != d_p:
                Route_D[d] = list()
                for (n_1, n_2) in L_d[d]:
                    if np.round(x[d, n_1, n_2].x) == 1:
                        # Route_D[d].append((n_1, n_2))
                        t1 = n_1 // S
                        s1 = n_1 % S
                        t2 = n_2 // S
                        s2 = n_2 % S
                        Route_D[d].append((t1, t2, s1, s2, n_1, n_2))
                        # Route_D[d].append(n_2)
                Route_D[d] = sorted(list(Route_D[d]), key=lambda x: x[0])
    elif m.status == GRB.OPTIMAL:
        X = {e for e in DL_d if x[e].x > 0.001}
        U = {e for e in RD if u[e].x > 0.001}
        Y = {e for e in RDL_rd if y[e].x > 0.001}
        R_match = {r for r in R if z[r].x > 0.001}
        mr = sum(z[r].x for r in R) / len(Rider)

        Route_D = dict()
        for d in D:
            if d != d_p:
                Route_D[d] = list()
                for (n_1, n_2) in L_d[d]:
                    if np.round(x[d, n_1, n_2].x) == 1:
                        # Route_D[d].append((n_1, n_2))
                        t1 = n_1 // S
                        s1 = n_1 % S
                        t2 = n_2 // S
                        s2 = n_2 % S
                        Route_D[d].append((t1, t2, s1, s2, n_1, n_2))
                        # Route_D[d].append(n_2)
                Route_D[d] = sorted(list(Route_D[d]), key=lambda x: x[0])
                # Route_D[d] = sorted(list(Route_D[d]))
    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        if not os.path.exists(config["m2m_output_loc"]):
            os.makedirs(config["m2m_output_loc"])
        m.write(config["m2m_output_loc"] + "model.ilp")
        raise ValueError("Infeasible solution!")
    else:
        print("Status code: {}".format(m.status))
    return (X, U, Y, Route_D, mr, m.ObjVal, R_match)


# %%
