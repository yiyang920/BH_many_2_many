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


def Many2Many(Rider, Driver, tau, tau2, ctr, config, fixed_route_D=None, start=None):
    """
    Ver 3.2:
    Relax variable x, y, u.
    Ver 3.1:
    1) When penalize ATT-SPTT ratio the SPTT is the shortest path travel time in the disaggregated network.
    2) Initialization of variable from previous optimization results
    Ver 3.0:
    This version considers flexible driver's depot.
    ==================================================================================================================
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
    7 (Rider only): SPTT in disaggregated network
    """
    VEH_CAP = config["VEH_CAP"]
    FIXED_ROUTE = config["FIXED_ROUTE"]
    REPEATED_TOUR = config["REPEATED_TOUR"]
    TIME_LIMIT = config["TIME_LIMIT"]
    MIP_GAP = config["MIP_GAP"]
    PENALIZE_RATIO = config["PENALIZE_RATIO"]
    beta = config["beta"]
    S = config["S"]
    T = config["T"]
    Tmin = np.min(np.r_[Rider[:, 2], Driver[:, 2]])
    Tmax = (
        np.max(np.r_[Rider[:, 3], Driver[:, 3]])
        if not REPEATED_TOUR
        else np.max(
            np.r_[
                Rider[:, 3],
                T // (Driver[:, 3] - Driver[:, 2]) * (Driver[:, 3] - Driver[:, 2]),
            ]
        )
    )

    # Time expanded network
    TN = nx.DiGraph()
    for t in range(Tmin, Tmax + 1):
        for i in range(S):
            for j in ctr[i]:
                if t + tau2[i, j] <= T:
                    TN.add_edge(
                        t * S + i, (t + tau2[i, j]) * S + j
                    )  # Every node in time-expaned network is defined as t_i*S+s_i
                    # which maps (t_i,s_i) into a unique n_i value

    # ---------------------------------------------------------
    # Rider data
    R, O_r, D_r, ED_r, LA_r, SP_r_agg, TW_r, T_r, V_r, SP_r = gp.multidict(
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
                Rider[i, 7],
            ]
            for i, r in enumerate(Rider[:, 0])
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

    # driver list with flex OD, bus mode or dial-a-ride mode
    D_flex_OD = config["flex_od_driver"]
    if D_flex_OD is None:
        D_flex_OD = []

    if REPEATED_TOUR:
        Duration_d = {d: Driver[i, 6] for i, d in enumerate(Driver[:, 0])}
        num_tour = {d: T // (Duration_d[d]) for d in set(D) - {d_p}}
        # rider r 登上 driver d 最早班次
        RD_on = {(r, d): ED_r[r] // (Duration_d[d]) for r in R for d in set(D) - {d_p}}
        # rider r 离开 driver d 的最晚班次
        RD_off = {(r, d): LA_r[r] // (Duration_d[d]) for r in R for d in set(D) - {d_p}}

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
            for d in set(D) - set(D_flex_OD) - {d_p}
        }
        pick.update({(r, d): ED_r[r] for r in R for d in D_flex_OD})
    else:
        # rider r 被 driver d 接乘的最早时间
        pick = {
            (r, d): np.maximum(ED_d[d] + tau[O_d[d], O_r[r]], ED_r[r])
            for r in R
            for d in set(D) - set(D_flex_OD) - {d_p}
        }  # no dummy driver
        pick.update({(r, d): ED_r[r] for r in R for d in D_flex_OD})

    # 从最早接乘时间开始计算，Rider r 被 driver d 放下的最早时间
    drop = {
        (r, d): pick[r, d] + tau[O_r[r], D_r[r]] for r in R for d in set(D) - {d_p}
    }  # no dummy driver, pickup time + shortest travel time of rider
    # 从放下的最早时间开始计算，driver d 到达其终点站的最早时间
    finish = {
        (r, d): drop[r, d] + tau[D_r[r], D_d[d]]
        for r in R
        for d in set(D) - set(D_flex_OD) - {d_p}
    }  # no dummy driver, latest return-to-depot time of driver
    finish.update({(r, d): drop[r, d] for r in R for d in D_flex_OD})

    if REPEATED_TOUR:
        RD = {
            (r, d)
            for r in R
            for d in set(D) - set(D_flex_OD) - {d_p}
            if drop[r, d] <= LA_r[r]  # 最早放下时间不大于r的最晚到达时间
            if finish[r, d]
            <= LA_d[d] + RD_on[r, d] * (Duration_d[d])  # 最早到站时间不大于该周期的最晚到终点站时间
            if RD_on[r, d] < num_tour[d]
        }

    else:
        RD = {
            (r, d)
            for r in R
            for d in set(D) - set(D_flex_OD) - {d_p}
            if drop[r, d] <= LA_r[r]
            if finish[r, d] <= LA_d[d]
        }

    RD.update({(r, d) for r in R for d in D_flex_OD})
    RD.update(set((r, d_p) for r in R))  # includes dummy driver

    # All feasible R-D match pairs which ensures both rider and driver
    # arrive their destinations in time
    if REPEATED_TOUR:
        drop2 = {
            (r, d): np.minimum(
                LA_r[r],
                LA_d[d] + RD_on[r, d] * (Duration_d[d]) - tau[D_r[r], D_d[d]],
            )
            for (r, d) in RD
            if d not in set(D_flex_OD) | {d_p}
        }

    else:
        drop2 = {
            (r, d): np.minimum(LA_r[r], LA_d[d] - tau[D_r[r], D_d[d]])
            for (r, d) in RD
            if d not in set(D_flex_OD) | {d_p}
        }  # no dummy driver, 必须在此之前drop rider才能使driver按时返回O_d
    drop2.update({(r, d): LA_r[r] for (r, d) in RD if d in set(D_flex_OD)})

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
        for d in set(D) - set(D_flex_OD) - {d_p}:
            Gs = list(np.where(tau[O_d[d], :] + tau[:, D_d[d]] <= TW_d[d])[0])
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
                        MN_d.setdefault(d, set()).update(T_d[d][ind] * S + s)
                else:
                    if s == O_d[d]:
                        SN_d[d] = T_d[d][ind] * S + s
                        EN_d[d] = T_d[d][ind] * S + s
                    else:
                        MN_d.setdefault(d, set()).update(T_d[d][ind] * S + s)

                N_d.setdefault(d, set()).update(T_d[d][ind] * S + s)

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
        # Node set for the flex OD drivers
        MN_d.update(
            {
                d: set(
                    node
                    for node in set(TN.nodes())
                    if node // S < TW_d[d] and node // S > (Tmax - TW_d[d])
                )
                for d in set(D_flex_OD)
            }
        )
        # End node set for the flex OD drivers
        EN_d.update(
            {
                d: set(node for node in set(TN.nodes()) if node // S >= TW_d[d])
                for d in set(D_flex_OD)
            }
        )
        # Start node set for the flex OD drivers
        SN_d.update(
            {
                d: set(
                    node for node in set(TN.nodes()) if node // S <= (Tmax - TW_d[d])
                )
                for d in set(D_flex_OD)
            }
        )

    else:
        for d in set(D) - set(D_flex_OD) - {d_p}:
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
                            MN_d.setdefault(d, set()).update(
                                (T_d[d][ind] + m * (Duration_d[d])) * S + s
                            )
                    else:
                        if s == O_d[d] and m == 0:
                            SN_d[d] = (
                                T_d[d][ind] * S + s
                            )  # need to be the first origin node
                            if LA_d[d] in T_d[d][ind]:
                                SN_d[d] = SN_d[d][SN_d[d] != LA_d[d] * S + O_d[d]]
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
                            MN_d.setdefault(d, set()).update(
                                (T_d[d][ind] + m * (Duration_d[d])) * S + s
                            )

                    N_d.setdefault(d, set()).update(
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
        # Node set for the flex OD drivers
        MN_d.update(
            {
                d: set(
                    node
                    for node in set(TN.nodes())
                    if node // S < (num_tour[d] * TW_d[d])
                    and node // S > (Tmax - num_tour[d] * TW_d[d])
                )
                for d in set(D_flex_OD)
            }
        )
        # End node set for the flex OD drivers
        EN_d.update(
            {
                d: set(
                    node
                    for node in set(TN.nodes())
                    if node // S >= (num_tour[d] * TW_d[d])
                )
                for d in set(D_flex_OD)
            }
        )
        # Start node set for the flex OD drivers
        SN_d.update(
            {
                d: set(
                    node
                    for node in set(TN.nodes())
                    if node // S <= (Tmax - num_tour[d] * TW_d[d])
                )
                for d in set(D_flex_OD)
            }
        )
    # Link set for the dummy driver
    L_d[d_p] = set((t * S + s, (t + 1) * S + s) for t in range(T) for s in range(S))
    # Link set for the flex OD drivers
    L_d.update({d: set(TN.edges()) for d in set(D_flex_OD)})

    # Link set for all the pairs of the dummy driver and the rider
    # and link set for all pairs of the flex OD drivers and the rider
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
        )
        for d in set(D_flex_OD):
            if (r, d) in RD:
                nodes = nx.dag.descendants(
                    TN, pick[r, d] * S + O_r[r]
                ) & nx.dag.ancestors(TN, drop2[r, d] * S + D_r[r])
                nodes.add(pick[r, d] * S + O_r[r])
                nodes.add(drop2[r, d] * S + D_r[r])
                TN_r = TN.subgraph(nodes)

                L_rd[r, d] = set(TN_r.edges())
                nodes = nodes - SN_r[r]
                N_rd[r, d] = nodes - EN_r[r]

    # Node set for each rider, including the waiting nodes, not SNs, and ENs
    RN_r = dict()
    for r in R:
        RN_r.setdefault(r, set())
        for d in D:
            if (r, d) in L_rd:
                for (n1, n2) in L_rd[r, d]:
                    RN_r[r].update({n1, n2})

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

    DN_d = [(d, n) for d in set(D) - {d_p} for n in MN_d[d]]  # not include dummy driver
    DR_r = {r: set(d for d in D if (r, d) in RD) for r in R}  # include dummy driver

    # fixed-route setting
    # fixed_route: dict with key as fixed route driver id,
    # value as array of time (first column) and stop (second column)
    if FIXED_ROUTE and fixed_route_D != None:
        FR_d = dict()
        for d in fixed_route_D:
            FR_d[d] = list()

        for d in fixed_route_D:
            for i in range(len(fixed_route_D[d]) - 1):
                for j in range(num_tour[d]):
                    FR_d[d].append(
                        (
                            (fixed_route_D[d][i][0] + j * (Duration_d[d])) * S
                            + fixed_route_D[d][i][1],
                            (fixed_route_D[d][i + 1][0] + j * (Duration_d[d])) * S
                            + fixed_route_D[d][i + 1][1],
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
    x = m.addVars(DL_d, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    y = m.addVars(RDL_rd, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    u = m.addVars(RD, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z = m.addVars(R, vtype=GRB.BINARY)

    if FIXED_ROUTE and fixed_route_D != None:
        for (d, n1, n2) in FRL_d:
            x[d, n1, n2].lb = 1
            x[d, n1, n2].ub = 1
        # for (d, n1, n2) in DL_d:
        #     if (d, n1, n2) not in FRL_d:
        #         x[d, n1, n2].lb = 0
        #         x[d, n1, n2].ub = 0
        m.update()

    if start and all(e for e in start):
        x_init, y_init, u_init, z_init = start
        for e in DL_d:
            x[e].start = 0
        for e in x_init:
            x[e].start = 1

        for e in RDL_rd:
            y[e].start = 0
        for e in y_init:
            y[e].start = 1

        for e in RD:
            u[e].start = 0
        for e in u_init:
            u[e].start = 1

        for e in R:
            z[e].start = 0
        for e in z_init:
            z[e].start = 1

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
            for d in set(D) - {d_p} - set(D_flex_OD)
        ),
        "source_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in SN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in SN_d[d])
            == 1
            for d in set(D_flex_OD)
        ),
        "flex_od_source_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in EN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in EN_d[d])
            == 1
            for d in set(D) - {d_p} - set(D_flex_OD)
        ),
        "dest_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in EN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in EN_d[d])
            == 1
            for d in set(D_flex_OD)
        ),
        "flex_od_dest_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 == n)
            - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 == n)
            == 0
            for (d, n) in DN_d  # DN_d not include dummy driver
            if d != d_p  # (just be safe, seems redundant)
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

    m.params.TimeLimit = TL
    m.params.MIPGap = MIP_GAP
    m.params.Method = 0
    m.optimize()

    if m.status == GRB.TIME_LIMIT:
        X = {e for e in DL_d if x[e].x > 0.001}
        U = {e for e in RD if u[e].x > 0.001}
        Y = {e for e in RDL_rd if y[e].x > 0.001}
        R_match = {r for r in R if z[r].x > 0.001}
        mr = sum(z[r].x for r in R) / len(R)

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
        mr = sum(z[r].x for r in R) / len(R)

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
        m.write(r"many2many_output\model.ilp")
        raise ValueError("Infeasible solution!")
    else:
        print("Status code: {}".format(m.status))
    return (X, U, Y, Route_D, mr, m.ObjVal, R_match)


# %%
if __name__ == "__main__":
    import yaml
    import pickle
    import os

    from trip_prediction import *
    from utils import *

    with open("config_m2m_gc.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)
    if not os.path.exists(config["figure_pth"]):
        os.makedirs(config["figure_pth"])

    result_loc = config["m2m_output_loc"]
    if not os.path.exists(result_loc):
        os.makedirs(result_loc)

    config["BUILD_ON"] = False
    config["FIXED_ROUTE"] = False
    config["PENALIZE_RATIO"] = False
    config["driver_set"] = [0]
    config["S"] = 39
    config["T"] = 41
    fraction = config["T"] / 300
    config["m2m_data_loc"] = "many2many_data\\no_FR_1_ATT_SPTT_test\\"
    config["m2m_output_loc"] = "many2many_output\\results\\no_FR_1_ATT_SPTT_test\\"
    config["figure_pth"] = "many2many_output\\figure\\no_FR_1_ATT_SPTT_test\\"
    config["flex_od_driver"] = [0]
    # Load data
    DELTA_t = config["DELTA_t"]
    # Load neighbor nodes information
    ctr = load_neighbor(config)
    # Load shortest travel time matrices
    tau, tau2 = load_tau(config)

    # Load mode choice input data
    (
        per_time,
        per_dist,
        per_emp,
        mdot_dat,
        dests_geo,
        D,
    ) = load_mc_input(config)
    # load ID converter
    id_converter = pickle.load(open(config["id_converter"], "rb"))
    id_converter_reverse = pickle.load(open(config["id_converter_reverse"], "rb"))
    O_p = [
        id_converter_reverse[o] if o in id_converter_reverse else 9999
        for o in mdot_dat.geoid_o
    ]

    # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
    D_p = {}  # dict of destination for each transportation mode
    D_p[2] = [
        id_converter_reverse[d] if d in id_converter_reverse else 9999
        for d in dests_geo.geoid3
    ]
    # Generate driver information
    # Load fixed routed infomation
    FR = load_FR(config)
    Route_fr_disagg = load_FR_disagg(config)
    Driver = get_driver(config)
    (
        trips_dict_pk,
        trips_dict_op,
        transit_trips_dict_pk,
        transit_trips_dict_op,
    ) = trip_prediction(
        id_converter, per_dist, per_emp, mdot_dat, dests_geo, D, per_time=per_time
    )
    trip_dict = disagg_2_agg_trip(transit_trips_dict_pk, config, fraction=fraction)
    Rider = get_rider(trip_dict, config)
    (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
        Rider, Driver, tau, tau2, ctr, config, fixed_route_D=FR
    )
    print("optimization finished, matching rate: {}%".format(mr * 100))
    # Post-processing: disaggregate Route_D
    Route_D_disagg = post_processing(Route_D, config)
    if config["FIXED_ROUTE"]:
        # Update route with the fixed route schedule
        Route_D_disagg.update(Route_fr_disagg)
    try:
        # Plot metrics
        plot_metric(Y, np.size(Rider, 0), config, 0)
    except:
        print("Warning: plot metric failed!")

    Y = [
        (
            r,
            d,
            n1 // config["S"],
            n2 // config["S"],
            n1 % config["S"],
            n2 % config["S"],
            n1,
            n2,
        )
        for (r, d, n1, n2) in Y
    ]
    Y = sorted(Y, key=operator.itemgetter(0, 2))

    U = sorted(list(U), key=operator.itemgetter(1, 0))

    for d in Route_D_disagg.keys():

        route_filename = result_loc + r"route_{}_disagg.csv".format(d)
        with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
            for row in Route_D_disagg[d]:
                csv_out.writerow(row)

    for d in Route_D.keys():

        route_filename = result_loc + r"route_{}_agg.csv".format(d)
        with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
            for row in Route_D[d]:
                csv_out.writerow(row)

    RDL_filename = result_loc + r"Y_rdl_agg.csv"
    with open(RDL_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["r", "d", "t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Y:
            csv_out.writerow(row)
        csv_out.writerow(
            [
                "Matching Rate: {}".format(mr),
            ]
        )

    RD_filename = result_loc + r"U_rd_agg.csv"
    with open(RD_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["r", "d"])
        for row in U:
            csv_out.writerow(row)
        csv_out.writerow(
            [
                "Matching Rate: {}".format(mr),
            ]
        )

    R_match = pd.DataFrame(data=sorted(list(R_match)), columns=["r"])
    R_match_filename = result_loc + r"R_matc_agg.csv"
    R_match.to_csv(R_match_filename, index=False)

    pickle.dump(
        transit_trips_dict_pk,
        open(config["m2m_data_loc"] + "transit_trips_dict_pk.p", "wb"),
    )
    try:
        # plot travel demand
        travel_demand_plot(transit_trips_dict_pk, config)
        # plot routes
        route_plot(Route_D_disagg, config)
    except:
        print("Warning: plot travel demand and/or routes failed!")

    # Record config info
    with open(result_loc + "config.yaml", "w") as fwrite:
        yaml.dump(config, fwrite)
    print("Finished!")
