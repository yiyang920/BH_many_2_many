"""
Disaggregate solution from many-to-many problem in a coarse graph.

Input: matched riders info, disaggregated transit routes.
"""
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict


def disagg_sol(
    Rider: np.ndarry,
    U_rd: dict,
    route_d: dict,
    tau2_disagg: np.ndarray,
    ctr_disagg,
    config: dict,
):
    """
    Rider: rider info in disaggregated network
    """
    S = config["S_disagg"]
    T = config["T"]
    Tmin, Tmax = min(station[0][0] for station in route_d.values), max(
        station[-1][1] for station in route_d.values
    )
    # Time expanded network
    # Every node in time-expaned network is defined as t_i*S+s_i
    # which maps (t_i,s_i) into a unique n_i value
    TN = nx.DiGraph()
    for t in range(Tmin, Tmax + 1):
        for i in range(S):
            for j in ctr_disagg[i]:
                if t + tau2_disagg[i, j] <= T:
                    TN.add_edge(t * S + i, (t + tau2_disagg[i, j]) * S + j)
    # Rider data
    R, O_r, D_r, ED_r, LA_r, SP_r_agg, TW_r, T_r, V_r, SP_r = gp.multidict(
        {
            r: [
                Rider[i, 4],
                Rider[i, 5],
                Rider[i, 2],
                Rider[i, 3],
            ]
            for i, r in enumerate(Rider[:, 0])
            if r in set(r for r, _ in U_rd)
        }
    )
    # Driver data
    # D = set(route_d.keys())
    V = set(s1 for station in route_d.values() for _, _, s1, *_ in station) | set(
        s2 for station in route_d.values() for _, _, _, s2, *_ in station
    )

    RL = set(
        (r, int(t1 * S + s1), int(t2 * S + s2))
        for r, d in U_rd
        for (t1, t2, s1, s2, *_) in route_d[d]
    )
    RL.update(
        set(r, t * S + s, (t + 1) * S + s) for t in range(Tmax) for s in V for r in R
    )
    RL_r = {rider: [(n1, n2) for r, n1, n2 in RL if r == rider] for rider in R}

    SN_r = {
        r: set((ED_r[r] + m) * S + O_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    EN_r = {
        r: set((LA_r[r] - m) * S + D_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    # Node set for each rider, including the waiting nodes, not SNs, and ENs
    RN_r = defaultdict(set)
    for r, n1, n2 in RL:
        RN_r[r].update({n1, n2})
    for r in R:
        RN_r[r] = RN_r[r] - SN_r[r]
        RN_r[r] = RN_r[r] - EN_r[r]
    ### Variables ###
    m = gp.Model("many2many")
    y = m.addVars(RL, vtype=GRB.BINARY)
    m.setObjective(1, GRB.MAXIMIZE)
    ### Constraints ###
    m.addConstrs(
        (
            gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n1 in SN_r[r])
            - gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n2 in SN_r[r])
            == 1
            for r in R
        ),
        "source_r",
    )

    m.addConstrs(
        (
            gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n2 in EN_r[r])
            - gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n1 in EN_r[r])
            == 1
            for r in R
        ),
        "dest_r",
    )

    m.addConstrs(
        (
            gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n1 == n)
            - gp.quicksum(y[r, n1, n2] for n1, n2 in RL_r[r] if n2 == n)
            == 0
            for r in R
            for n in RN_r[r]
        ),
        "trans_r",
    )
