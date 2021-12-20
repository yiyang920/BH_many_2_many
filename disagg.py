"""
Disaggregate solution from many-to-many problem in a coarse graph.

Input: matched riders info, disaggregated transit routes.
"""
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB


def disagg_sol(
    Rider: np.ndarry,
    U_rd: dict,
    route_d: dict,
    tau_disagg: np.ndarray,
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
                tau_disagg[Rider[i, 4], Rider[i, 5]],
                Rider[i, 3] - Rider[i, 2],
                np.arange(Rider[i, 2], Rider[i, 3] + 1),
                Rider[i, 6],
                Rider[i, 7],
            ]
            for i, r in enumerate(Rider[:, 0])
        }
    )
    RDL = gp.tuplelist(
        (r, d, int(t1 * S + s1), int(t2 * S + s2))
        for r, d in U_rd
        for (t1, t2, s1, s2, *_) in route_d[d]
    )

    ### Variables ###
    m = gp.Model("many2many")
    y = m.addVars(RDL, vtype=GRB.BINARY)
    m.setObjective(1, GRB.MAXIMIZE)
    ### Constraints ###
