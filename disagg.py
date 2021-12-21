"""
Disaggregate solution from many-to-many problem in a coarse graph.

Input: matched riders info, disaggregated transit routes.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict


def disagg_sol(
    Rider: np.ndarray,
    U_rd: dict,
    route_d: dict[int, list[tuple[int, int, int, int, int, int]]],
    config: dict,
) -> tuple[set[tuple[int, int, int, int, int, int]], dict]:
    """
    Rider: rider info in disaggregated network
    """
    TIME_LIMIT = config["TIME_LIMIT"]
    MIP_GAP = config["MIP_GAP"]
    PENALIZE_RATIO = config["PENALIZE_RATIO"]
    S = config["S_disagg"]
    Tmax = int(max([station[-1][1] for station in route_d.values()]))
    # Rider data
    R, O_r, D_r, ED_r, LA_r, SP_r = gp.multidict(
        {
            r: [
                Rider[i, 4],
                Rider[i, 5],
                Rider[i, 2],
                Rider[i, 3],
                Rider[i, 7],
            ]
            for i, r in enumerate(Rider[:, 0])
            if r in set(r for r, _ in U_rd)
        }
    )
    # Set of visited nodes by transits
    V = set(s1 for station in route_d.values() for _, _, s1, *_ in station) | set(
        s2 for station in route_d.values() for _, _, _, s2, *_ in station
    )

    RDL = set(
        (r, d, int(t1 * S + s1), int(t2 * S + s2))
        for r, d in U_rd
        for (t1, t2, s1, s2, *_) in route_d[d]
    )
    # Dummy driver id
    d_dummy = max(route_d.keys()) + 1
    RDL.update(
        set(
            (r, d_dummy, t * S + s, (t + 1) * S + s)
            for r in R
            for t in range(ED_r[r], Tmax + 1)
            for s in V
        )
    )
    RDL = gp.tuplelist(list(RDL))
    RDL_r = {rider: [(d, n1, n2) for r, d, n1, n2 in RDL if r == rider] for rider in R}

    SN_r = {
        r: set((ED_r[r] + m) * S + O_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    EN_r = {
        r: set((LA_r[r] - m) * S + D_r[r] for m in range(LA_r[r] - ED_r[r])) for r in R
    }
    # Node set for each rider, including the waiting nodes, not SNs, and ENs
    RN_r = defaultdict(set)
    for r, _, n1, n2 in RDL:
        RN_r[r].update({n1, n2})
    for r in R:
        RN_r[r] = RN_r[r] - SN_r[r]
        RN_r[r] = RN_r[r] - EN_r[r]

    ### Variables ###
    m = gp.Model("many2many")
    # y = m.addVars(RDL, vtype=GRB.BINARY)
    y = m.addVars(RDL, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    if PENALIZE_RATIO:
        LAMBDA = config["LAMBDA"]
        m.setObjective(
            gp.quicksum(
                LAMBDA
                / SP_r[r]
                * gp.quicksum(
                    (n2 // S - n1 // S) * y[r, d, n1, n2] for (d, n1, n2) in RDL_r[r]
                )
                for r in R
            ),
            GRB.MINIMIZE,
        )
    else:
        m.setObjective(1, GRB.MAXIMIZE)
    ### Constraints ###
    m.addConstrs(
        (
            gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n1 in SN_r[r])
            - gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n2 in SN_r[r])
            == 1
            for r in R
        ),
        "source_r",
    )

    m.addConstrs(
        (
            gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n2 in EN_r[r])
            - gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n1 in EN_r[r])
            == 1
            for r in R
        ),
        "dest_r",
    )

    m.addConstrs(
        (
            gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n1 == n)
            - gp.quicksum(y[r, d, n1, n2] for d, n1, n2 in RDL_r[r] if n2 == n)
            == 0
            for r in R
            for n in RN_r[r]
        ),
        "trans_r",
    )

    m.params.TimeLimit = TIME_LIMIT
    m.params.MIPGap = MIP_GAP
    m.optimize()

    if m.status == GRB.TIME_LIMIT or m.status == GRB.OPTIMAL:
        Y = {
            (r, d, n1 // S, n2 // S, n1 % S, n2 % S)
            for r, d, n1, n2 in RDL
            if y[r, d, n1, n2].x > 0.001
        }
    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(config["m2m_output_loc"] + r"model.ilp")
        raise ValueError("Infeasible solution!")
    else:
        print("Status code: {}".format(m.status))
    return (Y, m.ObjVal)


if __name__ == "__main__":
    import yaml, pickle
    import pandas as pd
    from direct_disagg import direct_disagg
    from utils import load_neighbor_disagg, load_tau_disagg

    with open("config_agg_disagg.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)
    config["m2m_data_loc"] = r"many2many_data\\disagg_test\\"
    config["m2m_output_loc"] = r"many2many_data\\disagg_test\\"
    # load existing results for debugging
    X = pickle.load(open(config["m2m_data_loc"] + r"X.p", "rb"))
    U = pickle.load(open(config["m2m_data_loc"] + r"U.p", "rb"))
    Y = pickle.load(open(config["m2m_data_loc"] + r"Y.p", "rb"))
    Route_D = pickle.load(open(config["m2m_data_loc"] + r"Route_D.p", "rb"))
    mr = pickle.load(open(config["m2m_data_loc"] + r"mr.p", "rb"))
    OBJ = pickle.load(open(config["m2m_data_loc"] + r"OBJ.p", "rb"))
    R_match = pickle.load(open(config["m2m_data_loc"] + r"R_match.p", "rb"))

    Driver = pickle.load(open(config["m2m_data_loc"] + r"Driver.p", "rb"))
    tau = pickle.load(open(config["m2m_data_loc"] + r"tau_agg.p", "rb"))
    agg_2_disagg_id = pickle.load(
        open(
            config["m2m_data_loc"] + r"agg_2_disagg_id_{}.p".format(1),
            "rb",
        ),
    )
    disagg_2_agg_id = pickle.load(
        open(
            config["m2m_data_loc"] + r"disagg_2_agg_id_{}.p".format(1),
            "rb",
        ),
    )
    # Load neighbor nodes information of disaggregated zones
    ctr_disagg = load_neighbor_disagg(config)
    # Load shortest travel time matrices of disaggregated zones
    tau_disagg, tau2_disagg = load_tau_disagg(config)
    OD_d = pd.read_csv(
        config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
    ).to_dict("index")
    Route_D = direct_disagg(
        Route_D,
        OD_d,
        Driver,
        tau_disagg,
        ctr_disagg,
        agg_2_disagg_id,
        disagg_2_agg_id,
        config,
    )
    Rider = np.loadtxt(
        config["m2m_data_loc"] + r"Rider_disagg.csv",
        skiprows=1,
        dtype=int,
        delimiter=",",
    )
    Y, obj = disagg_sol(Rider, U, Route_D, config)
    print(Y, obj)
