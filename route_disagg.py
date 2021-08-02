# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101

# Packages
import numpy as np
import networkx as nx
import itertools
import gurobipy as gp
from gurobipy import GRB


def route_disagg(
    Route_D_agg: dict,
    agg_2_disagg_id: dict,
    disagg_2_agg_id: dict,
    OD_d: dict,
    tau: "np.ndarray[np.int64]",
    tau2: "np.ndarray[np.int64]",
    Driver: "np.ndarray[np.int64]",
    ctr: dict,
    config: dict,
    fixed_route_D: dict = None,
) -> tuple:
    """
    Input:
    Route_D_agg := aggregated route for each driver, list of tuple (t1, t2, s1, s2, n1, n2)
    OD_d := dictionary of OD for each driver, {d: {"O": o, "D": d}}
    tau :=  zone-to-zone travel time of disaggregated network, numpy array
    tau2 := zone-to-zone travel time of disaggregated network
            with diagonal elements equal to one, numpy array
    ctr :=  a dictionary with key as zones and values be the set of neighbor
            zones for the key including the key itself (disaggregated network)
    """
    TL = config["TIME_LIMIT"]
    MIP_GAP = config["MIP_GAP"]
    FIXED_ROUTE = config["FIXED_ROUTE"]
    # number of disagg zones
    S = config["S_disagg"]
    # number of drivers
    D_len = len(Driver)
    # Time horizon of one tour of all drivers
    T = config["T_post_processing"]

    # Driver sets
    D = set(Route_D_agg.keys())
    # Time window for each driver. If repeated tour mode,
    # TW_d should be the time window for the first single tour
    TW_d = {d: Driver[i, 3] - Driver[i, 2] for i, d in enumerate(Driver[:, 0])}

    # Node set for each driver
    S_d = {
        d: set(
            s
            for station in route
            if station[1] <= TW_d[d]
            for s in agg_2_disagg_id[station[2]]
        )
        for d, route in Route_D_agg.items()
    }

    for d in D:
        # add destination station of each route
        S_d[d].update(agg_2_disagg_id[Route_D_agg[d][TW_d[d]][3]])

    # S_info is a dictionary with structure {s: {d: [t1, t2,...]}}
    S_info, Route_D_agg_2, S_d_tilda = dict(), dict(), dict()
    for d, route in Route_D_agg.items():
        for i, s in enumerate(route):
            # each driver only register its first tour info
            if s[1] <= TW_d[d]:
                # if (s[2] not in S_info) or (d not in S_info[s[2]]):
                S_info.setdefault(s[2], dict()).setdefault(d, list()).append(s[0])
                # encoding each node to the form of virtual node n_d^p,
                # where n_d^p = (p * S + s) * D + d
                Route_D_agg_2[i] = (S_info[s[2]][d].index(s[0]) * S + s[2]) * D + d
                # update virtual node set for each driver
                S_d_tilda[d].add(Route_D_agg_2[i])

    # Set of hubs and their visted set of drivers, by default the disaggregated hub is
    # the first subnode of each supernode
    # TODO: H_D for virtual nodes
    H_D = {
        agg_2_disagg_id[s][0]: set(d_info.keys())
        for s, d_info in S_info.items()
        if len(d_info) > 1  # if more than 1 driver visit node s
    }
    # Set of transfer hubs
    H = set(H_D)
    # Driver pairs, if d arrives no later than d' at hub h
    D_h = dict()
    for h, driver_set in H_D.items():
        d_combin = itertools.permutations(driver_set, 2)
        for (d1, d2) in d_combin:
            if (
                d1 != d2
                and S_info[disagg_2_agg_id[h]][d1] <= S_info[disagg_2_agg_id[h]][d2]
            ):
                D_h.setdefault(h, set()).add((d1, d2))

    # Time expanded network
    TN = nx.DiGraph()
    for t in range(0, T + 1):
        for i in range(S):
            for j in ctr[i]:
                if t + tau2[i, j] <= T:
                    # Every node in time-expaned network is defined as t_i * S + s_i
                    # which maps (t_i,s_i) into a unique n_i value
                    TN.add_edge(t * S + i, (t + tau2[i, j]) * S + j)

    L_d, C_dl, N_d, SN_d, EN_d, MN_d, HN_d = (
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
    )
    T_d = np.arange(T)
    # Generate link set for each driver
    for d in D:
        for s in S_d[d]:
            if OD_d[d]["O"] != OD_d[d]["D"]:
                if s == OD_d[d]["O"]:
                    SN_d[d] = T_d * S + s
                elif s == OD_d[d]["D"]:
                    EN_d[d] = T_d * S + s
                else:
                    MN_d.setdefault(d, set()).update(T_d * S + s)
            else:
                if s == OD_d[d]["O"]:
                    SN_d[d] = T_d * S + s
                    EN_d[d] = T_d * S + s
                else:
                    MN_d.setdefault(d, set()).update(T_d * S + s)
            if s in H:
                HN_d.setdefault(d, dict()).setdefault(s, set()).update(
                    T_d * S + s
                )  # node set of hubs

            N_d.setdefault(d, set()).update(T_d * S + s)
        TN_d = TN.subgraph(N_d[d])
        L_d[d] = set(TN_d.edges())
        C_dl.update({(d, n1, n2): n2 // S - n1 // S for (n1, n2) in L_d[d]})

    # Driver link set
    DL_d = set(
        (d, n1, n2)
        for driver, link in L_d.items()
        for (d, (n1, n2)) in itertools.product([driver], link)
    )
    # Driver node set without SN and EN
    DN_d = set((d, n) for d in D for n in MN_d[d])
    # # Driver node set with SN and EN
    # DN_d_full = set((d, n) for d in D for n in N_d[d])

    # fixed-route setting
    # fixed_route: dict with key as fixed route driver id,
    # value as array of time (first column) and stop (second column)
    if FIXED_ROUTE and fixed_route_D != None:
        FR_d = dict()
        for d in fixed_route_D:
            for i in range(len(fixed_route_D[d]) - 1):
                FR_d.setdefault(d, list()).append(
                    (
                        (fixed_route_D[d][i][0]) * S + fixed_route_D[d][i][1],
                        (fixed_route_D[d][i + 1][0]) * S + fixed_route_D[d][i + 1][1],
                    )
                )
        FRL_d = set(
            (d, n1, n2)
            for key, val in FR_d.items()
            for (d, (n1, n2)) in itertools.product([key], val)
        )

        DL_d.update(FRL_d)

    m = gp.Model("route_disagg")

    ### Variables ###
    x = m.addVars(DL_d, vtype=GRB.BINARY, name="x")

    if FIXED_ROUTE and fixed_route_D != None:
        for (d, n1, n2) in FRL_d:
            x[d, n1, n2].lb = 1
            x[d, n1, n2].ub = 1
        m.update()

    ### Objective ###
    m.setObjective(x.prod(C_dl), GRB.MINIMIZE)

    ### Constraints ###
    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in SN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in SN_d[d])
            == 1
            for d in D
        ),
        "source_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 in EN_d[d])
            # - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 in EN_d[d])
            == 1
            for d in D
        ),
        "dest_d",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 % S == s) >= 1
            for s in S_d[d] - set((OD_d[d]["O"], OD_d[d]["D"]))
        ),
        "visit_node_once",
    )

    m.addConstrs(
        (
            gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n2 == n)
            - gp.quicksum(x[d, n1, n2] for (n1, n2) in L_d[d] if n1 == n)
            == 0
            for (d, n) in DN_d  # no SN and EN
        ),
        "trans_d",
    )

    m.addConstrs(
        (
            gp.quicksum(
                x[d1, n1, n2]
                for (n1, n2) in L_d[d1]
                if n2 in HN_d[d1][h] - set(SN_d[d1]) - set(EN_d[d1]) and n2 // S <= u
            )
            - gp.quicksum(
                x[d2, n1, n2]
                for (n1, n2) in L_d[d2]
                if n2 in HN_d[d2][h] - set(SN_d[d2]) - set(EN_d[d2]) and n2 // S <= u
            )
            >= 0
            for h in H
            for (d1, d2) in D_h[h]
            if h not in set((OD_d[d1]["O"], OD_d[d1]["D"]))
            for u in range(0, T + 1)
        ),
        "hub_visit_d",
    )

    # m.addConstrs(
    #     (y[d, n1] + y[d, n2] >= 2 * x[d, n1, n2] for (d, n1, n2) in DL_d),
    #     "node_link_d_1",
    # )
    # m.addConstrs(
    #     (y[d, n1] + y[d, n2] <= 1 + x[d, n1, n2] for (d, n1, n2) in DL_d),
    #     "node_link_d_2",
    # )

    # m.addConstrs(
    #     (
    #         gp.quicksum(
    #             y[d, t * S + s] for t in range(0, T + 1) if (d, t * S + s) in DN_d_full
    #         )
    #         == 1
    #         for d in D
    #         for s in S_d[d] - set((OD_d[d]["O"], OD_d[d]["D"]))
    #     ),
    #     "node_visit_once_d",
    # )

    m.params.TimeLimit = TL
    m.params.MIPGap = MIP_GAP
    m.optimize()

    if m.status == GRB.TIME_LIMIT:
        X = {e for e in DL_d if x[e].x > 0.001}
        # Y = {e for e in DN_d_full if y[e].x > 0.001}

        Route_D = dict()
        for d in D:
            for (n1, n2) in L_d[d]:
                if np.round(x[d, n1, n2].x) == 1:
                    t1 = n1 // S
                    s1 = n1 % S
                    t2 = n2 // S
                    s2 = n2 % S
                    Route_D.setdefault(d, list()).append((t1, t2, s1, s2, n1, n2))
            Route_D[d] = sorted(list(Route_D[d]), key=lambda x: x[0])

            duration_d = int(Route_D[d][-1][1])
            num_tour_d = int(config["T"] // duration_d)

            for i in range(1, num_tour_d):
                Route_D[d] += [
                    (
                        t1 + duration_d * i,
                        t2 + duration_d * i,
                        s1,
                        s2,
                        (t1 + duration_d * i) * S + s1,
                        (t2 + duration_d * i) * S + s2,
                    )
                    for (t1, t2, s1, s2, *_) in Route_D[d]
                ]

    elif m.status == GRB.OPTIMAL:
        X = {e for e in DL_d if x[e].x > 0.001}
        # Y = {e for e in DN_d_full if y[e].x > 0.001}

        Route_D = dict()
        for d in D:
            for (n1, n2) in L_d[d]:
                if np.round(x[d, n1, n2].x) == 1:
                    t1 = n1 // S
                    s1 = n1 % S
                    t2 = n2 // S
                    s2 = n2 % S
                    Route_D.setdefault(d, list()).append((t1, t2, s1, s2, n1, n2))
            Route_D[d] = sorted(list(Route_D[d]), key=lambda x: x[0])

            duration_d = int(Route_D[d][-1][1])
            num_tour_d = int(config["T"] // duration_d)

            for i in range(1, num_tour_d):
                Route_D[d] += [
                    (
                        t1 + duration_d * i,
                        t2 + duration_d * i,
                        s1,
                        s2,
                        (t1 + duration_d * i) * S + s1,
                        (t2 + duration_d * i) * S + s2,
                    )
                    for (t1, t2, s1, s2, *_) in Route_D[d]
                    if t2 <= duration_d
                ]

    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(r"many2many_output\model.ilp")
        raise ValueError("Infeasible solution!")
    else:
        print("Status code: {}".format(m.status))
    return (X, Route_D, m.ObjVal)


if __name__ == "__main__":
    import yaml
    import pickle
    import os
    import csv
    import pandas as pd
    from utils import load_tau_disagg, load_neighbor_disagg, load_FR_m2m_gc

    test_driver_set = {
        0,
        1,
        2,
        3,
    }
    with open("config_m2m_gc.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)

    config["T_post_processing"] = 70
    config["m2m_output_loc"] = r"many2many_output\\results\\route_disagg_test\\"
    config["m2m_data_loc"] = r"many2many_data\\FR_and_1_ATT_SPTT_GC\\"
    config["figure_pth"] = r"many2many_output\\figure\\route_disagg_test\\"

    if not os.path.exists(config["figure_pth"]):
        os.makedirs(config["figure_pth"])

    result_loc = config["m2m_output_loc"]
    if not os.path.exists(result_loc):
        os.makedirs(result_loc)

    Route_D = pickle.load(open("Data\\debug\\Route_D.p", "rb"))
    for d in Route_D.keys():
        route_filename = result_loc + r"route_{}_agg.csv".format(d)
        with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
            for row in Route_D[d]:
                csv_out.writerow(row)
    agg_2_disagg_id = pickle.load(open("Data\\debug\\agg_2_disagg_id.p", "rb"))
    disagg_2_agg_id = {
        n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
    }
    config["S"] = len(agg_2_disagg_id)

    OD_d = pd.read_csv(
        config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
    ).to_dict("index")

    # Load fixed routed infomation
    _, FR, _ = load_FR_m2m_gc(
        agg_2_disagg_id, disagg_2_agg_id, config
    )  # aggregated routes
    # FR = {
    #     d: np.array([[s[0], agg_2_disagg_id[s[1]][0]] for s in route])
    #     for d, route in FR.items()
    #     if d in test_driver_set
    # }
    Route_D = {d: route for d, route in Route_D.items() if d in test_driver_set}
    tau, tau2 = load_tau_disagg(config)
    ctr = load_neighbor_disagg(config)

    X, Route_D, ObjVal = route_disagg(
        Route_D,
        agg_2_disagg_id,
        disagg_2_agg_id,
        OD_d,
        tau,
        tau2,
        ctr,
        config,
        fixed_route_D=FR,
    )
    for d in Route_D.keys():
        route_filename = result_loc + r"route_{}_disagg.csv".format(d)
        with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
            for row in Route_D[d]:
                csv_out.writerow(row)
