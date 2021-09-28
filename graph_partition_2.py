# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101
# %% Packages
import numpy as np
import itertools
import copy
import gurobipy as gp
from gurobipy import GRB


def graph_coarsening(V, D_l, E, K, config):
    """
    Input:
    V -- node set
    D_l -- demand matrix in dictionary
    E -- edge set, i.e. {(n1, n2)}
    K -- number of partitions
    config -- config of gurobipy, dict type

    Output:
    X -- {(v, k)} equal to 1 if node v belongs to partition k
    Z -- {(v, u, k)} equal to 1 if pair (v, u) belongs to partition k
    W -- {(v, u, k)} equal to 1 if edge (v, u) belongs to partition k
    P_V -- {k: [v]} dict with key as partition ID and value as list of nodes in the partition
    P_L -- {k: [(n1, n2)]} dict with key as partition ID and value as list of links in the partition
    m.ObjVal -- objective value
    """

    VK = set((v, k) for (v, k) in itertools.product(list(V), range(K)))
    EK = set((u, v, k) for ((u, v), k) in itertools.product(list(E), range(K)))
    LK = set((u, v, k) for (u, v, k) in itertools.product(list(V), list(V), range(K)))

    AV = {v: sum(t for (_, u), t in D_l.items() if u == v) for v in V}
    GV = {v: sum(t for (u, _), t in D_l.items() if u == v) for v in V}

    m = gp.Model("graph_coarsening")
    ### Variables ###
    x = m.addVars(VK, vtype=GRB.BINARY)
    # y = m.addVars(EK, vtype=GRB.BINARY)
    z = m.addVars(LK, vtype=GRB.BINARY)
    w = m.addVars(EK, vtype=GRB.BINARY)
    ### Model Objective ###
    # m.setObjective(
    #     gp.quicksum(
    #         gp.quicksum(
    #             C_l[n1, n2] * y[n1, n2, kk] for (n1, n2, kk) in LK_lk if kk == k
    #         )
    #         - z[k]
    #         for k in K_k
    #     ),
    #     GRB.MAXIMIZE,
    # )
    m.setObjective(
        gp.quicksum((AV[v] + GV[v]) * x[v, k] for (v, k) in VK)
        + gp.quicksum(D_l[u, v] * z[u, v, k] for (u, v, k) in LK),
        GRB.MINIMIZE,
    )

    ### Model Constraints ###
    m.addConstrs(
        (gp.quicksum(x[v, k] for k in range(K)) == 1 for v in V),
        "node_to_partition",
    )

    m.addConstrs(
        (x[u, k] + x[v, k] >= 2 * z[u, v, k] for (u, v, k) in LK),
        "node_link_1",
    )

    m.addConstrs(
        (x[u, k] + x[v, k] <= 1 + z[u, v, k] for (u, v, k) in LK),
        "node_link_2",
    )

    m.addConstrs(
        (x[u, k] + x[v, k] >= 2 * w[u, v, k] for (u, v, k) in EK),
        "node_link_3",
    )

    m.addConstrs(
        (
            gp.quicksum(w[u, v, k] for (u, v) in E)
            == gp.quicksum(x[v, k] for v in V) - 1
            for k in range(K)
        ),
        "spanning_tree_1",
    )

    m.addConstrs(
        (
            gp.quicksum(w[u, vv, k] for (u, vv) in E if vv == v)
            + gp.quicksum(w[vv, u, k] for (vv, u) in E if vv == v)
            - x[v, k]
            >= 0
            for k in range(K)
            for v in V
        ),
        "spanning_tree_2",
    )

    ### Gurobipy Parameters ###
    m.params.TimeLimit = config["TIME_LIMIT_GC"]
    m.params.MIPGap = config["MIP_GAP_GC"]

    m.optimize()

    # extract solutions
    if m.status == GRB.TIME_LIMIT:
        X = {(v, k) for (v, k) in VK if x[v, k].x > 0.001}
        Z = {(u, v, k) for (u, v, k) in LK if z[u, v, k].x > 0.001}

        P_N, P_L = dict(), dict()

        for (u, k) in X:
            _ = P_N.setdefault(k, []).append(u)

        for (u, v, k) in Z:
            _ = P_L.setdefault(k, []).append((u, v))

    elif m.status == GRB.OPTIMAL:
        X = {(v, k) for (v, k) in VK if x[v, k].x > 0.001}
        Z = {(u, v, k) for (u, v, k) in LK if z[u, v, k].x > 0.001}

        P_N, P_L = dict(), dict()

        for (u, k) in X:
            _ = P_N.setdefault(k, []).append(u)

        for (u, v, k) in Z:
            _ = P_L.setdefault(k, []).append((u, v))

    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(r"graph_coarsening_output\model.ilp")
        raise ValueError("Infeasible solution!")

    else:
        print("Status code: {}".format(m.status))

    return (X, Z, P_N, P_L, m.ObjVal)


if __name__ == "__main__":
    import yaml
    import pickle
    import os
    import networkx as nx
    from trip_prediction import trip_prediction
    from utils import (
        load_mc_input,
        load_neighbor_disagg,
        get_link_set_disagg,
        get_link_cost,
        post_processing,
        network_plot,
        update_tau_agg,
        load_tau_disagg,
        update_ctr_agg,
        travel_demand_plot,
    )

    with open("config_m2m_gc.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)
    config["figure_pth"] = r"many2many_output\\figure\\graph_coarsening_test\\"
    if not os.path.exists(config["figure_pth"]):
        os.makedirs(config["figure_pth"])

    ctr = load_neighbor_disagg(config)  # disagg station info
    # config["S"] = 39
    tau_disagg, tau2_disagg = load_tau_disagg(config)

    # Route_D_agg = pickle.load(open(r"Data\temp\Route_D.p", "rb"))

    # Route_D_disagg = post_processing(Route_D_agg, config)

    # Route_D_disagg = {k: v for k, v in Route_D_disagg.items() if k == 0}

    N, K = config["S_disagg"], config["K"]
    V = set(np.arange(N))

    # load ID converter
    id_converter = pickle.load(open(config["id_converter"], "rb"))

    # Load mode choice input data
    (
        per_time,
        per_dist,
        per_emp,
        mdot_dat,
        dests_geo,
        D,
    ) = load_mc_input(config)

    (
        trips_dict_pk,  # {(i, j): n}
        trips_dict_op,
        transit_trips_dict_pk,
        transit_trips_dict_op,
    ) = trip_prediction(
        id_converter, per_dist, per_emp, mdot_dat, dests_geo, D, per_time=per_time
    )

    _, E = get_link_set_disagg(config)
    # (N_n, L_l, C_l, K_k, w_out, w_in, w_sum, w_f) = get_link_cost(
    #     {k: 0.6 * v for k, v in trips_dict_pk.items()},
    #     tau_disagg,
    #     {},
    #     # Route_D_disagg,
    #     config,
    # )

    (X, Z, P_N, P_L, OBJ) = graph_coarsening(V, transit_trips_dict_pk, E, K, config)
    print("number of partition: {}".format(len(P_N)))
    # check if every partition forms a valid subgraph of the orginal graph
    for p, part in P_N.items():
        if len(part) > 1 and not all(
            any(i in ctr[j] for j in part if i != j) for i in part
        ):
            print("Warning: Partition does not form a subgraph! {}: {}".format(p, part))
        # Update agg_2_disagg_id and disagg_2_agg_id
        agg_2_disagg_id = dict()
        # Mapping from origin aggregated zone id to new aggregated zone id
        # of bus-visited aggregated zones, should be 1-to-1 mapping
        agg_2_agg_new_bus = dict()
        for idx, c in enumerate(P_N.keys()):
            agg_2_disagg_id[idx] = P_N[c]
        # P_N excludes the zones with bus service, need to add those excluded zones
        # into agg_2_disagg_id

    agg_2_disagg_id_bus = {}
    for idx, (part, nodes) in enumerate(agg_2_disagg_id_bus.items()):
        new_part_id = len(P_N) + idx
        agg_2_disagg_id[new_part_id] = nodes
        agg_2_agg_new_bus[part] = new_part_id

    disagg_2_agg_id = {
        n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
    }
    ctr_agg = update_ctr_agg(ctr, disagg_2_agg_id)
    tau, tau2 = update_tau_agg(ctr_agg, tau_disagg, agg_2_disagg_id, config)
    network_plot(tau, ctr_agg, disagg_2_agg_id, config)
    # plot travel demand
    travel_demand_plot(transit_trips_dict_pk, config)
    print("testing finished!")
