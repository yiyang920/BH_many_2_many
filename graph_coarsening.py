# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101
# %% Packages
import numpy as np
import itertools
import copy
import gurobipy as gp
from gurobipy import GRB


def graph_coarsening(N_n, W_n, L_l, C_l, K_k, config):
    """
    Input:
    N_n -- node set
    W_n -- node weight
    L_l -- link set, i.e. {(n1, n2)}
    C_l -- link cost, dict type with key as link, i.e. {(n1, n2): c}
    K_k -- set partitions
    config -- config of gurobipy, dict type

    Output:
    X -- {(n, k)} equal to 1 if node n belongs to partition k
    Y -- {(n1, n2, k)} equal to 1 if link (n1, n2) belongs to partition k
    P_N -- {k: [n]} dict with key as partition ID and value as list of nodes in the partition
    P_L -- {k:[(n1, n2)]} dict with key as partition ID and value as list of links in the partition
    m.ObjVal -- objective value
    """

    NK_nk = set((n, k) for (n, k) in itertools.product(list(N_n), list(K_k)))
    LK_lk = set(
        (n1, n2, k) for ((n1, n2), k) in itertools.product(list(L_l), list(K_k))
    )

    N = len(N_n)
    K = len(K_k)
    # soft thresholding of node weights
    soft_thred = 500
    W_n = {n: min(w, soft_thred) for n, w in W_n.items()}
    W_sum = sum(W_n.values())

    m = gp.Model("graph_coarsening")
    ### Variables ###
    x = m.addVars(NK_nk, vtype=GRB.BINARY, name="x")
    y = m.addVars(LK_lk, vtype=GRB.BINARY, name="y")
    z = m.addVars(K_k, vtype=GRB.BINARY, name="z")
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
        gp.quicksum(
            gp.quicksum(
                C_l[n1, n2] * y[n1, n2, kk] for (n1, n2, kk) in LK_lk if kk == k
            )
            + config["THETA"] * z[k]
            for k in K_k
        ),
        GRB.MAXIMIZE,
    )

    ### Model Constraints ###
    m.addConstrs(
        (gp.quicksum(x[n, k] for k in K_k) == 1 for n in N_n),
        "node_to_partition",
    )

    m.addConstrs(
        (x[n1, k] + x[n2, k] >= 2 * y[n1, n2, k] for (n1, n2, k) in LK_lk),
        "node_link_1",
    )

    m.addConstrs(
        (x[n1, k] + x[n2, k] <= 1 + y[n1, n2, k] for (n1, n2, k) in LK_lk),
        "node_link_2",
    )

    m.addConstrs(
        (gp.quicksum(x[n, k] for n in N_n) - z[k] >= 0 for k in K_k),
        "num_part",
    )

    if config["E-UNIFORM"]:
        m.addConstrs(
            (
                gp.quicksum(W_n[n] * x[n, k] for n in N_n)
                # <= W_sum / K + z[k]
                # <= (1 + config["EPSILON"]) * W_sum / K
                <= (1 + config["EPSILON"]) * soft_thred
                for k in K_k
            ),
            "e-uniform",
        )

    ### Gurobipy Parameters ###
    m.params.TimeLimit = config["TIME_LIMIT_GC"]
    m.params.MIPGap = config["MIP_GAP_GC"]

    m.optimize()

    # extract solutions
    if m.status == GRB.TIME_LIMIT:
        X = {(n, k) for (n, k) in NK_nk if x[n, k].x > 0.001}
        Y = {(n1, n2, k) for (n1, n2, k) in LK_lk if y[n1, n2, k].x > 0.001}

        P_N, P_L = dict(), dict()

        for (n, k) in X:
            _ = P_N.setdefault(k, []).append(n)

        for (n1, n2, k) in Y:
            _ = P_L.setdefault(k, []).append((n1, n2))

    elif m.status == GRB.OPTIMAL:
        X = {(n, k) for (n, k) in NK_nk if x[n, k].x > 0.001}
        Y = {(n1, n2, k) for (n1, n2, k) in LK_lk if y[n1, n2, k].x > 0.001}

        P_N, P_L = dict(), dict()

        for (n, k) in X:
            _ = P_N.setdefault(k, []).append(n)

        for (n1, n2, k) in Y:
            _ = P_L.setdefault(k, []).append((n1, n2))

    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(r"graph_coarsening_output\model.ilp")
        raise ValueError("Infeasible solution!")

    else:
        print("Status code: {}".format(m.status))

    return (X, Y, P_N, P_L, m.ObjVal)


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

    N_n = set(np.arange(N))
    K_k = set(np.arange(K))

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

    _, _ = get_link_set_disagg(config)
    (N_n, L_l, C_l, K_k, w_out, w_in, w_sum, w_f) = get_link_cost(
        {k: 0.6 * v for k, v in trips_dict_pk.items()},
        tau_disagg,
        {},
        # Route_D_disagg,
        config,
    )

    (X, Y, P_N, P_L, OBJ) = graph_coarsening(N_n, w_sum, L_l, C_l, K_k, config)
    print("number of partition: {}".format(len(P_N)))
    # check if every partition forms a valid subgraph of the orginal graph
    for p, part in P_N.items():
        if len(part) > 1 and not any(i in ctr[j] for i in part for j in part if i != j):
            raise ValueError(
                "Partition does not form a subgraph! {}: {}".format(p, part)
            )
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
