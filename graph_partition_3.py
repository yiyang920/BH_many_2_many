# %% Packages
import itertools
import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def graph_coarsening(TN, D_uv, N, K, config):
    ### Gurobipy Parameters ###
    m = gp.Model("graph_coarsening")
    m.params.OutputFlag = 1
    m.params.TimeLimit = config["TIME_LIMIT_GC"]
    m.params.MIPGap = config["MIP_GAP_GC"]
    m.params.IntFeasTol = 1e-9
    m.params.FeasibilityTol = 1e-9
    m.params.OptimalityTol = 1e-9

    V = set(i for i in range(N))
    E = set((u, v) for (u, v) in TN.edges())
    L = set(itertools.product(V, V))
    EV = set((u, v, c) for ((u, v), c) in itertools.product(E, V))
    LV = set((u, v, c) for (u, v, c) in itertools.product(V, V, V))
    
    d_sum = sum(D_uv.values())
    d2_sum = sum(v ** 2 for v in D_uv.values())
    
    ### Variables ###
    x = m.addVars(L, vtype=GRB.BINARY)
    y = m.addVars(LV, vtype=GRB.BINARY)
    f = m.addVars(EV)

    ### Model Objective ###
    m.setObjective(
        gp.quicksum(D_uv[u, v] * y[u, v, c] for (u, v, c) in LV) / d_sum
        + gp.quicksum(D_uv[u, v] ** 2 * (1 - y[u, v, c]) for (u, v, c) in LV) / d2_sum,
        sense=GRB.MINIMIZE,
    )

    ### Model Constraints ###
    m.addConstrs(
        (x.sum(v, "*") == 1 for v in V),
        "node_to_partition",
    )
    m.addConstr(
        (gp.quicksum(x[v, v] for v in V) == K),
        "num_of_partitions1",
    )
    m.addConstrs(
        (gp.quicksum(x[v, c] for v in V) <= (N - K) * x[c, c] for c in V),
        "num_of_partitions2"
    )
    m.addConstrs(
        (x[u, c] + x[v, c] - 2 * y[u, v, c] >= 0 for (u, v, c) in LV),
        "node_link_1",
    )
    m.addConstrs(
        (x[u, c] + x[v, c] - y[u, v, c] - 1 <= 0 for (u, v, c) in LV),
        "node_link_2",
    )
    m.addConstrs(
        (
            gp.quicksum(f[i, j, c] for (i, j) in TN.in_edges(v))
            - gp.quicksum(f[i, j, c] for (i, j) in TN.out_edges(v))
            == x[v, c]
            for c in V
            for v in V
            if v != c
        ),
        "flow_balance_1",
    )
    m.addConstrs(
        (
            gp.quicksum(f[i, j, c] for (i, j) in TN.in_edges(v)) - (N - K) * x[v, c]
            <= 0
            for c in V
            for v in V
            if v != c
        ),
        "flow_balance_2",
    )
    m.addConstrs(
        (gp.quicksum(f[i, j, c] for (i, j) in TN.in_edges(c)) == 0 for c in V),
        "flow_balance_3",
    )

    return (m, x, y, f, L, LV)


if __name__ == "__main__":
    import yaml
    import pickle
    import os
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
    tau_disagg, tau2_disagg = load_tau_disagg(config)
    N, K = config["S_disagg"], config["K"]

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

    TN, E = get_link_set_disagg(config)

    # Run graph coarsening
    m, x, y, f, L, LV = graph_coarsening(TN, transit_trips_dict_pk, N, K, config)

    m.optimize()

    # extract solutions
    if m.status == GRB.TIME_LIMIT:
        X = {(u, c) for (u, c) in L if round(x[u, c].x)}
        Y = {(u, v, c) for (u, v, c) in LV if round(y[u, v, c].x)}

        P_N, P_L = dict(), dict()

        for (u, c) in X:
            _ = P_N.setdefault(c, []).append(u)

        for (u, v, c) in Y:
            _ = P_L.setdefault(c, []).append((u, v))

    elif m.status == GRB.OPTIMAL:
        X = {(u, c) for (u, c) in L if round(x[u, c].x)}
        Y = {(u, v, c) for (u, v, c) in LV if round(y[u, v, c].x)}

        P_N, P_L = dict(), dict()

        for (u, c) in X:
            _ = P_N.setdefault(c, []).append(u)

        for (u, v, c) in Y:
            _ = P_L.setdefault(c, []).append((u, v))

    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(r"graph_coarsening_output\model.ilp")
        raise ValueError("Infeasible solution!")

    else:
        print("Status code: {}".format(m.status))

    print("number of partition: {}".format(len(P_N)))

    # check if every partition forms a valid subgraph of the orginal graph
    for p, part in P_N.items():
        G = TN.subgraph(part)
        if not nx.is_strongly_connected(G):
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
