import networkx as nx
import numpy as np
from itertools import product
from random import sample, seed

# from collections import deque
# from copy import deepcopy

# SEED = 1
# np.random.seed(seed=SEED)
# seed(SEED)


def BFS(TN, C_init, VP):
    bridges = dict()
    seen = set(C_init)
    # print(C_init)
    PV = {p: {v} for v, p in VP.items()}
    # queue = deque(list(j for u in C_init for j in TN.neighbors(u) if j not in seen))
    queue = set(j for u in C_init for j in TN.neighbors(u) if j not in seen)
    # print(PV, VP)
    # print(queue)
    while queue:
        # u = queue.popleft()
        u = sample(list(queue), 1)[0]
        queue.remove(u)

        interset = set(TN.neighbors(u)) & seen
        # if len(interset) > 1:
        #     bridges[v] = set(VP[n] for n in interset)
        # random select a neighbor which belongs to one supernode
        candidate = sample(list(interset), 1)[0]
        TN.nodes[u]["part"] = TN.nodes[candidate]["part"]
        part = VP[candidate]
        VP[u] = part
        PV[part].add(u)
        seen.add(u)
        # N = list(set(TN.neighbors(u)) - seen)
        # print(N)
        # if N:
        #     queue.extend(N)

        queue = queue | (set(TN.neighbors(u)) - seen)
    # print(seen)
    # print(VP)
    for v in TN.nodes():
        bridges[v] = {VP[j] for j in TN.neighbors(v)} | {VP[v]}
    border = {v for v in bridges.keys() if len(bridges[v]) > 1}
    # for v in VP.keys():
    # for k, p in PV.items():
    # if v in p:
    # print("zeros", VP[v] == k)
    # print(sum(len(j) for j in PV.values()))
    return TN, bridges, border, PV, VP


def find_hub(PV: dict[int, set[int]], tau2_disagg: np.ndarray) -> dict[int, int]:
    PV = {p: sorted(list(part)) for p, part in PV.items()}
    PH = dict()
    for p, part in PV.items():
        mtx = tau2_disagg[np.ix_(part, part)]
        PH[p] = part[np.argmin(np.sum(mtx, axis=1))]
    return PH


def get_objective(D_uv, PV, tau2_disagg):

    temp1 = sum(
        D_uv.get((u, v), 0) ** 2 + D_uv.get((v, u), 0) ** 2
        for c1, p1 in PV.items()
        for c2, p2 in PV.items()
        if c2 != c1
        for u in p1
        for v in p2
    )
    # temp2 = sum(
    #     D_uv.get((u, v), 0) ** 2 for p in PV.values() for (u, v) in product(p, p)
    # )
    temp2 = sum(
        D_uv.get((u, v), 0) ** 2 * (1) for p in PV.values() for (u, v) in product(p, p)
    )
    PH = find_hub(PV, tau2_disagg)
    temp3 = sum(
        tau2_disagg[h, v] + tau2_disagg[v, h] for p, h in PH.items() for v in PV[p]
    )

    return -1.0 * temp1 + 2.0 * temp2 + 1.0 * temp3


def local_search(TN, D_uv, N, K, config, tau2_disagg, V_exclude=None):
    if not V_exclude:
        V = set(TN.nodes)
    else:
        V = set(TN.nodes) - V_exclude
    E = set((u, v) for (u, v) in TN.edges())
    # LV = set((u, v, c) for (u, v, c) in product(V, V, V))
    # EL = set((u, v, a, c) for ((u, v), a, c) in product(E, V, V))

    # Generate initial supernodes
    C_init = sample(list(V), K)
    VP = dict()
    for i, u in enumerate(C_init):
        TN.nodes[u]["part"] = i
        VP[u] = i

    # Generate initial partition by BFS
    TN, bridges, border, PV, VP = BFS(TN, C_init, VP)

    OBJ = get_objective(D_uv, PV, tau2_disagg)

    # Conduct local search
    FIND_GRADIANT = -2
    MAX_CLUSTER = K - 1
    while FIND_GRADIANT != -1:
        FIND_GRADIANT = -1
        # print("border",border)
        ITERDONE = 0
        border_random = np.random.permutation(list(border))
        for v in border_random:
            ITERDONE = 0
            if len(PV[VP[v]]) < 2:
                continue
            if FIND_GRADIANT >= 0:
                # print("update bridges")
                bridges[FIND_GRADIANT] = {
                    VP[j] for j in TN.neighbors(FIND_GRADIANT)
                } | {VP[FIND_GRADIANT]}
                if len(bridges[FIND_GRADIANT]) == 1:
                    border.remove(FIND_GRADIANT)

                for u in TN.neighbors(FIND_GRADIANT):
                    bridges[u] = {VP[j] for j in TN.neighbors(u)} | {VP[u]}
                for u in TN.neighbors(FIND_GRADIANT):
                    border = border | {u} if len(bridges[u]) > 1 else border - {u}

                break

            current_part = VP[v]
            for part in bridges[v]:
                if part != current_part:

                    PV[current_part].remove(v)  # PV = {P: {v1, v2, ...}}
                    PV[part].add(v)
                    VP[v] = part

                    G = TN.subgraph(PV[current_part])
                    if not nx.is_strongly_connected(G):
                        PV[current_part].add(v)
                        PV[part].remove(v)
                        VP[v] = current_part
                        continue
                    # for k, p in PV.items():
                    # if v in p:
                    # print("fisrt", VP[v] == k )

                    OBJ_new = get_objective(D_uv, PV, tau2_disagg)
                    if OBJ_new < OBJ:
                        OBJ = OBJ_new
                        print(OBJ)
                        FIND_GRADIANT = v
                        # froze = {v}
                        ITERDONE = 1
                        break
                    # Recover mappings if OBJ not decreasing
                    PV[current_part].add(v)
                    PV[part].remove(v)
                    VP[v] = current_part
                    # for k, p in PV.items():
                    # if v in p:
                    # print(VP[v] == k )

            if not ITERDONE:
                for v in V:
                    current_part = VP[v]
                    if len(PV[current_part]) == 1:
                        for part in bridges[v]:
                            if part != current_part:
                                del PV[current_part]
                                PV[part].add(v)
                                VP[v] = part
                                OBJ_new = get_objective(D_uv, PV, tau2_disagg)
                                if OBJ_new < OBJ and K - 2 <= len(PV):
                                    OBJ = OBJ_new
                                    print(OBJ, "delete")
                                    FIND_GRADIANT = v
                                    # froze = {v}
                                    ITERDONE = 1
                                    break
                                # Recover mappings if OBJ not decreasing
                                PV[current_part] = {v}
                                PV[part].remove(v)
                                VP[v] = current_part
                    else:
                        G = TN.subgraph(PV[current_part]).copy()
                        G.remove_node(v)
                        MAX_CLUSTER_TEMP = MAX_CLUSTER
                        MAX_CLUSTER += 1
                        PV[MAX_CLUSTER] = {v}
                        VP[v] = MAX_CLUSTER

                        for component in nx.strongly_connected_components(G):
                            MAX_CLUSTER += 1
                            PV[MAX_CLUSTER] = set(component)
                            for item in component:
                                VP[item] = MAX_CLUSTER
                        del PV[current_part]
                        OBJ_new = get_objective(D_uv, PV, tau2_disagg)
                        if OBJ_new < OBJ and len(PV) <= K + 2:
                            OBJ = OBJ_new
                            print(OBJ, "add")
                            FIND_GRADIANT = v
                            # froze = {v}
                            ITERDONE = 1
                            for item in {
                                nei
                                for g in set(G.nodes) | {v}
                                for nei in set(TN.neighbors(g)) | {g}
                            }:
                                bridges[item] = {VP[j] for j in TN.neighbors(item)} | {
                                    VP[item]
                                }

                            break
                        PV[current_part] = set()
                        for part in range(MAX_CLUSTER_TEMP + 1, MAX_CLUSTER + 1):
                            PV[current_part].update(PV[part])
                            for item in PV[part]:
                                VP[item] = current_part
                            del PV[part]
                    if ITERDONE:
                        break

    PV = {i: p for i, (_, p) in enumerate(PV.items())}
    VP = {v: c for c, p in PV.items() for v in p}
    return PV, VP


if __name__ == "__main__":
    import yaml, time
    import pickle
    import os
    from trip_prediction import trip_prediction
    from utils import (
        load_mc_input,
        load_neighbor_disagg,
        get_link_set_disagg,
        # get_link_cost,
        # post_processing,
        network_plot,
        update_tau_agg,
        load_tau_disagg,
        update_ctr_agg,
        travel_demand_plot,
    )

    with open("config_m2m_gc.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)

    config[
        "figure_pth"
    ] = r"many2many_output\\figure\\graph_coarsening_local_search_test\\"
    if not os.path.exists(config["figure_pth"]):
        os.makedirs(config["figure_pth"])

    ctr = load_neighbor_disagg(config)  # disagg station info
    tau_disagg, tau2_disagg = load_tau_disagg(config)
    N, K = config["S_disagg"], config["K"]

    try:
        transit_trips_dict_pk = pickle.load(open("trips_dict_pk.p", "rb"))
    except:
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
    tic = time.perf_counter()
    PV, VP = local_search(TN, transit_trips_dict_pk, N, K, config, tau2_disagg)
    toc = time.perf_counter() - tic
    print(f"time: {toc}")
    # check if every partition forms a valid subgraph of the orginal graph
    for p, part in PV.items():
        G = TN.subgraph(part)
        if not nx.is_strongly_connected(G):
            print(
                "Warning: Partition does not form a connected subgraph! {}: {}".format(
                    p, part
                )
            )

    # Update agg_2_disagg_id and disagg_2_agg_id
    agg_2_disagg_id = dict()
    # Mapping from origin aggregated zone id to new aggregated zone id
    # of bus-visited aggregated zones, should be 1-to-1 mapping
    agg_2_agg_new_bus = dict()
    for idx, c in enumerate(PV.keys()):
        agg_2_disagg_id[idx] = list(PV[c])

    disagg_2_agg_id = {
        n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
    }
    ctr_agg = update_ctr_agg(ctr, disagg_2_agg_id)
    tau, tau2 = update_tau_agg(ctr_agg, tau_disagg, agg_2_disagg_id, config)
    network_plot(tau, ctr_agg, disagg_2_agg_id, config)
    # plot travel demand
    travel_demand_plot(transit_trips_dict_pk, config)
    print("testing finished!")
