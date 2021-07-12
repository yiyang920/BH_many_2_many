# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101
# %% import packages
import pandas as pd
import numpy as np
import pickle
import operator
import networkx as nx
import os


# from st_network3 import Many2Many
from trip_prediction import trip_prediction

from graph_coarsening import graph_coarsening
from utils import (
    load_neighbor_disagg,
    load_mc_input,
    get_link_cost,
    load_FR_disagg,
    update_ctr_agg,
    update_time_dist,
)


def init_partition(tau_disagg, config):
    Route_fr_disagg = load_FR_disagg(config) if config["FIXED_ROUTE"] else None

    if config["BUILD_ON"]:
        route_d = dict()
        if not isinstance(Route_fr_disagg, dict):
            Route_fr_disagg = dict()
        for idx, fname in config["build_on_list"].items():
            route_d[idx] = pd.read_csv(config["m2m_data_loc"] + fname)

        for d, df in route_d.items():
            Route_fr_disagg[d] = [
                (
                    t1,
                    t2,
                    s1,
                    s2,
                    t1 * config["S_disagg"] + s1,
                    t2 * config["S_disagg"] + s2,
                )
                for (t1, t2, s1, s2) in zip(
                    df.t.iloc[:-1],
                    df.t.iloc[1:],
                    df.s.iloc[:-1],
                    df.s.iloc[1:],
                )
            ]

    # load mode-choice input data
    (
        per_time,
        per_dist,
        per_emp,
        mdot_dat,
        dests_geo,
        D,
    ) = load_mc_input(config)
    # load ID converter between GEOID and zone_id
    id_converter = pickle.load(open(config["id_converter"], "rb"))
    id_converter_reverse = pickle.load(open(config["id_converter_reverse"], "rb"))

    # Update travel time and travel distance table
    od_travel_time, od_travel_dist = update_time_dist(Route_fr_disagg, config)
    # list of origins from mdot_dat trip data
    if od_travel_time and od_travel_dist:
        O_p = [
            id_converter_reverse[o] if o in id_converter_reverse else 9999
            for o in mdot_dat.geoid_o
        ]
        # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
        D_p = (
            {}
        )  # dict of destination for each transportation mode from mdot_dat trip data
        D_p[2] = [
            id_converter_reverse[d] if d in id_converter_reverse else 9999
            for d in dests_geo.geoid3
        ]
        per_time[2] = [
            od_travel_time[o, d] if (o != 9999) and (d != 9999) else 9999
            for (o, d) in zip(O_p, D_p[2])
        ]
        per_dist.dist3 = [
            od_travel_dist[o, d]
            if (o != 9999) and (d != 9999)
            else per_dist.dist3.iloc[i]
            for (i, (o, d)) in enumerate(zip(O_p, D_p[2]))
        ]
    # trip prediction
    (
        trips_dict_pk,
        trips_dict_op,
        transit_trips_dict_pk,
        transit_trips_dict_op,
    ) = trip_prediction(
        id_converter, per_dist, per_emp, mdot_dat, dests_geo, D, per_time=per_time
    )

    # graph coarsening
    (
        N_n,
        L_l,
        C_l,
        K_k,
        *_,
        w_sum,
        _,
    ) = get_link_cost(trips_dict_pk, tau_disagg, Route_fr_disagg, config)

    (_, _, P_N, P_L, _) = graph_coarsening(N_n, w_sum, L_l, C_l, K_k, config)

    # Record current agg2disagg of those bus-visited zones
    bus_zone = (
        set(stop[2] for _, route in Route_fr_disagg.items() for stop in route)
        if Route_fr_disagg
        else set()
    )
    # Update agg_2_disagg_id and disagg_2_agg_id
    agg_2_disagg_id = dict()

    for idx, c in enumerate(P_N.keys()):
        agg_2_disagg_id[idx] = P_N[c]
    # P_N excludes the zones with bus service, need to add those excluded zones
    # into agg_2_disagg_id

    for idx, node in enumerate(bus_zone):
        new_part_id = len(P_N) + idx
        agg_2_disagg_id[new_part_id] = [node]

    disagg_2_agg_id = {
        n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
    }
    # Update config dictionary: number of partitions of aggregated network
    config["S"] = len(agg_2_disagg_id)
    # Load neighbor nodes information of disaggregated zones
    ctr_disagg = load_neighbor_disagg(config)
    # Update ctr, i.e. station neighbor info of aggregated network
    ctr = update_ctr_agg(ctr_disagg, disagg_2_agg_id)

    file_dir = config["m2m_output_loc"] + "init_graph\\"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    pickle.dump(ctr, open(file_dir + "ctr_agg.p", "wb"))
    pickle.dump(ctr_disagg, open(file_dir + "ctr_disagg.p", "wb"))
    pickle.dump(agg_2_disagg_id, open(file_dir + "agg_2_disagg_id.p", "wb"))
    pickle.dump(disagg_2_agg_id, open(file_dir + "disagg_2_agg_id.p", "wb"))

    return ctr, agg_2_disagg_id, disagg_2_agg_id
