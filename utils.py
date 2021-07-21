# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# Disable all the "Passing unexpected keyword argument %r in function call" in this function
# pylint: disable=E0611, E1101, E1123, E1120
# import glob
import os
from typing import Any

os.environ["PROJ_LIB"] = r"C:\\Users\\SQwan\\miniconda3\\Library\\share"

import operator
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
from python_tsp.exact import solve_tsp_dynamic_programming
from collections import deque

# from python_tsp.distances import great_circle_distance_matrix


def load_neighbor(config):
    """
    Load neighbor nodes information of aggregated zones.
    """
    ctr_info = pickle.load(open(config["Station_agg"], "rb"))
    ctr = dict()
    zone_id = list(ctr_info.keys())
    for i in range(len(ctr_info)):
        ctr_info[i]["neighbours"].append(zone_id[i])
        ctr[i] = list(e for e in ctr_info[i]["neighbours"] if e < config["S"])

    return ctr


def load_neighbor_disagg(config):
    """
    Load neighbor nodes information of disaggregated zones.
    """
    ctr_info = pickle.load(open(config["Station"], "rb"))
    ctr = dict()
    zone_id = list(ctr_info.keys())
    for i in range(len(ctr_info)):
        ctr_info[i]["neighbours"].append(zone_id[i])
        ctr[i] = list(e for e in ctr_info[i]["neighbours"] if e < config["S_disagg"])

    return ctr


def get_link_set_disagg(config):
    """
    Generate link set of disaggregated network.
    """
    # load neighbor nodes information of disaggregated zones
    ctr = load_neighbor_disagg(config)
    # Construct a graph based on neighbor nodes information
    G = nx.DiGraph()
    for i in range(config["S_disagg"]):
        for j in ctr[i]:
            G.add_edge(i, j)

    L_l = set(G.edges())

    temp_dir = config["m2m_output_loc"] + "temp\\"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    pickle.dump(G, open(temp_dir + "G_disagg.p", "wb"))
    return G, {(i, j) for (i, j) in L_l if i != j}


def load_tau(config):
    """
    Load shortest travel time matrices of aggregated zones
    """
    # graph with edge cost as shortest travel time
    G_t = pickle.load(open(config["G_t_agg"], "rb"))
    # graph with edge cost as shortest travel distance
    G_d = pickle.load(open(config["G_d_agg"], "rb"))

    S = config["S"]
    DELTA_t = config["DELTA_t"]
    # round travel time to integer
    for _, _, d in G_t.edges(data=True):
        d["weight"] = np.rint(d["weight"])

    tau = np.zeros((S, S))
    tau2 = np.zeros((S, S))
    # TODO: convert travel time unit to the unit of many-2-many problem
    for i in range(S):
        for j in range(S):
            if i == j:
                tau[i, j] = 0
                tau2[i, j] = 1
            else:
                tau[i, j] = (
                    nx.shortest_path_length(G_t, source=i, target=j, weight="weight")
                    // DELTA_t
                )

                tau2[i, j] = tau[i, j]

    return (tau, tau2)


def load_tau_disagg(config):
    """
    Load shortest travel time matrices of disaggregated zones
    """
    # graph with edge cost as shortest travel time
    G_t = pickle.load(open(config["G_t"], "rb"))
    # graph with edge cost as shortest travel distance
    G_d = pickle.load(open(config["G_d"], "rb"))

    S = config["S_disagg"]
    DELTA_t = config["DELTA_t"]
    # round travel time to integer
    for _, _, d in G_t.edges(data=True):
        d["weight"] = np.rint(d["weight"])

    tau = np.zeros((S, S))
    tau2 = np.zeros((S, S))

    for i in range(S):
        for j in range(S):
            if i == j:
                tau[i, j] = 0
                tau2[i, j] = 1
            else:
                tau[i, j] = (
                    nx.shortest_path_length(G_t, source=i, target=j, weight="weight")
                    // DELTA_t
                )

                tau2[i, j] = tau[i, j]

    return (tau, tau2)


def load_mc_input(config):
    """
    Load mode choice input data
    """
    mc_fileloc = config["mc_fileloc"]

    per_dist = pd.read_csv(mc_fileloc + "dists_dat_auto.csv", sep=",")
    per_emp = pd.read_csv(mc_fileloc + "dests_emp.csv", sep=",")
    mdot_dat = pd.read_csv(mc_fileloc + "mdot_trips_dc.csv", sep=",")
    dests_geo = pd.read_csv(mc_fileloc + "dests_geoid.csv", sep=",")
    D = pd.read_csv(mc_fileloc + "distance.csv", sep=",", index_col=[0])

    per_time = {}  # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
    per_time[0] = list(
        pd.read_csv(
            mc_fileloc + "time_dat_walk.csv",
            usecols=[
                "time1",
            ],
        ).time1
    )
    per_time[1] = list(
        pd.read_csv(
            mc_fileloc + "time_dat_dar.csv",
            usecols=[
                "time2",
            ],
        ).time2
    )
    per_time[3] = list(
        pd.read_csv(
            mc_fileloc + "time_dat_auto.csv",
            usecols=[
                "time4",
            ],
        ).time4
    )
    per_time[4] = list(
        pd.read_csv(
            mc_fileloc + "time_dat_auto.csv",
            usecols=[
                "time5",
            ],
        ).time5
    )
    return per_time, per_dist, per_emp, mdot_dat, dests_geo, D


def load_mc_input_2(
    config: dict,
) -> tuple[
    dict[int, int],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load trip-based travel demand model input data,
    and return a zoneid-to-geoid converter.
    """
    mc_fileloc = config["mc_fileloc"]

    stop_zones_df = pd.read_csv(mc_fileloc + "stop_zones.csv")
    stop_zones = set(stop_zones_df["stop_zones"].apply(lambda x: x - 1))

    D = pd.read_csv(mc_fileloc + "distance.csv", index_col=0)
    id_converter = {i: idx for i, idx in enumerate(D.index)}

    gm_transTT = pd.read_csv(mc_fileloc + "gm_transit_time_min.csv", index_col=0)
    gm_transdist = pd.read_csv(mc_fileloc + "gm_transit_dist_km.csv", index_col=0)
    gm_autoTT = pd.read_csv(mc_fileloc + "gm_auto_time_min.csv", index_col=0)
    gm_autodist = pd.read_csv(mc_fileloc + "gm_auto_dist_km.csv", index_col=0)
    gm_wkTT = pd.read_csv(mc_fileloc + "gm_walk_time_min.csv", index_col=0)
    gm_wkdist = pd.read_csv(mc_fileloc + "gm_walk_dist_km.csv", index_col=0)
    est_dat = pd.read_csv(mc_fileloc + "mc_dat_trips_miss.csv")
    mdot_dat = est_dat.loc[est_dat["dat_id"] == 3, :]

    per_emp = pd.read_csv(mc_fileloc + "dests_temp_trips.csv")
    per_tdist = pd.read_csv(mc_fileloc + "tdists_dat_trips.csv")
    dests_geo = pd.read_csv(mc_fileloc + "tdests_geoid_trips.csv")
    per_ddist = pd.read_csv(mc_fileloc + "tdests_ddist_trips.csv")
    per_bustt = pd.read_csv(mc_fileloc + "tdests_bustt_trips.csv")
    per_drtt = pd.read_csv(mc_fileloc + "tdests_drtt_trips.csv")
    per_wktt = pd.read_csv(mc_fileloc + "tdests_wktt_trips.csv")

    return (
        id_converter,
        mdot_dat,
        stop_zones,
        per_ddist,
        per_tdist,
        per_drtt,
        per_bustt,
        per_wktt,
        per_emp,
        dests_geo,
        gm_autodist,
        gm_transdist,
        gm_autoTT,
        gm_transTT,
        gm_wkTT,
    )


def disagg_2_agg_trip(
    transit_trips_dict: dict[tuple[int, int], float],
    config: dict[str, Any],
    disagg_2_agg_id: dict[int, int] = None,
    fraction: float = 0.6,
) -> dict[tuple[int, int], int]:
    """
    Convert disaggregated trips into aggregated trips
    Rider info: Rounding trip number with threshold 0.5
    Multiply 0.6 (default) representing morning peak hour 7-10 am
    """
    # agg_2_disagg_id = pickle.load(open(config["agg_2_disagg_id"], "rb"))
    if disagg_2_agg_id is None:
        disagg_2_agg_id = pickle.load(open(config["disagg_2_agg_id"], "rb"))

    transit_trips_dict_agg = dict()
    for od, trips in transit_trips_dict.items():
        if (
            disagg_2_agg_id[od[0]],
            disagg_2_agg_id[od[1]],
        ) in transit_trips_dict_agg:
            transit_trips_dict_agg[
                disagg_2_agg_id[od[0]], disagg_2_agg_id[od[1]]
            ] += trips
        else:
            transit_trips_dict_agg[
                disagg_2_agg_id[od[0]], disagg_2_agg_id[od[1]]
            ] = trips

    trip_dict = {
        k: int(np.round(fraction * v))
        for k, v in transit_trips_dict_agg.items()
        if int(np.round(fraction * v)) > 0
        if k[0] != k[1]
    }

    return trip_dict


def get_rider(trip_dict, config):
    """
    Generate rider information as a numpy array
    """
    Rider = pd.DataFrame(columns=["ID", "NAN", "ED", "LA", "O", "D", "SL"])
    N_r = sum(trip_dict.values())
    O_r = list()
    D_r = list()

    for k, v in trip_dict.items():
        O_r += [k[0] for _ in range(v)]
        D_r += [k[1] for _ in range(v)]

    Rider["ID"] = np.arange(N_r)
    Rider["O"], Rider["D"] = O_r, D_r
    Rider["ED"] = 0

    # Rider["ED"] = np.random.randint(0, config["T"] // 5, N_r)
    Rider["LA"] = [config["T"] - 1] * N_r
    # Rider["LA"] = np.random.randint(config["T"] // 5 * 3, config["T"] // 5 * 4, N_r)
    Rider["SL"] = [10] * N_r

    Rider = Rider.fillna(999)
    Rider.to_csv(config["m2m_data_loc"] + r"Rider_agg.csv", index=False)
    return Rider.to_numpy(dtype=int, na_value=999)


def disagg_trip_get_rider(
    transit_trips_dict, config, tau_disagg, disagg_2_agg_id=None, fraction=0.6
):
    """
    Convert disaggregated trips into aggregated trips
    Generate rider information as a numpy array
    Rider info: Rounding trip number with threshold 0.5
    Multiply 0.6 (default) representing morning peak hour 7-10 am
    """
    # agg_2_disagg_id = pickle.load(open(config["agg_2_disagg_id"], "rb"))
    if disagg_2_agg_id is None:
        disagg_2_agg_id = pickle.load(open(config["disagg_2_agg_id"], "rb"))

    transit_trips_dict_round = {
        (o, d): int(np.round(fraction * trips))
        for (o, d), trips in transit_trips_dict.items()
        if int(np.round(fraction * trips)) > 0 and (o != d)
    }

    Rider = pd.DataFrame(columns=["ID", "NAN", "ED", "LA", "O", "D", "SL"])
    N_r = sum(transit_trips_dict_round.values())
    O_r = list()
    D_r = list()
    SP_r_foo = list()

    for (o, d), trips in transit_trips_dict_round.items():
        O_r += [disagg_2_agg_id[o]] * trips
        D_r += [disagg_2_agg_id[d]] * trips
        SP_r_foo += [tau_disagg[o, d]] * trips

    Rider["ID"] = np.arange(N_r)
    Rider["O"], Rider["D"] = O_r, D_r
    Rider["ED"] = 0
    # Rider["ED"] = np.random.randint(0, config["T"] // 5, N_r)
    Rider["LA"] = [config["T"] - 1] * N_r
    # Rider["LA"] = np.random.randint(config["T"] // 5 * 3, config["T"] // 5 * 4, N_r)
    Rider["SL"] = [10] * N_r

    SP_r = {n: SP_r_foo[n] for n in np.arange(N_r)}

    Rider = Rider.fillna(999)
    Rider.to_csv(config["m2m_data_loc"] + r"Rider_agg.csv", index=False)
    return SP_r, Rider.loc[:, ["ID", "NAN", "ED", "LA", "O", "D", "SL"]].to_numpy(
        dtype=int, na_value=999
    )


def load_FR(config):
    """
    Load aggregated fixed routes and build-on routes
    """
    if config["FIXED_ROUTE"]:
        df_blue_1 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_1_agg.csv")
        df_blue_2 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_1_agg.csv")
        df_red_1 = pd.read_csv(config["m2m_data_loc"] + r"red_fr_1_agg.csv")
        df_yellow_1 = pd.read_csv(config["m2m_data_loc"] + r"yellow_fr_1_agg.csv")

        blue_1 = df_blue_1.to_numpy()
        blue_2 = df_blue_2.to_numpy()
        red_1 = df_red_1.to_numpy()
        yellow_1 = df_yellow_1.to_numpy()
        FR = {
            0: blue_1,
            1: blue_1,
            2: red_1,
            3: yellow_1,
        }
    else:
        FR = None

    if config["BUILD_ON"]:
        if not isinstance(FR, dict):
            FR = {}
        for idx, fname in config["build_on_list"].items():
            FR[idx] = pd.read_csv(config["m2m_data_loc"] + fname).to_numpy()

    return FR


def load_FR_m2m_gc(agg_2_disagg_id, disagg_2_agg_id, config):
    """
    Load disaggregated fixed routed infomation and build-on routes.
    Load driver information in aggregated network.
    Used for mc-m2m-gc mode.
    """
    if config["REPEATED_TOUR"]:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver_rt.csv")
    else:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver.csv")
    Driver.O = Driver.O.apply(lambda x: disagg_2_agg_id[x])
    Driver.D = Driver.D.apply(lambda x: disagg_2_agg_id[x])
    if config["FIXED_ROUTE"]:
        duration_d = {d: duration for d, duration in zip(Driver.ID, Driver.Duration)}
        df_blue_1 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_1_disagg.csv")
        df_blue_1.s = df_blue_1.s.apply(lambda x: disagg_2_agg_id[x])
        df_blue_1 = df_blue_1.loc[df_blue_1.t <= duration_d[0], :]

        df_blue_2 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_1_disagg.csv")
        df_blue_2.s = df_blue_2.s.apply(lambda x: disagg_2_agg_id[x])
        df_blue_2 = df_blue_2.loc[df_blue_2.t <= duration_d[1], :]

        df_red_1 = pd.read_csv(config["m2m_data_loc"] + r"red_fr_1_disagg.csv")
        df_red_1.s = df_red_1.s.apply(lambda x: disagg_2_agg_id[x])
        df_red_1 = df_red_1.loc[df_red_1.t <= duration_d[2], :]

        df_yellow_1 = pd.read_csv(config["m2m_data_loc"] + r"yellow_fr_1_disagg.csv")
        df_yellow_1.s = df_yellow_1.s.apply(lambda x: disagg_2_agg_id[x])
        df_yellow_1 = df_yellow_1.loc[df_yellow_1.t <= duration_d[3], :]

        blue_1 = df_blue_1.to_numpy()
        blue_2 = df_blue_2.to_numpy()
        red_1 = df_red_1.to_numpy()
        yellow_1 = df_yellow_1.to_numpy()
        FR = {
            0: blue_1,
            1: blue_1,
            2: red_1,
            3: yellow_1,
        }
    else:
        FR = None
    # TODO: convert zone id!
    if config["BUILD_ON"]:
        if not isinstance(FR, dict):
            FR = dict()
        for d, fname in config["build_on_list"].items():
            temp = pd.read_csv(config["m2m_data_loc"] + fname)
            temp.s = temp.s.apply(lambda x: disagg_2_agg_id[x])
            FR[d] = temp.to_numpy()
    FR_disagg = (
        None
        if FR is None
        else {
            d: np.array([[s[0], agg_2_disagg_id[s[1]][0]] for s in route])
            for d, route in FR.items()
        }
    )
    return FR, FR_disagg, Driver.to_numpy(dtype=int, na_value=999)


def load_FR_disagg(config):
    """
    Load disaggregated fixed routed infomation
    """
    # num_tour_d = {}
    if config["FIXED_ROUTE"]:
        df_blue_1 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_1_disagg.csv")
        # num_tour_d[0] = config["T"] // df_blue_1.t.iloc[-1]
        df_blue_2 = pd.read_csv(config["m2m_data_loc"] + r"blue_fr_2_disagg.csv")
        # num_tour_d[1] = config["T"] // df_blue_2.t.iloc[-1]
        df_red_1 = pd.read_csv(config["m2m_data_loc"] + r"red_fr_1_disagg.csv")
        # num_tour_d[2] = config["T"] // df_red_1.t.iloc[-1]
        df_yellow_1 = pd.read_csv(config["m2m_data_loc"] + r"yellow_fr_1_disagg.csv")
        # num_tour_d[3] = config["T"] // df_yellow_1.t.iloc[-1]

        route_d = {0: df_blue_1, 1: df_blue_2, 2: df_red_1, 3: df_yellow_1}

        FR = {
            d: [
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
            for d, df in route_d.items()
        }
    else:
        FR = None
    return FR


def get_driver(config):
    """
    Load driver information in aggregated network.
    """
    if config["REPEATED_TOUR"]:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver_rt_agg.csv")
    else:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver_agg.csv")

    return Driver.to_numpy(dtype=int, na_value=999)


def get_driver_m2m_gc(disagg_2_agg_id, config):
    """
    Load driver information in disaggregated network.
    """
    if config["REPEATED_TOUR"]:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver_rt.csv")
    else:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver.csv")
    Driver.O = Driver.O.apply(lambda x: disagg_2_agg_id[x])
    Driver.D = Driver.D.apply(lambda x: disagg_2_agg_id[x])

    return Driver.to_numpy(dtype=int, na_value=999)


def get_driver_OD(config):
    """
    Load driver OD in disaggregated network
    """
    if config["REPEATED_TOUR"]:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver_rt.csv")
        OS_d = {d: o for d, o in zip(Driver.ID, Driver.O)}
        DS_d = {d: o for d, o in zip(Driver.ID, Driver.D)}
    else:
        Driver = pd.read_csv(config["m2m_data_loc"] + r"Driver.csv")
        OS_d = {d: o for d, o in zip(Driver.ID, Driver.O)}
        DS_d = {d: o for d, o in zip(Driver.ID, Driver.D)}
    return OS_d, DS_d


def update_time_dist(Route_D, config):
    """
    Update the transit travel time and distance between each pair of OD
    in disaggregated network, based on the optimization results.
    """
    S, T = config["S_disagg"], config["T"]
    # disagg graph with edge cost as shortest travel distance
    G_d = pickle.load(open(config["G_d"], "rb"))

    # time expanded network
    TN_t = nx.DiGraph()  # network with travel time info
    TN_d = nx.DiGraph()  # network with travel distance info

    if not Route_D:
        return None, None
    # add route links into the network
    for d in Route_D:
        for (_, _, _, _, n1, n2) in Route_D[d]:
            # set weight as travel time
            TN_t.add_edge(n1, n2, weight=n2 // S - n1 // S)
            # set weight as shortest path travel distance of the transit network
            TN_d.add_edge(
                n1,
                n2,
                weight=nx.shortest_path_length(
                    G_d, source=n1 % S, target=n2 % S, weight="weight"
                ),
            )

    # add waiting links into the network
    for t in range(T):
        for s in range(S):
            # set weight as waiting time
            TN_t.add_edge(t * S + s, (t + 1) * S + s, weight=1.0)
            TN_d.add_edge(t * S + s, (t + 1) * S + s, weight=0.0)

    od_travel_time = dict()
    od_travel_dist = dict()
    for i in range(S):
        for j in range(S):
            od_travel_time[i, j] = 9999
            od_travel_dist[i, j] = 9999

    # sp_t and sp_d are dictionary of dictionaries with sp_t[source][target]=L
    sp_t = {k: v for (k, v) in nx.shortest_path_length(TN_t, weight="weight")}
    sp_d = {k: v for (k, v) in nx.shortest_path_length(TN_d, weight="weight")}

    # update travel time influnced by bus routes
    for o in sp_t:
        for d in sp_t[o]:
            if od_travel_time[o % S, d % S] == 9999:
                od_travel_time[o % S, d % S] = sp_t[o][d]
            elif od_travel_time[o % S, d % S] > sp_t[o][d]:
                od_travel_time[o % S, d % S] = sp_t[o][d]

    # update travel distance influnced by bus routes
    for o in sp_d:
        for d in sp_d[o]:
            if od_travel_dist[o % S, d % S] == 9999:
                od_travel_dist[o % S, d % S] = sp_d[o][d]
            elif od_travel_dist[o % S, d % S] > sp_d[o][d]:
                od_travel_dist[o % S, d % S] = sp_d[o][d]

    return od_travel_time, od_travel_dist


def post_processing(Route_D, config, agg_2_disagg_id=None):
    """
    Disaggregate the bus route and solve the open TSP problem.
    Return an empty list if no route needs to be disaggregated.

    IMPORTANT: Current version of this function only considers
    bus/repeated-tour mode!
    """
    result_file_path = config["m2m_output_loc"]
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    try:
        OD_d = pd.read_csv(
            config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
        ).to_dict("index")
    except FileNotFoundError:
        print(
            "Warning: diver origin and destination data not found! TSP solution will alter the bus OD!"
        )
        OD_d = None
    driver_set = config["driver_set"]

    if not driver_set:
        return {}  # return an empty dict if no route needs to be disaggregated

    S = config["S_disagg"]  # number of zones
    T = config["T"]
    DELTA_t = config["DELTA_t"]  # x min

    if agg_2_disagg_id is None:
        agg_2_disagg_id = pickle.load(open(config["agg_2_disagg_id"], "rb"))

    # station info of disaggregated network
    ctr_info = pickle.load(open(config["Station"], "rb"))
    # graph with edge cost as shortest travel time, disagg network
    G_t = pickle.load(open(config["G_t"], "rb"))

    # round travel time to integer
    for _, _, d in G_t.edges(data=True):
        d["weight"] = np.rint(d["weight"])

    pth_dict_agg = {}
    for k, route in Route_D.items():
        pth_dict_agg[k] = list(set(s[2] for s in route))

    pth_dict_temp = {}
    for k, pth in pth_dict_agg.items():
        pth_dict_temp[k] = [agg_2_disagg_id[s] for s in pth]

    pth_dict = {}
    for k, pth in pth_dict_temp.items():
        pth_dict[k] = list()
        for lst in pth:
            pth_dict[k] += lst

    c_dict = {
        k: [ctr_info[zone]["lat_lon"] for zone in zone_lst]
        for k, zone_lst in pth_dict.items()
    }  # centroid coordinates of zones

    # Solve the Open TSP Problem
    tau = np.zeros((S, S))
    tau2 = np.zeros((S, S))

    for i in range(S):
        for j in range(S):
            if i == j:
                tau[i, j] = 0
                tau2[i, j] = 1
            else:
                tau[i, j] = (
                    nx.shortest_path_length(G_t, source=i, target=j, weight="weight")
                    // DELTA_t
                )
                tau2[i, j] = tau[i, j]
    Route_D_disagg = {}
    if config["REPEATED_TOUR"]:
        for d in driver_set:
            distance_matrix = np.zeros((len(c_dict[d]), len(c_dict[d])))
            for i in range(len(c_dict[d])):
                for j in range(len(c_dict[d])):
                    distance_matrix[i, j] = tau[pth_dict[d][i], pth_dict[d][j]]

            permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
            # check if all pair of successive zones are neighbors
            route_id = [pth_dict[d][i] for i in permutation]
            route_id.append(route_id[0])
            s = list()
            s.append(route_id[0])
            for previous, current in zip(route_id, route_id[1:]):
                if current in ctr_info[previous]["neighbours"]:
                    s.append(current)
                else:
                    sp = nx.shortest_path(
                        G_t, source=previous, target=current, weight="weight"
                    )
                    s += sp[1:]

            # make sure the bus starts at the origin station
            s.pop(-1)
            if OD_d:
                depot_idx = s.index(OD_d[d]["O"])
                foo = deque(s)
                foo.rotate(depot_idx)
                s = list(foo)
            # make sure the bus back to the depot
            s.append(s[0])

            # get bus schedule time based on tau matrix
            sch = list(
                np.cumsum(
                    [0]
                    + [
                        tau[s[i], s[i + 1]] if s[i] != s[i + 1] else 1
                        for i in range(len(s) - 1)
                    ]
                )
            )

            duration_d = sch[-1]
            num_tour_d = T // duration_d

            sch_temp = sch[1:]
            s_temp = s[1:]
            for i in np.arange(1, num_tour_d):
                sch += [t + duration_d * i for t in sch_temp]
                s += s_temp

            Route_D_disagg[d] = [
                (t1, t2, s1, s2, t1 * S + s1, t2 * S + s2)
                for (t1, t2, s1, s2) in zip(sch[:-1], sch[1:], s[:-1], s[1:])
            ]
    # TODO: DaR mode post processing procedure!

    return Route_D_disagg
    # # save files
    # df_fr = pd.DataFrame.from_dict(fr)
    # df_fr.to_csv(result_file_path + r"fr_{}.csv".format(d), index=False)

    # df_sch = pd.DataFrame()
    # df_sch["t1"], df_sch["t2"] = fr["t"][:-1], fr["t"][1:]
    # df_sch["s1"], df_sch["s2"] = fr["s"][:-1], fr["s"][1:]
    # df_sch.to_csv(result_file_path + r"sch_{}.csv".format(d), index=False)


def get_link_cost(trips_dict, tau_disagg, Route_D, config):
    """
    Compute the node weights, and generate the link cost of
    the disaggregated network. Node weight is a 3-dimensional
    tuple, i.e. (w_in, w_out, w_f).

    Remove nodes which have bus visited.
    """
    N, K = config["S_disagg"], config["K"]
    N_n = set(np.arange(N))
    K_k = set(np.arange(K))

    # load graph of disaggregated network
    G = pickle.load(open(config["m2m_output_loc"] + r"temp\G_disagg.p", "rb"))
    trip_sum = sum(trips_dict.values())

    # calculate outdegree and indegree of each node
    w_out = {
        n: sum(trip_count for ((o, _), trip_count) in trips_dict.items() if o == n)
        / trip_sum
        for n in N_n
    }
    w_in = {
        n: sum(trip_count for ((_, d), trip_count) in trips_dict.items() if d == n)
        / trip_sum
        for n in N_n
    }
    w_sum = {n: w_in[n] * trip_sum + w_out[n] * trip_sum for n in N_n}
    w_sum_max = max(w_sum.values())
    w_sum_mean = np.mean([v for v in w_sum.values()])

    # calculate bus frequency for each node
    w_f = dict()
    if Route_D:
        for _, route in Route_D.items():
            for station in route:
                if station[2] in w_f:
                    w_f[station[2]] += 1
                else:
                    w_f[station[2]] = 1
        # minimum distance from node n1 to any bus-visited nodes.
        dist_n = {
            n1: min(tau_disagg[n1, n2] for n2 in w_f.keys())
            for n1 in N_n - set(w_f.keys())
        }
        dist_max = max(dist_n.values())
    else:
        dist_n = {n: 1 for n in N_n}
        dist_max = 1

    # set default node's outdegree and indegree as 0 if the node is not in w_out or w_in
    for n in N_n:
        _ = w_out.setdefault(n, 0)
        _ = w_in.setdefault(n, 0)
        _ = w_out.setdefault(n, 0)
        _ = w_f.setdefault(n, 0)

    # w_vec = {n: w_in[n] + w_out[n] for n in N_n}

    # remove nodes with bus visited
    bus_visit = set(s for s, f in w_f.items() if f > 0)
    G.remove_nodes_from(bus_visit)
    N_n = N_n.difference(bus_visit)

    L_l = set((n1, n2) for (n1, n2) in G.edges() if n1 != n2)

    # d = {
    #     (n1, n2): np.sum(np.square(np.array(w_vec[n1]) - np.array(w_vec[n2])))
    #     for (n1, n2) in L_l
    # }

    # SIGMA = np.std([distance for distance in d.values()])
    # C_l = {
    #     (n1, n2): (np.exp(-d[n1, n2] / (2 * SIGMA ** 2)) + 1)
    #     * (1 - config["THETA"] * max(w_f[n1], w_f[n2]))
    #     * min(dist_n[n1], dist_n[n2])
    #     for (n1, n2) in L_l
    # }
    # C_l = {
    #     (n1, n2): min(dist_n[n1], dist_n[n2])
    #     / dist_max
    #     * w_sum_mean ** 2
    #     / (max(w_sum[n1], w_sum[n2], 1)) ** 2
    #     + 0.001
    #     for (n1, n2) in L_l
    # }
    C_l = dict()
    for n1, n2 in L_l:
        foo = (
            (
                min(dist_n[n1], dist_n[n2])
                / dist_max
                * w_sum_mean ** 2
                / (0.5 * (w_sum[n1] + w_sum[n2])) ** 2
            )
            if w_sum[n1] and w_sum[n2]
            else 20
        )
        C_l[n1, n2] = foo
        # C_l[n1, n2] = max(foo - 0.5, 0.1)

    return N_n, L_l, C_l, K_k, w_out, w_in, w_sum, w_f


def update_tau_agg(ctr_agg, tau_disagg, agg_2_disagg_id, config):
    """
    Update tau matrix of aggregated network, the travel time between each pair of
    neighboring nodes is the tsp time of two aggreagted zones + max travel time
    from one aggregated zone to another.
    """

    # calculate TSP distance for each aggregated zone
    tsp_c = dict()
    for c in ctr_agg.keys():
        p_size = len(agg_2_disagg_id[c])
        distance_matrix = np.zeros((p_size, p_size))
        for i in range(p_size):
            for j in range(p_size):
                distance_matrix[i, j] = tau_disagg[
                    agg_2_disagg_id[c][i], agg_2_disagg_id[c][j]
                ]

        _, tsp_c[c] = solve_tsp_dynamic_programming(distance_matrix)

    G = nx.Graph()
    for i in range(len(ctr_agg)):
        for j in ctr_agg[i]:
            travel_time = max(tsp_c[i], tsp_c[j]) + max(
                tau_disagg[m, n] for m in agg_2_disagg_id[i] for n in agg_2_disagg_id[j]
            )
            G.add_edge(i, j, weight=travel_time)

    K = len(ctr_agg)
    tau_agg = np.zeros((K, K))
    tau_agg2 = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                tau_agg[i, j] = 0
                tau_agg2[i, j] = 1
            else:
                tau_agg[i, j] = (
                    nx.shortest_path_length(G, source=i, target=j, weight="weight")
                    // config["DELTA_t"]
                )
                tau_agg2[i, j] = tau_agg[i, j]
    return tau_agg.astype("int32"), tau_agg2.astype("int32")


def update_ctr_agg(ctr_disagg, disagg_2_agg_id):
    """
    Update neighbor stations info of aggregated network
    """
    ctr_agg = dict()
    for c, neighbor in ctr_disagg.items():
        c_agg = disagg_2_agg_id[c]
        for n in neighbor:
            n_agg = disagg_2_agg_id[n]
            ctr_agg.setdefault(c_agg, set()).add(n_agg)
        ctr_agg[c_agg].add(c_agg)
    return {k: list(v) for k, v in ctr_agg.items()}


def update_driver(Driver, FR, agg_2_agg_new_bus, config):
    """
    Map fixed routes in aggregated network into disaggregated network.
    Current fixed routes will not be aggregated after graph coarsening,
    only a zone id change.

    Input:
    Route_D: aggregated route for each driver, dictionary {d: (t1, t2, s1, s2, n1, n2)}
    disagg_2_agg_id: id converter from origin aggregated network to new aggregated network
    of bus-visited zones

    Output:
    FR_new: fixed route in aggregated network
    Driver: driver info in aggregated network
    """
    # TODO: Dial-a-ride mode and flex OD mode
    Driver = pd.DataFrame(
        data=Driver, columns=["ID", "NAN", "ED", "LA", "O", "D", "Duration"]
    )

    FR_new = (
        {
            d: np.array(
                [
                    [
                        time_station[0],
                        agg_2_agg_new_bus[time_station[1]],
                    ]
                    for time_station in route
                ]
            )
            for d, route in FR.items()
        }
        if config["FIXED_ROUTE"] or config["BUILD_ON"]
        else None
    )

    # update Driver's OD info
    Driver.O = Driver.O.apply(
        lambda x: agg_2_agg_new_bus[x] if x in agg_2_agg_new_bus else x
    )
    Driver.D = Driver.D.apply(
        lambda x: agg_2_agg_new_bus[x] if x in agg_2_agg_new_bus else x
    )

    return FR_new, Driver.to_numpy(dtype=int)


# def update_driver(Driver, Route_D, disagg_2_agg_id, OS_d, DS_d, config):
#     """
#     Map fixed routes in aggregated network into disaggregated network.
#     Current fixed routes will not be aggregated after graph coarsening,
#     only a zone id change.

#     Input:
#     Route_D: disaggregated route for each driver, dictionary {d: (t1, t2, s1, s2, n1, n2)}
#     disagg_2_agg_id: id converter from disaggregated network to aggregated network

#     Output:
#     FR: fixed route in aggregated network
#     Driver: driver info in aggregated network
#     """
#     # TODO: Dial-a-ride mode!

#     # calculate tour duration for each route in disaggregated network
#     duration_d = dict()
#     # duration_d is the index of the time step of driver d
#     # visiting the last station before going back to the orgin station
#     if config["REPEATED_TOUR"]:
#         for d, time_station in Route_D.items():
#             OS = time_station[0][2]
#             ED = time_station[0][0]
#             for idx, (t1, t2, s1, s2, n1, n2) in enumerate(time_station):
#                 if t1 > ED and s1 == OS:
#                     duration_d[d] = idx - 1
#                     break

#     # update Driver data:
#     Driver = pd.DataFrame(
#         data=Driver, columns=["ID", "NAN", "ED", "LA", "O", "D", "Duration"]
#     )
#     # update FR data:
#     # FR is the dictionary {d: [t, disagg_2_agg_id[s]]} in the aggregated network,
#     # where disagg_2_agg_id[s] is supposed to be 1-to-1 mapping
#     FR = dict()
#     if config["FIXED_ROUTE"]:
#         for d in range(4):
#             FR[d] = np.array(
#                 [
#                     [time_station[0], disagg_2_agg_id[time_station[2]]]
#                     for idx, time_station in enumerate(Route_D[d])
#                     if idx <= duration_d[d]
#                 ]
#             )
#             Driver.loc[Driver.ID == d, "O"] = FR[d][0][1]
#             Driver.loc[Driver.ID == d, "D"] = FR[d][0][1]
#             Driver.loc[Driver.ID == d, "LA"] = Route_D[d][duration_d[d] + 1][0]
#             Driver.loc[Driver.ID == d, "Duration"] = Route_D[d][duration_d[d] + 1][0]
#     else:
#         FR = None

#     if config["BUILD_ON"]:  # by default BUILD_ON mode is with REPEATED_TOUR mode
#         if not isinstance(FR, dict):
#             FR = {}
#         for d, _ in config["build_on_list"].items():
#             FR[d] = np.array(
#                 [
#                     [time_station[0], disagg_2_agg_id[time_station[2]]]
#                     for idx, time_station in enumerate(Route_D[d])
#                     if idx <= duration_d[d]
#                 ]
#             )
#             Driver.loc[Driver.ID == d, "O"] = disagg_2_agg_id[OS_d[d]]
#             Driver.loc[Driver.ID == d, "D"] = disagg_2_agg_id[DS_d[d]]
#             Driver.loc[Driver.ID == d, "LA"] = Route_D[d][duration_d[d] + 1][0]
#             Driver.loc[Driver.ID == d, "Duration"] = Route_D[d][duration_d[d] + 1][0]

#     for d in config["driver_set"]:
#         Driver.loc[Driver.ID == d, "O"] = disagg_2_agg_id[OS_d[d]]
#         if config["REPEATED_TOUR"]:
#             Driver.loc[Driver.ID == d, "D"] = disagg_2_agg_id[DS_d[d]]
#         else:
#             Driver.loc[Driver.ID == d, "D"] = disagg_2_agg_id[DS_d[d]]
#         Driver.loc[Driver.ID == d, "LA"] = Route_D[d][duration_d[d] + 1][0]
#         Driver.loc[Driver.ID == d, "Duration"] = Route_D[d][duration_d[d] + 1][0]
#     return FR, Driver.astype("int32").to_numpy()


def plot_metric(Y, num_R, config, ITER):
    """
    generate distribution of detours plot and distribution of "actual travel time -
    shortest path travel time" ratio
    """
    figure_pth = config["figure_pth"]

    if not os.path.exists(figure_pth):
        os.makedirs(figure_pth)
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

    num_served = len(set(y[0] for y in Y))  # number of served rider

    Y_rdl = pd.DataFrame(Y, columns=["r", "d", "t1", "t2", "s1", "s2", "n1", "n2"])

    Y_rdl.r = Y_rdl.r.astype("int64")
    r_transfer = {}

    for (r, d) in zip(Y_rdl.r, Y_rdl.d):
        if r not in r_transfer:
            if d != (
                max(config["driver_set"]) + 1 if config["driver_set"] else 4
            ):  # exclude the dummy driver before the rider get on the first bus
                # dummy driver id be 4 when driver_set is None, i.e. benchmark case
                r_transfer[r] = {d}
        else:
            r_transfer[r].add(d)

    n_transfer = {}
    for r, D in r_transfer.items():
        if len(D) - 1 not in n_transfer:
            n_transfer[len(D) - 1] = 1
        else:
            n_transfer[len(D) - 1] += 1

    # calculate percentage:
    # n_transfer = {k: v / num_R * 100 for (k, v) in n_transfer.items()}
    n_transfer = {k: v / num_served for (k, v) in n_transfer.items()}

    # plot the distribution of transfers
    fig1, ax1 = plt.subplots()
    ax1.bar(n_transfer.keys(), n_transfer.values())
    ax1.set_xlabel("Number of transfers")
    ax1.set_ylabel("Percentage of passengers")
    # ax1.set_title("Distribution of transfers")
    ax1.set_xticks(np.arange(max(n_transfer.keys()) + 1))

    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(bottom=0, top=1.0)

    fig1.savefig(
        figure_pth + r"Distribution of transfers for riders_iter{}.png".format(ITER),
        bbox_inches="tight",
        dpi=100,
    )
    plt.close()

    (tau_agg, _) = load_tau(config)  # load shortest path travel time matrix

    trajectory_r = {
        rider: Y_rdl.loc[Y_rdl.r == rider, ["t1", "t2", "s1", "s2"]]
        for rider in Y_rdl.r.unique()
    }

    rider_info = {}
    for rider, table in trajectory_r.items():
        rider_info[rider] = {}
        rider_info[rider]["O"] = int(table.s1.iloc[0])
        rider_info[rider]["D"] = int(table.s2.iloc[-1])
        rider_info[rider]["duration"] = table.t2.iloc[-1] - table.t1.iloc[0]
        rider_info[rider]["shostest_path_duration"] = tau_agg[
            rider_info[rider]["O"], rider_info[rider]["D"]
        ]
        rider_info[rider]["duration_ratio"] = (
            (
                rider_info[rider]["duration"]
                / rider_info[rider]["shostest_path_duration"]
            )
            if rider_info[rider]["shostest_path_duration"] > 0
            else 1
            if rider_info[rider]["duration"] == 0
            else float("inf")
        )

    # plot the histogram of Travel time/Shortest path travel time
    fig2, ax2 = plt.subplots()
    ratio_list = [
        rider_info[rider]["duration_ratio"]
        for rider in rider_info.keys()
        if rider_info[rider]["duration_ratio"] < float("inf")
    ]
    ax2.hist(ratio_list, bins=20, weights=np.ones(len(ratio_list)) / len(ratio_list))
    ax2.set_xlabel("Travel time/Shortest path travel time")
    ax2.set_ylabel("Percentage of passengers")
    ax2.set_title("Distribution of travel time/Shortest path travel time")

    ax2.set_xticks(np.arange(1, 1 // config["LAMBDA"] + 1))

    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylim(bottom=0, top=1.0)

    fig2.savefig(
        figure_pth + r"Distribution of ratio{}.png".format(ITER),
        bbox_inches="tight",
        dpi=100,
    )

    plt.close()
    print("Max ATT-SPTT Ratio: {}".format(max(ratio_list)))


def plot_mr(mr_list, config):
    """
    Plot the matching rates over iterations
    """
    figure_pth = config["figure_pth"]
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(mr_list) + 1), mr_list, linewidth=1.5)
    # ax1.set_title("matching rate over iterations")
    ax1.set_ylabel("Percentage of served passengers")
    ax1.set_xlabel("Iteration")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xticks(np.arange(1, len(mr_list) + 1))
    fig1.savefig(
        figure_pth + r"matching rate.png",
        bbox_inches="tight",
        dpi=100,
    )

    plt.close()


def plot_r(R_list, config):
    """
    Plot the number of matched riders over iterations
    """
    figure_pth = config["figure_pth"]
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(R_list) + 1), R_list, linewidth=1.5)
    # ax1.set_title("number of matches over iterations")
    ax1.set_ylabel("Number of served passengers")
    ax1.set_xlabel("Iteration")
    ax1.set_xticks(np.arange(1, len(R_list) + 1))

    fig1.savefig(
        figure_pth + r"match num.png",
        bbox_inches="tight",
        dpi=100,
    )

    plt.close()


def travel_demand_plot(trips_dict, config):
    """
    Generate travel demand indegree and outdegree color map
    """
    S = config["S_disagg"]  # number of zones

    trip_sum = pd.DataFrame(columns=["zone_id", "indegree", "outdegree", "total"])
    trip_sum.zone_id = np.arange(S)
    for station in np.arange(S):
        trip_sum.loc[trip_sum.zone_id == station, "indegree"] = sum(
            [trips for (o, d), trips in trips_dict.items() if d == station]
        )
        trip_sum.loc[trip_sum.zone_id == station, "outdegree"] = sum(
            [trips for (o, d), trips in trips_dict.items() if o == station]
        )
        trip_sum.loc[trip_sum.zone_id == station, "total"] = (
            trip_sum.loc[trip_sum.zone_id == station, "outdegree"]
            + trip_sum.loc[trip_sum.zone_id == station, "indegree"]
        )
    trip_sum.to_csv(config["m2m_output_loc"] + "transit_trips_dict_pk.csv", index=False)

    gpd_shp_file = gpd.read_file(config["shapefile_zone_id"])
    gpd_shp_trip_sum = gpd_shp_file.merge(right=trip_sum, how="left", on="zone_id")

    # indegree plot
    fig = plt.figure(figsize=(30, 25))
    ax = fig.gca()

    vmin = gpd_shp_trip_sum.indegree.min()
    vmax = gpd_shp_trip_sum.indegree.max()

    gpd_shp_trip_sum.plot(
        column="indegree",
        ax=ax,
        legend=False,
        cmap="Reds",
        edgecolor="black",
    )

    cax = fig.add_axes([1, 0.1, 0.03, 0.8])

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbr = fig.colorbar(
        sm,
        cax=cax,
    )
    cbr.ax.tick_params(labelsize=30)

    fig.savefig(
        config["figure_pth"] + r"transit_trip_destinations.png",
        bbox_inches="tight",
        dpi=100,
    )
    plt.close()

    # outdegree plot
    fig = plt.figure(figsize=(30, 25))
    ax = fig.gca()

    vmin = gpd_shp_trip_sum.outdegree.min()
    vmax = gpd_shp_trip_sum.outdegree.max()

    gpd_shp_trip_sum.plot(
        column="outdegree",
        ax=ax,
        legend=False,
        cmap="Blues",
        edgecolor="black",
    )

    cax = fig.add_axes([1, 0.1, 0.03, 0.8])

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbr = fig.colorbar(
        sm,
        cax=cax,
    )
    cbr.ax.tick_params(labelsize=30)

    fig.savefig(
        config["figure_pth"] + r"transit_trip_origins.png", bbox_inches="tight", dpi=100
    )
    plt.close()

    # total demand plot
    fig = plt.figure(figsize=(30, 25))
    ax = fig.gca()

    vmin = gpd_shp_trip_sum.total.min()
    vmax = gpd_shp_trip_sum.total.max()

    gpd_shp_trip_sum.plot(
        column="total",
        ax=ax,
        legend=False,
        cmap="Purples",
        edgecolor="black",
    )

    for i, (xx, yy) in enumerate(
        zip(gpd_shp_trip_sum.centroid.x, gpd_shp_trip_sum.centroid.y), start=0
    ):
        plt.annotate(
            str(gpd_shp_trip_sum.zone_id[i]),
            (xx, yy),
            xytext=(5, 5),
            textcoords="offset points",
        )

    cax = fig.add_axes([1, 0.1, 0.03, 0.8])

    sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbr = fig.colorbar(
        sm,
        cax=cax,
    )
    cbr.ax.tick_params(labelsize=30)

    fig.savefig(
        config["figure_pth"] + r"transit_trip_total.png", bbox_inches="tight", dpi=100
    )
    plt.close()


def route_plot(Route_D, config):
    """
    Plot routes
    """
    gpd_shp_file = gpd.read_file(config["shapefile_zone_id"])
    gpd_shp_file_summary = gpd_shp_file.geometry.bounds
    lolon, uplon, lolat, uplat = (
        gpd_shp_file_summary.minx.min(),
        gpd_shp_file_summary.maxx.max(),
        gpd_shp_file_summary.miny.min(),
        gpd_shp_file_summary.maxy.max(),
    )

    bm = Basemap(
        llcrnrlon=lolon,
        llcrnrlat=lolat,
        urcrnrlon=uplon,
        urcrnrlat=uplat,
        resolution="i",
        projection="tmerc",
        lat_0=sum([uplat, lolat]) / 2.0,
        lon_0=sum([uplon, lolon]) / 2.0,
    )

    lons_dict = {
        idx: c.x for idx, c in zip(gpd_shp_file.zone_id, gpd_shp_file.centroid)
    }
    lats_dict = {
        idx: c.y for idx, c in zip(gpd_shp_file.zone_id, gpd_shp_file.centroid)
    }

    duration_d = dict()
    for d, route in Route_D.items():
        O = route[0][2]
        S1 = route[0][3]
        for idx, stamp in enumerate(route):
            if idx > 0 and stamp[2] == O and stamp[3] == S1:
                duration_d[d] = stamp[1]
                break
        if d not in duration_d:  # in case only one tour
            duration_d[d] = route[-1][1]
    s_list = {
        d: [
            (lons_dict[n1], lats_dict[n1])
            for (_, t2, n1, _, _, _) in route
            if t2 <= duration_d[d]
        ]
        for d, route in Route_D.items()
    }
    for d, stop_list in s_list.items():
        s_list[d].append(stop_list[0])

    for d, s in s_list.items():
        lons, lats = [c[0] for c in s], [c[1] for c in s]
        x, y = bm(lons, lats)
        fig = plt.figure(figsize=(30, 25))
        ax = fig.gca()
        bm.readshapefile(
            config["shapefile_zone_id"][:-4],
            "zone_id",
            ax=ax,
            linewidth=1,
        )
        bm.plot(
            x,
            y,
            color="black",
            ax=ax,
            linewidth=3.0,
        )
        fig.savefig(
            config["figure_pth"] + r"route_{}.png".format(d),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()


def plot_obj_m2m_gc(OBJ_list, config):
    """
    Plot objective values over m2m-gc iterations
    """
    figure_pth = config["figure_pth"]
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(OBJ_list) + 1), OBJ_list, linewidth=1.5)
    # ax1.set_title("matching rate over iterations")
    ax1.set_ylabel("Objective Value")
    ax1.set_xlabel("Iteration")
    ax1.set_xticks(np.arange(1, len(OBJ_list) + 1))
    fig1.savefig(
        figure_pth + r"obj_m2m_gc_{}.png".format(config["ITR_MC_M2M"]),
        bbox_inches="tight",
        dpi=100,
    )

    plt.close()


def network_plot(tau, ctr, disagg_2_agg_id, config):
    """
    Plot network
    """
    G = nx.Graph()
    for i in range(len(ctr)):
        for j in ctr[i]:
            travel_time = tau[i, j]
            G.add_edge(i, j, weight=travel_time)
    fig1 = plt.figure(figsize=(30, 25))
    ax1 = fig1.gca()
    nx.draw(
        G,
        ax=ax1,
        pos=nx.kamada_kawai_layout(G),
        with_labels=True,
    )
    fig_dir = config["figure_pth"] + "network_plot\\"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    try:
        fig1.savefig(
            fig_dir
            + r"graph_{}_{}.png".format(config["ITR_MC_M2M"], config["ITR_M2M_GC"]),
            bbox_inches="tight",
            dpi=100,
        )
    except KeyError:
        fig1.savefig(
            fig_dir + r"graph.png",
            bbox_inches="tight",
            dpi=100,
        )
    plt.close()

    # plot aggregated network
    gpd_shp_file = gpd.read_file(config["shapefile_zone_id"])
    gpd_shp_file["group"] = [disagg_2_agg_id[zone] for zone in gpd_shp_file.zone_id]

    gpd_shp_file = gpd_shp_file.dissolve(by="group")
    # copy poly series to new GeoDataFrame
    centroid_df = gpd_shp_file.copy(deep=True)
    # change the geometry to centroid of each polygon
    centroid_df.geometry = centroid_df.centroid

    fig2 = plt.figure(figsize=(30, 25))
    ax2 = fig2.gca()

    gpd_shp_file.boundary.plot(edgecolor="black", ax=ax2)
    _ = centroid_df.plot(ax=ax2, marker=".", color="black")

    for i, (xx, yy) in enumerate(
        zip(centroid_df.geometry.x, centroid_df.geometry.y), start=0
    ):
        plt.annotate(
            str(centroid_df.index[i]),
            (xx, yy),
            xytext=(5, 5),
            textcoords="offset points",
        )
    try:
        fig2.savefig(
            fig_dir
            + r"network_{}_{}.png".format(config["ITR_MC_M2M"], config["ITR_M2M_GC"]),
            bbox_inches="tight",
            dpi=100,
        )
    except KeyError:
        fig2.savefig(
            fig_dir + r"network.png",
            bbox_inches="tight",
            dpi=100,
        )
    plt.close()


if __name__ == "__main__":
    import csv
    import yaml
    from trip_prediction import trip_prediction

    with open("config_m2m_gc.yaml", "r+") as fopen:
        config = yaml.load(fopen, Loader=yaml.FullLoader)

    ctr_info = pickle.load(open(config["Station"], "rb"))  # disagg station info
    config["S"] = 39

    Route_D_agg = pickle.load(open(r"Data\temp\Route_D.p", "rb"))

    Route_D_disagg = post_processing(Route_D_agg, config)

    result_loc = r"E:many2many_output\\test\\"
    if not os.path.exists(result_loc):
        os.makedirs(result_loc)

    for d in Route_D_disagg.keys():
        route_filename = result_loc + r"route_{}_agg.csv".format(d)
        with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
            for row in Route_D_disagg[d]:
                csv_out.writerow(row)

    # od_travel_time, od_travel_dist = update_time_dist(Route_D_disagg, config)
    # print(od_travel_time)

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
    N_n, L_l, C_l, K_k, w_out, w_in, w_f = get_link_cost(
        trips_dict_pk, Route_D_disagg, config
    )
    print(min(C_l.values()))

    ctr_agg = load_neighbor(config)
    tau_disagg, _ = load_tau_disagg(config)
    agg_2_disagg_id = pickle.load(open(config["agg_2_disagg_id"], "rb"))

    tau_agg, tau_agg2 = update_tau_agg(ctr_agg, tau_disagg, agg_2_disagg_id, config)
    print(tau_agg)

# %%
