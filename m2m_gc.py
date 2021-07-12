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
import csv
import yaml
from st_network3 import Many2Many
from trip_prediction import trip_prediction
from init_partition import init_partition
from graph_coarsening import graph_coarsening
from route_disagg import route_disagg
from utils import (
    disagg_trip_get_rider,
    get_driver_m2m_gc,
    load_FR_m2m_gc,
    # load_neighbor,
    load_neighbor_disagg,
    get_link_set_disagg,
    load_tau_disagg,
    load_mc_input,
    # disagg_2_agg_trip,
    # get_rider,
    get_link_cost,
    # load_FR,
    load_FR_disagg,
    # get_driver,
    network_plot,
    plot_obj_m2m_gc,
    route_plot,
    travel_demand_plot,
    update_ctr_agg,
    update_driver,
    update_time_dist,
    post_processing,
    update_tau_agg,
    plot_metric,
    plot_mr,
    plot_r,
    # get_driver_OD,
)

# %% Initialize Parameters
with open("config_m2m_gc.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)


# %% Load data
DELTA_t = config["DELTA_t"]
print("Initializing aggregated network...")
# Generate link set of the disaggregated zones
_, _ = get_link_set_disagg(config)

# Load neighbor nodes information of disaggregated zones
ctr_disagg = load_neighbor_disagg(config)
# Load shortest travel time matrices of disaggregated zones
tau_disagg, tau2_disagg = load_tau_disagg(config)

# Initialize an aggregated network
(
    ctr,
    agg_2_disagg_id,
    disagg_2_agg_id,
) = init_partition(tau_disagg, config)

# Load mode choice input data
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

# # Load inital graph partition
# agg_2_disagg_id = pickle.load(open(config["agg_2_disagg_id"], "rb"))
# disagg_2_agg_id = pickle.load(open(config["disagg_2_agg_id"], "rb"))

# Get shortest travel time matrices of initial aggregated zones
tau, tau2 = update_tau_agg(ctr, tau_disagg, agg_2_disagg_id, config)
np.savetxt(
    config["m2m_output_loc"] + "tau_agg_{}_{}.csv".format(0, 0), tau, delimiter=","
)
# list of origins from mdot_dat trip data
O_p = [
    id_converter_reverse[o] if o in id_converter_reverse else 9999
    for o in mdot_dat.geoid_o
]

# 0: walk, 1: dar, 2: bus, 3: share, 4: auto
D_p = {}  # dict of destination for each transportation mode from mdot_dat trip data
D_p[2] = [
    id_converter_reverse[d] if d in id_converter_reverse else 9999
    for d in dests_geo.geoid3
]


# %% Mode choice - many to many problem iteration
# Generate driver information
# Load fixed routed infomation
FR, FR_disagg, Driver = load_FR_m2m_gc(
    agg_2_disagg_id, disagg_2_agg_id, config
)  # aggregated routes
Route_fr_disagg = load_FR_disagg(config)  # disaggregated routes
# Diver origin and destination data in disaggregated network
OD_d = pd.read_csv(
    config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
).to_dict("index")
# Driver = get_driver_m2m_gc(disagg_2_agg_id, config)  # aggregated version
# Load orgin and destination stations of drivers in disaggregated network
# OS_d, DS_d = get_driver_OD(config)

ITER_LIMIT_MC_M2M = config["ITER_LIMIT_MC_M2M"]

MR_list = {i: list() for i in np.arange(ITER_LIMIT_MC_M2M)}
R_list = {i: list() for i in np.arange(ITER_LIMIT_MC_M2M)}
OBJ_set_m2m_gc = {i: list() for i in np.arange(ITER_LIMIT_MC_M2M)}
OBJ_set_mc_m2m = list()
OBJ = None

while ITER_LIMIT_MC_M2M and not (OBJ and OBJ in set(OBJ_set_mc_m2m)):
    print(
        "entering mc-m2m iteration {}...".format(
            config["ITER_LIMIT_MC_M2M"] + 1 - ITER_LIMIT_MC_M2M
        )
    )
    if OBJ:
        OBJ_set_mc_m2m.append(OBJ)

    config["ITR_MC_M2M"] = config["ITER_LIMIT_MC_M2M"] - ITER_LIMIT_MC_M2M
    (
        trips_dict_pk,
        trips_dict_op,
        transit_trips_dict_pk,
        transit_trips_dict_op,
    ) = trip_prediction(
        id_converter, per_dist, per_emp, mdot_dat, dests_geo, D, per_time=per_time
    )
    print("trip generation finished!")
    # Generate Inital Rider and Driver Information
    # convert disaggregated trips into aggregated trips
    # Rider info: Rounding trip number with threshold 0.5
    # multiply by 0.6 representing morning peak hour 7-10am
    SP_r, Rider = disagg_trip_get_rider(
        transit_trips_dict_pk, config, tau_disagg, disagg_2_agg_id=disagg_2_agg_id
    )

    ITER_LIMIT_M2M_GC = config["ITER_LIMIT_M2M_GC"]
    OBJ = None
    while ITER_LIMIT_M2M_GC and not (
        OBJ and OBJ in OBJ_set_m2m_gc[config["ITR_MC_M2M"]]
    ):
        print(
            "entering m2m-gc iterations {}, number of aggregated zones {}...".format(
                config["ITER_LIMIT_M2M_GC"] + 1 - ITER_LIMIT_M2M_GC,
                len(agg_2_disagg_id),
            )
        )
        if OBJ:
            OBJ_set_m2m_gc[config["ITR_MC_M2M"]].append(OBJ)

        config["ITR_M2M_GC"] = config["ITER_LIMIT_M2M_GC"] - ITER_LIMIT_M2M_GC
        # plot current network
        network_plot(tau, ctr, disagg_2_agg_id, config)
        if not config["DEBUG_MODE"]:
            (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
                Rider,
                Driver,
                tau,
                tau2,
                ctr,
                config,
                fixed_route_D=FR,
                SP_r=SP_r,
            )
            print("optimization finished, matching rate: {}%".format(mr * 100))

            # save results for debugging
            pickle.dump(X, open(r"Data\temp\X.p", "wb"))
            pickle.dump(U, open(r"Data\temp\U.p", "wb"))
            pickle.dump(Y, open(r"Data\temp\Y.p", "wb"))
            pickle.dump(Route_D, open(r"Data\temp\Route_D.p", "wb"))
            pickle.dump(mr, open(r"Data\temp\mr.p", "wb"))
            pickle.dump(OBJ, open(r"Data\temp\OBJ.p", "wb"))
            pickle.dump(R_match, open(r"Data\temp\R_match.p", "wb"))
            pickle.dump(agg_2_disagg_id, open(r"Data\temp\agg_2_disagg_id.p", "wb"))
            pickle.dump(disagg_2_agg_id, open(r"Data\temp\disagg_2_agg_id.p", "wb"))
        else:
            # load existing results for debugging
            X = pickle.load(open(r"Data\temp\X.p", "rb"))
            U = pickle.load(open(r"Data\temp\U.p", "rb"))
            Y = pickle.load(open(r"Data\temp\Y.p", "rb"))
            Route_D = pickle.load(open(r"Data\temp\Route_D.p", "rb"))
            mr = pickle.load(open(r"Data\temp\mr.p", "rb"))
            OBJ = pickle.load(open(r"Data\temp\OBJ.p", "rb"))
            R_match = pickle.load(open(r"Data\temp\R_match.p", "rb"))
            agg_2_disagg_id = pickle.load(open(r"Data\temp\agg_2_disagg_id.p", "rb"))
            disagg_2_agg_id = pickle.load(open(r"Data\temp\disagg_2_agg_id.p", "rb"))
        # record matching rate, number of matches, and objective value
        MR_list[config["ITR_MC_M2M"]].append(mr)
        R_list[config["ITR_MC_M2M"]].append(len(R_match))

        # Post-processing: disaggregate Route_D
        Route_D_disagg = post_processing(
            Route_D, config, agg_2_disagg_id=agg_2_disagg_id
        )
        if config["FIXED_ROUTE"]:
            # Update route with the fixed route schedule
            Route_D_disagg.update(Route_fr_disagg)
        # print("begin route disaggregation...")
        # _, Route_D_disagg, _ = route_disagg(
        #     Route_D,
        #     agg_2_disagg_id,
        #     disagg_2_agg_id,
        #     OD_d,
        #     tau_disagg,
        #     tau2_disagg,
        #     ctr_disagg,
        #     config,
        #     fixed_route_D=FR_disagg,
        # )
        # print("route disaggregation finished!")

        # Record current agg2disagg of those bus-visited zones
        agg_2_disagg_id_bus = dict()
        for s in set(s[2] for route in Route_D_disagg.values() for s in route):
            agg_2_disagg_id_bus.setdefault(disagg_2_agg_id[s], list()).append(s)

        (
            N_n,
            L_l,
            C_l,
            K_k,
            w_out,
            w_in,
            w_sum,
            w_f,
        ) = get_link_cost(trips_dict_pk, tau_disagg, Route_D_disagg, config)

        (_, _, P_N, P_L, _) = graph_coarsening(N_n, w_sum, L_l, C_l, K_k, config)

        # Update agg_2_disagg_id and disagg_2_agg_id
        agg_2_disagg_id = dict()
        # Mapping from origin aggregated zone id to new aggregated zone id
        # of bus-visited aggregated zones, should be 1-to-1 mapping
        agg_2_agg_new_bus = dict()
        for idx, c in enumerate(P_N.keys()):
            agg_2_disagg_id[idx] = P_N[c]
        # P_N excludes the zones with bus service, need to add those excluded zones
        # into agg_2_disagg_id
        for idx, (part, nodes) in enumerate(agg_2_disagg_id_bus.items()):
            new_part_id = len(P_N) + idx
            agg_2_disagg_id[new_part_id] = nodes
            agg_2_agg_new_bus[part] = new_part_id

        disagg_2_agg_id = {
            n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
        }

        # Update config dictionary: number of partitions of aggregated network
        config["S"] = len(agg_2_disagg_id)
        # Update ctr, i.e. station neighbor info of aggregated network
        ctr = update_ctr_agg(ctr_disagg, disagg_2_agg_id)

        # Update tau matrix of aggregated network
        tau, tau2 = update_tau_agg(ctr, tau_disagg, agg_2_disagg_id, config)
        np.savetxt(
            config["m2m_output_loc"]
            + "tau_agg_{}_{}.csv".format(config["ITR_MC_M2M"], config["ITR_M2M_GC"]),
            tau,
            delimiter=",",
        )
        # Update rider data
        SP_r, Rider = disagg_trip_get_rider(
            transit_trips_dict_pk, config, tau_disagg, disagg_2_agg_id=disagg_2_agg_id
        )
        # Update driver data
        # Current fixed routes will not be aggregated after graph coarsening!
        # Map fixed routes in aggregated network into disaggregated network
        FR, Driver = update_driver(Driver, FR, agg_2_agg_new_bus, config)
        # FR, Driver = update_driver(
        #     Driver, Route_D_disagg, disagg_2_agg_id, OS_d, DS_d, config
        # )

        ITER_LIMIT_M2M_GC -= 1

    if OBJ:
        OBJ_set_m2m_gc[config["ITR_MC_M2M"]].append(OBJ)

    # Update travel time and travel distance table
    od_travel_time, od_travel_dist = update_time_dist(Route_D_disagg, config)

    per_time[2] = [
        od_travel_time[o, d] if (o != 9999) and (d != 9999) else 9999
        for (o, d) in zip(O_p, D_p[2])
    ]
    per_dist.dist3 = [
        od_travel_dist[o, d] if (o != 9999) and (d != 9999) else per_dist.dist3.iloc[i]
        for (i, (o, d)) in enumerate(zip(O_p, D_p[2]))
    ]

    # Plot objective values over m2m-gc iterations
    plot_obj_m2m_gc(OBJ_set_m2m_gc[config["ITR_MC_M2M"]], config)
    # Plot metrics
    plot_metric(Y, np.size(Rider, 0), config, config["ITR_MC_M2M"])

    ITER_LIMIT_MC_M2M -= 1

if OBJ:
    OBJ_set_mc_m2m.append(OBJ)
# plot matching rate curve and number of matched riders curve
plot_mr(
    [MR_list[i][-1] for i in range(config["ITER_LIMIT_MC_M2M"]) if MR_list[i]], config
)
plot_r([R_list[i][-1] for i in range(config["ITER_LIMIT_MC_M2M"]) if R_list[i]], config)
# plot travel demand
travel_demand_plot(transit_trips_dict_pk, config)
# plot routes
route_plot(Route_D_disagg, config)
# %% Store results
result_loc = config["m2m_output_loc"]
if not os.path.exists(result_loc):
    os.makedirs(result_loc)

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

U = sorted(list(U), key=operator.itemgetter(1, 0))

for d in Route_D_disagg.keys():

    route_filename = result_loc + r"route_{}_disagg.csv".format(d)
    with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Route_D_disagg[d]:
            csv_out.writerow(row)

for d in Route_D.keys():

    route_filename = result_loc + r"route_{}_agg.csv".format(d)
    with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Route_D[d]:
            csv_out.writerow(row)

RDL_filename = result_loc + r"Y_rdl_agg.csv"
with open(RDL_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d", "t1", "t2", "s1", "s2", "n1", "n2"])
    for row in Y:
        csv_out.writerow(row)
    csv_out.writerow(
        [
            "Matching Rate: {}".format(mr),
        ]
    )

RD_filename = result_loc + r"U_rd_agg.csv"
with open(RD_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d"])
    for row in U:
        csv_out.writerow(row)
    csv_out.writerow(
        [
            "Matching Rate: {}".format(mr),
        ]
    )

R_match = pd.DataFrame(data=sorted(list(R_match)), columns=["r"])
R_match_filename = result_loc + r"R_matc_agg.csv"
R_match.to_csv(R_match_filename, index=False)

pickle.dump(
    transit_trips_dict_pk,
    open(config["m2m_data_loc"] + "transit_trips_dict_pk.p", "wb"),
)

# Record config info
with open(result_loc + "config.yaml", "w") as fwrite:
    yaml.dump(config, fwrite)

print("Finished!")
