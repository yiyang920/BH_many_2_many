import csv, pickle, operator, os, yaml, math
import pandas as pd
import numpy as np
import networkx as nx
from st_network3 import Many2Many
from collections import defaultdict
from trip_prediction import trip_prediction
from graph_coarsening_local_search import local_search
from direct_disagg import direct_disagg
from utils import (
    disagg_2_agg_trip,
    disagg_trip_get_rider,
    get_driver_disagg,
    get_driver_m2m_gc,
    get_rider,
    load_FR_m2m_gc,
    load_mc_input_2,
    # load_neighbor,
    load_neighbor_disagg,
    get_link_set_disagg,
    load_scen_FR,
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

# Initialize Parameters
with open("config_agg_disagg.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)

if not os.path.exists(config["figure_pth"]):
    os.makedirs(config["figure_pth"])
if not os.path.exists(config["m2m_output_loc"]):
    os.makedirs(config["m2m_output_loc"])

# Load data
DELTA_t = config["DELTA_t"]
N, K = config["S_disagg"], config["K"]
print("Initializing aggregated network...")
# Generate link set of the disaggregated zones
_, _ = get_link_set_disagg(config)
# Load existing bus routes
FR_origin, V_bus = load_scen_FR(config)

# Load neighbor nodes information of disaggregated zones
ctr_disagg = load_neighbor_disagg(config)
# Load shortest travel time matrices of disaggregated zones
tau_disagg, tau2_disagg = load_tau_disagg(config)

# Initialize an aggregated network

# Load mode choice input data
# Load mode choice input data
(
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
) = load_mc_input_2(config)

if not config["DEMAND_MODEL"]:
    try:
        ttrips_mat = pickle.load(
            open(config["m2m_data_loc"] + "ttrips_mat.p", "rb"),
        )
    except:
        print("transit trip table not found, running travel demand model...")
        ttrips_mat = trip_prediction(
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
else:
    ttrips_mat = trip_prediction(
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
print("trip generation finished, entering optimization model...")

# get residual graph and edge set
TN, E = get_link_set_disagg(config, V_exclude=V_bus)
PV, VP = dict(), dict()
for comp in nx.connected_components(TN.to_undirected()):
    num_partition = len(PV)
    if len(comp) > config["K"]:
        K = math.floor(config["K"] / config["S_disagg"] * len(comp))
        TN_sub = TN.subgraph(comp).copy()
        PV_sub, VP_sub = local_search(TN_sub, ttrips_mat, N, K, config, tau2_disagg)
        PV.update({k + num_partition: v for k, v in PV_sub.items()})
    else:
        PV.update({num_partition: set(comp)})
num_partition = len(PV)
PV.update({num_partition + i: {v} for i, v in enumerate(V_bus)})
VP = {v: c for c, p in PV.items() for v in p}
# Update agg_2_disagg_id and disagg_2_agg_id
agg_2_disagg_id, disagg_2_agg_id = PV, VP
# Update config dictionary: number of partitions of aggregated network
config["S"] = len(agg_2_disagg_id)

ctr = update_ctr_agg(ctr_disagg, disagg_2_agg_id)
# Get shortest travel time matrices of initial aggregated zones
tau, tau2 = update_tau_agg(ctr, tau_disagg, agg_2_disagg_id, config)
np.savetxt(
    config["m2m_output_loc"] + "tau_agg_{}_{}.csv".format(0, 0), tau, delimiter=","
)

file_dir = config["m2m_output_loc"] + "init_graph\\"
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

pickle.dump(ctr, open(file_dir + "ctr_agg.p", "wb"))
pickle.dump(ctr_disagg, open(file_dir + "ctr_disagg.p", "wb"))
pickle.dump(agg_2_disagg_id, open(file_dir + "agg_2_disagg_id.p", "wb"))
pickle.dump(disagg_2_agg_id, open(file_dir + "disagg_2_agg_id.p", "wb"))

# Load Rider
trip_dict = disagg_2_agg_trip(
    ttrips_mat, config, disagg_2_agg_id=disagg_2_agg_id, fraction=3.0 / 24
)
Rider = get_rider(trip_dict, config)
# Load Driver
Driver = get_driver_m2m_gc(disagg_2_agg_id, config)
FR = {
    d: [
        (t1, t2, disagg_2_agg_id[s1], disagg_2_agg_id[s2])
        for (t1, t2, s1, s2, *_) in route
    ]
    for d, route in FR_origin.items()
}
# Diver origin and destination data in disaggregated network
OD_d = pd.read_csv(
    config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
).to_dict("index")

MR_list, R_list, OBJ_set_m2m_gc, OBJ_set_mc_m2m = list(), list(), list(), list()
OBJ = float("infinity")

ITER_LIMIT_M2M_GC = config["ITER_LIMIT_M2M_GC"]
while ITER_LIMIT_M2M_GC:
    print(
        "entering m2m-gc iterations {}, number of aggregated zones {}...".format(
            config["ITER_LIMIT_M2M_GC"] + 1 - ITER_LIMIT_M2M_GC,
            len(agg_2_disagg_id),
        )
    )
    config["ITR_M2M_GC"] = config["ITER_LIMIT_M2M_GC"] - ITER_LIMIT_M2M_GC
    # Run many-to-many model
    if not config["DEBUG_MODE"]:
        (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
            Rider, Driver, tau, tau2, ctr, config, fixed_route_D=FR
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

        pickle.dump(Driver, open(r"Data\temp\Driver.p", "wb"))
        pickle.dump(tau, open(r"Data\temp\tau_agg.p", "wb"))

    else:
        # load existing results for debugging
        X = pickle.load(open(r"Data\temp\X.p", "rb"))
        U = pickle.load(open(r"Data\temp\U.p", "rb"))
        Y = pickle.load(open(r"Data\temp\Y.p", "rb"))
        Route_D = pickle.load(open(r"Data\temp\Route_D.p", "rb"))
        mr = pickle.load(open(r"Data\temp\mr.p", "rb"))
        OBJ = pickle.load(open(r"Data\temp\OBJ.p", "rb"))
        R_match = pickle.load(open(r"Data\temp\R_match.p", "rb"))

    if OBJ in OBJ_set_m2m_gc:
        OBJ_set_m2m_gc.append(OBJ)
        break
    OBJ_set_m2m_gc.append(OBJ)
    # record matching rate, number of matches, and objective value
    MR_list.append(mr)
    R_list.append(len(R_match))
    # Post-processing: disaggregate Route_D
    Route_D_disagg = direct_disagg(
        Route_D,
        OD_d,
        Driver,
        tau_disagg,
        ctr_disagg,
        agg_2_disagg_id,
        disagg_2_agg_id,
        config,
    )
    if config["FIXED_ROUTE"]:
        # Update route with the fixed route schedule
        Route_D_disagg.update(FR_origin)

    # get residual graph and edge set
    V_bus = set(s1 for route in Route_D_disagg.values() for _, _, s1, *_ in route)
    TN, E = get_link_set_disagg(config, V_exclude=V_bus)
    PV, VP = dict(), dict()
    for comp in nx.connected_components(TN.to_undirected()):
        num_partition = len(PV)
        if len(comp) > config["K"]:
            K = math.floor(config["K"] / config["S_disagg"] * len(comp))
            TN_sub = TN.subgraph(comp).copy()
            PV_sub, VP_sub = local_search(TN_sub, ttrips_mat, N, K, config, tau2_disagg)
            PV.update({k + num_partition: v for k, v in PV_sub.items()})
        else:
            PV.update({num_partition: set(comp)})
    num_partition = len(PV)
    PV.update({num_partition + i: {v} for i, v in enumerate(V_bus)})
    VP = {v: c for c, p in PV.items() for v in p}
    # Update agg_2_disagg_id and disagg_2_agg_id
    agg_2_disagg_id, disagg_2_agg_id = PV, VP
    # Update config dictionary: number of partitions of aggregated network
    config["S"] = len(agg_2_disagg_id)

    ctr = update_ctr_agg(ctr_disagg, disagg_2_agg_id)
    # Get shortest travel time matrices of initial aggregated zones
    tau, tau2 = update_tau_agg(ctr, tau_disagg, agg_2_disagg_id, config)
    np.savetxt(
        config["m2m_output_loc"] + "tau_agg_{}_{}.csv".format(0, 0), tau, delimiter=","
    )

    pickle.dump(ctr, open(file_dir + "ctr_agg.p", "wb"))
    pickle.dump(ctr_disagg, open(file_dir + "ctr_disagg.p", "wb"))
    pickle.dump(agg_2_disagg_id, open(file_dir + "agg_2_disagg_id.p", "wb"))
    pickle.dump(disagg_2_agg_id, open(file_dir + "disagg_2_agg_id.p", "wb"))

    # Load Rider
    trip_dict = disagg_2_agg_trip(
        ttrips_mat, config, disagg_2_agg_id=disagg_2_agg_id, fraction=3.0 / 24
    )
    Rider = get_rider(trip_dict, config)
    # Load Driver
    Driver = get_driver_m2m_gc(disagg_2_agg_id, config)
    FR = {
        d: [
            (t1, t2, disagg_2_agg_id[s1], disagg_2_agg_id[s2])
            for (t1, t2, s1, s2, *_) in route
        ]
        for d, route in FR_origin.items()
    }
    ITER_LIMIT_M2M_GC -= 1

# plot matching rate curve and number of matched riders curve
plot_mr(MR_list, config)
plot_r(R_list, config)
# plot travel demand
travel_demand_plot(trip_dict, config)
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
    trip_dict,
    open(config["m2m_data_loc"] + "trip_dict.p", "wb"),
)

# Record config info
with open(result_loc + "config.yaml", "w") as fwrite:
    yaml.dump(config, fwrite)

print("Finished!")
