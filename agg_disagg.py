import csv, pickle, operator, os, yaml
import pandas as pd
import numpy as np
import networkx as nx
from st_network3 import Many2Many
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
FR, V = load_scen_FR(config)

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
TN, E = get_link_set_disagg(config, V_exclude=V)
PV, VP = local_search(TN, ttrips_mat, N, K, config, V_exclude=V)

# Update agg_2_disagg_id and disagg_2_agg_id
agg_2_disagg_id = dict()

for idx, c in enumerate(PV.keys()):
    agg_2_disagg_id[idx] = PV[c]
for idx, node in enumerate(V):
    new_part_id = len(PV) + idx
    agg_2_disagg_id[new_part_id] = {node}

disagg_2_agg_id = {
    n: partition for partition, nodes in agg_2_disagg_id.items() for n in nodes
}

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
# Diver origin and destination data in disaggregated network
OD_d = pd.read_csv(
    config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
).to_dict("index")

MR_list, R_list, OBJ_set_m2m_gc, OBJ_set_mc_m2m = dict(), dict(), dict(), list()
OBJ = None

ITER_LIMIT_M2M_GC = config["ITER_LIMIT_M2M_GC"]
while ITER_LIMIT_M2M_GC and not (OBJ and OBJ in OBJ_set_m2m_gc[config["ITR_MC_M2M"]]):
    print(
        "entering m2m-gc iterations {}, number of aggregated zones {}...".format(
            config["ITER_LIMIT_M2M_GC"] + 1 - ITER_LIMIT_M2M_GC,
            len(agg_2_disagg_id),
        )
    )
    if OBJ:
        OBJ_set_m2m_gc[config["ITR_MC_M2M"]].append(OBJ)
    config["ITR_M2M_GC"] = config["ITER_LIMIT_M2M_GC"] - ITER_LIMIT_M2M_GC
    # Run many-to-many model
    if not config["DEBUG_MODE"]:
        (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
            Rider, Driver, tau, tau2, ctr, V, config, fixed_route_D=FR
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
