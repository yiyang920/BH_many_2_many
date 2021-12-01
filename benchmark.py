import csv, pickle, operator, os, yaml, time
import pandas as pd
import numpy as np
import networkx as nx
from st_network3_1 import Many2Many
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
    get_rider_agg_diagg,
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
with open("config_benchmark.yaml", "r+") as fopen:
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
FR, V_bus = load_scen_FR(config)

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
        ttrips_dict = pickle.load(
            open(config["m2m_data_loc"] + "ttrips_mat.p", "rb"),
        )
    except:
        print("transit trip table not found, running travel demand model...")
        ttrips_dict = trip_prediction(
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
    ttrips_dict = trip_prediction(
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

disagg_2_agg_id = {v: v for v in range(config["S_disagg"])}
# Load Rider
Rider = get_rider_agg_diagg(
    ttrips_dict,
    config,
    tau_disagg,
    disagg_2_agg_id={v: v for v in range(config["S_disagg"])},
    fraction=3.0 / 24,
)
# Load Driver
# Load Driver
Driver = get_driver_m2m_gc(disagg_2_agg_id, config)
# Load existing bus routes
if config["FIXED_ROUTE"]:
    FR_origin, V_bus = load_scen_FR(config)
else:
    FR_origin, V_bus = dict(), set()
FR = {
    d: [
        (t1, t2, disagg_2_agg_id[s1], disagg_2_agg_id[s2])
        for (t1, t2, s1, s2, *_) in route
    ]
    for d, route in FR_origin.items()
}
OD_d = pd.read_csv(
    config["m2m_data_loc"] + "Driver_OD_disagg.csv", index_col=["ID"]
).to_dict("index")

# Run many-to-many model
if not config["DEBUG_MODE"]:
    tic = time.perf_counter()
    (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
        Rider,
        Driver,
        tau_disagg,
        tau2_disagg,
        ctr_disagg,
        config,
        fixed_route_D=FR,
    )
    toc = time.perf_counter() - tic
    print(
        "optimization finished, matching rate: {}%, time elapsed: {}s".format(
            mr * 100, toc
        )
    )

    # save results for debugging
    pickle.dump(X, open(r"Data\temp\X.p", "wb"))
    pickle.dump(U, open(r"Data\temp\U.p", "wb"))
    pickle.dump(Y, open(r"Data\temp\Y.p", "wb"))
    pickle.dump(Route_D, open(r"Data\temp\Route_D.p", "wb"))
    pickle.dump(mr, open(r"Data\temp\mr.p", "wb"))
    pickle.dump(OBJ, open(r"Data\temp\OBJ.p", "wb"))
    pickle.dump(R_match, open(r"Data\temp\R_match.p", "wb"))

    pickle.dump(Driver, open(r"Data\temp\Driver.p", "wb"))
    pickle.dump(tau_disagg, open(r"Data\temp\tau_agg.p", "wb"))

else:
    # load existing results for debugging
    X = pickle.load(open(r"Data\temp\X.p", "rb"))
    U = pickle.load(open(r"Data\temp\U.p", "rb"))
    Y = pickle.load(open(r"Data\temp\Y.p", "rb"))
    Route_D = pickle.load(open(r"Data\temp\Route_D.p", "rb"))
    mr = pickle.load(open(r"Data\temp\mr.p", "rb"))
    OBJ = pickle.load(open(r"Data\temp\OBJ.p", "rb"))
    R_match = pickle.load(open(r"Data\temp\R_match.p", "rb"))

Route_D_disagg = Route_D
Route_D_disagg.update(FR)
# plot travel demand
travel_demand_plot(ttrips_dict, config)
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
    ttrips_dict,
    open(config["m2m_data_loc"] + "trip_dict.p", "wb"),
)

# Record config info
with open(result_loc + "config.yaml", "w") as fwrite:
    yaml.dump(config, fwrite)

print("Finished!")
