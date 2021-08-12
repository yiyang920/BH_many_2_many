"""
Travel-Demand model to Many-To-Many model iterations

Version 2:
    New trip-based travel demand model.
"""
# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# pylint: disable=E0611, E1101
# %% import packages
import pandas as pd
import numpy as np
import pickle
import operator
import os
import csv
import yaml

from st_network2 import Many2Many
from trip_prediction2 import trip_prediction

from utils import (
    load_mc_input_2,
    load_neighbor,
    load_tau,
    disagg_2_agg_trip,
    get_rider,
    load_FR,
    load_FR_disagg,
    get_driver,
    post_processing,
    plot_metric,
    plot_mr,
    plot_r,
    travel_demand_plot,
    route_plot,
)

# Debug mode will skip many-to-many problem and load exisiting optimization results
# %% Initialize Parameters
with open("config_mc_m2m_2.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)

# %% Load data
DELTA_t = config["DELTA_t"]
# Load neighbor nodes information (agg)
ctr = load_neighbor(config)
# Load shortest travel time matrices (agg)
tau, tau2 = load_tau(config)

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

# reversed ID converter
id_converter_reverse = {geoid: zoneid for zoneid, geoid in id_converter.items()}

# %% Mode choice - many to many problem iteration
# Generate driver information
# Load fixed routed infomation
FR = load_FR(config)
Route_fr_disagg = load_FR_disagg(config)
Driver = get_driver(config)

MR_list = list()
R_list = list()
OBJ_set = set()
OBJ = None
ITER_LIMIT = config["ITER_LIMIT"]

while ITER_LIMIT and not (OBJ and OBJ_set and OBJ in OBJ_set):
    print("entering iteration {}...".format(config["ITER_LIMIT"] + 1 - ITER_LIMIT))

    if OBJ:
        OBJ_set.add(OBJ)
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
    # Generate Rider and Driver Information
    # convert disaggregated trips into aggregated trips
    # Rider info: Rounding trip number with threshold 0.5
    # multiply by 0.6 representing morning peak hour 7-10am
    trip_dict = disagg_2_agg_trip(ttrips_mat, config, fraction=3.0 / 24)
    Rider = get_rider(trip_dict, config)

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

    # record matching rate, number of matches, and objective value
    MR_list.append(mr)
    R_list.append(len(R_match))

    # Post-processing: disaggregate Route_D
    Route_D_disagg = post_processing(Route_D, config)
    if config["FIXED_ROUTE"]:
        # Update route with the fixed route schedule
        Route_D_disagg.update(Route_fr_disagg)

    # Plot metrics
    plot_metric(Y, np.size(Rider, 0), config, config["ITER_LIMIT"] - ITER_LIMIT)

    ITER_LIMIT -= 1


# Plot matching rate curve and number of matched riders curve
plot_mr(MR_list, config)
plot_r(R_list, config)

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
    ttrips_mat,
    open(config["m2m_data_loc"] + "ttrips_mat.p", "wb"),
)

# Record config info
with open(result_loc + "config.yaml", "w") as fwrite:
    yaml.dump(config, fwrite)

# plot travel demand
travel_demand_plot(ttrips_mat, config)
# plot routes
route_plot(Route_D_disagg, config)

print("Finished!")
