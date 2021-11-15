"""
Many-To-Many model for Dial-a-Ride service

Version 3:
    Substract matched riders by buses, 
    and run many-2-many for the rest of the rider.
    Simulation on disaggregated network.
"""
# import packages
import pandas as pd
import numpy as np
import pickle, csv, operator, os, yaml

from st_network4 import Many2Many

# from trip_prediction2 import trip_prediction
from utils import (
    load_mc_input_2,
    load_neighbor_disagg,
    load_tau_disagg,
    disagg_2_agg_trip,
    get_rider,
    get_driver_disagg,
    # load_FR_disagg,
    # get_driver,
    plot_metric_disagg,
    # post_processing,
    # plot_metric,
    plot_mr,
    plot_r,
    travel_demand_plot,
    route_plot,
    load_scen_FR,
)

# Initialize Parameters
with open("config_DaR.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)
if not os.path.exists(config["m2m_data_loc"]):
    os.makedirs(config["m2m_data_loc"])
if not os.path.exists(config["m2m_output_loc"]):
    os.makedirs(config["m2m_output_loc"])
if not os.path.exists(config["figure_pth"]):
    os.makedirs(config["figure_pth"])
# Load data
DELTA_t = config["DELTA_t"]
# Load neighbor nodes information (disagg)
ctr = load_neighbor_disagg(config)
# Load shortest travel time matrices (agg)
tau, tau2 = load_tau_disagg(config)

# disaggregate network, dis_2_agg is 1-to-1 mapping
disagg_2_agg_id = {k: k for k in range(config["S_disagg"])}

# Mode choice - many to many problem iteration
# Generate driver information
# Load DaR routed infomation
Driver = get_driver_disagg(config)

MR_list = list()
R_list = list()
OBJ_set = set()
OBJ = None
ITER_LIMIT = config["ITER_LIMIT"]
trip_mat = np.zeros((config["S_disagg"], config["S_disagg"]))
while ITER_LIMIT and not (OBJ and OBJ_set and OBJ in OBJ_set):
    print("entering iteration {}...".format(config["ITER_LIMIT"] + 1 - ITER_LIMIT))

    if OBJ:
        OBJ_set.add(OBJ)
    Rider = pd.read_csv(config["m2m_data_loc"] + "Rider.csv", index_col=["ID"])
    Rider_matched = list(pd.read_csv(config["m2m_data_loc"] + "U_rd_disagg.csv").r)
    Rider = Rider.drop(Rider_matched).to_numpy(dtype=int, na_value=999)
    for r in Rider:
        o_r, d_r = r[3], r[4]
        trip_mat[o_r][d_r] = trip_mat[o_r][d_r] + 1
    trip_mat = np.ceil(trip_mat / 3).astype(int)
    trip_dict = {
        (i, j): trip_mat[i][j]
        for i in range(len(trip_mat))
        for j in range(len(trip_mat[0]))
        if trip_mat[i][j]
    }
    Rider = get_rider(trip_dict, config)
    print("trip generation finished, entering optimization model...")

    # Run many-to-many model
    if not config["DEBUG_MODE"]:
        (X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
            Rider,
            Driver,
            tau,
            tau2,
            ctr,
            set(range(config["S_disagg"])),
            config,
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
    # Plot metrics
    n_transfer, ratio_list = plot_metric_disagg(
        Y, np.size(Rider, 0), config, config["ITER_LIMIT"] - ITER_LIMIT
    )
    pickle.dump(
        n_transfer,
        open(
            config["m2m_output_loc"]
            + "n_transfer_{}.p".format(config["ITER_LIMIT"] - ITER_LIMIT),
            "wb",
        ),
    )
    pickle.dump(
        ratio_list,
        open(
            config["m2m_output_loc"]
            + "ratio_list_{}.p".format(config["ITER_LIMIT"] - ITER_LIMIT),
            "wb",
        ),
    )

    ITER_LIMIT -= 1


# Plot matching rate curve and number of matched riders curve
plot_mr(MR_list, config)
plot_r(R_list, config)
# Store results
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

for d in Route_D.keys():

    route_filename = result_loc + r"route_{}_disagg.csv".format(d)
    with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Route_D[d]:
            csv_out.writerow(row)

RDL_filename = result_loc + r"Y_rdl_disagg.csv"
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

RD_filename = result_loc + r"U_rd_disagg.csv"
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
    trip_mat,
    open(config["m2m_data_loc"] + "ttrips_mat.p", "wb"),
)

# Record config info
with open(result_loc + "config.yaml", "w") as fwrite:
    yaml.dump(config, fwrite)

# plot travel demand
travel_demand_plot(trip_dict, config)
# plot routes
route_plot(Route_D, config)

print("Finished!")
