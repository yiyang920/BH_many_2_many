import csv, pickle, operator, os, yaml, math, time
import pandas as pd
import numpy as np
import networkx as nx
from disagg import disagg_sol
from st_network3_1 import Many2Many
from collections import defaultdict
from trip_prediction import trip_prediction
from graph_coarsening_local_search import local_search
from direct_disagg import direct_disagg
from utils import (
    # disagg_2_agg_trip,
    # disagg_trip_get_rider,
    # get_driver_disagg,
    get_driver_m2m_gc,
    # get_rider,
    get_rider_agg_diagg,
    # load_FR_m2m_gc,
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
    # update_driver,
    # update_time_dist,
    # post_processing,
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

ITER_LIMIT_M2M_GC = config["ITER_LIMIT_M2M_GC"]

X = pickle.load(open(r"Data\temp\X.p", "rb"))
U = pickle.load(open(r"Data\temp\U.p", "rb"))
Y = pickle.load(open(r"Data\temp\Y.p", "rb"))
Route_D = pickle.load(open(r"Data\temp\Route_D.p", "rb"))
mr = pickle.load(open(r"Data\temp\mr.p", "rb"))
OBJ = pickle.load(open(r"Data\temp\OBJ.p", "rb"))
R_match = pickle.load(open(r"Data\temp\R_match.p", "rb"))

Driver = pickle.load(open(r"Data\temp\Driver.p", "rb"))
tau = pickle.load(open(r"Data\temp\tau_agg.p", "rb"))
agg_2_disagg_id = pickle.load(
    open(
        r"Data\temp\agg_2_disagg_id_{}.p".format(
            config["ITER_LIMIT_M2M_GC"] + 1 - ITER_LIMIT_M2M_GC
        ),
        "rb",
    ),
)
disagg_2_agg_id = pickle.load(
    open(
        r"Data\temp\disagg_2_agg_id_{}.p".format(
            config["ITER_LIMIT_M2M_GC"] + 1 - ITER_LIMIT_M2M_GC
        ),
        "rb",
    ),
)
