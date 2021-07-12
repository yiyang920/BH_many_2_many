# pylint: disable=E0611, E1101
#%% import packages
import pandas as pd
import numpy as np
import pickle
import operator
import networkx as nx
import time
import itertools
import random
import copy
import sys
import os
import csv
from scipy.spatial import distance
import scipy.sparse as sp
from scipy.spatial import distance
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from st_network2 import Many2Many
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import great_circle_distance_matrix
from trip_prediction import trip_prediction

# import osmnx as ox
# from joblib import Parallel, delayed, load, dump
# from mpl_toolkits.basemap import Basemap

#%% Initialize Parameters and Load Data===========================================
config = dict()
config["S"] = 69
config["T"] = 185
config["beta"] = 0  # x DELTA_t (mins) time flexibility budget
config["VEH_CAP"] = 20
config["FIXED_ROUTE"] = True
config["REPEATED_TOUR"] = True
config["TIME_LIMIT"] = 1000
config["MIP_GAP"] = 0.2
DELTA_t = 1  # discrete time interval in minutes

# Load neighbor nodes information
ctr_info = pickle.load(open(r"Data\temp\Station.p", "rb"))
ctr = dict()
zone_id = list(ctr_info.keys())
for i in range(len(ctr_info)):
    ctr_info[i]["neighbours"].append(zone_id[i])
    ctr[i] = list(e for e in ctr_info[i]["neighbours"] if e < config["S"])


S = config["S"]
# graph with edge cost as shortest travel time
G_t = pickle.load(open(r"Data\temp\G_t.p", "rb"))
# graph with edge cost as shortest travel distance
G_d = pickle.load(open(r"Data\temp\G_d.p", "rb"))

# round travel time to integer
for _, _, d in G_t.edges(data=True):
    d["weight"] = np.rint(d["weight"])

tau = np.zeros((S, S))
tau2 = np.zeros((S, S))
# TO DO: convert travel time unit to the unit of many-2-many problem
for i in range(S):
    for j in range(S):
        if i == j:
            tau[i, j] = 0
            tau2[i, j] = 1
        else:
            tau[i, j] = nx.shortest_path_length(
                G_t, source=i, target=j, weight="weight"
            )

            tau2[i, j] = tau[i, j]


# Mode choice model trip prediction
# load mode choice input data
mc_fileloc = r"mc_input_data\\"
per_dist = pd.read_csv(mc_fileloc + "dists_dat.csv", sep=",")
per_emp = pd.read_csv(mc_fileloc + "dests_emp.csv", sep=",")
mdot_dat = pd.read_csv(mc_fileloc + "mdot_trips_dc.csv", sep=",")
dests_geo = pd.read_csv(mc_fileloc + "dests_geoid.csv", sep=",")
D = pd.read_csv(mc_fileloc + "distance.csv", sep=",", index_col=[0])
# load ID converter
id_converter = pickle.load(open(r"Data\id_converter.p", "rb"))

#%% Mode Choice Model for Trip Prediction===========================================
(
    trips_dict_pk,
    trips_dict_op,
    transit_trips_dict_pk,
    transit_trips_dict_op,
) = trip_prediction(id_converter, per_dist, per_emp, mdot_dat, dests_geo, D)

#%% Generate Rider and Driver Information===========================================
# Rider info: Rounding trip number with threshold 0.5
# multiply by 0.6 representing morning peak hour 7-10am

m2m_data_loc = r"many2many_data\\FR\\"

trip_dict = {
    k: int(np.round(0.6 * v))
    for k, v in transit_trips_dict_pk.items()
    if int(np.round(0.6 * v)) > 0
    if k[0] != k[1]
}

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
Rider.to_csv(m2m_data_loc + r"Rider.csv", index=False)
Rider = Rider.to_numpy(dtype=int, na_value=999)

# Generate driver information
# Load fixed routed infomation
if config["FIXED_ROUTE"]:
    df_blue_1 = pd.read_csv(m2m_data_loc + r"blue_fr_1.csv")
    df_blue_2 = pd.read_csv(m2m_data_loc + r"blue_fr_1.csv")
    df_red_1 = pd.read_csv(m2m_data_loc + r"red_fr_1.csv")
    df_yellow_1 = pd.read_csv(m2m_data_loc + r"yellow_fr_1.csv")

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

if config["REPEATED_TOUR"]:
    Driver = pd.read_csv(m2m_data_loc + r"Driver_rt.csv")
else:
    Driver = pd.read_csv(m2m_data_loc + r"Driver.csv")

Driver = Driver.to_numpy(dtype=int, na_value=999)

#%% Run many-to-many model
(X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
    Rider, Driver, tau, tau2, ctr, config, fixed_route_D=FR
)
#%% Store results
result_loc = r"test_data\result\3_25_2021\\"
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

    route_filename = result_loc + r"route_{}.csv".format(d)
    with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Route_D[d]:
            csv_out.writerow(row)

RDL_filename = result_loc + r"Y_rdl.csv"
with open(RDL_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d", "t1", "t2", "s1", "s2", "n1", "n2"])
    for row in Y:
        csv_out.writerow(row)
    csv_out.writerow(
        ["Matching Rate: {}".format(mr),]
    )

RD_filename = result_loc + r"U_rd.csv"
with open(RD_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d"])
    for row in U:
        csv_out.writerow(row)
    csv_out.writerow(
        ["Matching Rate: {}".format(mr),]
    )

R_match = pd.DataFrame(data=sorted(list(R_match)), columns=["r"])
R_match_filename = result_loc + r"R_match.csv"
R_match.to_csv(R_match_filename, index=False)
