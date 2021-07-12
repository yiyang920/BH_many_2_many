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
import csv
from scipy.spatial import distance
import scipy.sparse as sp
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import great_circle_distance_matrix
from scipy.spatial import distance
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from st_network2 import Many2Many

# import osmnx as ox
# from mpl_toolkits.basemap import Basemap
# from joblib import Parallel, delayed, load, dump


np.random.seed(1)

config = dict()
config["S"] = 10
config["T"] = 60
config["beta"] = 0  # x DELTA_t (mins) time flexibility budget
config["VEH_CAP"] = 10
config["TIME_LIMIT"] = 500
config["FIXED_ROUTE"] = True
config["REPEATED_TOUR"] = True
config["MIP_GAP"] = 0.2
DELTA_t = 1  # discrete time interval in minutes

# load neighbor nodes information
ctr_info = pickle.load(open(r"test_data\Station.p", "rb"))
ctr = dict()
zone_id = list(ctr_info.keys())
for i in range(len(ctr_info)):
    ctr_info[i]["neighbours"].append(zone_id[i])
    ctr[i] = list(e for e in ctr_info[i]["neighbours"] if e < config["S"])


# graph with edge cost as shortest travel time
G_t = pickle.load(open(r"test_data\G_t.p", "rb"))
# graph with edge cost as shortest travel distance
G_d = pickle.load(open(r"test_data\G_d.p", "rb"))

# round travel time to integer
for _, _, d in G_t.edges(data=True):
    d["weight"] = np.rint(d["weight"])

S = config["S"]

tau = np.zeros((S, S))
tau2 = np.zeros((S, S))
# TO DO: convert travel time unit to the unit of many-2-many problem
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

# df = pd.read_csv(r"Data\zone_id.csv")
# sources = list(zip(df["lon"], df["lat"]))
# sources = sources[: config["S"]]
# tau = great_circle_distance_matrix(sources)
# tau = tau / 1000 / 50 * 60 // DELTA_t
# tau2 = copy.deepcopy(tau)
# for i in range(config["S"]):
#     for j in range(config["S"]):
#         if i == j:
#             tau2[i, j] = 1


Rider = pd.read_csv(r"test_data\Rider.csv")
Rider["O"] = Rider["O"].apply(lambda x: int(x % config["S"]))
Rider["D"] = Rider["D"].apply(lambda x: int(x % config["S"]))

Rider = Rider.loc[Rider["O"] != Rider["D"], :]

Rider_len = len(Rider)
Rider["ED"] = np.random.randint(0, config["T"], Rider_len)

for i in Rider.index:
    Rider.loc[i, "LA"] = int(
        Rider.loc[i, "ED"]
        + tau[int(Rider.loc[i, "O"]), int(Rider.loc[i, "D"])]
        + np.random.randint(0, config["T"] - Rider.loc[i, "ED"], 1)
    )

Rider.loc[Rider["LA"] >= config["T"], "LA"] = config["T"] - 1
# Rider.to_csv(r"test_data\Rider.csv")
Rider = Rider.to_numpy(dtype=int, na_value=999)

if config["REPEATED_TOUR"]:
    Driver = pd.read_csv(r"test_data\Driver_rt.csv")
else:
    Driver = pd.read_csv(r"test_data\Driver.csv")
    # Driver["O"] = Driver["O"].apply(lambda x: int(x % 10))
    # Driver["D"] = Driver["D"].apply(lambda x: int(x % 10))
    # Driver_len = len(Driver)
    # Driver["ED"] = np.random.randint(0, 1, Driver_len)
    # Driver["LA"] = config["T"]
    # # Driver.to_csv(r"test_data\Driver.csv")
Driver = Driver.to_numpy(dtype=int, na_value=999)


if config["FIXED_ROUTE"]:
    df = pd.read_csv(r"test_data\FR.csv")
    df2numpy = df.to_numpy(dtype=int, na_value=999)
    FR = {0: df2numpy}

else:
    FR = None

(X, U, Y, Route_D, mr, OBJ, R_match) = Many2Many(
    Rider, Driver, tau, tau2, ctr, config, fixed_route_D=FR
)


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

    route_filename = r"test_data\result\route_{}.csv".format(d)
    with open(route_filename, "w+", newline="", encoding="utf-8") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(["t1", "t2", "s1", "s2", "n1", "n2"])
        for row in Route_D[d]:
            csv_out.writerow(row)

RDL_filename = r"test_data\result\Y_rdl.csv"
with open(RDL_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d", "t1", "t2", "s1", "s2", "n1", "n2"])
    for row in Y:
        csv_out.writerow(row)
    csv_out.writerow(
        ["Matching Rate: {}".format(mr),]
    )

RD_filename = r"test_data\result\U_rd.csv"
with open(RD_filename, "w+", newline="", encoding="utf-8") as csvfile:
    csv_out = csv.writer(csvfile)
    csv_out.writerow(["r", "d"])
    for row in U:
        csv_out.writerow(row)
    csv_out.writerow(
        ["Matching Rate: {}".format(mr),]
    )

R_match = pd.DataFrame(data=sorted(list(R_match)), columns=["r"])
R_match_filename = r"test_data\result\R_match.csv"
R_match.to_csv(R_match_filename, index=False)

