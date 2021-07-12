# Disable all the "No name %r in module %r violations" in this function
# Disable all the "%s %r has no %r member violations" in this function
# Disable all the "Passing unexpected keyword argument %r in function call" in this function
# pylint: disable=E0611, E1101, E1123, E1120

# %% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import fiona
import os
import pickle
import googlemaps
import networkx as nx
import itertools
import csv
import glob
from collections import Counter
from datetime import datetime
from pprint import pprint
from mpl_toolkits.basemap import Basemap
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import great_circle_distance_matrix
from shapely.geometry import Point, shape, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

import shapely.speedups

shapely.speedups.enable()

if not os.path.exists(r"Data"):
    os.makedirs(r"Data")

if not os.path.exists(r"Data\temp"):
    os.makedirs(r"Data\temp")

route_file_path = r"test_data\result\FR_and_2_BO\\"
result_file_path = r"many2many_output\FR_and_2_BO\\"
driver_set = {
    4,
    5,
    # 6,
}  # optimized route's driver id

if not os.path.exists(result_file_path):
    os.makedirs(result_file_path)

# %% Zone Disaggregation
S = 69  # number of zones
DELTA_t = 1  # x min

agg_2_disagg_id = pickle.load(open(r"Data\agg_2_disagg_id.p", "rb"))
disagg_2_agg_id = pickle.load(open(r"Data\disagg_2_agg_id.p", "rb"))

ctr_info = pickle.load(open(r"Data\temp\Station.p", "rb"))
# graph with edge cost as shortest travel time
G_t = pickle.load(open(r"Data\temp\G_t.p", "rb"))

# round travel time to integer
for _, _, d in G_t.edges(data=True):
    d["weight"] = np.rint(d["weight"])


route_dict = {}

for d in driver_set:
    route_dict[d] = pd.read_csv(
        route_file_path + r"route_{}_agg.csv".format(d), encoding="utf-8"
    )
# for i, f in enumerate(glob.glob(route_file_path + "route_*_agg.csv")):
#     route_dict[i] = pd.read_csv(f, encoding="utf-8")

N_route = len(route_dict)  # number of route

pth_dict_agg = {}
for k, route in route_dict.items():
    pth_dict_agg[k] = list(set((route["s1"])))

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
# %% Solve the Open TSP Problem
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

for d in driver_set:
    distance_matrix = np.zeros((len(c_dict[d]), len(c_dict[d])))
    for i in range(len(c_dict[d])):
        for j in range(len(c_dict[d])):
            distance_matrix[i, j] = tau[pth_dict[d][i], pth_dict[d][j]]

    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    # %% Check if all pair of successive zones are neighbors
    route_id = [pth_dict[d][i] for i in permutation]

    s = list()
    s.append(route_id[0])
    for previous, current in zip(route_id, route_id[1:]):
        if current in ctr_info[previous]["neighbours"]:
            s.append(current)
        else:
            sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
            s += sp[1:]

    # make sure the bus back to the depot
    s.append(s[0])

    # get bus schedule time based on tau matrix
    sch = np.cumsum(
        [0]
        + [tau[s[i], s[i + 1]] if s[i] != s[i + 1] else 1 for i in range(len(s) - 1)]
    )
    fr = {"t": sch, "s": s * 1}

    # save files
    df_fr = pd.DataFrame.from_dict(fr)
    df_fr.to_csv(result_file_path + r"fr_{}.csv".format(d), index=False)

    df_sch = pd.DataFrame()
    df_sch["t1"], df_sch["t2"] = fr["t"][:-1], fr["t"][1:]
    df_sch["s1"], df_sch["s2"] = fr["s"][:-1], fr["s"][1:]
    df_sch.to_csv(result_file_path + r"sch_{}.csv".format(d), index=False)


# %%
