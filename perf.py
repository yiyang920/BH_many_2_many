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
import shapely.speedups
import matplotlib.pyplot as plt

result_pth = r"E:\Codes\BH_data_preprocess_and_DP\test_data\result\FR_and_2_BO\\"
figure_pth = r"many2many_output\figure\\FR_and_2_BO\\"

if not os.path.exists(figure_pth):
    os.makedirs(figure_pth)
# %% Calculate the number of transfers
S = 69  # number of zones
DELTA_t = 1  # x min

Y_rdl = pd.read_csv(result_pth + r"Y_rdl_agg.csv")
MR = Y_rdl.r.iloc[-1]
Y_rdl = Y_rdl.iloc[:-1]
Y_rdl.r = Y_rdl.r.astype("int64")

r_transfer = {}
for (r, d) in zip(Y_rdl.r, Y_rdl.d):
    if r not in r_transfer:
        if d != 6:  # exclude the dummy driver before the rider get on the first bus
            r_transfer[r] = {d}
    else:
        r_transfer[r].add(d)

n_transfer = {}
for r, D in r_transfer.items():
    if len(D) - 1 not in n_transfer:
        n_transfer[len(D) - 1] = 1
    else:
        n_transfer[len(D) - 1] += 1
# plot
fig1, ax1 = plt.subplots()
ax1.bar(n_transfer.keys(), n_transfer.values())
ax1.set_xlabel("Number of transfers")
ax1.set_ylabel("Rider count")
ax1.set_title("Distribution of transfer for riders")
ax1.set_xticks(np.arange(max(n_transfer.keys()) + 1))

ax1.set_ylim(bottom=0, top=85)

fig1.savefig(
    figure_pth + r"Distribution of transfers for riders.png",
    bbox_inches="tight",
    dpi=100,
)
# _ = plt.show()
# %% Calculate the number of actual travel time/shortest path travel time ratio
# graph with edge cost as shortest travel time
G_t_agg = pickle.load(open(r"Data\temp\G_t_agg.p", "rb"))

S_agg = 39  # number of zones
DELTA_t = 1  # x min
tau_agg = np.zeros((S, S))
tau2_agg = np.zeros((S, S))
# round travel time to integer
for _, _, d in G_t_agg.edges(data=True):
    d["weight"] = np.rint(d["weight"])

for i in range(S_agg):
    for j in range(S_agg):
        if i == j:
            tau_agg[i, j] = 0
            tau2_agg[i, j] = 1
        else:
            tau_agg[i, j] = (
                nx.shortest_path_length(G_t_agg, source=i, target=j, weight="weight")
                // DELTA_t
            )
            tau2_agg[i, j] = tau_agg[i, j]

trajectory_r = {
    rider: Y_rdl.loc[Y_rdl.r == rider, ["t1", "t2", "s1", "s2"]]
    for rider in Y_rdl.r.unique()
}

rider_info = {}
for rider, table in trajectory_r.items():
    rider_info[rider] = {}
    rider_info[rider]["O"] = int(table.s1.iloc[0])
    rider_info[rider]["D"] = int(table.s2.iloc[-1])
    rider_info[rider]["duration"] = table.t2.iloc[-1] - table.t1.iloc[0]
    rider_info[rider]["shostest_path_duration"] = tau_agg[
        rider_info[rider]["O"], rider_info[rider]["D"]
    ]
    rider_info[rider]["duration_ratio"] = (
        (rider_info[rider]["duration"] / rider_info[rider]["shostest_path_duration"])
        if rider_info[rider]["shostest_path_duration"] > 0
        else 1
        if rider_info[rider]["duration"] == 0
        else float("inf")
    )

# plot
fig2, ax2 = plt.subplots()
ax2.hist(
    [
        rider_info[rider]["duration_ratio"]
        for rider in rider_info.keys()
        if rider_info[rider]["duration_ratio"] < float("inf")
    ],
    bins=20,
)
ax2.set_xlabel("Travel time/Shortest path travel time")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of travel time/Shortest path travel time")

ax2.set_ylim(bottom=0, top=85)


fig2.savefig(
    figure_pth + r"Distribution of ratio.png",
    bbox_inches="tight",
    dpi=100,
)


# %%
