import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import fiona
import os

os.environ["PROJ_LIB"] = r"C:\\Users\\SQwan\\miniconda3\\Library\\share"
import pickle
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

if not os.path.exists(r"Data"):
    os.makedirs(r"Data")

if not os.path.exists(r"Data\temp"):
    os.makedirs(r"Data\temp")


def locate_point(point, geoseries):
    """
    Checker function which determines if a coordinate is within
    the geoseries and return the zone id where the point is
    located
    """
    idx_list = geoseries.index[geoseries.geometry.contains(point)].tolist()
    return idx_list[0] if len(idx_list) >= 1 else np.nan


def station_arrange(L, tau):
    count = 1
    foo = list()
    for i, (cur, nex) in enumerate(zip(L[:-1], L[1:])):
        if i == len(L) - 2:
            if cur != nex:
                foo.append(
                    (cur, count - tau[cur, nex] + 1 if count + 1 > tau[cur, nex] else 1)
                )
                foo.append((nex, 1) if cur != nex else (cur, 1))
                break
            else:
                foo.append((cur, 1))
                break

        if cur != nex:
            foo.append(
                (cur, count - tau[cur, nex] + 1 if count + 1 > tau[cur, nex] else 1)
            )
            count = 0

        count += 1

    ans = list()
    for s, count in foo:
        ans += [s] * int(count)
    return ans


print("Encoding routes...")
# Route stations
blue_station = [
    Point(42.116034, -86.452483),
    Point(42.115074, -86.465348),
    Point(42.110256, -86.463736),
    Point(42.110254, -86.457768),
    Point(42.101634, -86.448961),
    Point(42.101646, -86.441657),
    Point(42.102544, -86.436052),
    Point(42.088709, -86.437108),
    Point(42.084822, -86.437156),
    Point(42.080667, -86.434759),
    Point(42.085583, -86.433805),
    Point(42.085637, -86.424232),
    Point(42.082548, -86.421979),
    Point(42.082242, -86.418849),
    Point(42.077808, -86.424668),
    Point(42.102544, -86.436052),
    Point(42.107206, -86.446024),
    Point(42.109733, -86.447242),
    Point(42.116034, -86.452483),
]
blue_station = [Point(p.y, p.x) for p in blue_station]

red_station = [
    Point(42.101646, -86.441657),
    Point(42.101634, -86.448961),
    Point(42.116034, -86.452483),
    Point(42.115074, -86.465348),
    Point(42.111264, -86.481872),
    Point(42.088810, -86.478394),
    Point(42.084126, -86.486379),
    Point(42.079074, -86.493490),
    Point(42.033439, -86.513542),
    Point(42.026502, -86.516012),
    Point(42.086425, -86.440537),
    Point(42.101646, -86.441657),
]
red_station = [Point(p.y, p.x) for p in red_station]

yellow_station = [
    Point(42.118494913335645, -86.45082973186932),
    Point(42.13082775201815, -86.4538851865351),
    Point(42.13268958444188, -86.45128880811971),
    Point(42.124573800847095, -86.4460383743168),
    Point(42.121903066372475, -86.4390957589761),
    Point(42.116026992072754, -86.4296080933503),
    Point(42.11587877166418, -86.43641202669362),
    Point(42.112791181420455, -86.4407060644722),
    Point(42.10241413329736, -86.43602474092258),
    Point(42.10241413, -86.43602474),
    Point(42.11279118, -86.44070606),
    Point(42.11587877, -86.43641203),
    Point(42.11602699, -86.42960809),
    Point(42.12190307, -86.43909576),
    Point(42.1245738, -86.44603837),
    Point(42.13268958, -86.45128881),
    Point(42.13082775, -86.45388519),
    Point(42.11849491, -86.45082973),
]
yellow_station = [Point(p.y, p.x) for p in yellow_station]


brown_df = pd.read_csv(r"Data\new routes\brown_inbound1.csv", index_col=["id"])
brown_station = [Point(x, y) for x, y in zip(brown_df.xcoord, brown_df.ycoord)]

yellow_revised_df = pd.read_csv(
    r"Data\new routes\yellow_revised_inbound1.csv", index_col=["id"]
)
yellow_revised_station = [
    Point(x, y) for x, y in zip(yellow_revised_df.xcoord, yellow_revised_df.ycoord)
]

red_revised_df = pd.read_csv(
    r"Data\new routes\red_revised_inbound1.csv", index_col=["id"]
)
red_revised_station = [
    Point(x, y) for x, y in zip(red_revised_df.xcoord, red_revised_df.ycoord)
]

blue_revised_df = pd.read_csv(r"Data\new routes\blue inbound1.csv", index_col=["id"])
blue_revised_station = [
    Point(x, y) for x, y in zip(blue_revised_df.xcoord, blue_revised_df.ycoord)
]

grey_df = pd.read_csv(r"Data\new routes\grey inbound1.csv", index_col=["id"])
grey_station = [Point(x, y) for x, y in zip(grey_df.xcoord, grey_df.ycoord)]

purple_df = pd.read_csv(r"Data\new routes\purple inbound1.csv", index_col=["id"])
purple_station = [Point(x, y) for x, y in zip(purple_df.xcoord, purple_df.ycoord)]

#
gdf = gpd.read_file(r"Data\shapefile\zone_id.shp")

blue_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in blue_station]
blue_station_id = [
    v
    for i, v in enumerate(blue_station_id)
    if i < len(blue_station_id) - 1
    # if v != blue_station_id[i + 1]
]
blue_station_id += [gdf.loc[locate_point(blue_station[-1], gdf), "zone_id"]]

red_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in red_station]
red_station_id = [
    v
    for i, v in enumerate(red_station_id)
    if i < len(red_station_id) - 1
    # if v != red_station_id[i + 1]
]
red_station_id += [gdf.loc[locate_point(red_station[-1], gdf), "zone_id"]]

yellow_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in yellow_station]
yellow_station_id = [
    v
    for i, v in enumerate(yellow_station_id)
    if i < len(yellow_station_id) - 1
    # if v != yellow_station_id[i + 1]
]
yellow_station_id += [gdf.loc[locate_point(yellow_station[-1], gdf), "zone_id"]]

brown_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in brown_station]
brown_station_id = [
    v
    for i, v in enumerate(brown_station_id)
    if i < len(brown_station_id) - 1
    # if v != brown_station_id[i + 1]
]
brown_station_id += [gdf.loc[locate_point(brown_station[-1], gdf), "zone_id"]]

yellow_revised_station_id = [
    gdf.loc[locate_point(p, gdf), "zone_id"] for p in yellow_revised_station
]
yellow_revised_station_id = [
    v
    for i, v in enumerate(yellow_revised_station_id)
    if i < len(yellow_revised_station_id) - 1
    # if v != yellow_revised_station_id[i + 1]
]
yellow_revised_station_id += [
    gdf.loc[locate_point(yellow_revised_station[-1], gdf), "zone_id"]
]

red_revised_station_id = [
    gdf.loc[locate_point(p, gdf), "zone_id"] for p in red_revised_station
]
red_revised_station_id = [
    v
    for i, v in enumerate(red_revised_station_id)
    if i < len(red_revised_station_id) - 1
    # if v != red_revised_station_id[i + 1]
]
red_revised_station_id += [
    gdf.loc[locate_point(red_revised_station[-1], gdf), "zone_id"]
]

blue_revised_station_id = [
    gdf.loc[locate_point(p, gdf), "zone_id"] for p in blue_revised_station
]
blue_revised_station_id = [
    v
    for i, v in enumerate(blue_revised_station_id)
    if i < len(blue_revised_station_id) - 1
    # if v != red_revised_station_id[i + 1]
]
blue_revised_station_id += [
    gdf.loc[locate_point(blue_revised_station[-1], gdf), "zone_id"]
]

grey_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in grey_station]
grey_station_id = [
    v
    for i, v in enumerate(grey_station_id)
    if i < len(grey_station_id) - 1
    # if v != brown_station_id[i + 1]
]
grey_station_id += [gdf.loc[locate_point(grey_station[-1], gdf), "zone_id"]]

purple_station_id = [gdf.loc[locate_point(p, gdf), "zone_id"] for p in purple_station]
purple_station_id = [
    v
    for i, v in enumerate(purple_station_id)
    if i < len(purple_station_id) - 1
    # if v != brown_station_id[i + 1]
]
purple_station_id += [gdf.loc[locate_point(purple_station[-1], gdf), "zone_id"]]


# graph with edge cost as shortest travel time
G_t = pickle.load(open(r"Data\temp\G_t.p", "rb"))

S = 69  # number of zones
DELTA_t = 1  # x min
tau = np.zeros((S, S))
tau2 = np.zeros((S, S))
# round travel time to integer
for _, _, d in G_t.edges(data=True):
    d["weight"] = np.rint(d["weight"])

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

blue_station_id = station_arrange(blue_station_id, tau2)
red_station_id = station_arrange(red_station_id, tau2)
yellow_station_id = station_arrange(yellow_station_id, tau2)
red_revised_station_id = station_arrange(red_revised_station_id, tau2)
yellow_revised_station_id = station_arrange(yellow_revised_station_id, tau2)
brown_station_id = station_arrange(brown_station_id, tau2)
grey_station_id = station_arrange(grey_station_id, tau2)
purple_station_id = station_arrange(purple_station_id, tau2)
blue_revised_station_id = station_arrange(blue_revised_station_id, tau2)

# Load neighbor nodes information
ctr_info = pickle.load(open(r"Data\temp\Station.p", "rb"))
# graph with edge cost as shortest travel time
G_t = pickle.load(open(r"Data\temp\G_t.p", "rb"))

for node in ctr_info.keys():
    ctr_info[node]["neighbours"].append(node)

s_blue = list()
s_blue.append(blue_station_id[0])
for previous, current in zip(blue_station_id[:-1], blue_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_blue.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_blue += sp[1:]

s_red = list()
s_red.append(red_station_id[0])
for previous, current in zip(red_station_id[:-1], red_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_red.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_red += sp[1:]

s_yellow = list()
s_yellow.append(yellow_station_id[0])
for previous, current in zip(yellow_station_id[:-1], yellow_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_yellow.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_yellow += sp[1:]

s_red_revised = list()
s_red_revised.append(red_revised_station_id[0])
for previous, current in zip(red_revised_station_id[:-1], red_revised_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_red_revised.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_red_revised += sp[1:]

s_yellow_revised = list()
s_yellow_revised.append(yellow_revised_station_id[0])
for previous, current in zip(
    yellow_revised_station_id[:-1], yellow_revised_station_id[1:]
):
    if current in ctr_info[previous]["neighbours"]:
        s_yellow_revised.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_yellow_revised += sp[1:]

s_blue_revised = list()
s_blue_revised.append(blue_revised_station_id[0])
for previous, current in zip(blue_revised_station_id[:-1], blue_revised_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_blue_revised.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_blue_revised += sp[1:]

s_brown = list()
s_brown.append(brown_station_id[0])
for previous, current in zip(brown_station_id[:-1], brown_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_brown.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_brown += sp[1:]

s_grey = list()
s_grey.append(grey_station_id[0])
for previous, current in zip(grey_station_id[:-1], grey_station_id[1:]):
    if current in ctr_info[previous]["neighbours"]:
        s_grey.append(current)
    else:
        sp = nx.shortest_path(G_t, source=previous, target=current, weight="weight")
        s_grey += sp[1:]

s_purple = [39]


# build route schedule
sch_blue_1 = np.cumsum(
    [0] + [tau2[s_blue[i], s_blue[i + 1]] for i in range(len(s_blue) - 1)]
)

# Rotate the blue bus station list with 30 mins
idx = np.where(sch_blue_1 >= 30)[0][0]
s_blue_2 = s_blue[-idx:] + s_blue[:-idx]

sch_blue_2 = np.cumsum(
    [0] + [tau2[s_blue_2[i], s_blue_2[i + 1]] for i in range(len(s_blue_2) - 1)]
)


sch_red_1 = np.cumsum(
    [0] + [tau2[s_red[i], s_red[i + 1]] for i in range(len(s_red) - 1)]
)

sch_yellow_1 = np.cumsum(
    [0] + [tau2[s_yellow[i], s_yellow[i + 1]] for i in range(len(s_yellow) - 1)]
)

sch_red_revised_1 = np.cumsum(
    [0]
    + [
        tau2[s_red_revised[i], s_red_revised[i + 1]]
        for i in range(len(s_red_revised) - 1)
    ]
)

sch_yellow_revised_1 = np.cumsum(
    [0]
    + [
        tau2[s_yellow_revised[i], s_yellow_revised[i + 1]]
        for i in range(len(s_yellow_revised) - 1)
    ]
)

sch_blue_revised_1 = np.cumsum(
    [0]
    + [
        tau2[s_blue_revised[i], s_blue_revised[i + 1]]
        for i in range(len(s_blue_revised) - 1)
    ]
)

sch_brown_1 = np.cumsum(
    [0] + [tau2[s_brown[i], s_brown[i + 1]] for i in range(len(s_brown) - 1)]
)

sch_grey_1 = np.cumsum(
    [0] + [tau2[s_grey[i], s_grey[i + 1]] for i in range(len(s_grey) - 1)]
)

sch_purple_1 = np.array([0])


# combine schedule and stop id
blue_fr_1 = {"t": sch_blue_1, "s": s_blue * 1}
blue_fr_2 = {"t": sch_blue_2, "s": s_blue_2 * 1}

df_blue_fr_1 = pd.DataFrame.from_dict(blue_fr_1)
df_blue_fr_2 = pd.DataFrame.from_dict(blue_fr_2)

red_fr_1 = {"t": sch_red_1, "s": s_red * 1}
df_red_fr_1 = pd.DataFrame.from_dict(red_fr_1)

yellow_fr_1 = {"t": sch_yellow_1, "s": s_yellow * 1}
df_yellow_fr_1 = pd.DataFrame.from_dict(yellow_fr_1)

red_revised_fr_1 = {"t": sch_red_revised_1, "s": s_red_revised * 1}
df_red_revised_fr_1 = pd.DataFrame.from_dict(red_revised_fr_1)

yellow_revised_fr_1 = {"t": sch_yellow_revised_1, "s": s_yellow_revised * 1}
df_yellow_revised_fr_1 = pd.DataFrame.from_dict(yellow_revised_fr_1)

blue_revised_fr_1 = {"t": sch_blue_revised_1, "s": s_blue_revised * 1}
df_blue_revised_fr_1 = pd.DataFrame.from_dict(blue_revised_fr_1)

brown_fr_1 = {"t": sch_brown_1, "s": s_brown * 1}
df_brown_fr_1 = pd.DataFrame.from_dict(brown_fr_1)

grey_fr_1 = {"t": sch_grey_1, "s": s_grey * 1}
df_grey_fr_1 = pd.DataFrame.from_dict(grey_fr_1)

purple_fr_1 = {"t": sch_purple_1, "s": s_purple * 1}
df_purple_fr_1 = pd.DataFrame.from_dict(purple_fr_1)

df_blue_fr_1.to_csv(r"many2many_data\blue_fr_1.csv", index=False)
df_blue_fr_2.to_csv(r"many2many_data\blue_fr_2.csv", index=False)
df_red_fr_1.to_csv(r"many2many_data\red_fr_1.csv", index=False)
df_yellow_fr_1.to_csv(r"many2many_data\yellow_fr_1.csv", index=False)
df_red_revised_fr_1.to_csv(r"many2many_data\red_revised_fr_1.csv", index=False)
df_blue_revised_fr_1.to_csv(r"many2many_data\blue_revised_fr_1.csv", index=False)
df_yellow_revised_fr_1.to_csv(r"many2many_data\yellow_revised_fr_1.csv", index=False)
df_brown_fr_1.to_csv(r"many2many_data\brown_fr_1.csv", index=False)
df_grey_fr_1.to_csv(r"many2many_data\grey_fr_1.csv", index=False)
df_purple_fr_1.to_csv(r"many2many_data\purple_fr_1.csv", index=False)

print("Routes all encoded!")