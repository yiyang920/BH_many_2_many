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

from collections import Counter
from datetime import datetime
from pprint import pprint
from mpl_toolkits.basemap import Basemap

from shapely.geometry import Point, shape, Polygon
from shapely.ops import unary_union

import shapely.speedups

shapely.speedups.enable()

if not os.path.exists(r"Data"):
    os.makedirs(r"Data")

if not os.path.exists(r"Data\temp"):
    os.makedirs(r"Data\temp")

USE_GMAPS_API = False
# %% Generate mode choice data using Google Maps API
if USE_GMAPS_API:
    # your google map API key
    gmap_key = "AIzaSyCAMHXBSQrYIppBxu_GAKCMhR4rYSPuB8k"
    # travel time and distance by walk
    station = pickle.load(open(r"Data\temp\Station.p", "rb"))
    num_station = len(station)
    gmaps = googlemaps.Client(key=gmap_key)
    # datetime needs to be no later than current time, format: Y,M,D,h,m,s
    currenttime = datetime(2021, 6, 1, 12, 0, 0)

    # construct graphs of travel time and travel distance
    G_d = nx.Graph()
    G_t = nx.Graph()
    G_d.add_nodes_from(np.arange(num_station))
    G_t.add_nodes_from(np.arange(num_station))

    for i in range(num_station):
        for j in station[i]["neighbours"]:
            if i < j:
                direction_result = gmaps.directions(
                    tuple(station[i]["lat_lon"]),
                    tuple(station[j]["lat_lon"]),
                    mode="walking",
                    avoid="ferries",
                    departure_time=currenttime,
                )

                w_t = np.round(
                    direction_result[0]["legs"][0]["duration"][u"value"] / 60.0,
                    decimals=2,
                )
                w_d = np.round(
                    direction_result[0]["legs"][0]["distance"][u"value"] * 0.000621371,
                    decimals=2,
                )

                G_t.add_edge(i, j, weight=w_t)
                G_d.add_edge(i, j, weight=w_d)

    pickle.dump(G_t, open(r"Data\temp\G_t_walk.p", "wb"))
    pickle.dump(G_d, open(r"Data\temp\G_d_walk.p", "wb"))

# %% generate OD for each trip
# load ID converter
id_converter = pickle.load(open("Data\\id_converter.p", "rb"))
id_converter_reverse = pickle.load(open("Data\\id_converter_reverse.p", "rb"))
df_O = pd.read_csv(
    r"mc_input_data\dests_dat.csv",
    usecols=[
        "geoid_o",
    ],
)
df_D = pd.read_csv(r"mc_input_data\dests_geoid.csv")
# %% generate walk travel time table
# graph with edge cost as shortest travel time (min)
G_t_walk = pickle.load(open(r"Data\temp\G_t_walk.p", "rb"))

time1 = [
    nx.shortest_path_length(
        G_t_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid1)
]

time2 = [
    nx.shortest_path_length(
        G_t_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid2)
]

time3 = [
    nx.shortest_path_length(
        G_t_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid3)
]

time4 = [
    nx.shortest_path_length(
        G_t_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid4)
]

time5 = [
    nx.shortest_path_length(
        G_t_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid5)
]

df_walk_time_dat = pd.DataFrame(
    data={
        "time1": time1,
        "time2": time2,
        "time3": time3,
        "time4": time4,
        "time5": time5,
    }
)

df_walk_time_dat.to_csv(r"mc_input_data\\time_dat_walk.csv")
# %% generate auto & dial-a-ride travel time tables
# graph with edge cost as shortest travel time (min)
G_t = pickle.load(open(r"Data\temp\G_t.p", "rb"))

time1 = [
    nx.shortest_path_length(
        G_t,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid1)
]

time2 = [
    nx.shortest_path_length(
        G_t,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid2)
]

time3 = [
    nx.shortest_path_length(
        G_t,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid3)
]

time4 = [
    nx.shortest_path_length(
        G_t,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid4)
]

time5 = [
    nx.shortest_path_length(
        G_t,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid5)
]

df_auto_time_dat = pd.DataFrame(
    data={
        "time1": time1,
        "time2": time2,
        "time3": time3,
        "time4": time4,
        "time5": time5,
    }
)

foo = df_auto_time_dat.copy(deep=True)
df_dar_time_dat = foo.applymap(lambda x: x * 40 / 35 if x != 9999 else 9999)

df_dar_time_dat.to_csv(r"mc_input_data\time_dat_dar.csv")
df_auto_time_dat.to_csv(r"mc_input_data\time_dat_auto.csv")

# %% generate walk travel distance table
# graph with edge cost as shortest travel distance (km)
G_d_walk = pickle.load(open(r"Data\temp\G_d_walk.p", "rb"))

dist1 = [
    nx.shortest_path_length(
        G_d_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid1)
]

dist2 = [
    nx.shortest_path_length(
        G_d_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid2)
]

dist3 = [
    nx.shortest_path_length(
        G_d_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid3)
]

dist4 = [
    nx.shortest_path_length(
        G_d_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid4)
]

dist5 = [
    nx.shortest_path_length(
        G_d_walk,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid5)
]

df_walk_dist_dat = pd.DataFrame(
    data={
        "dist1": dist1,
        "dist2": dist2,
        "dist3": dist3,
        "dist4": dist4,
        "dist5": dist5,
    }
)

df_walk_time_dat.to_csv(r"mc_input_data\dists_dat_walk.csv")

# %% generate walk travel distance table
# graph with edge cost as shortest travel distance (km)
G_d = pickle.load(open(r"Data\temp\G_d.p", "rb"))

dist1 = [
    nx.shortest_path_length(
        G_d,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid1)
]

dist2 = [
    nx.shortest_path_length(
        G_d,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid2)
]

dist3 = [
    nx.shortest_path_length(
        G_d,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid3)
]

dist4 = [
    nx.shortest_path_length(
        G_d,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid4)
]

dist5 = [
    nx.shortest_path_length(
        G_d,
        source=id_converter_reverse[i],
        target=id_converter_reverse[j],
        weight="weight",
    )
    if i in id_converter_reverse and j in id_converter_reverse
    else 9999
    for (i, j) in zip(df_O.geoid_o, df_D.geoid5)
]

df_auto_dist_dat = pd.DataFrame(
    data={
        "dist1": dist1,
        "dist2": dist2,
        "dist3": dist3,
        "dist4": dist4,
        "dist5": dist5,
    }
)

df_dar_dist_dat = pd.DataFrame(
    data={
        "dist1": dist1,
        "dist2": dist2,
        "dist3": dist3,
        "dist4": dist4,
        "dist5": dist5,
    }
)

df_auto_dist_dat.to_csv(r"mc_input_data\dists_dat_auto.csv")
df_dar_dist_dat.to_csv(r"mc_input_data\dists_dat_dar.csv")
# %%
