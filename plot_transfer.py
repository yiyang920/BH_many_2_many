import pickle, yaml, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import (
    disagg_2_agg_trip,
    get_rider,
)

with open("config_mc_m2m_3.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)

figure_pth = config["figure_pth"]

if not os.path.exists(figure_pth):
    os.makedirs(figure_pth)

Y_df = pd.read_csv(config["m2m_output_loc"] + "Y_rdl_disagg.csv")
Y_df = Y_df.iloc[:-1, :].astype("int64")
Y = list(Y_df.to_records(index=False))
R = set(Y_df.r)
r_transfer = {}

dumm_d = (
    max(config["driver_set"]) + 1 if config["driver_set"] else len(config["FR_list"])
)

for (r, d) in zip(Y_df.r, Y_df.d):
    if d != dumm_d:
        r_transfer.setdefault(r, set()).add(d)

n_transfer = {}
for r, D in r_transfer.items():
    n_transfer.setdefault(len(D) - 1, 0)
    n_transfer[len(D) - 1] += 1

n_transfer = {k: v / len(R) for (k, v) in n_transfer.items()}

# # plot the distribution of transfers
# fig1, ax1 = plt.subplots()
# ax1.bar(n_transfer.keys(), n_transfer.values())
# ax1.set_xlabel("Number of transfers")
# ax1.set_ylabel("Percentage of passengers")
# # ax1.set_title("Distribution of transfers")
# ax1.set_xticks(np.arange(max(n_transfer.keys()) + 1))

# ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
# ax1.set_ylim(bottom=0, top=1.0)

# fig1.savefig(
#     figure_pth + r"Distribution of transfers for riders.png",
#     bbox_inches="tight",
#     dpi=100,
# )
# plt.close()

ttrips_mat = pickle.load(
    open(config["m2m_data_loc"] + "ttrips_mat.p", "rb"),
)

# disaggregate network, dis_2_agg is 1-to-1 mapping
disagg_2_agg_id = {k: k for k in range(config["S_disagg"])}
trip_dict = disagg_2_agg_trip(
    ttrips_mat, config, disagg_2_agg_id=disagg_2_agg_id, fraction=3 / 24
)
Rider = get_rider(trip_dict, config)
on_time, travel_time, wait_stop, wait_time = dict(), dict(), dict(), dict()
for i, r, d, t1, t2, s1, s2 in zip(
    Y_df.index, Y_df.r, Y_df.d, Y_df.t1, Y_df.t2, Y_df.s1, Y_df.s2
):
    if s1 == Rider[r][4] and r not in on_time:
        on_time[r] = t1
    if s2 == Rider[r][5]:
        travel_time[r] = t2 - on_time[r]
    if i == 0:
        if d == dumm_d:
            wait_stop[r] = (t1, Y_df.loc[i - 1, "d"])
    else:
        if d == dumm_d and (
            Y_df.loc[i - 1, "r"] != r or Y_df.loc[i - 1, "d"] != dumm_d
        ):
            wait_stop[r] = (t1, Y_df.loc[i - 1, "d"])
        elif (
            (Y_df.loc[i - 1, "r"], Y_df.loc[i - 1, "d"]) == (r, dumm_d)
            and d != dumm_d
            and d != wait_stop[r][1]
        ):
            wait_time.setdefault(r, 0)
            wait_time[r] += t1 - wait_stop[r][0]

print(
    "Average transfer waiting time: {}".format(
        np.mean(list(wait_time.values())) if wait_time else 0
    )
)
print("Average travel time: {}".format(np.mean(list(travel_time.values()))))
