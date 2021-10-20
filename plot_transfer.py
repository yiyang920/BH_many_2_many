import pickle, yaml, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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

# plot the distribution of transfers
fig1, ax1 = plt.subplots()
ax1.bar(n_transfer.keys(), n_transfer.values())
ax1.set_xlabel("Number of transfers")
ax1.set_ylabel("Percentage of passengers")
# ax1.set_title("Distribution of transfers")
ax1.set_xticks(np.arange(max(n_transfer.keys()) + 1))

ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylim(bottom=0, top=1.0)

fig1.savefig(
    figure_pth + r"Distribution of transfers for riders.png",
    bbox_inches="tight",
    dpi=100,
)
plt.close()

pickle.dump(
    n_transfer,
    open(
        config["m2m_output_loc"] + "n_transfer_{}.p".format(0),
        "wb",
    ),
)
