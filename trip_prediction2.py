from func_deco import my_timer
import pandas as pd
import numpy as np
import os
import pickle
import csv

"""
Modes:
    0: walk
    1: bus
    2: dial-a-ride
    3: shared drive
    4: drive
"""

# Functions
# FN:calculate mode choice logsums
# There are two mode choice models;
# one for shopping trip purposes and one for all other trip purposes
def mc_utilities(mdot_per_dat: pd.DataFrame, ascs: tuple, skims: tuple) -> float:

    worker = mdot_per_dat["emp"] == 1
    health = mdot_per_dat["purpose"] == 12
    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2
    noLow = mdot_per_dat["income"] > 2
    medInc = 2 < mdot_per_dat["income"] < 5

    # Parameters
    scale = (
        0.453627
        if mdot_per_dat["dat_id"] == 1
        else 0.443440
        if mdot_per_dat["dat_id"] == 2
        else 1.0000
    )

    asc_walk = ascs[0]  # -1.100 #-2.8100
    asc_dar = ascs[2]  # -3.9990 #-5.0600
    asc_bus = ascs[1]  # -3.9600 #-4.3600
    asc_share = ascs[3]  # -1.800 #-2.500
    asc_drive = ascs[4]  # 0.0000

    B_time = -0.035010
    B_cost = -0.136460

    B_workerDrive = 0.706155
    B_elderShare = 0.592326

    B_healthDrive = 1.161821
    B_healthTransit = 1.291629

    B_womenTransit = 1.097161
    B_womenShare = 0.730061

    wk_time = skims[4]
    bus_time = skims[3] * medInc
    dar_time = (skims[0] / 30) * 60 * medInc
    auto_time = skims[2] * medInc

    bus_cost = (1 + (skims[1] * 0.05)) * noLow
    dar_cost = (2 + (skims[0] * 0.1)) * noLow
    share_cost = ((skims[0] * 0.6) / 2.5) * medInc
    drive_cost = (skims[0] * 0.6) * medInc

    av_wk = mdot_per_dat["av_wk"]
    av_dar = mdot_per_dat["av_dar"]
    av_bus = mdot_per_dat["av_bus"]
    av_share = int(mdot_per_dat["hh_auto"] > 0)
    av_auto = mdot_per_dat["av_auto"]

    mu_transit = 1.59
    mu_auto = 1.15

    # mode utilities
    u_walk = asc_walk + B_time * wk_time
    u_bus = (
        asc_bus
        + B_time * bus_time
        + B_cost * bus_cost
        + B_womenTransit * woman
        + B_healthTransit * health
    )
    u_dar = (
        asc_dar
        + B_time * dar_time
        + B_cost * dar_cost
        + B_womenTransit * woman
        + B_healthTransit * health
    )
    u_share = (
        asc_share
        + B_time * auto_time
        + B_cost * share_cost
        + B_womenTransit * woman
        + B_elderShare * elder
    )
    u_drive = (
        asc_drive
        + B_time * auto_time
        + B_cost * drive_cost
        + B_healthDrive * health
        + B_workerDrive * worker
    )

    exp_ls_transit = np.exp(
        (
            np.log(
                av_bus * np.exp(scale * mu_transit * u_bus)
                + av_dar * np.exp(scale * mu_transit * u_dar)
                + 1e-50
            )
        )
        / mu_transit
    )
    exp_ls_auto = np.exp(
        (
            np.log(
                av_auto * np.exp(scale * mu_auto * u_drive)
                + av_share * np.exp(scale * mu_auto * u_share)
                + 1e-50
            )
        )
        / mu_auto
    )
    return np.log(
        sum([av_wk * np.exp(scale * u_walk), exp_ls_auto, exp_ls_transit]) + 1e-50
    )


def mc_utilities_shop(mdot_per_dat: pd.DataFrame, ascs: tuple, skims: tuple) -> float:

    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2
    noLow = mdot_per_dat["income"] > 2
    medInc = 2 < mdot_per_dat["income"] < 5

    # Parameters
    scale = (
        0.274
        if mdot_per_dat["dat_id"] == 1
        else 0.38
        if mdot_per_dat["dat_id"] == 2
        else 1.0000
    )

    asc_walk = ascs[0]  # -1.100 #-2.8100
    asc_dar = ascs[2]  # -3.9990 #-5.0600
    asc_bus = ascs[1]  # -3.9600 #-4.3600
    asc_share = ascs[3]  # -1.800 #-2.500
    asc_drive = ascs[4]  # 0.0000

    B_time = -0.035010
    B_cost = -0.036

    B_elderShare = 1.14

    B_womenTransit = 0.976
    B_womenWalk = -6.22

    wk_time = skims[4]
    bus_time = skims[3] * medInc
    dar_time = (skims[0] / 30) * 60 * medInc
    auto_time = skims[2] * medInc

    bus_cost = (1 + (skims[1] * 0.05)) * noLow
    dar_cost = (2 + (skims[0] * 0.1)) * noLow
    share_cost = ((skims[0] * 0.6) / 2.5) * medInc
    drive_cost = (skims[0] * 0.6) * medInc

    av_wk = mdot_per_dat["av_wk"]
    av_dar = mdot_per_dat["av_dar"]
    av_bus = mdot_per_dat["av_bus"]
    av_share = mdot_per_dat["hh_auto"] > 0
    av_auto = mdot_per_dat["av_auto"]

    # mode utilities
    u_walk = asc_walk + B_time * wk_time + B_womenWalk * woman
    u_bus = asc_bus + B_time * bus_time + B_cost * bus_cost + B_womenTransit * woman
    u_dar = asc_dar + B_time * dar_time + B_cost * dar_cost + B_womenTransit * woman
    u_share = (
        asc_share + B_time * auto_time + B_cost * share_cost + B_elderShare * elder
    )
    u_drive = asc_drive + B_time * auto_time + B_cost * drive_cost

    utilities = [
        av_wk * np.exp(scale * u_walk),
        av_bus * np.exp(scale * u_bus),
        av_dar * np.exp(scale * u_dar),
        av_share * np.exp(scale * u_share),
        av_auto * np.exp(scale * u_drive),
    ]

    return np.log(sum(utilities) + 1e-50)


# FN:calculate mode Probabilities
def mc_probs(
    mdot_per_dat: pd.DataFrame, ascs: tuple, skims: tuple
) -> tuple[float, float, float, float, float]:
    worker = mdot_per_dat["emp"] == 1
    health = mdot_per_dat["purpose"] == 12
    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2
    noLow = mdot_per_dat["income"] > 2
    medInc = 2 < mdot_per_dat["income"] < 5

    # Parameters
    scale = (
        0.453627
        if mdot_per_dat["dat_id"] == 1
        else 0.443440
        if mdot_per_dat["dat_id"] == 2
        else 1.0000
    )

    asc_walk = ascs[0]  # -1.100 #-2.8100
    asc_dar = ascs[2]  # -3.9990 #-5.0600
    asc_bus = ascs[1]  # -3.9600 #-4.3600
    asc_share = ascs[3]  # -1.800 #-2.500
    asc_drive = ascs[4]  # 0.0000

    B_time = -0.035010
    B_cost = -0.136460

    B_workerDrive = 0.706155
    B_elderShare = 0.592326

    B_healthDrive = 1.161821
    B_healthTransit = 1.291629

    B_womenTransit = 1.097161
    B_womenShare = 0.730061

    wk_time = skims[4]
    bus_time = skims[3] * medInc
    dar_time = (skims[0] / 30) * 60 * medInc
    auto_time = skims[2]

    bus_cost = (1 + (skims[1] * 0.05)) * noLow
    dar_cost = (2 + (skims[0] * 0.1)) * noLow
    share_cost = ((skims[0] * 0.6) / 2.5) * medInc
    drive_cost = (skims[0] * 0.6) * medInc

    av_wk = mdot_per_dat["av_wk"]
    av_dar = mdot_per_dat["av_dar"]
    av_bus = mdot_per_dat["av_bus"]
    av_share = mdot_per_dat["hh_auto"] > 0
    av_auto = mdot_per_dat["av_auto"]

    mu_transit = 1.59
    mu_auto = 1.15

    # mode utilities
    u_walk = asc_walk + B_time * wk_time
    u_bus = (
        asc_bus
        + B_time * bus_time
        + B_cost * bus_cost
        + B_womenTransit * woman
        + B_healthTransit * health
    )
    u_dar = (
        asc_dar
        + B_time * dar_time
        + B_cost * dar_cost
        + B_womenTransit * woman
        + B_healthTransit * health
    )
    u_share = (
        asc_share
        + B_time * auto_time
        + B_cost * share_cost
        + B_womenTransit * woman
        + B_elderShare * elder
    )
    u_drive = (
        asc_drive
        + B_time * auto_time
        + B_cost * drive_cost
        + B_healthDrive * health
        + B_workerDrive * worker
    )

    exp_ls_transit = np.exp(
        (
            np.log(
                av_bus * np.exp(scale * mu_transit * u_bus)
                + av_dar * np.exp(scale * mu_transit * u_dar)
                + 1e-50
            )
        )
        / mu_transit
    )

    exp_ls_auto = np.exp(
        (
            np.log(
                av_auto * np.exp(scale * mu_auto * u_drive)
                + av_share * np.exp(scale * mu_auto * u_share)
                + 1e-50
            )
        )
        / mu_auto
    )

    utilities = [
        av_wk * np.exp(scale * u_walk),
        av_bus * np.exp(scale * mu_transit * u_bus),
        av_dar * np.exp(scale * mu_transit * u_dar),
        av_share * np.exp(scale * mu_auto * u_share),
        av_auto * np.exp(scale * mu_auto * u_drive),
        exp_ls_auto,
        exp_ls_transit,
    ]

    foo = sum([utilities[6], utilities[5], utilities[0]])
    p_trans = utilities[6] / foo if foo else np.inf
    foo = sum([utilities[6], utilities[5], utilities[0]])
    p_auto = utilities[5] / foo if foo else np.inf
    foo = sum([utilities[6], utilities[5], utilities[0]])
    p_wk = utilities[0] / foo if foo else np.inf

    if p_trans > 0 and np.isfinite(p_trans):
        foo = sum([utilities[1], utilities[2]])
        p_dar = (utilities[2] / foo) * p_trans if foo else 0.0
        p_bus = (utilities[1] / foo) * p_trans if foo else 0.0
    else:
        foo = sum([utilities[1], utilities[2]])
        p_dar = utilities[2] / foo if foo else 0.0
        p_bus = utilities[1] / foo if foo else 0.0

    if p_auto > 0 and np.isfinite(p_auto):
        foo = sum([utilities[3], utilities[4]])
        p_drive = (utilities[4] / foo) * p_auto if foo else 0.0
        p_share = (utilities[3] / foo) * p_auto if foo else 0.0
    else:
        foo = sum([utilities[3], utilities[4]])
        p_drive = utilities[4] / foo if foo else 0.0
        p_share = utilities[3] / foo if foo else 0.0
    return (p_wk, p_bus, p_dar, p_share, 1 - sum([p_wk, p_bus, p_dar, p_share]))


def mc_probs_shop(
    mdot_per_dat: pd.DataFrame, ascs: tuple, skims: tuple
) -> tuple[float, float, float, float, float]:

    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2
    noLow = mdot_per_dat["income"] > 2
    medInc = 2 < mdot_per_dat["income"] < 5

    # Parameters
    scale = (
        0.274
        if mdot_per_dat["dat_id"] == 1
        else 0.38
        if mdot_per_dat["dat_id"] == 2
        else 1.0000
    )

    asc_walk = ascs[0]  # -1.100 #-2.8100
    asc_dar = ascs[2]  # -3.9990 #-5.0600
    asc_bus = ascs[1]  # -3.9600 #-4.3600
    asc_share = ascs[3]  # -1.800 #-2.500
    asc_drive = ascs[4]  # 0.0000

    B_time = -0.035010
    B_cost = -0.036

    B_elderShare = 1.14

    B_womenTransit = 0.976
    B_womenWalk = -6.22

    wk_time = skims[4]
    bus_time = skims[3] * medInc
    dar_time = (skims[0] / 30) * 60 * medInc
    auto_time = skims[2] * medInc

    bus_cost = (1 + (skims[1] * 0.05)) * noLow
    dar_cost = (2 + (skims[0] * 0.1)) * noLow
    share_cost = ((skims[0] * 0.6) / 2.5) * medInc
    drive_cost = (skims[0] * 0.6) * medInc

    av_wk = mdot_per_dat["av_wk"]
    av_dar = mdot_per_dat["av_dar"]
    av_bus = mdot_per_dat["av_bus"]
    av_share = int(mdot_per_dat["hh_auto"] > 0)
    av_auto = mdot_per_dat["av_auto"]

    # mode utilities

    u_walk = asc_walk + B_time * wk_time + B_womenWalk * woman
    u_bus = asc_bus + B_time * bus_time + B_cost * bus_cost + B_womenTransit * woman
    u_dar = asc_dar + B_time * dar_time + B_cost * dar_cost + B_womenTransit * woman
    u_share = (
        asc_share + B_time * auto_time + B_cost * share_cost + B_elderShare * elder
    )
    u_drive = asc_drive + B_time * auto_time + B_cost * drive_cost

    utilities = [
        av_wk * np.exp(scale * u_walk),
        av_bus * np.exp(scale * u_bus),
        av_dar * np.exp(scale * u_dar),
        av_share * np.exp(scale * u_share),
        av_auto * np.exp(scale * u_drive),
    ]

    p_wk = utilities[0] / sum(utilities)
    p_bus = utilities[1] / sum(utilities)
    p_dar = utilities[2] / sum(utilities)
    p_share = utilities[3] / sum(utilities)

    return (p_wk, p_bus, p_dar, p_share, 1 - sum((p_wk, p_bus, p_dar, p_share)))


# FN:calculate destination choice utility
def dc_utility(dis: float, emp: float, mcls: float) -> float:
    # parameters
    # b_dist = -0.6000
    b_mcls = 0.80
    b_size = 0.87

    return np.exp(b_size * np.log(emp + 1.0) + b_mcls * mcls)


@my_timer
def trip_prediction(
    id_converter: dict,
    mdot_dat: pd.DataFrame,
    stop_zones: set,
    per_ddist: pd.DataFrame,
    per_tdist: pd.DataFrame,
    per_drtt: pd.DataFrame,
    per_bustt: pd.DataFrame,
    per_wktt: pd.DataFrame,
    per_emp: pd.DataFrame,
    dests_geo: pd.DataFrame,
    gm_autodist: pd.DataFrame,
    gm_transdist: pd.DataFrame,
    gm_autoTT: pd.DataFrame,
    gm_transTT: pd.DataFrame,
    gm_wkTT: pd.DataFrame,
) -> dict[tuple[int, int] : float]:
    # Data Prep
    # Filter trips traveling in bus zones
    id_converter_rev = {geoid: zoneid for zoneid, geoid in id_converter.items()}
    mdot_dat_busz = pd.DataFrame(columns=mdot_dat.columns)

    for i in mdot_dat.index:
        if (
            id_converter_rev[mdot_dat.loc[i, "geoid_o"]] in stop_zones
            and id_converter_rev[mdot_dat.loc[i, "geoid_d"]] in stop_zones
        ):
            mdot_dat_busz = mdot_dat_busz.append(mdot_dat.loc[i, :], ignore_index=False)

    N = len(mdot_dat_busz)
    N_ZONE = len(id_converter)
    N_MODE = 10
    # convert mode IDs
    modeid = mdot_dat_busz["mode"].apply(
        lambda x: 0 if x == 1 else 4 if x == 4 else 3 if x == 5 else 1 if x == 8 else 2
    )
    mdot_dat_busz["mode"] = modeid

    # Calculate MCLS for each person trip
    mcls = np.zeros((N, N_MODE))
    shop_ascs = (-1.218793, -9.075928, -10.991232, -1.192324, 0.000000)
    other_ascs = (-0.6818053, -8.4196311, -8.0848393, -1.0033526, 0.0000000)

    # MCLS
    for i in range(N):
        for j in range(N_MODE):
            skims = (
                per_ddist.iloc[i, j],
                per_tdist.iloc[i, j],
                per_drtt.iloc[i, j],
                per_bustt.iloc[i, j],
                per_wktt.iloc[i, j],
            )

            if 7 < mdot_dat_busz.purpose.iloc[i] <= 10:
                mcls[i, j] = mc_utilities_shop(
                    mdot_dat_busz.iloc[i, :],
                    shop_ascs,
                    skims,
                )
            else:
                mcls[i, j] = mc_utilities(mdot_dat_busz.iloc[i, :], other_ascs, skims)

    # Predict all trip destination volumes: Output OD matrix for all trip volumes
    dc_prob = np.zeros((N, N_MODE))
    exp_u_holder = np.zeros(N_MODE)
    trips_mat = dict()

    for i in range(N):
        for j in range(N_MODE):
            exp_u_holder[j] = dc_utility(
                per_ddist.iloc[i, j], per_emp.iloc[i, j], mcls[i, j]
            )
        dc_prob[i, 0:-1] = exp_u_holder[0:-1] / np.sum(exp_u_holder)
        dc_prob[i, -1] = 1 - np.sum(dc_prob[i, 0:-1])

        # Which zone Indx is the origin?
        o = id_converter_rev[mdot_dat_busz["geoid_o"].iloc[i]]
        # Which zone Indx is the predicted destination?
        d = id_converter_rev[dests_geo.iloc[i, np.argmax(dc_prob[i, :])]]
        trips_mat.setdefault((o, d), 0)
        trips_mat[o, d] += mdot_dat_busz["weight"].iloc[i]

    bus_probs = np.zeros((N_ZONE, N_ZONE))
    dar_probs = np.zeros((N_ZONE, N_ZONE))
    mc_probs_h = np.zeros((N, N_ZONE, N_ZONE, 6))

    skims_dict = {
        (zoneid_o, zoneid_d): (
            (gm_autodist.loc[geoid_o, str(geoid_d)] + 0.1) * 0.62137,
            (gm_transdist.loc[geoid_o, str(geoid_d)] + 0.1) * 0.62137,
            (gm_autoTT.loc[geoid_o, str(geoid_d)] + 0.1065206),
            (gm_transTT.loc[geoid_o, str(geoid_d)] + 0.1331507),
            (gm_wkTT.loc[geoid_o, str(geoid_d)] + 1.331507),
        )
        for zoneid_o, geoid_o in id_converter.items()
        for zoneid_d, geoid_d in id_converter.items()
    }  # Add min dist and convert to miles

    for (i, j), skims in skims_dict.items():
        for trip in range(N):
            if 7 < mdot_dat_busz["purpose"].iloc[trip] <= 10:
                p = mc_probs_shop(mdot_dat_busz.iloc[trip, :], shop_ascs, skims)
            else:
                p = mc_probs(mdot_dat_busz.iloc[trip, :], other_ascs, skims)
            mc_probs_h[trip, i, j, :] = np.array(
                list(p) + [mdot_dat_busz["weight"].iloc[trip]]
            )

        bus_probs[i, j] = np.round_(
            np.sum(mc_probs_h[:, i, j, 1] * mc_probs_h[:, i, j, 5])
            / np.sum(mc_probs_h[:, i, j, 5]),
            decimals=7,
        )
        dar_probs[i, j] = np.round_(
            np.sum(mc_probs_h[:, i, j, 2] * mc_probs_h[:, i, j, 5])
            / np.sum(mc_probs_h[:, i, j, 5]),
            decimals=7,
        )

    # Calculate Bus + Dar Volumes, by destination
    ttrips_mat = dict()
    for od, volumn in trips_mat.items():
        ttrips_mat[od] = volumn * dar_probs[od] + volumn * bus_probs[od]

    return ttrips_mat


if __name__ == "__main__":
    # %%
    print("start testing...")
    # id_converter = pickle.load(open("Data/id_converter.p", "rb"))

    # Load data
    mc_fileloc = "mc_input_data2/"

    stop_zones_df = pd.read_csv(mc_fileloc + "stop_zones.csv")
    stop_zones = set(stop_zones_df["stop_zones"].apply(lambda x: x - 1))

    D = pd.read_csv(mc_fileloc + "distance.csv", index_col=0)
    id_converter = {i: idx for i, idx in enumerate(D.index)}

    gm_transTT = pd.read_csv(mc_fileloc + "gm_transit_time_min.csv", index_col=0)
    gm_transdist = pd.read_csv(mc_fileloc + "gm_transit_dist_km.csv", index_col=0)
    gm_autoTT = pd.read_csv(mc_fileloc + "gm_auto_time_min.csv", index_col=0)
    gm_autodist = pd.read_csv(mc_fileloc + "gm_auto_dist_km.csv", index_col=0)
    gm_wkTT = pd.read_csv(mc_fileloc + "gm_walk_time_min.csv", index_col=0)
    gm_wkdist = pd.read_csv(mc_fileloc + "gm_walk_dist_km.csv", index_col=0)
    est_dat = pd.read_csv(mc_fileloc + "mc_dat_trips_miss.csv")
    mdot_dat = est_dat.loc[est_dat["dat_id"] == 3, :]

    per_emp = pd.read_csv(mc_fileloc + "dests_temp_trips.csv")
    per_tdist = pd.read_csv(mc_fileloc + "tdists_dat_trips.csv")
    dests_geo = pd.read_csv(mc_fileloc + "tdests_geoid_trips.csv")
    per_ddist = pd.read_csv(mc_fileloc + "tdests_ddist_trips.csv")
    per_bustt = pd.read_csv(mc_fileloc + "tdests_bustt_trips.csv")
    per_drtt = pd.read_csv(mc_fileloc + "tdests_drtt_trips.csv")
    per_wktt = pd.read_csv(mc_fileloc + "tdests_wktt_trips.csv")
    # %%
    ttrips_mat = trip_prediction(
        id_converter,
        mdot_dat,
        stop_zones,
        per_ddist,
        per_tdist,
        per_drtt,
        per_bustt,
        per_wktt,
        per_emp,
        dests_geo,
        gm_autodist,
        gm_transdist,
        gm_autoTT,
        gm_transTT,
        gm_wkTT,
    )

    transit_trip = np.zeros((len(id_converter), len(id_converter)))
    for (i, j), trips in ttrips_mat.items():
        transit_trip[i, j] = trips

    result_loc = "E:/Codes/BH_data_preprocess_and_DP/mc_output2/"
    if not os.path.exists(result_loc):
        os.makedirs(result_loc)

    np.savetxt(
        result_loc + "transit_trips.csv", transit_trip, fmt="%.7f", delimiter=","
    )
    print("finish!")

# %%
