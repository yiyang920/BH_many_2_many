import pandas as pd
import numpy as np
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
def mc_utilities(mdot_per_dat: object, ascs: tuple, skims: tuple) -> float:

    worker = mdot_per_dat["emp"] == 1
    health = mdot_per_dat["purpose"] == 12
    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2

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
    bus_time = skims[3]
    dar_time = (skims[0] / 30) * 60
    auto_time = skims[2]

    bus_cost = 1 + (skims[1] * 0.05)
    dar_cost = 2 + (skims[0] * 0.1)
    share_cost = (skims[0] * 0.6) / 2.5
    drive_cost = skims[0] * 0.6

    av_wk = 1
    av_dar = 1
    av_bus = 1
    av_share = 1
    av_auto = 1

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
            )
        )
        / mu_transit
    )
    exp_ls_auto = np.exp(
        (
            np.log(
                av_auto * np.exp(scale * mu_auto * u_drive)
                + av_share * np.exp(scale * mu_auto * u_share)
            )
        )
        / mu_auto
    )
    return np.log(sum([av_wk * np.exp(scale * u_walk), exp_ls_auto, exp_ls_transit]))


def mc_utilities_shop(mdot_per_dat: object, ascs: tuple, skims: tuple) -> float:

    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2

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
    bus_time = skims[3]
    dar_time = (skims[0] / 30) * 60
    auto_time = skims[2]

    bus_cost = 1 + (skims[1] * 0.05)
    dar_cost = 2 + (skims[0] * 0.1)
    share_cost = (skims[0] * 0.6) / 2.5
    drive_cost = skims[0] * 0.6

    av_wk = 1
    av_dar = 1
    av_bus = 1
    av_share = 1
    av_auto = 1

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

    return np.log(sum(utilities))


# FN:calculate mode Probabilities
def mc_probs(mdot_per_dat, ascs, skims) -> tuple:
    worker = mdot_per_dat["emp"] == 1
    health = mdot_per_dat["purpose"] == 12
    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2

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
    bus_time = skims[3]
    dar_time = (skims[0] / 30) * 60
    auto_time = skims[2]

    bus_cost = 1 + (skims[1] * 0.05)
    dar_cost = 2 + (skims[0] * 0.1)
    share_cost = (skims[0] * 0.6) / 2.5
    drive_cost = skims[0] * 0.6

    av_wk = 1
    av_dar = 1
    av_bus = 1
    av_share = 1
    av_auto = 1

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
            )
        )
        / mu_transit
    )

    exp_ls_auto = np.exp(
        (
            np.log(
                av_auto * np.exp(scale * mu_auto * u_drive)
                + av_share * np.exp(scale * mu_auto * u_share)
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
    p_trans = utilities[6] / foo if foo else float("inf")
    foo = sum([utilities[6], utilities[5], utilities[0]])
    p_auto = utilities[5] / foo if foo else float("inf")
    foo = sum([utilities[6], utilities[5], utilities[0]])
    p_wk = utilities[0] / foo if foo else float("inf")

    if p_trans > 0 and p_trans < float("inf"):
        p_dar = (utilities[2] / sum([utilities[1], utilities[2]])) * p_trans
        p_bus = (utilities[1] / sum([utilities[1], utilities[2]])) * p_trans
    else:
        p_dar = utilities[2] / sum([utilities[1], utilities[2]])
        p_bus = utilities[1] / sum([utilities[1], utilities[2]])

    if p_auto > 0 and p_auto < float("inf"):
        p_drive = (utilities[4] / sum([utilities[3], utilities[4]])) * p_auto
        p_share = (utilities[3] / sum([utilities[3], utilities[4]])) * p_auto
    else:
        p_drive = utilities[4] / sum([utilities[3], utilities[4]])
        p_share = utilities[3] / sum([utilities[3], utilities[4]])
    return (p_wk, p_bus, p_dar, p_share, 1 - sum([p_wk, p_bus, p_dar, p_share]))


def mc_probs_shop(mdot_per_dat, ascs, skims) -> tuple:

    elder = mdot_per_dat["age"] > 4
    woman = mdot_per_dat["gender"] == 2

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
    bus_time = skims[3]
    dar_time = (skims[0] / 30) * 60
    auto_time = skims[2]

    bus_cost = 1 + (skims[1] * 0.05)
    dar_cost = 2 + (skims[0] * 0.1)
    share_cost = (skims[0] * 0.6) / 2.5
    drive_cost = skims[0] * 0.6

    av_wk = 1
    av_dar = 1
    av_bus = 1
    av_share = 1
    av_auto = 1

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

    return (p_wk, p_bus, p_dar, p_share, 1 - sum(p_wk, p_bus, p_dar, p_share))


# FN:calculate destination choice utility
def dc_utility(dis: float, emp: float, mcls: float) -> float:
    # parameters
    # b_dist = -0.6000
    b_mcls = 1.00
    b_size = 0.87

    return np.exp(b_size * np.log(emp + 1) + b_mcls * mcls)


def trip_prediction(
    id_converter: dict,
    mdot_dat: object,
    stop_zones: set,
    per_ddist: object,
    per_tdist: object,
    per_drtt: object,
    per_bustt: object,
    per_wktt: object,
    per_emp: object,
    dests_geo: object,
    gm_autodist: object,
    gm_transdist: object,
    gm_autoTT: object,
    gm_transTT: object,
    gm_wkTT: object,
):
    # Data Prep
    # Filter trips traveling in bus zones
    id_converter_rev = {geoid: zoneid for zoneid, geoid in id_converter.items()}
    mdot_dat_busz = pd.DataFrame(columns=mdot_dat.columns)

    for i in mdot_dat.index:
        if (
            id_converter_rev[mdot_dat.loc[i, "geoid_o"]] in stop_zones
            and id_converter_rev[mdot_dat.loc[i, "geoid_d"]] in stop_zones
        ):
            mdot_dat_busz.append(mdot_dat.loc[i, :])

    N = len(mdot_dat_busz)
    N_MODE = 10
    # convert mode IDs
    mdot_dat_busz["modeid"] = mdot_dat_busz.mode.apply(
        lambda x: 0 if x == 1 else 4 if x == 4 else 3 if x == 5 else 1 if x == 8 else 2
    )

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

    # Calculate Destination Choice Probabilities
    exp_u_holder = np.zeros((1, N_MODE))
    dc_prob = np.zeros((N, N_MODE))

    for i in range(N):
        for j in range(N_MODE):
            exp_u_holder[j] = dc_utility(
                per_ddist.iloc[i, j], per_emp.iloc[i, j], mcls[i, j]
            )
            dc_prob[i, 0:-1] = exp_u_holder[0:-1] / np.sum(exp_u_holder)
            dc_prob[i, -1] = 1 - np.sum(dc_prob[i, 0:-1])

    # Expansion
    # Predict all trip destination volumes: Output OD matrix for all trip volumes
    trips_mat = dict()
    for trip in range(len(mdot_dat_busz)):
        geoid_o = mdot_dat_busz.geoid_o.iloc[trip]
        for mode in range(N_MODE):
            geoid_d = dests_geo.iloc[trip, mode]
            id_o, id_d = id_converter[geoid_o], id_converter[geoid_d]
            trips_mat.setdefault((id_o, id_d), 0)
            trips_mat[id_o, id_d] += (
                dc_prob[trip, mode] * mdot_dat_busz.weight.iloc[trip]
            )

    # Predict all transit volumes by Destination:
    # Output OD matrix for transit (Bus + DialARide) trip volumes
    transit_trips_mat, t_probs = dict(), dict()
    for trip in range(len(mdot_dat_busz)):
        geoid_o = mdot_dat_busz.geoid_o.iloc[trip]
        for mode in range(N_MODE):
            geoid_d = dests_geo.iloc[trip, mode]
            id_o, id_d = id_converter[geoid_o], id_converter[geoid_d]
            skims = (
                (gm_autodist.loc[geoid_o, str(geoid_d)] + 0.1) * 0.62137,
                (gm_transdist[geoid_o, str(geoid_d)] + 0.1) * 0.62137,
                (gm_autoTT[geoid_o, str(geoid_d)] + 0.1065206),
                (gm_transTT[geoid_o, str(geoid_d)] + 0.1331507),
                (gm_wkTT[geoid_o, str(geoid_d)] + 1.331507),
            )  # Add min dist and convert to miles

            if 7 < mdot_dat_busz.purpose.iloc[i] <= 10:
                p = mc_probs_shop(mdot_dat_busz.iloc[trip, :], shop_ascs, skims)
            else:
                p = mc_probs(mdot_dat_busz.iloc[trip, :], other_ascs, skims)

            t_probs.setdefault(geoid_o, list()).append(
                list(p) + [mdot_dat_busz.weight.iloc[trip]]
            )

    # for i in id_converter.keys():
    #     for j in range(N_MODE):
    #         rows =
