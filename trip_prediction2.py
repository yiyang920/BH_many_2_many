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
def mc_utilities(mdot_per_dat, ascs, skims) -> float:

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


def mc_utilities_shop(mdot_per_dat, ascs, skims) -> float:

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
def dc_utility(dis, emp, mcls) -> float:
    # parameters
    # b_dist = -0.6000
    b_mcls = 1.00
    b_size = 0.87

    return np.exp(b_size * np.log(emp + 1) + b_mcls * mcls)


def trip_prediction(id_converter: dict, mdot_dat: object, stop_zones: set):
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

    # convert mode IDs
    mdot_dat_busz["modeid"] = mdot_dat_busz.mode.apply(
        lambda x: 0 if x == 1 else 4 if x == 4 else 3 if x == 5 else 1 if x == 8 else 2
    )
    # TODO: I AM HERE!
