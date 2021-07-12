import pandas as pd
import numpy as np


def mc_utilities_pk(dist, mdot_per_dat, per_time={}):
    work = mdot_per_dat["emp"] == 0

    # Parameters
    asc_walk = (-0.7659, -1.2387)
    asc_dar = (-1.6562, -1.2189)
    asc_bus = (-2.3093, -2.7091)
    asc_share = (-0.7987, -0.9036)
    asc_drive = (0.0000, 0.0000)

    B_time = (-0.2292, -0.3149)
    B_timeWalk = (-2.8100, -1.2014)
    B_cost = (-1.0689, -0.8501)

    B_worker = (2.1674, 1.8431)

    if per_time:
        wk_time = (dist / 4) * 60
        bus_time = (dist / 30) * 60
        dar_time = (dist / 35) * 60
        auto_time = (dist / 40) * 60
    else:  # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
        wk_time = per_time[0] if 0 in per_time else (dist / 4) * 60  # gg maps api
        dar_time = (
            per_time[1] if 1 in per_time else (dist / 35) * 60
        )  # autotime * 35/40
        bus_time = per_time[2] if 2 in per_time else (dist / 30) * 60  # model output
        auto_time = per_time[4] if 4 in per_time else (dist / 40) * 60  # gg map api

    bus_cost = 1 + dist * 0.05
    dar_cost = 2 + dist * 0.1
    share_cost = dist * 0.6 / 2.5
    drive_cost = dist * 0.6

    umin = 0.001
    u_walk = asc_walk[0] + B_timeWalk[0] * wk_time + umin
    u_bus = asc_bus[0] + B_time[0] * bus_time + B_cost[0] * bus_cost + umin
    u_dar = asc_dar[0] + B_time[0] * dar_time + B_cost[0] * dar_cost + umin
    u_share = asc_share[0] + B_time[0] * auto_time + B_cost[0] * share_cost + umin
    u_drive = B_worker[0] * work + B_time[0] * auto_time + B_cost[0]

    utilities_exp = np.sum(
        (np.exp(u_walk), np.exp(u_bus), np.exp(u_dar), np.exp(u_share), np.exp(u_drive))
    )
    return np.log(utilities_exp + 1e-5)


def mc_utilities_op(dist, mdot_per_dat, per_time={}):
    work = mdot_per_dat["emp"] == 0

    # Parameters
    asc_walk = (-0.7659, -1.2387)
    asc_dar = (-1.6562, -1.2189)
    asc_bus = (-2.3093, -2.7091)
    asc_share = (-0.7987, -0.9036)
    asc_drive = (0.0000, 0.0000)

    B_time = (-0.2292, -0.3149)
    B_timeWalk = (-2.8100, -1.2014)
    B_cost = (-1.0689, -0.8501)

    B_worker = (2.1674, 1.8431)

    if per_time:
        wk_time = (dist / 4) * 60
        bus_time = (dist / 30) * 60
        dar_time = (dist / 35) * 60
        auto_time = (dist / 40) * 60
    else:  # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
        wk_time = per_time[0] if 0 in per_time else (dist / 4) * 60  # gg maps api
        dar_time = (
            per_time[1] if 1 in per_time else (dist / 35) * 60
        )  # autotime * 35/40
        bus_time = per_time[2] if 2 in per_time else (dist / 30) * 60  # model output
        auto_time = per_time[4] if 4 in per_time else (dist / 40) * 60  # gg map api

    bus_cost = 1 + dist * 0.05
    dar_cost = 2 + dist * 0.1
    share_cost = dist * 0.6 / 2.5
    drive_cost = dist * 0.6

    # off-peak travel
    umin = 0.001
    u_walk = asc_walk[1] + B_timeWalk[1] * wk_time + umin
    u_bus = asc_bus[1] + B_time[1] * bus_time + B_cost[1] * bus_cost + umin
    u_dar = asc_dar[1] + B_time[1] * dar_time + B_cost[1] * dar_cost + umin
    u_share = asc_share[1] + B_time[1] * auto_time + B_cost[1] * share_cost + umin
    u_drive = B_worker[1] * work + B_time[1] * auto_time + B_cost[1] * drive_cost + umin

    utilities_exp = np.sum(
        (np.exp(u_walk), np.exp(u_bus), np.exp(u_dar), np.exp(u_share), np.exp(u_drive))
    )

    return np.log(utilities_exp + 1e-5)


def mc_probs_pk(dist, per_time={}):
    dist = dist + 0.5

    # Parameters
    asc_walk = (-0.7659, -1.2387)
    asc_dar = (-1.6562, -1.2189)
    asc_bus = (-2.3093, -2.7091)
    asc_share = (-0.7987, -0.9036)
    asc_drive = (0.0000, 0.0000)

    B_time = (-0.2292, -0.3149)
    B_timeWalk = (-2.8100, -1.2014)
    B_cost = (-1.0689, -0.8501)

    B_worker = (2.1674, 1.8431)

    if per_time:
        wk_time = (dist / 4) * 60
        bus_time = (dist / 30) * 60
        dar_time = (dist / 35) * 60
        auto_time = (dist / 40) * 60
    else:  # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
        wk_time = per_time[0] if 0 in per_time else (dist / 4) * 60  # gg maps api
        dar_time = (
            per_time[1] if 1 in per_time else (dist / 35) * 60
        )  # autotime * 35/40
        bus_time = per_time[2] if 2 in per_time else (dist / 30) * 60  # model output
        auto_time = per_time[4] if 4 in per_time else (dist / 40) * 60  # gg map api

    bus_cost = 1 + dist * 0.05
    dar_cost = 2 + dist * 0.1
    share_cost = dist * 0.6 / 2.5
    drive_cost = dist * 0.6

    # peak travel
    umin = 0.001
    u_walk = asc_walk[0] + B_timeWalk[0] * wk_time + umin
    u_bus = asc_bus[0] + B_time[0] * bus_time + B_cost[0] * bus_cost + umin
    u_dar = asc_dar[0] + B_time[0] * dar_time + B_cost[0] * dar_cost + umin
    u_share = asc_share[0] + B_time[0] * auto_time + B_cost[0] * share_cost + umin
    u_drive = B_worker[0] * 1 + B_time[0] * auto_time + B_cost[0] * drive_cost + umin

    utilities_exp = np.sum(
        (np.exp(u_walk), np.exp(u_bus), np.exp(u_dar), np.exp(u_share), np.exp(u_drive))
    )

    utils = np.array(
        (
            np.exp(u_walk),
            np.exp(u_bus),
            np.exp(u_dar),
            np.exp(u_share),
            np.exp(u_drive),
        )
    )

    probs = utils[0:3] / utilities_exp
    probs = np.append(probs, (1 - np.sum(probs),))

    return tuple(probs)


def mc_probs_op(dist, per_time={}):
    dist = dist + 0.5

    # Parameters
    asc_walk = (-0.7659, -1.2387)
    asc_dar = (-1.6562, -1.2189)
    asc_bus = (-2.3093, -2.7091)
    asc_share = (-0.7987, -0.9036)
    asc_drive = (0.0000, 0.0000)

    B_time = (-0.2292, -0.3149)
    B_timeWalk = (-2.8100, -1.2014)
    B_cost = (-1.0689, -0.8501)

    B_worker = (2.1674, 1.8431)

    if per_time:
        wk_time = (dist / 4) * 60
        bus_time = (dist / 30) * 60
        dar_time = (dist / 35) * 60
        auto_time = (dist / 40) * 60
    else:  # 0: walk, 1: dar, 2: bus, 3: share, 4: auto
        wk_time = per_time[0] if 0 in per_time else (dist / 4) * 60  # gg maps api
        dar_time = (
            per_time[1] if 1 in per_time else (dist / 35) * 60
        )  # autotime * 35/40
        bus_time = per_time[2] if 2 in per_time else (dist / 30) * 60  # model output
        auto_time = per_time[4] if 4 in per_time else (dist / 40) * 60  # gg map api

    bus_cost = 1 + dist * 0.05
    dar_cost = 2 + dist * 0.1
    share_cost = dist * 0.6 / 2.5
    drive_cost = dist * 0.6

    # off-peak travel
    umin = 0.001
    u_walk = asc_walk[1] + B_timeWalk[1] * wk_time + umin
    u_bus = asc_bus[1] + B_time[1] * bus_time + B_cost[1] * bus_cost + umin
    u_dar = asc_dar[1] + B_time[1] * dar_time + B_cost[1] * dar_cost + umin
    u_share = asc_share[1] + B_time[1] * auto_time + B_cost[1] * share_cost + umin
    u_drive = B_worker[1] * 0 + B_time[1] * auto_time + B_cost[1] * drive_cost + umin

    utilities_exp = np.sum(
        (np.exp(u_walk), np.exp(u_bus), np.exp(u_dar), np.exp(u_share), np.exp(u_drive))
    )

    utils = np.array(
        (
            np.exp(u_walk),
            np.exp(u_bus),
            np.exp(u_dar),
            np.exp(u_share),
            np.exp(u_drive),
        )
    )

    probs = utils[0:3] / utilities_exp
    probs = np.append(probs, (1 - np.sum(probs),))

    return tuple(probs)


def dc_utility(dis, emp, mcls):
    # parameters
    b_dist = -0.6000

    b_dist025 = 0.35
    b_dist5 = 0.15
    b_dist10 = 0.00
    b_dist15 = 0.02
    b_dist20 = 0.005
    b_dist30 = 0.00
    b_dist40 = 0.00

    pw_dist = np.array(
        (b_dist025, b_dist5, b_dist10, b_dist15, b_dist20, b_dist30, b_dist30)
    )

    b_mcls = 1.0000
    b_size = 1.0000

    dis_025 = max(dis - 0.25, 0)
    dis_5 = max(dis - 5, 0)
    dis_10 = max(dis - 10, 0)
    dis_15 = max(dis - 15, 0)
    dis_20 = max(dis - 20, 0)
    dis_30 = max(dis - 30, 0)
    dis_40 = max(dis - 40, 0)

    d = np.array((dis_025, dis_5, dis_10, dis_15, dis_20, dis_30, dis_40))

    uility = np.exp(
        b_dist * dis
        + np.sum(np.multiply(pw_dist, d))
        + b_size * np.log(emp + 1)
        + b_mcls * mcls
    )

    return uility
