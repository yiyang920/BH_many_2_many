import numpy as np
import pandas as pd
from python_tsp.exact import solve_tsp_dynamic_programming


def direct_disagg(
    Route_D_agg: dict[int, list[tuple[int, int, int, int, int, int]]],
    OD_d: dict[int, dict[str, int]],
    OD_r: dict[int, int],
    Driver: "np.ndarray[np.int64]",
    tau_disagg: "np.ndarray[np.int64]",
    tau_agg: "np.ndarray[np.int64]",
    ctr: dict[int, list[int]],
    agg_2_disagg_id: dict[int, list[int]],
    disagg_2_agg_id: dict[int, int],
    config: dict[str, any],
    fixed_route_D: dict = None,
):
    """
    Direct disaggregation algorithm:
    Input:
    Route_D_agg -> {d: list((t1, t2, s1, s2, n1, n2))}
    OD_d -> {d: {"O": o, "D": d}}, disaggregated network
    OD_r -> {r: {"O": o, "D": d}}, OD of served riders, disaggregated network
    tau_disagg ->  zone-to-zone travel time of disaggregated network, numpy array
    tau_agg -> zone-to-zone travel time of disaggregated network
            with diagonal elements equal to one, numpy array
    ctr ->  a dictionary with key as zones and values be the set of neighbor
            zones for the key including the key itself (disaggregated network)
    """
    # time horizon of one tour of all drivers
    T = config["T_post_processing"]
    # number of disagg zones
    S = config["S_disagg"]

    # set of drivers, drivers' origins and destinations of disagg network
    D = set(OD_d.keys())

    # disagg nodes need to be visited to serve passengers
    S_r = set()
    for OD_dictionary in OD_r.values():
        S_r.add(OD_dictionary["O"])
        S_r.add(OD_dictionary["D"])

    # time window for each driver. If repeated tour mode,
    # TW_d should be the time window for the first single tour
    TW_d = {d: Driver[i, 3] - Driver[i, 2] for i, d in enumerate(Driver[:, 0])}
    # each driver only register its first tour info
    for d in D:
        Route_D_agg[d] = [
            station for station in Route_D_agg[d] if station[1] <= TW_d[d]
        ]
    # node set for each driver, disagg
    S_d = {
        d: set(
            s
            for station in route
            # if station[1] <= TW_d[d]
            for s in agg_2_disagg_id[station[2]]
            # if s in S_r
        )
        for d, route in Route_D_agg.items()
    }
    for d in D:
        # add destination station of each route
        S_d[d].update(agg_2_disagg_id[Route_D_agg[d][-1][3]])

    # S_info is a dictionary with structure {s: {d: [t1, t2,...]}}
    # S_d_tilda: virtual node set for each driver {d: (s, d, t)}
    # H_d_tilda: virtual hub set for each driver {d: (s, d, t)}
    S_info, Route_D_agg_2, S_d_tilda, H_d_tilda = dict(), dict(), dict(), dict()
    for d, route in Route_D_agg.items():
        for i, s in enumerate(route):
            # each driver only register its first tour info
            if s[1] <= TW_d[d]:
                # if (s[2] not in S_info) or (d not in S_info[s[2]]):
                S_info.setdefault(s[2], dict()).setdefault(d, list()).append(s[0])
        # register destination station
        S_info.setdefault(route[-1][3], dict()).setdefault(d, list()).append(s[1])
    H_d = dict()  # hub set for each driver, no time info
    for s, d_dict in S_info.items():
        if len(d_dict) < 2:
            for d, t_list in d_dict.items():
                for t in t_list:
                    S_d_tilda.setdefault(d, list()).append((s, d, t))
        else:
            for d, t_list in d_dict.items():
                for t in t_list:
                    H_d_tilda.setdefault(d, list()).append((s, d, t))
                    H_d.setdefault[d, list()].append(s)

    df_route_agg = pd.DataFrame()
    for d, route in Route_D_agg.items():
        df_route_agg = pd.concat(
            [df_route_agg, pd.DataFrame({d: route})], ignore_index=True, axis=1
        )
    os = {d: OD_d[d]["O"] for d in D}  # origin sub-node within each super-node
    ds = dict()  # end sub-node within each super-node

    # t_d is the leaving time of each sub-node for each driver
    Route_D_disagg, t_d = (dict(), dict())
    sub_station = (
        list()
    )  # list[tuple[int, int, int, int]], list of (t1, t2, s1, s2) within each super-node
    for index, row in df_route_agg.iterrows():
        for d in D:
            Route_D_disagg.setdefault(d, list())
            t_d.setdefault(d, 0)
            # if not the end of route for driver d
            if np.notna(row[d]):
                station = list(set(agg_2_disagg_id[row[d][2]]))

                # make sure the DS_d will always be the destination node
                if row[d][1] == TW_d[d]:
                    station.remove(OD_d[d]["D"])

                # in case the last super-node has only one sub-node, i.e. DS_d,
                # where after removing DS_d, station will be an empty list
                if not station:
                    Route_D_disagg[d].append(
                        (
                            ds[d],
                            OD_d[d]["D"],
                            t_d[d],
                            t_d[d] + tau_disagg[ds[d], OD_d[d]["D"]],
                        )
                    )
                    t_d[d] += tau_disagg[ds[d], OD_d[d]["D"]]
                    ds[d] = OD_d[d]["D"]
                # when station has at least one sub-node
                else:
                    # if not the first super-node, os_d will be the closest sub-node
                    # to the ds_d[d] from previous partition
                    if index > 0:
                        os[d] = station[
                            np.argmin(tau_disagg[ds[d], s] for s in station)
                        ]

                    p_size = len(station)
                    distance_matrix = np.zeros((p_size, p_size))
                    for i in range(p_size):
                        for j in range(p_size):
                            if i != j:
                                distance_matrix[i, j] = tau_disagg[
                                    station[i], station[j]
                                ]
                    sub_route, _ = solve_tsp_dynamic_programming(distance_matrix)
                    # index of the first substation
                    O_idx = sub_route.index(os[d])
                    # rotate to the first substation
                    sub_route = sub_route[O_idx:] + sub_route[:O_idx]

                    if row[d][1] < TW_d[d]:
                        ds[d] = sub_route[-1]
                    else:
                        sub_route.append(OD_d[d]["D"])
                        ds[d] = OD_d[d]["D"]

                    for s1, s2 in zip(sub_route[:-1], sub_route[1:]):
                        sub_station.append(
                            (s1, s2, t_d[d], t_d[d] + tau_disagg[s1, s2])
                        )
                        t_d[d] += tau_disagg[s1, s2]

                    Route_D_disagg[d] += sub_station
