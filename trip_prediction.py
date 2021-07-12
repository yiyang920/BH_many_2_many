import pandas as pd
import numpy as np
import pickle
import csv

from mc_utilities import (
    mc_utilities_pk,
    mc_utilities_op,
    mc_probs_pk,
    mc_probs_op,
    dc_utility,
)


def trip_prediction(
    id_converter, per_dist, per_emp, mdot_dat, dests_geo, D, per_time={}
):
    # Calculate MCLS for each person trip
    N = len(mdot_dat)
    N_MODE = 5
    mcls_peak = np.zeros((N, N_MODE))
    mcls_offpeak = np.zeros((N, N_MODE))

    exp_u_holder_pk = np.zeros(N_MODE)
    exp_u_holder_op = np.zeros(N_MODE)
    dc_prob_pk = np.zeros((N, N_MODE))
    dc_prob_op = np.zeros((N, N_MODE))

    # Predicted trip location (peak and off peak)
    pred_dests_pk = np.zeros(N)
    pred_dests_op = np.zeros(N)

    for i in range(N):
        for j in range(N_MODE):
            if not per_time:
                t = {mode: per_time[mode][i] for mode in per_time.keys()}
            else:
                t = {}

            # Calculate MCLS - peak trips
            mcls_peak[i, j] = mc_utilities_pk(
                per_dist.iloc[i, j], mdot_dat.iloc[i, :], per_time=t
            )
            # Calculate MCLS - off peak trips
            mcls_offpeak[i, j] = mc_utilities_op(
                per_dist.iloc[i, j], mdot_dat.iloc[i, :], per_time=t
            )

            # Calculate Destination Choice Utilites - peak trips
            exp_u_holder_pk[j] = dc_utility(
                per_dist.iloc[i, j], per_emp.iloc[i, j], mcls_peak[i, j]
            )
            # Calculate Destination Choice Utilites - off-peak trips
            exp_u_holder_op[j] = dc_utility(
                per_dist.iloc[i, j], per_emp.iloc[i, j], mcls_offpeak[i, j]
            )

        # Calculate Destination Choice Probabilities - peak trips
        dc_prob_pk[i, 0 : N_MODE - 2] = exp_u_holder_pk[0 : N_MODE - 2] / np.sum(
            exp_u_holder_pk
        )
        dc_prob_pk[i, N_MODE - 1] = 1 - np.sum(dc_prob_pk[i, 0 : N_MODE - 2])
        # Calculate Destination Choice Probabilities - off-peak trips
        dc_prob_op[i, 0 : N_MODE - 2] = exp_u_holder_op[0 : N_MODE - 2] / np.sum(
            exp_u_holder_op
        )
        dc_prob_op[i, N_MODE - 1] = 1 - np.sum(dc_prob_pk[i, 0 : N_MODE - 2])

        # Get predicted trip location (peak and off peak)
        pred_dests_pk[i] = dests_geo.iloc[i, np.argmax(dc_prob_pk[i, :])]
        pred_dests_op[i] = dests_geo.iloc[i, np.argmax(dc_prob_op[i, :])]

    # Expansion
    # Generate distance matrix from MDOT dataset (broader than study region)
    data_pk = pd.DataFrame(
        data={
            "geoid_o": mdot_dat.loc[:, "geoid_o"],
            "pre_dests_pk": pred_dests_pk,
            "PERRKWT0": mdot_dat.loc[:, "PERRKWT0"],
        }
    )
    data_pk = data_pk.loc[mdot_dat["u_perid"] == 1, :]

    data_op = pd.DataFrame(
        data={
            "geoid_o": mdot_dat.loc[:, "geoid_o"],
            "pre_dests_op": pred_dests_op,
            "PERRKWT0": mdot_dat.loc[:, "PERRKWT0"],
        }
    )
    data_op = data_op.loc[mdot_dat["u_perid"] == 0, :]

    trips_mat_pk = {
        (i, j): np.sum(
            data_pk.loc[
                (
                    (data_pk["geoid_o"] == id_converter[i])
                    & (data_pk["pre_dests_pk"] == id_converter[j])
                ),
                "PERRKWT0",
            ]
        )
        for i in id_converter
        for j in id_converter
    }

    trips_mat_op = {
        (i, j): np.sum(
            data_op.loc[
                (
                    (data_op["geoid_o"] == id_converter[i])
                    & (data_op["pre_dests_op"] == id_converter[j])
                ),
                "PERRKWT0",
            ]
        )
        for i in id_converter
        for j in id_converter
    }

    transit_trips_mat_pk = dict()
    transit_trips_mat_op = dict()

    for i in id_converter:
        for j in id_converter:
            if not per_time:
                t = {mode: per_time[mode][i] for mode in per_time.keys()}
            else:
                t = {}
            if str(id_converter[i]) in D.columns:
                probs_pk = mc_probs_pk(
                    D.loc[id_converter[i], str(id_converter[j])], per_time=t
                )
                transit_trips_mat_pk[i, j] = (probs_pk[1] + probs_pk[2]) * trips_mat_pk[
                    i, j
                ]
                probs_op = mc_probs_op(
                    D.loc[id_converter[i], str(id_converter[j])], per_time=t
                )
                transit_trips_mat_op[i, j] = (probs_op[1] + probs_op[2]) * trips_mat_op[
                    i, j
                ]

    return trips_mat_pk, trips_mat_op, transit_trips_mat_pk, transit_trips_mat_op


if __name__ == "__main__":
    print("start testing...")
    id_converter = pickle.load(open(r"Data\id_converter.p", "rb"))

    # Load data
    mc_fileloc = r"mc_input_data\\"
    per_dist = pd.read_csv(mc_fileloc + "dists_dat.csv", sep=",")
    per_emp = pd.read_csv(mc_fileloc + "dests_emp.csv", sep=",")
    mdot_dat = pd.read_csv(mc_fileloc + "mdot_trips_dc.csv", sep=",")
    dests_geo = pd.read_csv(mc_fileloc + "dests_geoid.csv", sep=",")
    D = pd.read_csv(mc_fileloc + "distance.csv", sep=",", index_col=[0])

    (
        trips_mat_pk,
        trips_mat_op,
        transit_trips_mat_pk,
        transit_trips_mat_op,
    ) = trip_prediction(id_converter, per_dist, per_emp, mdot_dat, dests_geo, D)

    trips_mat_day = {
        (i, j): trips_mat_pk[i, j] + trips_mat_op[i, j]
        for i in range(len(id_converter))
        for j in range(len(id_converter))
    }
    transit_trips_mat_day = {
        (i, j): transit_trips_mat_pk[i, j] + transit_trips_mat_op[i, j]
        for i in range(len(id_converter))
        for j in range(len(id_converter))
    }

    transit_trip_pk = np.zeros((len(id_converter), len(id_converter)))
    transit_trip_op = np.zeros((len(id_converter), len(id_converter)))
    transit_trip_day = np.zeros((len(id_converter), len(id_converter)))

    trip_pk = np.zeros((len(id_converter), len(id_converter)))
    trip_op = np.zeros((len(id_converter), len(id_converter)))
    trip_day = np.zeros((len(id_converter), len(id_converter)))

    for i in range(len(id_converter)):
        for j in range(len(id_converter)):
            transit_trip_pk[i, j] = transit_trips_mat_pk[i, j]
            transit_trip_op[i, j] = transit_trips_mat_op[i, j]
            transit_trip_day[i, j] = (
                transit_trips_mat_pk[i, j] + transit_trips_mat_op[i, j]
            )

            trip_pk[i, j] = trips_mat_pk[i, j]
            trip_op[i, j] = trips_mat_op[i, j]
            trip_day[i, j] = trips_mat_day[i, j]

    trips_df = pd.DataFrame(
        index=np.arange(len(id_converter)),
        columns=[
            "trips_pk_in",
            "trips_pk_out",
            "trips_op_in",
            "trips_op_out",
            "trips_day_in",
            "trips_day_out",
            "transit_trips_pk_in",
            "transit_trips_pk_out",
            "transit_trips_op_in",
            "transit_trips_op_out",
            "transit_trips_day_in",
            "transit_trips_day_out",
        ],
    )

    for i in range(len(id_converter)):
        trips_df.loc[i, "trips_pk_in"] = sum(
            v for (k, v) in trips_mat_pk.items() if k[1] == i
        )
        trips_df.loc[i, "trips_op_in"] = sum(
            v for (k, v) in trips_mat_op.items() if k[1] == i
        )
        trips_df.loc[i, "trips_pk_out"] = sum(
            v for (k, v) in trips_mat_pk.items() if k[0] == i
        )
        trips_df.loc[i, "trips_op_out"] = sum(
            v for (k, v) in trips_mat_op.items() if k[0] == i
        )
        trips_df.loc[i, "trips_day_in"] = (
            trips_df.loc[i, "trips_pk_in"] + trips_df.loc[i, "trips_op_in"]
        )
        trips_df.loc[i, "trips_day_out"] = (
            trips_df.loc[i, "trips_pk_out"] + trips_df.loc[i, "trips_op_out"]
        )

        trips_df.loc[i, "transit_trips_pk_in"] = sum(
            v for (k, v) in transit_trips_mat_pk.items() if k[1] == i
        )
        trips_df.loc[i, "transit_trips_op_in"] = sum(
            v for (k, v) in transit_trips_mat_op.items() if k[1] == i
        )
        trips_df.loc[i, "transit_trips_pk_out"] = sum(
            v for (k, v) in transit_trips_mat_pk.items() if k[0] == i
        )
        trips_df.loc[i, "transit_trips_op_out"] = sum(
            v for (k, v) in transit_trips_mat_op.items() if k[0] == i
        )
        trips_df.loc[i, "transit_trips_day_in"] = (
            trips_df.loc[i, "transit_trips_pk_in"]
            + trips_df.loc[i, "transit_trips_op_in"]
        )
        trips_df.loc[i, "transit_trips_day_out"] = (
            trips_df.loc[i, "transit_trips_pk_out"]
            + trips_df.loc[i, "transit_trips_op_out"]
        )

    trips_df.index.name = "zone_id"

    # Save files=====================================================
    trips_df.to_csv(r"mc_output\trips_sum.csv")
    np.savetxt(r"mc_output\transit_trips_pk.csv", transit_trip_pk, delimiter=",")
    np.savetxt(r"mc_output\transit_trips_op.csv", transit_trip_op, delimiter=",")
    np.savetxt(r"mc_output\transit_trips_day.csv", transit_trip_day, delimiter=",")
    np.savetxt(r"mc_output\trips_pk.csv", trip_pk, delimiter=",")
    np.savetxt(r"mc_output\trips_op.csv", trip_op, delimiter=",")
    np.savetxt(r"mc_output\trips_day.csv", trip_day, delimiter=",")

    with open(r"mc_output\transit_trips_dict_pk.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in transit_trips_mat_pk.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])
    with open(r"mc_output\transit_trips_dict_op.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in transit_trips_mat_op.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])
    with open(r"mc_output\transit_trips_dict_day.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in transit_trips_mat_day.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])

    with open(r"mc_output\trips_dict_pk.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in trips_mat_pk.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])
    with open(r"mc_output\trips_dict_op.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in trips_mat_op.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])
    with open(r"mc_output\trips_dict_day.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zone_id_o", "zone_id_d", "trips"])
        for key, value in trips_mat_day.items():
            if value > 0:
                writer.writerow([key[0], key[1], value])

    print(
        "total transit trips for peak hour: {}".format(
            sum(transit_trips_mat_pk.values())
        )
    )
    print(
        "total transit trips for off-peak hour: {}".format(
            sum(transit_trips_mat_op.values())
        )
    )

    print(
        "total transit trips for the day: {}".format(
            sum(transit_trips_mat_day.values())
        )
    )
