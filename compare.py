import pickle
import numpy as np

result_loc = "many2many_output//results//"
n_transfer_current = pickle.load(open(result_loc + "current//n_transfer_0.p", "rb"))
n_transfer_const1 = pickle.load(open(result_loc + "contrained_1//n_transfer_0.p", "rb"))
n_transfer_const2 = pickle.load(open(result_loc + "contrained_2//n_transfer_0.p", "rb"))
n_transfer_exp1 = pickle.load(
    open(result_loc + "expansion_1_double//n_transfer_0.p", "rb")
)
n_transfer_exp2 = pickle.load(
    open(result_loc + "expansion_2_double//n_transfer_0.p", "rb")
)

print(
    "average transfer current: {}".format(
        sum(k * v for k, v in n_transfer_current.items())
    )
)
print(
    "average transfer constrained 1: {}".format(
        sum(k * v for k, v in n_transfer_const1.items())
    )
)
print(
    "average transfer constrained 2: {}".format(
        sum(k * v for k, v in n_transfer_const2.items())
    )
)
print(
    "average transfer expansion 1: {}".format(
        sum(k * v for k, v in n_transfer_exp1.items())
    )
)
print(
    "average transfer expansion 2: {}".format(
        sum(k * v for k, v in n_transfer_exp2.items())
    )
)

ratio_list_current = pickle.load(open(result_loc + "current//ratio_list_0.p", "rb"))
ratio_list_const1 = pickle.load(open(result_loc + "contrained_1//ratio_list_0.p", "rb"))
ratio_list_const2 = pickle.load(open(result_loc + "contrained_2//ratio_list_0.p", "rb"))
ratio_list_exp1 = pickle.load(
    open(result_loc + "expansion_1_double//ratio_list_0.p", "rb")
)
ratio_list_exp2 = pickle.load(
    open(result_loc + "expansion_2_double//ratio_list_0.p", "rb")
)

print("average ATT-SPTT ratio current: {}".format(np.mean(ratio_list_current)))
print("average ATT-SPTT ratio constrained 1: {}".format(np.mean(ratio_list_const1)))
print("average ATT-SPTT ratio constrained 2: {}".format(np.mean(ratio_list_const2)))
print("average ATT-SPTT ratio expansion 1: {}".format(np.mean(ratio_list_exp1)))
print("average ATT-SPTT ratio expansion 2: {}".format(np.mean(ratio_list_exp2)))

factor = 180 // 60
ratio_list_current_DaR = (
    pickle.load(open(result_loc + "current_Dar_40//ratio_list_0.p", "rb")) * factor
)
ratio_list_const1_DaR = (
    pickle.load(open(result_loc + "contrained_1_Dar_40//ratio_list_0.p", "rb")) * factor
)
ratio_list_const2_DaR = (
    pickle.load(open(result_loc + "contrained_2_Dar_40//ratio_list_0.p", "rb")) * factor
)
ratio_list_exp1_DaR = (
    pickle.load(open(result_loc + "expansion_1_double_Dar_40//ratio_list_0.p", "rb"))
    * factor
)
ratio_list_exp2_DaR = (
    pickle.load(open(result_loc + "expansion_2_double_Dar_40//ratio_list_0.p", "rb"))
    * factor
)

print(
    "average ATT-SPTT ratio current DaR: {}".format(
        np.mean(ratio_list_current + ratio_list_current_DaR)
    )
)
print(
    "average ATT-SPTT ratio constrained 1 DaR: {}".format(
        np.mean(ratio_list_const1 + ratio_list_const1_DaR)
    )
)
print(
    "average ATT-SPTT ratio constrained 2 DaR: {}".format(
        np.mean(ratio_list_const2 + ratio_list_const2_DaR)
    )
)
print(
    "average ATT-SPTT ratio expansion 1 DaR: {}".format(
        np.mean(ratio_list_exp1 + ratio_list_exp1_DaR)
    )
)
print(
    "average ATT-SPTT ratio expansion 2 DaR: {}".format(
        np.mean(ratio_list_exp2 + ratio_list_exp2_DaR)
    )
)
