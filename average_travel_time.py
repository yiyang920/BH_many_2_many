import pickle, yaml
import numpy as np
import pandas as pd

with open("config_DaR.yaml", "r+") as fopen:
    config = yaml.load(fopen, Loader=yaml.FullLoader)

data_loc = "many2many_data/"
result_loc = "many2many_output/results/"

Rider_current_DaR = pd.read_csv(data_loc + "current_DaR_40\\Rider.csv")
Rider_const1_DaR = pd.read_csv(data_loc + "contrained_1_DaR_40\\Rider.csv")
Rider_const2_DaR = pd.read_csv(data_loc + "contrained_2_DaR_40\\Rider.csv")
Rider_expansion_1_double_DaR = pd.read_csv(
    data_loc + "expansion_1_double_DaR_40\\Rider.csv"
)
Rider_expansion_2_double_DaR = pd.read_csv(
    data_loc + "expansion_2_double_DaR_40\\Rider.csv"
)

ratio_list_current_DaR = pickle.load(
    open(result_loc + "current_Dar_40//ratio_list_0.p", "rb")
)
ratio_list_const1_DaR = pickle.load(
    open(result_loc + "contrained_1_Dar_40//ratio_list_0.p", "rb")
)
ratio_list_const2_DaR = pickle.load(
    open(result_loc + "contrained_2_Dar_40//ratio_list_0.p", "rb")
)
ratio_list_exp1_DaR = pickle.load(
    open(result_loc + "expansion_1_double_Dar_40//ratio_list_0.p", "rb")
)
ratio_list_exp2_DaR = pickle.load(
    open(result_loc + "expansion_2_double_Dar_40//ratio_list_0.p", "rb")
)
