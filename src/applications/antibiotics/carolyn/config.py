import numpy as np
from scipy.io import loadmat

from config import DIR_DATA_CAROLYN

datasets = {
    "10000": [
        '20170715_Fowler_1-23_LB_10000.mat',
        '20170715_Fowler_24-46_LB_10000.mat',
        '20170716_Fowler_47-69_LB_10000.mat',
        '20170716_Fowler_70-92_LB_10000.mat',
        '20170717_Fowler_93-106_LB_10000.mat',
        '20170719_Fowler_107-129_LB_10000.mat',
        '20170720_Fowler_130-149_LB_10000.mat',
        '20170721_Fowler_150-172_LB_10000.mat',
        '20170722_Fowler_173-195_LB_10000.mat',
        '20170722_Fowler_196-218_LB_10000.mat',
        '20170805_Anderson_1-20_LB_10000.mat',
        '20170806_Anderson_21-40_LB_10000.mat',
        '20170806_Anderson_41-59_LB_10000.mat',
    ],
    "lambda_MOI1_10000": [
        '20170807_Fowler_1-23_LB_lambda_MOI-1_10000.mat',
        '20170807_Fowler_24-46_LB_lambda_MOI-1_10000.mat',
        '20170808_Fowler_47-69_LB_lambda_MOI-1_10000.mat',
        '20170808_Fowler_70-92_LB_lambda_MOI-1_10000.mat',
        '20170809_Fowler_116-138_LB_lambda_MOI-1_10000.mat',
        '20170809_Fowler_139-161_LB_lambda_MOI-1_10000.mat',
        '20170809_Fowler_93-115_LB_lambda_MOI-1_10000.mat',
        '20170810_Fowler_162-184_LB_lambda_MOI-1_10000.mat',
        '20170810_Fowler_185-207_LB_lambda_MOI-1_10000.mat',
        '20170810_Fowler_208-218_LB_lambda_MOI-1_10000.mat',
        '20170811_Anderson_1-23_LB_lambda_MOI-1_10000.mat',
        '20170811_Anderson_24-46_LB_lambda_MOI-1_10000.mat',
        '20170811_Anderson_47-59_LB_lambda_MOI-1_10000.mat',
    ],
    "100": [
        '20170818_Fowler_1-23_LB_100.mat',
        '20170818_Fowler_24-46_LB_100.mat',
        '20170818_Fowler_47-69_LB_100.mat',
        '20170819_Fowler_116-138_LB_100.mat',
        '20170819_Fowler_70-92_LB_100.mat',
        '20170819_Fowler_93-115_LB_100.mat',
        '20170820_Fowler_139-161_LB_100.mat',
        '20170820_Fowler_162-184_LB_100.mat',
        '20170820_Fowler_185-207_LB_100.mat',
        '20170821_Fowler_208-218_LB_100.mat',
        '20170821_Anderson_1-23_LB_100.mat',
        '20170821_Anderson_24-46_LB_100.mat',
        '20170821_Anderson_47-59_LB_100.mat',
    ],
    "Carb-5ugml_10000": [
        '20170903_Fowler_1-23_LB_Carb-5ugml_10000.mat',
        '20170904_Fowler_24-46_LB_Carb-5ugml_10000.mat',
        '20170904_Fowler_47-69_LB_Carb-5ugml_10000.mat',
        '20170905_Fowler_70-92_LB_Carb-5ugml_10000.mat',
        '20170906_Fowler_93-115_LB_Carb-5ugml_10000.mat',
        '20170907_Fowler_139-161_LB_Carb-5ugml_10000.mat',
        '20170908_Fowler_162-184_LB_Carb-5ugml_10000.mat',
        '20170908_Fowler_185-207_LB_Carb-5ugml_10000.mat',
        '20170909_Fowler_116-138_LB_Carb-5ugml_10000.mat',
        '20170909_Fowler_208-218_LB_Carb-5ugml_10000.mat',
        '20170910_Anderson_1-23_LB_Carb-5ugml_10000.mat',
        '20170910_Anderson_24-46_LB_Carb-5ugml_10000.mat',
        '20170911_Anderson_47-59_LB_Carb-5ugml_10000.mat',
    ],
}

strain_names = loadmat(DIR_DATA_CAROLYN / "data_dependency_files" / "strainNames.mat")
fowler_labels_1 = strain_names["myLabelLB"].ravel()
fowler_labels_2 = strain_names["myLabel"].ravel()

anderson_labels = np.array([1,2,3,4,5,6,7,9,10,13,16,17,18,19,20,21,24,25,27,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,63,64,65,66,67,68,70,71,])
