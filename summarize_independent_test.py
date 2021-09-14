#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 09:44:41 2021

@author: tsukiyamashou
"""

import os
import numpy as np
import pandas as pd

def import_txt(filename):
    #index_list = ["test_sensitivity:", "test_specificity:", "test_accuracy:", "test_mcc:", "test_auc:", "test_precision:", "test_recall:", "test_f1:", "test_AUPRC:"]
    index_list = ["test_sensitivity:", "test_specificity:", "test_accuracy:", "test_mcc:", "test_auc:"]
    res = []
    with open(filename) as f:
        reader = f.readlines()
        
        for index in index_list:
            for row in reader:
                if(index in row):
                    res.append(float(row.split(": ")[-1]))
        
    return res

def output_csv_pandas(filename, data):
    data.to_csv(filename)

deep_model_list = ["LSTM", "CNN"]
encoding_list = ["BE", "Contextual-BE", "NCPNF", "Contextual-NCPNF", "w2v"]

#path = "/Users/kurata/Documents/5mC_pred/data/results/5CV"
#out_path = "/Users/kurata/Documents/5mC_pred/data/results/5-fold_cross-val"
met_index = ["sensitivity", "specificity", "accuracy", "mcc", "auc"]
met_type = ["5CV", "independent_test"]


for met in met_type:
    path = "/Users/kurata/Documents/5mC_pred/data/results/" + str(met) 
    out_path = "/Users/kurata/Documents/5mC_pred/data/results/" + str(met) 
    for model in deep_model_list:
        for enc in encoding_list:
            res_all = []
            for i in range(1,6):
                res = import_txt(path + "/" + model + "/" + str(enc) + "/" + str(i) + "/deep_HV_result.txt")
                res_all.append(res)
            ave_res = np.mean(np.array(res_all), axis = 0).tolist()
            res_all = pd.DataFrame(np.array(res_all).T, index = met_index, columns = [1, 2, 3, 4, 5])
            res_all["average"] = ave_res
            output_csv_pandas(out_path + "/" + model + "/performances_" + str(model) + "_" + str(enc) + ".csv", res_all)



        
        
        
        
        
        
        
        
        
        
        