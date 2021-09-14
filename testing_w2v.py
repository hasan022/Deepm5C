#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:24 2021

@author: tsukiyamashou
"""

import sys
sys.path.append("C:/Users/sho/Documents/project_1_7_20/program")
import os
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
#import torch_optimizer as optim
from torch import optim
from torch.utils.data import BatchSampler
import numpy as np
from numpy import argmax
import joblib
import argparse
from gensim.models import KeyedVectors
from gensim.models import word2vec
import copy
import json
#from Bert import BERT
#from Bert_bidirectional_LSTM import BERT
#from GRU_network import Gru
#from LSTM_network import Lstm
#from GRU_network_bidirectional import bGRU
from LSTM_network_bidirectional import bLSTM
#from CNN_LSTM_network import CNN_LSTM
#from CNN_network import CNN
import collections
import time
import pickle
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
#from class_balanced_loss import CBLoss
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
from sklearn.model_selection import StratifiedKFold
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
        
    return data

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)

    return data

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)

    return data

def emb_seq(seq, w2v_model, num = 4):
    #aa_dict = {"A": [1,0,0,0], "T": [0,1,0,0], "G": [0,0,1,0], "C": [0,0,0,1], "N": [0,0,0,0]}
    #seq_emb = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([4 * num]) for i in range(len(seq) - num + 1)])
    
    seq_emb = np.array([np.array(w2v_model[seq[i:i+num]]) for i in range(len(seq) - num + 1)])

    return seq_emb

def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)

class pv_data_sets(data.Dataset):
    #def __init__(self, data_sets):
    def __init__(self, data_sets, w2v_model):
        super().__init__()
        self.w2v_model = w2v_model
        self.seq = data_sets["seq"].values.tolist()
        self.labels = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"].values.tolist()),1]).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #emb_mat = emb_seq(self.seq[idx])
        emb_mat = emb_seq(self.seq[idx], self.w2v_model)
        label = self.labels[idx]
        
        return torch.tensor(emb_mat).cuda().float(), torch.tensor(label).cuda()

class burt_process():
    def __init__(self, out_path, deep_model_path, batch_size = 32, features = 100, thresh = 0.5):
        self.out_path = out_path
        self.deep_model_path = deep_model_path
        #self.model_type = model_type
        self.batch_size = batch_size
        self.features = features
        self.thresh = thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def pre_training(self, dataset, w2v_model):
        os.makedirs(self.out_path, exist_ok = True) 
        data_all = pv_data_sets(dataset, w2v_model)
        #balanced_sampler_train = BinarySampler(train_data_sets["label"].values.tolist(), self.tra_batch_size)
        #train_loader = DataLoader(dataset = tra_data_all, batch_sampler = balanced_sampler_train)
        loader = DataLoader(dataset = data_all, batch_size = self.batch_size, shuffle=False)

        #if(self.model_type == "bert"):
            #from Bert import BERT
            #net = BERT(n_layers = 3, d_model = self.features, n_heads = 4, d_dim = 100, d_ff = 400, time_seq = 41 - 4 + 1).cuda()
        #elif(self.model_type == "bert_lstm"):
            #from Bert_bidirectional_LSTM import BERT as BERT_LSTM
            #net = BERT_LSTM(n_layers = 3, d_model = self.features, n_heads = 4, d_dim = 100, d_ff = 400, lstm_hidden_size = 128).cuda()
        #elif(self.model_type == "gru"):
            #net = Gru(features = self.features, gru_hidden_size = 128).cuda()
        #elif(self.model_type == "lstm"):
        #net = Lstm(features = self.features, lstm_hidden_size = 128).cuda()
        #elif(self.model_type == "bgru"):
            #net = bGRU(features = self.features, gru_hidden_size = 128).cuda()
        #elif(self.model_type == "blstm"):
        net = bLSTM(features = self.features, lstm_hidden_size = 128).cuda()
        #elif(self.model_type == "cnn_lstm"):
            #net = CNN_LSTM(features = self.features, lstm_hidden_size = 128).cuda()
        #elif(self.model_type == "cnn"):
            #net = CNN(features = self.features, time_size = 41 - 4 + 1).cuda()
        #else:
            #print("Error")
            #sys.exit()

        net.load_state_dict(torch.load(self.deep_model_path, map_location = self.device))
            
        with open(self.out_path + "/deep_HV_result.txt", 'w') as f:
            print(self.out_path, file = f, flush=True)
            print("The number of training data:" + str(len(dataset)), file = f, flush=True)
            
            self.probs, self.labels = [], []
            
            print("testing...", file = f, flush=True)
            net.eval()
            for i, (emb_mat, label) in enumerate(loader):
                with torch.no_grad():
                    outputs = net(emb_mat)
                        
                self.probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                self.labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
            #print("test_threshold:: value: %f" % (str(self.thresh)), file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    met = metrics_dict[key](self.labels, self.probs, thresh = self.thresh)
                else:
                    met = metrics_dict[key](self.labels, self.probs)
                print("test_" + key + ": " + str(met), file = f, flush=True)
                
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.labels, self.probs, thresh = self.thresh)
            print("test_true_negative:: value: %f" % (tn_t), file = f, flush=True)
            print("test_false_positive:: value: %f" % (fp_t), file = f, flush=True)
            print("test_false_negative:: value: %f" % (fn_t), file = f, flush=True)
            print("test_true_positive:: value: %f" % (tp_t), file = f, flush=True)
        
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputpath', help='Path')
parser.add_argument('-o', '--outpath', help='Path')
parser.add_argument('-t', '--threshold', help='Path', default = 0.5)
parser.add_argument('-dm', '--deep_model_path', help='Path')
parser.add_argument('-wm', '--w2v_model_path', help='Path')
#parser.add_argument('-dt', '--deep_model_type', help='Path')

test_path = parser.parse_args().inputpath
out_path = parser.parse_args().outpath
threshold = float(parser.parse_args().threshold)
deep_model_path = parser.parse_args().deep_model_path
w2v_model_path = parser.parse_args().w2v_model_path
#model_type = parser.parse_args().deep_model_type 

w2v_model = word2vec.Word2Vec.load(w2v_model_path)
dataset = file_input_csv(test_path ,index_col = None)

#for model_type in ["bert","bert_lstm","gru","lstm","bgru","blstm","cnn_lstm","cnn"]:
for i in range(1, 6):
    results_probs = []
    net = burt_process(out_path + "/" + str(i), deep_model_path + "/" + str(i) + "/data_model/deep_model", batch_size = 64, thresh = threshold)
    net.pre_training(dataset, w2v_model)
    results_probs.append(net.probs)
    results_probs.append(net.labels)
    results_probs = pd.DataFrame(results_probs, index = ["probability_scores", "labels"]).transpose()
    output_csv_pandas(out_path + "/" + str(i) + "/prob_scores.csv", results_probs)





























