#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:24 2021

@author: mehedi
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
from sklearn.metrics import precision_recall_curve
import joblib
import argparse
from gensim.models import KeyedVectors
from gensim.models import word2vec
import copy
import json

from LSTM_network_bidirectional import bLSTM
#from CNN_LSTM_network import CNN_LSTM

import time
import pickle
import sklearn.metrics as metrics
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

#def emb_seq(seq, num = 4):
def emb_seq(seq, w2v_model, num = 4):
    #aa_dict = {"A": [1,0,0,0], "T": [0,1,0,0], "G": [0,0,1,0], "C": [0,0,0,1], "N": [0,0,0,0]}
    #seq_emb = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([4 * num]) for i in range(len(seq) - num + 1)])
    
    seq_emb = np.array([np.array(w2v_model[seq[i:i+num]]) for i in range(len(seq) - num + 1)])

    return seq_emb

class pv_data_sets(data.Dataset):
    #def __init__(self, data_sets):
    def __init__(self, data_sets, w2v_model):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist()
        self.labels = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"].values.tolist()),1]).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #emb_mat = emb_seq(self.seq[idx])
        emb_mat = emb_seq(self.seq[idx], w2v_model)
        label = self.labels[idx]
        
        return torch.tensor(emb_mat).cuda().float(), torch.tensor(label).cuda()

class BinarySampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.index = np.arange(len(self.labels))
        np.random.shuffle(self.index)
        self.labels_temp = self.labels[self.index]
        self.batch_count = int(len(self.labels)/batch_size)
        self.index = self.index[0:self.batch_count*batch_size]
        self.labels_temp = self.labels_temp[0:self.batch_count*batch_size]
       
        self.batch_index_all = []
        self.skf = StratifiedKFold(self.batch_count)
        for _, batch_index in self.skf.split(self.index, self.labels_temp):
            self.batch_index_all.append(batch_index)

    def __iter__(self):
        for i in range(len(self.batch_index_all)):
            yield self.index[self.batch_index_all[i]]

    def __len__(self):
        return self.batch_count

class burt_process():
    def __init__(self, out_path, tra_batch_size = 128, val_batch_size = 128, features = 100, lr = 0.00001, n_epoch = 10000, early_stop = 20, thresh = 0.5, loss_type = "balanced"):
        self.out_path = out_path
        self.tra_batch_size = tra_batch_size
        self.val_batch_size = val_batch_size
        self.features = features
        self.lr = lr
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.thresh = thresh
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #def pre_training(self, train_data_sets, val_data_sets):
    def pre_training(self, train_data_sets, val_data_sets, w2v_model):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)
       
        #tra_data_all = pv_data_sets(train_data_sets)
        tra_data_all = pv_data_sets(train_data_sets, w2v_model)
        balanced_sampler_train = BinarySampler(train_data_sets["label"].values.tolist(), self.tra_batch_size)
        train_loader = DataLoader(dataset = tra_data_all, batch_sampler = balanced_sampler_train)
        #train_loader = DataLoader(dataset = tra_data_all, batch_size = self.tra_batch_size, shuffle=True)

        #val_data_all = pv_data_sets(val_data_sets)
        val_data_all = pv_data_sets(val_data_sets, w2v_model)
        val_loader = DataLoader(dataset = val_data_all, batch_size = self.val_batch_size, shuffle=True)
        
        #net = BERT(n_layers = 6, d_model = self.features, n_heads = 6, d_dim = 100, d_ff = 400).cuda()
        #net = BERT(n_layers = 6, d_model = self.features, n_heads = 6, d_dim = 100, d_ff = 400, lstm_hidden_size = 128).cuda()
        #net = Gru(features = self.features, gru_hidden_size = 128).cuda()
        #net = Lstm(features = self.features, lstm_hidden_size = 128).cuda()
        #net = bGRU(features = self.features, gru_hidden_size = 128).cuda()
        net = bLSTM(features = self.features, lstm_hidden_size = 128).cuda()
        #net = CNN_LSTM(features = self.features, lstm_hidden_size = 128).cuda()        
        #net = CNN(features = self.features, time_size = 41 - int(self.features/4) + 1).cuda()
        opt = optim.Adam(params = net.parameters(), lr = self.lr)
        
        if(self.loss_type == "balanced"):
            criterion = nn.BCELoss()
            
        min_loss = 1000
        early_stop_count = 0
        with open(self.out_path + "/deep_HV_result.txt", 'w') as f:
            print(self.out_path, file = f, flush=True)
            print("The number of training data:" + str(len(train_data_sets)), file = f, flush=True)
            print("The number of validation data:" + str(len(val_data_sets)), file = f, flush=True)
            
            for epoch in range(self.n_epoch):
                train_losses, val_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
                
                print("epoch_" + str(epoch + 1) + "=====================", file = f, flush=True) 
                print("train...", file = f, flush=True)
                net.train()
                
                for i, (emb_mat, label) in enumerate(train_loader):
                    opt.zero_grad()
                    outputs = net(emb_mat)
                    
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.99, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    loss.backward()
                    opt.step()
                    
                    train_losses.append(float(loss.item()))
                    train_probs.extend(outputs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    train_labels.extend(label.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())

                train_thresh = 0.5
                print("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), file = f, flush=True)
                print("val_threshold:: value: %f, epoch: %d" % (train_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](train_labels, train_probs, thresh = train_thresh)
                    else:
                        metrics = metrics_dict[key](train_labels, train_probs)
                    print("train_" + key + ": " + str(metrics), file = f, flush=True)
                    
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh = train_thresh)
                print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                print("validation...", file = f, flush=True)
                net.eval()
                for i, (emb_mat, label) in enumerate(val_loader):
                    with torch.no_grad():
                        outputs = net(emb_mat)
                        
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.99, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    if(np.isnan(loss.item()) == False):
                        val_losses.append(float(loss.item()))
                        
                    val_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    val_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
                val_thresh = 0.5
                #precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
                #fscore = (2 * precision * recall) / (precision + recall)
                #val_thresh = thresholds[argmax(fscore)]
                loss_epoch = sum(val_losses) / len(val_losses)
                print("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), file = f, flush=True)
                print("validation_thresh:: value: %f, epoch: %d" % (val_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](val_labels, val_probs, thresh = val_thresh)
                    else:
                        metrics = metrics_dict[key](val_labels, val_probs)
                    print("validation_" + key + ": " + str(metrics), file = f, flush=True)
                
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_probs, thresh = val_thresh)
                print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                if loss_epoch < min_loss:
                    early_stop_count = 0
                    min_loss = loss_epoch
                    os.makedirs(self.out_path + "/data_model", exist_ok=True)
                    os.chdir(self.out_path + "/data_model")
                    torch.save(net.state_dict(), "deep_model")
                    #final_thresh = val_thresh
                    final_thresh = 0.5
                    final_val_probs = val_probs
                    final_val_labels = val_labels
                    final_train_probs = train_probs
                    final_train_labels = train_labels
                    
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop:
                        print('Traning can not improve from epoch {}\tBest loss: {}'.format(epoch + 1 - self.early_stop, min_loss), file = f, flush=True)
                        break
                    
            print(final_thresh, file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = final_thresh)
                    val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = final_thresh)
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print("train_" + key + ": " + str(train_metrics), file = f, flush=True)
                print("test_" + key + ": " + str(val_metrics), file = f, flush=True)
        
        
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--path', help='Path')
parser.add_argument('-o', '--outpath', help='Path')
parser.add_argument('-l', '--losstype', help='Path', default = "balanced", choices=["balanced", "imbalanced"])
parser.add_argument('-m', '--model_path', help='Path')

path = parser.parse_args().path
out_path = parser.parse_args().outpath
loss_type = parser.parse_args().losstype
model_path = parser.parse_args().model_path

w2v_model = word2vec.Word2Vec.load(model_path)

for i in range(1, 6):
    train_dataset = file_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv" ,index_col = None)
    val_dataset = file_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv" ,index_col = None)

    net = burt_process(out_path + "/" + str(i), loss_type = loss_type)
    net.pre_training(train_dataset, val_dataset, w2v_model)






























