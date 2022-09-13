# Copyright (c) Rahil Gholamipoor

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils import data
import torch.nn.functional as F



def load_data(path):
    with open(path, 'rb') as data:
        data_set = pickle.load(data)      
    return data_set 



class RawDataset(data.Dataset):
    def __init__(self, dataset, window_size, num_crops):
        self.dataset  = dataset 
        self.window_size = window_size 
        self.num_crops = num_crops
        
    def __len__(self):
        return len(self.dataset)    
    
    def resize(self, seq):
        index = np.random.randint(len(seq) - self.window_size + 1)
        return np.expand_dims(seq[index:index+self.window_size], axis=0)
    
    def __getitem__(self, index):
        pat_data = self.dataset[index]
        seq_len = len(pat_data[0])
        id_pat = pat_data[1] 
        # we randomly sample windows of size 'window_size' as many as 'num_crop' from the same hour
        croped_lst = [self.resize(pat_data[0]) for i in range(self.num_crops)]
        return croped_lst, pat_data[1], pat_data[2], pat_data[3], pat_data[4]
    

    
def get_data(x):
    bs, _, l = x.size()

    HR = x[:,0,:]   
    HR_q = x[:,1,:]
    HR_q_onehot =  F.one_hot(HR_q.view(1,-1), 101).view(HR_q.size(0), HR_q.size(1), 101)
    HR = torch.mul(HR.unsqueeze(2).float(), HR_q_onehot.float())
    HR = HR.permute(0,2,1)   

    SPo2 = x[:,2,:]   
    SPo2_q = x[:,3,:] 
    SPo2_q_onehot = F.one_hot(SPo2_q.reshape(1,-1), 101).reshape(SPo2_q.size(0), SPo2_q.size(1), 101)
    SPo2 = torch.mul(SPo2.unsqueeze(2).float() , SPo2_q_onehot.float())
    SPo2 = SPo2.permute(0,2,1)

    B_p = x[:,4,:].float().unsqueeze(1)
    activity_value = x[:,6,:].float().unsqueeze(1)
    B_w = x[:,9,:].float().unsqueeze(1)

    HRV = x[:,10,:]  
    HRV_q = x[:,11,:]
    HRV_q_onehot = F.one_hot(HRV_q.view(1,-1), 101).view(HRV_q.size(0), HRV_q.size(1), 101)
    HRV = torch.mul(HRV.unsqueeze(2).float(), HRV_q_onehot.float())
    HRV = HRV.permute(0,2,1)

    RR = x[:,12,:]
    RR_q = x[:,13,:]
    RR_q_onehot = F.one_hot(RR_q.view(1,-1), 101).view(RR_q.size(0), RR_q.size(1), 101)
    RR = torch.mul(RR.unsqueeze(2).float() , RR_q_onehot.float())
    RR = RR.permute(0,2,1)

    Energy = x[:,14,:] 
    Energy_q = x[:,15,:]
    Energy_q_onehot = F.one_hot(Energy_q.view(1,-1), 101).view(Energy_q.size(0), Energy_q.size(1), 101)
    Energy = torch.mul(Energy.unsqueeze(2).float() , Energy_q_onehot.float())
    Energy = Energy.permute(0,2,1)
    
    local_temp_1 = ((x[:,18,:].float())/100).unsqueeze(1)
    local_temp_2 = ((x[:,19,:].float())/100).unsqueeze(1)

    object_temp_1 = ((x[:,20,:].float())/100).unsqueeze(1)
    object_temp_2 = ((x[:,21,:].float())/10).unsqueeze(1)

    data = torch.cat((HR, SPo2, B_p, activity_value, B_w, HRV, RR, Energy, local_temp_1, local_temp_2,
                     object_temp_1, object_temp_2), dim=1)
    return data 



def h_preparation(loader, model, device):
    with torch.no_grad():
        h_lst = []
        label_lst = []
        s_lst = []
        e_lst = []
        
        for i, ((x, label, _, time_s, time_e)) in enumerate(loader):
            x = torch.cat(x, dim=1)
            bs, num_crop, L, nf = x.size() # batch size, num. of crops, length of input sequence, number of features
            x = x.view(bs * num_crop, L, nf)
            x = (x.permute(0,2,1)).to(device)
            
            h = model(get_data(x))     
            h_lst.append(h) 
            
            label_lst.append(label)
            s_lst.append(time_s)
            e_lst.append(time_e)
            
        h_total = torch.cat(h_lst, dim=0)
        label_total = torch.cat(label_lst, dim=0)
        s_tensor = torch.cat(s_lst, dim=0)
        e_tensor = torch.cat(e_lst, dim=0)
        
        return h_total, label_total, s_tensor, e_tensor

    
    
def score_ps(h_query, h_ref, label_query, label_ref, start, end, num_crops):
    '''
     Reference set is the collection of all patient specific hours from training set
    '''
    cos_score_lst = []
    cos_norm_lst = []
    label_lst = []
    start_lst = []
    end_lst = []
    label_query_repeat = label_query.repeat_interleave(num_crops)
    label_ref = label_ref.repeat_interleave(num_crops)
    with torch.no_grad():
        norm_query = h_query.norm(dim=1).view(-1,1)
        h_query = F.normalize(h_query, dim=1 , p=2)
        h_ref = F.normalize(h_ref, dim=1 , p=2)
        unique_query = torch.unique(label_query)
        for i in unique_query:
            if i.item() in label_ref:
                h_ref1 = h_ref[label_ref==i.item()]
                h_query1 = h_query[label_query_repeat==i.item()]
                norm_query1 = norm_query[label_query_repeat==i.item()]
                scores = torch.matmul(h_query1, h_ref1.transpose(1,0))   #[query_bs, ref_bs]
                cos_score, indices_best = scores.topk(1, dim=1, largest=True, sorted=True) 
                cos_score_lst.append(cos_score.squeeze())  
                cos_norm_lst.append(cos_score * norm_query1)
                label_lst.append(label_query[label_query==i.item()])
                start_lst.append(start[label_query==i.item()])
                end_lst.append(end[label_query==i.item()])

        lbl_updated = torch.cat(label_lst, dim=0).squeeze()
        start_updated = torch.cat(start_lst, dim=0).squeeze()
        end_updated = torch.cat(end_lst, dim=0).squeeze()
        
        return torch.cat(cos_score_lst, dim=0), lbl_updated, start_updated, end_updated    


    
def calculate_aucroc(s_1, s_2): 
    with torch.no_grad():   
        l1 = torch.zeros(s_1.size(0))
        l2 = torch.ones(s_2.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
        scores = torch.cat((s_1, s_2), dim=0).cpu().numpy()
        FPR, TPR, _ = roc_curve(label, scores, pos_label=0)
        return FPR, TPR, auc(FPR, TPR)

    
def daily_scores(labels, start_, end_, score):
    lbl_unq = torch.unique(labels)
    all_scores = []
    for k in range(len(lbl_unq)):   
        start = start_[labels==lbl_unq[k]]
        end = end_[labels==lbl_unq[k]]
        days = torch.unique(start[:, 0:3], dim=0) #[year, month, day]
        for j in range(len(days)):
            # idx = year & month & day
            idx = (start[:, 0]==days[j,0])&(start[:, 1]==days[j, 1])&(start[:, 2]==days[j, 2])
            score_hourly = ((score.cpu()[labels==lbl_unq[k]])[idx])
            all_scores.append(score_hourly.mean().item())
    return torch.tensor(all_scores)    

def standard_error_of_mean(input_x, title):
    y = np.array(input_x)
    error = np.round(np.array(np.std(input_x)), 3)
    mean = np.round(np.mean(y), 3)
    print(f'{title} scores: {mean}±{error}')
    return mean, error


def find_x_best(h_query, h_ref, num_neighbour=1, T=-1):
    with torch.no_grad():
        norm_query = h_query.norm(dim=1).view(-1,1)
        h_query = h_query/norm_query
        h_ref = h_ref/h_ref.norm(dim=1).view(-1, 1)
        
        if T!=-1:
            scores = torch.exp(torch.matmul(h_query, h_ref.transpose(1,0))/T)
            print("take exp")
        else:
            scores = torch.matmul(h_query, h_ref.transpose(1,0))   #[query_bs, ref_bs]
        
        if num_neighbour!=-1:
            cos_score, indices_best = scores.topk(num_neighbour, dim=1, largest=True, sorted=True) 
            cos_score = cos_score.mean(dim=-1).view(-1, 1)
        else:
            cos_score = scores.mean(dim=1).view(-1, 1)
            
        cos_norm_score = cos_score * norm_query

        return cos_score.squeeze(), cos_norm_score.squeeze(), indices_best.squeeze()
    
    
