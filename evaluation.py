import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"

import torch
import torch.nn as nn
torch.set_num_threads(6)
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import math
import scipy
import pickle
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from src.utils_d import *
from src.resnet1d import ResNet1D

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    
    bootstrapping = False # bootstrapping the minor class
    removing_buffer = False
    num_crops = 4
    batch_size = 64
    audio_window = 2000
    t = -1
    nb = 1
    cohort = 'Out' # options = ['In', 'Out', 'Total']
    case = 'Infectious' # options = ['Infectious', 'all_SCC']
    
    if case == 'Infectious':
        column = "Regular day_infectious [1 = 'regular day', 0 = 'irregular day']"
    else:    
        column = "Regular day_all [1 = 'regular day', 0 = 'irregular day']"
        
    ############### loading datasets ###############
    if cohort == 'In':
        if case == 'Infectious':
            h_net_path = 'enc_ckpts/train_I_NXent_tem0.1_notbalanced_infect_Ts/h_net.pt'
            data_path = '/alto/shared/SCC_newdataset/InPatients/Indatasets_infection'
        else:
            h_net_path = 'enc_ckpts/train_I_NXent_tem0.1_notbalanced_allSCCs_Ts/h_net.pt'
            data_path = '/alto/shared/SCC_newdataset/InPatients/Indatasets_allSCCs'
            
    if cohort == 'Out':
        if case == 'Infectious':
            h_net_path = 'enc_ckpts/train_O_NXent_lars_tem0.1_notbalanced_infect_Ts/h_net.pt'
            data_path = '/alto/shared/SCC_newdataset/OutPatients/Outdatasets_infection'
        else:
            h_net_path = 'enc_ckpts/train_O_NXent_lars_tem0.1_notbalanced_allSCCs_Ts/h_net.pt'
            data_path = '/alto/shared/SCC_newdataset/OutPatients/Outdatasets_allSCCs'          

    if cohort == 'Total':
        if case == 'Infectious':
            h_net_path = 'enc_ckpts/train_2C_supcon_tem0.1_notbalanced_infect_Ts/h_net.pt'
            data_path_out = '/alto/shared/SCC_newdataset/OutPatients/Outdatasets_infection' 
            data_path_in = '/alto/shared/SCC_newdataset/InPatients/Indatasets_infection'
        else:
            h_net_path = 'enc_ckpts/train_2C_supcon_tem0.1_notbalanced_allSCCs_Ts/h_net.pt'
            data_path_out = '/alto/shared/SCC_newdataset/OutPatients/Outdatasets_allSCCs' 
            data_path_in = '/alto/shared/SCC_newdataset/InPatients/Indatasets_allSCCs' 
    
    if cohort in {'In', 'Out'}:
        train_data = load_data(f'{data_path}/train_set.pickle')
        test_data = load_data(f'{data_path}/test_set.pickle')
        ood_data = load_data(f'{data_path}/ood_set.pickle')
    
    else:
        train_outdata = f'{data_path_out}/train_set.pickle'
        test_outdata = f'{data_path_out}/test_set.pickle'
        ood_outdata = f'{data_path_out}/ood_set.pickle'

        train_indata = f'{data_path_in}/train_set.pickle'
        test_indata = f'{data_path_in}/test_set.pickle'
        ood_indata = f'{data_path_in}/ood_set.pickle'
        
        train_data = load_data(train_outdata) + load_data(train_indata)
        test_data = load_data(test_outdata) + load_data(test_indata)
        ood_data = load_data(ood_outdata) + load_data(ood_indata)
        
    ### removing buffers from SCC
    if removing_buffer:
        ood_lst = []
        specific_keys = remove_buffer(column)
        for i in range(len(ood_data)):
            key = str([(ood_data[i][-2][0], ood_data[i][-2][1], ood_data[i][-2][2]), ood_data[i][-4]])
            if key in specific_keys:
                if ood_data[i][-4] not in {1002, 1026}:
                    ood_lst.append(ood_data[i])
        ###             
        test_lst = []
        for i in range(len(test_data)):
            if test_data[i][-4] not in {1002, 1026}:
                    test_lst.append(test_data[i])

        ### checking out the number of samples of patients with IDs {1002 and 1026} that we already excluded       
        l_test = len(test_lst)
        l_ood = len(ood_lst)

        test_label = []
        for i in range(l_test):
            test_label.append(test_lst[i][-4])
        print("test: ", sorted(Counter(np.array(test_label)).items()))

        ood_label = []
        for i in range(l_ood):
            ood_label.append(ood_lst[i][-4])   
        print("ood: ", sorted(Counter(np.array(ood_label)).items()))

        ### bootstrapping the minor class
        if bootstrapping:
            if l_test<l_ood:
                print("resample the test set") 
                test_bt, _ = resample(test_lst, test_label, n_samples=l_ood, replace=True)
                test_lst = test_bt

            else:
                print("resample the ood set")
                ood_bt, _ = resample(ood_lst, ood_label, n_samples=l_test, replace=True)
                ood_lst = ood_bt

    ### sampling 4 random crops
    training_set = RawDataset(train_data, audio_window, num_crops=num_crops)
    test_set = RawDataset(test_data, audio_window, num_crops=num_crops) 
    ood_set = RawDataset(ood_data, audio_window, num_crops=num_crops) 

    print("# training hours:", len(training_set))
    print("# test hours:", len(test_set))
    print("# ood hours:", len(ood_set))


    train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=False) 
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    ood_loader = data.DataLoader(ood_set, batch_size=batch_size, shuffle=True)
    
    ############### loading trained model ###############
    h_net_ckpt = torch.load(h_net_path)
    h_net = ResNet1D(
            in_channels=512, 
            base_filters=64, 
            kernel_size=16, 
            stride=2, 
            groups=32, 
            n_block=24, 
            n_classes=128, 
            downsample_gap=6,         
            increasefilter_gap=12, 
            use_do=True)

    h_net.load_state_dict(h_net_ckpt['model'])
    h_net = nn.DataParallel(h_net)
    h_net = h_net.to(device)
    h_net.eval() 
    
    ############### computing scores ###############
    with torch.no_grad():
        h_train, label_train, start_train, end_train = h_preparation(train_loader, h_net, device)
        h_test, label_test, start_test, end_test = h_preparation(test_loader, h_net, device)
        h_ood, label_ood, start_ood, end_ood = h_preparation(ood_loader, h_net, device)
        
        if cohort in {'In', 'Out'}:
            test_cos, test_cos_norm = find_x_best(h_test, h_train, num_neighbour=nb, T=t)
            ood_cos, ood_cos_norm = find_x_best(h_ood, h_train, num_neighbour=nb, T=t)
        
        else:
            train_h_ds = TensorDataset(h_train.cpu())
            train_h_loader = DataLoader(train_h_ds, batch_size=5000, 
                                       shuffle=False, drop_last=False)        

            test_h_ds = TensorDataset(h_test.cpu())
            test_h_loader = DataLoader(test_h_ds, batch_size=5000, 
                                       shuffle=False, drop_last=False)

            ood_h_ds = TensorDataset(h_ood.cpu())
            ood_h_loader = DataLoader(ood_h_ds, batch_size=5000,
                                     shuffle=False, drop_last=False)

            test_cos, test_cos_norm = calculate_similarity(test_h_loader, train_h_loader, device, num_neighbour=nb, T=t)
            ood_cos, ood_cos_norm = calculate_similarity(ood_h_loader, train_h_loader, device, num_neighbour=nb, T=t)
        
        test_cos = (test_cos.view(len(test_cos)//num_crops, num_crops)).mean(-1)
        ood_cos = ood_cos.view(len(ood_cos)//num_crops, num_crops).mean(-1)

        cos_auc = calculate_aucroc(test_cos, ood_cos)
        
        print('------------------------------------ hourly ------------------------------------')
        print(f'{cohort} cohort,{case}, AUROC:{np.round(cos_auc, 4)}')
        print("# test scores:", len(test_cos))
        print("# ood scores:",len(ood_cos))
        print("95% Confidence Interval of AUROC:", AUROC_95CI(test_cos, ood_cos, positive=1))
        standard_error_of_mean(1 - test_cos.cpu().numpy(), 'regular')
        standard_error_of_mean(1 - ood_cos.cpu().numpy(), 'SCC')
        print("unpaired t-test:", scipy.stats.ttest_ind(test_cos.cpu(), ood_cos.cpu(), equal_var=False))
        print("(Sensitivity95, Specificity):", sen_spec95(test_cos, ood_cos))
        fpr_h, tpr_h, _ = cutoff_youdens(test_cos, ood_cos)
        print(f'Optimal sensitivity/speceficity :{np.round(tpr_h[0],4)}/{np.round(1-fpr_h[0], 4)}')
        
        print('------------------------------------ daily ------------------------------------')
        daily_scores_ood = daily_scores(label_ood, start_ood, end_ood, ood_cos)
        daily_scores_test = daily_scores(label_test, start_test, end_test, test_cos)
        cos_auc = calculate_aucroc(daily_scores_test, daily_scores_ood)
        print(f'{cohort},{case}, AUROC:{np.round(cos_auc, 4)}')
        print("# test scores:", len(daily_scores_test))
        print("# ood scores:", len(daily_scores_ood))
        print("95% Confidence Interval of AUROC:", AUROC_95CI(daily_scores_test, daily_scores_ood, positive=1))
        standard_error_of_mean(1 - daily_scores_test.cpu().numpy(), 'regular')
        standard_error_of_mean(1 - daily_scores_ood.cpu().numpy(), 'SCC')
        print("unpaired t-test:", scipy.stats.ttest_ind(daily_scores_test.cpu(), daily_scores_ood.cpu(), equal_var=False))
        print("(Sensitivity95, Specificity):", sen_spec95(daily_scores_test, daily_scores_ood))
        fpr_d, tpr_d, _ = cutoff_youdens(daily_scores_test, daily_scores_ood)
        print(f'Optimal sensitivity/speceficity :{np.round(tpr_d[0],4)}/{np.round(1-fpr_d[0], 4)}')
        
        
        
