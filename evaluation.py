import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4,5,6"

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
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--ps', default=False, type=str, help='Patient Specific')
    parser.add_argument('--save_path', type=str, default='', help='where to save results')
    parser.add_argument('--save_results', type=str, default = False)
    parser.add_argument('--model_path', type=str, default='', help='path to checkpoint')
    parser.add_argument('--data_path', type=str, default='', help='path to data')
    parser.add_argument('--bootstrapping', default=True, type=str, help='Bootstrapping the minor class')
    parser.add_argument('--num_crops', default=6, type=int, help='Number of crops per sample')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--window_size', default=1000, type=int, help='Length of input sequence')
    parser.add_argument('--t', default=-1, type=int, help='Temperature')
    parser.add_argument('--nb', default=1, type=int, help='Number of neighbours')
    parser.add_argument('--cohort', default='Outpatient', type=str, choices=["Inpatient", "Outpatient", "Total"])
    parser.add_argument('--case', default='allSCCs', type=str, choices=["Infectious", "allSCCs"])
    args = parser.parse_args()
        
    ############################################# loading datasets #############################################
    
    assert os.path.exists(args.data_path)
    train_data = load_data(f'{args.data_path}/train_set.pickle') # training regular hours
    test_data = load_data(f'{args.data_path}/test_set.pickle') # test regular hours
    ood_data = load_data(f'{args.data_path}/ood_set.pickle') # out-of-distribution irregular hours

    ##############################  bootstrapping the minority class ########################################### 
    
    if args.bootstrapping:
        l_test = len(test_data)
        l_ood = len(ood_data)
        
        test_label = [test_data[i][-4] for i in range(l_test)]
        ood_label = [ood_data[i][-4] for i in range(l_ood)]
        
        if l_test<l_ood:
            print("Resampling the regular test set\n") 
            test_bt, _ = resample(test_data, test_label, n_samples=l_ood, replace=True)
            test_data = test_bt

        else:
            print("Resampling the irregular test set\n")
            ood_bt, _ = resample(ood_data, ood_label, n_samples=l_test, replace=True)
            ood_data = ood_bt

    print("train  dataset: ", sorted(Counter(np.array([train_data[i][-4] for i in range(len(train_data))])).items()))  
    print("test dataset: ", sorted(Counter(np.array([test_data[i][-4] for i in range(len(test_data))])).items()))
    print("ood dataset: ", sorted(Counter(np.array([ood_data[i][-4] for i in range(len(ood_data))])).items()))
        
    ##################################### data preparation #######################################################
    
    training_set = RawDataset(train_data, args.window_size, num_crops=args.num_crops) 
    test_set = RawDataset(test_data, args.window_size, num_crops=args.num_crops) 
    ood_set = RawDataset(ood_data, args.window_size, num_crops=args.num_crops) 

    print("# training hours:", len(training_set))
    print("# test hours:", len(test_set))
    print("# ood hours:", len(ood_set))


    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False) 
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    ood_loader = data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False)
    
    ############################## loading checkpoint ##########################################################
    
    assert os.path.exists(args.model_path)
    h_net_ckpt = torch.load(args.model_path)
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
    
    ############################## computing scores ##########################################################
    
    with torch.no_grad():
        
        h_train, label_train, start_train, end_train = h_preparation(train_loader, h_net, device)
        h_test, label_test, start_test, end_test = h_preparation(test_loader, h_net, device)
        h_ood, label_ood, start_ood, end_ood = h_preparation(ood_loader, h_net, device)
      
        
        if args.ps:
            print("-----------------------------------------------------\n")
            print("Evaluation strategy is patient specific \n")
            print("Reference set is the collection of all patient specific hours from training set \n")
           
            print("Best matches of regular hours to reference set build the null distriubtion \n")
            test_cos, label_test, start_test, end_test = score_ps(h_test, h_train, label_test, label_train, start_test, end_test, args.num_crops)
            test_cos = (test_cos.view(len(test_cos)//args.num_crops, args.num_crops)).mean(-1)
            
            print("Best matches of ood hours to reference set form the alternative distribution \n")
            ood_cos, label_ood, start_ood, end_ood = score_ps(h_ood, h_train, label_ood, label_train, start_ood, end_ood, args.num_crops) 
                                                              
            ood_cos = (ood_cos.view(len(ood_cos)//args.num_crops, args.num_crops)).mean(-1)
            
        else:
            print("-----------------------------------------------------\n")
            print("Evaluation strategy is non-patient specific \n")
            print("Reference set is the whole training set \n")
            
            print("Best matches of regular hours to reference set build the null distriubtion \n")
            test_cos, _, _ = find_x_best(h_test, h_train, num_neighbour=args.nb, T=args.t)    
            test_cos = (test_cos.view(len(test_cos)//args.num_crops, args.num_crops)).mean(-1)
            
            print("Best matches of ood hours to reference set form the alternative distribution \n")
            ood_cos, _, _ = find_x_best(h_ood, h_train, num_neighbour=args.nb, T=args.t)
            ood_cos = (ood_cos.view(len(ood_cos)//args.num_crops, args.num_crops)).mean(-1)
        


        standard_error_of_mean(1 - test_cos.cpu().numpy(), 'regular samples')
        standard_error_of_mean(1 - ood_cos.cpu().numpy(), 'irregular samples')
        
        print("-----------------------------------------------------\n")
        print('If we collect scores from all patients and then form null and alternative distributions \n')
        print("%AUROC given hourly scores:", np.round(calculate_aucroc(test_cos, ood_cos)[-1] * 100, 2))
        
        # creating daily scores from hourly scores
        daily_ood = daily_scores(label_ood, start_ood, end_ood, ood_cos)
        daily_test = daily_scores(label_test, start_test, end_test, test_cos)
        print("%AUROC given daily scores:", np.round(calculate_aucroc(daily_test, daily_ood)[-1] * 100, 2))
        print("-----------------------------------------------------\n")
        
        print('If we form null and alternative distributions for each single patient and then take average \n')
        tprs = []
        tprs_d = []
        mean_fpr = np.linspace(0, 1, 100)
        auc_lst = []
        aucroc_d_lst = []
        k = 0
        
        for i in torch.unique(label_test):
            if i in torch.unique(label_ood): 
                test_id = test_cos[label_test==i.item()]
                ood_id = ood_cos[label_ood==i.item()]
                fpr, tpr, aucroc = calculate_aucroc(-ood_id, -test_id)
                auc_lst.append(aucroc)
                k = k + 1
                
                interp_tpr = np.interp(mean_fpr, fpr, tpr) # interpolation
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                
                daily_scores_ood = daily_scores(label_ood[label_ood==i.item()], start_ood[label_ood==i.item()], 
                                                end_ood[label_ood==i.item()], ood_cos[label_ood==i.item()])
                daily_scores_test = daily_scores(label_test[label_test==i.item()], start_test[label_test==i.item()],
                                                 end_test[label_test==i.item()], test_cos[label_test==i.item()])
                fpr_d, tpr_d, aucroc_d = calculate_aucroc(-daily_scores_ood, -daily_scores_test)
                aucroc_d_lst.append(aucroc_d)
                
                interp_tpr_d = np.interp(mean_fpr, fpr_d, tpr_d)
                interp_tpr_d[0] = 0.0
                tprs_d.append(interp_tpr_d)
        

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_hourly_auc = auc(mean_fpr, mean_tpr)
        
        mean_tprs_d = np.mean(tprs_d, axis=0)
        mean_tprs_d[-1] = 1.0
        mean_daily_auc = auc(mean_fpr, mean_tprs_d)
        
        
        print("%AUROC given hourly scores:", np.round(mean_hourly_auc * 100, 2))
        print("%AUROC given daily scores:", np.round(mean_daily_auc * 100, 2))

        print("-----------------------------------------------------")
        
        if args.save_results:
            assert os.path.exists(args.save_path)
            tensors = {'FPR_h': mean_fpr, 'TPR_h': mean_tpr, 'FPR_d': mean_fpr, 'TPR_d': mean_tprs_d, 'test_d':daily_test,
                  'ood_d':daily_ood, 'test_h':test_cos, 'ood_h':ood_cos}
            print("saveing ...")
            torch.save(tensors, f'{args.save_path}/{args.cohort}_{args.case}_ps_{args.ps}')
