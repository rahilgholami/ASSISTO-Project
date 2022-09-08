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


## Utilities
from __future__ import print_function
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3,4,5"

## Libraries
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

## Torch
import torch
torch.set_num_threads(6)
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

## Custom Imports
from src.utils import *
from src.utils import SupConLoss
from src.resnet1d import ResNet1D

device = 'cuda'
#############################################################

def get_data(x):
    #HR,HR_quality,SpO2,SpO2_quality,value_classification,Value_Activity,Quality_classification,
    #steps,HRV,HRV_quality,Respiration_Rate,RR_quality

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
        #steps = x[:,8,:].float().unsqueeze(1)
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
    
#############################################################

def x_h_preparation(loader, model, device):        
    with torch.no_grad():
        h_lst = []
        for i, ((x, _, _, _, _)) in enumerate(loader):
            x = (x.permute(0,2,1)).to(device)
            h = model(get_data(x))            
            h_lst.append(h)        
        h_total = torch.cat(h_lst, dim=0)
        return h_total

def find_x_best(h_query, h_ref):
    with torch.no_grad():
        norm_query = h_query.norm(dim=1).view(-1,1)
        h_query = h_query/norm_query
        h_ref = h_ref/h_ref.norm(dim=1).view(-1, 1)
        scores = torch.matmul(h_query, h_ref.transpose(1,0))   #[query_bs, ref_bs]
        cos_score, best_idx = scores.topk(1, dim=1)  #select top k best  
        cos_norm_score = cos_score * norm_query
        return cos_score.squeeze(), cos_norm_score.squeeze(), norm_query
    
def calculate_aucroc(s_1, s_2, name): 
    with torch.no_grad():
        file = open(name+"_auroc.txt","a+")    
        l1 = torch.zeros(s_1.size(0))
        l2 = torch.ones(s_2.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
        scores = torch.cat((s_1, s_2), dim=0).cpu().numpy()
        FPR, TPR, _ = roc_curve(label, scores, pos_label=0)
        print(f"AUC: {auc(FPR, TPR)}") 
        file.write(f"AUC :{auc(FPR, TPR)} \r\n")        
        file.close() 
        return auc(FPR, TPR)    
#############################################################    

def test(args, model, exp, device, train_loader, test_loader, ood_loader):
    model.eval()
    print("in test")
    with torch.no_grad():
        h_train = x_h_preparation(train_loader, model, device)
        h_test = x_h_preparation(test_loader, model, device)
        h_ood = x_h_preparation(ood_loader, model, device)
        test_cos, test_cos_norm, test_norm = find_x_best(h_test, h_train)
        ood_cos, ood_cos_norm, ood_norm = find_x_best(h_ood, h_train)

        cos_auc = calculate_aucroc(test_cos, ood_cos, f"results/{exp}/_CosineSim")  
        cos_norm_auc = calculate_aucroc(test_cos_norm, ood_cos_norm, f"results/{exp}/_CosineSim-Norm") 
        norm_auc = calculate_aucroc(test_norm, ood_norm, f"results/{exp}/_Norm")
    return cos_auc       

#############################################################

def train(args, model, device, loader, optimizer, epoch, writer):
    model.train()
    loader = tqdm(loader)
    total_loss = 0
    #criterion = SupConLoss(temperature=args.temperature, base_temperature=args.temperature)
    
    for batch_idx, (x_1, x_2, labels) in enumerate(loader):
        optimizer.zero_grad()
        x_1 = (x_1.permute(0,2,1)).to(device) 
        x_2 = (x_2.permute(0,2,1)).to(device)
        
        bsz = labels.shape[0]
        x_all = torch.cat((get_data(x_1), get_data(x_2)), dim=0) 
        h = model(x_all)
        h = normalize(h)
        
        sim_matrix = torch.mm(h, h.t())
        loss = NT_xent(sim_matrix, temperature=args.temperature) 
        
        #bsz = labels.shape[0]
        #h1, h2 = torch.split(h, [bsz, bsz], dim=0)
        #features = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1) #[bsz, n_views, hdim]
        #loss = criterion(features, labels.to(device))
        
        loss.backward()
        optimizer.step()
        
        loader.set_description(f'Train epoch: {epoch}; loss: {loss.item():.5f}')    
        total_loss+=loss
        
    with torch.no_grad(): 
        ave_loss = total_loss/float(len(loader))
        writer.add_scalar("Loss", ave_loss.item(), global_step=epoch, walltime=0.001)  
        
#############################################################

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train_data', default=None, help='path to training data')
    parser.add_argument('--test_data', default=None, help='path to test set')
    parser.add_argument('--ood_data', default=None, help='path to ood data')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--temperature', type=int, default=0.07, help='temperature in the loss function')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=int, default=1e-3)
    parser.add_argument('--window-size', type=int, default=1000) 
    parser.add_argument('--hdim', type=int, default=128, help='dimension of latent space')
    parser.add_argument('--input-channel', type=int, default=512, help='number of input channels')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)

    model = ResNet1D(
            in_channels=args.input_channel, 
            base_filters=64,
            kernel_size=16, 
            stride=2, 
            groups=32, 
            n_block=24, 
            n_classes=args.hdim, 
            downsample_gap=6,         
            increasefilter_gap=12, 
            use_do=True)
    model = nn.DataParallel(model)
    model = model.to(device)

    ## Loading the dataset
    params = {'num_workers': 6,
              'pin_memory': False} if use_cuda else {}

    print('===> loading training dataset')
    trainset = load_data(args.train_data)
    training_set = RawDataset(trainset, args.window_size) 
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) 
    print("trainingset size:", len(training_set))
    
    training_set1 = RawDataset(trainset, args.window_size, train=False) 
    train_loader1 = data.DataLoader(training_set1, batch_size=args.batch_size, shuffle=False) 
    print("trainingset size:", len(training_set1))
    
    test_set = RawDataset(load_data(args.test_data), args.window_size, train=False) 
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print("testset size:", len(test_set))
    
    ood_set = RawDataset(load_data(args.ood_data), args.window_size, train=False) 
    ood_loader = data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False)
    print("oodset size:", len(ood_set))
    
    ## optimizer  
    optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, lr= args.lr, weight_decay=args.weight_decay, amsgrad=True)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('===> Model total parameter: {}\n'.format(model_params))
    
    exp = f'train_NXent_tem{args.temperature}_w{args.window_size}'
    if not os.path.isdir('runs'):
        os.makedirs('runs') 
    if not os.path.isdir(f'results/{exp}'):
        os.makedirs(f'results/{exp}') 
    if not os.path.isdir(f'ckpts/{exp}'):
        print("Creating the directory")
        os.makedirs(f'ckpts/{exp}') 
        
    writer = SummaryWriter(f'runs/{exp}')
    ## Start training 
    cos_auc_old = 0
    for epoch in range(args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f'ckpts/{exp}/h_net.pt')
        if epoch%10 ==0:
            cos_auc = test(args, model, exp, device, train_loader1, test_loader, ood_loader)

    writer.close()

if __name__ == '__main__':
    main()
