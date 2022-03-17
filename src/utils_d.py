import torch
from torch.utils import data
import torch.nn.functional as F
import math
import pickle
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from openpyxl.utils import get_column_letter, column_index_from_string

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
        h1_lst = []
        h2_lst = []
        h3_lst = []
        label_lst = []
        s_lst = []
        e_lst = []
        
        for i, ((x, label, _, time_s, time_e)) in enumerate(loader):
            print("excluding some patients with ids 1002 and 1026")
            idx =  (label==1002) + (label==1026)
            x = torch.cat(x, dim=1)
            x = x[~idx, :, :, :]
            bs, num_crop, L, nf = x.size()
            x = x.view(bs*num_crop, L, nf)
            x = (x.permute(0,2,1)).to(device)
            h = model(get_data(x))     
            h_lst.append(h) 
            
            label_lst.append(label[~idx])
            s_lst.append(time_s[~idx])
            e_lst.append(time_e[~idx])
            
        h_total = torch.cat(h_lst, dim=0)
        label_total = torch.cat(label_lst, dim=0)
        s_tensor = torch.cat(s_lst, dim=0)
        e_tensor = torch.cat(e_lst, dim=0)
        
        return h_total, label_total, s_tensor, e_tensor

def find_x_best(h_query, h_ref, num_neighbour=1, T=-1):
    with torch.no_grad():
        norm_query = h_query.norm(dim=1).view(-1,1)
        h_query = h_query/norm_query
        h_ref = h_ref/h_ref.norm(dim=1).view(-1, 1)
        
        if T!=-1:
            scores = torch.exp(torch.matmul(h_query, h_ref.transpose(1,0))/T)
        else:
            scores = torch.matmul(h_query, h_ref.transpose(1,0))   #[query_bs, ref_bs]
        
        if num_neighbour!=-1:
            cos_score, _ = scores.topk(num_neighbour, dim=1, largest=True, sorted=True) 
            print(cos_score.size())
            cos_score = cos_score.mean(dim=-1).view(-1, 1)
        else:
            cos_score = scores.mean(dim=1).view(-1, 1)
            
        cos_norm_score = cos_score * norm_query
        return cos_score.squeeze(), cos_norm_score.squeeze()

    


def calculate_aucroc(s_1, s_2): 
    with torch.no_grad():   
        l1 = torch.zeros(s_1.size(0))
        l2 = torch.ones(s_2.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
        scores = torch.cat((s_1, s_2), dim=0).cpu().numpy()
        FPR, TPR, _ = roc_curve(label, scores, pos_label=0)
        return auc(FPR, TPR)
    
    
def dist_plot(score_ind, score_ood, title):
    with torch.no_grad():
        plt.figure()
        num_bins = 100  
        plt.hist(score_ind.cpu().tolist(), bins=num_bins, alpha=0.8)
        plt.hist(score_ood.cpu().tolist(), bins=num_bins, alpha=0.8)                   
        plt.xlabel(title, fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend(['test', 'ood-test'], loc='upper right')
        plt.show()
        
        
class RawDataset(data.Dataset):
    def __init__(self, dataset, audio_window, num_crops, downsample=False):
        self.dataset  = dataset 
        self.audio_window = audio_window 
        self.num_crops = num_crops
        self.downsample = downsample
    def __len__(self):
        return len(self.dataset)    
    
    def resize(self, seq):
        index = np.random.randint(len(seq) - self.audio_window + 1)
        return np.expand_dims(seq[index:index+self.audio_window], axis=0)
    def __getitem__(self, index):
        pat_data = self.dataset[index]
        seq_len = len(pat_data[0])
        id_pat = pat_data[1] 
        croped_lst = [self.resize(pat_data[0]) for i in range(self.num_crops)]
        return croped_lst, pat_data[1], pat_data[2], pat_data[3], pat_data[4]
        
        
def load_data(path):
    with open(path, 'rb') as data:
        data_set = pickle.load(data)      
    return data_set  


def calculate_similarity(query_loader, ref_loader, device, num_neighbour, T):
    with torch.no_grad():
        cos_score_list = []
        cos_norm_score_list = []
        for i , (f_query_lst) in enumerate(query_loader):
            f_query = f_query_lst[0] 
            f_query = f_query.to(device)
            print(f_query.size())
            norm_query = f_query.norm(dim=1).view(-1,1)
            f_query = F.normalize(f_query, dim=1 , p=2)
            for j , (f_ref_lst) in enumerate(ref_loader):
                f_ref = f_ref_lst[0]
                f_ref = f_ref.to(device)
                f_ref = F.normalize(f_ref, dim=1 , p=2)
                if T != -1:
                    tmp_scores = torch.exp(torch.matmul(f_query, f_ref.transpose(1,0))/T)   
                else:
                    tmp_scores = torch.matmul(f_query, f_ref.transpose(1,0))   
                if j == 0 :
                    scores = tmp_scores.clone()
                else:
                    scores = torch.cat((scores, tmp_scores), dim=1)
                
            if num_neighbour != -1 :            
                cos_score, _ = scores.topk(num_neighbour, dim=1, largest=True, sorted=True) 
                cos_score = cos_score.mean(dim=1).view(-1, 1)               
            else:
                cos_score = scores.mean(dim=1).view(-1, 1)     
                
            cos_norm_score = cos_score * norm_query
            cos_score_list.append(cos_score)
            cos_norm_score_list.append(cos_norm_score)
            
        cos_score = torch.cat(cos_score_list, dim=0)
        cos_norm_score = torch.cat(cos_norm_score_list, dim=0)
        return  cos_score.squeeze(), cos_norm_score.squeeze()
    
    

def quantile(x):
    Q1 = np.quantile(x, 0.25)
    median = np.quantile(x, 0.50)
    Q3 = np.quantile(x, 0.75)
    IQR = Q3-Q1
    return np.round(median, 3), np.round(IQR, 3)


def standard_error_of_mean(input_x, title):
    y = np.array(input_x)
    e = np.array(np.std(input_x))
    mean = np.mean(y)
    print(f'{title} scores: {np.round(mean, 3)}±{np.round(e, 3)}')
    return 
    

def sen_spec95(s_1, s_2):
    l1 = torch.zeros(s_1.size(0))
    l2 = torch.ones(s_2.size(0))
    label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
    scores = torch.cat((1/s_1, 1/s_2), dim=0).cpu().numpy()
    FPR, TPR, thresholds = roc_curve(label, scores, pos_label=1)
    
    a = TPR[(0.95<=TPR)] #TPR:sensitivity
    sen = a[0]
    b = (FPR[(0.95<=TPR)]) #FPR:1-specificity
    spec = (1-b)[0]
    return np.round(sen, 4), np.round(spec, 4)




def AUROC_95CI(test, ood, positive=1):
    l1 = torch.zeros(test.size(0))
    l2 = torch.ones(ood.size(0))
    y_true = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
    y_score = torch.cat((test, ood), dim=0).cpu().numpy()
    FPR, TPR, _ = roc_curve(y_true, y_score, pos_label=0)
    AUC = auc(FPR, TPR)

    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (np.round(lower, 2), np.round(upper, 2))


def make_dict(label, time_s, time_e, score):
    lbl_unq = torch.unique(label)
    Dict_scores = {}
    with torch.no_grad():
        for i in range(len(lbl_unq)):
            start = time_s[label==lbl_unq[i]]
            end = time_e[label==lbl_unq[i]]
            days = torch.unique(start[:, 0:3], dim=0) #[year,month,day]
            for j in range(len(days)):
                # idx = year & month & day
                idx = (start[:, 0]==days[j,0])&(start[:, 1]==days[j,1])&(start[:, 2]==days[j,2])
                vec_score = torch.zeros([1, 24])
                indx_score = start[idx][:, 3]
                vec_score[:, indx_score] = (score.cpu())[label==lbl_unq[i]][idx]
                dict_key = str([(days[j,0].item(), days[j,1].item(), days[j,2].item()), lbl_unq[i].item()])
                Dict_scores[dict_key]= vec_score
    return Dict_scores


def dict_daily_scores(label, time_s, time_e, score):
    lbl_unq = torch.unique(label)
    Dict_scores = {}
    with torch.no_grad():
        for i in range(len(lbl_unq)):
            start = time_s[label==lbl_unq[i]]
            end = time_e[label==lbl_unq[i]]
            days = torch.unique(start[:, 0:3], dim=0) #[year,month,day]
            for j in range(len(days)):
                # idx = year & month & day
                idx = (start[:, 0]==days[j,0])&(start[:, 1]==days[j,1])&(start[:, 2]==days[j,2])
                indx_score = start[idx][:, 3]
                dict_key = str([(days[j,0].item(), days[j,1].item(), days[j,2].item()), lbl_unq[i].item()])
                Dict_scores[dict_key] = ((score.cpu())[label==lbl_unq[i]][idx]).mean().item()
    return Dict_scores


def daily_scores(labels, start_, end_, score):
    lbl_unq = torch.unique(labels)
    all_scores = []
    for k in range(len(lbl_unq)):   
        start = start_[labels==lbl_unq[k]]
        end = end_[labels==lbl_unq[k]]
        days = torch.unique(start[:, 0:3], dim=0) #[year,month,day]
        for j in range(len(days)):
            # idx = year & month & day
            idx = (start[:, 0]==days[j,0])&(start[:, 1]==days[j,1])&(start[:, 2]==days[j,2])
            score_hourly = ((score.cpu()[labels==lbl_unq[k]])[idx])
            all_scores.append(score_hourly.mean().item())
    return torch.tensor(all_scores)

def export_scores_to_excel(currentSheet, letter_range, Dict_scores):
    for row in range(2, 1914 + 1):
        cell_ID = "{}{}".format("A", row)
        cell_date = "{}{}".format("C", row)
        year = (currentSheet[cell_date].value).year
        month = (currentSheet[cell_date].value).month
        day = (currentSheet[cell_date].value).day
        cell_name = str([(year, month, day), currentSheet[cell_ID].value])

        if cell_name in Dict_scores.keys():
            value = Dict_scores[cell_name]
            print(cell_name)
            i = 0
            for column in range(column_index_from_string(letter_range[0]), column_index_from_string(letter_range[-1])+1):
                currentSheet["{}{}".format(get_column_letter(column), row)] = round(value[0][i].item(), 3)
                i +=1
    return currentSheet

def save_scores(s_1, s_2, path):
    with torch.no_grad():   
        l1 = torch.zeros(s_1.size(0))
        l2 = torch.ones(s_2.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
        scores = torch.cat((s_1, s_2), dim=0).cpu().numpy()
        FPR, TPR, thresholds = roc_curve(label, scores, pos_label=0)
        roc_auc = auc(FPR, TPR)
        tensors = {'FPR': FPR, 'TPR': TPR, 'scores_test':s_1, 'scores_ood':s_2}
        if path is not None:
            torch.save(tensors, path)
    return FPR, TPR, roc_auc 


def plot_roc(FPR, TPR, roc_auc):
    plt.figure(figsize=(7,5))
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=12, fontname='Arial')
    plt.legend(loc="lower right")
    plt.show()

def cutoff_youdens(s_1, s_2):
    '''
    Youden Index is a measure for evaluating the biomarker effectiveness
    J = {Se - (1 - Sp)} = {tpr - fpr}
    '''
    l1 = torch.zeros(s_1.size(0))
    l2 = torch.ones(s_2.size(0))
    label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu().numpy()
    scores = torch.cat((1/s_1, 1/s_2), dim=0).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] 
    return fpr[thresholds==optimal_threshold], tpr[thresholds==optimal_threshold], optimal_threshold


def remove_buffer(input_column):
    xls = pd.ExcelFile(r"data_preparation/220311_H2017A_Annotation_Rahil.xlsx") 
    data_sheet = xls.parse(0)
    data_sheet = data_sheet[:1913]

    # filter necessary columns 
    data_sheet = data_sheet.filter(items=["Pat-ID", "Date", input_column])
    Dict_SCC_infectious_performance = {}
    for row in range(0, 1913): 
        cell_ID = "Pat-ID"
        cell_date = "Date"
        cell_SCC_infectious_performance = input_column
        year = (data_sheet[cell_date][row]).year
        month = (data_sheet[cell_date][row]).month
        day = (data_sheet[cell_date][row]).day
        key = str([(year, month, day), data_sheet[cell_ID][row]])
        value = data_sheet[cell_SCC_infectious_performance][row]
        if not math.isnan(value):
            Dict_SCC_infectious_performance[key]=int(value)
    specific_keys = [k for k,v in Dict_SCC_infectious_performance.items() if v == 0]        
    return specific_keys
    
### 95% confidence interval sensitivity and specificity
# from sklearn.metrics import confusion_matrix
# CM=confusion_matrix(label.cpu(), pred.cpu())
# TN = CM[0][0] #Healthy people correctly identified as healthy
# FN = CM[1][0] #Sick people incorrectly identified as healthy
# TP = CM[1][1] #Sick people correctly identified as sick
# FP = CM[0][1] #Healthy people incorrectly identified as sick
# sensitivity  = TP / (TP+FN)
# specificity  = TN / (TN+FP)
# print('sensitivity:', sensitivity)
# print('specificity:', specificity)
 
# Se_sen = 1.96 * np.sqrt((sensitivity*(1-sensitivity))/(TP+FN))    
# Se_spe = 1.96 * np.sqrt((specificity*(1-specificity))/(TN+FP)) 