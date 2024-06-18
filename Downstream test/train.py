# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import csv
import time
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
#from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

from data_entry import select_train_loader, select_eval_loader
from model_entry import select_model
from options import prepare_train_args
from logger import Logger
from torch_utils import load_match_dict

from list_dataset import *

CASF_2016='/dataset'
ids_CASF_2016="%s/out_id_CASF_2016_5A.npy"%(CASF_2016)
ligs_CASF_2016="%s/out_ligand_CASF_2016_5A.bin"%(CASF_2016)
prots_CASF_2016="%s/out_protein_CASF_2016_5A.bin"%(CASF_2016)


pdbids_CASF_2016 = np.load(ids_CASF_2016)
graphsl_CASF_2016= load_graphs(ligs_CASF_2016)
graphsp_CASF_2016= load_graphs(prots_CASF_2016)
graphsl_CASF_2016 = graphsl_CASF_2016[0]
graphsp_CASF_2016 = graphsp_CASF_2016[0]

CASF_2013='dataset'
ids_CASF_2013="%s/out_id_CASF_2013_5A.npy"%(CASF_2013)
ligs_CASF_2013="%s/out_ligand_CASF_2013_5A.bin"%(CASF_2013)
prots_CASF_2013="%s/out_protein_CASF_2013_5A.bin"%(CASF_2013)

pdbids_CASF_2013 = np.load(ids_CASF_2013)
graphsl_CASF_2013= load_graphs(ligs_CASF_2013)
graphsp_CASF_2013= load_graphs(prots_CASF_2013)
graphsl_CASF_2013 = graphsl_CASF_2013[0]
graphsp_CASF_2013 = graphsp_CASF_2013[0]



mae_loss_fn = torch.nn.L1Loss()
def calculate_rmse(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse
def calculate_mae(predictions, targets):
    mae = mae_loss_fn(predictions, targets)
    #rmse = torch.sqrt(mse)
    return mae

def calculate_pearson_coefficient(predictions, targets):
    # 标准化预测值和目标值
    predictions = (predictions - predictions.mean()) / predictions.std()
    targets = (targets - targets.mean()) / targets.std()

    # 计算皮尔逊相关系数
    pearson_coefficient = torch.mean(predictions * targets)

    return pearson_coefficient


def PCC_RMSE(y_pred,y_true, alpha=0.8):
    rmse=calculate_rmse(y_pred,y_true)
    pearsonR=calculate_pearson_coefficient(y_pred,y_true)
    loss = alpha *(1-pearsonR) + (1 - alpha) * rmse

    return loss

def PCC_MAE(y_pred,y_true, alpha=0.8):
    mae=calculate_mae(y_pred,y_true)
    pearsonR=calculate_pearson_coefficient(y_pred,y_true)
    loss = alpha *(1-pearsonR) + (1 - alpha) * mae

    return loss


mae_loss_fn = torch.nn.L1Loss()






class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
        self.logger = Logger(args)
        print(self.args)
        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        #self.model = select_model(args)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model=select_model(args)
        if args.load_model_path != '':
            if args.load_not_strict:
                #load_match_dict(self.model, args.load_model_path)
                pass
            else:
                print('调用预训练')
                self.model.load_state_dict(torch.load(args.load_model_path))#.state_dict())

        #self.model = torch.nn.DataParallel(self.model)分布式训练
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr
                                          , betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay
                                         )

    def train(self):
        loss_list=[]
        personR_list=[]
        for epoch in range(self.args.epochs):
            train_loss=self.train_per_epoch(epoch)
            test_loss,personR=self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model,epoch)#保存模型
            print('epoch {:d} | train loss {:.4f} | val loss {:.4f}'.format(epoch, train_loss, test_loss))
            print('PersonR {:.4f}'.format(personR[0]))
            loss_list.append([float(train_loss), float(test_loss)])
            personR_list.append(personR)
        train_loss=pd.DataFrame(data=loss_list)#数据有三列，列名分别为one,two,three
        train_loss.to_csv('loss_list.csv',encoding='gbk')
        personR_list_csv=pd.DataFrame(data=personR_list)#数据有三列，列名分别为one,two,three
        personR_list_csv.to_csv('personR_list.csv',encoding='gbk')
        #eval_loss=pd.DataFrame(data=eval_loss)#数据有三列，列名分别为one,two,three
        #train_loss.to_csv('eval_loss.csv',encoding='gbk')
        
        
            

    def train_per_epoch(self, epoch,aux_weight=0.001):
        # switch to train mode
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
        #device = torch.device('cuda:0')
        self.model.train()
     
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
             
            pdbids,bgl_0,bgp_0,bgl_1,bgp_1,label= data
            bgl_0=bgl_0.to(device)
            bgp_0=bgp_0.to(device)
            bgl_1=bgl_1.to(device)
            bgp_1=bgp_1.to(device)
            label=label.to(device)
            self.model.to(device)
            #print(bgl)
            #print(bgp)
            #print(label)
            output_2,output_3,output_0 ,output_1,regression,_,_ = self.model(bgl_0,bgp_0,bgl_1,bgp_1)#.to(device)
            #print(pred)
            #print(len(pred))
            label=label.t()
            #pred,label = self.step(data)
            #loss =self.compute_loss(pred, label).to(device)#, is_train=True)
            #print('loss：',pred.size(),label.size())
            #print('loss：',pred.size(),label.size())
            #loss_fn=torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)

            loss = torch.nn.functional.mse_loss(regression, label)
            #loss = PCC_MAE(regression, label)
            #loss = mae_loss_fn(regression, label)
            loss.backward()
            self.optimizer.step()
            
            
            #metrics = self.compute_metrics(pred, label, is_train=False)
            #loss =self.compute_loss(pred, label).to(device)
            
            
            #for key in metrics.keys():
                #self.logger.record_scalar(key, metrics[key])
            return loss

            # monitor training progress
            #if i % self.args.print_freq == 0:
                #print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss)) 

    def val_per_epoch(self, epoch):
        
        self.model.eval()
        for i, data in enumerate(self.val_loader):

            pdbids,bgl_0,bgp_0,bgl_1,bgp_1,label= data
            bgl_0=bgl_0.to(device)
            bgp_0=bgp_0.to(device)
            bgl_1=bgl_1.to(device)
            bgp_1=bgp_1.to(device)
            label=label.to(device)
            self.model.to(device)
            
            label=label.t()
            with torch.no_grad():
                output_2,output_3,output_0 ,output_1,regression,_,_= self.model(bgl_0,bgp_0,bgl_1,bgp_1)#.to(device)
            loss = torch.nn.functional.mse_loss(regression, label)
            #loss_fn=torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
            #loss = PCC_MAE(regression, label)
            #loss = mae_loss_fn(regression, label)
            #return loss
        
        
        
        
        prediction=[]
        for index,i in enumerate(pdbids_CASF_2016):
            bgl = dgl.batch([graphsl_CASF_2016[index]])
            bgp = dgl.batch([graphsp_CASF_2016[index]])
            
            output_1,output_2,GNN,PLEC,pred,_,_=self.model(bgl.to(device),bgp.to(device),bgl.to(device),bgp.to(device))
            #Input= torch.cat([output_0,output_1],axis=1)
            #Input_list = output_0.detach().numpy().tolist()
    
            #model_2.eval()
            #pred=model_2(torch.tensor(Input_list))
            prediction.append(float(pred))
        
        CASF_2016_target='/dataset/INDEX_refined_data.2016'
        contents = []
        #with open(CASF_2016_target, 'r') as f:
        #    for line in f.readlines():
        #        if line[0] != "#":
        #            splitted_elements = line.split()
        #            if len(splitted_elements) == 6:
        #                contents.append(splitted_elements[:5] + splitted_elements[6:])
        #            else:
        #                pass
        #df = pd.DataFrame(contents, columns=(
        #        'pdb_code', 'resl', 'release_year',
        #        'logKa', 'Ka'))
        #df.set_index('pdb_code',inplace = True)
        with open(CASF_2016_target, 'r') as f:
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    if len(splitted_elements) == 8:
                        contents.append(splitted_elements[:5] + splitted_elements[6:])
                    else:
                        pass
        df = pd.DataFrame(contents, columns=(
                'pdb_code', 'resolution', 'release_year',
                'logKa', 'Kd/Ki', 'reference', 'ligand_name'))
        df.set_index('pdb_code',inplace = True)

        
        
        
        dataframe = pd.DataFrame()
        dataframe['pdbids']=pdbids_CASF_2016
        dataframe['prediction']=prediction
        dataframe.set_index('pdbids',inplace = True)
        LogKas=[]
        for i in pdbids_CASF_2016:
            logKa=df.loc[i,'logKa']
            LogKas.append(logKa)
        dataframe['logKa']=LogKas
        
        from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
        from scipy import stats
        LogKas = list(map(float, LogKas))
        prediction = [round(i,2) for i in prediction]
        #try:
        Pearsonr= stats.pearsonr(LogKas,prediction)
        
    
        
        
        
        return loss,Pearsonr
        
        
        
        
        
        
        
        
    def step(self, data):
        pdbids, bgl, bgp,label= data
        bgl=bgl.to(device)
        bgp=bgp.to(device)
        label=label.to(device)
        
        label = Variable(label,requires_grad=True)
        
        
        self.model.to(device)
        with torch.no_grad():
            pred = self.model(bgl, bgp).to(device)
        return pred, label

    def compute_metrics(self, pred, gt, is_train):
        # you can call functions in metrics.py
        l1 = (pred - gt).abs().mean()
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics

    def gen_imgs_to_write(self, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            #prefix + '输入值': img,#[0],
            prefix + '预测值': pred[0],
            prefix + '真实值': label[0]
        }

    def compute_loss(self, pred, gt):
        #if self.args.loss == 'l1':
        #loss = (pred - gt).abs().mean()
        #elif self.args.loss == 'ce':
        #loss = torch.nn.functional.cross_entropy(pred, gt)
        #else:
        loss = torch.nn.functional.mse_loss(pred, gt)
        return loss


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
    