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

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict

from data.list_dataset import *

def nt_xent_loss(out_1,out_2, temperature, eps=1e-5):
  
    #out = torch.cat([out_1, out_2], dim=0)
    print('out_1',out_1.size())
    print('out_2',out_2.size())
    cov = torch.mm(out_1, out_2.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)
      
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    #pos = torch.cat([pos, pos], dim=0)
    
    return -torch.log(pos / (neg + eps)).mean()
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 *self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nt_xent_criterion = NTXentLoss(device, batch_size=300, temperature=0.1,use_cosine_similarity=True)#, config['batch_size'], **config['loss'])
        self.model=select_model(args)
        if args.load_model_path != '':
            if args.load_not_strict:
                #load_match_dict(self.model, args.load_model_path)
                pass
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        #self.model = torch.nn.DataParallel(self.model)分布式训练
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr
                                          , betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay
                                         )

    def train(self):
        loss_list=[]
        for epoch in range(self.args.epochs):
            train_loss=self.train_per_epoch(epoch)
            test_loss=self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model,epoch)#保存模型
            print('epoch {:d} | train loss {:.4f} | val loss {:.4f}'.format(epoch, train_loss, test_loss))
            loss_list.append([float(train_loss), float(test_loss)])
        train_loss=pd.DataFrame(data=loss_list)#数据有三列，列名分别为one,two,three
        train_loss.to_csv('loss_list.csv',encoding='gbk')
        #eval_loss=pd.DataFrame(data=eval_loss)#数据有三列，列名分别为one,two,three
        #train_loss.to_csv('eval_loss.csv',encoding='gbk')
        
        
            

    def train_per_epoch(self, epoch,aux_weight=0.001):
        # switch to train mode
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
        #device = torch.device('cuda:0')
        self.model.train()
     
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
             
            pdbids,bgl_0,bgp_0,bgl_1,bgp_1= data
            bgl_0=bgl_0.to(device)
            bgp_0=bgp_0.to(device)
            bgl_1=bgl_1.to(device)
            bgp_1=bgp_1.to(device)
            self.model.to(device)
            #print(bgl)
            #print(bgp)
            #print(label)
            output_2,output_3,output_0 ,output_1,regression = self.model(bgl_0,bgp_0,bgl_1,bgp_1)#.to(device)
            
            
            #output_2 = F.normalize(output_2, dim=0)
            #output_3 = F.normalize(output_3, dim=0)
            loss=self.step(output_2,output_3,temperature=0.1)
            # compute gradient and do Adam step
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

            pdbids,bgl_0,bgp_0,bgl_1,bgp_1= data
            bgl_0=bgl_0.to(device)
            bgp_0=bgp_0.to(device)
            bgl_1=bgl_1.to(device)
            bgp_1=bgp_1.to(device)
            self.model.to(device)
            with torch.no_grad():
                output_2,output_3,output_0 ,output_1,regression= self.model(bgl_0,bgp_0,bgl_1,bgp_1)#.to(device)
                #output_2 = F.normalize(output_2, dim=0)
                #output_3 = F.normalize(output_3, dim=0)
            loss=self.step(output_2,output_3,temperature=0.1)
            return loss
    #def step(self, data):
     #   pdbids, bgl, bgp,label= data
     #   bgl=bgl.to(device)
     #   bgp=bgp.to(device)
     #   label=label.to(device)
        
     #   label = Variable(label,requires_grad=True)
        
        
     #   self.model.to(device)
     #   with torch.no_grad():
     #       pred = self.model(bgl, bgp).to(device)
     #   return pred, label

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

    def step(self, xis, xjs,temperature, n_iter=0,batchsize=300):
        # get the representations and the projections
        #ris, zis = model(xis)  # [N,C]


        # normalize projection feature vectors
        #zis = F.normalize(zis, dim=1)
        #zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(xis, xjs)#,temperature)
        return loss


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
    