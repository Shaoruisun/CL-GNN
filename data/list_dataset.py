import torch as th
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset  # , DataLoader
import pandas as pd
import numpy as np
# from copy import deepcopy
from rdkit import Chem
from joblib import Parallel, delayed
import os
import tempfile
import shutil
#from utils.viz import mol_to_graph, load_mol, prot_to_graph
#from utils.extract_pocket_prody import extract_pocket

file='/home/zyj/RTMScore复现/data/pdbbind_v2020_原始数据/protein'

class PDBbindDataset(Dataset):
    def __init__(self,
                 ids=None,
                 ligs_0=None,
                 prots_0=None,
                 ligs_1=None,
                 prots_1=None
                 ):
        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
        if isinstance(ligs_0, np.ndarray) or isinstance(ligs_0, tuple) or isinstance(ligs_0, list):
            if isinstance(ligs_0[0], dgl.DGLGraph):
                self.graphsl_0 = ligs_0
            else:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
        else:
            try:
                self.graphsl_0, _ = load_graphs(ligs_0)
            except:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')

        if isinstance(prots_0, np.ndarray) or isinstance(prots_0, tuple) or isinstance(prots_0, list):
            if isinstance(prots_0[0], dgl.DGLGraph):
                self.graphsp_0 = prots_0
            else:
                raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')
        else:
            try:
                self.graphsp_0, _ = load_graphs(prots_0)
            except:
                raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')
                
                
                
        if isinstance(ligs_1, np.ndarray) or isinstance(ligs_1, tuple) or isinstance(ligs_1, list):
            if isinstance(ligs_1[0], dgl.DGLGraph):
                self.graphsl_1 = ligs_1
            else:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
        else:
            try:
                self.graphsl_1, _ = load_graphs(ligs_1)
            except:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')

        if isinstance(prots_1, np.ndarray) or isinstance(prots_1, tuple) or isinstance(prots_1, list):
            if isinstance(prots_1[0], dgl.DGLGraph):
                self.graphsp_1 = prots_1
            else:
                raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')
        else:
            try:
                self.graphsp_1, _ = load_graphs(prots_1)
            except:
                raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')

                
        #self.graphsl_0 = list(self.graphsl_0)
        #self.graphsp_0 = list(self.graphsp_0)
        
        #self.graphsl_1 = list(self.graphsl_1)
        #self.graphsp_1 = list(self.graphsp_1)
        
        print(len(self.pdbids),len(self.graphsl_0),len(self.graphsp_0),len(self.graphsl_1),len(self.graphsp_1))

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """


        return self.pdbids[idx], self.graphsl_0[idx], self.graphsp_0[idx],self.graphsl_1[idx], self.graphsp_1[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.pdbids)

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        return train_inds, val_inds
