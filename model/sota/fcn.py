import torch as th
import torch.nn.functional as F
import dgl
import numpy as np
import random
import dgl.function as fn
from torch import nn
import pandas as pd
import torch

from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor
from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
from dgllife.model.gnn.wln import WLN
from dgllife.model.model_zoo.gat_predictor import GATPredictor


class ECIF_GNN(nn.Module):
	def __init__(self,dropout):#,in_feats,hidden_size,dropout):
		super(ECIF_GNN, self).__init__()
		self.dropout = dropout
		self.lig_model = AttentiveFPPredictor(node_feat_size=41,edge_feat_size=10,n_tasks=64,num_layers=3,graph_feat_size=200,dropout=0,num_timesteps=4)
		self.prot_model = AttentiveFPPredictor(node_feat_size=41, edge_feat_size=5,n_tasks=64, num_layers=3,graph_feat_size=200,dropout=0,num_timesteps=4)
		#self.lig_model =GATPredictor(in_feats=41, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None, biases=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=64, predictor_hidden_feats=128, predictor_dropout=0.0)
		#self.prot_model =GATPredictor(in_feats=41, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None, biases=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=64, predictor_hidden_feats=128, predictor_dropout=0.0)          
		#self.lig_model = DGLGraphTransformer(in_channels=41, edge_features=6, num_hidden_channels=64,activ_fn=th.nn.SiLU(),transformer_residual=True,num_attention_heads=4,norm_to_apply='batch',dropout_rate=0.15,num_layers=6)
		#self.prot_model = DGLGraphTransformer(in_channels=41, edge_features=5, num_hidden_channels=64,activ_fn=th.nn.SiLU(),transformer_residual=True,num_attention_heads=4,norm_to_apply='batch',dropout_rate=0.15,num_layers=6)
		self.MLP= nn.Sequential(#nn.Dropout(self.dropout),
                                    #nn.BatchNorm1d(256),
                                     #nn.LeakyReLU(),
                                     #nn.Dropout(0.1),
                                     #nn.Linear(in_features=256, out_features=128, bias=True),
                                     #nn.BatchNorm1d(128),
                                     nn.Linear(in_features=128, out_features=64, bias=True),nn.ReLU())
		self.MLP_2 = nn.Sequential(nn.Linear(in_features=128, out_features=64),#, bias=True),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=32),#, bias=True),
                                     #nn.BatchNorm1d(32),
                                     nn.ReLU(),
                                    # nn.Linear(in_features=32, out_features=16),#, bias=True),
                                     #nn.BatchNorm1d(32),
                                     #nn.ReLU(),
                                     #nn.Dropout(self.dropout),
                                     nn.Linear(in_features=32, out_features=1))#,nn.ReLU())#, bias=True))          
        
        
	def forward(self,bgl_0 ,bgp_0,bgl_1 ,bgp_1): 
		#print('atom_ndata:',bgl_0.ndata['atom'].float().size())   
		#print('edata_bond:',bgl_0.edata['bond'].float().size())
		#print('ndata_feats:',bgp_0.ndata['feats'].float().size())
		#print('edata_feats:',bgp_0.edata['feats'].float().size()) 
		h_l_0 = self.lig_model(bgl_0,bgl_0.ndata['atom'].float(), bgl_0.edata['bond'].float())
		h_p_0 = self.prot_model(bgp_0,bgp_0.ndata['feats'].float(), bgp_0.edata['feats'].float())
		#print('atom_ndata:',bgl.ndata['atom'].float().size())   
		#print('edata_bond:',bgl.edata['bond'].float().size())
		#print('ndata_feats:',bgp.ndata['feats'].float().size())
		#print('edata_feats:',bgp.edata['feats'].float().size())        
		#print('h_l:',h_l.size())
		#print('h_p:',h_p.size())
		h_l_1 = self.lig_model(bgl_1,bgl_1.ndata['atom'].float(), bgl_1.edata['bond'].float())
		h_p_1 = self.prot_model(bgp_1,bgp_1.ndata['feats'].float(), bgp_1.edata['feats'].float())
		output_0 = torch.cat([h_l_0,h_p_0],axis=1)
		output_1 = torch.cat([h_l_1,h_p_1],axis=1)
		#print("output_1:",output.size())
		output_2 = self.MLP(output_0)
		output_3 = self.MLP(output_1)
		#output = self.MLP(PLEC)
		#print("output_2:",output.size())
		regression = self.MLP_2(output_0) 
		return F.normalize(output_2, dim=1),F.normalize(output_3, dim=1),output_0 ,output_1,regression 
		#return output_2,output_3,output_0 ,output_1,regression 
    