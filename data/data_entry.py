from data.list_dataset import PDBbindDataset
#from list_dataset import PDBbindDataset
from torch.utils.data import DataLoader
import dgl
import numpy as np
import torch
from options import prepare_train_args
import argparse
args=prepare_train_args()
data_dir='/home/zyj/RTMScore复现/dataset'
data=PDBbindDataset(ids="%s/out_id_pre_train_protein.npy"%(data_dir),ligs_0="%s/out_ligand_pre_train_1.bin"%(data_dir),prots_0="%s/out_protein_pre_train_1.bin"%(data_dir),ligs_1="%s/out_ligand_pre_train_2.bin"%(data_dir),prots_1="%s/out_protein_pre_train_2.bin"%(data_dir))
#data=PDBbindDataset(ids="%s/out_id.npy"%(data_dir),ligs="%s/out_ligand.bin"%(data_dir),prots="%s/out_protein.bin"%(data_dir),fingerprint="%s/out_PLEC.npy"%(data_dir))
train_inds, val_inds =data.train_and_test_split(valnum=20000, seed=1234)
print('训练集个数',len(train_inds))
print('测试集个数',len(val_inds))
def collate(data):
	pdbids, graphsl_0, graphsp_0,graphsl_1, graphsp_1= map(list, zip(*data))
	bgl_0 = dgl.batch(graphsl_0)
	bgp_0 = dgl.batch(graphsp_0)
	for nty in bgl_0.ntypes:
		bgl_0.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgl_0.canonical_etypes:
		bgl_0.set_e_initializer(dgl.init.zero_initializer, etype=ety)
	for nty in bgp_0.ntypes:
		bgp_0.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgp_0.canonical_etypes:
		bgp_0.set_e_initializer(dgl.init.zero_initializer, etype=ety)

	bgl_1 = dgl.batch(graphsl_1)
	bgp_1 = dgl.batch(graphsp_1)
	for nty in bgl_1.ntypes:
		bgl_1.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgl_1.canonical_etypes:
		bgl_1.set_e_initializer(dgl.init.zero_initializer, etype=ety)
	for nty in bgp_1.ntypes:
		bgp_1.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgp_1.canonical_etypes:
		bgp_1.set_e_initializer(dgl.init.zero_initializer, etype=ety)    
        
        
	return pdbids,bgl_0,bgp_0,bgl_1,bgp_1


def get_dataset_by_type_train(args, is_train=False):
    type2data = {
        'PDBbind2020': PDBbindDataset(ids=data.pdbids[train_inds],
							ligs_0=np.array(data.graphsl_0)[train_inds],
							prots_0=np.array(data.graphsp_0)[train_inds],
							ligs_1=np.array(data.graphsl_1)[train_inds],
							prots_1=np.array(data.graphsp_1)[train_inds])
    }
    dataset = type2data['PDBbind2020']
    return dataset
def get_dataset_by_type_val(args, is_train=False):
    type2data = {
        'PDBbind2020': PDBbindDataset(ids=data.pdbids[val_inds],
							ligs_0=np.array(data.graphsl_0)[val_inds],
							prots_0=np.array(data.graphsp_0)[val_inds],
							ligs_1=np.array(data.graphsl_1)[val_inds],
							prots_1=np.array(data.graphsp_1)[val_inds])
    }
    dataset = type2data['PDBbind2020']
    return dataset


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type_train(args, True)
    print('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,args.batch_size, shuffle=True, num_workers=8,collate_fn=collate,drop_last=False)#, pin_memory=True, drop_last=False)
    return train_loader

#args.batch_size
def select_eval_loader(args):
    eval_dataset = get_dataset_by_type_val(args)
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset,args.batch_size, shuffle=False, num_workers=8,collate_fn=collate,drop_last=False)#, pin_memory=True, drop_last=False)
    return val_loader
