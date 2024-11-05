import numpy as np
import pandas as pd
from mol2graph import pdbbind_handle,drop_nodes,permute_edges,subgraph
import os
from copy import deepcopy
import multiprocessing

filelist = os.listdir('/data')
protein_path='/data/'
ligand_path='/data/'
results = []
error=[]
for index,i in enumerate(filelist):
#def handle(index,i):
    #output=pdbbind_handle(i,protein_path,ligand_path,5.0)
    #results.append(output)
    a=i.split("_")
    fingerprints =[]
    ids=[]
    graphs_p=[]   
    graphs_l=[]
    protein_path_pdb='%s/%s/'%(protein_path,i)+a[0]+'_handle_pocket_5.0.pdb'
    ligand_path_pdb='%s/%s/'%(protein_path,i)+a[0]+'_'+a[2]+'_'+a[3]+ "_"+a[4]+'.pdb'
    try:
        pdbid, gp, gl=pdbbind_handle(i,protein_path_pdb,ligand_path_pdb,10.0)
        ids.append(pdbid)
        gp_drop_nodes=drop_nodes(deepcopy(gp),0.2)
        gp_permute_edges=permute_edges(deepcopy(gp),0.2)
        gp_subgraph=subgraph(deepcopy(gp),0.2)


        #gl_drop_nodes=drop_nodes(deepcopy(gl),0.2)
        #gl_permute_edges=permute_edges(deepcopy(gl),0.2)
        #gl_subgraph=subgraph(deepcopy(gl),0.2)
        
        #del gl_subgraph.nodes['_N'].data['_ID']
        #del gl_subgraph.edata['_ID']
        del gp_subgraph.nodes['_N'].data['_ID']
        del gp_subgraph.edata['_ID']
        #print()

        #print('gl:',gl,gl_drop_nodes,gl_permute_edges,gl_subgraph)
        print('gp:',gp,gp_drop_nodes,gp_permute_edges,gp_subgraph)
        graphs_p.append(gp)
        graphs_p.append(gp_drop_nodes)
        graphs_p.append(gp_permute_edges)
        graphs_p.append(gp_subgraph)
        
        
        #graphs_l.append(gl)
        #graphs_l.append(gl_drop_nodes)
        #graphs_l.append(gl_permute_edges)
        #graphs_l.append(gl_subgraph)
        #results.append(output)
        #data=list(PLEC(ligand, protein=receptor, size=1024,  depth_protein=5,depth_ligand=1,distance_cutoff=5, sparse=False))
 
        #fingerprints.append(data)
        #np.save("%s/%s/out_id.npy"%(protein_path,i), ids)
        #np.save("%s/%s/out_protein.npy"%(protein_path,i), graphs_p)    
        #np.save("%s/%s/out_ligand.npy"%(protein_path,i), graphs_l)
        #save_graphs("out_PLEC.bin", PLECs)
        #np.save("%s/%s/out_PLEC.npy"%(protein_path,i),fingerprints)                
    except:
        print(index)
    
#from multiprocessing import Pool
#import os, time, random
#import tempfile
#if __name__=='__main__':
#    #print('Parent process %s.' % os.getpid())
#    p = Pool(20)
#    #f = open('/data/pdb/pdb_id.txt')
    #lines = f.readlines()  
#    for index,i in enumerate(filelist):
#        #print(index)
#        p.apply_async(handle, args=(index,i,))
#    print('Waiting for all subprocesses done...')
#    p.close()
#    p.join()
#    print('All subprocesses done.')
