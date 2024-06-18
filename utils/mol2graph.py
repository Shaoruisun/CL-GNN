from dgllife.data.pdbbind import PDBBind
from dgllife.utils import multiprocess_load_molecules
from dgl.data.utils import get_download_dir,extract_archive
import dgl.backend as F
import glob
import os
import multiprocessing
from tqdm import tqdm

ROOT_DIR = os.getcwd()
print(f'Current working directory : {ROOT_DIR}')
from functools import partial

import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re
import dgl
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl.data.utils import save_graphs, load_graphs, load_labels
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances

METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","U","PU","TH"] 
RES_MAX_NATOMS=24

def prot_to_graph(prot, cutoff):
	"""obtain the residue graphs"""
	u = mda.Universe(prot)
	g = dgl.DGLGraph()
	# Add nodes
	num_residues = len(u.residues)
	g.add_nodes(num_residues)
	
	res_feats = np.array([calc_res_features(res) for res in u.residues])
	g.ndata["feats"] = th.tensor(res_feats)
	edgeids, distm = obatin_edge(u, cutoff)	
	src_list, dst_list = zip(*edgeids)
	g.add_edges(src_list, dst_list)
	
	g.ndata["ca_pos"] = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))	
	g.ndata["center_pos"] = th.tensor(u.atoms.center_of_mass(compound='residues'))
	dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
	cadist = th.tensor([dis_matx_ca[i,j] for i,j in edgeids]) * 0.1
	dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
	cedist = th.tensor([dis_matx_center[i,j] for i,j in edgeids]) * 0.1
	edge_connect =  th.tensor(np.array([check_connect(u, x, y) for x,y in zip(src_list, dst_list)]))
	g.edata["feats"] = th.cat([edge_connect.view(-1,1), cadist.view(-1,1), cedist.view(-1,1), th.tensor(distm)], dim=1)
	g.ndata.pop("ca_pos")
	g.ndata.pop("center_pos")
	#res_max_natoms = max([len(res.atoms) for res in u.residues])
	g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	#g.ndata["posmask"] = th.tensor([[1]* len(res.atoms)+[0]*(RES_MAX_NATOMS-len(res.atoms)) for res in u.residues]).bool()
	#g.ndata["atnum"] = th.tensor([len(res.atoms) for res in u.residues])
	return g


def obtain_ca_pos(res):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms("name CA").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
	try:
		if res.phi_selection() is not None:
			phi = res.phi_selection().dihedral.value()
		else:
			phi = 0
		if res.psi_selection() is not None:
			psi = res.psi_selection().dihedral.value()
		else:
			psi = 0
		if res.omega_selection() is not None:
			omega = res.omega_selection().dihedral.value()
		else:
			omega = 0
		if res.chi1_selection() is not None:
			chi1 = res.chi1_selection().dihedral.value()
		else:
			chi1 = 0
		return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
	except:
		return [0, 0, 0, 0]

def calc_res_features(res):
	return np.array(one_of_k_encoding_unk(obtain_resname(res), 
										['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +          #32  residue type	
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)

def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname

##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
	edgeids = []
	dismin = []
	dismax = []
	for res1, res2 in permutations(u.residues, 2):
		dist = calc_dist(res1, res2)
		if dist.min() <= cutoff:
			edgeids.append([res1.ix, res2.ix])
			dismin.append(dist.min()*0.1)
			dismax.append(dist.max()*0.1)
	return edgeids, np.array([dismin, dismax]).T



def check_connect(u, i, j):
	if abs(i-j) != 1:
		return 0
	else:
		if i > j:
			i = j
		nb1 = len(u.residues[i].get_connections("bonds"))
		nb2 = len(u.residues[i+1].get_connections("bonds"))
		nb3 = len(u.residues[i:i+2].get_connections("bonds"))
		if nb1 + nb2 == nb3 + 1:
			return 1
		else:
			return 0
		
	

def calc_dist(res1, res2):
	#xx1 = res1.atoms.select_atoms('not name H*')
	#xx2 = res2.atoms.select_atoms('not name H*')
	#dist_array = distances.distance_array(xx1.positions,xx2.positions)
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array
	#return dist_array.max()*0.1, dist_array.min()*0.1



def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
       'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 
		'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 
		'Cu', 'Mn', 'Mo', 'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]
                # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])	
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


	
def load_mol(molpath, explicit_H=False, use_chirality=True):
	# load mol
	if re.search(r'.pdb$', molpath):
		mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
	elif re.search(r'.mol2$', molpath):
		mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
	elif re.search(r'.sdf$', molpath):			
		mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
	else:
		raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
	
	if use_chirality:
		Chem.AssignStereochemistryFrom3D(mol)
	return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=False):
	"""
	mol: rdkit.Chem.rdchem.Mol
	explicit_H: whether to use explicit H
	use_chirality: whether to use chirality
	"""   	
				
	g = dgl.DGLGraph()
	# Add nodes
	num_atoms = mol.GetNumAtoms()
	g.add_nodes(num_atoms)
	
	atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
	if use_chirality:
		chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
		chiral_arr = np.zeros([num_atoms,3]) 
		for (i, rs) in chiralcenters:
			if rs == 'R':
				chiral_arr[i, 0] =1 
			elif rs == 'S':
				chiral_arr[i, 1] =1 
			else:
				chiral_arr[i, 2] =1 
		atom_feats = np.concatenate([atom_feats,chiral_arr],axis=1)
			
	g.ndata["atom"] = th.tensor(atom_feats)
	
	# obtain the positions of the atoms
	atomCoords = mol.GetConformer().GetPositions()
	g.ndata["pos"] = th.tensor(atomCoords)
	
	# Add edges
	src_list = []
	dst_list = []
	bond_feats_all = []
	num_bonds = mol.GetNumBonds()
	for i in range(num_bonds):
		bond = mol.GetBondWithIdx(i)
		u = bond.GetBeginAtomIdx()
		v = bond.GetEndAtomIdx()
		bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
		src_list.extend([u, v])
		dst_list.extend([v, u])		
		bond_feats_all.append(bond_feats)
		bond_feats_all.append(bond_feats)
	
	g.add_edges(src_list, dst_list)
	#normal_all = []
	#for i in etype_feature_all:
	#	normal = etype_feature_all.count(i)/len(etype_feature_all)
	#	normal = round(normal, 1)
	#	normal_all.append(normal)
	
	g.edata["bond"] = th.tensor(np.array(bond_feats_all))
	#g.edata["normal"] = th.tensor(normal_all)
	
	#dis_matx = distance_matrix(g.ndata["pos"], g.ndata["pos"])
	#g.edata["dist"] = th.tensor([dis_matx[i,j] for i,j in zip(*g.edges())]) * 0.1	
	return g

def mol_to_graph2(pro, lig, cutoff=10, explicit_H=False, use_chirality=False):
	protein = load_mol(pro, explicit_H=explicit_H, use_chirality=use_chirality) 
	ligand = load_mol(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	gl = mol_to_graph(ligand)
	gp = prot_to_graph(protein, cutoff)
	return gp, gl


def PN_graph_construction_and_featurization_and_save(pro,lig,pdbid):
	#pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s"%(args.dir, x))]
	#if args.parallel:
	##else:
	try: 
		gp, gl = mol_to_graph2(pro,lig,
						cutoff=10)#print(pdbid)#print(gp)#print(gl)
		
	except:
		return pdbid  
		#print("%s failed to generare the graph"%pdbid)  
	#	gp, gl = None, None
	result = []
	result.append(gp)
	result.append(gl)   
	save_graphs("%s.bin"%pdbid,list(result))

    
    
class PDBBind_v2020(PDBBind):
    def __init__(self, 
                 subset, 
                 pdb_version='v2015', 
                 load_binding_pocket=True, 
                 remove_coreset_from_refinedset=True, 
                 sanitize=False, 
                 calc_charges=False, 
                 remove_hs=False, 
                 use_conformation=True,
                 zero_padding=True, 
                 num_processes=None, 
                 local_path=None,
                 distance_bins=[1.5, 2.5, 3.5, 4.5],
                 save_bin_files=True):
        print(" ")
        print("---Using PDBBind v2020 compatible loader---")
        print(" ")
        
        self.save_bin_files = save_bin_files
        self.pdb_version = pdb_version
        self.subset = subset
        self.distance_bins = distance_bins
        
        super().__init__(subset, 
                         pdb_version, 
                         load_binding_pocket, 
                         remove_coreset_from_refinedset, 
                         sanitize, 
                         calc_charges, 
                         remove_hs, 
                         use_conformation,
                         zero_padding, 
                         num_processes, 
                         local_path)
        
    def _read_data_files(self, pdb_version, subset, load_binding_pocket, remove_coreset_from_refinedset, local_path):
        """Download and extract pdbbind data files specified by the version"""
        root_dir_path = get_download_dir()
        if local_path:
            print(" ")
            print("--Using Local Path--")
            print(" ")
            if local_path[-1] != '/':
                local_path += '/'
            index_label_file = glob.glob(local_path + '*' + subset + '*data*')[0]
            
        elif pdb_version == 'v2015':
            print(" ")
            print("--v2015--")
            print(" ")
            
            self._url = 'dataset/pdbbind_v2015.tar.gz'
            data_path = root_dir_path + '/pdbbind_v2015.tar.gz'
            extracted_data_path = root_dir_path + '/pdbbind_v2015'
            download(_get_dgl_url(self._url), path=data_path, overwrite=False)
            extract_archive(data_path, extracted_data_path)

            if subset == 'core':
                index_label_file = extracted_data_path + '/v2015/INDEX_core_data.2013'
            elif subset == 'refined':
                index_label_file = extracted_data_path + '/v2015/INDEX_refined_data.2015'
            else:
                raise ValueError('Expect the subset_choice to be either core or refined, got {}'.format(subset))
                
        elif pdb_version == 'v2020':
            print(" ")
            print("--v2020--")
            print(" ")
            
            root_dir_path = ROOT_DIR
            print(f'root_dir_path : {root_dir_path}')
            
            # Others
            #print("--v2020 other PL--")
            #
            #data_path = root_dir_path + '/PDBbind_v2020_other_PL.tar.gz'
            #print(f'data_path : {data_path}')
            #extracted_data_path = root_dir_path + '/pdbbind_v2020'
            #extract_archive(data_path, extracted_data_path)
            
            # Refined
            print("--v2020 refined--")
            data_path = root_dir_path + '/PDBbind_v2020_refined.tar.gz'
            print(f'data_path : {data_path}')
            extracted_data_path = root_dir_path + '/pdbbind_v2020'
            extract_archive(data_path, extracted_data_path, overwrite=True)

            #index_label_file = extracted_data_path + '/v2020-other-PL/index/INDEX_general_PL_data.2020'
            if subset == 'core':
                print("--v2020 core--")
                # Read index file for refined dataset 
                #index_label_file = extracted_data_path + '/v2020-other-PL/index/INDEX_refined_data.2020'
                index_label_file = extracted_data_path + '/refined-set/index/INDEX_refined_data.2020'
                # Core
                data_path = root_dir_path + '/CASF-2016.tar.gz'
                extracted_data_path = root_dir_path + '/pdbbind_v2020'
                extract_archive(data_path, extracted_data_path, overwrite=True)
                core_dir = extracted_data_path + '/CASF-2016/coreset'
                print(f'core_dir : {core_dir}')
            elif subset == 'refined':
                #index_label_file = extracted_data_path + '/v2020-other-PL/index/INDEX_refined_data.2020'
                index_label_file = extracted_data_path + '/refined-set/index/INDEX_refined_data.2020'
            elif subset == 'general':
                #index_label_file = extracted_data_path + '/v2020-other-PL/index/INDEX_general_PL_data.2020'
                index_label_file = extracted_data_path + '/refined-set/index/INDEX_general_PL_data.2020'
                
        elif pdb_version == 'v2007':
            print(" ")
            print("--v2007--")
            print(" ")
            
            self._url = 'dataset/pdbbind_v2007.tar.gz'
            data_path = root_dir_path + '/pdbbind_v2007.tar.gz'
            extracted_data_path = root_dir_path + '/pdbbind_v2007'
            download(_get_dgl_url(self._url), path=data_path, overwrite=False)
            extract_archive(data_path, extracted_data_path, overwrite=False)
            extracted_data_path += '/home/ubuntu' # extra layer 

            # DataFrame containing the pdbbind_2007_agglomerative_split.txt
            self.agg_split = pd.read_csv(extracted_data_path + '/v2007/pdbbind_2007_agglomerative_split.txt')
            self.agg_split.rename(columns={'PDB ID':'PDB_code', 'Sequence-based assignment':'sequence', 'Structure-based assignment':'structure'}, inplace=True)
            self.agg_split.loc[self.agg_split['PDB_code']=='1.00E+66', 'PDB_code'] = '1e66' # fix typo
            if subset == 'core':
                index_label_file = extracted_data_path + '/v2007/INDEX.2007.core.data'
            elif subset == 'refined':
                index_label_file = extracted_data_path + '/v2007/INDEX.2007.refined.data'
            else:
                raise ValueError('Expect the subset_choice to be either core or refined, got {}'.format(subset))
                
                
        print("")         
        print("index_label_file")        
        print(index_label_file)
        print("") 
        contents = []
        with open(index_label_file, 'r') as f:
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    if pdb_version == 'v2015' or pdb_version == 'v2020':
                        if len(splitted_elements) == 8:
                            # Ignore "//"
                            contents.append(splitted_elements[:5] + splitted_elements[6:])
                        else:
                            print('Incorrect data format.')
                            print(splitted_elements)

                    elif pdb_version == 'v2007':
                        if len(splitted_elements) == 6:
                            contents.append(splitted_elements)
                        else:
                            contents.append(splitted_elements[:5] + [' '.join(splitted_elements[5:])])

        if pdb_version == 'v2015' or pdb_version == 'v2020':
            self.df = pd.DataFrame(contents, columns=(
                'PDB_code', 'resolution', 'release_year',
                '-logKd/Ki', 'Kd/Ki', 'reference', 'ligand_name'))
        elif pdb_version == 'v2007':
            self.df = pd.DataFrame(contents, columns=(
                'PDB_code', 'resolution', 'release_year',
                '-logKd/Ki', 'Kd/Ki', 'cluster_ID'))
         
        if local_path:
            pdb_path = local_path
        elif pdb_version == 'v2020' and subset != 'core':
            pdb_path = os.path.join(extracted_data_path, 'v2020-other-PL')
            print('Loading PDBBind data from', pdb_path)
            pdb_dirs = glob.glob(pdb_path + '/*')
            
            pdb_path = os.path.join(extracted_data_path, 'refined-set')
            print('Loading PDBBind data from', pdb_path)
            pdb_dirs += glob.glob(pdb_path + '/*')
        elif pdb_version == 'v2020' and subset == 'core':
            pdb_path = os.path.join(extracted_data_path, 'CASF-2016/coreset')
            print('Loading PDBBind data from', pdb_path)
            pdb_dirs = glob.glob(pdb_path + '/*')
            
        else:
            pdb_path = os.path.join(extracted_data_path, pdb_version)
            print('Loading PDBBind data from', pdb_path)
            pdb_dirs = glob.glob(pdb_path + '/*')
        
        pdb_dirs = [pdb_dir for pdb_dir in pdb_dirs if '.' not in pdb_dir.split('/')[-1]]
        
        print("")
        print('pdb_dirs from dirs')
        print(f"data length : {len(pdb_dirs)}")
        print("")
        
        ## pdb and pdb_dirs update
        dict_pdb_dirs = {pdb_dir.split('/')[-1]: pdb_dir for pdb_dir in pdb_dirs}
        self.df['pdb_paths'] = self.df['PDB_code'].map(dict_pdb_dirs)  
        self.df = self.df.dropna().drop_duplicates().reset_index().drop(columns=['index'])
        
        # remove core set from refined set if using refined
        if remove_coreset_from_refinedset and subset == 'refined' and pdb_version != 'v2020':
            if local_path:
                core_path = glob.glob(local_path + '*core*data*')[0]
            elif pdb_version == 'v2015':
                core_path = extracted_data_path + '/v2015/INDEX_core_data.2013'
            elif pdb_version == 'v2007':
                core_path = extracted_data_path + '/v2007/INDEX.2007.core.data'

            core_pdbs = []
            with open(core_path,'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if fields[0] != "#":
                        core_pdbs.append(fields[0])
                        
            non_core_ids = []
            for i in range(len(self.df)):
                if self.df['PDB_code'][i] not in core_pdbs:
                    non_core_ids.append(i)
            self.df = self.df.iloc[non_core_ids]
                        
        if remove_coreset_from_refinedset and subset != 'core' and pdb_version == 'v2020':
            core_pdb_path = os.path.join(extracted_data_path, 'CASF-2016/coreset')
            print('Loading PDBBind data from', core_pdb_path)
            core_pdb_dirs = glob.glob(core_pdb_path + '/*')
            
            core_pdb_dirs = [core_pdb_dir for core_pdb_dir in core_pdb_dirs if '.' not in core_pdb_dir]
            
            core_pdbs = []
            for core_pdb in core_pdb_dirs: 

                core_pdbs.append(core_pdb.split('/')[-1])

            non_core_ids = []
            for i in range(len(self.df)):
                if self.df['PDB_code'][i] not in core_pdbs:
                    non_core_ids.append(i)
            self.df = self.df.iloc[non_core_ids]
            
        
        # The final version of self.df
        pdbs = self.df['PDB_code'].tolist()
        pdb_dirs = self.df['pdb_paths'].tolist()

        
        print("")
        print('pdbs')
        #print(pdbs)
        print(f"data length : {len(pdbs)}")
        print("")
        
        print("")
        print('pdb_dirs from dirs')
        print(f"data length : {len(pdb_dirs)}")
        print("")

        self.ligand_files = [os.path.join(pdb_dir, '{}_ligand.sdf'.format(pdb_dir.split('/')[-1])) for pdb_dir in pdb_dirs if pdb_dir.split('/')[-1] in pdbs]

        if load_binding_pocket:
            self.protein_files = [os.path.join(pdb_dir, '{}_pocket.pdb'.format(pdb_dir.split('/')[-1])) for pdb_dir in pdb_dirs if pdb_dir.split('/')[-1] in pdbs]
        else:
            self.protein_files = [os.path.join(pdb_dir, '{}_protein.pdb'.format(pdb_dir.split('/')[-1])) for pdb_dir in pdb_dirs if pdb_dir.split('/')[-1] in pdbs]
    def _preprocess(self, load_binding_pocket,
                    sanitize, calc_charges, remove_hs, use_conformation,
                    construct_graph_and_featurize, zero_padding, num_processes):
        """Preprocess the dataset.

        The pre-processing proceeds as follows:

        1. Load the dataset
        2. Clean the dataset and filter out invalid pairs
        3. Construct graphs
        4. Prepare node and edge features

        Parameters
        ----------
        load_binding_pocket : bool
            Whether to load binding pockets or full proteins.是否装载绑定袋或完整的蛋白质。
        sanitize : bool
            Whether sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        calc_charges : bool
            Whether to add Gasteiger charges via RDKit.计算电荷 Setting this to be True will enforce
            ``sanitize`` to be True.
        remove_hs : bool 是否去氢
            Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
            slow for large molecules.
        use_conformation : bool是否需要从蛋白质和配体中提取分子构象。
            Whether we need to extract molecular conformation from proteins and ligands.
        construct_graph_and_featurize : callable构造一个用于gnn的DGLHeteroGraph
            Construct a DGLHeteroGraph for the use of GNNs. Mapping self.ligand_mols[i],
            self.protein_mols[i], self.ligand_coordinates[i] and self.protein_coordinates[i]
            to a DGLHeteroGraph. Default to :func:`ACNN_graph_construction_and_featurization`.
        zero_padding : bool是否进行零填充。
            Whether to perform zero padding. While DGL does not necessarily require zero padding,
            pooling operations for variable length inputs can introduce stochastic behaviour, which
            is not desired for sensitive scenarios.
        num_processes : int or None
            Number of worker processes to use. If None,
            then we will use the number of CPUs in the system.
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()#multiprocessing包是Python中的多进程管理包。
        num_processes = min(num_processes, len(self.df))

        print('Loading ligands...')
        ligands_loaded = multiprocess_load_molecules(self.ligand_files,
                                                     sanitize=sanitize,
                                                     calc_charges=calc_charges,
                                                     remove_hs=remove_hs,
                                                     use_conformation=use_conformation,
                                                     num_processes=num_processes)

        print('Loading proteins...')
        proteins_loaded = multiprocess_load_molecules(self.protein_files,
                                                      sanitize=sanitize,
                                                      calc_charges=calc_charges,
                                                      remove_hs=remove_hs,
                                                      use_conformation=use_conformation,
                                                      num_processes=num_processes)

        self._filter_out_invalid(ligands_loaded, proteins_loaded, use_conformation)
        self.df = self.df.iloc[self.indices]
                
        self.labels = F.zerocopy_from_numpy(self.df[self.task_names].values.astype(np.float32))
        print('Finished cleaning the dataset, '
              'got {:d}/{:d} valid pairs'.format(len(self), len(self.ligand_files))) # account for the ones use_conformation failed

        # Prepare zero padding
        if zero_padding:
            max_num_ligand_atoms = 0
            max_num_protein_atoms = 0
            for i in range(len(self)):
                max_num_ligand_atoms = max(
                    max_num_ligand_atoms, self.ligand_mols[i].GetNumAtoms())
                max_num_protein_atoms = max(
                    max_num_protein_atoms, self.protein_mols[i].GetNumAtoms())
        else:
            max_num_ligand_atoms = None
            max_num_protein_atoms = None

        #construct_graph_and_featurize = partial(construct_graph_and_featurize, 
                            #max_num_ligand_atoms=max_num_ligand_atoms,
                            #max_num_protein_atoms=max_num_protein_atoms)

        print('Start constructing graphs and featurizing them.')
        num_mols = len(self)
        
        # Run this after the filter
        pdbs = self.df['PDB_code'].tolist()
        pdb_dirs = self.df['pdb_paths'].tolist()
        pdb_version = self.pdb_version
        subset = self.subset
        self.df.to_csv(ROOT_DIR + f'/{pdb_version}-{subset}-{pdbs[0]}-{pdbs[-1]}-{len(pdbs)}.csv')
        
        if self.save_bin_files: 
            for i in tqdm(range(len(self.labels)), desc="Loading..."):
                print(ROOT_DIR+'/'+'pdbbind_v2020/CASF-2016/coreset/'+pdbs[i]+'/'+pdbs[i]+'_pocket.pdb')
                PN_graph_construction_and_featurization_and_save(ROOT_DIR+'/'+'pdbbind_v2020/CASF-2016/coreset/'+pdbs[i]+'/'+pdbs[i]+'_pocket.pdb',
                    ROOT_DIR+'/'+'pdbbind_v2020/CASF-2016/coreset/'+pdbs[i]+'/'+pdbs[i]+'_ligand.sdf',pdbs[i])

        else:
            pool = multiprocessing.Pool(processes=num_processes)
            self.graphs = pool.starmap(#construct_graph_and_featurize, 
                                       zip(self.ligand_mols, self.protein_mols,
                                           self.ligand_coordinates, self.protein_coordinates))
        
        print(f'Done constructing {len(self.labels)} graphs.')
        
    def __getitem__(self, item):
        """Get the datapoint associated with the index.

        Parameters
        ----------
        item : int
            Index for the datapoint.

        Returns
        -------
        int
            Index for the datapoint.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the ligand molecule.
        rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the protein molecule.
        DGLGraph or tuple of DGLGraphs
            Pre-processed DGLGraph with features extracted.
            For ACNN, a single DGLGraph;
            For PotentialNet, a tuple of DGLGraphs that consists of a molecular graph and a KNN graph of the complex.
        Float32 tensor
            Label for the datapoint.
        """
        if self.save_bin_files:
            return item, self.ligand_mols[item], self.protein_mols[item], self.labels[item]                               
        else:
            return item, self.ligand_mols[item], self.protein_mols[item], \
               self.graphs[item], self.labels[item]
    
