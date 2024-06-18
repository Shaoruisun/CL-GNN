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


def PN_graph_construction_and_featurization_and_save(ligand_mol,
	                                                  protein_mol,
                                                     ligand_coordinates,
                                                     protein_coordinates,
                                                     pdb_id,
                                                     label,
                                                     
                                                     
                                                     max_num_ligand_atoms=None,
                                                     max_num_protein_atoms=None,
                                                     max_num_neighbors=4,
                                                     distance_bins=[1.5, 2.5, 3.5, 4.5],
                                                     strip_hydrogens=False):
    """Graph construction and featurization for `PotentialNet for Molecular Property Prediction
     <https://pubs.acs.org/doi/10.1021/acscentsci.8b00507>`__.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    protein_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    ligand_coordinates : Float Tensor of shape (V1, 3)
        Atom coordinates in a ligand.
    protein_coordinates : Float Tensor of shape (V2, 3)
        Atom coordinates in a protein.
    max_num_ligand_atoms : int or None
        Maximum number of atoms in ligands for zero padding, which should be no smaller than
        ligand_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_protein_atoms : int or None
        Maximum number of atoms in proteins for zero padding, which should be no smaller than
        protein_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_neighbors : int
        Maximum number of neighbors allowed for each atom when constructing KNN graph. Default to 4.
    distance_bins : list of float
        Distance bins to determine the edge types.
        Edges of the first edge type are added between pairs of atoms whose distances are less than `distance_bins[0]`.
        The length matches the number of edge types to be constructed.
        Default `[1.5, 2.5, 3.5, 4.5]`.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.

    Returns
    -------
    complex_bigraph : DGLGraph
        Bigraph with the ligand and the protein (pocket) combined and canonical features extracted.
        The atom features are stored as DGLGraph.ndata['h'].
        The edge types are stored as DGLGraph.edata['e'].
        The bigraphs of the ligand and the protein are batched together as one complex graph.
    complex_knn_graph : DGLGraph
        K-nearest-neighbor graph with the ligand and the protein (pocket) combined and edge features extracted based on distances.
        The edge types are stored as DGLGraph.edata['e'].
        The knn graphs of the ligand and the protein are batched together as one complex graph.

    """

    assert ligand_coordinates is not None, 'Expect ligand_coordinates to be provided.'
    assert protein_coordinates is not None, 'Expect protein_coordinates to be provided.'
    if max_num_ligand_atoms is not None:
        assert max_num_ligand_atoms >= ligand_mol.GetNumAtoms(), \
            'Expect max_num_ligand_atoms to be no smaller than ligand_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_ligand_atoms, ligand_mol.GetNumAtoms())
    if max_num_protein_atoms is not None:
        assert max_num_protein_atoms >= protein_mol.GetNumAtoms(), \
            'Expect max_num_protein_atoms to be no smaller than protein_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_protein_atoms, protein_mol.GetNumAtoms())

    if strip_hydrogens:
        # Remove hydrogen atoms and their corresponding coordinates
        ligand_atom_indices_left = filter_out_hydrogens(ligand_mol)
        protein_atom_indices_left = filter_out_hydrogens(protein_mol)
        ligand_coordinates = ligand_coordinates.take(ligand_atom_indices_left, axis=0)
        protein_coordinates = protein_coordinates.take(protein_atom_indices_left, axis=0)
    else:
        ligand_atom_indices_left = list(range(ligand_mol.GetNumAtoms()))
        protein_atom_indices_left = list(range(protein_mol.GetNumAtoms()))

    # Node featurizer for stage 1
    atoms = ['H','N','O','C','P','S','F','Br','Cl','I','Fe','Zn','Mg','Na','Mn','Ca','Co','Ni','Se','Cu','Cd','Hg','K']
    atom_total_degrees = list(range(5))
    atom_formal_charges = [-1, 0, 1]
    atom_implicit_valence = list(range(4))
    atom_explicit_valence = list(range(8))
    atom_concat_featurizer = ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=atoms), 
                                               partial(atom_total_degree_one_hot, allowable_set=atom_total_degrees),
                                               partial(atom_formal_charge_one_hot, allowable_set=atom_formal_charges),
                                               atom_is_aromatic,
                                               partial(atom_implicit_valence_one_hot, allowable_set=atom_implicit_valence),
                                               partial(atom_explicit_valence_one_hot, allowable_set=atom_explicit_valence)])
    PN_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})

    # Bond featurizer for stage 1
    bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    PN_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})

    # construct graphs for stage 1
    ligand_bigraph = mol_to_bigraph(ligand_mol, add_self_loop=False,
                                    node_featurizer=PN_atom_featurizer,
                                    edge_featurizer=PN_bond_featurizer,
                                    canonical_atom_order=False) # Keep the original atomic order)
    protein_bigraph = mol_to_bigraph(protein_mol, add_self_loop=False,
                                     node_featurizer=PN_atom_featurizer,
                                     edge_featurizer=PN_bond_featurizer,
                                     canonical_atom_order=False)

    
	return ligand_bigraph,protein_bigraph
    
   
def int_2_one_hot(a):
	"""Convert integer encodings on a vector to a matrix of one-hot encoding"""
	n = len(a)
	b = np.zeros((n, a.max()+1))
	b[np.arange(n), a] = 1
	return b


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



def calc_atom_features(atom, explicit_H):
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
                                                  [0, 1, 2, 3, 4,5,6,7])	
    return np.array(results)


def calc_bond_features(bond, use_chirality):
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


	
def load_mol(molpath, explicit_H, use_chirality):
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


def mol_to_graph(mol, explicit_H, use_chirality):
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

def mol_to_graph2(pro, lig, cutoff=10.0, explicit_H=False,use_chirality=True):
	protein = load_mol(pro, explicit_H=explicit_H, use_chirality=use_chirality) 
	ligand = load_mol(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	gl = mol_to_graph(ligand,explicit_H=explicit_H, use_chirality=use_chirality)
	gp = prot_to_graph(protein, cutoff)
	return gp, gl

def mol_to_graph3(pro, lig, cutoff=10.0, explicit_H=False,use_chirality=True):
	#protein = load_mol(pro, explicit_H=explicit_H, use_chirality=use_chirality) 
	ligand = load_mol(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	gl = mol_to_graph(ligand,explicit_H=explicit_H, use_chirality=use_chirality)
	#gp = prot_to_graph(pro, cutoff)
	return gl


def pdbbind_handle(pdbid,prot_path,lig_path,cutoff):
	#try:
		#lig_path = "%s/%s/%s_ligand.pdb"%(lig_path, pdbid, pdbid)
		#print(lig_path)
	#except:
		#lig_path = "%s/%s/%s_ligand.sdf"%(lig_path, pdbid, pdbid)
		#print(lig_path)
	prot_path = "%s/%s/%s_protein_handle_pocket_%s.pdb"%(prot_path,pdbid,pdbid,cutoff)
	lig_path_pdb = "%s/%s/%s_ligand.pdb"%(lig_path, pdbid, pdbid)
	lig_path_sdf = "%s/%s/%s_ligand.sdf"%(lig_path, pdbid, pdbid)
	lig_path_mol2 = "%s/%s/%s_ligand.mol2"%(lig_path, pdbid, pdbid)
	try: 
		gp,gl = mol_to_graph2(prot_path, 
							lig_path_pdb, 
							cutoff=cutoff,
							explicit_H=False,use_chirality=False)
     
		#print('蛋白：',gp)
		#print('药物：',gl)
	except:
		try:
			gp,gl = mol_to_graph2(prot_path, 
							lig_path_mol2, 
							cutoff=cutoff,
							explicit_H=False,use_chirality=False)
		except:
			#try:
			gp, gl = mol_to_graph2(prot_path, 
							lig_path_sdf, 
							cutoff=cutoff,
							explicit_H=False,use_chirality=False)
			#except: ZeroDivisionError:
				#print("%s failed to generare the graph"%pdbid)
				#gp, gl = None, None
	return pdbid, gp, gl
