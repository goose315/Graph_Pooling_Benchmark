# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:11:05 2022

@author: Fanding Xu
"""

import pandas as pd
import numpy as np
import torch
from itertools import chain, combinations, product
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from IPython.display import SVG, Image
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
import scipy.sparse as sp
from distinctipy import distinctipy


# colors = [(0.2, 0.13333333333333333, 0.5333333333333333),
#           (0.06666666666666667, 0.4666666666666667, 0.2),
#           (0.26666666666666666, 0.6666666666666666, 0.6),
#           (0.5333333333333333, 0.8, 0.9333333333333333),
#           (0.8666666666666667, 0.8, 0.4666666666666667),
#           (0.8, 0.4, 0.4666666666666667),
#           (0.6666666666666666, 0.26666666666666666, 0.6),
#           (0.5333333333333333, 0.13333333333333333, 0.3333333333333333),
#           (0.7372549019607844, 0.8235294117647058, 0.9333333333333333),
#           (0.0, 0.803921568627451, 0.4)]


colors = distinctipy.get_colors(50, pastel_factor=0.4, colorblind_type='Deuteranomaly')[8:]
# colors = [(255, 182, 193),  (255, 192, 203),  (220, 20, 60),  (255, 240, 245),  (219, 112, 147),  (255, 105, 180),  (255, 20, 147),  (199, 21, 133),  (218, 112, 214),  (216, 191, 216),  (221, 160, 221),  (238, 130, 238),  (255, 0, 255),  (255, 0, 255),  (139, 0, 139),  (128, 0, 128),  (186, 85, 211),  (148, 0, 211),  (153, 50, 204),  (75, 0, 130),  (138, 43, 226),  (147, 112, 219),  (123, 104, 238),  (106, 90, 205),  (72, 61, 139),  (230, 230, 250),  (248, 248, 255),  (0, 0, 255),  (0, 0, 205),  (25, 25, 112),  (0, 0, 139),  (0, 0, 128),  (65, 105, 225),  (100, 149, 237),  (176, 196, 222),  (119, 136, 153),  (112, 128, 144),  (30, 144, 255),  (240, 248, 255),  (70, 130, 180),  (135, 206, 250),  (135, 206, 235),  (0, 191, 255),  (173, 216, 230),  (176, 224, 230),  (95, 158, 160),  (240, 255, 255),  (225, 255, 255),  (175, 238, 238),  (0, 255, 255),  (212, 242, 231),  (0, 206, 209),  (47, 79, 79),  (0, 139, 139),  (0, 128, 128),  (72, 209, 204),  (32, 178, 170),  (64, 224, 208),  (127, 255, 170),  (0, 250, 154),  (0, 255, 127),  (245, 255, 250),  (60, 179, 113),  (46, 139, 87),  (240, 255, 240),  (144, 238, 144),  (152, 251, 152),  (143, 188, 143),  (50, 205, 50),  (0, 255, 0),  (34, 139, 34),  (0, 128, 0),  (0, 100, 0),  (127, 255, 0),  (124, 252, 0),  (173, 255, 47),  (85, 107, 47),  (245, 245, 220),  (250, 250, 210),  (255, 255, 240),  (255, 255, 224),  (255, 255, 0),  (128, 128, 0),  (189, 183, 107),  (255, 250, 205),  (238, 232, 170),  (240, 230, 140),  (255, 215, 0),  (255, 248, 220),  (218, 165, 32),  (255, 250, 240),  (253, 245, 230),  (245, 222, 179),  (255, 228, 181),  (255, 165, 0),  (255, 239, 213),  (255, 235, 205),  (255, 222, 173),  (250, 235, 215),  (210, 180, 140),  (222, 184, 135),  (255, 228, 196),  (255, 140, 0),  (250, 240, 230),  (205, 133, 63),  (255, 218, 185),  (244, 164, 96),  (210, 105, 30),  (139, 69, 19),  (255, 245, 238),  (160, 82, 45),  (255, 160, 122),  (255, 127, 80),  (255, 69, 0),  (233, 150, 122),  (255, 99, 71),  (255, 228, 225),  (250, 128, 114),  (255, 250, 250),  (240, 128, 128),  (188, 143, 143),  (205, 92, 92),  (255, 0, 0),  (165, 42, 42),  (178, 34, 34),  (139, 0, 0),  (128, 0, 0)]
# colors = np.array(colors)/255
# colors = colors[np.random.permutation(colors.shape[0])]
# colors = [tuple(x) for x in colors]




def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol
def mol_with_bond_index( mol ):
    bonds = mol.GetNumBonds()
    for idx in range( bonds ):
        mol.GetBondWithIdx( idx ).SetProp( 'molBondMapNumber', str( mol.GetBondWithIdx( idx ).GetIdx() ) )
    return mol
 
def get_atom_features( atom ):
    atom_features = {}
    atom_features['type'] = atom.GetAtomicNum()                                 # int
    atom_features['aromatic'] = int(atom.GetIsAromatic())                       # bool2int
    atom_features['Hs_num'] = atom.GetTotalNumHs(includeNeighbors=True)         # int
    atom_features['formal_charge'] = atom.GetFormalCharge()                     # int
    atom_features['chirality'] = atom.GetChiralTag().numerator                  # 0:unspecified 1:CW 2:CCW 3:other
    atom_features['hybridization'] = atom.GetHybridization().numerator          # 0:unspecified 1:S 2:SP 3:SP2 4:SP3 5:SP3D 6:SP3D2
    return atom_features
    

def get_atom_features_vector( atom, type_length ):
    atom_features = get_atom_features(atom)
    # one-hot encoding
    atom_type = np.zeros(type_length) # type_length决定了原子类型最大数量
    atom_type[atom_features['type']-1] = 1
    atom_Hs = np.zeros(5)
    atom_Hs[atom_features['Hs_num']] = 1
    atom_formal_charge = np.zeros(5)
    atom_formal_charge[atom_features['formal_charge']] = 1
    atom_chirality = np.zeros(4)
    atom_chirality[atom_features['chirality']] = 1
    atom_hybridization = np.zeros(6)
    atom_hybridization[atom_features['hybridization']-1] = 1
    
    atom_features_vector = np.concatenate((atom_type,
                                           np.array([atom_features['aromatic']]),
                                           atom_Hs,
                                           atom_formal_charge,
                                           atom_chirality,
                                           atom_hybridization
                                           ),axis=0)
    return atom_features_vector


def get_atoms_features_matrix( mol, type_length = 100 ):
    atoms_features_matrix = []
    for atom in mol.GetAtoms():
        fv = get_atom_features_vector(atom, type_length)
        atoms_features_matrix.append(fv)
    atoms_features_matrix = np.array(atoms_features_matrix)
    # # 在原子属性的最后一项加上了该原子的部分电荷
    # contribs = get_atoms_partial_charge(mol)
    # atoms_features_matrix = np.concatenate([atoms_features_matrix,
    #                                         contribs.reshape(-1,1)], axis=1)
    
    return atoms_features_matrix 

def get_atoms_partial_charge( mol ):
    AllChem.ComputeGasteigerCharges(mol)
    contribs = np.array([mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())])
    return contribs.reshape(-1,1)

def get_bond_features( bond ):
    bond_features = {}
    tp = 4 if bond.GetBondType().numerator == 12 else bond.GetBondType().numerator
    bond_features['type'] = tp if tp in (0,1,2,3,4) else 5                      # 0:unspecified 1:single 2:double 3:triple 4:aromatic 5:other
    bond_features['conjugation'] = int(bond.GetIsConjugated())                  # bool2int
    bond_features['ring'] = int(bond.IsInRing())                                # bool2int
    bond_features['stereo'] = bond.GetStereo().numerator                        # 0:one 1:any 2:Z 3:E 4:cic 5:trans
    return bond_features    

def get_bond_features_vector( bond ):
    bond_features = get_bond_features(bond)
    # one-hot encoding
    bond_type = np.zeros(5)
    bond_type[bond_features['type']-1] = 1
    bond_stereo = np.zeros(6)
    bond_stereo[bond_features['stereo']] = 1
    
    bond_features_vector = np.concatenate((bond_type,
                                           np.array([bond_features['conjugation']]),
                                           np.array([bond_features['ring']]),
                                           bond_stereo
                                           ),axis=0)
    return bond_features_vector

def get_bonds_features_matrix( mol ):

    bonds_features_matrix = []
    for bond in mol.GetBonds():
        op = bond.GetBeginAtomIdx()
        ed = bond.GetEndAtomIdx()
        fv = get_bond_features_vector(bond)
        bonds_features_matrix.append(([op,ed], fv))
        bonds_features_matrix.append(([ed,op], fv))
    bonds_features_matrix = sorted(bonds_features_matrix)
    bonds_indices = [i[0] for i in bonds_features_matrix]
    bonds_features_matrix = [i[1] for i in bonds_features_matrix]   
    return np.array(bonds_features_matrix), np.array(bonds_indices)



def comps_visualize(mol, comps,
                    tars=None, edge_index=None,
                    size=(500, 500), label='',
                    count_in_node_edges=True,
                    multi_fig=True, form='png'):
    assert form in ['png', 'svg']
    if multi_fig:
        return comps_visualize_multi(mol, comps,
                                     tars=tars, edge_index=edge_index,
                                     size=size, label=label,
                                     count_in_node_edges=count_in_node_edges,
                                     form = form)
    else:
        return comps_visualize_single(mol, comps,
                                      tars=tars, edge_index=edge_index,
                                      size=size, label=label,
                                      count_in_node_edges=count_in_node_edges,
                                      form = form)



def comps_visualize_multi(mol, comps,
                          tars=None, edge_index=None,
                          size=(500, 500), label='',
                          count_in_node_edges=True, form='png'):
   
    if tars is not None:
        assert edge_index is not None, "edge_index is needed when using edge target information"
        tars = [x.T.cpu().tolist() for x in tars]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.T.cpu().tolist()
    
            
    if form == 'png':
        drawer = rdMolDraw2D.MolDraw2DCairo
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG
        
    comps = [x.cpu().numpy().astype(np.int32) for x in comps]
    atoms = mol.GetNumAtoms()
    imgs = []
    s_nodes = []
    groups = []
    tar_last = []
    for layer, comp in enumerate(comps):
        if tars is not None:
            tar = tars[layer]
        if layer != 0:
            temp = comp
            comp = np.zeros(atoms, dtype=np.int32)
            """
            affine s small size comp (post layers) to atoms_num size
            """
            for i in range(temp.size):
                comp[groups[i]] = temp[i]
            if tars is not None:
                tar = [[groups[i], groups[j]] for i,j in tar]
                tar = [list(product(x[0], x[1])) for x in tar]
                tar = list(map(list, chain.from_iterable(tar)))
                tar = [x for x in tar if x in edge_index]
                if count_in_node_edges:
                    tar = tar + tar_last
                    
        value, counts = np.unique(comp, return_counts=True)
        s_nodes = value[counts > 1]
        groups = []
        groups_e = []
        for s_node in s_nodes:
            group = np.where(comp == s_node)[0].tolist()
            groups.append(group)
            group_e = []
            for atom_pair in combinations(group, 2):
                bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
                if bond is not None:
                    if tars is not None:  
                        if list(atom_pair) in tar:
                            group_e.append(bond.GetIdx())
                    else:
                        group_e.append(bond.GetIdx())
            groups_e.append(group_e)
            
        atom_cols = {}
        bond_cols = {}
        atom_list = []
        bond_list = []
        for i, (hit_atom, hit_bond) in enumerate(zip(groups, groups_e)):
            for at in hit_atom:
                atom_cols[at] = colors[i % len(colors)]
                atom_list.append(at)
            for bd in hit_bond:
                bond_cols[bd] = colors[i % len(colors)]
                bond_list.append(bd)
        d = drawer(size[0], size[1])
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_list,
                                            highlightAtomColors=atom_cols,
                                            highlightBonds=bond_list,
                                            highlightBondColors=bond_cols)
        d.FinishDrawing()
        img=d.GetDrawingText()
        imgs.append(img)        
        """
        pre-process for next layer: set single atoms as independent groups
        and insert in LIST: groups
        """
        diff = list(set(range(atoms))^set(atom_list))
        for i in diff:
            groups.insert(comp[i], [i])
        tar_last = tar[:] 
    return imgs


def comps_visualize_single(mol, comps,
                           tars=None, edge_index=None,
                           size=(500, 500), label='',
                           count_in_node_edges=True, form='png'):
    if tars is not None:
        assert edge_index is not None, "edge_index is needed when using edge target information"
        tars = [x.T.cpu().tolist() for x in tars]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.T.cpu().tolist()
    
    if form == 'png':
        drawer = rdMolDraw2D.MolDraw2DCairo
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG
        
    comps = [x.cpu().numpy().astype(np.int32) for x in comps]
    
    atoms = mol.GetNumAtoms()
    atom_cols = {}
    bond_cols = {}

    s_nodes = []
    groups = []
    tar_last = []
    op_color = 0
    for layer, comp in enumerate(comps):
        if tars is not None:
            tar = tars[layer]
        if layer != 0:
            temp = comp
            comp = np.zeros(atoms, dtype=np.int32)
            """
            affine s small size comp (post layers) to atoms_num size
            """
            for i in range(temp.size):
                comp[groups[i]] = temp[i]
                    
            if tars is not None:
                tar = [[groups[i], groups[j]] for i,j in tar]
                tar = [list(product(x[0], x[1])) for x in tar]
                tar = list(map(list, chain.from_iterable(tar)))
                tar = [x for x in tar if x in edge_index]
                if count_in_node_edges:
                    tar = tar + tar_last
        value, counts = np.unique(comp, return_counts=True)
        s_nodes = value[counts > 1]
        groups = []
        groups_e = []
        for s_node in s_nodes:
            group = np.where(comp == s_node)[0].tolist()
            groups.append(group)
            group_e = []
            for atom_pair in combinations(group, 2):
                bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
                if bond is not None:
                    if tars is not None:  
                        if list(atom_pair) in tar:
                            group_e.append(bond.GetIdx())
                    else:
                        group_e.append(bond.GetIdx())
                    
            groups_e.append(group_e)
            
        atom_list = []
        for i, (hit_atom, hit_bond) in enumerate(zip(groups, groups_e)):
            for at in hit_atom:
                if at in atom_cols:
                    atom_cols[at].append(colors[(i + op_color) % len(colors)])
                else:
                    atom_cols[at] = [colors[(i + op_color) % len(colors)]]
                atom_list.append(at)
            for bd in hit_bond:
                if bd in bond_cols:
                    bond_cols[bd].append(colors[(i + op_color) % len(colors)])
                else:
                    bond_cols[bd] = [colors[(i + op_color) % len(colors)]]
        try:
            op_color += i + 1
        except:
            print("Null extraction: block {:d}".format(layer))
        """
        pre-process for next layer: set single atoms as independent groups
        and insert in LIST: groups
        """
        diff = list(set(range(atoms))^set(atom_list))
        for i in diff:
            groups.insert(comp[i], [i])
        tar_last = tar[:] 
    
    h_rads = {}
    h_lw_mult = {}
    d = drawer(size[0], size[1])
    dos = d.drawOptions()
    dos.atomHighlightsAreCircles = False
    dos.fillHighlights = False
    dos.bondLineWidth = 2
    dos.scaleBondWidth = True
    d.DrawMoleculeWithHighlights(mol, label, atom_cols, bond_cols, h_rads, h_lw_mult, -1)
    d.FinishDrawing()
    img=d.GetDrawingText()
    return img








































def mol_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []
    bonds = []
    decomp_1 = []
    decomp_2 = []
    res = [list(x[0]) for x in list(BRICS.FindBRICSBonds(mol))]
    res_bonds = sorted([[single_bond, [single_bond[1], single_bond[0]]] for single_bond in res])
    ccs = []
    for bond in mol.GetBonds():
        op_atom = bond.GetBeginAtom()
        ed_atom = bond.GetEndAtom()
        op = op_atom.GetIdx()
        ed = ed_atom.GetIdx()
        bonds.append([op, ed])
        bonds.append([ed, op])
        if op_atom.GetAtomicNum() == 6 and ed_atom.GetAtomicNum() == 6:
            ccs.append([op, ed])
            ccs.append([ed, op])
        
    bonds = sorted(bonds)
    all_bonds = bonds[:]
    for bond in res:
        if bond in bonds:
            bonds.remove(bond)
        if [bond[1], bond[0]] in bonds:
            bonds.remove([bond[1], bond[0]])    
            
    # function that spilt bond clusters
    def merge_cliques(bonds):
        bonds = [np.array([x]) for x in bonds]
        for i in range(len(bonds)-1):
            if i >= len(bonds):
                break
            for j in range(i+1, len(bonds)):
                if j >= len(bonds):
                    break
                if len(set(bonds[i].flatten()) & set(bonds[j].flatten())) > 0:
                    bonds[i] = np.append(bonds[i], bonds[j], axis=0)
                    bonds[j] = np.array([])
            bonds = [x for x in bonds if len(x) > 0]
        bonds = [x for x in bonds if len(x) > 0]
        return [x.tolist() for x in bonds]
    
    decomp_2 = merge_cliques(bonds)
    res_ex = []
    # break bonds between rings and non-ring atoms
    for bond in bonds:
        if mol.GetAtomWithIdx(bond[0]).IsInRing() and not mol.GetAtomWithIdx(bond[1]).IsInRing():
            res_ex.append([bond, [bond[1], bond[0]]])
            bonds.remove(bond)
            bonds.remove([bond[1], bond[0]])

    # split clusters that an atom connects more than 2 neighbors        
    for atom in mol.GetAtoms():
        op = atom.GetIdx()
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            # if op not in np.array(res).flatten() and op not in np.array(res_ex).flatten():
            for nei in atom.GetNeighbors():
                ed = nei.GetIdx()
                if [op, ed] in bonds:
                    res_ex.append([[op, ed], [ed, op]])
                    bonds.remove([op, ed])
                    bonds.remove([ed, op])
                    
    decomp_1 = merge_cliques(bonds)
    # for single_bond in res_ex:
    #     decomp_1.append(single_bond)
    decomp_1 += res_ex
    decomp_1 = sorted(decomp_1)
    decomp_2 = sorted(decomp_2)
    return decomp_1, decomp_2, res_bonds, all_bonds, ccs


def draw_decomp(mol, decomp, size=(500, 200)):
    np.random.seed(777) 
    colors = [(255, 182, 193),  (255, 192, 203),  (220, 20, 60), (219, 112, 147),  (255, 105, 180),  (255, 20, 147),  (199, 21, 133),  (218, 112, 214),  (216, 191, 216),  (221, 160, 221),  (238, 130, 238),  (255, 0, 255),  (255, 0, 255),  (139, 0, 139),  (128, 0, 128),  (186, 85, 211),  (148, 0, 211),  (153, 50, 204),  (75, 0, 130),  (138, 43, 226),  (147, 112, 219),  (123, 104, 238),  (106, 90, 205),  (72, 61, 139),  (0, 0, 255),  (0, 0, 205),  (25, 25, 112),  (0, 0, 139),  (0, 0, 128),  (65, 105, 225),  (100, 149, 237),  (176, 196, 222),  (119, 136, 153),  (112, 128, 144),  (30, 144, 255),  (70, 130, 180),  (135, 206, 250),  (135, 206, 235),  (0, 191, 255),  (173, 216, 230),  (176, 224, 230),  (95, 158, 160),  (175, 238, 238),  (0, 255, 255),  (212, 242, 231),  (0, 206, 209),  (47, 79, 79),  (0, 139, 139),  (0, 128, 128),  (72, 209, 204),  (32, 178, 170),  (64, 224, 208),  (127, 255, 170),  (0, 250, 154),  (0, 255, 127),  (60, 179, 113),  (46, 139, 87),  (144, 238, 144),  (152, 251, 152),  (143, 188, 143),  (50, 205, 50),  (0, 255, 0),  (34, 139, 34),  (0, 128, 0),  (0, 100, 0),  (127, 255, 0),  (124, 252, 0),  (173, 255, 47),  (85, 107, 47),  (250, 250, 210),  (255, 255, 0),  (128, 128, 0),  (189, 183, 107),  (255, 250, 205),  (238, 232, 170),  (240, 230, 140),  (255, 215, 0),  (218, 165, 32),  (245, 222, 179),  (255, 228, 181),  (255, 165, 0),  (255, 239, 213),  (255, 235, 205),  (255, 222, 173),  (250, 235, 215),  (210, 180, 140),  (222, 184, 135),  (255, 228, 196),  (255, 140, 0),  (250, 240, 230),  (205, 133, 63),  (255, 218, 185),  (244, 164, 96),  (210, 105, 30),  (139, 69, 19),  (160, 82, 45),  (255, 160, 122),  (255, 127, 80),  (255, 69, 0),  (233, 150, 122),  (255, 99, 71),  (250, 128, 114),  (240, 128, 128),  (188, 143, 143),  (205, 92, 92),  (255, 0, 0),  (165, 42, 42),  (178, 34, 34),  (139, 0, 0),  (128, 0, 0)]
    colors = np.array(colors)/255
    colors = colors[np.random.permutation(colors.shape[0])]
    colors = [tuple(x) for x in colors]
    atom_cols = {}
    bond_cols = {}
    atom_list = []
    bond_list = []
    for i, bond_group in enumerate(decomp):
        hit_atom = np.unique(np.array(bond_group).flatten()).tolist()
        hit_bond = [mol.GetBondBetweenAtoms(x[0], x[1]).GetIdx() for x in bond_group]
        hit_bond = np.unique(np.array(hit_bond).flatten()).tolist()
        for at in hit_atom:
            atom_cols[at] = colors[i % len(colors)]
            atom_list.append(at)
        for bd in hit_bond:
            bond_cols[bd] = colors[i % len(colors)]
            bond_list.append(bd)
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol,
                                       # highlightAtoms=atom_list,
                                       # highlightAtomColors=atom_cols,
                                       highlightBonds=bond_list,
                                       highlightBondColors=bond_cols)
    d.FinishDrawing()
    png=d.GetDrawingText()
    return png


def get_decomp_mask(mol, label=False):
    decomp_1, decomp_2, res_bonds, all_bonds, ccs = mol_decomp(mol)
    scaffold = mol.GetSubstructMatches(MurckoScaffold.GetScaffoldForMol(mol))
    scaffolds = list(chain.from_iterable(scaffold))
    reses = list(chain.from_iterable(res_bonds))
    label_1 = np.ones(len(all_bonds), dtype=np.int32)
    label_2 = np.ones(len(all_bonds), dtype=np.int32)
    for i, bond in enumerate(all_bonds):        
        if bond[0] in scaffolds and bond[1] in scaffolds:
            label_1[i] = 0  
            label_2[i] = 0
        elif bond in reses or bond in ccs:
            label_1[i] = 0
    labels = np.vstack((label_1, label_2)).T      
    def decomp2mask(decomp):
        mask = []
        for i, bonds in enumerate(decomp):
            for bond in bonds:
                mask.append((bond, i))
        return [x[1] for x in sorted(mask)]
     
    mask_1 = np.array(decomp2mask(sorted(decomp_1 + res_bonds)))
    mask_2 = np.array(decomp2mask(sorted(decomp_2 + res_bonds))) # notice that mask here has shape (-1, 2), where mask[:, 0] is bond mask and mask[:, 1] is bond label
    masks = np.vstack((mask_1, mask_2)).T
    if label:
        return np.hstack((masks, labels)) # where [:, 0] and [:, 1] are masks and [:, 2] is label
    return masks


def get_decomp_labels(mol):
    n_atoms = mol.GetNumAtoms()
    assert n_atoms != 1
    bonds = []
    res = [list(x[0]) for x in list(BRICS.FindBRICSBonds(mol))]
    res_bonds = sorted([[single_bond, [single_bond[1], single_bond[0]]] for single_bond in res])
    ccs = []
    for bond in mol.GetBonds():
        op_atom = bond.GetBeginAtom()
        ed_atom = bond.GetEndAtom()
        op = op_atom.GetIdx()
        ed = ed_atom.GetIdx()
        bonds.append([op, ed])
        bonds.append([ed, op])
        if op_atom.GetAtomicNum() == 6 and ed_atom.GetAtomicNum() == 6:
            ccs.append([op, ed])
            ccs.append([ed, op])
    all_bonds = sorted(bonds)
    scaffold = mol.GetSubstructMatches(MurckoScaffold.GetScaffoldForMol(mol))
    scaffolds = list(chain.from_iterable(scaffold))
    reses = list(chain.from_iterable(res_bonds))
    label_1 = np.ones(len(all_bonds), dtype=np.int32)
    label_2 = np.ones(len(all_bonds), dtype=np.int32)
    for i, bond in enumerate(all_bonds):        
        if bond[0] in scaffolds and bond[1] in scaffolds:
            label_1[i] = 0  
            label_2[i] = 0
        elif bond in reses or bond in ccs:
            label_1[i] = 0
    labels = np.vstack((label_1, label_2)).T 
    return labels




































































# def get_fungroups( file='./group/FunctionalGroups2.csv' ):
#     df = pd.read_csv(file)
#     return [Chem.MolFromSmarts(x) for x in df['smarts'].values], df['label'].values

# def get_pool_labels( mol, bonds_indices, groups, labels, hierarchy=[0, 2] ):
#     """
#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.mol
#     bonds_indices: Array
#     groups : list
#     labels : list
#     hierarchy : list, optional
#         for example, hierarchy = [0, 2, 4] means that there are 3 pooling layers
#         the 1st layer considers those groups with num_bonds >0 and <=2
#         the 2nd layer considers those groups with num_bonds >2 and <=4
#         the 3rd layer considers those groups with num_bonds >=4
#         if your model has only one pooling layer, you can set hierarchy = [0]

#     Returns
#     -------
#     bond_labels : Array
#         np.array with size (len(bonds_indices), len(hierarchy))
#         saves the pooling labels of each bonds (dim 0) in hierarchical layers (dim 1)
#         -1 means no pooling labels
#     """
#     groups = np.array(groups, dtype=object)
#     labels = np.array(labels, dtype=np.int64)
#     group_bond_num = np.array([x.GetNumBonds() for x in groups])
#     subs = []
#     for i in range(1, len(hierarchy)):
#         mask = np.logical_and(group_bond_num > hierarchy[i-1],
#                               group_bond_num <= hierarchy[i])
#         subs_i = [mol.GetSubstructMatches(x) for x in groups[mask]]
#         subs_i = [list(x) for x in subs_i]
#         for i in range(len(subs_i)):
#             for j in range(len(subs_i[i])):
#                 subs_i[i][j] = [subs_i[i][j], labels[mask][i]]
#         subs.append([x for x in subs_i if x])
        
#     mask = group_bond_num > hierarchy[-1]    
#     subs_i = [mol.GetSubstructMatches(x) for x in groups[mask]]
#     subs_i = [list(x) for x in subs_i]
#     for i in range(len(subs_i)):
#         for j in range(len(subs_i[i])):
#             subs_i[i][j] = [subs_i[i][j], labels[mask][i]]
#     subs.append([x for x in subs_i if x])
    
#     bond_labels = -np.ones([len(bonds_indices), len(hierarchy)])    
#     for i in range(len(subs)):
#         subs_list = list(chain.from_iterable(subs[i]))
#         for j in range(len(bonds_indices)):
#             bond_label = is_bond_in_subs(bonds_indices[j], subs_list)
#             if bond_label is not None:
#                 bond_labels[j][i] = bond_label
#     return bond_labels
            

# def is_bond_in_subs(bond, subs):
#     for sub in subs:
#         if bond[0] in sub[0] and bond[1] in sub[0]:
#             return np.array(sub[1])
#     return None





















































