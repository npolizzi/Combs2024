from collections import defaultdict
from .clashfilter import df_ideal_ala, rel_coords_dict, Clash, ClashVDM, make_pose_df, \
    backbone_str, Contact, make_df_corr, VdmReps, rec_dd
import pickle
import numpy as np
from .transformation import get_rot_trans
from prody import calcPhi, calcPsi, writePDB, AtomGroup, getDihedral
from sklearn.neighbors import NearestNeighbors
from .convex_hull import AlphaHull
from numba import jit
import time
import os
import copy
import random
import itertools
from scipy.spatial.distance import cdist
from functools import partial
from multiprocessing import Pool
import gzip
from os import path
import pandas as pd
from .constants import coords_cols
from .dataframe import make_df_from_prody
from .functions import get_ABPLE


_dir = os.path.dirname(__file__)
path_to_ideal_ala = os.path.join(_dir, '../files/ideal_alanine_bb_only.pkl')


def load_ideal_ala():
    return pd.read_pickle(path_to_ideal_ala)


df_ideal_ala = dict()
load_ideal(df_ideal_ala, 'SC')
load_ideal(df_ideal_ala, 'HNCA')
load_ideal(df_ideal_ala, 'CO')
load_ideal(df_ideal_ala, 'PHI_PSI')


df_ideal_ala_atoms = dict(SC=['HA', 'C', 'CA', 'N'],
                          HNCA=['HA', 'H', 'CA', 'N'],
                          CO=['HA', 'C', 'O'],
                          PHI_PSI=['HA', 'C', 'CA', 'N'])


# Have Lig dataframe.  Need corr dataframe for each iFG.  For each iFG, grab
# coords from Lig df and look up in iFG NN.


class Template:

    def __init__(self, pdb):
        self.pdb = pdb  # pdb should be prody object poly-gly with CA hydrogens for design.
        self.dataframe = make_df_from_prody(self.pdb)
        self.alpha_hull = None
        self.phi_psi_dict = dict()
        self.set_phi_psi()

    @staticmethod
    def get_bb_sel(pdb):
        return pdb.select(backbone_str).copy()

    def set_phi_psi(self):
        for s in set(self.dataframe.seg_chain_resnum):
            self.phi_psi_dict[s] = tuple(self.get_phi_psi(s[0], s[1], s[2]))

    @staticmethod
    def calc_phi(res_bef, res):
        """
        res_bef and res are pre-sorted by N, Ca, C

        Parameters
        ----------
        res_bef
        res

        Returns
        -------

        """
        cm1 = res_bef[res_bef['name'] == 'C'][['c_x', 'c_y', 'c_z']].values
        c = res[['c_x', 'c_y', 'c_z']].values
        return getDihedral(cm1, c[0,:], c[1,:], c[2,:], radian=False)

    @staticmethod
    def calc_psi(res, res_aft):
        """
        res_aft and res are pre-sorted by N, Ca, C

        Parameters
        ----------
        res
        res_aft

        Returns
        -------

        """
        cp1 = res_aft[res_aft['name'] == 'N'][['c_x', 'c_y', 'c_z']].values
        c = res[['c_x', 'c_y', 'c_z']].values
        return getDihedral(c[0,:], c[1,:], c[2,:], cp1, radian=False)

    def set_phi_psi_abple(self):
        cols = ['segment', 'chain', 'resnum', 'resname', 'name', 'c_x', 'c_y', 'c_z']
        df = self.dataframe[cols]
        df_name_order = pd.DataFrame(dict(name=['N', 'CA', 'C']))
        df = pd.merge(df_name_order, df, on='name')  # sorts df by N, Ca, C atoms
        df.sort_values('name', inplace=True)
        data = []
        for seg, g_seg in df.groupby('segment'):
            for ch, g_ch in g_seg.groupby('chain'):
                resnums = sorted(set(g_ch.resnum))
                gs_rn = g_ch.groupby('resnum')
                for i, rn in enumerate(resnums):
                    g_rn = gs_rn.get_group(rn)
                    if rn - 1 in resnums:
                        phi = self.calc_phi(gs_rn.get_group(rn - 1), g_rn)
                    else:
                        phi = np.nan
                    if rn + 1 in resnums:
                        psi = self.calc_psi(g_rn, gs_rn.get_group(rn + 1))
                    else:
                        psi = np.nan
                    resname = g_rn.resname.iat[0]
                    abple = get_ABPLE(resname, phi, psi)
                    data.append((seg, ch, rn, phi, psi, abple))
        cols_phipsi = ['segment', 'chain', 'resnum', 'phi', 'psi', 'ABPLE']
        df_phipsi = pd.DataFrame(data, columns=cols_phipsi)
        self.dataframe = pd.merge(self.dataframe, df_phipsi, on=['segment', 'chain', 'resnum'])

    def get_phi_psi(self, seg, chain, resnum):
        res = self.pdb[seg, chain, resnum]
        
        try:
            phi = calcPhi(res)
        except ValueError:
            phi = None
        
        try:
            psi = calcPsi(res)
        except ValueError:
            psi = None
            
        return phi, psi

    def set_alpha_hull(self, pdb_w_CB, alpha=9):
        self.pdb_w_CB = pdb_w_CB
        self.alpha_hull = AlphaHull(alpha)
        self.alpha_hull.set_coords(pdb_w_CB)
        self.alpha_hull.calc_hull()


class Design:

    def __init__(self):
        self.interaction_csts = None  # list of csts per iFG.


# @jit("f8[:,:](f8[:,:], f8, f8[:,:], f8)", nopython=True)
# def rottrans(x, m_com, R, t_com):
#     return np.dot((x - m_com), R) + t_com

# vecsd1 = ['vec_don1_x', 'vec_don1_y', 'vec_don1_z']
# vecsd2 = ['vec_don2_x', 'vec_don2_y', 'vec_don2_z']
# vecsd3 = ['vec_don3_x', 'vec_don3_y', 'vec_don3_z']
# vecsacc = ['vec_acc_x',  'vec_acc_y', 'vec_acc_z']

class Load:
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.path = kwargs.get('path', './')  # path to sig reps
        # self.designable = list()  # list of tuples (segment, chain, residue number)
        self.sequence_csts = kwargs.get('sequence_csts') # keys1 are tuples (seq, ch, #), keys2 are label,
                                               # vals are allowed residue names (three letter code).
        self.dataframe = pd.DataFrame()
        self.dataframe_grouped = None
        self._rot = defaultdict(dict)
        self._mobile_com = defaultdict(dict)
        self._target_com = defaultdict(dict)
        self._sig_reps = defaultdict(dict)
        self._ideal_ala_df = defaultdict(dict)
        self._nonclashing = list()
        self.filetype = kwargs.get('filetype', '.feather')
        self.remove_from_df = kwargs.get('remove_from_df') # e.g. {1: {'chain': 'Y', 'name': 'CB', 'resname': 'ASN'},
                                                           #       2: {'chain': 'Y', 'name': 'CG', 'resname': 'GLN'}}

    @staticmethod
    def _get_targ_coords(template, label, seg, chain, resnum):
        sel_str = 'segment ' + seg + ' chain ' + chain + ' resnum ' + str(resnum) + ' name '
        cs = []
        for n in rel_coords_dict[label]:
            try:
                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
            except AttributeError:
                try:
                    cs = []
                    for n in ['N', '1H', 'CA']:
                        cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                    return np.stack(cs)
                except AttributeError:
                    try:
                        cs = []
                        for n in ['N', 'H1', 'CA']:
                            cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                        return np.stack(cs)
                    except AttributeError:
                        sel_str = 'chain ' + chain + ' resnum ' + str(resnum) + ' name '
                        cs = []
                        for n in rel_coords_dict[label]:
                            try:
                                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                            except AttributeError:
                                cs = []
                                for n in ['N', '1H', 'CA']:
                                    cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                                return np.stack(cs)
                        return np.stack(cs)
        return np.stack(cs)

        # return np.array([template.pdb.select(sel_str + n).getCoords()[0]
        #                 for n in rel_coords_dict[label]])

    @staticmethod
    def _get_mob_coords(df, label):
        # cs = []
        # for n in rel_coords_dict[label]:
        #     try:
        #         cs.append(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten())
        #     except AttributeError:
        #         cs = []
        #         for n in ['N', 'H1', 'CA']:
        #             cs.append(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten())
        # return np.stack(cs)
        return np.stack(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten()
                        for n in rel_coords_dict[label])

    def set_rot_trans(self, template):
        for seg, chain, resnum in self.sequence_csts.keys():
            for label, df in df_ideal_ala.items():
                mob_coords = self._get_mob_coords(df, label)
                targ_coords = self._get_targ_coords(template, label, seg, chain, resnum)
                R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
                self._rot[label][(seg, chain, resnum)] = R
                self._mobile_com[label][(seg, chain, resnum)] = m_com
                self._target_com[label][(seg, chain, resnum)] = t_com
                df_ = df.copy()
                df_[['c_x', 'c_y', 'c_z']] = np.dot(df_[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
                self._ideal_ala_df[label][(seg, chain, resnum)] = df_

    def _import_sig_reps(self):
        labels_resns = defaultdict(set)
        for tup in self.sequence_csts.keys():
            for label in self.sequence_csts[tup].keys():
                labels_resns[label] |= set(self.sequence_csts[tup][label])
        for label in labels_resns.keys():
            for resn in labels_resns[label]:
                reppath = self.path + label + '/' + resn + self.filetype
                try:
                    if self.filetype == '.feather':
                        # print(label, resn)
                        self._sig_reps[label][resn] = pd.read_feather(reppath)
                    elif self.filetype == '.pkl':
                        with open(reppath, 'rb') as infile:
                            self._sig_reps[label][resn] = pickle.load(infile)
                except FileNotFoundError:
                    # print('hmm')
                    pass

    # def _import_additional_sig_rep(self, path, label, resn):
    #     try:
    #         with open(path + label + '/' + resn + '.pkl', 'rb') as infile:
    #             self._sig_reps[label][resn] = pickle.load(infile)
    #     except FileNotFoundError:
    #         pass

    @staticmethod
    def _get_phi_psi_df(df, phi, psi, phipsi_width=60):
        if phi is not None:
            phi_high = df['phi'] < (phi + (phipsi_width / 2))
            phi_low = df['phi'] > (phi - (phipsi_width / 2))
        else:
            phi_high = np.array([True] * len(df))
            phi_low = phi_high
        if psi is not None:
            psi_high = df['psi'] < (psi + (phipsi_width / 2))
            psi_low = df['psi'] > (psi - (phipsi_width / 2))
        else:
            psi_high = np.array([True] * len(df))
            psi_low = psi_high
        return df[phi_high & phi_low & psi_high & psi_low]

    @staticmethod
    def chunk_df(df_gr, gr_chunk_size=100):
        grs = list()
        for i, (n, gr) in enumerate(df_gr):
            grs.append(gr)
            if (i + 1) % gr_chunk_size == 0:
                yield pd.concat(grs)
                grs = list()

    def _load(self, template, seg, chain, resnum, **kwargs):
        # dataframe = pd.DataFrame()
        # t0 = time.time()
        phipsi_width = kwargs.get('phipsi_width', 60)

        dfs = list()
        for label in self.sequence_csts[(seg, chain, resnum)].keys():
            print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
            if label == 'PHI_PSI':
                df_list = list()
                phi, psi = template.get_phi_psi(seg, chain, resnum)
                for resn in self.sequence_csts[(seg, chain, resnum)][label]:
                    df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
                                                     phi, psi, phipsi_width)
                    df_list.append(df_phipsi)
                df = pd.concat(df_list)
            else:
                df = pd.concat([self._sig_reps[label][resn]
                                for resn in self.sequence_csts[(seg, chain, resnum)][label]])

            if self.remove_from_df is not None:
                for d in self.remove_from_df.values():
                    tests = []
                    for col, val in d.items():
                        tests.append(df[col] == val)
                    tests = np.array(tests).T
                    tests = tests.all(axis=1)
                    df = df.loc[~tests]

            m_com = self._mobile_com[label][(seg, chain, resnum)]
            t_com = self._target_com[label][(seg, chain, resnum)]
            R = self._rot[label][(seg, chain, resnum)]
            print('transforming coordinates...')
            # df['coords'] = df['coords'].transform(lambda x: np.dot((x - m_com), R) + t_com)
            # df['coords'] = rottrans(np.stack(df.coords), m_com, R, t_com)
            # df['coords'] = list(np.dot(np.stack(df.coords) - m_com, R) + t_com)
            # df[['c_x', 'c_y', 'c_z']] = np.dot(df[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
            df[coords[:3]] = np.dot(df[coords[:3]] - m_com, R) + t_com
            df[coords[3:6]] = np.dot(df[coords[3:6]] - m_com, R) + t_com
            df[coords[6:9]] = np.dot(df[coords[6:9]] - m_com, R) + t_com
            df[coords[9:12]] = np.dot(df[coords[9:12]] - m_com, R) + t_com
            df[coords[12:15]] = np.dot(df[coords[12:15]] - m_com, R) + t_com
            df[coords[15:18]] = np.dot(df[coords[15:18]] - m_com, R) + t_com
            df[coords[18:21]] = np.dot(df[coords[18:21]] - m_com, R) + t_com
            df[coords[21:]] = np.dot(df[coords[21:]] - m_com, R) + t_com
            # df.loc[df['vec_acc_x'].notna(), vecsacc] = np.dot(df[vecsacc].dropna().values, R)
            # df.loc[df['vec_don1_x'].notna(), vecsd1] = np.dot(df[vecsd1].dropna().values, R)
            # df.loc[df['vec_don2_x'].notna(), vecsd2] = np.dot(df[vecsd2].dropna().values, R)
            # df.loc[df['vec_don3_x'].notna(), vecsd3] = np.dot(df[vecsd3].dropna().values, R)
            df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
            df['seg_chain_resnum_'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            # df['seg_chain_resnum'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            ###NEW STUFF FOR CLASH FILTER TEST
            df['str_index'] = df['iFG_count'] + '_' + df['vdM_count'] + '_' + df['query_name'] + '_' + df['seg_chain_resnum_']
            # df['hash'] = df['str_index'].apply(hash)
            print('making transformed dataframe...')
            dfs.append(df)
            # dataframe = pd.concat((dataframe, df), sort=False)
        dataframe = pd.concat(dfs, sort=False, ignore_index=True)
    
        print('removing clashes...')
        df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
        self._nonclashing.append(df_nonclash)
        # self._nonclashing.append(dataframe)  ##CHANGE previous 3 lines BACK!!
        # tf = time.time()
        # print('loaded (' + seg + ', ' + chain + ', ' + str(resnum) + ') in ' + str(tf-t0) + ' seconds.')
        # print('concatenating non-clashing to dataframe')
        # self.dataframe = pd.concat((self.dataframe, df_nonclash))
        # self._set_grouped_dataframe()
    
        ###### IF WANT TO CHUNK DATAFRAME, UNCOMMENT below code and COMMENT above block ###
        # print('removing clashes...')
        # dataframe_gr = dataframe.groupby(['iFG_count', 'vdM_count',
        #                                   'query_name', 'seg_chain_resnum'])
        #
        # for df_chnk in self.chunk_df(dataframe_gr, gr_chunk_size=500):
        #     df_nonclash = self._remove(df_chnk, template, seg, chain, resnum, **kwargs)
        #     print('concatenating non-clashing chunk to dataframe')
        #     self.dataframe = pd.concat((self.dataframe, df_nonclash), ignore_index=True)
        # self._set_grouped_dataframe()

    # @staticmethod
    # def _remove(dataframe, template, seg, chain, resnum, **kwargs):
    #     cla = Clash(df=template.dataframe, df_query=dataframe,
    #                 exclude=(resnum, chain, seg), **kwargs)
    #     cla.find()
    #     return cla.nonclashing_vdms
    @staticmethod
    def _remove(dataframe, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        cla = ClashVDM(dfq=dataframe, dft=template.dataframe)
        # cla.set_grouping(['iFG_count', 'vdM_count',
        #                   'query_name', 'seg_chain_resnum'])
        # cla.set_grouping(['iFG_count', 'vdM_count', 'query_name'])  # commented out for clashfilter test
        cla.set_grouping('str_index')
        cla.set_exclude((resnum, chain, seg))
        cla.setup()
        cla.find(**kwargs)
        tf = time.time()
        print('time:', tf-t0)
        return cla.dfq_clash_free

    def load(self, template, **kwargs):
        if not self._sig_reps:
            self._import_sig_reps()

        if not self._rot:
            self.set_rot_trans(template)

        for seg, chain, resnum in self.sequence_csts.keys():
            self._load(template, seg, chain, resnum, **kwargs)

        print('concatenating non-clashing to dataframe')
        t0 = time.time()
        self.dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
        self._nonclashing = list()
        self._sig_reps = defaultdict(dict)
        tf = time.time() - t0
        print('concatenated in ' + str(tf) + ' seconds.')
        self._set_grouped_dataframe()

    def load_additional(self, template, sequence_csts, **kwargs):
        seq_csts = defaultdict(dict)
        seq_csts_copy = copy.deepcopy(self.sequence_csts)
        for seg_ch_rn in sequence_csts.keys():
            if seg_ch_rn not in self.sequence_csts.keys():
                seq_csts[seg_ch_rn] = sequence_csts[seg_ch_rn]
                seq_csts_copy[seg_ch_rn] = sequence_csts[seg_ch_rn]

        if len(seq_csts.keys()) > 0:
            self.path = kwargs.get('path', self.path)
            self.sequence_csts = seq_csts
            self._import_sig_reps()
            self.set_rot_trans(template)
            self._nonclashing = list()

            for seg, chain, resnum in self.sequence_csts.keys():
                self._load(template, seg, chain, resnum, **kwargs)

            print('concatenating non-clashing to dataframe')
            t0 = time.time()
            _dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
            self.dataframe = pd.concat((self.dataframe, _dataframe), sort=False, ignore_index=True)
            self._nonclashing = list()
            self._sig_reps = defaultdict(dict)
            tf = time.time() - t0
            print('concatenated in ' + str(tf) + ' seconds.')
            self._set_grouped_dataframe()
            self.sequence_csts = seq_csts_copy
            return True
        else:
            return False


    def _set_grouped_dataframe(self):
        # self.dataframe_grouped = self.dataframe.groupby(['iFG_count', 'vdM_count',
        #                                                  'query_name', 'seg_chain_resnum'])
        self.dataframe_grouped = self.dataframe.groupby('str_index')


class VdM(Load):
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'iFG_type')
        self.neighbors = None
        self.num_iFG_atoms = 0
        self.dataframe_iFG_coords = None
        self.neighbors = None
        self.ligand_iFG_corr_sorted = None
        self.path_to_pdbs = kwargs.get('path_to_pdbs', './')  # for printing vdMs
        if self.path_to_pdbs[-1] != '/':
            self.path_to_pdbs += '/'
        self.ligand_iFG_corr = kwargs.get('ligand_iFG_corr')  # use make_df_corr for dataframe

    # def _get_ifg_coords(self, df):
    #     df_ifg = df[df.chain == 'Y']
    #     # df_ifg_corr = pd.merge(self.ligand_iFG_corr[['resname', 'name']],
    #     #                        df_ifg, on=['resname', 'name'])
    #     df_ifg_corr = pd.merge(self.ligand_iFG_corr_sorted[['resname', 'name']],
    #                            df_ifg, on=['resname', 'name'])
    #     return np.stack(df_ifg_corr['coords']).reshape(-1)

    def _get_num_iFG_atoms(self):
        d = defaultdict(set)
        for n1, g1 in self.ligand_iFG_corr.groupby('lig_resname'):
            for n2, g2 in g1.groupby('resname'):
                d[n1] |= {len(g2)}
        for key in d.keys():
            assert len(d[key]) == 1, 'Problem with ligand iFG correspondence?'
            self.num_iFG_atoms += d[key].pop()

    def set_sorted_lig_corr(self):
        self.ligand_iFG_corr_sorted = self.ligand_iFG_corr.sort_values(by=['lig_resname', 'lig_name'])

    def set_neighbors(self, rmsd=0.4):
        if self.ligand_iFG_corr_sorted is None:
            self.set_sorted_lig_corr()

        if self.num_iFG_atoms == 0:
            self._get_num_iFG_atoms()

        #DONT NEED GROUPED BECAUSE OF INDEXING TRICK BELOW
        # if not self.dataframe_grouped:
        #     self._set_grouped_dataframe()

        df_ifg = self.dataframe[self.dataframe['chain'] == 'Y']
        df_ifg = pd.merge(self.ligand_iFG_corr_sorted[['resname', 'name']], df_ifg,
                          how='inner', on=['resname', 'name'], sort=False)
        # df_ifg = df_ifg.sort_values(by=['resname','name'])
        # df_ifg.set_index(['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum'], inplace=True)
        # df_ifg_gr = df_ifg[['c_x', 'c_y', 'c_z']].groupby(df_ifg.index, sort=False)
        # self.dataframe_iFG_coords = df_ifg_gr.apply(lambda x: x.values.flatten())
        M = int(len(df_ifg) / self.num_iFG_atoms)
        N = self.num_iFG_atoms
        R = np.arange(len(df_ifg))
        # print('M', M)
        # print('N', N)
        # print('len(df_ifg)', len(df_ifg))
        # # self.dataframe_iFG_coords = np.stack(df_ifg.coords).reshape(M, N)
        # self.neighbors = NearestNeighbors(radius=np.sqrt(self.num_iFG_atoms) * rmsd, algorithm='ball_tree')
        # self.neighbors.fit(np.stack(self.dataframe_iFG_coords))
        inds = np.array([R[i::M] for i in range(M)]).flatten()
        # print('inds.shape', inds.shape)
        # self.dataframe_iFG_coords = df_ifg[:M][['iFG_count', 'vdM_count',
        #                                         'query_name', 'seg_chain_resnum']]
        self.dataframe_iFG_coords = df_ifg[:M]['str_index']
        self.neighbors = NearestNeighbors(radius=np.sqrt(self.num_iFG_atoms) * rmsd, algorithm='ball_tree')
        self.neighbors.fit(df_ifg.iloc[inds][['c_x', 'c_y', 'c_z']].values.reshape(M, N * 3))

    def _print_vdm(self, group_name, df_group, outpath, out_name_tag='', full_fragments=False,
                   with_bb=False, bfactor_field=None):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        if not full_fragments:
            if not with_bb:
                ProdyAG(**dict(bfactor_field=bfactor_field)).print_ag(group_name, df_group, outpath, out_name_tag)
            elif with_bb:
                label = set(df_group.label).pop()
                bb_names = rel_coords_dict[label]
                seg_chain_resnum = set(df_group.seg_chain_resnum).pop()
                df_ala = self._ideal_ala_df[label][seg_chain_resnum].copy()
                df_ala['segment'] = set(df_group.segment).pop()
                df_ala['resname'] = set(df_group.resname_vdm).pop()
                df_ala_bbsel = df_ala[df_ala['name'].isin(bb_names)]
                df = pd.concat((df_group, df_ala_bbsel))
                pdag = ProdyAG(**dict(bfactor_field=bfactor_field))
                pdag.print_ag(group_name, df, outpath, out_name_tag='_' + label + out_name_tag)
        else:
            pass # update to print full fragments

    def print_vdms(self):
        pass


class ProdyAG:
    def __init__(self, **kwargs):
        self.resnums = list()
        self.names = list()
        self.resnames = list()
        self.coords = list()
        self.chids = list()
        self.segments = list()
        self.elements = list()
        self.bfactors = list()
        self.print_field_to_bfactor = kwargs.get('bfactor_field', None)
        self.ag = None
        self.df_group = None
        self.group_name = None

    def set(self, group_name, df_group):
        self.group_name = group_name
        self.df_group = df_group
        
        for n, d in df_group.iterrows():
            self.resnums.append(d['resnum'])
            name = d['name']
            self.names.append(name)
            self.resnames.append(d['resname'])
            self.coords.append(d[['c_x', 'c_y', 'c_z']])
            self.chids.append(d['chain'])
            self.segments.append(d['segment'])
            if self.print_field_to_bfactor:
                #print(self.print_field_to_bfactor)
                self.bfactors.append(d[self.print_field_to_bfactor])
            if name[0].isdigit():
                self.elements.append(name[1])
            else:
                self.elements.append(name[0])

    def set_ag(self, group_name, df_group):
        if not self.resnums:
            self.set(group_name, df_group)
        
        self.ag = AtomGroup(self.group_name)
        self.ag.setResnums(self.resnums)
        self.ag.setNames(self.names)
        self.ag.setResnames(self.resnames)
        self.ag.setCoords(np.array(self.coords, dtype='float'))
        self.ag.setChids(self.chids)
        self.ag.setSegnames(self.segments)
        self.ag.setElements(self.elements)
        if self.bfactors:
        # print(self.bfactors)
            self.ag.setBetas(self.bfactors)

    def print_ag(self, group_name, df_group, outpath, out_name_tag=''):
        if self.ag is None:
            self.set_ag(group_name, df_group)

        file = list()
        if isinstance(self.group_name, list):
            for e in self.group_name:
                if isinstance(e, tuple):
                    e = '-'.join([str(i) for i in e])
                file.append(str(e))
        else:
            file.append(str(self.group_name))
        filename = 'ag_' + '_'.join(file) + out_name_tag + '.pdb.gz'

        if outpath[-1] != '/':
            outpath += '/'
        writePDB(outpath + filename, self.ag)


class Ligand(Load):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataframe_frag_coords = None
        self.poses = list()
        self.csts = None
        self.csts_gr = None
        self.num_heavy = kwargs.get('num_heavy', 26) #34)
        self.num_total = kwargs.get('num_total', 59)
        self.percent_buried = kwargs.get('percent_buried', 0.5)
        self.isin_field = kwargs.get('isin_field', 'name')
        self.isin = kwargs.get('isin', ['C10', 'C12', 'C13', 'C8', 'C17',
                                        'C24', 'C1', 'C2', 'C3', 'C4', 'C5',
                                        'C6', 'C15', 'N5', 'C7', 'C22', 'C18',
                                        'C16', 'C14', 'C44', 'N2', 'C19',
                                        'C23', 'C20', 'C21', 'C25'])

    def set_csts(self, path_to_cst_file):
        self.csts = make_cst_df(path_to_cst_file)
        self.csts_gr = self.csts.groupby('cst_group')

    def _get_frag_coords(self, df, vdm):
        # df_corr = pd.merge(vdm.ligand_iFG_corr[['lig_resname', 'lig_name']].drop_duplicates(),
        #                    df, on=['lig_resname', 'lig_name'])
        df_corr = pd.merge(vdm.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates(), df,
                           how='inner', on=['lig_resname', 'lig_name'], sort=False)
        # df_corr = df_corr.sort_values(by=['lig_resname', 'lig_name'])
        # return np.stack(df_corr['coords']).reshape(1, -1)
        return df_corr[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)

    def _load(self, template, seg, chain, resnum, **kwargs):
        # dataframe = pd.DataFrame()
        t0 = time.time()
        phipsi_width = kwargs.get('phipsi_width', 40)

        dfs = list()
        for label in self.sequence_csts[(seg, chain, resnum)].keys():
            print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
            if label == 'PHI_PSI':
                df_list = list()
                phi, psi = template.get_phi_psi(seg, chain, resnum)
                for resn in self.sequence_csts[(seg, chain, resnum)][label]:
                    df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
                                                     phi, psi, phipsi_width)
                    df_list.append(df_phipsi)
                df = pd.concat(df_list)
            else:
                df = pd.concat([self._sig_reps[label][resn]
                                for resn in self.sequence_csts[(seg, chain, resnum)][label]])

            m_com = self._mobile_com[label][(seg, chain, resnum)]
            t_com = self._target_com[label][(seg, chain, resnum)]
            R = self._rot[label][(seg, chain, resnum)]
            print('transforming coordinates...')
            # df['coords'] = df['coords'].transform(lambda x: np.dot((x - m_com), R) + t_com)
            # df['coords'] = rottrans(np.stack(df.coords), m_com, R, t_com)
            # df['coords'] = list(np.dot(np.stack(df.coords) - m_com, R) + t_com)
            df[coords[:3]] = np.dot(df[coords[:3]] - m_com, R) + t_com
            df[coords[3:6]] = np.dot(df[coords[3:6]] - m_com, R) + t_com
            df[coords[6:9]] = np.dot(df[coords[6:9]] - m_com, R) + t_com
            df[coords[9:12]] = np.dot(df[coords[9:12]] - m_com, R) + t_com
            df[coords[12:15]] = np.dot(df[coords[12:15]] - m_com, R) + t_com
            df[coords[15:18]] = np.dot(df[coords[15:18]] - m_com, R) + t_com
            df[coords[18:21]] = np.dot(df[coords[18:21]] - m_com, R) + t_com
            df[coords[21:]] = np.dot(df[coords[21:]] - m_com, R) + t_com
            # df[['c_x', 'c_y', 'c_z']] = np.dot(df[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
            # df.loc[df['vec_acc_x'].notna(), vecsacc] = np.dot(df[vecsacc].dropna().values, R)
            # df.loc[df['vec_don1_x'].notna(), vecsd1] = np.dot(df[vecsd1].dropna().values, R)
            # df.loc[df['vec_don2_x'].notna(), vecsd2] = np.dot(df[vecsd2].dropna().values, R)
            # df.loc[df['vec_don3_x'].notna(), vecsd3] = np.dot(df[vecsd3].dropna().values, R)
            df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
            # df['seg_chain_resnum'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            df['seg_chain_resnum_'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            # df['seg_chain_resnum'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            ###NEW STUFF FOR CLASH FILTER TEST
            df['str_index'] = df['iFG_count'] + '_' + df['vdM_count'] + '_' + df['query_name'] + '_' + df['seg_chain_resnum_']
            # df['hash'] = df['str_index'].apply(hash)
            print('making transformed dataframe...')
            dfs.append(df)
            # dataframe = pd.concat((dataframe, df), sort=False)
        dataframe = pd.concat(dfs, sort=False)

        print('removing clashes...')
        df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
        if len(df_nonclash) > 0:
            print('removing exposed ligands...')
            # df_nonclash.reset_index(inplace=True, drop=True)
            df_alpha = self.remove_alpha_hull(df_nonclash, template)
            self._nonclashing.append(df_alpha)
        # print('removing exposed ligands...')
        # df_alpha = self.remove_alpha_hull(dataframe, template)
        # if len(df_alpha) > 0:
        #     print('removing clashing ligands...')
        #     # df_nonclash.reset_index(inplace=True, drop=True)
        #     df_nonclash = self._remove(df_alpha, template, seg, chain, resnum, **kwargs)
        #     self._nonclashing.append(df_nonclash)
        tf = time.time()
        print('loaded ligand for (' + seg + ', ' + chain + ', ' + str(resnum) + ') in ' + str(tf - t0) + ' seconds.')

    # def _load(self, template, seg, chain, resnum, **kwargs):
    #     dataframe = pd.DataFrame()
    #     phipsi_width = kwargs.get('phipsi_width', 40)
    #
    #     for label in self.sequence_csts[(seg, chain, resnum)].keys():
    #         print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
    #         if label == 'PHI_PSI':
    #             df_list = list()
    #             phi, psi = template.get_phi_psi(seg, chain, resnum)
    #             for resn in self.sequence_csts[(seg, chain, resnum)][label]:
    #                 df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
    #                                                  phi, psi, phipsi_width)
    #                 df_list.append(df_phipsi)
    #             df = pd.concat(df_list)
    #         else:
    #             df = pd.concat([self._sig_reps[label][resn]
    #                             for resn in self.sequence_csts[(seg, chain, resnum)][label]])
    #
    #         m_com = self._mobile_com[label][(seg, chain, resnum)]
    #         t_com = self._target_com[label][(seg, chain, resnum)]
    #         R = self._rot[label][(seg, chain, resnum)]
    #         print('transforming coordinates...')
    #         # df['coords'] = df['coords'].transform(lambda x: np.dot((x - m_com), R) + t_com)
    #         # df['coords'] = rottrans(np.stack(df.coords), m_com, R, t_com)
    #         df['coords'] = list(np.dot(np.stack(df.coords) - m_com, R) + t_com)
    #         df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
    #         print('making transformed dataframe...')
    #         dataframe = pd.concat((dataframe, df), sort=False)
    #
    #     print('removing clashes...')
    #     # dataframe_gr = dataframe.groupby(['iFG_count', 'vdM_count',
    #     #                                   'query_name', 'seg_chain_resnum'])
    #
    #     df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
    #     print('removing exposed ligands...')
    #     if len(df_nonclash) > 0:
    #         df_nonclash.reset_index(inplace=True, drop=True)
    #         df_alpha = self.remove_alpha_hull(df_nonclash, template)
    #         self.dataframe = pd.concat((self.dataframe, df_alpha), ignore_index=True)
    #         # self.dataframe['lig_resname'] = self.dataframe['resname']
    #         # self.dataframe['lig_name'] = self.dataframe['name']
    # #        for df_chnk in self.chunk_df(dataframe_gr, gr_chunk_size=500):
    # #            df_nonclash = self._remove(df_chnk, template, seg, chain, resnum, **kwargs)
    # #            print('concatenating non-clashing chunk to dataframe')
    # #            self.dataframe = pd.concat((self.dataframe, df_nonclash), ignore_index=True)
    # #        self.dataframe['in_hull'] = None
    #         self._set_grouped_dataframe()

    # @staticmethod
    # def _remove(dataframe, template, seg, chain, resnum, **kwargs):
    #     cla = Clash(df=template.dataframe, df_query=dataframe,
    #                 exclude=None, **kwargs)
    #     cla.find()
    #     return cla.nonclashing_vdms
    @staticmethod
    def _remove(dataframe, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        cla = Clash(dfq=dataframe, dft=template.dataframe)
        # cla.set_grouping(['iFG_count', 'vdM_count',
        #                   'query_name', 'seg_chain_resnum'])
        # cla.set_grouping(['iFG_count', 'vdM_count', 'query_name'])
        cla.set_grouping('str_index')
        cla.find()
        tf = time.time()
        print('time:', tf-t0)
        return cla.dfq_clash_free
    
    # def print_lig(self, lig, group_name, out_path, out_name):
    #     df_lig = self.dataframe_grouped.get_group(group_name)
    #     coords = np.stack(df_lig[df_lig['name'] == n]['coords'].item() for n in lig.getNames())
    #     lig_copy = lig.copy()
    #     lig_copy.setCoords(coords)
    #     writePDB(out_path + out_name, lig_copy)
    @staticmethod
    def print_lig(group_name, df_group, outpath, out_name_tag=''):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        ProdyAG().print_ag(group_name, df_group, outpath, out_name_tag)

#     def remove_alpha_hull(self, df, template, num_heavy=34, num_total=59):
#         """Removes ligands with less than *percent_buried* heavy atoms
#          within the alpha hull of the template.  This also adds the columns
#          *in_hull* and *dist_to_hull* to the (surviving) ligand dataframe"""
#
#         df_lig_heav = df[~df['atom_type_label'].isin(['h_alkyl', 'h_pol', 'h_aro'])]
#         df_lig_heav_gr = df_lig_heav.groupby(['iFG_count', 'vdM_count',
#                                               'query_name', 'seg_chain_resnum'])
#         indices = list()
#         lig_index = list()
#         in_outs = list()
#         for n, lig in df_lig_heav_gr:
#             # coords = np.stack(lig['coords'])
#             coords = lig[['c_x', 'c_y', 'c_z']].values
# #            in_out = template.alpha_hull.pnts_in_hull(coords)
#             tf, in_out = template.alpha_hull.pnts_in_hull_threshold(coords, self.percent_buried)
# #            in_out_arr = np.array(in_out)
# #            percent_in = np.sum(in_out_arr) / in_out_arr.size
#             if tf: #percent_in >= percent_buried:
#                 indices.append(n)
#                 lig_index.extend(lig.index)
#                 in_outs.extend(in_out)
#                 # df.loc[lig.index, 'in_hull'] = in_out
#                 # dists = template.alpha_hull.get_pnts_distance(coords)
#                 # self.dataframe.loc[lig_heav.index, 'dist_to_hull'] = dists
#         df.loc[lig_index, 'in_hull'] = in_outs
#
#         df_indices = pd.DataFrame(indices, columns=['iFG_count', 'vdM_count',
#                                                     'query_name', 'seg_chain_resnum'])
#         df = pd.merge(df, df_indices, on=['iFG_count', 'vdM_count',
#                                           'query_name', 'seg_chain_resnum'])
#         return df
    def remove_alpha_hull(self, df, template):
        """Removes ligands with less than *percent_buried* heavy atoms
         within the alpha hull of the template.  This also adds the columns
         *in_hull* and *dist_to_hull* to the (surviving) ligand dataframe"""

        # dfh = df[~df['atom_type_label'].isin(['h_alkyl', 'h_pol', 'h_aro'])]
        # dfh = df[df['atom_type_label'].isin(['c_alkyl', 'c_aro'])]
        dfh = df[df[self.isin_field].isin(self.isin)]
        inout = template.alpha_hull.pnts_in_hull(dfh[['c_x', 'c_y', 'c_z']].values)
        index = dfh.index
        df.loc[index, 'in_hull'] = inout
        inout = inout.reshape(-1, self.num_heavy)
        # bur = inout.sum(axis=1) > np.floor((self.percent_buried * self.num_heavy))
        bur = inout.sum(axis=1) >= self.percent_buried * self.num_heavy
        df_inds = np.arange(len(df)).reshape(-1, self.num_total)
        buried_inds = df_inds[bur].flatten()
        df_bur = df.iloc[buried_inds]
        return df_bur

    def find_frag_neighbors(self, vdms, template, hb_only=False, return_rmsd=False):
        if not self.dataframe_grouped:
            self._set_grouped_dataframe()

        # self.dataframe_frag_coords = self.dataframe_grouped.apply(lambda x: self._get_frag_coords(df=x))
        for ind, lig in self.dataframe_grouped:
            pose = Pose()
            dfs_to_append = list()
            lig = lig.copy()
            for vdm in vdms:
                lig_coords = self._get_frag_coords(lig, vdm)
                if return_rmsd:
                    dist, ind_neighbors = vdm.neighbors.radius_neighbors(lig_coords, return_distance=True)
                    dist = dist[0]
                    # print('len dist', len(dist))
                    ind_neighbors = ind_neighbors[0]
                    # print('len ind', len(ind_neighbors))
                    rmsds = dist / np.sqrt(vdm.num_iFG_atoms)
                else:
                    ind_neighbors = vdm.neighbors.radius_neighbors(lig_coords, return_distance=False)[0]
                if ind_neighbors.size > 0:
                    # indices = [vdm.dataframe_iFG_coords.index[i] for i in ind_neighbors]
                    # df_indices = pd.DataFrame(indices, columns=['iFG_count', 'vdM_count',
                    #                                             'query_name', 'seg_chain_resnum'])
                    # print(df_indices)
                    if return_rmsd:
                        df_uniq = pd.DataFrame(vdm.dataframe_iFG_coords.iloc[ind_neighbors])
                        df_uniq['rmsd_to_query'] = rmsds
                        # print('len df uniq', len(df_uniq))
                        # print('len rmsds', len(rmsds))
                        # print('shape rmsds', rmsds.shape)
                        df_uniq = df_uniq.drop_duplicates()
                        to_concat = []
                        # print(df_uniq.columns)
                        # print(df_uniq)
                        # print('vals', df_uniq[['str_index', 'rmsd_to_query']].values)
                        for str_index, rmsd in df_uniq[['str_index', 'rmsd_to_query']].values:
                            d = vdm.dataframe_grouped.get_group(str_index).copy()
                            d['rmsd_to_query'] = rmsd
                            to_concat.append(d)
                        df_to_append = pd.concat(to_concat, sort=False, ignore_index=True)
                    else:
                        df_uniq = vdm.dataframe_iFG_coords.iloc[ind_neighbors].drop_duplicates()
                    # df_to_append = pd.merge(vdm.dataframe, df_uniq, on=['iFG_count', 'vdM_count',
                    #                                                         'query_name', 'seg_chain_resnum'])  #THIS IS REALLY SLOW
                    # df_to_append = pd.concat([vdm.dataframe_grouped.get_group(g[1:]) for g in df_uniq.itertuples()],
                    #                          sort=False, ignore_index=True)
                        df_to_append = pd.concat([vdm.dataframe_grouped.get_group(g) for g in
                                              df_uniq.values], sort=False, ignore_index=True)
                    print('appending to possible pose...')
                    dfs_to_append.append(df_to_append)
            if len(dfs_to_append) > 0:
                pose._vdms = pd.concat(dfs_to_append)
                if 'num_tag' in pose._vdms.columns:
                    pose._vdms = pose._vdms.drop('num_tag', axis=1).drop_duplicates()
                else:
                    pose._vdms = pose._vdms.drop_duplicates()
                pose.ligand = lig
                pose.ligand_gr_name = ind
                # print('pose._vdms=', pose._vdms)
            # if len(pose._vdms) > 0:
                print('getting non clashing pose...')
                pose.set_nonclashing_vdms()
                if len(pose.vdms) > 0:
                    # print('pose.vdms=', pose.vdms)
                    print('checking pose constraints...')
                    sc_bb = pd.concat((pose.vdms_sidechains, template.dataframe), sort=False)
                    lig_con = Contact(sc_bb, lig)
                    lig_con.find()
                    # print(lig_con.df_contacts)
                    if len(lig_con.df_contacts) > 0:
                        inout = template.alpha_hull.pnts_in_hull(lig[['c_x', 'c_y', 'c_z']].values)
                        lig.loc[:, 'in_hull'] = inout
                        dist_bur = template.alpha_hull.get_pnts_distance(lig[['c_x', 'c_y', 'c_z']].values)
                        lig.loc[:, 'dist_in_hull'] = dist_bur
                        if self.check_csts(lig, lig_con.df_contacts):
                            print('pose found...')
                            pose.lig_contacts = lig_con.df_contacts
                            if hb_only:
                                hb_contacts = pose.lig_contacts[pose.lig_contacts.contact_type == 'hb']
                                df_hb = get_vdms_hbonding_to_lig(pose, hb_contacts)
                                bb_contacts = hb_contacts[hb_contacts.iFG_count_q.isnull()]
                                all_vdms = []
                                all_vdms.append(df_hb)
                                if len(bb_contacts) > 0:
                                    bb_only_vdms = pose.vdms[
                                        pose.vdms.seg_chain_resnum.isin(set(bb_contacts.seg_chain_resnum_q))].groupby(
                                        pose.groupby).filter(lambda x: 'X' not in set(x.chain))
                                    # for n, g in bb_only_vdms.groupby('seg_chain_resnum'):
                                    for n, g in bb_only_vdms.groupby('seg_chain_resnum_'):
                                        min_clust_num = min(g.cluster_number)
                                        for n_, g_ in g[g.cluster_number == min_clust_num].groupby(pose.groupby):
                                            all_vdms.append(g_)
                                            break
                                df_all_vdms = pd.concat(all_vdms, sort=False)
                                if len(df_all_vdms) == 0:
                                    print('just kidding...no real H-bonds.')
                                    continue
                                df = pd.merge(pose.vdms, df_all_vdms[pose.groupby].drop_duplicates(), on=pose.groupby)
                                pose.vdms = df
                                pose.vdms_sidechains = df_hb
                                pose.lig_contacts = hb_contacts
                            pose._vdms = None
                            lig_heavy_in_hull = lig[(lig[self.isin_field].isin(self.isin)) & lig.in_hull]
                            pose.ligand_num_heavy_buried = len(lig_heavy_in_hull)
                            pose.ligand_avg_heavy_buried_dist = lig_heavy_in_hull.dist_in_hull.mean()
                            pose.lig_csts = self.csts
                            pose.lig_csts_gr = self.csts_gr
                            self.poses.append(pose)

    def check_csts(self, lig, lig_contacts):
        if not self.csts_gr:
            return True
        for n, cst_gr in self.csts_gr:
            atom_cst_tests = list()
            for i, cst in cst_gr.iterrows():
                if cst['contact_type']:
                    resname = lig_contacts['resname_t'] == cst['lig_resname']
                    name = lig_contacts['name_t'] == cst['lig_name']
                    lig_atom = lig_contacts[resname & name]
                    if any(lig_atom['contact_type'].isin(cst['contact_type'])):
                        atom_cst_tests.append(True)
                    else:
                        atom_cst_tests.append(False)
                if pd.notnull(cst['burial']):
                    resname = lig['lig_resname'] == cst['lig_resname']
                    name = lig['lig_name'] == cst['lig_name']
                    lig_atom = lig[resname & name]
                    lig_burial = lig_atom['in_hull'].item()
                    if cst['burial'] == lig_burial:
                        atom_cst_tests.append(True)
                    else:
                        atom_cst_tests.append(False)
                if pd.notnull(cst['dist_buried']):
                    resname = lig['lig_resname'] == cst['lig_resname']
                    name = lig['lig_name'] == cst['lig_name']
                    lig_atom = lig[resname & name]
                    lig_dist_buried = lig_atom['dist_in_hull'].item()
                    if cst['dist_buried_lessthan']:
                        if lig_dist_buried < cst['dist_buried']:
                            atom_cst_tests.append(True)
                        else:
                            atom_cst_tests.append(False)
                    else:
                        if lig_dist_buried > cst['dist_buried']:
                            atom_cst_tests.append(True)
                        else:
                            atom_cst_tests.append(False)
            if not any(atom_cst_tests):  # If any group cst fails, the function returns False
                return False
        return True

    def score_poses(self, template):
        for pose in self.poses:
            pose.find_opt(template)
            pose.set_total_opt_en(template)


def get_heavy(row):
    name = row['name']
    if name[0] == 'H':
        return False
    if name[:2] in {'1H', '2H', '3H', '4H'}:
        return False
    return True


class Pose:

    def __init__(self, **kwargs):
        self.score_designability = None
        self.score_opt = None
        self.ligand = None  # ligand dataframe
        self.ligand_gr_name = None
        self._vdms = None #dict()  # keys are vdm names, vals are dataframes, pre-clash removal
        self.vdms = None #dict()  # keys are vdm names, vals are dataframes, post-clash removal
        self.vdms_sidechains = None
        # self.groupby = kwargs.get('groupby', ['iFG_count', 'vdM_count',
        #                                       'query_name', 'seg_chain_resnum'])
        self.groupby = kwargs.get('groupby', ['str_index'])
        self.lig_contacts = None
        self.ligand_num_heavy_buried = None
        self.ligand_avg_heavy_buried_dist = 0
        self.hb_net = list()
        self.num_buns_lig = 0
        self._poseleg_number = 0
        self.lig_csts = None
        self.lig_csts_gr = None
        self.opt_en_sidechains = 0
        self.opt_res_sidechains = None
        self.opt_en = 0
        self.opt_vdms = None
        self.vdms_sidechains_opt = []

    def set_lig_csts(self, path_to_cst_file):
        self.lig_csts = make_cst_df(path_to_cst_file)
        self.lig_csts_gr = self.lig_csts.groupby('cst_group')

    def set_nonclashing_vdms(self):
        vdms_x = self._vdms[self._vdms.chain == 'X'].copy()
        cla = Clash(vdms_x, self.ligand, **dict(tol=0.1))
        cla.set_grouping(self.groupby)
        cla.find(return_clash_free=True, return_clash=True)
        df = pd.merge(self._vdms, cla.dfq_clash[self.groupby],
                    on=self.groupby, how='outer', indicator=True, sort=False)
        df = df[df['_merge'] == 'left_only'].drop(columns='_merge')
        self.vdms = df
        self.vdms_sidechains = cla.dfq_clash_free

    @staticmethod
    def get_counts(phi, psi, phi_vec, psi_vec):
        # Note this doesn't account for periodic conditions...
        phi_vec[np.isnan(phi_vec)] = 1000  # makes sure np.nan fails and doesn't give runtime warning.
        psi_vec[np.isnan(psi_vec)] = 1000  # makes sure np.nan fails and doesn't give runtime warning.
        if (phi == None) and (psi != None)\
                and isinstance(phi_vec, np.ndarray) and isinstance(psi_vec, np.ndarray):
            counts = ((phi_vec == np.nan)
                      & (psi_vec < psi + 20) & (psi_vec > psi - 20)).sum()
        elif (phi != None) and (psi == None)\
                and isinstance(phi_vec, np.ndarray) and isinstance(psi_vec, np.ndarray):
            counts = ((phi_vec < phi + 20) & (phi_vec > phi - 20)
                      & (psi_vec == np.nan)).sum()
        elif (phi == None) and (psi == None)\
                and isinstance(phi_vec, np.ndarray) and isinstance(psi_vec, np.ndarray):
            counts = ((phi_vec == np.nan)
                      & (psi_vec == np.nan)).sum()
        elif isinstance(phi, float) and isinstance(psi, float)\
                and isinstance(phi_vec, np.ndarray) and isinstance(psi_vec, np.ndarray):
            counts = ((phi_vec < phi + 20) & (phi_vec > phi - 20)
                      & (psi_vec < psi + 20) & (psi_vec > psi - 20)).sum()
        else:
            counts = 0
        return counts

    @staticmethod
    def _get_phipsi(phi, psi, phi_vec, psi_vec):
        # Note this doesn't account for periodic conditions...
        phi_vec[np.isnan(phi_vec)] = 1000  # makes sure np.nan fails and doesn't give runtime warning.
        psi_vec[np.isnan(psi_vec)] = 1000  # makes sure np.nan fails and doesn't give runtime warning.
        if (phi == None) and (psi != None):
            tf = ((phi_vec == np.nan)
                      & (psi_vec < psi + 20) & (psi_vec > psi - 20))
        elif (phi != None) and (psi == None):
            tf = ((phi_vec < phi + 20) & (phi_vec > phi - 20)
                      & (psi_vec == np.nan))
        elif (phi == None) and (psi == None):
            tf = ((phi_vec == np.nan)
                      & (psi_vec == np.nan))
        elif isinstance(phi, float) and isinstance(psi, float):
            tf = ((phi_vec < phi + 20) & (phi_vec > phi - 20)
                      & (psi_vec < psi + 20) & (psi_vec > psi - 20))
        else:
            tf = [False] * len(psi_vec)
        return tf

    def calc_ss_score(self, ifg_name, aa, iFG_count, vdM_count, query_name, phi, psi):
        # phi_vec = ss_score_dict[ifg_name][label][aa]['phi']
        # psi_vec = ss_score_dict[ifg_name][label][aa]['psi']
        df = cluster_dict[ifg_name]['PHI_PSI'][aa]
        vdm = df[(df.iFG_count == iFG_count) & (df.vdM_count == vdM_count) & (df.query_name == query_name)]
        if len(vdm) == 0:
            print('vdm was not found in PHIPSI cluster df', ifg_name, aa,
                  'vdm=', iFG_count, vdM_count, query_name)
            return 0
        cluster_number = vdm['cluster_number'].iat[0]
        cluster_size = vdm['cluster_size'].iat[0]
        phi_vec = df['phi'].values
        psi_vec = df['psi'].values
        label = 'PHI_PSI'
        if not isinstance(phi_vec, np.ndarray):
            print(ifg_name, label, aa, 'phi', phi_vec)
        if not isinstance(psi_vec, np.ndarray):
            print(ifg_name, label, aa, 'psi', phi_vec)
        num_counts = self.get_counts(phi, psi, phi_vec, psi_vec)
        total_counts = phi_vec.size
        f = num_counts / total_counts
        expected = cluster_size * f
        df_ = df[df.cluster_number == cluster_number]
        clu_phi_vec = df_.phi.values #ss_score_dict[ifg_name][label][aa][cluster_number]['phi']
        clu_psi_vec = df_.psi.values #ss_score_dict[ifg_name][label][aa][cluster_number]['psi']
        observed = self.get_counts(phi, psi, clu_phi_vec, clu_psi_vec)
        return np.log((observed + 1) / (expected + 1))

    # def calc_score(self, ifg_name, label, aa, cluster_number, cluster_size, phi, psi):
    #     df = cluster_dict[ifg_name][label][aa]
    #     # total = len(df)
    #     # f = cluster_size / total
    #     phi_vec = df['phi'].values
    #     psi_vec = df['psi'].values
    #     tf = self._get_phipsi(phi, psi, phi_vec, psi_vec)
    #     df_ = df['cluster_number'][tf]
    #     total_phipsi = len(df_)
    #     # expected_w_phipsi = f * total_phipsi
    #     observed_w_phipsi = (df_ == cluster_number).sum()
    #     a = 0.01
    #     if (total_phipsi == 0) or (observed_w_phipsi == 0):
    #         p = 0
    #     else:
    #         p = observed_w_phipsi / total_phipsi
    #     # return -1 * np.log((observed_w_phipsi + 1) / (expected_w_phipsi + 1))
    #     return -1 * np.log(p + a)

    def calc_score(self, ifg_name, aa, iFG_count, vdM_count, query_name, phi, psi):
        df = cluster_dict[ifg_name]['PHI_PSI'][aa]

        vdm = df[(df.iFG_count == iFG_count) & (df.vdM_count == vdM_count) & (df.query_name == query_name)]
        a = 0.01
        if len(vdm) == 0:
            print('vdm was not found in PHIPSI cluster df', ifg_name, aa,
                  'vdm=', iFG_count, vdM_count, query_name)
            return -1 * np.log(a)
        cluster_number = vdm['cluster_number'].iat[0]
        # cluster_size = vdm['cluster_size'].iat[0]

        total = len(df)
        num_clusters = df.centroid.sum()
        # f = cluster_size / total
        phi_vec = df['phi'].values
        psi_vec = df['psi'].values
        tf = self._get_phipsi(phi, psi, phi_vec, psi_vec)
        frac_phipsi = tf.sum()/total
        df_ = df[tf]
        # if len(df_) > 0:
        #     avg = df_.groupby('cluster_number').size().mean()
        # else:
        #     avg = 0
        # total_phipsi = len(df_)
        # expected_w_phipsi = f * total_phipsi
        observed_w_phipsi = (df_['cluster_number'] == cluster_number).sum()
        expected_w_phipsi = total * 1/num_clusters * frac_phipsi
        if (expected_w_phipsi == 0) or (observed_w_phipsi == 0):
            p = 0
        else:
            p = (observed_w_phipsi) / (expected_w_phipsi)
        # return -1 * np.log((observed_w_phipsi + 1) / (expected_w_phipsi + 1))
        return -1 * np.log(p + a)

    def calc_c_phi_psi_score(self, ifg_name, aa, iFG_count, vdM_count, query_name):
        df = cluster_dict[ifg_name]['PHI_PSI'][aa]
        total = len(df)
        num_clusters = df.centroid.sum()
        vdm = df[(df.iFG_count == iFG_count) & (df.vdM_count == vdM_count) & (df.query_name == query_name)]
        if len(vdm) == 0:
            print('vdm was not found in PHIPSI cluster df', ifg_name, aa,
                  'vdm=', iFG_count, vdM_count, query_name)
            return 0
        cluster_number = vdm.cluster_number.iat[0]
        # f = cluster_size / total
        expected = total / num_clusters
        observed = (df['cluster_number'] == cluster_number).sum()
        p = (observed) / (expected)
        # return -1 * np.log((observed_w_phipsi + 1) / (expected_w_phipsi + 1))
        return np.log(p)

    def get_ss_score(self, df, template):
        cluster_number = df['cluster_number'].iat[0]
        cluster_size = df['cluster_size'].iat[0]
        aa = df['resname_vdm'].iat[0]
        # label = df['label'].iat[0]
        iFG_count = df['iFG_count'].iat[0]
        vdM_count = df['vdM_count'].iat[0]
        query_name = df['query_name'].iat[0]
        # cluster_number = vdm.cluster_number.iat[0]
        ifg_name = df['iFG_name'].iat[0]
        seg_chain_resnum = df['seg_chain_resnum'].iat[0]
        phi, psi = template.phi_psi_dict[seg_chain_resnum]
        ss_score = self.calc_ss_score(ifg_name, aa, iFG_count, vdM_count, query_name, phi, psi)
        # ss_score = self.calc_score(ifg_name, label, aa, cluster_number, cluster_size, phi, psi)
        print(ifg_name, aa, cluster_number, cluster_size, seg_chain_resnum, phi, psi, ss_score)
        return ss_score

    def get_p_score(self, df, template):
        iFG_count = df['iFG_count'].iat[0]
        vdM_count = df['vdM_count'].iat[0]
        query_name = df['query_name'].iat[0]
        aa = df['resname_vdm'].iat[0]
        label = df['label'].iat[0]
        ifg_name = df['iFG_name'].iat[0]
        seg_chain_resnum = df['seg_chain_resnum'].iat[0]
        phi, psi = template.phi_psi_dict[seg_chain_resnum]
        # ss_score = self.calc_ss_score(ifg_name, label, aa, cluster_number, cluster_size, phi, psi)
        p_score = self.calc_score(ifg_name, aa, iFG_count, vdM_count, query_name, phi, psi)
        print(ifg_name, label, iFG_count, vdM_count, query_name, aa, seg_chain_resnum, phi, psi, p_score)
        return p_score

    def get_c_phi_psi_score(self, df):
        aa = df['resname_vdm'].iat[0]
        iFG_count = df['iFG_count'].iat[0]
        vdM_count = df['vdM_count'].iat[0]
        query_name = df['query_name'].iat[0]
        ifg_name = df['iFG_name'].iat[0]
        seg_chain_resnum = df['seg_chain_resnum'].iat[0]
        c_phi_psi_score = self.calc_c_phi_psi_score(ifg_name, aa, iFG_count, vdM_count, query_name)
        # ss_score = self.calc_score(ifg_name, label, aa, cluster_number, cluster_size, phi, psi)
        print(ifg_name, aa, iFG_count, vdM_count, query_name, seg_chain_resnum, c_phi_psi_score)
        return c_phi_psi_score

    def find_opt(self, template):
        if len(self.vdms_sidechains) == 0:
            return
        vdms = self.vdms_sidechains.copy()
        zero_pt_en = np.log(0.01) + 0.01 # p_score can only be as large as -np.log(0.01) due to a = 0.01
        vdms['ss_score_local'] = 0
        vdms['p_score'] = 0
        vdms['c_phi_psi_score'] = 0
        es = dict()
        ep = defaultdict(dict)
        grs = vdms.groupby('seg_chain_resnum_')
        d = defaultdict(dict)
        gby = ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum_']
        for site, gr in grs:
            for vdm_name, subgroup in gr.groupby(gby):
                d[site][vdm_name] = subgroup
                ss_score = self.get_ss_score(subgroup, template)
                p_score = self.get_p_score(subgroup, template)
                c_phi_psi_score = self.get_c_phi_psi_score(subgroup)
                # es[vdm_name] = -1 * subgroup['score'].iat[0] - ss_score
                es[vdm_name] = p_score + zero_pt_en # p_score can only be as large as 100 due to a = 0.01
                vdms.loc[subgroup.index, 'ss_score_local'] = ss_score
                vdms.loc[subgroup.index, 'p_score'] = p_score
                vdms.loc[subgroup.index, 'c_phi_psi_score'] = c_phi_psi_score

        # opts = []
        # for m in reversed(range(1, len(list(d.keys())) + 1)):
        #     for sites in itertools.combinations(list(d.keys()), m):
        sites = list(d.keys())
        for i in range(len(sites)):
            for vdm_name_i in list(d[sites[i]].keys()):
                si = es[vdm_name_i]
                pmin = 0
                for j in sorted(set(range(len(sites))) - {i}):
                    _p = []
                    for vdm_name_j in d[sites[j]].keys():
                        try:
                            _p.append(ep[vdm_name_i][vdm_name_j])
                        except:
                            cla = Clash(d[sites[i]][vdm_name_i].copy(), d[sites[j]][vdm_name_j].copy(), **dict(tol=0.1))
                            cla.set_grouping(self.groupby)
                            cla.find(return_clash_free=True)
                            if len(cla.dfq_clash_free) == 0:
                                ep[vdm_name_i][vdm_name_j] = np.inf
                                ep[vdm_name_j][vdm_name_i] = np.inf
                                _p.append(np.inf)
                            else:
                                ep[vdm_name_i][vdm_name_j] = 0
                                ep[vdm_name_j][vdm_name_i] = 0
                                _p.append(0)
                    pmin += np.min(_p)

                for vdm_name_k in d[sites[i]].keys():
                    sk = es[vdm_name_k]
                    pmax = 0
                    for j in sorted(set(range(len(sites))) - {i}):
                        _p = []
                        for vdm_name_j in d[sites[j]].keys():
                            try:
                                _p.append(ep[vdm_name_k][vdm_name_j])
                            except:
                                cla = Clash(d[sites[i]][vdm_name_k].copy(), d[sites[j]][vdm_name_j].copy(),
                                            **dict(tol=0.1))
                                cla.set_grouping(self.groupby)
                                cla.find(return_clash_free=True)
                                if len(cla.dfq_clash_free) == 0:
                                    ep[vdm_name_k][vdm_name_j] = np.inf
                                    ep[vdm_name_j][vdm_name_k] = np.inf
                                    _p.append(np.inf)
                                else:
                                    ep[vdm_name_k][vdm_name_j] = 0
                                    ep[vdm_name_j][vdm_name_k] = 0
                                    _p.append(0)
                        pmax += np.max(_p)

                    if si + pmin > sk + pmax:
                        d[sites[i]].pop(vdm_name_i)
                        break

        for key in d.keys():
            d[key]['None'] = 0
        es['None'] = 0
        for key1 in list(ep.keys()):
            ep[key1]['None'] = 0
            for key2 in list(ep[key1].keys()):
                ep['None'][key2] = 0
        ep['None']['None'] = 0

        # print('es', es)
        # print('ep', ep)

        opt_en = np.inf
        opt_res = list()
        keys = [list(v) for v in d.values()]
        for combo in itertools.product(*keys):
            c_en = np.sum(es[key_] for key_ in combo)
            c_en += np.sum(ep[key1_][key2_]
                           for key1_, key2_ in itertools.combinations(combo, 2))
            if c_en < opt_en:
                opt_en = c_en
                opt_res = combo
        # opts.append((opt_en, opt_res))
        # opt_en, opt_res = sorted(opts)[0]
        self.opt_en_sidechains = opt_en
        self.opt_res_sidechains = opt_res
        _gps = vdms.groupby(gby)
        print(opt_en, opt_res)
        if set(opt_res) != {'None'}:
            to_concat = [_gps.get_group(g) for g in opt_res if g != 'None']
            if len(to_concat) > 0:
                self.vdms_sidechains_opt = pd.concat(to_concat)

    # WORKING CODE BELOW
    # def find_opt(self, template):
    #     if len(self.vdms_sidechains) == 0:
    #         return
    #     vdms = self.vdms_sidechains.copy()
    #     vdms['ss_score_local'] = 0
    #     es = dict()
    #     ep = defaultdict(dict)
    #     grs = vdms.groupby('seg_chain_resnum_')
    #     d = defaultdict(dict)
    #     gby = ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum_']
    #     for site, gr in grs:
    #         for vdm_name, subgroup in gr.groupby(gby):
    #             d[site][vdm_name] = subgroup
    #             ss_score = self.get_ss_score(subgroup, template)
    #             # es[vdm_name] = -1 * subgroup['score'].iat[0] - ss_score
    #             es[vdm_name] = ss_score
    #             vdms.loc[subgroup.index, 'ss_score_local'] = ss_score
    #     sites = list(d.keys())
    #     for i in range(len(sites)):
    #         for vdm_name_i in list(d[sites[i]].keys()):
    #             si = es[vdm_name_i]
    #             pmin = 0
    #             for j in sorted(set(range(len(sites))) - {i}):
    #                 _p = []
    #                 for vdm_name_j in d[sites[j]].keys():
    #                     try:
    #                         _p.append(ep[vdm_name_i][vdm_name_j])
    #                     except:
    #                         cla = Clash(d[sites[i]][vdm_name_i].copy(), d[sites[j]][vdm_name_j].copy(), **dict(tol=0.1))
    #                         cla.set_grouping(self.groupby)
    #                         cla.find(return_clash_free=True)
    #                         if len(cla.dfq_clash_free) == 0:
    #                             ep[vdm_name_i][vdm_name_j] = np.inf
    #                             ep[vdm_name_j][vdm_name_i] = np.inf
    #                             _p.append(np.inf)
    #                         else:
    #                             ep[vdm_name_i][vdm_name_j] = 0
    #                             ep[vdm_name_j][vdm_name_i] = 0
    #                             _p.append(0)
    #                 pmin += np.min(_p)
    #
    #             for vdm_name_k in d[sites[i]].keys():
    #                 sk = es[vdm_name_k]
    #                 pmax = 0
    #                 for j in sorted(set(range(len(sites))) - {i}):
    #                     _p = []
    #                     for vdm_name_j in d[sites[j]].keys():
    #                         try:
    #                             _p.append(ep[vdm_name_k][vdm_name_j])
    #                         except:
    #                             cla = Clash(d[sites[i]][vdm_name_k].copy(), d[sites[j]][vdm_name_j].copy(),
    #                                         **dict(tol=0.1))
    #                             cla.set_grouping(self.groupby)
    #                             cla.find(return_clash_free=True)
    #                             if len(cla.dfq_clash_free) == 0:
    #                                 ep[vdm_name_k][vdm_name_j] = np.inf
    #                                 ep[vdm_name_j][vdm_name_k] = np.inf
    #                                 _p.append(np.inf)
    #                             else:
    #                                 ep[vdm_name_k][vdm_name_j] = 0
    #                                 ep[vdm_name_j][vdm_name_k] = 0
    #                                 _p.append(0)
    #                     pmax += np.max(_p)
    #
    #                 if si + pmin > sk + pmax:
    #                     d[sites[i]].pop(vdm_name_i)
    #                     break
    #
    #     for key in d.keys():
    #         d[key]['None'] = 0
    #     es['None'] = 0
    #     for key1 in list(ep.keys()):
    #         ep[key1]['None'] = 0
    #         for key2 in list(ep[key1].keys()):
    #             ep['None'][key2] = 0
    #     ep['None']['None'] = 0
    #
    #     # print('es', es)
    #     # print('ep', ep)
    #
    #     opt_en = np.inf
    #     opt_res = list()
    #     keys = [list(v) for v in d.values()]
    #     for combo in itertools.product(*keys):
    #         c_en = np.sum(es[key_] for key_ in combo)
    #         c_en += np.sum(ep[key1_][key2_]
    #                        for key1_, key2_ in itertools.combinations(combo, 2))
    #         if c_en < opt_en:
    #             opt_en = c_en
    #             opt_res = combo
    #
    #     self.opt_en_sidechains = opt_en
    #     self.opt_res_sidechains = opt_res
    #     _gps = vdms.groupby(gby)
    #     print(opt_en, opt_res)
    #     if set(opt_res) != {'None'}:
    #         self.vdms_sidechains_opt = pd.concat([_gps.get_group(g) for g in opt_res if g != 'None'])

    def set_total_opt_en(self, template):
        self.opt_en = self.opt_en_sidechains
        vdms_bb = self.vdms.groupby(self.groupby).filter(lambda g: 'X' in g.chain)
        if len(vdms_bb) == 0:
            if len(self.vdms_sidechains_opt) == 0:
                return
            str_indices = set(self.vdms_sidechains_opt.str_index)
            self.opt_vdms = self.vdms[self.vdms.str_index.isin(str_indices)]
            opt_w_ss = self.vdms_sidechains_opt[['str_index', 'ss_score_local', 'p_score',
                                                 'c_phi_psi_score']].drop_duplicates()
            self.opt_vdms = pd.merge(self.opt_vdms, opt_w_ss, on='str_index')
            return
        vdms_bb['ss_score_local'] = 0
        vdms_bb['p_score'] = 0
        vdms_bb['c_phi_psi_score'] = 0
        grs = vdms_bb.groupby('seg_chain_resnum_')
        str_indices = set()
        for n, g in grs:
            # i = g.score.values.argmax()
            # s = g.score.values[i]
            subgroups = g.groupby(self.groupby)
            scores = []
            for subname, subgroup in subgroups:
                ss_score = self.get_ss_score(subgroup, template)
                p_score = self.get_p_score(subgroup, template)
                c_phi_psi_score = self.get_c_phi_psi_score(subgroup)
                # c_score = subgroup['score'].iat[0]
                score = p_score #ss_score #+ c_score
                scores.append((score, subname))
                vdms_bb.loc[subgroup.index, 'ss_score_local'] = ss_score
                vdms_bb.loc[subgroup.index, 'p_score'] = p_score
                vdms_bb.loc[subgroup.index, 'c_phi_psi_score'] = c_phi_psi_score
            # best_score = sorted(scores, reverse=True)[0]
            best_score = sorted(scores)[0]
            # i = g['score', 'ss_score'].values.sum(axis=1).argmax()
            # s = g['score', 'ss_score'].values[i].sum()
            # self.opt_en += -1 * best_score[0]
            self.opt_en += best_score[0]
            str_index = best_score[1]
            str_indices.add(str_index)
        vdms = vdms_bb[vdms_bb.str_index.isin(str_indices)]['str_index', 'ss_score_local', 'p_score', 'c_phi_psi_score']
        if len(self.vdms_sidechains_opt) > 0:
            # str_indices.update(set(self.vdms_sidechains_opt.str_index))
            vdms = pd.concat([vdms, self.vdms_sidechains_opt[['str_index', 'ss_score_local', 'p_score',
                                                              'c_phi_psi_score']]]).drop_duplicates()
        # self.opt_vdms = self.vdms[self.vdms.str_index.isin(str_indices)]
        self.opt_vdms = pd.merge(self.vdms, vdms, on='str_index')

    # def find_opt(self):
    #     vdms = self.vdms_sidechains.copy()
    #     grs_ = vdms.groupby('seg_chain_resnum_')
    #     d = {n: gr.groupby(['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum_']) for n, gr in grs_}
    #     es_ = defaultdict(dict)
    #     ep_dict = defaultdict(dict)
    #     index_dict = dict()
    #     key_dict = dict()
    #     for k1, k2 in itertools.combinations(d.keys(), 2):
    #         grs1 = d[k1]
    #         # grs = _grs.groupby(['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum_'])
    #         grs2 = d[k2]
    #         N = len(grs1)
    #         M = len(grs2)
    #         key_dict[k1] = {i: key for i, key in enumerate(list(grs1.groups.keys()))}
    #         index_dict[k1] = {key: i for i, key in enumerate(list(grs1.groups.keys()))}
    #         index_dict[k2] = {key: i for i, key in enumerate(list(grs2.groups.keys()))}
    #         ep = np.zeros((M, N))
    #         es = np.zeros(N)
    #         for gr1, gr2 in itertools.product(*[grs1, grs2]):
    #             key_gr1 = gr1[0]
    #             key_gr2 = gr2[0]
    #             ind1 = index_dict[k1][key_gr1]
    #             ind2 = index_dict[k2][key_gr2]
    #             if key_gr1 in ep_dict:
    #                 if key_gr2 in ep_dict[key_gr1]:
    #                     ep[ind2, ind1] = ep_dict[key_gr1][key_gr2]
    #             else:
    #                 cla = Clash(gr1[1].copy(), gr2[1].copy(), **dict(tol=0.1))
    #                 cla.set_grouping(self.groupby)
    #                 cla.find(return_clash_free=True)
    #                 if len(cla.dfq_clash_free) == 0:
    #                     ep[ind2, ind1] = np.inf
    #                     ep_dict[key_gr1][key_gr2] = np.inf
    #                     ep_dict[key_gr2][key_gr1] = np.inf
    #                 else:
    #                     ep_dict[key_gr1][key_gr2] = 0
    #                     ep_dict[key_gr2][key_gr1] = 0
    #         # ep = ep + ep.T
    #         # ep_dict[n] = ep
    #         for gr in grs1:
    #             key_gr = gr[0]
    #             ind = index_dict[k1][key_gr]
    #             es[ind] = -1 * gr[1]['score'].iat[0]
    #
    #         to_del = _dee(es, ep)
    #         # to_del_keys = [key_dict[i] for i in to_del]
    #
    #         # grs = vdms.groupby('seg_chain_resnum_')
    #         # d = {n: gr for n, gr in grs}
    #         indices = sorted(set(range(N)) - set(to_del))
    #         for i in range(len(indices)):
    #             key = key_dict[k1][indices[i]]
    #             es_[key[-1]][key] = es[indices[i]]
    #
    #     opt_en = np.inf
    #     opt_res = list()
    #     keys = [list(v.keys()) for v in es_.values()]
    #     print(es_)
    #     print(ep_dict)
    #     for combo in itertools.product(*keys):
    #         c_en = np.sum(es_[key_[-1]][key_] for key_ in combo)
    #         c_en += np.sum(ep_dict[key1_][key2_]
    #                        for key1_, key2_ in itertools.combinations(combo, 2))
    #         if c_en < opt_en:
    #             opt_en = c_en
    #             opt_res = combo
    #
    #     return opt_en, opt_res

    # def find_opt(self):
    #     vdms = self.vdms_sidechains.copy()
    #     grs = vdms.groupby(['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum_'])
    #     N = len(grs)
    #     key_dict = {i: key for i, key in enumerate(list(grs.groups.keys()))}
    #     index_dict = {key: i for i, key in enumerate(list(grs.groups.keys()))}
    #     ep = np.zeros((N, N))
    #     es = np.zeros(N)
    #     for gr1, gr2 in itertools.combinations(grs, 2):
    #         key_gr1 = gr1[0]
    #         key_gr2 = gr2[0]
    #         ind1 = index_dict[key_gr1]
    #         ind2 = index_dict[key_gr2]
    #         if key_gr1[-1] == key_gr2[-1]:
    #             # ep[ind1, ind2] = np.inf
    #             continue
    #         cla = Clash(gr1[1].copy(), gr2[1].copy(), **dict(tol=0.1))
    #         cla.set_grouping(self.groupby)
    #         cla.find(return_clash_free=True)
    #         if len(cla.dfq_clash_free) == 0:
    #             ep[ind1, ind2] = np.inf
    #     ep = ep + ep.T
    #     for gr in grs:
    #         key_gr = gr[0]
    #         ind = index_dict[key_gr]
    #         es[ind] = -1 * gr[1]['score'].iat[0]
    #
    #     to_del = _dee(es, ep)
    #     # to_del_keys = [key_dict[i] for i in to_del]
    #
    #     # grs = vdms.groupby('seg_chain_resnum_')
    #     # d = {n: gr for n, gr in grs}
    #     es_ = defaultdict(dict)
    #     indices = sorted(set(range(N)) - set(to_del))
    #     for i in range(len(indices)):
    #         key = key_dict[indices[i]]
    #         es_[key[-1]][key] = es[indices[i]]
    #
    #     opt_en = np.inf
    #     opt_res = list()
    #     keys = [list(v.keys()) for v in es_.values()]
    #     print(es_)
    #     for combo in itertools.product(*keys):
    #         c_en = np.sum(es[index_dict[key_]] for key_ in combo)
    #         c_en += np.sum(ep[index_dict[key1_], index_dict[key2_]]
    #                        for key1_, key2_ in itertools.combinations(combo, 2))
    #         if c_en < opt_en:
    #             opt_en = c_en
    #             opt_res = combo
    #
    #     return opt_en, opt_res

    # def find_opt_en(self, Es, Ep):
    #     l = len(Es.keys())
    #     opt_res = list(zip([-1] * l, Es.keys()))
    #     opt_en = np.inf
    #     ranges = map(range, map(len, Es.values()))
    #     for inds in itertools.product(*ranges):
    #         tuples = list(zip(inds, Es.keys()))
    #         c_en = np.sum(Es[key][ind] for ind, key in tuples)
    #         c_en += np.sum(Ep[t1[1]][t2[1]][t1[0], t2[0]] for t1, t2 in itertools.combinations(tuples, 2))
    #         if c_en < opt_en:
    #             opt_en = c_en
    #             opt_res = tuples
    #     return opt_en, opt_res

    # def find_opt(self):
    #     pass

    def set_lig_buns(self, template):
        # con = Contact(template.dataframe, self.ligand)
        # con.find()
        # for field in ['iFG_count', 'vdM_count', 'query_name']:
        #     con.df_contacts[field + '_t'] = con.df_contacts[field]
        bun_acc, bun_don = get_bun_hb_atoms(df=self.ligand, template=template)
        self.num_buns_lig = len(bun_acc) + len(bun_don)
        self.lig_acc_buns = bun_acc
        self.lig_don_buns = bun_don

    def set_num_hb_net_lig_bun_interactions(self, template):
        for hbnet in self.hb_net:
            num_lig_ints = self._set_num_hb_net_lig_bun_interactions(hbnet, template)
            hbnet.num_lig_ints = num_lig_ints

    def _set_num_hb_net_lig_bun_interactions(self, hbnet, template):
        con = Contact(hbnet.primary, self.ligand)
        con.find()
        bun_acc, bun_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=con.df_contacts)
        return self.num_buns_lig - (len(bun_acc) + len(bun_don))

    def _set_num_pl_lig_bun_interactions(self, pl, template):
        con = Contact(pl, self.ligand)
        con.find()
        bun_acc, bun_don, sat_acc, sat_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=con.df_contacts, return_satisfied=True)
        num_lig_bun_ints = self.num_buns_lig - (len(bun_acc) + len(bun_don))
        sat_acc_set = set()
        sat_don_set = set()
        if len(sat_acc) > 0:
            sat_acc_set = set(sat_acc.name)
        if len(sat_don) > 0:
            sat_don_set = set(sat_don.name)
        return num_lig_bun_ints, sat_acc_set, sat_don_set


    @staticmethod
    def _print_vdm(vdm, group_name, df_group, outpath, out_name_tag='', full_fragments=False, with_bb=False):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        if not full_fragments:
            if not with_bb:
                ProdyAG().print_ag(group_name, df_group, outpath, out_name_tag)
            elif with_bb:
                label = set(df_group.label).pop()
                bb_names = rel_coords_dict[label]
                seg_chain_resnum = set(df_group.seg_chain_resnum).pop()
                df_ala = vdm._ideal_ala_df[label][seg_chain_resnum].copy()
                df_ala['segment'] = set(df_group.segment).pop()
                df_ala['resname'] = set(df_group.resname_vdm).pop()
                df_ala_bbsel = df_ala[df_ala['name'].isin(bb_names)]
                df = pd.concat((df_group, df_ala_bbsel))
                ProdyAG().print_ag(group_name, df, outpath, out_name_tag='_' + label + out_name_tag)
        else:
            pass

    def print_opt(self, outpath, template, include_iFG=True, include_template=False):
        template = template.dataframe
        template['p_score'] = 0
        template['resname_vdm'] = template['resname']
        vdms = pd.merge(self.opt_vdms, template[['resnum', 'chain', 'segment', 'seg_chain_resnum']].drop_duplicates(),
                        on='seg_chain_resnum', suffixes=['', '_t'])

        vdms.loc[vdms.chain == 'Y', 'segment_t'] = 'Y'

        if not include_iFG:
            vdms = vdms[vdms.chain == 'X']

        template = template[template.apply(get_heavy, axis=1)].copy()
        if not include_template:
            template = pd.merge(template, vdms['seg_chain_resnum'].drop_duplicates(), on='seg_chain_resnum')
        template['resnum_t'] = template['resnum']
        template['chain_t'] = template['chain']
        template['segment_t'] = template['segment']
        ligand = self.ligand.copy()
        ligand['resname_vdm'] = ligand['resname']
        ligand['resnum_t'] = ligand['resnum']
        ligand['chain_t'] = 'L'
        ligand['segment_t'] = 'L'
        ligand['p_score'] = 0

        cols = ['c_x', 'c_y', 'c_z', 'name', 'resnum_t', 'chain_t', 'segment_t', 'seg_chain_resnum', 'resname', 'p_score']
        vdms = pd.concat([vdms, template[cols]])
        vdms.reset_index(drop=True, inplace=True)
        for n, g in vdms.groupby('seg_chain_resnum'):
            g_ = g['resname_vdm'][~g.resname_vdm.isna()]
            if len(g_) > 0:
                resname = g_.iat[0]
            else:
                resname = g['resname'].iat[0]
            vdms.loc[g.index, 'resname_vdm'] = resname

        cols = ['c_x', 'c_y', 'c_z', 'name', 'resname_vdm', 'resnum_t', 'chain_t', 'segment_t', 'p_score']
        vdms = pd.concat([vdms, ligand[cols]])

        vdms = vdms.sort_values(['seg_chain_resnum', 'iFG_count', 'vdM_count', 'name'])
        # scores = np.sum(vdms[['score', 'ss_score_local']], axis=1)
        scores = vdms['p_score'].values
        ag = AtomGroup()
        ag.setCoords(vdms[['c_x', 'c_y', 'c_z']].values)
        ag.setResnums(vdms['resnum_t'].values)
        ag.setResnames(vdms['resname_vdm'].values)
        ag.setNames(vdms['name'].values)
        ag.setChids(vdms['chain_t'].values)
        ag.setSegnames(vdms['segment_t'].values)
        ag.setBetas(scores)
        # ag.setOccupancies(1)
        occ = np.ones(len(vdms))
        writePDB(outpath, ag, occupancy=occ)

        # vars = ['c_score', 'ss_score', 'total_score', 'resnum',
        #         'chain', 'segment', 'resname', 'iFG_name', 'label',
        #         'iFG_count', 'vdM_count', 'query_name']
        # vars = ['c_score', 'c_phi_psi_score', 'ss_score', 'p_score', 'resnum',
        #         'chain', 'segment', 'resname', 'iFG_name', 'label',
        #         'iFG_count', 'vdM_count', 'query_name']
        vars = ['c_phi_psi_score', 'ss_score', 'p_score', 'resnum',
                'chain', 'segment', 'resname', 'iFG_name', 'label',
                'iFG_count', 'vdM_count', 'query_name']

        # if outpath.split('.')[-1] == 'pdb':
        #     f = open(outpath, 'a')
        # elif outpath.split('.')[-1] == 'gz':
        #     f = gzip.open(outpath, 'ab')

        f = open(outpath, 'a')
        f.write('# COMBS info \n')
        f.write('# Total_COMBS_Energy= ' + str(np.round(self.opt_en, 6)) + ' \n')
        f.write('# Number of apolar ligand heavy atoms buried= ' + str(self.ligand_num_heavy_buried) + ' \n')
        f.write('# Mean distance buried of apolar ligand heavy atoms= '
                + str(np.round(self.ligand_avg_heavy_buried_dist, 5)) + ' \n')
        f.write('# BEGIN_COMBS_ENERGIES_TABLE ' + outpath.split('/')[-1] + ' \n')
        f.write(' '.join(vars) + ' \n')

        for n, v in self.opt_vdms.groupby(self.groupby):
            # _c_score = v['score'].iat[0]
            # c_score = str(np.round(_c_score, 6))
            _c_score =  v['c_phi_psi_score'].iat[0]
            c_phi_psi_score = str(np.round(_c_score, 6))
            _ss_score = v['ss_score_local'].iat[0]
            ss_score = str(np.round(_ss_score, 6))
            _p_score = v['p_score'].iat[0]
            # p_score = str(np.round(_p_score, 6))
            # _total_score = _c_score + _ss_score
            _total_score = _p_score
            total_score = str(np.round(_total_score, 6))
            seg_chain_resnum = v['seg_chain_resnum'].iat[0]
            resnum = str(seg_chain_resnum[2])
            chain = seg_chain_resnum[1]
            segment = seg_chain_resnum[0]
            resname = v['resname_vdm'].iat[0]
            iFG_name = v['iFG_name'].iat[0]
            label = v['label'].iat[0]
            iFG_count = v['iFG_count'].iat[0]
            vdM_count = v['vdM_count'].iat[0]
            query_name = v['query_name'].iat[0]
            # _vars = [c_score, c_phi_psi_score, ss_score, total_score, resnum,
            #     chain, segment, resname, iFG_name, label,
            #     iFG_count, vdM_count, query_name]
            _vars = [c_phi_psi_score, ss_score, total_score, resnum,
                chain, segment, resname, iFG_name, label,
                iFG_count, vdM_count, query_name]
            f.write(' '.join(_vars) + ' \n')

        f.write('# END_COMBS_ENERGIES_TABLE ' + outpath.split('/')[-1] + ' \n')
        f.close()

    def print_vdms(self, vdm, outpath, out_name_tag='', full_fragments=False, with_bb=False):
        for n, gr in self.vdms.groupby(self.groupby):
            self._print_vdm(vdm, n, gr, outpath, out_name_tag, full_fragments, with_bb)

    def print_hb_nets(self, min_bun_only=True, outdir='./'):
        for k, hbnet in enumerate(self.hb_net):
            self._print_hb_net(hbnet, k, min_bun_only, outdir)

    def _print_hb_net(self, hbnet, k, min_bun_only=True, outdir='./'):
        if outdir[-1] != '/':
            outdir += '/'

        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        pl = PoseLegs()
        pl.get_poselegs(hbnet)
        pl.drop_duplicates()
        min_bun = min(pl.num_buns_uniq)
        for i, (nbuns, df) in enumerate(zip(pl.num_buns_uniq, pl.poselegs_uniq)):
            can_print = True
            if min_bun_only:
                if nbuns != min_bun:
                    can_print = False
            if can_print:
                ProdyAG().print_ag('h', df, outdir, '_' + str(k) + '_' + str(i) + '_' + str(nbuns) + 'numbuns')

    def make_pose_legs(self, template):
        for hbnet in self.hb_net:
            hbnet.pose_legs = self._make_pose_legs(hbnet, template)

    def _make_pose_legs(self, hbnet, template):
        pl = PoseLegs()
        pl.get_poselegs(hbnet)
        pl.drop_duplicates()
        for n, p in zip(pl.num_buns_uniq, pl.poselegs_uniq):
        # for n, p in zip(pl.num_buns, pl.poselegs):
            p['poseleg_number'] = self._poseleg_number
            p['num_buns'] = n
            num_lig_ints, sat_acc_set, sat_don_set = self._set_num_pl_lig_bun_interactions(p, template)
            p['num_lig_ints'] = num_lig_ints
            acc = '_'.join(i for i in sat_acc_set)
            don = '_'.join(i for i in sat_don_set)
            p['sat_acc_set'] = acc
            p['sat_don_set'] = don
            self._poseleg_number += 1
        return pl

    # def _get_clash_pose_legs(self, pl1, pl2, inds1, inds2):
    #     """returns true if unique residues from each pose are clashing"""
    #     if (dists == 0).any():
    #         m = pd.merge(pl1, pl2[self.groupby], how='outer', on=self.groupby, indicator=True)
    #         pl1_uniq = m[m['_merge'] == 'left_only'].drop('_merge', axis=1)
    #         m = pd.merge(pl2, pl1[self.groupby], how='outer', on=self.groupby, indicator=True)
    #         pl2_uniq = m[m['_merge'] == 'left_only'].drop('_merge', axis=1)
    #         con = Contact(pl1_uniq, pl2_uniq)
    #     else:
    #         con = Contact(pl1, pl2)
    #     con.find()
    #     return 'cl' in set(con.df_contacts.contact_type)

    # def _get_clash_pose_legs(self, pl1, pl2, inds1, inds2):
    #     """returns true if unique residues from each pose are clashing"""
    #     if len(inds1) == len(pl1) and len(inds2) == len(pl2):
    #         con = Contact(pl1, pl2)
    #     else:
    #         con = Contact(pl1.iloc[inds1], pl2.iloc[inds2])
    #     con.find()
    #     return 'cl' in set(con.df_contacts.contact_type)
    def _get_clash_pose_legs(self, pl1, pl2, inds1, inds2):
        """returns true if unique residues from each pose are clashing"""
        if len(inds1) == len(pl1) and len(inds2) == len(pl2):
            # cla = Clash(pl1, pl2, **dict(q_grouping=['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum']))
            cla = Clash(pl1, pl2, **dict(q_grouping='str_index'))
        elif len(inds2) == 0:
            return False
        else:
            # cla = Clash(pl1.iloc[inds1], pl2.iloc[inds2], **dict(q_grouping=['iFG_count', 'vdM_count',
            #                                                                  'query_name', 'seg_chain_resnum'],
            #                                                      tol=0.05))
            cla = Clash(pl1.iloc[inds1], pl2.iloc[inds2], **dict(q_grouping='str_index', tol=0.05))
        cla.find()
        return len(cla.dfq_clash_free) != len(pl1)

    def _get_pairwise_hb_int_pose_legs(self, pl1, pl2, template, inds1, inds2, **kwargs):
        if len(inds1) == len(pl1) and len(inds2) == len(pl2):
            df = pd.concat((pl1, pl2), sort=False)
        elif len(inds2) == 0:
            df = pl1
        else:
            if ('num_tag' in pl1.columns) or ('num_tag' in pl2.columns):
                # df = pd.concat((pl1, pl2), sort=False).drop('num_tag', axis=1).drop_duplicates()
                df = pd.concat((pl1.iloc[inds1], pl2.iloc[inds2]), sort=False).drop('num_tag', axis=1)
            else:
                # df = pd.concat((pl1, pl2), sort=False).drop_duplicates()
                df = pd.concat((pl1.iloc[inds1], pl2.iloc[inds2]), sort=False)
        num_acc, num_don = get_num_bun_hb_atoms(df, template, self, **kwargs)
        return num_acc + num_don

    # @staticmethod
    # def make_its(mat_len, nproc=8):
    #     n = int(np.floor(mat_len / nproc))
    #     its = [[i, j] for i, j in zip(range(0, mat_len, n), range(n - 1, mat_len, n))]
    #     its[-1][-1] = mat_len
    #     return its
    #
    def make_pls_sorted(self):
        pls = []
        pls_nums = []
        for hbnet in self.hb_net:
            for p in hbnet.pose_legs.poselegs_uniq:
                pls.append(p)
                pls_nums.append(set(p.poseleg_number).pop())
        inds_sorted = sorted(range(len(pls)), key=lambda x: pls_nums[x])
        pls_sorted = [pls[i] for i in inds_sorted]
        self.pls_sorted = pls_sorted

    def set_lig_bb_contacts(self):
        self.lig_bb_contacts = self.lig_contacts[self.lig_contacts.iFG_count_q.isnull()]
    #
    # def make_energy_arrays_mp(self, template, nproc=8, **kwargs):
    #     its = self.make_its(len(self.pls_sorted), nproc)
    #     make_en_arrays = partial(self._make_energy_arrays, template, **kwargs)
    #     with Pool(nproc) as p:
    #         results = p.starmap(make_en_arrays, its, 1)
    #     self.pairwise_array = np.sum([r[1] for r in results], axis=0)
    #     self.single_array = np.sum([r[0] for r in results], axis=0)
    #
    # def _make_energy_arrays(self, template, m, j, **kwargs):
    #     pairwise_array = np.zeros((len(self.pls_sorted), len(self.pls_sorted)), dtype='float')
    #     single_array = np.zeros(len(self.pls_sorted), dtype='float')
    #     for i in range(m, j):
    #         pl1 = self.pls_sorted[i]
    #         single_array[i] = set(pl1.num_buns).pop() - set(pl1.num_lig_ints).pop() # * 1.5
    #         for k in range(i + 1, len(self.pls_sorted)):
    #             pl2 = self.pls_sorted[k]
    #             a1 = pl1[['c_x', 'c_y', 'c_z']].values
    #             a2 = pl2[['c_x', 'c_y', 'c_z']].values
    #             dists = cdist(a1, a2)
    #             print(set(pl1.poseleg_number).pop(), set(pl2.poseleg_number).pop())
    #             if (dists <= 4.8).any():
    #                 # print('pl1 pl2')
    #                 # print(pl1)
    #                 # print(pl2)
    #                 if is_subset(a1, a2):  # one poseleg is a subset of the other, so make them clash
    #                     pairwise_array[i, k] = 50
    #                 else:
    #                     a1_, a2_, inds1, inds2 = remove_dups(a1, a2, return_inds=True)
    #                     dists_ = cdist(a1_, a2_)
    #                     if (dists_ <= 1).any():
    #                         pairwise_array[i, k] = 50
    #                     elif self._get_clash_pose_legs(pl1, pl2, inds1, inds2):
    #                         pairwise_array[i, k] = 50
    #                     else:
    #                         atoms_acc1 = pl1[pl1.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
    #                         atoms_don1 = pl1[pl1.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
    #                         atoms_acc2 = pl2[pl2.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
    #                         atoms_don2 = pl2[pl2.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
    #                         d1 = cdist(atoms_acc1, atoms_don2)
    #                         d2 = cdist(atoms_don1, atoms_acc2)
    #                         if (d1 <= 3.25).any() or (d2 <= 3.25).any():
    #                             nb = self._get_pairwise_hb_int_pose_legs(pl1, pl2, template, inds1, inds2, **kwargs)
    #                             # new_hbs = max(( (set(pl1.num_buns).pop() + set(pl2.num_buns).pop()) / 2 ) - nb, 0)
    #                             new_hbs = max( min(set(pl1.num_buns).pop(), set(pl2.num_buns).pop()) - nb, 0)
    #                             pairwise_array[i, k] = (-1 * new_hbs)
    #                         else:
    #                             pairwise_array[i, k] = 0
    #                 # else:
    #                 #     pairwise_array[i, k] = 0
    #             else:
    #                 pairwise_array[i, k] = 0
    #             pairwise_array[k, i] = pairwise_array[i, k]
    #     return single_array, pairwise_array

    def make_energy_arrays(self, template, **kwargs):
    # def make_energy_arrays(self):
        pls = []
        pls_nums = []
        for hbnet in self.hb_net:
            for p in hbnet.pose_legs.poselegs_uniq:
                pls.append(p)
                pls_nums.append(set(p.poseleg_number).pop())
        inds_sorted = sorted(range(len(pls)), key=lambda x: pls_nums[x])
        pls_sorted = [pls[i] for i in inds_sorted]
        self.pls_sorted = pls_sorted
        pairwise_array = np.zeros((len(pls_sorted), len(pls_sorted)), dtype='float')
        single_array = np.zeros(len(pls_sorted), dtype='float')
        for i in range(len(pls_sorted)):
            pl1 = pls_sorted[i]
            single_array[i] = set(pl1.num_buns).pop() - set(pl1.num_lig_ints).pop() # * 1.5
            for k in range(i + 1, len(pls)):
                pl2 = pls_sorted[k]
                a1 = pl1[['c_x', 'c_y', 'c_z']].values
                a2 = pl2[['c_x', 'c_y', 'c_z']].values
                dists = cdist(a1, a2)
                print(set(pl1.poseleg_number).pop(), set(pl2.poseleg_number).pop())
                if (dists <= 4.8).any():
                    # print('pl1 pl2')
                    # print(pl1)
                    # print(pl2)
                    if is_subset(a1, a2):  # one poseleg is a subset of the other, so make them clash
                        pairwise_array[i, k] = 50
                    else:
                        a1_, a2_, inds1, inds2 = remove_dups(a1, a2, return_inds=True)
                        dists_ = cdist(a1_, a2_)
                        if (dists_ <= 1).any():
                            pairwise_array[i, k] = 50
                        elif self._get_clash_pose_legs(pl1, pl2, inds1, inds2):
                            pairwise_array[i, k] = 50
                        else:
                            atoms_acc1 = pl1[pl1.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_don1 = pl1[pl1.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_acc2 = pl2[pl2.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_don2 = pl2[pl2.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
                            d1 = cdist(atoms_acc1, atoms_don2)
                            d2 = cdist(atoms_don1, atoms_acc2)
                            if (d1 <= 3.25).any() or (d2 <= 3.25).any():
                                nb = self._get_pairwise_hb_int_pose_legs(pl1, pl2, template, inds1, inds2, **kwargs)
                                # new_hbs = max(( (set(pl1.num_buns).pop() + set(pl2.num_buns).pop()) / 2 ) - nb, 0)
                                new_hbs = max(max(set(pl1.num_buns).pop(), set(pl2.num_buns).pop()) - nb, 0)
                                if new_hbs == 0:
                                    pairwise_array[i, k] = 0
                                # else:
                                #     names_don1 = set(set(pl1.sat_don_set).pop().split('_')) - {''}
                                #     names_don2 = set(set(pl2.sat_don_set).pop().split('_')) - {''}
                                #     names_acc1 = set(set(pl1.sat_acc_set).pop().split('_')) - {''}
                                #     names_acc2 = set(set(pl2.sat_acc_set).pop().split('_')) - {''}
                                #     print(names_don1, names_don2, names_acc1, names_acc2)
                                #     num_lig_ints = len(names_don1 | names_don2) + len(names_acc1 | names_acc2)
                                #     print('num', num_lig_ints)
                                #     ek = set(pl2.num_buns).pop() - set(pl2.num_lig_ints).pop()
                                #     pairwise_array[i, k] = nb - num_lig_ints - single_array[i] - ek
                                # if new_hbs == 0:
                                #     pairwise_array[i, k] = 0.1
                                else:
                                    pairwise_array[i, k] = (-1 * new_hbs)
                            else:
                                pairwise_array[i, k] = 0
                    # else:
                    #     pairwise_array[i, k] = 0
                else:
                    pairwise_array[i, k] = 0
                pairwise_array[k, i] = pairwise_array[i, k]
        self.single_array = single_array
        self.pairwise_array = pairwise_array

    # def dee(self):
    #     Es = self.single_dict.copy()
    #     Ep = self.pairwise_dict.copy()
    #     del_dict = None
    #
    #     key_map = {i: key for i, key in enumerate(Es.keys())}
    #     es = np.array([v for v in Es.values()])
    #     ep = np.zeros((es.size, es.size))
    #     for i in range(es.size):
    #         for j in range(i + 1, es.size):
    #             ep[i][j] = Ep[key_map[i]][key_map[j]]
    #             ep[j][i] = ep[i][j]
    #
    #     to_del = []
    #     for key1 in Es.keys():
    #         for key2 in set(Es.keys()) - {key1}:
    #             pair_ens = []
    #             for key3 in set(Es.keys()) - {key1, key2}:
    #                 pair_ens.append(Ep[key1][key2] - Ep[key3][key2])
    #             cond = Es[key1] - Es[key2] + min(pair_ens)
    #             if cond > 0:
    #                 to_del.append(key1)
    #                 break
    #     if to_del:
    #         for key in to_del:
    #             Es.pop(key)
    #             Ep.pop(key)
    #             for k in Ep.keys():
    #                 Ep[k].pop(key)
    #         del_dict = to_del
    #
    #     self.Es_dee = Es
    #     self.Ep_dee = Ep
    #     self.del_dict = del_dict

    def dee(self):
        # Es = copy.deepcopy(self.single_dict)
        # Ep = copy.deepcopy(self.pairwise_dict)

        n = self.single_array.size
        es = self.single_array
        ep = self.pairwise_array
        # es = np.array([v for v in Es.values()], dtype='float')
        # ep = np.zeros((es.size, es.size), dtype='float')
        # for i in range(es.size):
        #     for j in range(i + 1, es.size):
        #         ep[i][j] = Ep[i][j]
        #         ep[j][i] = ep[i][j]

        to_del = _dee(es, ep)
        es = np.delete(es, to_del)
        ep = np.delete(ep, to_del, axis=0)
        ep = np.delete(ep, to_del, axis=1)
        self.pls_sorted_dee = [self.pls_sorted[i] for i in range(n) if i not in to_del]

        # to_del = list(to_del)
        #
        # if to_del:
        #     for key in to_del:
        #         Es.pop(key)
        #         Ep.pop(key)
        #         for k in Ep.keys():
        #             Ep[k].pop(key)
        #     del_dict = to_del

        self.Es_dee = es
        self.Ep_dee = ep
        self.to_delete = to_del

    # def run_mc(self, num_pose_legs, trials=10000, kT=1):
    #     all_keys = set(range(len(self.single_array)))
    #     _keys = random.sample(list(all_keys), num_pose_legs)
    #     all_keys_minus = all_keys - set(_keys)
    #     e_old = self._energy_fn_singles(_keys) + self._energy_fn_pairs(_keys)
    #     energies = []
    #     keys = []
    #     energies.append(e_old)
    #     keys.append(_keys)
    #     for_pop = list(range(num_pose_legs))
    #     for _ in range(trials):
    #         old_keys = copy.deepcopy(_keys)
    #         _keys.pop(random.choice(for_pop))
    #         _keys.append(random.choice(list(all_keys_minus)))
    #         e = self._energy_fn_singles(_keys) + self._energy_fn_pairs(_keys)
    #         p = min(1, np.exp(-(e - e_old) / kT))
    #         if np.random.rand() <= p:
    #             energies.append(e)
    #             keys.append(copy.deepcopy(_keys))
    #             e_old = e
    #         else:
    #            _keys = old_keys
    #         all_keys_minus = all_keys - set(_keys)
    #     self.mc_energies = energies
    #     self.mc_keys = keys

    # def _make_nonpairwise_en(self, _keys, template, burial_depth=1, **kwargs):
    #     clash_en = self._calc_clash_en(_keys)
    #     if clash_en > 0:
    #         return clash_en
    #
    #     df = pd.concat((self.pls_sorted[i] for i in _keys), sort=False)
    #     if ('num_tag' in df.columns):
    #         df = df.drop('num_tag', axis=1).drop_duplicates()
    #     else:
    #         df = df.drop_duplicates()
    #     numacc, numdon = get_num_bun_hb_atoms(df, template, self, burial_depth, **kwargs)
    #     df_con = Contact(df, self.ligand)
    #     df_con.find()
    #     contacts_ = df_con.df_contacts
    #     all_lig_contacts = pd.concat((contacts_, self.lig_bb_contacts), sort=False)
    #     lig = Ligand()
    #     lig.csts_gr = self.lig_csts_gr
    #     if not lig.check_csts(self.ligand, all_lig_contacts):
    #         return 300
    #     lig_bun_acc, lig_bun_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=contacts_)
    #     num_lig_ints = self.num_buns_lig - len(lig_bun_acc) - len(lig_bun_don)
    #     num_buns_pls = numacc + numdon
    #     # print(num_buns_pls, num_lig_ints)
    #     return num_buns_pls - num_lig_ints + clash_en
    def _make_nonpairwise_en(self, _keys, template, burial_depth=1, **kwargs):
        clash_en = self._calc_clash_en(_keys)
        if clash_en > 0:
            return clash_en

        df = pd.concat((self.pls_sorted[i] for i in _keys), sort=False)
        if ('num_tag' in df.columns):
            df = df.drop('num_tag', axis=1).drop_duplicates()
        else:
            df = df.drop_duplicates()
        acc, don = _get_bun_hb_atoms(df, template, self, burial_depth, **kwargs)
        costs = 0
        if len(acc) > 0:
            costs += np.sum(acc['distance_to_hull'].values/50)
        if len(don) > 0:
            costs += np.sum(don['distance_to_hull'].values/50)
        df_con = Contact(df, self.ligand)
        df_con.find()
        contacts_ = df_con.df_contacts
        all_lig_contacts = pd.concat((contacts_, self.lig_bb_contacts), sort=False)
        lig = Ligand()
        lig.csts_gr = self.lig_csts_gr
        if not lig.check_csts(self.ligand, all_lig_contacts):
            return 50
        lig_bun_acc, lig_bun_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=contacts_)
        if len(lig_bun_acc) > 0:
            costs += np.sum(lig_bun_acc['distance_to_hull'].values/40)
        if len(lig_bun_don) > 0:
            costs += np.sum(lig_bun_don['distance_to_hull'].values/40)
        return costs

    def make_clash_en_matrix(self):
        print('making clash matrix...')
        pairwise_array = np.zeros((len(self.pls_sorted), len(self.pls_sorted)), dtype='float')
        for i in range(len(self.pls_sorted)):
            pl1 = self.pls_sorted[i]
            for k in range(i + 1, len(self.pls_sorted)):
                # print(i, k)
                pl2 = self.pls_sorted[k]
                a1 = pl1[['c_x', 'c_y', 'c_z']].values
                a2 = pl2[['c_x', 'c_y', 'c_z']].values
                dists = cdist(a1, a2)
                # print(set(pl1.poseleg_number).pop(), set(pl2.poseleg_number).pop())
                if (dists <= 4.8).any():
                    if is_subset(a1, a2):  # one poseleg is a subset of the other, so make them clash
                        pairwise_array[i, k] = 50
                    else:
                        a1_, a2_, inds1, inds2 = remove_dups(a1, a2, return_inds=True)
                        dists_ = cdist(a1_, a2_)
                        if (dists_ <= 1).any():
                            pairwise_array[i, k] = 50
                        elif self._get_clash_pose_legs(copy.deepcopy(pl1), copy.deepcopy(pl2), inds1, inds2):
                            pairwise_array[i, k] = 50
        pairwise_array = pairwise_array + pairwise_array.T
        self.clash_matrix = pairwise_array

    def make_single_array(self):
        single_array = np.zeros(len(self.pls_sorted), dtype='float')
        for i in range(len(self.pls_sorted)):
            pl1 = self.pls_sorted[i]
            single_array[i] = set(pl1.num_buns).pop() - set(pl1.num_lig_ints).pop()
        self.single_array = single_array

    # def make_weights(self, kT=2):
    def make_weights(self):
        # Q = np.sum(np.exp(-1 * self.single_array / kT))
        # self.weights = [np.exp(-1 * i / kT) / Q for i in self.single_array]
        un = np.unique(self.single_array)
        n = len(un)
        self.weights = np.zeros(len(self.single_array))
        for u in un:
            self.weights[self.single_array == u] = n
            if u > 0:
                n -= 1

    def _calc_clash_en(self, _keys):
        return sum(self.clash_matrix[i, j] for i, j in itertools.combinations(_keys, 2))

    # def run_mc_nonpairwise(self, num_pose_legs, template, burial_depth=1, trials=10000, kT=1, **kwargs):  # test nopairwise decomp speed
    #     all_keys = set(range(len(self.pls_sorted)))
    #     _keys = random.sample(list(all_keys), num_pose_legs)
    #     all_keys_minus = all_keys - set(_keys)
    #     e_old = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
    #     energies = []
    #     keys = []
    #     energies.append(e_old)
    #     keys.append(_keys)
    #     for_pop = list(range(num_pose_legs))
    #     for i in range(trials):
    #         if i % 100 == 0:
    #             print('iteration', i + 1)
    #         old_keys = copy.deepcopy(_keys)
    #         _keys.pop(random.choice(for_pop))
    #         _keys.append(random.choice(list(all_keys_minus)))
    #         e = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
    #         p = min(1, np.exp(-(e - e_old) / kT))
    #         if np.random.rand() <= p:
    #             print('old', e_old, 'new', e, 'accepted')
    #             energies.append(e)
    #             keys.append(copy.deepcopy(_keys))
    #             e_old = e
    #         else:
    #             print('old', e_old, 'new', e, 'rejected')
    #             _keys = old_keys
    #         all_keys_minus = all_keys - set(_keys)
    #     self.mc_energies = energies
    #     self.mc_keys = keys

    def run_mc_nonpairwise(self, num_pose_legs, template, burial_depth=1, trials=10000, kT=1, num_non_clash=500, **kwargs):  # test nopairwise decomp speed
        all_keys = range(len(self.pls_sorted))
        _keys = random.sample(list(all_keys), num_pose_legs)
        e_old = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
        energies = []
        keys = []
        energies.append(e_old)
        keys.append(_keys)
        for_pop = list(range(num_pose_legs))
        j = 0
        for i in range(trials):
            if i % 100 == 0:
                print('iteration', i + 1)
            old_keys = copy.deepcopy(_keys)
            oldkey = 1
            newkey = 1
            while oldkey == newkey:
                oldkey = _keys.pop(random.choice(for_pop))
                newkey = random.choices(all_keys, weights=self.weights, k=1)[0]
                if newkey in _keys:
                    _keys.append(oldkey)
                    oldkey = newkey
                else:
                    _keys.append(newkey)
            e = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
            if e < 10:
                j += 1
            p = min(1, np.exp(-(e - e_old) / kT))
            if np.random.rand() <= p:
                print('old', e_old, 'new', e, 'accepted')
                energies.append(e)
                keys.append(copy.deepcopy(_keys))
                e_old = e
            else:
                print('old', e_old, 'new', e, 'rejected')
                _keys = old_keys
            if j == num_non_clash:
                break
            if j % 50 == 0:
                print('nonclashing', j + 1)
        self.mc_energies = energies
        self.mc_keys = keys

    def run_sim_anneal(self, num_pose_legs, template, annealing_sched, burial_depth=1, trials=10000,
                       **kwargs):  # test nopairwise decomp speed
        all_keys = range(len(self.pls_sorted))
        _keys = random.sample(list(all_keys), num_pose_legs)
        e_old = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
        energies = []
        keys = []
        energies.append(e_old)
        keys.append(_keys)
        for_pop = list(range(num_pose_legs))
        for i in range(trials):
            kT = annealing_sched[i]
            if i % 100 == 0:
                print('iteration', i + 1, 'kT=', kT)
            old_keys = copy.deepcopy(_keys)
            oldkey = 1
            newkey = 1
            while oldkey == newkey:
                oldkey = _keys.pop(random.choice(for_pop))
                newkey = random.choices(all_keys, weights=self.weights, k=1)[0]
                if newkey in _keys:
                    _keys.append(oldkey)
                    oldkey = newkey
                else:
                    _keys.append(newkey)
            e = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
            p = min(1, np.exp(-(e - e_old) / kT))
            if np.random.rand() <= p:
                print('old', e_old, 'new', e, 'accepted')
                energies.append(e)
                keys.append(copy.deepcopy(_keys))
                e_old = e
            else:
                print('old', e_old, 'new', e, 'rejected')
                _keys = old_keys
        self.mc_energies = energies
        self.mc_keys = keys

    # def _energy_fn(self, key1, key2):
    #     return self.Es_dee[key1] + self.Es_dee[key2] + self.Ep_dee[key1][key2]

    def _energy_fn_singles(self, keys):
        return sum(self.single_array[key] for key in keys)

    def _energy_fn_pairs(self, keys):
        return sum(self.pairwise_array[key1, key2] for key1, key2 in itertools.combinations(keys, 2))
        # return sum(0 if (self.pairwise_array[key1, key2] == 0)
        #            else (self.pairwise_array[key1, key2] + self.single_array[key1] + self.single_array[key2])
        #            for key1, key2 in itertools.combinations(keys, 2))



def make_cst_df(path_to_cst_file):
    groups = list()
    resnames = list()
    names = list()
    contacts = list()
    burials = list()
    dists_buried = list()
    dists_lessthan = list()
    with open(path_to_cst_file, 'r') as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            spl = line.split()
            if len(spl) < 1:
                continue
            group = int(spl[0].strip())
            resname = spl[1].strip()
            name = spl[2].strip()
            try:
                CO_ind = spl.index('CO')
            except ValueError:
                CO_ind = None
            try:
                BU_ind = spl.index('BU')
            except ValueError:
                BU_ind = None
            try:
                DB_ind = spl.index('DB')
            except ValueError:
                DB_ind = None

            if BU_ind:
                burial = spl[BU_ind + 1]
                if burial == 'buried':
                    burial = True
                elif burial == 'exposed':
                    burial = False
                else:
                    raise ValueError('burial must be "exposed" or "buried"')
            else:
                burial = None

            if DB_ind:
                dist = spl[DB_ind + 1]
                if dist[0] == '<':
                    dist_lessthan = True
                elif dist[0] == '>':
                    dist_lessthan = False
                else:
                    raise ValueError('distance buried must be "<" or ">" a number, e.g. "<0.5')
                dist = float(dist[1:])
            else:
                dist = None
                dist_lessthan = None

            # CO_set = set()
            # if CO_ind and BU_ind:
            #     if CO_ind < BU_ind:
            #         for co in spl[CO_ind + 1:BU_ind]:
            #             CO_set |= {co}
            #     else:
            #         for co in spl[BU_ind + 2:]:
            #             CO_set |= {co}
            # elif CO_ind and not BU_ind:
            #     for co in spl[CO_ind + 1:]:
            #         CO_set |= {co}
            CO_set = set()
            if CO_ind:
                CO_set = set(spl[CO_ind + 1].split(','))

            groups.append(group)
            resnames.append(resname)
            names.append(name)
            contacts.append(CO_set)
            burials.append(burial)
            dists_buried.append(dist)
            dists_lessthan.append(dist_lessthan)
    data = dict(cst_group=groups, lig_resname=resnames, lig_name=names,
                contact_type=contacts, burial=burials, dist_buried=dists_buried,
                dist_buried_lessthan=dists_lessthan)
    return pd.DataFrame(data)


acceptor_atom_types = ['n', 'o', 'f']

# path_to_sig_dict = defaultdict(dict)
# path_to_sig_dict['SER'] = '/Volumes/bkup1/combs/database/representatives/hb_only/hydroxyl/20181009/'
# path_to_sig_dict['THR'] = '/Volumes/bkup1/combs/database/representatives/hb_only/hydroxyl/20181009/'
# path_to_sig_dict['TYR'] = '/Volumes/bkup1/combs/database/representatives/hb_only/hydroxyl/20181009/'
# path_to_sig_dict['ASN'] = '/Volumes/bkup1/combs/database/representatives/hb_only/carboxamide/20181002/'
# path_to_sig_dict['GLN'] = '/Volumes/bkup1/combs/database/representatives/hb_only/carboxamide/20181002/'
# path_to_sig_dict['ASP'] = '/Volumes/bkup1/combs/database/representatives/hb_only/carboxylate/20181009/'
# path_to_sig_dict['GLU'] = '/Volumes/bkup1/combs/database/representatives/hb_only/carboxylate/20181009/'
# path_to_sig_dict['ARG'] = '/Volumes/bkup1/combs/database/representatives/hb_only/arginine/20181009/'
# path_to_sig_dict['LYS'] = '/Volumes/bkup1/combs/database/representatives/hb_only/lysine/20181009/'
# path_to_sig_dict['backboneNH'] = '/Volumes/bkup1/combs/database/representatives/hb_only/backboneNH/20181002/'
# path_to_sig_dict['backboneCO'] = '/Volumes/bkup1/combs/database/representatives/hb_only/backboneCO/20181002/'
# path_to_sig_dict['HIS'] = defaultdict(dict)
# path_to_sig_dict['HIS']['ND1']['ACC'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoleacc/20181009/'
# path_to_sig_dict['HIS']['NE2']['ACC'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoleacc/20181009/'
# path_to_sig_dict['HIS']['ND1']['DON'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoledon/20181009/'
# path_to_sig_dict['HIS']['NE2']['DON'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoledon/20181009/'
# path_to_sig_dict['HIS']['HD1']['DON'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoledon/20181009/'
# path_to_sig_dict['HIS']['HE2']['DON'] = '/Volumes/bkup1/combs/database/representatives/hb_only/imidazoledon/20181009/'

path_to_sig_dict = defaultdict(dict)
path_to_sig_dict['SER'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['THR'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['TYR'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['ASN'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/carboxamide/20181002/'
path_to_sig_dict['GLN'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/carboxamide/20181002/'
path_to_sig_dict['ASP'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/carboxylate/20181009/'
path_to_sig_dict['GLU'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/carboxylate/20181009/'
path_to_sig_dict['ARG'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/arginine/20181009/'
path_to_sig_dict['LYS'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/lysine/20181009/'
path_to_sig_dict['backboneNH'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/backboneNH/20181002/'
path_to_sig_dict['backboneCO'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/backboneCO/20181002/'
path_to_sig_dict['HIS'] = defaultdict(dict)
path_to_sig_dict['HIS']['ND1']['ACC'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoleacc/20181009/'
path_to_sig_dict['HIS']['NE2']['ACC'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoleacc/20181009/'
path_to_sig_dict['HIS']['ND1']['DON'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['NE2']['DON'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['HD1']['DON'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['HE2']['DON'] = '/wynton/scratch/nick.polizzi/combs/database/representatives/hb_only/imidazoledon/20181009/'


dict_corr_dict = defaultdict(dict)
#hydroxyl
dict_corr_dict['SER'] = dict(SER=dict(SER=dict(CB='CB', OG='OG'),
                                      THR=dict(CB='CB', OG1='OG'),
                                      TYR=dict(CZ='CB', OH='OG')))
dict_corr_dict['THR'] = dict(THR=dict(SER=dict(CB='CB', OG='OG1'),
                                      THR=dict(CB='CB', OG1='OG1'),
                                      TYR=dict(CZ='CB', OH='OG1')))
dict_corr_dict['TYR'] = dict(TYR=dict(SER=dict(CB='CZ', OG='OH'),
                                      THR=dict(CB='CZ', OG1='OH'),
                                      TYR=dict(CZ='CZ', OH='OH')))
#carboxamide
dict_corr_dict['ASN'] = dict(ASN=dict(GLN=dict(NE2='ND2', CD='CG', OE1='OD1', CG='CB'),
                                      ASN=dict(ND2='ND2', CG='CG', OD1='OD1', CB='CB')))
dict_corr_dict['GLN'] = dict(GLN=dict(GLN=dict(NE2='NE2', CD='CD', OE1='OE1', CG='CG'),
                                      ASN=dict(ND2='NE2', CG='CD', OD1='OE1', CB='CG')))

#carboxylate
dict_corr_dict['ASP'] = dict(ASP=dict(GLU=dict(OE2='OD2', CD='CG', OE1='OD1', CG='CB'),
                                      ASP=dict(OD2='OD2', CG='CG', OD1='OD1', CB='CB')))
dict_corr_dict['GLU'] = dict(GLU=dict(GLU=dict(OE2='OE2', CD='CD', OE1='OE1', CG='CG'),
                                      ASP=dict(OD2='OE2', CG='CD', OD1='OE1', CB='CG')))

#guanidinium
dict_corr_dict['ARG'] = dict(ARG=dict(ARG=dict(NE='NE', CZ='CZ', NH1='NH1', NH2='NH2')))

#amino
dict_corr_dict['LYS'] = dict(LYS=dict(LYS=dict(CE='CE', NZ='NZ')))
dict_corr_dict['HIS'] = defaultdict(dict)
dict_corr_dict['HIS']['NE2']['ACC'] = dict(HIS=dict(HID=dict(CE1='CE1', NE2='NE2', CD2='CD2'),
                                                    HIE=dict(CE1='CE1', ND1='NE2', CG='CD2')
                                                    ))
dict_corr_dict['HIS']['HE2']['DON'] = dict(HIS=dict(HIE=dict(CE1='CE1', NE2='NE2', CD2='CD2'),
                                                    HID=dict(CE1='CE1', ND1='NE2', CG='CD2'),
                                                    TRP=dict(CD1='CE1', NE1='NE2', CE2='CD2')
                                                    ))
dict_corr_dict['HIS']['ND1']['ACC'] = dict(HIS=dict(HID=dict(CE1='CE1', NE2='ND1', CD2='CG'),
                                                    HIE=dict(CE1='CE1', ND1='ND1', CG='CG')
                                                    ))
dict_corr_dict['HIS']['HD1']['DON'] = dict(HIS=dict(HIE=dict(CE1='CE1', NE2='ND1', CD2='CG'),
                                                    HID=dict(CE1='CE1', ND1='ND1', CG='CG'),
                                                    TRP=dict(CD1='CE1', NE1='ND1', CE2='CG')
                                                    ))

dict_corr_dict['backboneNH'] = defaultdict(dict)
dict_corr_dict['backboneCO'] = defaultdict(dict)
dict_corr_dict['backboneNH']['HIS'] = defaultdict(dict)
dict_corr_dict['backboneCO']['HIS'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ASN'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ASN']['HD22'] = dict(ASN=dict(GLN={'HE21': 'HD22', 'NE2': 'ND2'},
                                                            ASN={'HD21': 'HD22', 'ND2': 'ND2'},
                                                            ALA={'H': 'HD22', 'N': 'ND2'},
                                                            GLY={'H': 'HD22', 'N': 'ND2'}))
dict_corr_dict['backboneNH']['ASN']['HD21'] = dict(ASN=dict(GLN={'HE21': 'HD21', 'NE2': 'ND2'},
                                                            ASN={'HD21': 'HD21', 'ND2': 'ND2'},
                                                            ALA={'H': 'HD21', 'N': 'ND2'},
                                                            GLY={'H': 'HD21', 'N': 'ND2'}))
dict_corr_dict['backboneNH']['GLN'] = defaultdict(dict)
dict_corr_dict['backboneNH']['GLN']['HE22'] = dict(GLN=dict(GLN=dict(HE21='HE22', NE2='NE2'),
                                                    ASN=dict(HD21='HE22', ND2='NE2'),
                                                    ALA=dict(H='HE22', N='NE2'),
                                                    GLY=dict(H='HE22', N='NE2')))
dict_corr_dict['backboneNH']['GLN']['HE21'] = dict(GLN=dict(GLN=dict(HE21='HE21', NE2='NE2'),
                                                    ASN=dict(HD21='HE21', ND2='NE2'),
                                                    ALA=dict(H='HE21', N='NE2'),
                                                    GLY=dict(H='HE21', N='NE2')))
dict_corr_dict['backboneNH']['ARG'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ARG']['HE'] = dict(ARG=dict(GLN=dict(HE21='HE', NE2='NE'),
                                                    ASN=dict(HD21='HE', ND2='NE'),
                                                    ALA=dict(H='HE', N='NE'),
                                                    GLY=dict(H='HE', N='NE')))
dict_corr_dict['backboneNH']['ARG']['HH21'] = dict(ARG=dict(GLN=dict(HE21='HH21', NE2='NH2'),
                                                    ASN=dict(HD21='HH21', ND2='NH2'),
                                                    ALA=dict(H='HH21', N='NH2'),
                                                    GLY=dict(H='HH21', N='NH2')))
dict_corr_dict['backboneNH']['ARG']['HH22'] = dict(ARG=dict(GLN=dict(HE21='HH22', NE2='NH2'),
                                                    ASN=dict(HD21='HH22', ND2='NH2'),
                                                    ALA=dict(H='HH22', N='NH2'),
                                                    GLY=dict(H='HH22', N='NH2')))
dict_corr_dict['backboneNH']['ARG']['HH12'] = dict(ARG=dict(GLN=dict(HE21='HH12', NE2='NH1'),
                                                    ASN=dict(HD21='HH12', ND2='NH1'),
                                                    ALA=dict(H='HH12', N='NH1'),
                                                    GLY=dict(H='HH12', N='NH1')))
dict_corr_dict['backboneNH']['ARG']['HH11'] = dict(ARG=dict(GLN=dict(HE21='HH11', NE2='NH1'),
                                                    ASN=dict(HD21='HH11', ND2='NH1'),
                                                    ALA=dict(H='HH11', N='NH1'),
                                                    GLY=dict(H='HH11', N='NH1')))
dict_corr_dict['backboneNH']['LYS'] = defaultdict(dict)
dict_corr_dict['backboneNH']['LYS']['HZ1'] = dict(LYS=dict(GLN=dict(HE21='HZ1', NE2='NZ'),
                                                    ASN=dict(HD21='HZ1', ND2='NZ'),
                                                    ALA=dict(H='HZ1', N='NZ'),
                                                    GLY=dict(H='HZ1', N='NZ')))
dict_corr_dict['backboneNH']['LYS']['HZ2'] = dict(LYS=dict(GLN=dict(HE21='HZ2', NE2='NZ'),
                                                    ASN=dict(HD21='HZ2', ND2='NZ'),
                                                    ALA=dict(H='HZ2', N='NZ'),
                                                    GLY=dict(H='HZ2', N='NZ')))
dict_corr_dict['backboneNH']['LYS']['HZ3'] = dict(LYS=dict(GLN=dict(HE21='HZ3', NE2='NZ'),
                                                    ASN=dict(HD21='HZ3', ND2='NZ'),
                                                    ALA=dict(H='HZ3', N='NZ'),
                                                    GLY=dict(H='HZ3', N='NZ')))

dict_corr_dict['backboneNH']['HIS']['HE2'] = dict(HIS=dict(GLN=dict(HE21='HE2', NE2='NE2'),
                                                           ASN=dict(HD21='HE2', ND2='NE2'),
                                                           ALA=dict(H='HE2', N='NE2'),
                                                           GLY=dict(H='HE2', N='NE2')))
dict_corr_dict['backboneNH']['HIS']['HD1'] = dict(HIS=dict(GLN=dict(HE21='HD1', NE2='ND1'),
                                                                  ASN=dict(HD21='HD1', ND2='ND1'),
                                                                  ALA=dict(H='HD1', N='ND1'),
                                                                  GLY=dict(H='HD1', N='ND1')))
dict_corr_dict['backboneCO']['HIS']['ND1'][1] = dict(HIS=dict(GLN=dict(OE1='ND1', CD='CG'),
                                                                  ASN=dict(OD1='ND1', CG='CG'),
                                                                  ALA=dict(O='ND1', C='CG'),
                                                                  GLY=dict(O='ND1', C='CG')))
dict_corr_dict['backboneCO']['HIS']['ND1'][2] = dict(HIS=dict(GLN=dict(OE1='ND1', CD='CE1'),
                                                                  ASN=dict(OD1='ND1', CG='CE1'),
                                                                  ALA=dict(O='ND1', C='CE1'),
                                                                  GLY=dict(O='ND1', C='CE1')))
dict_corr_dict['backboneCO']['HIS']['NE2'][1] = dict(HIS=dict(GLN=dict(OE1='NE2', CD='CE1'),
                                                                  ASN=dict(OD1='NE2', CG='CE1'),
                                                                  ALA=dict(O='NE2', C='CE1'),
                                                                  GLY=dict(O='NE2', C='CE1')))
dict_corr_dict['backboneCO']['HIS']['NE2'][2] = dict(HIS=dict(GLN=dict(OE1='NE2', CD='CD2'),
                                                                  ASN=dict(OD1='NE2', CG='CD2'),
                                                                  ALA=dict(O='NE2', C='CD2'),
                                                                  GLY=dict(O='NE2', C='CD2')))
dict_corr_dict['backboneCO']['ASN'] = defaultdict(dict)
dict_corr_dict['backboneCO']['GLN'] = defaultdict(dict)
dict_corr_dict['backboneCO']['ASP'] = defaultdict(dict)
dict_corr_dict['backboneCO']['GLU'] = defaultdict(dict)
dict_corr_dict['backboneCO']['ASN']['OD1'] = dict(ASN=dict(GLN=dict(OE1='OD1', CD='CG'),
                                                    ASN=dict(OD1='OD1', CG='CG'),
                                                    ALA=dict(O='OD1', C='CG'),
                                                    GLY=dict(O='OD1', C='CG')))
dict_corr_dict['backboneCO']['GLN']['OE1'] = dict(GLN=dict(GLN=dict(OE1='OE1', CD='CD'),
                                                    ASN=dict(OD1='OE1', CG='CD'),
                                                    ALA=dict(O='OE1', C='CD'),
                                                    GLY=dict(O='OE1', C='CD')))
dict_corr_dict['backboneCO']['ASP']['OD1'] = dict(ASP=dict(GLN=dict(OE1='OD1', CD='CG'),
                                                    ASN=dict(OD1='OD1', CG='CG'),
                                                    ALA=dict(O='OD1', C='CG'),
                                                    GLY=dict(O='OD1', C='CG')))
dict_corr_dict['backboneCO']['ASP']['OD2'] = dict(ASP=dict(GLN=dict(OE1='OD2', CD='CG'),
                                                    ASN=dict(OD1='OD2', CG='CG'),
                                                    ALA=dict(O='OD2', C='CG'),
                                                    GLY=dict(O='OD2', C='CG')))
dict_corr_dict['backboneCO']['GLU']['OE1'] = dict(GLU=dict(GLN=dict(OE1='OE1', CD='CD'),
                                                    ASN=dict(OD1='OE1', CG='CD'),
                                                    ALA=dict(O='OE1', C='CD'),
                                                    GLY=dict(O='OE1', C='CD')))
dict_corr_dict['backboneCO']['GLU']['OE2'] = dict(GLU=dict(GLN=dict(OE1='OE2', CD='CD'),
                                                    ASN=dict(OD1='OE2', CG='CD'),
                                                    ALA=dict(O='OE2', C='CD'),
                                                    GLY=dict(O='OE2', C='CD')))

backboneNHs = ['HIS', 'ASN', 'GLN', 'ARG', 'LYS']
backboneCOs = ['HIS', 'ASN', 'GLN', 'ASP', 'GLU']

remove_from_df_dict = defaultdict(dict)
remove_from_df_dict['SER'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['THR'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['TYR'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['LYS'] = {1: {'chain': 'Y', 'name': 'CD'}}
remove_from_df_dict['backboneNH'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CG', 'resname': 'ASN'},
                                     3: {'chain': 'Y', 'name': 'CD', 'resname': 'GLN'}}
remove_from_df_dict['backboneCO'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CB'},
                                     3: {'chain': 'Y', 'name': 'CG', 'resname': 'GLN'}}

class HBNet:
    def __init__(self):
        self.primary = pd.DataFrame()
        self.num_buns = 0
        self.secondary = dict()  # Pose()
        self.num_lig_ints = 0
        self.pose_legs = None


class SecondaryPoses:
    def __init__(self, **kwargs):
        self.poses = None
        self.template = None
        self.vdm_dict = defaultdict(dict)
        self.num_recusion = 0
        self.recursion_limit = kwargs.get('recursion_limit', 2)
        self.path_to_sig_dict = kwargs.get('path_to_sig_dict', path_to_sig_dict)
        self.dict_corr_dict = kwargs.get('dict_corr_dict', dict_corr_dict)
        self.remove_from_df_dict = kwargs.get('remove_from_df_dict', remove_from_df_dict)
        self.do_not_design = kwargs.get('do_not_design')  # format of list of seg_chain_resnums, e.g. [('A', 'A', 5), ...]

    def load_primary_poses(self, poses):
        self.poses = poses

    def load_template(self, template):
        self.template = template

    def find_secondary(self, outdir=None, **kwargs):
        reset_dict = kwargs.get('reset_dict', True)
        j = 0
        for k, pose in enumerate(self.poses):
            self._find_secondary(pose, outdir, k, **kwargs)
            j += 1
            if reset_dict and j == 3:
                self.vdm_dict = defaultdict(dict)
                j = 0
    def _find_secondary(self, pose, outdir=None, pose_num=1, **kwargs):
        """

        :param pose:
        :param template:
        :return:


        1). get set of hb atoms of ligand that are hbonding.
        2a). for each hb atom:
            get vdms that are hbonding.
            2b). for each vdm:
                    get set of possible hb atoms.
                    for each hb atom of vdm:
                        if hb atom is not hbonding (or is not solvent exposed):
                            -load vdms (SC and SC-phipsi only) onto residues of template with CA atoms w/in 10A of hb atom
                            -prune clashing vdms (template and ligand and vdm)
                            -find neighbors (include first shell vdms from pose)
                            -if there are neighbors, check if they satisfy hbonds of the vdm.

        """

        burialdepth = kwargs.get('burialdepth', 1)
        hb_contacts = pose.lig_contacts[pose.lig_contacts.contact_type == 'hb']
        df_hb = get_vdms_hbonding_to_lig(pose, hb_contacts)
        if len(df_hb) > 0:
            vdmrep = VdmReps(df_hb, **dict(grouping=pose.groupby))
            vdmrep.find_all_reps_dict(rmsd=0.4)
            df_hb = pd.concat(vdmrep.reps_dict.values())
            has_hbs = ''
        else:
            has_hbs = '_no_hb'

        df_hb_gr = df_hb.groupby(pose.groupby)
        print('Searching through ' + str(len(df_hb_gr)) + ' vdMs for secondary H-bonders...')
        recurse = kwargs.get('recurse')
        if recurse is not None:
            if self.num_recusion >= self.recursion_limit:
                print('Breaking recursion')
                self.num_recusion -= 1
                for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                    if recurse is not None:
                        print('recursive vdM number ' + str(i + 1) + '...')
                    else:
                        print('first-shell vdM number ' + str(i + 1) + '...')
                    # print('original vdm_df:')
                    # print(vdm_df)
                    vdm_df = vdm_df.copy()
                    print(set(vdm_df.resname_vdm).pop())
                    num_accs, num_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                          contacts=None, pose=pose, grouping=None, **kwargs)
                    # print('vdm_df after get bun:')
                    # print(vdm_df)
                    hbnet = HBNet()
                    hbnet.num_buns = num_accs + num_dons
                    hbnet.primary = vdm_df

                    # dist_bur = self.template.alpha_hull.get_pnts_distance(lig[['c_x', 'c_y', 'c_z']].values)
                    # lig.loc[:, 'dist_in_hull'] = dist_bur

                    pose.hb_net.append(hbnet)
            else:
                for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                    if recurse is not None:
                        print('recursive vdM number ' + str(i + 1) + '...')
                    else:
                        print('first-shell vdM number ' + str(i + 1) + '...')
                    # print('original vdm_df:')
                    # print(vdm_df)
                    vdm_df = vdm_df.copy()
                    print(set(vdm_df.resname_vdm).pop())
                    bun_accs, bun_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                          contacts=None, pose=pose, grouping=None)
                    # print('vdm_df after get bun:')
                    # print(vdm_df)
                    hbnet = HBNet()
                    hbnet.num_buns = len(bun_accs) + len(bun_dons)
                    hbnet.primary = vdm_df

                    if recurse is not None:
                        num_accs, num_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                              contacts=None, pose=pose, grouping=None, **kwargs)
                        hbnet.num_buns = num_accs + num_dons
                    num_vdm_recusion = self.num_recusion
                    for n, atom_row in bun_accs.iterrows():
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='ACC')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='ACC', **kwargs)
                    for n, atom_row in bun_dons.iterrows():
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='DON')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='DON', **kwargs)
                    pose.hb_net.append(hbnet)
        else:
            self.num_recusion = 0

            for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                print('first-shell vdM number ' + str(i + 1) + '...')
                # print('original vdm_df:')
                # print(vdm_df)
                vdm_df = vdm_df.copy()
                print(set(vdm_df.resname_vdm).pop())
                bun_accs, bun_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                      contacts=None, pose=pose, grouping=None)
                # print('vdm_df after get bun:')
                # print(vdm_df)
                accs, dons = get_hb_atoms(vdm_df)
                hbnet = HBNet()
                hbnet.num_buns = len(bun_accs) + len(bun_dons)
                hbnet.primary = vdm_df
                num_vdm_recusion = self.num_recusion
                # for n, atom_row in bun_accs.iterrows():
                for n, atom_row in accs.iterrows():  # find buttressing interactions for all polar atoms in first shell
                    if atom_row['resname_vdm'] != 'TRP':
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='ACC')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='ACC', **kwargs)
                # for n, atom_row in bun_dons.iterrows():
                for n, atom_row in dons.iterrows():
                    if atom_row['resname_vdm'] != 'TRP':
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='DON')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='DON', **kwargs)
                pose.hb_net.append(hbnet)

        if outdir is not None:
            if outdir[-1] != '/':
                outdir += '/'

            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass
            with open(outdir + 'pose' + str(pose_num) + has_hbs + '.pkl', 'wb') as outfile:
                pickle.dump(pose, outfile)

    def load_sec(self, atom_row, hb_type):
        sel = self.template.pdb.select('name CA within 10 of c',
                                       c=np.array([atom_row.c_x, atom_row.c_y,
                                                   atom_row.c_z]))
        if self.do_not_design is None:
            template_seg_chain_resnums = set(
                zip(sel.getSegnames(), sel.getChids(), sel.getResnums())) - \
                                     {atom_row.seg_chain_resnum}
        else:
            template_seg_chain_resnums = set(
                zip(sel.getSegnames(), sel.getChids(), sel.getResnums())) - \
                                         {atom_row.seg_chain_resnum} - set(self.do_not_design)

        ## MAKE DF of only hbonding vdms since that's all we are looking for here.
        resname = atom_row.resname_vdm
        if resname == 'HIS':
            try:
                name = atom_row['name']
                path_to_sig = self.path_to_sig_dict[resname][name][hb_type]
                dict_corr = self.dict_corr_dict[resname][name][hb_type]
            except KeyError:
                print('HIS KeyError:' + resname + ', ' + name + ', ' + hb_type)
                return
        else:
            try:
                path_to_sig = self.path_to_sig_dict[resname]
                dict_corr = self.dict_corr_dict[resname]
            except KeyError:
                print('KeyError:' + resname)
                return
        df_lig_corr = make_df_corr(dict_corr)
        remove_from_df = self.remove_from_df_dict[resname]

        sc_set = {f[:3] for f in [ff for ff in os.listdir(path_to_sig + 'SC') if ff[-3:] == 'pkl']}
        phi_psi_set = {f[:3] for f in [ff for ff in os.listdir(path_to_sig + 'PHI_PSI') if ff[-3:] == 'pkl']}
        hydrophobes_set = {'ALA', 'PHE', 'LEU', 'VAL', 'PRO', 'ILE', 'MET'}
        cys_set = {'CYS'}
        gly_set = {'GLY'}
        charged_set = {'ASP', 'GLU', 'ARG', 'LYS'}
        seq_csts = defaultdict(dict)
        for seg, ch, rn in template_seg_chain_resnums:
            for label in ['SC', 'PHI_PSI']:
                if label == 'SC':
                    resnames = sc_set - hydrophobes_set - cys_set - gly_set
                if label == 'PHI_PSI':
                    resnames = phi_psi_set - hydrophobes_set - cys_set - gly_set
                seq_csts[(seg, ch, rn)][label] = resnames

        if (resname in self.vdm_dict.keys()) and (hb_type in self.vdm_dict[resname].keys()):
            #kwargs = dict(name='vdm', path=path_to_sig,
            #              ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
            # additional_loaded = self.vdm_dict[resname].load_additional(self.template, seq_csts, **kwargs)
            additional_loaded = self.vdm_dict[resname][hb_type].load_additional(self.template, seq_csts)
            if additional_loaded:
                # self.vdm_dict[resname][hb_type].set_neighbors(rmsd=2.0)
                self.vdm_dict[resname][hb_type].set_neighbors(rmsd=1.5)

        else:
            kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
            self.vdm_dict[resname][hb_type] = VdM(**kwargs)
            self.vdm_dict[resname][hb_type].load(self.template)
            self.vdm_dict[resname][hb_type].set_neighbors(rmsd=1.5)

        atom_type = atom_row['atom_type_label']
        if (resname in backboneNHs) and (hb_type == 'DON') and (atom_type) == 'h_pol':
            print('loading backboneNH vdMs')
            try:
                name = atom_row['name']
                path_to_sig = self.path_to_sig_dict['backboneNH']
                dict_corr = self.dict_corr_dict['backboneNH'][resname][name]
                df_lig_corr = make_df_corr(dict_corr)
                remove_from_df = self.remove_from_df_dict['backboneNH']
            except KeyError:
                print('KeyError: backboneNH and ' + resname)
                return

            if 'backboneNH' in self.vdm_dict.keys():
                kwargs = dict(name='vdm', path=path_to_sig,
                              ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                additional_loaded = self.vdm_dict['backboneNH']['parent'].load_additional(self.template, seq_csts)
                if additional_loaded:
                    self.vdm_dict['backboneNH']['parent'].set_neighbors(rmsd=0.9)
                    print('copying additional backboneNH vdms to ' + resname + '...')
                    self.vdm_dict['backboneNH'][resname][name] = VdM(**kwargs)
                    # self.vdm_dict['backboneNH'][resname][name].neighbors = copy.copy(
                    #     self.vdm_dict['backboneNH']['parent'].neighbors)
                    # self.vdm_dict['backboneNH'][resname][name].dataframe_iFG_coords = copy.copy(self.vdm_dict['backboneNH'][
                    #     'parent'].dataframe_iFG_coords)
                    # self.vdm_dict['backboneNH'][resname][name].dataframe = copy.copy(self.vdm_dict['backboneNH'][
                    #     'parent'].dataframe)
                    # self.vdm_dict['backboneNH'][resname][name].dataframe_grouped = copy.copy(self.vdm_dict['backboneNH'][
                    #     'parent'].dataframe_grouped)
                    self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                        self.vdm_dict['backboneNH']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                        self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr, on=['resname', 'name'])
                    
            else:
                kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                              ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                self.vdm_dict['backboneNH'] = defaultdict(dict)
                self.vdm_dict['backboneNH']['parent'] = VdM(**kwargs)
                self.vdm_dict['backboneNH']['parent'].load(self.template)
                self.vdm_dict['backboneNH']['parent'].set_neighbors(rmsd=0.9)

                print('copying backboneNH vdms to ' + resname + '...')
                self.vdm_dict['backboneNH'][resname][name] = VdM(**kwargs)
                # self.vdm_dict['backboneNH'][resname][name].neighbors = copy.copy(self.vdm_dict['backboneNH']['parent'].neighbors)
                # self.vdm_dict['backboneNH'][resname][name].dataframe_iFG_coords = copy.copy(self.vdm_dict['backboneNH']['parent'].dataframe_iFG_coords)
                # self.vdm_dict['backboneNH'][resname][name].dataframe = copy.copy(self.vdm_dict['backboneNH']['parent'].dataframe)
                # self.vdm_dict['backboneNH'][resname][name].dataframe_grouped = copy.copy(self.vdm_dict['backboneNH']['parent'].dataframe_grouped)
                self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr_sorted = pd.merge(self.vdm_dict['backboneNH']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                                                                                       self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr, on=['resname', 'name'])

        if (resname in backboneCOs) and (hb_type == 'ACC'):
            print('loading backboneCO vdMs')
            name = atom_row['name']
            path_to_sig = self.path_to_sig_dict['backboneCO']
            remove_from_df = self.remove_from_df_dict['backboneCO']
            if resname == 'HIS':
                try:
                    dict_corr = self.dict_corr_dict['backboneCO'][resname][name][1]
                    df_lig_corr = make_df_corr(dict_corr)
                except KeyError:
                    print('HIS KeyError:' + resname + ', ' + name + ', ' + hb_type)
                    return
            else:
                try:
                    dict_corr = self.dict_corr_dict['backboneCO'][resname][name]
                    df_lig_corr = make_df_corr(dict_corr)
                except KeyError:
                    print('KeyError: backboneCO and ' + resname)
                    return
            kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)

            if 'backboneCO' in self.vdm_dict.keys():
                additional_loaded = self.vdm_dict['backboneCO']['parent'].load_additional(self.template, seq_csts)
                if additional_loaded:
                    self.vdm_dict['backboneCO']['parent'].set_neighbors(rmsd=0.9)
                    if resname == 'HIS':
                        print('copying additional backboneCO vdms to ' + resname + '...')
                        self.vdm_dict['backboneCO'][resname][name] = dict()
                        for ii in range(1, 3):
                            dict_corr = self.dict_corr_dict['backboneCO'][resname][name][ii]
                            df_lig_corr = make_df_corr(dict_corr)
                            kwargs = dict(name='vdm', path=path_to_sig,
                                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                            self.vdm_dict['backboneCO'][resname][name][ii] = VdM(**kwargs)
                            # self.vdm_dict['backboneCO'][resname][name][ii].neighbors = copy.copy(
                            #     self.vdm_dict['backboneCO']['parent'].neighbors)
                            # self.vdm_dict['backboneCO'][resname][name][ii].dataframe_iFG_coords = copy.copy(self.vdm_dict['backboneCO'][
                            #     'parent'].dataframe_iFG_coords)
                            # self.vdm_dict['backboneCO'][resname][name][ii].dataframe = copy.copy(self.vdm_dict['backboneCO'][
                            #     'parent'].dataframe)
                            # self.vdm_dict['backboneCO'][resname][name][ii].dataframe_grouped = copy.copy(self.vdm_dict['backboneCO'][
                            #     'parent'].dataframe_grouped)
                            self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr_sorted = pd.merge(
                                self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                                self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr, on=['resname', 'name'])
                    else:
                        print('copying additional backboneCO vdms to ' + resname + '...')
                        self.vdm_dict['backboneCO'][resname][name] = VdM(**kwargs)
                        # self.vdm_dict['backboneCO'][resname][name].neighbors = copy.copy(
                        #     self.vdm_dict['backboneCO']['parent'].neighbors)
                        # self.vdm_dict['backboneCO'][resname][name].dataframe_iFG_coords = copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe_iFG_coords)
                        # self.vdm_dict['backboneCO'][resname][name].dataframe = copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe)
                        # self.vdm_dict['backboneCO'][resname][name].dataframe_grouped = copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe_grouped)
                        self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                            self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                            self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr, on=['resname', 'name'])

            else:
                self.vdm_dict['backboneCO'] = defaultdict(dict)
                self.vdm_dict['backboneCO']['parent'] = VdM(**kwargs)
                self.vdm_dict['backboneCO']['parent'].load(self.template)
                self.vdm_dict['backboneCO']['parent'].set_neighbors(rmsd=0.9)

                print('copying backboneCO vdms to ' + resname + '...')
                if resname == 'HIS':
                    self.vdm_dict['backboneCO'][resname][name] = dict()
                    for ii in range(1, 3):
                        dict_corr = self.dict_corr_dict['backboneCO'][resname][name][ii]
                        df_lig_corr = make_df_corr(dict_corr)
                        kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                                      ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                        self.vdm_dict['backboneCO'][resname][name][ii] = VdM(**kwargs)
                        # self.vdm_dict['backboneCO'][resname][name][ii].neighbors = copy.copy(
                        #     self.vdm_dict['backboneCO']['parent'].neighbors)
                        # self.vdm_dict['backboneCO'][resname][name][ii].dataframe_iFG_coords = \
                        # copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe_iFG_coords)
                        # self.vdm_dict['backboneCO'][resname][name][ii].dataframe = copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe)
                        # self.vdm_dict['backboneCO'][resname][name][ii].dataframe_grouped = copy.copy(self.vdm_dict['backboneCO'][
                        #     'parent'].dataframe_grouped)
                        self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr_sorted = pd.merge(
                            self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                            self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr, on=['resname', 'name'])
                else:
                    self.vdm_dict['backboneCO'][resname][name] = VdM(**kwargs)
                    # self.vdm_dict['backboneCO'][resname][name].neighbors = copy.copy(
                    #     self.vdm_dict['backboneCO']['parent'].neighbors)
                    # self.vdm_dict['backboneCO'][resname][name].dataframe_iFG_coords = copy.copy(self.vdm_dict['backboneCO'][
                    #     'parent'].dataframe_iFG_coords)
                    # self.vdm_dict['backboneCO'][resname][name].dataframe = copy.copy(self.vdm_dict['backboneCO']['parent'].dataframe)
                    # self.vdm_dict['backboneCO'][resname][name].dataframe_grouped = copy.copy(self.vdm_dict['backboneCO'][
                    #     'parent'].dataframe_grouped)
                    self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                        self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                        self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr, on=['resname', 'name'])


    def find_sec(self, pose, vdm_df, hbnet, atom_row, hb_type, **kwargs):
        #print(atom_row)
        # hbnet = copy.deepcopy(hbnet)
        # print('atom_row name:', atom_row['name'])
        first_shell_df = pose.vdms_sidechains[pose.vdms_sidechains.seg_chain_resnum
                                              != atom_row.seg_chain_resnum]
        resname = atom_row.resname_vdm
        name = atom_row['name']
        sec_pose = Pose()
        #print('vdm_df input for coords:')
        #print(vdm_df)
        coords = _get_frag_coords(vdm_df, self.vdm_dict[resname][hb_type])

        # print(resname, name, len(self.vdm_dict[resname][hb_type].dataframe))
        #print('coords:', coords)
        try:
            ind_neighbors = \
                self.vdm_dict[resname][hb_type].neighbors.radius_neighbors(coords, return_distance=False)[0]
        except ValueError:
            print('ValueError: duplicated vdM? skipping...')
            print(resname, hb_type)
            print(coords)
            ind_neighbors = np.array([])

        df_to_append = []
        if ind_neighbors.size > 0:
            df_uniq = self.vdm_dict[resname][hb_type].dataframe_iFG_coords.iloc[
                ind_neighbors].drop_duplicates()
            # df_to_append = pd.concat(
            #     [self.vdm_dict[resname][hb_type].dataframe_grouped.get_group(g[1:]) for g in
            #      df_uniq.itertuples()],
            #     sort=False, ignore_index=True)
            df_to_append = pd.concat(
                [self.vdm_dict[resname][hb_type].dataframe_grouped.get_group(g) for g in
                 df_uniq.values],
                sort=False, ignore_index=True)
            print('found secondary vdms...')
            
        if ('backboneNH' in self.vdm_dict.keys()) and (resname in self.vdm_dict['backboneNH']) and (name in self.vdm_dict['backboneNH'][resname]):
            coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneNH'][resname][name])
            # print('coords:', coords)
            # print('backboneNH', resname, name, len(self.vdm_dict['backboneNH'][resname][name].dataframe))
            # print('backboneNH parent', len(self.vdm_dict['backboneNH']['parent'].dataframe))
            try:
                # ind_neighbors_bb = \
                #     self.vdm_dict['backboneNH'][resname][name].neighbors.radius_neighbors(coords, return_distance=False)[0]
                ind_neighbors_bb = \
                    self.vdm_dict['backboneNH']['parent'].neighbors.radius_neighbors(coords,
                                                                                          return_distance=False)[0]
            except ValueError:
                print('ValueError: duplicated vdM? skipping...')
                print(resname, name)
                print(coords)
                ind_neighbors_bb = np.array([])

            if ind_neighbors_bb.size > 0:
                # df_uniq_bb = self.vdm_dict['backboneNH'][resname][name].dataframe_iFG_coords.iloc[
                #     ind_neighbors_bb].drop_duplicates()
                df_uniq_bb = self.vdm_dict['backboneNH']['parent'].dataframe_iFG_coords.iloc[
                    ind_neighbors_bb].drop_duplicates()
                # df_to_append_bb = pd.concat(
                #     [self.vdm_dict['backboneNH'][resname][name].dataframe_grouped.get_group(g[1:]) for g in
                #      df_uniq_bb.itertuples()],
                #     sort=False, ignore_index=True)
                # df_to_append_bb = pd.concat(
                #     [self.vdm_dict['backboneNH'][resname][name].dataframe_grouped.get_group(g) for g in
                #      df_uniq_bb.values],
                #     sort=False, ignore_index=True)
                df_to_append_bb = pd.concat(
                    [self.vdm_dict['backboneNH']['parent'].dataframe_grouped.get_group(g) for g in
                     df_uniq_bb.values],
                    sort=False, ignore_index=True)
                print('found backboneNH secondary vdms...')
                if len(df_to_append) > 0:
                    df_to_append = pd.concat((df_to_append, df_to_append_bb))
                else:
                    df_to_append = df_to_append_bb

        if ('backboneCO' in self.vdm_dict.keys()) and (resname in self.vdm_dict['backboneCO']) and (
            name in self.vdm_dict['backboneCO'][resname]):
            if resname == 'HIS':
                for jj in range(1, 3):
                    coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneCO'][resname][name][jj])
                    # print('coords:', coords)
                    # print('backboneCO', resname, name, len(self.vdm_dict['backboneCO'][resname][name][jj].dataframe))
                    # print('backboneCO parent', len(self.vdm_dict['backboneCO']['parent'].dataframe))
                    try:
                        # ind_neighbors_bb = \
                        #     self.vdm_dict['backboneCO'][resname][name][jj].neighbors.radius_neighbors(coords,
                        #                                                                           return_distance=False)[0]
                        ind_neighbors_bb = \
                            self.vdm_dict['backboneCO']['parent'].neighbors.radius_neighbors(coords,
                                                                                                      return_distance=False)[0]
                    except ValueError:
                        print('ValueError: duplicated vdM? skipping...')
                        print(resname, name)
                        print(coords)
                        ind_neighbors_bb = np.array([])

                    if ind_neighbors_bb.size > 0:
                        # df_uniq_bb = self.vdm_dict['backboneCO'][resname][name][jj].dataframe_iFG_coords.iloc[
                        #     ind_neighbors_bb].drop_duplicates()
                        df_uniq_bb = self.vdm_dict['backboneCO']['parent'].dataframe_iFG_coords.iloc[
                            ind_neighbors_bb].drop_duplicates()
                        # df_to_append_bb = pd.concat(
                        #     [self.vdm_dict['backboneCO'][resname][name][jj].dataframe_grouped.get_group(g[1:]) for g in
                        #      df_uniq_bb.itertuples()],
                        #     sort=False, ignore_index=True)
                        # df_to_append_bb = pd.concat(
                        #     [self.vdm_dict['backboneCO'][resname][name][jj].dataframe_grouped.get_group(g) for g in
                        #      df_uniq_bb.values],
                        #     sort=False, ignore_index=True)
                        df_to_append_bb = pd.concat(
                            [self.vdm_dict['backboneCO']['parent'].dataframe_grouped.get_group(g) for g in
                             df_uniq_bb.values],
                            sort=False, ignore_index=True)
                        print('found backboneCO secondary vdms...')
                        if len(df_to_append) > 0:
                            df_to_append = pd.concat((df_to_append, df_to_append_bb))
                        else:
                            df_to_append = df_to_append_bb
            else:
                # print('backboneCO', resname, name, len(self.vdm_dict['backboneCO'][resname][name].dataframe))
                # print('backboneCO parent', len(self.vdm_dict['backboneCO']['parent'].dataframe))
                coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneCO'][resname][name])
                # print('coords:', coords)
                try:
                    # ind_neighbors_bb = \
                    #     self.vdm_dict['backboneCO'][resname][name].neighbors.radius_neighbors(coords,
                    #                                                                           return_distance=False)[0]
                    ind_neighbors_bb = \
                        self.vdm_dict['backboneCO']['parent'].neighbors.radius_neighbors(coords,
                                                                                              return_distance=False)[0]
                except ValueError:
                    print('ValueError: duplicated vdM? skipping...')
                    ind_neighbors_bb = np.array([])

                if ind_neighbors_bb.size > 0:
                    # df_uniq_bb = self.vdm_dict['backboneCO'][resname][name].dataframe_iFG_coords.iloc[
                    #     ind_neighbors_bb].drop_duplicates()
                    df_uniq_bb = self.vdm_dict['backboneCO']['parent'].dataframe_iFG_coords.iloc[
                        ind_neighbors_bb].drop_duplicates()
                    # df_to_append_bb = pd.concat(
                    #     [self.vdm_dict['backboneCO'][resname][name].dataframe_grouped.get_group(g[1:]) for g in
                    #      df_uniq_bb.itertuples()],
                    #     sort=False, ignore_index=True)
                    # df_to_append_bb = pd.concat(
                    #     [self.vdm_dict['backboneCO'][resname][name].dataframe_grouped.get_group(g) for g in
                    #      df_uniq_bb.values], sort=False, ignore_index=True)
                    df_to_append_bb = pd.concat(
                        [self.vdm_dict['backboneCO']['parent'].dataframe_grouped.get_group(g) for g in
                         df_uniq_bb.values], sort=False, ignore_index=True)
                    print('found backboneCO secondary vdms...')
                    if len(df_to_append) > 0:
                        df_to_append = pd.concat((df_to_append, df_to_append_bb))
                    else:
                        df_to_append = df_to_append_bb

        if len(first_shell_df) > 0 and len(df_to_append) > 0:
            _vdms = pd.concat((first_shell_df, df_to_append)).drop_duplicates()
        elif len(first_shell_df) > 0:
            _vdms = first_shell_df
        elif len(df_to_append) > 0:
            _vdms = df_to_append
        else:
            _vdms = []

        if len(_vdms) > 0:
            sec_pose._vdms = _vdms
            vdm_df = vdm_df.copy()
            vdm_df['lig_resname'] = vdm_df.resname
            vdm_df['lig_name'] = vdm_df.name
            sec_pose.ligand = pd.concat((vdm_df, pose.ligand), sort=False)
            sec_pose.set_nonclashing_vdms()
            if len(sec_pose.vdms) > 0:
                sec_pose._vdms = None
                # print('pose.vdms=', pose.vdms)
                print('checking secondary pose for H-bond...')
                # vdm_lig_bb = pd.concat((sec_pose.ligand, self.template.dataframe), sort=False)
                # sc_lig = pd.concat((sec_pose.vdms_sidechains, pose.ligand), sort=False)
                # lig_con = Contact(sc_lig, vdm_df)
                lig_con = Contact(sec_pose.vdms_sidechains, vdm_df)
                lig_con.find()
                # print(lig_con.df_contacts)
                if len(lig_con.df_contacts) > 0:
                    #print('lig_con.df_contacts')
                    #print(lig_con.df_contacts)
                    #print('lig_con.df_contacts trunc')
                    #print(lig_con.df_contacts[['atom_type_label_q', 'name_q', 'resname_q', 'atom_type_label_t', 'name_t', 'resname_t', 'contact_type']])
                    #print('name_t:', lig_con.df_contacts.name_t)
                    # print('atom_row name:', atom_row['name'])
                    hb_contacts = lig_con.df_contacts[
                        (lig_con.df_contacts.name_t == atom_row['name'])
                        & (lig_con.df_contacts.contact_type == 'hb')]
                    #print('hb_contacts before ACC')
                    #print(hb_contacts)
                    if hb_type == 'ACC':
                        hb_contacts = hb_contacts[hb_contacts.atom_type_label_q == 'h_pol']
                    if len(hb_contacts) > 0:
                        #print('hb_contacts:')
                        #print(hb_contacts)
                        df_hb_sec = get_vdms_hbonding_to_lig(sec_pose, hb_contacts)
                        # find if hbonding atom exists in contacts now. if so, keep only vdm sidechains that are hbonding
                        if len(df_hb_sec) > 0:
                            df_hb_sec = df_hb_sec.drop('num_tag', axis=1).drop_duplicates()
                            sec_pose.vdms_sidechains = df_hb_sec
                            sec_pose.vdms = df_hb_sec
                            print('secondary H-bonders found!')
                            hbnet.secondary[name] = sec_pose
                            lig_con = Contact(sec_pose.vdms_sidechains, vdm_df)
                            lig_con.find()
                            sec_pose.lig_contacts = lig_con.df_contacts
                            kwargs['recurse'] = True
                            print('Beginning recursion...')
                            self.num_recusion += 1
                            self._find_secondary(sec_pose, **kwargs)
        # pose.hb_net.append(hbnet)


def get_vdms_hbonding_to_lig(pose, hb_contacts=None):
    groupby_q = [p + '_q' for p in pose.groupby]
    if hb_contacts is None:
        df_hb = pd.merge(pose.vdms_sidechains,
                         pose.lig_contacts[pose.lig_contacts.contact_type == 'hb'][groupby_q].drop_duplicates(),
                         left_on=pose.groupby,
                         right_on=groupby_q).drop(groupby_q, axis=1)
    else:
        df_hb = pd.merge(pose.vdms_sidechains,
                         hb_contacts[groupby_q].drop_duplicates(),
                         left_on=pose.groupby,
                         right_on=groupby_q).drop(groupby_q, axis=1)
    return df_hb


def get_hb_atoms(df, grouping=None):
    # return vdm_df[vdm_df.c_D_x.notna() | vdm_df.c_A1_x.notna()]
    atoms_acc = df[df.c_A1_x.notna()]
    atoms_don = df[df.c_D_x.notna()]
    heavy_atoms_don = df[df.atom_type_label != 'h_pol']
    if grouping is None:
        # grouping = ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum', 'name']
        grouping = ['str_index', 'name']
    if len(atoms_don) > 0 and len(heavy_atoms_don) > 0:
        don_merged = pd.merge(atoms_don, heavy_atoms_don[grouping].drop_duplicates(), on=grouping, how='outer',
                              indicator=True)
        atoms_don = don_merged[don_merged._merge == 'left_only'].drop('_merge', axis=1)
    return atoms_acc, atoms_don


def count_num_donors_acceptors(df):
    num_acc = len(df[df.c_A1_x.notna()])
    num_don = len(df[df.c_D_x.notna()])
    num_don -= len(df[df.c_H2_x.notna()])  # removes heavy atoms from donor count, such that donors only counts Hs.
    return num_acc + num_don


def get_bun_hb_atoms(df, template, burial_depth=1, contacts=None, pose=None, grouping=None, return_satisfied=False, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    bun_atoms_acc = pd.DataFrame()
    bun_atoms_don = pd.DataFrame()
    # df = df.copy()
    df['distance_to_hull'] = get_distance_to_hull(df, template)

    if kwargs:
        omit = kwargs.get('omit')
        # groupby = kwargs.get('groupby', ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum'])
        groupby = kwargs.get('groupby', 'str_index')
        pl = pose.ligand.copy()
        pl2 = pose.ligand.copy()
        pl.set_index(groupby, inplace=True, drop=False)
        num_accs = []
        num_dons = []
        for n, g in pl2.groupby(groupby):
            # if n == omit:
            #     continue
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((pl[~pl.index.isin(g.index)], df, template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            # print(contacts_[['name_q','lig_name_t', 'contact_type']])
            acc, don = get_bun_hb_atoms(g, template, contacts=contacts_)
            num_accs.append(len(acc))
            num_dons.append(len(don))
        acc, don = get_bun_hb_atoms(df, template, burial_depth, contacts, pose, grouping)
        num_accs.append(len(acc))
        num_dons.append(len(don))
        return sum(num_accs), sum(num_dons)
    
    atoms_acc = df[df.c_A1_x.notna()]
    atoms_don = df[df.c_D_x.notna()]
    heavy_atoms_don = df[df.atom_type_label != 'h_pol']
    if grouping is None:
        # grouping = ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum', 'name']
        grouping = ['str_index', 'name']
    if len(atoms_don) > 0 and len(heavy_atoms_don) > 0:
        don_merged = pd.merge(atoms_don, heavy_atoms_don[grouping].drop_duplicates(), on=grouping, how='outer',
                              indicator=True)
        atoms_don = don_merged[don_merged._merge == 'left_only'].drop('_merge', axis=1)
        
    if contacts is None and pose is not None:
        lig_bb = pd.concat((pose.ligand, template.dataframe), sort=False)
        df_con = Contact(lig_bb, df)
        df_con.find()
        contacts = df_con.df_contacts
    elif contacts is None and pose is None:
        # raise '*contacts* are not supplied, so *pose* must not be None.'
        if len(atoms_acc) > 0:
            # atoms_acc = atoms_acc.copy()
            # atoms_acc['distance_to_hull'] = get_distance_to_hull(atoms_acc, template)
            bun_atoms_acc = atoms_acc[atoms_acc.distance_to_hull >= burial_depth]
        if len(atoms_don) > 0:
            # atoms_don = atoms_don.copy()
            # atoms_don['distance_to_hull'] = get_distance_to_hull(atoms_don, template)
            bun_atoms_don = atoms_don[atoms_don.distance_to_hull >= burial_depth]
        return bun_atoms_acc, bun_atoms_don
    hb_contacts = contacts[contacts.contact_type == 'hb']

    if return_satisfied:
        sat_acc = pd.DataFrame()
        sat_don = pd.DataFrame()
        if len(atoms_acc) > 0:
            atoms_acc, sat_acc = remove_satisfied_accs(atoms_acc, hb_contacts, grouping, return_satisfied)
            atoms_acc = atoms_acc.copy()
            sat_acc = sat_acc.copy()

        if len(atoms_don) > 0:
            atoms_don, sat_don = remove_satisfied_dons(atoms_don, hb_contacts, grouping, return_satisfied)
            atoms_don = atoms_don.copy()
            sat_don = sat_don.copy()
    else:
        if len(atoms_acc) > 0:
            atoms_acc = remove_satisfied_accs(atoms_acc, hb_contacts, grouping).copy()

        if len(atoms_don) > 0:
            atoms_don = remove_satisfied_dons(atoms_don, hb_contacts, grouping).copy()

    if len(atoms_acc) > 0:
        # atoms_acc['distance_to_hull'] = get_distance_to_hull(atoms_acc, template)
        # print('acc')
        # print(atoms_acc[['name', 'resname_vdm', 'distance_to_hull']])
        bun_atoms_acc = atoms_acc[atoms_acc.distance_to_hull >= burial_depth]
        
    if len(atoms_don) > 0:
        # atoms_don['distance_to_hull'] = get_distance_to_hull(atoms_don, template)
        # print('don')
        # print(atoms_don[['name', 'resname_vdm', 'distance_to_hull']])
        bun_atoms_don = atoms_don[atoms_don.distance_to_hull >= burial_depth]

    if return_satisfied:
        return bun_atoms_acc, bun_atoms_don, sat_acc, sat_don
    else:
        return bun_atoms_acc, bun_atoms_don


def get_num_bun_hb_atoms(df, template, pose, burial_depth=1, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    df = pd.concat((df, pose.ligand), sort=False)

    if kwargs:
        omit = kwargs.get('omit')
        # groupby = kwargs.get('groupby', ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum'])
        groupby = kwargs.get('groupby', 'str_index')
        df2 = df.copy()
        df.set_index(groupby, inplace=True, drop=False)
        num_accs = []
        num_dons = []
        for n, g in df2.groupby(groupby):
            # if n == omit:
            #     continue
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((df[~df.index.isin(g.index)], template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            # print(contacts_[['name_q','lig_name_t', 'contact_type']])
            acc, don = get_bun_hb_atoms(g, template, burial_depth, contacts=contacts_)
            num_accs.append(len(acc))
            num_dons.append(len(don))
        return sum(num_accs), sum(num_dons)


def _get_bun_hb_atoms(df, template, pose, burial_depth=1, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    df = pd.concat((df, pose.ligand), sort=False)
    if 'num_tag' in df.columns:
        df = df.drop('num_tag', axis=1)

    if kwargs:
        omit = kwargs.get('omit')
        # groupby = kwargs.get('groupby', ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum'])
        groupby = kwargs.get('groupby', 'str_index')
        df2 = df.copy()
        df.set_index(groupby, inplace=True, drop=False)
        accs = []
        dons = []
        for n, g in df2.groupby(groupby):
            # if n == omit:
            #     continue
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((df[~df.index.isin(g.index)], template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            # print(contacts_[['name_q','lig_name_t', 'contact_type']])
            acc, don = get_bun_hb_atoms(g, template, burial_depth, contacts=contacts_)
            if len(acc) > 0:
                accs.append(acc)
            if len(don) > 0:
                dons.append(don)
        if accs:
            df_accs = pd.concat(accs, sort=False).drop_duplicates()
        else:
            df_accs = pd.DataFrame(columns=df.columns)
        if dons:
            df_dons = pd.concat(dons, sort=False).drop_duplicates()
        else:
            df_dons = pd.DataFrame(columns=df.columns)

        return df_accs, df_dons


def remove_satisfied_accs(atoms_acc, hb_contacts, grouping, return_satisfied=False):
    hb_contacts = hb_contacts[hb_contacts.atom_type_label_q == 'h_pol']
    if return_satisfied:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_acc = pd.merge(atoms_acc, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_acc = merged_acc[merged_acc._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            sat_atoms_acc = merged_acc[merged_acc._merge == 'both'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_acc, sat_atoms_acc
        else:
            return atoms_acc, pd.DataFrame(columns=atoms_acc.columns)
    else:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_acc = pd.merge(atoms_acc, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_acc = merged_acc[merged_acc._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_acc
        else:
            return atoms_acc
    
    
def remove_satisfied_dons(atoms_don, hb_contacts, grouping, return_satisfied=False):
    if return_satisfied:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_don = pd.merge(atoms_don, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_don = merged_don[merged_don._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            sat_atoms_don = merged_don[merged_don._merge == 'both'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_don, sat_atoms_don
        else:
            return atoms_don, pd.DataFrame(columns=atoms_don.columns)
    else:
        t_keys = [p + '_t' for p in grouping]
        merged_don = pd.merge(atoms_don, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                              right_on=t_keys, indicator=True)
        atoms_don = merged_don[merged_don._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
        return atoms_don


def get_distance_to_hull(df, template):
    return template.alpha_hull.get_pnts_distance(df[['c_x', 'c_y', 'c_z']].values)


def _get_frag_coords(df, vdm):
    df_corr = pd.merge(vdm.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates(), df.drop_duplicates(),
                       how='inner', left_on=['lig_resname', 'lig_name'], right_on=['resname_vdm', 'name'], sort=False)
    return df_corr[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)


class PoseLegs:
    def __init__(self):
        self.poselegs = []
        self.num_buns = []
        self.poselegs_uniq = []
        self.num_buns_uniq = []

    def get_poselegs(self, hbnet):
        self.poselegs.append(hbnet.primary)
        self.num_buns.append(hbnet.num_buns)

        if hbnet.secondary:
            for pose in hbnet.secondary.values():
                for hbnet_ in pose.hb_net:
                    hbnet_ = copy.deepcopy(hbnet_)
                    concated = pd.concat((hbnet.primary, hbnet_.primary), sort=False)
                    hbnet_.primary = concated
                    self.get_poselegs(hbnet_)

    def drop_duplicates(self):
        dropped = [pl.drop(['num_tag', 'rmsd_from_centroid'], axis=1).sort_values(
            ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum', 'name']).reset_index(drop=True) for pl in
                   self.poselegs]
        for i, (nb, pl) in enumerate(zip(self.num_buns, dropped)):
            drop = False
            for pl_ in dropped[i + 1:]:
                if pl.equals(pl_):
                    drop = True
            if not drop:
                self.poselegs_uniq.append(pl)
                self.num_buns_uniq.append(nb)


# @jit("f8[:](f8[:], f8[:,:])", nopython=True)
# def _dee(es, ep):
#     to_del = np.zeros(es.size)
#     pair_ens = np.zeros(es.size)
#     w = 0
#     for i in range(es.size):
#         for j in range(es.size):
#             if j == i:
#                 continue
#             for k in range(es.size):
#                 pair_ens[k] = ep[i, j] - ep[k, j]
#             cond = es[i] - es[j] + pair_ens.min()
#             if cond > 0.0:
#                 to_del[w] = i
#                 w += 1
#                 break
#     return to_del[:w]
@jit("f8[:](f8[:], f8[:,:])", nopython=True)
def _dee(es, ep):
    to_del = np.zeros(es.size)
    pair_ens = np.zeros(ep.shape[0])
    w = 0
    for i in range(es.size):
        for j in range(es.size):
            if j == i:
                continue
            for k in range(ep.shape[0]):
                pair_ens[k] = ep[i, j] - ep[k, j]
            cond = es[i] - es[j] + pair_ens.min()
            # es[i] + sum over k(ep[i, k]) > max over j( es[j] + sum over k(ep[j, k]) )
            if cond > 0.0:
                to_del[w] = i
                w += 1
                break
    return to_del[:w]


def remove_dups(ar1, ar2, return_inds=False):
    sh1 = ar1.shape[0]
    stacked = np.vstack((ar1, ar2))
    un, inds = np.unique(stacked, axis=0, return_index=True)
    inds_ = set(inds)
    inds1 = set(range(sh1)) & inds_
    inds__ = list(np.array(list(inds_ - inds1)) - sh1)
    if return_inds:
        inds1 = list(inds1)
        return ar1[inds1], ar2[inds__], inds1, inds__
    else:
        return ar1[np.array(list(inds1))], ar2[inds__]


def is_subset(ar1, ar2):
    if ar1.shape[0] < ar2.shape[0]:
        ar1_ = ar1
        ar1 = ar2
        ar2 = ar1_
    sh1 = ar1.shape[0]
    stacked = np.vstack((ar1, ar2))
    un, inds = np.unique(stacked, axis=0, return_index=True)
    inds_ = set(inds)
    inds1 = set(range(sh1)) & inds_
    inds__ = list(np.array(list(inds_ - inds1)) - sh1)
    if inds__:
        return False
    else:
        return True
