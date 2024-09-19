from sys import path_importer_cache
from .constants import inv_one_letter_code, load_columns, hb_cols, coords_cols, atom_type_dict
from collections import defaultdict
import pandas as pd
import warnings
from pandas import read_parquet, merge, concat, DataFrame, notnull
import numpy as np
from .functions import df_ideal_ala, listdir_mac, read_lig_txt, get_heavy, print_dataframe, \
    fast_concat, make_lig_atom_type_dict, make_lig_hbond_dict, make_df_from_prody, \
    chunks, get_HA2, get_HA3, df_is_subset, make_empty_df, cg_dfs, flip_cg_coords, \
    cg_dicts, rotamer_dfs, flip_coords_from_reference_df, get_angle_diff
from .cluster import get_max, Cluster
from .transformation import superpose_df, get_rot_trans, apply_transform_to_coords_cols, \
    apply_transform
from .contacts import Clash, ClashVDM, Contact
from .constants import cgs_that_flip, flip_dict
import os
from sklearn.neighbors import NearestNeighbors, BallTree
import itertools
from prody import AtomGroup, writePDB, parsePDB
from multiprocessing import Pool
from functools import partial
from pickle import dump
import random
import pickle
import time
from scipy.spatial.distance import cdist


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


phi_psi_dict = dict(phi=dict(phi=20, psi=100),
                    psi=dict(phi=100, psi=20),
                    phi_psi=dict(phi=20, psi=20),
                    sc=dict(phi=30, psi=30))


class Group:
    """This class replaces pandas groupby get_group fn with
    much faster group retrieval."""
    def __init__(self, dict):
        self.dict = dict

    def get_group(self, key):
        return self.dict[key]


class Residue:
    def __init__(self):
        self.CG = ''
        self.resname = ''
        self.seg_chain_resnum = None
        self.hbond_only = False
        self.bbdep = 0
        self.enriched = False
        self.top = None
        self.CA_burial = None
        self.phi_psi_dict = phi_psi_dict
        self.rotamer = False
        self.dssp = False
        self.cg_is_hb_donor = False
        self.cg_is_hb_acceptor = False
        self.cg_is_not_hb_donor = False
        self.cg_is_not_hb_acceptor = False

    def set_attr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Sample:
    def __init__(self, **kwargs):
        # self.df_template = None
        self.path_to_resfile = kwargs.get('path_to_resfile', './resfile.txt')
        self.path_to_database = kwargs.get('path_to_database')
        self.path_to_ligand_database = kwargs.get('path_to_ligand_database')
        self.residue_dict = defaultdict(dict)
        self.cg_dict = defaultdict(list)
        self.cg_dict_grs = dict()
        self.cgs = set()
        self.ligand_dataframe = []
        self.ligand_dataframe_grs = None
        self.cg_neighbors = dict()
        self.cg_dataframe_neighbors = dict()
        self.is_acceptor = dict()
        self.is_donor = dict()
        self.cg_seg_chain_resnum_scores = dict()
        self.ligand_vdm_correspondence = None
        self.ligand_vdm_correspondence_grs = None
        self.cg_alts = defaultdict(set)
        self.cg_atom_map = defaultdict(set)
        self.ligand_names = None
        self.atom_cg_map = defaultdict(set)
        self.atom_cg_map_by_atomtype = defaultdict(set)
        self.cg_ligand_coverage = defaultdict(dict)
        self.cg_rmsds = dict()
        self.cg_num_atoms = dict()
        self._max_dist_criterion = False
        self.ligand_neighbors_indices = defaultdict(dict)
        self.ligand_neighbors_dists = defaultdict(dict)
        self.ligand_dataframe_neighbors = None
        self._ligand_neighbors_indices = None
        self._ligand_neighbors_dists = None
        self._ligand_dataframe_neighbors = None
        self.constraints = None
        self.constraints_gr = None
        self.poses = []
        self._poses = []
        self.pose_clusters_members = []
        self.pose_clusters_centroids = []
        self.vdm_clusters_members = dict()
        self.vdm_clusters_centroids = dict()
        self._pairwise_dict = defaultdict(dict)
        self.ligand_conformers = None
        self.pose_groups = []
        self.pose_groups_dict = dict()
        self.filtered_lig_indices = None
        self.leftover_filtered_lig_indices = None
        self._cg_com_tree = dict()
        self.cg_weights = dict()
        self.ligand_rep = None
        self.ligand_atom_types = dict()
        self.vdms_buried_unsat = defaultdict(set)
        self.vdm_pose_map = defaultdict(list)
        self.poss_vdms_for_buried_unsatisfied = None
        self._sample_buried_unsat = None
        self.nbr_vdm_map = defaultdict(dict)
        self.pose_nbr_map = dict() # defaultdict(lambda: defaultdict(set))

    def __copy__(self, no_vdms=True, keep_vdm_info=False, keep_nbrs=False, minimal_info=False,
                 include_poses_in_minimal_info=True):
        s = Sample()
        for k, v in vars(self).items():
            if minimal_info:
                cols = ['constraints', 'constraints_gr', 'ligand_conformers',
                        '_max_dist_criterion', 'cg_num_atoms', 'cg_rmsds', 'cg_ligand_coverage',
                        'ligand_vdm_correspondence', 'ligand_vdm_correspondence_grs', 'residue_dict', 'path_to_ligand_database',
                        'path_to_database', 'path_to_resfile', 'cg_alts', 'cg_atom_map', 'atom_cg_map', 
                        'ligand_names', 'ligand_rep', 'atom_cg_map_by_atomtype', 'ligand_atom_types']
                if include_poses_in_minimal_info:
                    cols.append('poses')
                if k not in cols:
                    continue
            else:
                if no_vdms and not keep_nbrs and k in ['cg_dict_grs', 'cg_neighbors', 'cg_dataframe_neighbors']:
                    continue
                if no_vdms and keep_nbrs and not keep_vdm_info and k in ['cg_dict_grs', 'cg_dataframe_neighbors', 'cg_dict']:
                    continue
                if no_vdms and k == 'cg_dict':
                    cols_dfx = ['resname_rota', 'CG', 'rota', 'probe_name', 'seg_chain_resnum']
                    for k2, v2 in self.cg_dict.items():
                        s.cg_dict[k2] = v2[cols_dfx].drop_duplicates()
                    continue
            s.__dict__[k] = v
        return s

    def set_loaded_ligand_data(self, s):
        self.ligand_dataframe = s.ligand_dataframe
        self.ligand_dataframe_grs = s.ligand_dataframe_grs
        s.ligand_dataframe = None
        s.ligand_dataframe_grs = None

    def read_resfile(self):
        residues = []
        with open(self.path_to_resfile, 'r') as infile:
            for line in infile:
                if line[0] == '#':
                    continue
                if len(line.strip()) == 0:
                    continue
                split_line = line.split(',')
                _seg_chain_resnum = split_line[0].split()
                _seg_chain_resnum[0] = int(_seg_chain_resnum[0])
                if _seg_chain_resnum[-1] == '_':
                    _seg_chain_resnum[-1] = ''
                _seg_chain_resnum.reverse()
                seg_chain_resnum = tuple(_seg_chain_resnum)
                res_dict = dict()
                for item in split_line[1:]:
                    item_spl = item.split()
                    if item_spl[0] == 'CG':
                        cgs = item_spl[1:]
                    elif item_spl[0] == 'phi_psi_tol':
                        res_dict[item_spl[0]] = item_spl[1:]
                    elif len(item_spl) == 2:
                        if item_spl[1] in ['True', 'False']:
                            res_dict[item_spl[0]] = bool(item_spl[1])
                        else:
                            try:
                                res_dict[item_spl[0]] = int(item_spl[1])
                            except:
                                res_dict[item_spl[0]] = item_spl[1]
                    else:
                        res_dict[item_spl[0]] = True
                for cg in cgs:  # make sure CG is in every line of resfile.
                    for aa in set(res_dict['PIKAA']):
                        res = Residue()
                        res.seg_chain_resnum = seg_chain_resnum
                        res.CG = cg
                        res.resname = inv_one_letter_code[aa]
                        if 'phi_psi_tol' in res_dict:
                            len_pst = len(res_dict['phi_psi_tol'])
                            if len_pst % 3 != 0:
                                raise Exception('phi_psi_tol formatted incorrectly in resfile.')
                            for i in range(0, len_pst, 3):
                                key1, key2, val = res_dict['phi_psi_tol'][i:i + 3]
                                res.phi_psi_dict[key1][key2] = float(val)
                            res_dict.pop('phi_psi_tol')
                        res.set_attr(**res_dict)
                        residues.append(res)
        for res in residues:
            if res.resname not in self.residue_dict[res.CG]:
                self.residue_dict[res.CG][res.resname] = [res]
            else:
                self.residue_dict[res.CG][res.resname].append(res)

    @staticmethod
    def _load_res(res, df_parent, template, filter_by_phi_psi, filter_by_phi_psi_exclude_sc, aa,
                    gr_names, ignore_CG_for_clash_check, vdW_tolerance=0.1):
        print('        Loading residue', res.seg_chain_resnum)

        if aa == 'PRO':
            df_targ_res = template.dataframe[template.dataframe['seg_chain_resnum'] == res.seg_chain_resnum]
            phi = df_targ_res['phi'].iat[0]
            psi = df_targ_res['psi'].iat[0]
            if not ((phi < -35) & (phi > -105) & (psi > -70)):
                print('            Residue Phi/Psi does not pass PRO Phi/Psi filter')
                return DataFrame()
            m1 = list(res.seg_chain_resnum)
            m1[-1] = m1[-1] - 1
            m1 = tuple(m1)
            if m1 in set(template.dataframe['seg_chain_resnum']):
                df_targ_res_m1 = template.dataframe[template.dataframe['seg_chain_resnum'] == tuple(m1)]
                psi_m1 = df_targ_res_m1['psi'].iat[0]
                if psi_m1 < -90:
                    print('            Preceding residue Psi does not pass preceding-PRO Psi filter')
                    return DataFrame()

        if gr_names is not None:
            grs = df_parent.groupby(['CG', 'rota', 'probe_name'])
            df_indices = list(itertools.chain(*[grs.groups[gr_name] for gr_name in grs.groups.keys()
                                                        if gr_name in gr_names]))
            df = df_parent[df_parent.index.isin(df_indices)].copy()
        else:
            df = df_parent.copy()

        if res.cg_is_not_hb_acceptor:
            print('            setting cg as not an hb acceptor')
            df_y = df[df['chain'] == 'Y']
            inds = np.arange(len(df_y))
            tf_hb = ~df_y['partners_hb'].isna()
            inds_hb = inds[tf_hb]
            if tf_hb.any():
                tf_acc = df_y['partners_hb'][tf_hb].str.startswith('H')
                if tf_acc.any():
                    inds_acc = inds_hb[tf_acc]
                    acc = df_y.iloc[inds_acc].groupby(['CG', 'rota', 'probe_name'])
                    grs = df.groupby(['CG', 'rota', 'probe_name'])

                    df_indices = list(itertools.chain(*[grs.groups[gr_name] for gr_name in grs.groups.keys()
                                                        if gr_name not in acc.groups.keys()]))

                    df = df[df.index.isin(df_indices)]

                    if len(df) == 0:
                        print('            no vdms due to no cg that is not an hb acceptor')
                        return DataFrame()

        if res.cg_is_not_hb_donor:
            print('            setting cg as not an hb donor')
            df_y = df[df['chain'] == 'Y']
            inds = np.arange(len(df_y))
            tf_hb = ~df_y['partners_hb'].isna()
            inds_hb = inds[tf_hb]
            if tf_hb.any():
                tf_don = ~df_y['partners_hb'][tf_hb].str.startswith('H')
                if tf_don.any():
                    inds_don = inds_hb[tf_don]
                    don = df_y.iloc[inds_don].groupby(['CG', 'rota', 'probe_name'])
                    grs = df.groupby(['CG', 'rota', 'probe_name'])

                    df_indices = list(itertools.chain(*[grs.groups[gr_name] for gr_name in grs.groups.keys()
                                                        if gr_name not in don.groups.keys()]))

                    df = df[df.index.isin(df_indices)]

                    if len(df) == 0:
                        print('            no vdms due to no cg that is not an hb donor')
                        return DataFrame()

        hb_str = ''
        if res.hbond_only:
            df_y = df[df['chain'] == 'Y']
            global_filters_y = [
                df_y['contact_hb'] == True,
            ]

            global_filter_y = np.all(global_filters_y, axis=0)
            df_y = df_y[global_filter_y]

            if len(df_y) == 0:
                print('            no vdms due to no h-bonding')
                return DataFrame()

            df = merge(df, df_y[['CG', 'rota', 'probe_name']].drop_duplicates(),
                       on=['CG', 'rota', 'probe_name'])

            if res.cg_is_hb_acceptor or res.cg_is_hb_donor:
                df_y = df[df['chain'] == 'Y']  # all vdMs here are hbonding
                global_filters_y = []

                if res.cg_is_hb_acceptor:
                    tf = df_y['partners_hb'].str.startswith('H') # CG has a partner that is donating a H
                    # global_filters_y.append(~df_y['c_A1_x'].isna())
                    # global_filters_y.append((~df_y['partners_hb'].isna()) & (df_y['partners_hb'].str.startswith('H')))
                    global_filters_y.append(tf)

                elif res.cg_is_hb_donor:
                    # global_filters_y.append(~df_y['c_D_x'].isna())
                    # global_filters_y.append((~df_y['partners_hb'].isna()) & (~df_y['partners_hb'].str.startswith('H')))
                    tf = ~df_y['partners_hb'].str.startswith('H') # CG has a partner that is not donating a H
                    global_filters_y.append(tf)

                global_filter_y = np.all(global_filters_y, axis=0)
                df_y = df_y[global_filter_y]

                if len(df_y) == 0:
                    print('            no vdms due to no h-bonding')
                    return DataFrame()

                df = merge(df, df_y[['CG', 'rota', 'probe_name']].drop_duplicates(),
                        on=['CG', 'rota', 'probe_name'])
                # df = fast_merge(df, df_y[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)

            # print('            Num hbonding vdMs', len(df[['CG', 'rota', 'probe_name']].drop_duplicates()))
            hb_str = 'hb_'

        df_targ_res = template.dataframe[template.dataframe['seg_chain_resnum'] == res.seg_chain_resnum]
        abple = df_targ_res['ABPLE'].iat[0]
        ABPLE_3mer = df_targ_res['ABPLE_3mer'].iat[0]
        if 'dist_to_hull' in df_targ_res.columns:
            targ_dist_to_hull = df_targ_res[df_targ_res['name'] == 'CA']['dist_to_hull'].iat[0]
        if 'dssp' in df_targ_res.columns:
            dssp = df_targ_res['dssp'].iat[0]
            dssp_3mer = df_targ_res['dssp_3mer'].iat[0]

        if res.bbdep != 0:
            if res.dssp:
                local_filters_xy = [df['sc_rep_fine_dssp']]
                if res.top is not None:
                    local_filters_xy.append(df['cluster_rank_' + hb_str + 'dssp_' + dssp] <= res.top)
                if res.enriched:
                    local_filters_xy.append(df['C_score_' + hb_str + 'dssp_' + dssp] > 0)
            else:
                local_filters_xy = [df['sc_rep_fine_ABPLE']]
                if res.top is not None:
                    local_filters_xy.append(df['cluster_rank_' + hb_str + 'ABPLE_' + abple] <= res.top)
                if res.enriched:
                    local_filters_xy.append(df['C_score_' + hb_str + 'ABPLE_' + abple] > 0)
            local_filter_xy = np.all(local_filters_xy, axis=0)
            df = df[local_filter_xy]
            df_x_ca = df[(df['chain'] == 'X') & (df['name'] == 'CA')]
            if res.bbdep == 1:
                if res.dssp:
                    if 'dssp' not in df_targ_res.columns:
                        raise Exception('Template PDB needs dssp set.')
                    ABPLE_col = 'dssp'
                    ABPLE_val = dssp
                else:
                    ABPLE_col = 'ABPLE'
                    ABPLE_val = abple
            elif res.bbdep == 3:
                if res.dssp:
                    if 'dssp' not in df_targ_res.columns:
                        raise Exception('Template PDB needs dssp set.')
                    ABPLE_col = 'dssp_3mer'
                    ABPLE_val = dssp_3mer
                else:
                    ABPLE_col = 'ABPLE_3mer'
                    ABPLE_val = ABPLE_3mer
            filters_x_ca = [df_x_ca[ABPLE_col] == ABPLE_val]
            if res.CA_burial is not None:
                filters_x_ca.append(df_x_ca['dist_to_hull'] > (targ_dist_to_hull + res.CA_burial))

            if res.rotamer:
                if res.resname not in ['GLY', 'ALA']:
                    rot = df_targ_res['rotamer'].iat[0]
                    filters_x_ca.append(df_x_ca['rotamer'] == rot)

            filter_x_ca = np.all(filters_x_ca, axis=0)
            df_x_ca = df_x_ca[filter_x_ca]
            if len(df_x_ca) == 0:
                print('            no vdms due to bb_dep, burial, or rotamer')
                return DataFrame()
            
            if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
                contact_types = df_x_ca['contact_type'].values
                phis = df_x_ca['phi'].values
                template_phi = df_targ_res['phi'].iat[0]
                phi_diffs = get_angle_diff(phis, template_phi)
                psis = df_x_ca['psi'].values
                template_psi = df_targ_res['psi'].iat[0]
                psi_diffs = get_angle_diff(psis, template_psi)
                indices = np.arange(len(df_x_ca))
                passed_phi_psi_filter = np.zeros(len(df_x_ca), dtype=bool)
                for contact_type in phi_psi_dict.keys():
                    if filter_by_phi_psi_exclude_sc:
                        if contact_type == 'sc' and aa != 'GLY':
                            mask = contact_types == contact_type
                            passed_phi_psi_filter[mask] = True
                            continue
                    mask = contact_types == contact_type
                    if not mask.any():
                        continue
                    phi_tol = res.phi_psi_dict[contact_type]['phi']
                    psi_tol = res.phi_psi_dict[contact_type]['psi']
                    phi_diffs_masked = phi_diffs[mask]
                    phi_mask = phi_diffs_masked <= 2 * phi_tol
                    psi_diffs_masked = psi_diffs[mask]
                    psi_mask = psi_diffs_masked <= 2 * psi_tol
                    phi_psi_mask = phi_mask & psi_mask
                    passed_phi_psi_filter[indices[mask][phi_psi_mask]] = True
                df_x_ca = df_x_ca[passed_phi_psi_filter]
                if len(df_x_ca) == 0:
                    print('            no vdms due to Phi/Psi filter')
                    return DataFrame()

            # if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
            #     gs = []
            #     for n, g in df_x_ca.groupby('contact_type'):
            #         if filter_by_phi_psi_exclude_sc:
            #             if n == 'sc' and res.resname != 'GLY':
            #                 gs.append(g)
            #                 continue
            #         phi_tol = res.phi_psi_dict[n]['phi']
            #         psi_tol = res.phi_psi_dict[n]['psi']

            #         phi_psi_filters = []
            #         phi_high = df_targ_res['phi'].iat[0] + phi_tol
            #         if phi_high > 180:
            #             f1 = (g['phi'] <= 180).values
            #             f2 = (g['phi'] <= phi_high - 180).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['phi'] <= phi_high).values)

            #         psi_high = df_targ_res['psi'].iat[0] + psi_tol
            #         if psi_high > 180:
            #             f1 = (g['psi'] <= 180).values
            #             f2 = (g['psi'] <= psi_high - 360).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['psi'] <= psi_high).values)

            #         phi_low = df_targ_res['phi'].iat[0] - phi_tol
            #         if phi_low < -180:
            #             f1 = (g['phi'] >= -180).values
            #             f2 = (g['phi'] >= 360 - phi_low).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['phi'] >= phi_low).values)

            #         psi_low = df_targ_res['psi'].iat[0] - psi_tol
            #         if psi_low < -180:
            #             f1 = (g['psi'] >= -180).values
            #             f2 = (g['psi'] >= 360 - psi_low).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['psi'] >= psi_low).values)

            #         phi_psi_filter = np.all(phi_psi_filters, axis=0)
            #         if not phi_psi_filter.any():
            #             return DataFrame()
            #         gs.append(g[phi_psi_filter])
            #     if len(gs) > 0:
            #         df_x_ca = fast_concat(gs)
            #     if len(df_x_ca) == 0:
            #         print('            no vdms due to Phi/Psi filter')
            #         return DataFrame()
            df = merge(df, df_x_ca[['CG', 'rota', 'probe_name']].drop_duplicates(),
                       on=['CG', 'rota', 'probe_name'])
            # df = fast_merge(df, df_x_ca[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)
            df = df[~((df['chain'] == 'X') & (df['name'] == 'CA'))]

        if res.bbdep == 0:
            local_filters_xy = [df['sc_rep_fine_bb_ind']]
            if res.top is not None:
                local_filters_xy.append(df['cluster_number'] <= res.top)
            if res.enriched:
                local_filters_xy.append(df['C_score_' + hb_str + 'bb_ind'] > 0)
            local_filter_xy = np.all(local_filters_xy, axis=0)
            df = df[local_filter_xy]

            if res.rotamer:
                if res.resname not in ['GLY', 'ALA']:
                    df_x_ca = df[(df['chain'] == 'X') & (df['name'] == 'CA')]
                    rot = df_targ_res['rotamer'].iat[0]
                    df_x_ca = df_x_ca[df_x_ca['rotamer'] == rot]
                    df = merge(df, df_x_ca[['CG', 'rota', 'probe_name']].drop_duplicates(),
                            on=['CG', 'rota', 'probe_name'])

                    if len(df) == 0:
                        print('            no vdms due to rotamer')
                        return DataFrame()

            if res.CA_burial is not None:
                df_x_ca = df[(df['chain'] == 'X') & (df['name'] == 'CA')]
                df_x_ca = df_x_ca[df_x_ca['dist_to_hull'] > (targ_dist_to_hull + res.CA_burial)]
                df = merge(df, df_x_ca[['CG', 'rota', 'probe_name']].drop_duplicates(),
                           on=['CG', 'rota', 'probe_name'])

                if len(df) == 0:
                    print('            no vdms due to burial')
                    return DataFrame()

            if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
                df_x_ca = df[(df['chain'] == 'X') & (df['name'] == 'CA')]
                contact_types = df_x_ca['contact_type'].values
                phis = df_x_ca['phi'].values
                template_phi = df_targ_res['phi'].iat[0]
                phi_diffs = get_angle_diff(phis, template_phi)
                psis = df_x_ca['psi'].values
                template_psi = df_targ_res['psi'].iat[0]
                psi_diffs = get_angle_diff(psis, template_psi)
                indices = np.arange(len(df_x_ca))
                passed_phi_psi_filter = np.zeros(len(df_x_ca), dtype=bool)
                for contact_type in phi_psi_dict.keys():
                    if filter_by_phi_psi_exclude_sc:
                        if contact_type == 'sc' and aa != 'GLY':
                            mask = contact_types == contact_type
                            passed_phi_psi_filter[mask] = True
                            continue
                    mask = contact_types == contact_type
                    if not mask.any():
                        continue
                    phi_tol = res.phi_psi_dict[contact_type]['phi']
                    psi_tol = res.phi_psi_dict[contact_type]['psi']
                    phi_diffs_masked = phi_diffs[mask]
                    phi_mask = phi_diffs_masked <= 2 * phi_tol
                    psi_diffs_masked = psi_diffs[mask]
                    psi_mask = psi_diffs_masked <= 2 * psi_tol
                    phi_psi_mask = phi_mask & psi_mask
                    passed_phi_psi_filter[indices[mask][phi_psi_mask]] = True
                df_x_ca = df_x_ca[passed_phi_psi_filter]
                if len(df_x_ca) == 0:
                    print('            no vdms due to Phi/Psi filter')
                    return DataFrame()

            # if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
            #     df_x_ca = df[(df['chain'] == 'X') & (df['name'] == 'CA')]
            #     gs = []
            #     for n, g in df_x_ca.groupby('contact_type'):
            #         if filter_by_phi_psi_exclude_sc:
            #             if n == 'sc' and res.resname != 'GLY':
            #                 gs.append(g)
            #                 continue
            #         phi_tol = res.phi_psi_dict[n]['phi']
            #         psi_tol = res.phi_psi_dict[n]['psi']

            #         phi_psi_filters = []
            #         phi_high = df_targ_res['phi'].iat[0] + phi_tol
            #         if phi_high > 180:
            #             f1 = (g['phi'] <= 180).values
            #             f2 = (g['phi'] <= phi_high - 180).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['phi'] <= phi_high).values)

            #         psi_high = df_targ_res['psi'].iat[0] + psi_tol
            #         if psi_high > 180:
            #             f1 = (g['psi'] <= 180).values
            #             f2 = (g['psi'] <= psi_high - 360).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['psi'] <= psi_high).values)

            #         phi_low = df_targ_res['phi'].iat[0] - phi_tol
            #         if phi_low < -180:
            #             f1 = (g['phi'] >= -180).values
            #             f2 = (g['phi'] >= 360 - phi_low).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['phi'] >= phi_low).values)

            #         psi_low = df_targ_res['psi'].iat[0] - psi_tol
            #         if psi_low < -180:
            #             f1 = (g['psi'] >= -180).values
            #             f2 = (g['psi'] >= 360 - psi_low).values
            #             f = f1 | f2
            #             phi_psi_filters.append(f)
            #         else:
            #             phi_psi_filters.append((g['psi'] >= psi_low).values)

            #         phi_psi_filter = np.all(phi_psi_filters, axis=0)
            #         if phi_psi_filter.any():
            #             gs.append(g[phi_psi_filter])
            #     if len(gs) > 1:
            #         df_x_ca = fast_concat(gs)
            #     elif len(gs) == 1:
            #         df_x_ca = gs[0]
            #     elif len(df_x_ca) == 0:
            #         print('            no vdms due to Phi/Psi filter')
            #         return DataFrame()
                df = merge(df, df_x_ca[['CG', 'rota', 'probe_name']].drop_duplicates(),
                           on=['CG', 'rota', 'probe_name'])

            df = df[~((df['chain'] == 'X') & (df['name'] == 'CA'))]

        num_poss_vdms = len(df[['CG', 'rota', 'probe_name']].drop_duplicates())
        targ_coords = \
            merge(df_ideal_ala['name'],
                template.dataframe[template.dataframe['seg_chain_resnum'] == res.seg_chain_resnum],
                on='name')[['c_x', 'c_y', 'c_z']].values
        mob_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values
        df = superpose_df(mob_coords, targ_coords, df)
        if not ignore_CG_for_clash_check:
            clash = Clash(df[df['chain'] == 'Y'].copy(), template.dataframe, tol=vdW_tolerance)
            clash.set_grouping(['CG', 'rota', 'probe_name'])
            clash.find()
            df = merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(),
                    on=['CG', 'rota', 'probe_name'])
            # df = fast_merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)
            # print('num vdms after Y clash', len(df[['CG', 'rota', 'probe_name']].drop_duplicates()))
            if len(df) == 0:
                print('            no vdms due to clashing CGs')
                return DataFrame()
        if aa != 'GLY':
            clash = ClashVDM(df[df['chain'] == 'X'].copy(), template.dataframe)
            clash.set_grouping(['CG', 'rota', 'probe_name'])
            clash.set_exclude(res.seg_chain_resnum)
            clash.setup()
            if res.rotamer:
                clash.find(**dict(tol=0.3))
            else:
                clash.find(**dict(tol=vdW_tolerance))
            df = merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(),
                       on=['CG', 'rota', 'probe_name'])
            # df = fast_merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)

        if len(df) == 0:
            print('            no vdms due to clashing sidechains')
            return DataFrame()
        df['seg_chain_resnum'] = [res.seg_chain_resnum] * len(df)
        # df.loc[:, 'seg_chain_resnum'] = [res.seg_chain_resnum] * len(df)
        print('            Added ', len(df[['CG', 'rota', 'probe_name']].drop_duplicates()),
              'vdMs of', num_poss_vdms, 'possible before clash filter.')
        return df

    def _load_aa(self, aa, cg, template, filter_by_phi_psi, filter_by_phi_psi_exclude_sc,
                 cg_vdm_gr_names, ignore_CG_for_clash_check, vdW_tolerance=0.1):
        print('    Loading', aa)
        try:
            df_parent = read_parquet(self.path_to_database + cg + '/' + aa + '.parquet.gzip',
                                 columns=load_columns)
        except:
            df_parent = read_parquet(self.path_to_database + cg + '/' + aa + '.parquet.gzip')
            cols = list(set(df_parent.columns) & set(load_columns))
            df_parent = df_parent[cols]
            cols_na = set(load_columns) - set(df_parent.columns)
            for col in cols_na:
                df_parent[col] = np.nan
            df_parent = df_parent[load_columns]

        global_filters_xy = [
            df_parent['resnum'] == 10,
            ~((df_parent['chain'] == 'X') & (
                df_parent['name'].isin(['N', 'C', 'O', 'H', 'HA', 'HA1']))), # keep HA2 HA3 for Gly "sidechain"
        ]
        global_filter_xy = np.all(global_filters_xy, axis=0)
        df_parent = df_parent[global_filter_xy]

        if cg_vdm_gr_names is not None:
            try:
                with open(self.path_to_database + '../vdMs_gr_indices/' + cg + '/' + aa + '.pkl', 'rb') as infile:
                    groups = pickle.load(infile)
            except:
                print('vdM_gr_indices folder not found in database.')
                grs = df_parent.groupby(['CG', 'rota', 'probe_name'])
                groups = grs.groups

            gr_names = set()
            for seg_chain_resnum, gns in cg_vdm_gr_names[cg][aa].items():
                gr_names.update(gns)

            df_indices = list(itertools.chain(*[groups[gr_name] for gr_name in groups.keys()
                                                        if gr_name in gr_names]))
            df_parent = df_parent[df_parent.index.isin(df_indices)].copy()

        dd = []
        for res in self.residue_dict[cg][aa]:
            if cg_vdm_gr_names is not None:
                if res.seg_chain_resnum not in cg_vdm_gr_names[cg][aa]:
                    continue
                gr_names = cg_vdm_gr_names[cg][aa][res.seg_chain_resnum]
            else:
                gr_names = None
            loaded = self._load_res(res, df_parent=df_parent, template=template,
                                    filter_by_phi_psi=filter_by_phi_psi,
                                     filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                     aa=aa, gr_names=gr_names, 
                                     ignore_CG_for_clash_check=ignore_CG_for_clash_check,
                                     vdW_tolerance=vdW_tolerance)
            if len(loaded) > 0:
                dd.append(loaded)
        return fast_concat(dd)

    def _get_ligand_cg_df_for_lookup(self, CG_type, CG_group, cg_df, ignore_rmsd_column=()):
        if len(ignore_rmsd_column) > 0:
            dfs = []
            for (_cg, _cg_gr), g in self.ligand_vdm_correspondence.groupby(['CG_type', 'CG_group']):
                if (_cg, _cg_gr) in ignore_rmsd_column:
                    dfs.append(g)
                else:
                    dfs.append(g[g['rmsd'] == True])
            lig_vdm_corr = concat(dfs)
        elif 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence
        is_cg_gr = lig_vdm_corr['CG_group'] == CG_group
        is_cg_type = lig_vdm_corr['CG_type'] == CG_type
        vdm_lig_corr_cg_gr = lig_vdm_corr[is_cg_type & is_cg_gr][
                                ['resname', 'name', 'lig_resname', 'lig_name']].drop_duplicates()
        dfy = pd.merge(cg_df, vdm_lig_corr_cg_gr, on=['resname', 'name'])
        resnames = dfy.resname.unique()
        if len(resnames) > 1:
            dfy = dfy[dfy.resname == resnames[0]]
        dfy = dfy[['lig_resname', 'lig_name']]
        df_lig_cg = pd.merge(dfy, self.ligand_dataframe, left_on=['lig_resname', 'lig_name'],
                                right_on=['resname', 'name'])
        return df_lig_cg

    def _load_ligands_low_mem(self, template, df_cgs, df_cgs_grs, filter_by_phi_psi=False, 
                            filter_by_phi_psi_exclude_sc=True, 
                           superpose_to_cgs=None, residue_chunk_size=10, lig_chunk_size=10000,
                           frac_non_hb_heavy_buried=0.4, hull_tolerance=0,
                           vdW_tolerance=0.0
                           ):
        unique_cg_cg_gr_number = 0
        cg_num = 0 
        for cg in superpose_to_cgs.keys():
            cg_ligand_dataframe = []
            unique_vdm_number = 0
            cg_num += 1

            if cg not in self.residue_dict:
                continue
            print('Loading vdMs of', cg)
            available_aas = [f.split('.')[0] for f in os.listdir(self.path_to_database + cg)
                                if f[0] != '.']
            iterator = [aa for aa in self.residue_dict[cg].keys() if aa in available_aas]
            for aa in iterator:
                #####
                print('    Loading', aa)
                try:
                    df_parent = read_parquet(self.path_to_database + cg + '/' + aa + '.parquet.gzip',
                                        columns=load_columns)
                except:
                    df_parent = read_parquet(self.path_to_database + cg + '/' + aa + '.parquet.gzip')
                    cols = list(set(df_parent.columns) & set(load_columns))
                    df_parent = df_parent[cols]
                    cols_na = set(load_columns) - set(df_parent.columns)
                    for col in cols_na:
                        df_parent[col] = np.nan
                    df_parent = df_parent[load_columns]

                global_filters_xy = [
                    df_parent['resnum'] == 10,
                    ~((df_parent['chain'] == 'X') & (
                        df_parent['name'].isin(['N', 'C', 'O', 'H', 'HA', 'HA1']))), # keep HA2 HA3 for Gly "sidechain"
                ]
                global_filter_xy = np.all(global_filters_xy, axis=0)
                df_parent = df_parent[global_filter_xy]
                #####
                ligand_dataframe = []
                for res_chunk in chunks(self.residue_dict[cg][aa], residue_chunk_size):
                    vdms = []
                    for res in res_chunk:
                        loaded = self._load_res(res, df_parent=df_parent, template=template,
                                                filter_by_phi_psi=filter_by_phi_psi,
                                                filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                                aa=aa, gr_names=None, ignore_CG_for_clash_check=False)
                        if len(loaded) > 0:
                            vdms.append(loaded)
                    if len(vdms) == 0:
                        continue
                    vdms = fast_concat(vdms)
                    lig_dfs = []
                    for cg_gr_num in superpose_to_cgs[cg]:
                        cg_gr = df_cgs_grs.get_group((cg, cg_gr_num))
                        df_cg = vdms[vdms['chain'] == 'Y']
                        df_cg = merge(cg_gr[['resname', 'name']], df_cg, on=['resname', 'name'], sort=False)
                        num_cg_atoms = set(cg_gr.groupby('resname').size()).pop()
                        M = int(len(df_cg) / num_cg_atoms)
                        N = num_cg_atoms
                        R = np.arange(len(df_cg))
                        inds = np.array([R[i::M] for i in range(M)]).flatten()
                        # dataframe_cg_coords = df_cg[['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum']][:M]
                        vectors = df_cg[['c_x', 'c_y', 'c_z']].values[inds].reshape(M, N * 3)
                        gn = cg_gr_num
                        gr = self.ligand_vdm_correspondence_grs.get_group((cg, gn))
                        for z, ligand_conformer in enumerate(self.ligand_conformers):
                            print('        Superposing ligands of', cg, gn, 'for conformer', z+1, '...')
                            gr_ = pd.merge(df_cgs, gr, on=['resname', 'name', 'CG_type', 'CG_group'])
                            gr_.reset_index(inplace=True, drop=True)
                            gr_ = gr_[['lig_resname', 'lig_name']].drop_duplicates()
                            df_cg = pd.merge(gr_, ligand_conformer, left_on=['lig_resname', 'lig_name'],
                                            right_on=['resname', 'name'])
                            coords_lig_cg = df_cg[['c_x', 'c_y', 'c_z']].values
                            lig_coords_cols_vals = ligand_conformer[coords_cols].values
                            df_lig_coords = []
                            for i in range(vectors.shape[0]):
                                coords_cg = vectors[i].reshape(-1, 3)
                                R, mob_coords_com, targ_coords_com = get_rot_trans(mob_coords=coords_lig_cg,
                                                                                targ_coords=coords_cg, weights=None)
                                new_lig_coords_cols_vals = apply_transform_to_coords_cols(R, mob_coords_com,
                                                                                        targ_coords_com,lig_coords_cols_vals)
                                df_lig_coords.append(new_lig_coords_cols_vals.astype(np.float32))
                            num_rows = lig_coords_cols_vals.shape[0]
                            for _df_lig_coords in chunks(df_lig_coords, chunk_size=lig_chunk_size):
                                num_ligs = len(_df_lig_coords)
                                cg_ids = np.array([[i] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
                                rota_ids = np.array([[unique_cg_cg_gr_number] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                                probe_name_ids = np.array([[unique_vdm_number] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                                seg_chain_resnum_ids = np.array([[z] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                                cg_type_ids = np.array([[cg_num] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                                cg_gr_ids = np.array([[1] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                                atom_type_labels = ligand_conformer['atom_type_label'].values
                                atom_type_label_ids = np.array([[atom_type_labels] for _ in range(num_ligs)], dtype=str).flatten()
                                name_labels = ligand_conformer['name'].values
                                name_label_ids = np.array([[name_labels] for _ in range(num_ligs)], dtype=str).flatten()
                                resname_labels = ligand_conformer['resname'].values
                                resname_label_ids = np.array([[resname_labels] for _ in range(num_ligs)], dtype=str).flatten()
                                labels = np.array([cg_ids, rota_ids, probe_name_ids, seg_chain_resnum_ids, cg_type_ids, cg_gr_ids]).T
                                cols = ['CG', 'rota', 'probe_name', 'seg_chain_resnum', 'CG_type', 'CG_group']
                                cols.extend(coords_cols)
                                df_lig_coords_concat = np.vstack(_df_lig_coords)
                                vals = np.hstack((labels, df_lig_coords_concat))
                                df = pd.DataFrame(vals, columns=cols)
                                df['atom_type_label'] = atom_type_label_ids
                                df['resname'] = resname_label_ids
                                df['name'] = name_label_ids
                                df['resnum'] = 10
                                unique_vdm_number += 1
                                print('            ', num_ligs, 'ligands entering clash/burial filters...')
                                result = self._ligand_gauntlet(df, template, frac_non_hb_heavy_buried=frac_non_hb_heavy_buried,
                                                            hull_tolerance=hull_tolerance, vdW_tolerance=vdW_tolerance)
                                if not result:
                                    print('            ', 'No', 'ligands passed clash/burial filters.')
                                else:
                                    df, _, num_ligs = result
                                    print('            ', num_ligs, 'ligands passed clash/burial filters.')
                                    if len(df) > 0:
                                        lig_dfs.append(df)
                        unique_cg_cg_gr_number += 1
                    if len(lig_dfs) > 0:  # cascade concatenations to save memory space.
                        lig_dfs = fast_concat(lig_dfs)
                        ligand_dataframe.append(lig_dfs)
                if len(ligand_dataframe) > 0:
                    ligand_dataframe = fast_concat(ligand_dataframe)
                    cg_ligand_dataframe.append(ligand_dataframe)
            if len(cg_ligand_dataframe) > 0:
                cg_ligand_dataframe = fast_concat(cg_ligand_dataframe)
                self.ligand_dataframe.append(cg_ligand_dataframe)
        if len(self.ligand_dataframe) == 0:
            raise Exception('        No ligands were successfully loaded.')
        self.ligand_dataframe = fast_concat(self.ligand_dataframe)
        self.ligand_dataframe_grs = self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
                                                                   'seg_chain_resnum', 'CG_type', 'CG_group'])
        print('        Loaded', len(self.ligand_dataframe_grs), 'ligands')

    def _get_segchresnum_and_coord_dicts(self, template, lig_vdm_corr, df_cgs,
                                        ignore_rmsd_column=()):

        if len(ignore_rmsd_column) > 0:
            dfs = []
            for (_cg, _cg_gr), g in self.ligand_vdm_correspondence.groupby(['CG_type', 'CG_group']):
                if (_cg, _cg_gr) in ignore_rmsd_column:
                    dfs.append(g)
                else:
                    dfs.append(g[g['rmsd'] == True])
            lig_vdm_corr = concat(dfs)
        elif 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence

        df_cgs = lig_vdm_corr[['resname', 'name', 'CG_type', 'CG_group']].drop_duplicates()

        allowed_seg_chain_resnums = defaultdict(set)
        for cg in self.residue_dict.keys():
            for aa in self.residue_dict[cg].keys():
                for res in self.residue_dict[cg][aa]:
                    allowed_seg_chain_resnums[cg].add(res.seg_chain_resnum)

        template_ca = template.dataframe[template.dataframe['name'] == 'CA']
        ca_coords = template_ca[['c_x', 'c_y', 'c_z']].values
        ca_seg_chain_resnums = template_ca['seg_chain_resnum'].values
        nbrs_template_ca = NearestNeighbors(radius=15).fit(ca_coords)
        seg_chain_resnum_dict = defaultdict(dict)
        cg_coords_dict = defaultdict(dict)

        for (cg, gn), gr in lig_vdm_corr.groupby(['CG_type', 'CG_group']):
            print('Finding template-backbone neighbors of ligands', cg, gn, '...')
            gn_ = int(gn)
            gr_ = merge(df_cgs, gr, on=['resname', 'name', 'CG_type', 'CG_group'])
            gr_.reset_index(inplace=True, drop=True)
            gr_ = gr_[['lig_resname', 'lig_name']].drop_duplicates()
            # df_cg = merge(gr_, self.ligand_dataframe, left_on=['lig_resname', 'lig_name'],
            #               right_on=['resname', 'name'])
            _cg_df = cg_dfs[cg]
            df_cg = self._get_ligand_cg_df_for_lookup(cg, gn_, _cg_df, 
                                                        ignore_rmsd_column=ignore_rmsd_column) 
            num_cg_atoms = len(gr_)
            # print(df_cg)
            M = int(len(df_cg) / num_cg_atoms)
            N = num_cg_atoms
            R = np.arange(len(df_cg))
            inds_ = np.array([R[i::M] for i in range(M)]).flatten()
            # print(inds_)
            inds = inds_.reshape(M, N)
            if self.ligand_dataframe_neighbors is None:
                dataframe_cg_coords = df_cg[['CG', 'rota', 'probe_name',
                                                 'seg_chain_resnum', 'CG_type', 'CG_group']][:M]
                self.ligand_dataframe_neighbors = dataframe_cg_coords
            vectors = df_cg[['c_x', 'c_y', 'c_z']].values 
            cg_coords_dict[cg][gn] = vectors[inds_].reshape(M, N*3)
            ind_neighbors = nbrs_template_ca.radius_neighbors(vectors, return_distance=False) 
            
            for row in range(M):
                combined_inds = np.concatenate([ind_neighbors[i] for i in inds[row]])
                scrns = {ca_seg_chain_resnums[j] for j in combined_inds if ca_seg_chain_resnums[j] 
                         in allowed_seg_chain_resnums[cg]}
                for scrn in scrns:
                    try:
                        seg_chain_resnum_dict[scrn][cg][gn_].add(row)
                    except:
                        seg_chain_resnum_dict[scrn][cg] = defaultdict(set)       
                        seg_chain_resnum_dict[scrn][cg][gn_].add(row)

        return seg_chain_resnum_dict, cg_coords_dict

    def _get_cg_vdm_gr_names(self, template, seg_chain_resnum_dict, cg_coords_dict,
                            path_to_nbrs_database_, path_to_nbrs_database_groupnames,
                            distance_metric):

        cg_vdm_gr_names = defaultdict(dict)

        allowed_seg_chain_resnums = defaultdict(dict)
        for cg in self.residue_dict.keys():
            for aa in self.residue_dict[cg].keys():
                for res in self.residue_dict[cg][aa]:
                    try:
                        allowed_seg_chain_resnums[res.seg_chain_resnum][cg].add(aa)
                    except:
                        allowed_seg_chain_resnums[res.seg_chain_resnum][cg] = set()
                        allowed_seg_chain_resnums[res.seg_chain_resnum][cg].add(aa)

        for seg_chain_resnum in seg_chain_resnum_dict.keys():

            print('Finding neighbors of', seg_chain_resnum, '...')
            seg_aa, chain_aa, res_aa = seg_chain_resnum
            if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                        (template.dataframe['chain'] == chain_aa) &
                                        (template.dataframe['resnum'] == res_aa)].copy()
                m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                for _name in ['N', 'CA', 'C']])
                t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com) 
            else:
                R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]                         
            for cg in seg_chain_resnum_dict[seg_chain_resnum].keys():
                if cg not in allowed_seg_chain_resnums[seg_chain_resnum]:
                    continue
                print('\t', cg)
                for aa in self.residue_dict[cg].keys():
                    if aa not in allowed_seg_chain_resnums[seg_chain_resnum][cg]:
                        continue
                    print('\t\t', aa)
                    #open cg/aa
                    if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                        print('\t\t\t AA not found in CG database. Skipping...')
                        continue
                    if distance_metric == 'rmsd':
                        with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            nbrs = pickle.load(f)
                        num_cg_atoms = nbrs._fit_X.shape[1] / 3 
                    elif distance_metric == 'maxdist':
                        with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            nbrs = pickle.load(f)
                    with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                        groupnames = pickle.load(f)
                    
                    for cg_gr, lig_indices in seg_chain_resnum_dict[seg_chain_resnum][cg].items():
                        print('\t\t\t', cg_gr)
                        cg_coords = cg_coords_dict[cg][cg_gr]
                        if len(cg_coords) == 0:
                            continue
                        N = cg_coords.shape[-1]
                        lig_indices = list(lig_indices)
                        cg_coords = cg_coords[lig_indices].reshape(-1, int(N/3), 3)
                        cg_coords = apply_transform(R, mob_com, targ_com, cg_coords)
                        cg_coords = cg_coords.reshape(-1, N)

                        if distance_metric == 'rmsd':
                            radius = self.cg_rmsds[cg] * np.sqrt(num_cg_atoms)
                        else:
                            radius = self.cg_max_dists[cg]
                        inds = nbrs.radius_neighbors(cg_coords, radius=radius, return_distance=False)

                        for i in range(len(inds)):
                            nbr_indices = inds[i]
                            nbr_groupnames = {groupnames[k] for k in nbr_indices}
                            try:
                                cg_vdm_gr_names[cg][aa][seg_chain_resnum].update(nbr_groupnames)
                            except:
                                cg_vdm_gr_names[cg][aa] = defaultdict(set)
                                cg_vdm_gr_names[cg][aa][seg_chain_resnum].update(nbr_groupnames)
        return cg_vdm_gr_names

    def _load_vdms_ligands_low_mem(self, template, path_to_database=None, filter_by_phi_psi=False, 
                                  filter_by_phi_psi_exclude_sc=True, 
                                  superpose_to_cgs=None, residue_chunk_size=100, lig_chunk_size=10000,
                                  frac_non_hb_heavy_buried=0.4, hull_tolerance=0, distance_metric='rmsd',
                                  cg_rmsds=None, cg_max_dists=None, max_dist_criterion=False,
                                  ignore_rmsd_column=(), vdW_tolerance=0.0, use_preloaded_ligands=False):

        """
        for each cg in cgs_to_be_superposed:
            for each aa:
                load up to 100 residues at a time
                for residues in residue_blocks_of_100:
                    for res in residues:
                        load vdms allowed by resfile
                        for cg_gr in superpose_to_cgs[cg]:
                           load ligands (superpose or load from db, filter)
                        remove vdms

        get residue ca nbrs of loaded ligands. dictionary of (cg, cg_gr) : residue nbrs (list with indices same as lig indices)
        make dictionary with residue as keys, cg, cg_gr as next keys, and list of ligand indices as values

        for seg_chain_resnum in residues_dict.keys():
            transform ligand coordinates to reference frame of residue
            for cg in residues_dict[seg_chain_resnum].keys():
                for allowed aa at residue
                    open cg/aa for residue
                    for cg_gr in residue/cg
                        extract cg coords for each ligand
                        lig_coords
                        look up nbrs (all_nbr_indices = nbrs.radius_nbrs(lig_coords))
                        for i, nbr_indices in enumerate(all_nbr_indices):
                            lig_gr_name = lig_gr_names[i]
                            vdm_gr_names = []
                            for nbr_index in nbr_indices:
                                _vdm_gr_name = vdm_group_names[nbr_index]
                                _vdm_gr_name.append(cg)
                                _vdm_gr_name.append(seg_ch_resnum)
                                vdm_gr_name = (cg, _vdm_gr_name)
                                vdm_gr_names.append(vdm_gr_name)
                            if len(vdm_gr_names) > 0:
                                add to dict lig_vdm_gr_names[lig_gr_name].extend(vdm_gr_names)


        goal is to iterate through the loaded ligands and add the appropriate vdms to the pose
        
        loaded_ligs_grs 
            is grouped dataframe of loaded ligands

        lig_vdm_gr_names 
            is a dictionary with key of ligand_gr_name and 
            value is a list of tuples of the form (cg, vdm_gr_name)

        vdm_gr_name 
            has the form (CG, rota, probe_name, CG_type, seg_chain_resnum)

        self.cg_dict_grs[cg]
            vdms are only those that are nbrs to a ligand

        for lig_gr_name in loaded_ligs_grs.groups.keys():
            vdms = []
            vdm_gr_names = lig_vdm_gr_names[lig_gr_name]
            for cg, vdm_gr_name in vdm_gr_names:
                if vdm_gr_name in self.cg_dict_grs[cg].groups:
                    vdm = self.cg_dict_grs[cg].get_group[vdm_gr_name]
                    vdms.append(vdm)

        then do clash filter and everything else about adding to pose.

        """
        if path_to_database is None:
            path_to_database = self.path_to_database

        path_to_nbrs_database = path_to_database + '../nbrs/'

        if superpose_to_cgs is None:
            superpose_to_cgs = defaultdict(set)
            for cg, cg_gr in self.ligand_vdm_correspondence_grs.groups.keys():
                superpose_to_cgs[cg].add(cg_gr)

        if 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence
        df_cgs = lig_vdm_corr[['resname', 'name', 'CG_type', 'CG_group']].drop_duplicates()
        df_cgs_grs = df_cgs.groupby(['CG_type', 'CG_group'])

        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'

        self.cg_rmsds = {cg: 0.4 for cg in self.cgs}
        if cg_rmsds is not None:
            for key, val in cg_rmsds.items():
                self.cg_rmsds[key] = val

        if max_dist_criterion:
            self._max_dist_criterion = True
            self.cg_max_dists = {cg: 0.6 for cg in self.cgs}
            if cg_max_dists is not None:
                for key, val in cg_max_dists.items():
                    self.cg_max_dists[key] = val

        if not use_preloaded_ligands:
            self._load_ligands_low_mem(template, 
                                    df_cgs=df_cgs, 
                                    df_cgs_grs=df_cgs_grs, 
                                    filter_by_phi_psi=filter_by_phi_psi, 
                                    filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc, 
                                    superpose_to_cgs=superpose_to_cgs, 
                                    residue_chunk_size=residue_chunk_size, 
                                    lig_chunk_size=lig_chunk_size,
                                    frac_non_hb_heavy_buried=frac_non_hb_heavy_buried, 
                                    hull_tolerance=hull_tolerance, 
                                    vdW_tolerance=vdW_tolerance
                            )

        seg_chain_resnum_dict, cg_coords_dict = \
                    self._get_segchresnum_and_coord_dicts(template, 
                                                         lig_vdm_corr=lig_vdm_corr, 
                                                         df_cgs=df_cgs,
                                                         ignore_rmsd_column=ignore_rmsd_column)

        cg_vdm_gr_names = self._get_cg_vdm_gr_names(template, 
                                                    seg_chain_resnum_dict=seg_chain_resnum_dict, 
                                                    cg_coords_dict=cg_coords_dict,
                                                    path_to_nbrs_database_=path_to_nbrs_database_, 
                                                    path_to_nbrs_database_groupnames=path_to_nbrs_database_groupnames,
                                                    distance_metric=distance_metric)

        return cg_vdm_gr_names

    def load_vdms_ligands_low_mem(self, template, path_to_database=None, filter_by_phi_psi=False, 
                            filter_by_phi_psi_exclude_sc=True, 
                           superpose_to_cgs=None, residue_chunk_size=100, lig_chunk_size=10000,
                           frac_non_hb_heavy_buried=0.4, hull_tolerance=0, distance_metric='rmsd',
                           cg_rmsds=None, cg_max_dists=None, max_dist_criterion=False,
                           ignore_rmsd_column=(), vdW_tolerance_vdms=0.1, vdW_tolerance_ligands=0.0,
                           use_preloaded_ligands=False):

        cg_vdm_gr_names = self._load_vdms_ligands_low_mem(template,
                                                          path_to_database=path_to_database,
                                                          filter_by_phi_psi=filter_by_phi_psi,
                                                          filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                                          superpose_to_cgs=superpose_to_cgs,
                                                          residue_chunk_size=residue_chunk_size,
                                                          lig_chunk_size=lig_chunk_size,
                                                          frac_non_hb_heavy_buried=frac_non_hb_heavy_buried, 
                                                          hull_tolerance=hull_tolerance, 
                                                          distance_metric=distance_metric,
                                                          cg_rmsds=cg_rmsds, 
                                                          cg_max_dists=cg_max_dists, 
                                                          max_dist_criterion=max_dist_criterion,
                                                          ignore_rmsd_column=ignore_rmsd_column,
                                                          vdW_tolerance=vdW_tolerance_ligands,
                                                          use_preloaded_ligands=use_preloaded_ligands)

        self.load_vdms(template, filter_by_phi_psi=filter_by_phi_psi,
                        filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                        num_cpus=None, run_parallel=False, cg_vdm_gr_names=cg_vdm_gr_names,
                        ignore_CG_for_clash_check=False, vdW_tolerance=vdW_tolerance_vdms)

    def load_vdms(self, template, filter_by_phi_psi=False, filter_by_phi_psi_exclude_sc=True,
                  num_cpus=None, run_parallel=False, cg_vdm_gr_names=None, 
                  ignore_CG_for_clash_check=False, vdW_tolerance=0.1):
        # add way to make kernel density
        num_cpus = num_cpus or os.cpu_count() - 2
        if run_parallel:
            cg_dict = dict()
            cg_dict_grs = dict()
            with Pool(num_cpus) as p:
                for cg in self.residue_dict.keys():
                    print('Loading', cg)
                    available_aas = [f.split('.')[0] for f in os.listdir(self.path_to_database + cg)
                                        if f[0] != '.']
                    f = partial(self._load_aa, cg=cg, template=template,
                                filter_by_phi_psi=filter_by_phi_psi,
                                filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                cg_vdm_gr_names=cg_vdm_gr_names, 
                                ignore_CG_for_clash_check=ignore_CG_for_clash_check,
                                vdW_tolerance=vdW_tolerance)
                    iterator = [aa for aa in self.residue_dict[cg].keys() if aa in available_aas]
                    # num_cpus = min(num_cpus, len(iterator))
                    dfs = p.map(f, iterator, chunksize=1) #chuncksize=int(len(iterator) / num_cpus))
                    # cg_dict[cg] = fast_concat(dfs)
                    cg_dict[cg] = concat(dfs, axis=0, ignore_index=True)
                    if len(cg_dict[cg]) == 0:
                        continue
                # self.cg_dict[cg] = concat(self.cg_dict[cg])
                    cg_dict[cg].loc[:, 'CG_type'] = cg
                    # cg_dict[cg].reset_index(drop=True, inplace=True)
                    try:
                        cg_dict[cg].loc[cg_dict[cg]['rotamer'].isna(), 'rotamer'] = 'OUTLIER'
                    except:
                        cg_dict[cg]['rotamer'] = cg_dict[cg]['rotamer'].astype(str)
                        cg_dict[cg].loc[cg_dict[cg]['rotamer'].isna(), 'rotamer'] = 'NAN'
                        cg_dict[cg]['rotamer'] = cg_dict[cg]['rotamer'].astype('category')
                    cg_dict[cg] = cg_dict[cg].drop_duplicates() # thought i didn't need this, but...
                    cg_dict_grs[cg] = cg_dict[cg].groupby(['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                    print('Loaded ', len(cg_dict_grs[cg]), 'vdMs of', cg)
                    # cg_dict_grs[cg] = Group(dict(tuple(cg_dict[cg].groupby(['CG', 'rota', 'probe_name', 'seg_chain_resnum']))))
                    # print('Loaded ', len(cg_dict_grs[cg].dict.keys()), 'vdMs of', cg)
            self.cg_dict = cg_dict
            self.cg_dict_grs = cg_dict_grs
        else:
            for cg in self.residue_dict.keys():
                if cg_vdm_gr_names is not None:
                    if cg not in cg_vdm_gr_names:
                        continue
                print('Loading', cg)
                available_aas = [f.split('.')[0] for f in os.listdir(self.path_to_database + cg)
                                 if f[0] != '.']
                for aa in self.residue_dict[cg].keys():
                    if cg_vdm_gr_names is not None:
                        if aa not in cg_vdm_gr_names[cg]:
                            continue
                    if aa not in available_aas:
                        continue
                    vdms = self._load_aa(aa, cg=cg, template=template,
                                         filter_by_phi_psi=filter_by_phi_psi,
                                         filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                         cg_vdm_gr_names=cg_vdm_gr_names,
                                         ignore_CG_for_clash_check=ignore_CG_for_clash_check,
                                         vdW_tolerance=vdW_tolerance)
                    if len(vdms) > 0:   
                        self.cg_dict[cg].append(vdms)
                if len(self.cg_dict[cg]) == 0:
                    continue
                elif len(self.cg_dict[cg]) == 1:
                    self.cg_dict[cg] = self.cg_dict[cg][0]
                else:
                    self.cg_dict[cg] = fast_concat(self.cg_dict[cg])
                # self.cg_dict[cg] = concat(self.cg_dict[cg], ignore_index=True)
                self.cg_dict[cg].loc[:, 'CG_type'] = cg
                # self.cg_dict[cg].reset_index(drop=True, inplace=True)
                try:
                    self.cg_dict[cg].loc[self.cg_dict[cg]['rotamer'].isna(), 'rotamer'] = 'OUTLIER'
                except:
                    self.cg_dict[cg]['rotamer'] = self.cg_dict[cg]['rotamer'].astype(str)
                    self.cg_dict[cg].loc[self.cg_dict[cg]['rotamer'].isna(), 'rotamer'] = 'NAN'
                    self.cg_dict[cg]['rotamer'] = self.cg_dict[cg]['rotamer'].astype('category')

                self.cg_dict[cg] = self.cg_dict[cg].drop_duplicates() # thought i didn't need this, but...
                self.cg_dict_grs[cg] = self.cg_dict[cg].groupby(['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                print('Loaded ', len(self.cg_dict_grs[cg]), 'vdMs of', cg)
                # self.cg_dict_grs[cg] = Group(
                #     dict(tuple(self.cg_dict[cg].groupby(['CG', 'rota', 'probe_name', 'seg_chain_resnum']))))
                # print('Loaded ', len(self.cg_dict_grs[cg].dict.keys()), 'vdMs of', cg)

    def set_ligand_vdm_correspondence(self, path_to_ligand_file):
        ligand_vdm_correspondence = read_lig_txt(path_to_ligand_file)
        # self.ligand_vdm_correspondence = ligand_vdm_correspondence.sort_values(by=['lig_resname', 'lig_name'], ignore_index=True)
        rows = []
        for _, gr in ligand_vdm_correspondence.groupby(['CG_type', 'CG_group']):
            rn_grs = [rn_gr for n, rn_gr in gr.groupby('resname')]
            len_gr = len(rn_grs[0])
            num_grs = len(rn_grs)
            for i in range(len_gr):
                for j in range(num_grs):
                    rows.append(rn_grs[j].iloc[[i]])
        self.ligand_vdm_correspondence = concat(rows)
        # self.ligand_vdm_correspondence = ligand_vdm_correspondence

        self.ligand_vdm_correspondence_grs = self.ligand_vdm_correspondence.groupby(['CG_type', 'CG_group'])
        self.cgs = set(self.ligand_vdm_correspondence['CG_type'])
        for (cg, cg_gr), g in self.ligand_vdm_correspondence_grs:
            self.cg_atom_map[(cg, cg_gr)] = set(g.lig_name)
        # self.cg_atom_map = {(cg, cg_gr): set(g.lig_name) for (cg, cg_gr), g in self.ligand_vdm_correspondence_grs}

        for n, group in self.ligand_vdm_correspondence_grs:
            self.cg_ligand_coverage[n[0]][n[1]] = group['CG_ligand_coverage'].iat[0]

        for cg1, cg_gr1 in self.ligand_vdm_correspondence_grs.groups:
            for cg2, cg_gr2 in self.ligand_vdm_correspondence_grs.groups:
                if (cg1, cg_gr1) == (cg2, cg_gr2):
                    continue
                if self._is_alt_cg(cg1, cg_gr1, cg2, cg_gr2, self.cg_atom_map, 
                                   self.ligand_vdm_correspondence, self.ligand_vdm_correspondence_grs):
                    self.cg_alts[(cg1, cg_gr1)].add((cg2, cg_gr2))

        for cg, names in self.cg_atom_map.items():
            for name in names:
                self.atom_cg_map[name].add(cg)

        self.ligand_names = {n for names in self.cg_atom_map.values() for n in names}

    def set_atom_cg_map_by_atomtype(self):
        self.ligand_atom_types = {n: at for _, (n, at) in self.ligand_rep[['name', 'atom_type_label']].iterrows()}
        polars = {'h_pol', 'n', 'o', 's'}
        for (cg_, cg_gr_), df in self.ligand_vdm_correspondence_grs:
            for _, row in df.iterrows():
                try:
                    resn = row['resname']
                    nm = row['name']
                    lig_name = row['lig_name']
                    lig_atom_type_label = self.ligand_rep[self.ligand_rep['name'] == lig_name]['atom_type_label'].iat[0]
                    cg_atom_type_label = atom_type_dict[resn][nm]
                    if lig_atom_type_label in polars and cg_atom_type_label in polars:
                        self.atom_cg_map_by_atomtype[lig_name].add((cg_, cg_gr_))
                    elif lig_atom_type_label not in polars and cg_atom_type_label not in polars:
                        self.atom_cg_map_by_atomtype[lig_name].add((cg_, cg_gr_))
                except IndexError:
                    raise IndexError('Could not find ligand name {} in ligand_rep.\
                                     Check mislabeled atoms in ligand.txt file'.format(lig_name))

    @staticmethod
    def _is_alt_cg(cg1, cg_gr1, cg2, cg_gr2, atom_map, lig_vdm_corr, lig_vdm_corr_grs):
        d1 = lig_vdm_corr_grs.get_group((cg1, cg_gr1))[['lig_name', 'resname', 'name']]
        d2 = lig_vdm_corr_grs.get_group((cg2, cg_gr2))[['lig_name', 'resname', 'name']]
        set1 = set(map(tuple, d1.values)) 
        set2 = set(map(tuple, d2.values))
        for m1 in set1.copy():
            if m1 in set2.copy():
                set1.remove(m1)
                set2.remove(m1)
        dict_pair = defaultdict(list)
        for a,r,b in set1:
            dict_pair[a].append((r,b)) 
        for a,r,b in set2:
            dict_pair[a].append((r,b))
        for key in dict_pair.keys():
            name_set = set()
            for resn, nm in dict_pair[key]:
                name_set.add(atom_type_dict[resn][nm])
            len_set_no_polars = len(name_set - {'h_pol', 'n', 'o', 's'})
            if len_set_no_polars != len(name_set):
                return False
        uniq_names = atom_map[(cg1, cg_gr1)] - atom_map[(cg2, cg_gr2)]
        if len(uniq_names) / len(atom_map[(cg1, cg_gr1)]) < 0.5:
            for uniq_name in uniq_names:
                f1 = lig_vdm_corr.CG_type==cg1
                f2 = lig_vdm_corr.CG_group==cg_gr1
                f3 = lig_vdm_corr.lig_name==uniq_name
                resn, nm = lig_vdm_corr[f1 & f2 & f3][['resname', 'name']].values[0]
                if atom_type_dict[resn][nm] in {'h_pol', 'n', 'o', 's'}:
                    return False
            return True
        else:
            return False

    def set_cg_neighbors(self, max_dist_criterion=False, cg_rmsds=None, cg_max_dists=None):
        print('Setting neighbors...')
        if self.ligand_vdm_correspondence is None:
            raise Exception('Set ligand vdm correspondence before making Neighbors')

        self.cg_rmsds = {cg: 0.4 for cg in self.cg_dict}
        if cg_rmsds is not None:
            for key, val in cg_rmsds.items():
                self.cg_rmsds[key] = val

        if max_dist_criterion:
            self._max_dist_criterion = True
            self.cg_max_dists = {cg: 0.6 for cg in self.cg_dict}
            if cg_max_dists is not None:
                for key, val in cg_max_dists.items():
                    self.cg_max_dists[key] = val

        #this fn works because df_corr is sorted by lig_resname and lig_name
        # to create an alternating (by name) df structure.
        if 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence
        lig_vdm_cols = ['resname', 'name', 'CG_type', 'CG_group', 
                        'is_not_donor', 'is_donor', 'is_not_acceptor', 'is_acceptor']
        df_cgs = lig_vdm_corr[lig_vdm_cols].drop_duplicates()
        loaded_cgs = set()
        for (cg, cg_gr_num), cg_gr in df_cgs.groupby(['CG_type', 'CG_group']):
            if cg in loaded_cgs:
                continue
            loaded_cgs.add(cg)
            if cg not in self.cg_dict:
                continue
            if len(self.cg_dict[cg]) == 0:
                continue
            df_cg = self.cg_dict[cg][self.cg_dict[cg]['chain'] == 'Y']
            df_cg = merge(cg_gr[['resname', 'name']], df_cg, on=['resname', 'name'], sort=False)
            rmsd = self.cg_rmsds[cg]
            num_cg_atoms = set(cg_gr.groupby('resname').size()).pop()
            self.cg_num_atoms[cg] = num_cg_atoms
            M = int(len(df_cg) / num_cg_atoms)
            N = num_cg_atoms
            R = np.arange(len(df_cg))
            inds = np.array([R[i::M] for i in range(M)]).flatten()
            dataframe_cg_coords = df_cg[['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum']][:M].values
            vectors = df_cg[['c_x', 'c_y', 'c_z']].values[inds].reshape(M, N * 3)
            if max_dist_criterion:
                max_dist = self.cg_max_dists[cg]
                neighbors = NearestNeighbors(radius=max_dist, algorithm='ball_tree', metric=get_max)
            else:
                neighbors = NearestNeighbors(radius=np.sqrt(num_cg_atoms) * rmsd, algorithm='ball_tree')
            neighbors.fit(vectors)
            self.cg_neighbors[cg] = neighbors
            self.cg_dataframe_neighbors[cg] = dataframe_cg_coords
            df_cg_type = df_cgs[df_cgs.CG_type == cg]
            if df_cg_type['is_not_donor'].any() or df_cg_type['is_donor'].any():
                if cg not in self.is_donor:
                    print('Storing CG', cg, 'donor info...')
                    is_donor = df_cg['is_donor'][:M].values
                    self.is_donor[cg] = is_donor
            if df_cg_type['is_not_acceptor'].any() or df_cg_type['is_acceptor'].any():
                if cg not in self.is_acceptor:
                    print('Storing CG', cg, 'acceptor info...')
                    is_acceptor = df_cg['is_acceptor'][:M].values
                    self.is_acceptor[cg] = is_acceptor
            seg_chain_resnum_scores = df_cg[['seg_chain_resnum', 'C_score_bb_ind']][:M].values
            self.cg_seg_chain_resnum_scores[cg] = seg_chain_resnum_scores
    
    def set_cg_com_neighbors(self):
        for cg in self.cg_neighbors.keys():
            num_atoms = self.cg_num_atoms[cg]
            coms = self.cg_neighbors[cg]._fit_X.reshape(-1, num_atoms, 3).mean(axis=1)
            self._cg_com_tree[cg] = BallTree(coms)

    def find_ligand_cg_neighbors(self, maxdists=None, rmsds=None):
        """

        Parameters
        ----------
        dists : dict
            Dictionary with keys of CG names and values of distances for neighbor search.
            If rmsd is desired, the desired rmsd should be multiplied by the sqrt of the
            number of atoms in the CG, in order to give the correct distance.

        Returns
        -------

        """
        if 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence
        df_cgs = lig_vdm_corr[['resname', 'name', 'CG_type', 'CG_group']].drop_duplicates()
        for (cg, gn), gr in lig_vdm_corr.groupby(['CG_type', 'CG_group']):
            if cg not in self.cg_dict:
                continue
            if cg not in self.cg_neighbors:
                continue
            if len(self.cg_dict[cg]) == 0:
                continue
            print('Finding neighbors of', cg, gn, '...')
            gn_ = int(gn)
            gr_ = merge(df_cgs, gr, on=['resname', 'name', 'CG_type', 'CG_group'])
            gr_.reset_index(inplace=True, drop=True)
            gr_ = gr_[['lig_resname', 'lig_name']].drop_duplicates()
            df_cg = merge(gr_, self.ligand_dataframe, left_on=['lig_resname', 'lig_name'],
                          right_on=['resname', 'name'])
            num_cg_atoms = len(gr_)
            M = int(len(df_cg) / num_cg_atoms)
            N = num_cg_atoms
            R = np.arange(len(df_cg))
            inds = np.array([R[i::M] for i in range(M)]).flatten()
            if self.ligand_dataframe_neighbors is None:
                dataframe_cg_coords = df_cg[['CG', 'rota', 'probe_name',
                                                 'seg_chain_resnum', 'CG_type', 'CG_group']][:M]
                self.ligand_dataframe_neighbors = dataframe_cg_coords
            vectors = df_cg[['c_x', 'c_y', 'c_z']].values[inds].reshape(M, N * 3)
            if maxdists is not None and cg in maxdists:
                if type(self.cg_neighbors[cg].metric) == str:
                    raise Exception('cg_neighbors are not set with maximum distance metric (get_max). \
                                    Perhaps you meant to use rmsds instead?')
                dist, ind_neighbors = self.cg_neighbors[cg].radius_neighbors(vectors, return_distance=True,
                                                                            radius=maxdists[cg])
            elif rmsds is not None and cg in rmsds:
                if self.cg_neighbors[cg].metric not in ['minkowski', 'euclidean']:
                    raise Exception('cg_neighbors are not set with euclidean metric, so rmsds will not be computed. \
                                    Perhaps you meant to use maxdists instead?')
                dist, ind_neighbors = self.cg_neighbors[cg].radius_neighbors(vectors, return_distance=True,
                                                                            radius=np.sqrt(num_cg_atoms) * rmsds[cg])
            else:
                dist, ind_neighbors = self.cg_neighbors[cg].radius_neighbors(vectors, return_distance=True)
            if gr['is_not_donor'].any():
                is_not_don = ~self.is_donor[cg]
                inds_not_don = np.where(is_not_don)[0]
                for i in range(ind_neighbors.shape[0]):
                    _ind_neighbors = ind_neighbors[i]
                    tf = np.in1d(_ind_neighbors, inds_not_don)
                    ind_neighbors[i] = _ind_neighbors[tf]
                    dist[i] = dist[i][tf]
            elif gr['is_donor'].any():
                is_don = self.is_donor[cg]
                inds_don = np.where(is_don)[0]
                for i in range(ind_neighbors.shape[0]):
                    _ind_neighbors = ind_neighbors[i]
                    tf = np.in1d(_ind_neighbors, inds_don)
                    ind_neighbors[i] = _ind_neighbors[tf]
                    dist[i] = dist[i][tf]
            elif gr['is_not_acceptor'].any():
                is_not_acc = ~self.is_acceptor[cg]
                inds_not_acc = np.where(is_not_acc)[0]
                for i in range(ind_neighbors.shape[0]):
                    _ind_neighbors = ind_neighbors[i]
                    tf = np.in1d(_ind_neighbors, inds_not_acc)
                    ind_neighbors[i] = _ind_neighbors[tf]
                    dist[i] = dist[i][tf]
            elif gr['is_acceptor'].any():
                is_acc = self.is_acceptor[cg]
                inds_acc = np.where(is_acc)[0]
                for i in range(ind_neighbors.shape[0]):
                    _ind_neighbors = ind_neighbors[i]
                    tf = np.in1d(_ind_neighbors, inds_acc)
                    ind_neighbors[i] = _ind_neighbors[tf]
                    dist[i] = dist[i][tf]
            self.ligand_neighbors_indices[cg][gn_] = ind_neighbors
            self.ligand_neighbors_dists[cg][gn_] = dist

    def get_lig_cg_coords(self, lig_df, random_CG_gr_choice=True, CG_gr_index=None):
        if 'rmsd' in self.ligand_vdm_correspondence.columns:
            lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = self.ligand_vdm_correspondence
        lig_cg_coords = dict()
        for cg_lig_cov, vdm_lig_corr in lig_vdm_corr.groupby('CG_ligand_coverage'):
            for cg, vdm_lig_corr_cg in vdm_lig_corr.groupby('CG_type'):
                cg_grs = sorted(set(vdm_lig_corr_cg['CG_group']))
                if random_CG_gr_choice:
                    cg_grs = [random.choice(cg_grs)]
                elif CG_gr_index is not None:
                    cg_grs = [cg_grs[CG_gr_index]]
                for cg_gr in cg_grs:
                    vdm_lig_corr_cg_gr = vdm_lig_corr_cg[vdm_lig_corr_cg['CG_group'] == cg_gr][
                        ['lig_resname', 'lig_name']].drop_duplicates()
                    df_lig_cg = pd.merge(vdm_lig_corr_cg_gr, lig_df, left_on=['lig_resname', 'lig_name'],
                                         right_on=['resname', 'name'])
                    lig_cg_coords[(cg, cg_gr)] = df_lig_cg[['c_x', 'c_y', 'c_z']].values.flatten()
        return lig_cg_coords

    def _remove_dist_buried(self, df_ligs, template):
        if self.constraints is None:
            return df_ligs
        csts = self.constraints[((~self.constraints['dist_buried'].isna())
                        & (self.constraints['contact_type'] == set())
                        & (self.constraints['Has_vdM_in_pose'].isna()) | 
                        ~self.constraints['atom_dist_resnum'].isna())]
        # csts = self.constraints
        csts_grs = csts.groupby('cst_group')
        df_t = template.dataframe
        cst_filters = []
        for n, gr in csts_grs:
            if len(gr) > 1 and not ((~gr.dist_buried.isna()).all() or (~gr.atom_dist_resnum.isna()).all()):
                continue
            gr_cst_filters = []
            for m, row in gr.iterrows():
                lig_resname = row.lig_resname
                lig_name = row.lig_name
                if not np.isnan(row.dist_buried):
                    dist_buried = row.dist_buried
                    lessthan = row.dist_lessthan
                    df_filter = (df_ligs.resname == lig_resname) & (df_ligs.name == lig_name)
                    if dist_buried == 0:
                        if 'in_hull' not in df_ligs.columns:
                            in_hull = template.alpha_hull.pnts_in_hull(df_ligs[['c_x', 'c_y', 'c_z']][df_filter].values)
                        else:
                            in_hull = df_ligs['in_hull'][df_filter].values
                        if lessthan:
                            gr_cst_filters.append(~in_hull) # <0 means not in hull
                        else:
                            gr_cst_filters.append(in_hull)
                    else:
                        if 'dist_to_template_hull' not in df_ligs.columns:
                            dist_to_template_hull = np.array(template.alpha_hull.get_pnts_distance(df_ligs[['c_x', 'c_y', 'c_z']][df_filter].values))
                        else:
                            dist_to_template_hull = df_ligs['dist_to_template_hull'][df_filter].values
                        if lessthan:
                            gr_cst_filters.append(dist_to_template_hull < dist_buried)
                        else:
                            gr_cst_filters.append(dist_to_template_hull >= dist_buried)
                elif not np.isnan(row.atom_dist_resnum):
                    atom_dist_resnum = row.atom_dist_resnum
                    atom_dist_chain = row.atom_dist_chain
                    atom_dist_name = row.atom_dist_name
                    df_filter = (df_ligs.resname == lig_resname) & (df_ligs.name == lig_name)
                    df_template_atom_filter = (df_t.name == atom_dist_name) & (df_t.chain == atom_dist_chain) & (df_t.resnum == atom_dist_resnum)
                    df_lig_atom_coords = df_ligs[['c_x', 'c_y', 'c_z']][df_filter].values
                    df_template_atom_coords = df_t[['c_x', 'c_y', 'c_z']][df_template_atom_filter].values
                    dists = cdist(df_lig_atom_coords, df_template_atom_coords)
                    atom_dist_filters = []
                    if not np.isnan(row.atom_dist_lessthan):
                        atom_dist_filters.append(dists < row.atom_dist_lessthan)
                    if not np.isnan(row.atom_dist_greaterthan):
                        atom_dist_filters.append(dists > row.atom_dist_greaterthan)
                    gr_cst_filters.append(np.array(atom_dist_filters).all(axis=0))
             
            cst_filters.append(np.array(gr_cst_filters).any(axis=0))

        if len(cst_filters) == 0:
            return df_ligs

        cst_filter = np.array(cst_filters).all(axis=0)
        df_lig_ids = df_ligs[['CG', 'rota', 'probe_name']][df_filter] # df_filter should give all unique labels each time it is defined
        return merge(df_ligs, df_lig_ids[cst_filter], on=['CG', 'rota', 'probe_name'])

    # @staticmethod
    def _ligand_gauntlet(self, df, template, frac_non_hb_heavy_buried, hull_tolerance, max_count=0, vdW_tolerance=0.0):
        clash = Clash(df.copy(), template.dataframe, tol=vdW_tolerance)
        clash.set_grouping(['CG', 'rota', 'probe_name'])
        clash.find()

        df = merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(),
                   on=['CG', 'rota', 'probe_name'])
        # df = fast_merge(df, clash.dfq_clash_free[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)
        if len(df) == 0:
            print('                No ligands due to clashing')
            return df, max_count, 0

        if frac_non_hb_heavy_buried > 0:
            if hull_tolerance == 0:
                df['in_hull'] = template.alpha_hull.pnts_in_hull(df[['c_x', 'c_y', 'c_z']].values)
            else:
                df['dist_to_template_hull'] = template.alpha_hull.get_pnts_distance(df[['c_x', 'c_y', 'c_z']].values)
                df['in_hull'] = df['dist_to_template_hull'] >= 0 + hull_tolerance
            df_ = df[(df[hb_cols].isna().all(axis=1)) &
                    ~(df['atom_type_label'].isin(['h_pol', 'h_alkyl', 'h_aro']))].copy()
            if max_count == 0:
                max_count = len(df_) / len(df_[['CG', 'rota', 'probe_name']].drop_duplicates())
            min_heavy_atoms_buried = np.floor(frac_non_hb_heavy_buried * max_count)
            heavy_atoms_buried = df_.groupby(['CG', 'rota', 'probe_name'])['in_hull'].transform('sum')
            df_['heavy_atoms_buried'] = heavy_atoms_buried
            df_['frac_heavy_atoms_buried'] = heavy_atoms_buried / max_count
            df_ = df_[df_['heavy_atoms_buried'] >= min_heavy_atoms_buried]
            if len(df_) == 0:
                print('                No ligands due to burial')
                return df_, max_count, 0
            drop_dup_df_ = df_[['CG', 'rota', 'probe_name',
                                'heavy_atoms_buried',
                                'frac_heavy_atoms_buried']].drop_duplicates()
            num_ligs = len(drop_dup_df_)
            df = merge(df, drop_dup_df_, on=['CG', 'rota', 'probe_name'])
            len_df = len(df)

            df = self._remove_dist_buried(df, template)
            len_df_ = len(df)
            if len_df_ != len_df:
                num_ligs = len(df[['CG', 'rota', 'probe_name']].drop_duplicates())

            if hull_tolerance == 0:
                df['dist_to_template_hull'] = template.alpha_hull.get_pnts_distance(df[['c_x', 'c_y', 'c_z']].values)
            return df, max_count, num_ligs
        else:
            df = self._remove_dist_buried(df, template)
            df['frac_heavy_atoms_buried'] = np.nan
            df['heavy_atoms_buried'] = np.nan
            df['dist_to_template_hull'] = np.nan
            df['in_hull'] = True
            return df, max_count, len(df[['CG', 'rota', 'probe_name']].drop_duplicates())

    def _assign_spurious_lig_attr(self, dfs, template):
        for df in dfs:
            df = self._remove_dist_buried(df, template)
            df['frac_heavy_atoms_buried'] = np.nan
            df['heavy_atoms_buried'] = np.nan
            df['dist_to_template_hull'] = np.nan
            df['in_hull'] = True

    def _load_ligands_aa(self, aa, cg, cg_group, template,
                         use_ligs_of_loaded_vdms_only, frac_non_hb_heavy_buried,
                         hull_tolerance, vdW_tolerance=0.0):
        max_count = 0
        dataframes = []
        cg_gr = int(cg_group)
        print('        Loading ligands of', aa)
        if use_ligs_of_loaded_vdms_only:
            cols_dfx = ['resname_rota', 'CG', 'rota', 'probe_name', 'seg_chain_resnum']
            dfx = self.cg_dict[cg][cols_dfx].drop_duplicates()
            dfx = dfx[dfx['resname_rota'] == aa]
            if len(dfx) == 0:
                print('            No parent ligands')
                return DataFrame()
            df_parent = read_parquet(self.path_to_ligand_database[cg] + cg +
                                     '/' + cg_group + '/' + aa + '.parquet.gzip')
            df_parent = merge(df_parent,
                              dfx[['CG', 'rota', 'probe_name']].drop_duplicates(),
                              on=['CG', 'rota', 'probe_name'])
            # df_parent = fast_merge(df_parent, dfx[['CG', 'rota', 'probe_name']].drop_duplicates(), columns=None)

            if len(df_parent) == 0:
                print('            No parent ligands')
                return DataFrame()
        else:
            df_parent = read_parquet(self.path_to_ligand_database[cg] + cg +
                                     '/' + cg_group + '/' + aa + '.parquet.gzip')

        num_poss_ligands = len(df_parent[['CG', 'rota', 'probe_name']].drop_duplicates())
        for res in self.residue_dict[cg][aa]:
            print('            Loading', num_poss_ligands, 'ligands of residue', res.seg_chain_resnum)
            df = df_parent.copy()
            if use_ligs_of_loaded_vdms_only:
                filter = dfx['seg_chain_resnum'] == res.seg_chain_resnum
                df = merge(df, dfx[['CG', 'rota', 'probe_name']][filter].drop_duplicates(),
                           on=['CG', 'rota', 'probe_name'])
                # df = fast_merge(df, dfx[['CG', 'rota', 'probe_name']][filter].drop_duplicates(), columns=None)
                if len(df) == 0:
                    print('                No ligands due to no vdMs')
                    continue

            # Need to add code for when use_ligs_of_loaded_vdms_only=False.

            targ_coords = \
                merge(df_ideal_ala['name'],
                      template.dataframe[template.dataframe['seg_chain_resnum'] == res.seg_chain_resnum],
                      on='name')[['c_x', 'c_y', 'c_z']].values
            mob_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values
            df = superpose_df(mob_coords, targ_coords, df)

            df, max_count, num_ligs = self._ligand_gauntlet(df, template, frac_non_hb_heavy_buried,
                                                  hull_tolerance, max_count=max_count, vdW_tolerance=vdW_tolerance)

            df['seg_chain_resnum'] = [res.seg_chain_resnum] * len(df)
            df['CG_group'] = cg_gr
            df['CG_type'] = cg
            print('                Appending', num_ligs, 'ligands')
            if len(df) > 0:
                dataframes.append(df)
        if len(dataframes) > 0:
            return fast_concat(dataframes)
        else:
            return DataFrame()

    def _load_ligand_conformer(self, path_to_lig_pdb, path_to_lig_params, lig_resname,
                              remove_atom_from_hb_dict=None, ligand_dataframe=None):
        if ligand_dataframe is None:
            lig = parsePDB(path_to_lig_pdb)
            hb_dict = make_lig_hbond_dict(lig_resname, path_to_lig_params)
            if remove_atom_from_hb_dict is not None:
                [hb_dict[lig_resname].pop(atom_name) for atom_name in remove_atom_from_hb_dict]
            atom_type_dict = make_lig_atom_type_dict(lig_resname, path_to_lig_params)
            ligand_dataframe = make_df_from_prody(lig, can_hbond_dict=hb_dict, lig_atom_types_dict=atom_type_dict)
        return ligand_dataframe

    def load_ligand_conformer(self, path_to_lig_pdb, path_to_lig_params, lig_resname,
                               remove_atom_from_hb_dict=None, ligand_dataframe=None):
        ligand_dataframe = self._load_ligand_conformer(path_to_lig_pdb, path_to_lig_params, lig_resname,
                               remove_atom_from_hb_dict=remove_atom_from_hb_dict, ligand_dataframe=ligand_dataframe)
        self.ligand_conformers = [ligand_dataframe]

    def load_ligand_conformers(self, paths_to_lig_pdbs, path_to_lig_params, lig_resname,
                               remove_atom_from_hb_dict=None, ligand_dataframes=None):
        if ligand_dataframes is not None:
            if type(ligand_dataframes == list):
                self.ligand_conformers = ligand_dataframes
            else:
                raise Exception('ligand_dataframes must be a list of ligand dataframes.')

        else:
            ligand_dfs = []
            for path_to_lig_pdb in paths_to_lig_pdbs:
                ligand_dataframe = self._load_ligand_conformer(path_to_lig_pdb, path_to_lig_params, lig_resname,
                                                               remove_atom_from_hb_dict=remove_atom_from_hb_dict,
                                                               ligand_dataframe=None)
                ligand_dfs.append(ligand_dataframe)
            self.ligand_conformers = ligand_dfs

    def _superpose_ligands_to_CGs(self, df_lig_coords_and_index_and_conformer_num, ligand_conformer, template, num_rows,
                                  frac_non_hb_heavy_buried, hull_tolerance, vdW_tolerance=0.0):
        _df_lig_coords, k, z = df_lig_coords_and_index_and_conformer_num
        num_ligs = len(_df_lig_coords)
        cg_ids = np.array([[i] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        rota_ids = np.array([[k] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        probe_name_ids = np.array([[z] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        seg_chain_resnum_ids = np.array([[1] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        cg_type_ids = np.array([[1] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        cg_gr_ids = np.array([[1] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
        atom_type_labels = ligand_conformer['atom_type_label'].values
        atom_type_label_ids = np.array([[atom_type_labels] for i in range(num_ligs)], dtype=str).flatten()
        name_labels = ligand_conformer['name'].values
        name_label_ids = np.array([[name_labels] for i in range(num_ligs)], dtype=str).flatten()
        resname_labels = ligand_conformer['resname'].values
        resname_label_ids = np.array([[resname_labels] for i in range(num_ligs)], dtype=str).flatten()
        labels = np.array([cg_ids, rota_ids, probe_name_ids, seg_chain_resnum_ids, cg_type_ids, cg_gr_ids]).T
        cols = ['CG', 'rota', 'probe_name', 'seg_chain_resnum', 'CG_type', 'CG_group']
        cols.extend(coords_cols)
        df_lig_coords_concat = np.vstack(_df_lig_coords)
        vals = np.hstack((labels, df_lig_coords_concat))
        df = pd.DataFrame(vals, columns=cols)
        df['atom_type_label'] = atom_type_label_ids
        df['resname'] = resname_label_ids
        df['name'] = name_label_ids
        df['resnum'] = 10
        print('    ', num_ligs, 'ligands entering clash/burial filters...')
        result = self._ligand_gauntlet(df, template, frac_non_hb_heavy_buried=frac_non_hb_heavy_buried,
                                       hull_tolerance=hull_tolerance, vdW_tolerance=vdW_tolerance)
        if not result:
            print('        ', 'No', 'ligands passed clash/burial filters.')
            return None
        else:
            df, _, num_ligs = result
            print('        ', num_ligs, 'ligands passed clash/burial filters.')
            if len(df) > 0:
                return df

    def superpose_ligands_to_CGs(self, template, frac_non_hb_heavy_buried=0.4,
                                 hull_tolerance=0, chunk_size=10000, run_parallel=False,
                                 num_cpus=None, cgs_groups=dict(), vdW_tolerance=0.0):
        if self.ligand_conformers is None:
            raise Exception('Ligand conformer must be loaded first. See load_ligand_conformer().')

        if run_parallel:
            num_cpus = num_cpus or os.cpu_count() - 2

            with Pool(num_cpus) as p:
                if 'rmsd' in self.ligand_vdm_correspondence.columns:
                    lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
                else:
                    lig_vdm_corr = self.ligand_vdm_correspondence
                df_cgs = lig_vdm_corr[['resname', 'name', 'CG_type', 'CG_group']].drop_duplicates()
                all_dfs = []
                for z, ligand_conformer in enumerate(self.ligand_conformers):
                    df_lig_coords = []
                    lig_coords_cols_vals = ligand_conformer[coords_cols].values
                    num_rows = lig_coords_cols_vals.shape[0]
                    for (cg, gn), gr in lig_vdm_corr.groupby(['CG_type', 'CG_group']):
                        if cg not in self.cg_neighbors:
                            continue
                        if cgs_groups:
                            if cg not in cgs_groups:
                                continue
                            if gn not in cgs_groups[cg]:
                                continue
                        print('Superposing ligands of', cg, gn, 'for conformer', z+1, '...')
                        gr_ = pd.merge(df_cgs, gr, on=['resname', 'name', 'CG_type', 'CG_group'])
                        gr_.reset_index(inplace=True, drop=True)
                        gr_ = gr_[['lig_resname', 'lig_name']].drop_duplicates()
                        df_cg = pd.merge(gr_, ligand_conformer, left_on=['lig_resname', 'lig_name'],
                                         right_on=['resname', 'name'])
                        coords_lig_cg = df_cg[['c_x', 'c_y', 'c_z']].values
                        for i in range(self.cg_neighbors[cg]._fit_X.shape[0]):
                            coords_cg = self.cg_neighbors[cg]._fit_X[i].reshape(-1, 3)
                            R, mob_coords_com, targ_coords_com = get_rot_trans(mob_coords=coords_lig_cg,
                                                                               targ_coords=coords_cg, weights=None)
                            new_lig_coords_cols_vals = apply_transform_to_coords_cols(R, mob_coords_com,
                                                                                      targ_coords_com, lig_coords_cols_vals)
                            df_lig_coords.append(new_lig_coords_cols_vals.astype(np.float32))

                    if len(df_lig_coords) > 0:
                        f = partial(self._superpose_ligands_to_CGs, template=template,
                                    ligand_conformer=ligand_conformer, num_rows=num_rows,
                                    frac_non_hb_heavy_buried=frac_non_hb_heavy_buried,
                                    hull_tolerance=hull_tolerance)
                        iterator = ((coords, k, z) for k, coords in enumerate(chunks(df_lig_coords, chunk_size=chunk_size)))
                        dfs = p.map(f, iterator, chunksize=1)  # chuncksize=int(len(iterator) / num_cpus))
                        dfs = [df_ for df_ in dfs if df_ is not None]
                        all_dfs.extend(dfs)
        else:
            all_dfs = []
            if 'rmsd' in self.ligand_vdm_correspondence.columns:
                lig_vdm_corr = self.ligand_vdm_correspondence[self.ligand_vdm_correspondence['rmsd'] == True]
            else:
                lig_vdm_corr = self.ligand_vdm_correspondence
            df_cgs = lig_vdm_corr[['resname', 'name', 'CG_type', 'CG_group']].drop_duplicates()
            for z, ligand_conformer in enumerate(self.ligand_conformers):
                for k, ((cg, gn), gr) in enumerate(lig_vdm_corr.groupby(['CG_type', 'CG_group'])):
                    if cg not in self.cg_dict:
                        continue
                    if cgs_groups:
                        if cg not in cgs_groups:
                            continue
                        if gn not in cgs_groups[cg]:
                            continue
                    print('Superposing ligands of', cg, gn, 'for conformer', z+1, '...')
                    gr_ = pd.merge(df_cgs, gr, on=['resname', 'name', 'CG_type', 'CG_group'])
                    gr_.reset_index(inplace=True, drop=True)
                    gr_ = gr_[['lig_resname', 'lig_name']].drop_duplicates()
                    df_cg = pd.merge(gr_, ligand_conformer, left_on=['lig_resname', 'lig_name'],
                                     right_on=['resname', 'name'])
                    coords_lig_cg = df_cg[['c_x', 'c_y', 'c_z']].values
                    lig_coords_cols_vals = ligand_conformer[coords_cols].values
                    df_lig_coords = []
                    for i in range(self.cg_neighbors[cg]._fit_X.shape[0]):
                        coords_cg = self.cg_neighbors[cg]._fit_X[i].reshape(-1, 3)
                        R, mob_coords_com, targ_coords_com = get_rot_trans(mob_coords=coords_lig_cg,
                                                                           targ_coords=coords_cg, weights=None)
                        new_lig_coords_cols_vals = apply_transform_to_coords_cols(R, mob_coords_com,
                                                                                  targ_coords_com,lig_coords_cols_vals)
                        df_lig_coords.append(new_lig_coords_cols_vals.astype(np.float32))
                    num_rows = lig_coords_cols_vals.shape[0]
                    for j, _df_lig_coords in enumerate(chunks(df_lig_coords, chunk_size=chunk_size)):
                        num_ligs = len(_df_lig_coords)
                        cg_ids = np.array([[i] * num_rows for i in range(num_ligs)], dtype=np.float32).flatten()
                        rota_ids = np.array([[k] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                        probe_name_ids = np.array([[j] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                        seg_chain_resnum_ids = np.array([[z] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                        cg_type_ids = np.array([[1] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                        cg_gr_ids = np.array([[1] * num_rows for _ in range(num_ligs)], dtype=np.float32).flatten()
                        atom_type_labels = ligand_conformer['atom_type_label'].values
                        atom_type_label_ids = np.array([[atom_type_labels] for _ in range(num_ligs)], dtype=str).flatten()
                        name_labels = ligand_conformer['name'].values
                        name_label_ids = np.array([[name_labels] for _ in range(num_ligs)], dtype=str).flatten()
                        resname_labels = ligand_conformer['resname'].values
                        resname_label_ids = np.array([[resname_labels] for _ in range(num_ligs)], dtype=str).flatten()
                        labels = np.array([cg_ids, rota_ids, probe_name_ids, seg_chain_resnum_ids, cg_type_ids, cg_gr_ids]).T
                        cols = ['CG', 'rota', 'probe_name', 'seg_chain_resnum', 'CG_type', 'CG_group']
                        cols.extend(coords_cols)
                        df_lig_coords_concat = np.vstack(_df_lig_coords)
                        vals = np.hstack((labels, df_lig_coords_concat))
                        df = pd.DataFrame(vals, columns=cols)
                        df['atom_type_label'] = atom_type_label_ids
                        df['resname'] = resname_label_ids
                        df['name'] = name_label_ids
                        df['resnum'] = 10
                        print('    ', num_ligs, 'ligands entering clash/burial filters...')
                        result = self._ligand_gauntlet(df, template, frac_non_hb_heavy_buried=frac_non_hb_heavy_buried,
                                                       hull_tolerance=hull_tolerance, vdW_tolerance=vdW_tolerance)
                        if not result:
                            print('        ', 'No', 'ligands passed clash/burial filters.')
                        else:
                            df, _, num_ligs = result
                            print('        ', num_ligs, 'ligands passed clash/burial filters.')
                            if len(df) > 0:
                                all_dfs.append(df)

        if len(all_dfs) == 0:
            raise Exception('No ligands were successfully loaded.')
        self.ligand_dataframe = fast_concat(all_dfs)
        # self.ligand_dataframe['lig_resname'] = self.ligand_dataframe['resname']
        # self.ligand_dataframe['lig_name'] = self.ligand_dataframe['name']
        self.ligand_dataframe_grs = self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
                                                                   'seg_chain_resnum', 'CG_type', 'CG_group'])
        print('Loaded ', len(self.ligand_dataframe_grs), 'ligands')

    def load_ligands(self, template, use_ligs_of_loaded_vdms_only=True,
                     frac_non_hb_heavy_buried=0.4,
                     path_to_ligand_database=None, hull_tolerance=0,
                     run_parallel=False, num_cpus=None, exclude_cgs=()):
        if path_to_ligand_database is not None:
            self.path_to_ligand_database = path_to_ligand_database
        self.ligand_dataframe = []
        #output self.ligand_dict[cg_name][cg_group] = dataframe_of_ligands
        #above structure doesn't really matter.
        # can just do self.df_ligands = dataframe_of_ligands
        # actually, ligands from different groups might have same identifier (cg,rota, probe_name)
        # so the first option is prob the best.
        #cg_name -> cg_group -> abple -> resname

        num_cpus = num_cpus or os.cpu_count() - 2

        if type(self.path_to_ligand_database) == str:
            path_lig_db_dict = dict()
            for cg in self.residue_dict.keys():
                path_lig_db_dict[cg] = self.path_to_ligand_database
            self.path_to_ligand_database = path_lig_db_dict
        for key, val in self.path_to_ligand_database.items():
            if val[-1] != '/':
                val += '/'
                self.path_to_ligand_database[key] = val

        if run_parallel:
            ligand_dataframes = []
            with Pool(num_cpus) as p:
                for cg in self.residue_dict.keys():
                    if cg in exclude_cgs:
                        continue
                    print('Loading ligands of', cg)
                    for cg_group in [d for d in os.listdir(self.path_to_ligand_database[cg] + cg + '/') if d[0] != '.']:
                        print('    cg_group', cg_group)
                        available_aas = [f.split('.')[0] for f in
                                         os.listdir(self.path_to_ligand_database[cg] + cg + '/' + cg_group)
                                         if f[0] != '.']
                        f = partial(self._load_ligands_aa, cg=cg, cg_group=cg_group, template=template,
                                 use_ligs_of_loaded_vdms_only=use_ligs_of_loaded_vdms_only,
                                    frac_non_hb_heavy_buried=frac_non_hb_heavy_buried,
                                 hull_tolerance=hull_tolerance)
                        iterator = [aa for aa in self.residue_dict[cg].keys() if aa in available_aas]
                        if len(iterator) == 0:
                            print('        No ligands to load in the Ligand Database.')
                        # num_cpus = min(num_cpus, len(iterator))
                        dfs = p.map(f, iterator, chunksize=1)  # chuncksize=int(len(iterator) / num_cpus))
                        dfs = [df_ for df_ in dfs if len(df_) > 0]
                        if len(dfs) > 0:
                            ligand_dataframes.append(fast_concat(dfs))
            if len(ligand_dataframes) == 0:
                raise Exception('No ligands were successfully loaded. Stopping the run.')
            self.ligand_dataframe = fast_concat(ligand_dataframes)
        else:
            for cg in self.residue_dict.keys():
                if cg in exclude_cgs:
                    continue
                print('Loading ligands of', cg)
                for cg_group in [d for d in os.listdir(self.path_to_ligand_database[cg] + cg + '/') if d[0] != '.']:
                    print('    cg_group', cg_group)
                    available_aas = [f.split('.')[0] for f in
                                     os.listdir(self.path_to_ligand_database[cg] + cg + '/' + cg_group)
                                     if f[0] != '.']
                    iterator = [aa for aa in self.residue_dict[cg].keys() if aa in available_aas]
                    if len(iterator) == 0:
                        print('        No ligands to load in the Ligand Database.')
                    for aa in iterator:
                        self.ligand_dataframe.append(self._load_ligands_aa(aa, cg, cg_group, template,
                                                                           use_ligs_of_loaded_vdms_only,
                                                                           frac_non_hb_heavy_buried,
                                                                           hull_tolerance))
            dfs = [df_ for df_ in self.ligand_dataframe if len(df_) > 0]
            if len(dfs) > 0:
                self.ligand_dataframe = fast_concat(dfs)
            else:
                raise Exception('No ligands were successfully loaded. Stopping the run.')
        # if len(self.ligand_dataframe) == 0:
        #     raise Exception('No ligands passed any filters.')
        # self.ligand_dataframe.reset_index(drop=True, inplace=True)
        self.ligand_dataframe_grs = self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
                                                                   'seg_chain_resnum', 'CG_type', 'CG_group'])
        print('Loaded ', len(self.ligand_dataframe_grs), 'ligands')
        # self.ligand_dataframe_grs = Group(dict(tuple(self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
        #                                                                             'seg_chain_resnum', 'CG_type', 'CG_group']))))
        # print('Loaded ', len(self.ligand_dataframe_grs.dict.keys()), 'ligands')

    def set_ligand_rep(self, force_set=False):
        if (self.ligand_rep is None and self.ligand_dataframe_grs is not None) or force_set:
            self.ligand_rep = self.ligand_dataframe_grs.get_group(list(self.ligand_dataframe_grs.groups.keys())[0])

    def load_poses(self, poses):
        self._poses = poses
        self.ligand_dataframe = fast_concat([pose.ligand for pose in poses])
        self.ligand_dataframe_grs = self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
                                                                   'seg_chain_resnum', 'CG_type', 'CG_group'])

    def load_static_ligands(self, ligand_dataframes, template):
        df0 = ligand_dataframes[0]
        if 'CG' not in df0.columns:
            for i, df in enumerate(ligand_dataframes):
                df.loc[:, 'CG'] = i
                df.loc[:, 'rota'] = i
                df.loc[:, 'probe_name'] = i
                df.loc[:, 'seg_chain_resnum'] = i
                df.loc[:, 'CG_type'] = i
                df.loc[:, 'CG_group'] = i

        self._assign_spurious_lig_attr(ligand_dataframes, template)

        if len(ligand_dataframes) == 1:
            self.ligand_dataframe = ligand_dataframes[0]
        else:
            self.ligand_dataframe = fast_concat(ligand_dataframes)

        self.ligand_dataframe_grs = self.ligand_dataframe.groupby(['CG', 'rota', 'probe_name',
                                                                   'seg_chain_resnum', 'CG_type', 'CG_group'])

    def set_constraints(self, path_to_cst_file):
        groups = list()
        resnames = list()
        names = list()
        cg_types = list()
        cg_groups = list()
        contacts = list()
        dists_buried = list()
        dists_lessthan = list()
        hasvdms = list()
        hasvdms_inpose = list()
        atom_dists_lessthan = list()
        atom_dists_greaterthan = list()
        atom_dist_resnums = list()
        atom_dist_chains = list() 
        atom_dist_names = list()
        with open(path_to_cst_file, 'r') as infile:
            for line in infile:
                if line.startswith('#'):
                    continue
                if '#' in line:
                    comment_index = line.index('#')
                    line = line[:comment_index]
                spl = line.split()
                if len(spl) < 1:
                    continue
                group = int(spl[0].strip())
                resname = spl[1].strip()
                name = spl[2].strip()
                #cg_type = spl[3].strip()
                #cg_grs = []
                if 'AD' in spl:
                    ad_ind = spl.index('AD')
                    atom_dist_resnums.append(int(spl[ad_ind + 1].strip()))
                    atom_dist_chains.append(spl[ad_ind + 2].strip())
                    atom_dist_names.append(spl[ad_ind + 3].strip())
                    if '<' in line:
                        for s in spl[ad_ind + 1:]:
                            if '>' in s:
                                atom_dists_greaterthan.append(float(s.split('>')[1]))
                    else:
                        atom_dists_greaterthan.append(np.nan)
                    if '>' in line:
                        for s in spl[ad_ind + 1:]:
                            if '<' in s:
                                atom_dists_lessthan.append(float(s.split('<')[1]))
                    else:
                        atom_dists_lessthan.append(np.nan)
                else:
                    atom_dist_resnums.append(np.nan)
                    atom_dist_chains.append(None)
                    atom_dist_names.append(None)
                    atom_dists_greaterthan.append(np.nan)
                    atom_dists_lessthan.append(np.nan)

                if 'CGt' in spl and 'CGgr' in spl:
                    cgt_ind = spl.index('CGt')
                    cggr_ind = spl.index('CGgr')
                    cg_type = spl[cgt_ind + 1]
                    cg_grs = [int(c) for c in spl[cggr_ind + 1].split(',')]
                else:
                    cg_type = None
                    cg_grs = []
                try:
                    CO_ind = spl.index('CO')
                except ValueError:
                    CO_ind = None
                if 'HVM' in spl:
                    hasvdm = True
                else:
                    hasvdm = False

                try:
                    hasvdmp_ind = spl.index('HVMp')
                except:
                    hasvdmp_ind = None

                if hasvdmp_ind:
                    try:
                        hasvdmp = int(spl[hasvdmp_ind + 1])
                    except:
                        raise Exception('Has_vdM_in_pose not set correctly in cst file.')
                else:
                    hasvdmp = np.nan

                try:
                    DB_ind = spl.index('DB')
                except ValueError:
                    DB_ind = None

                if DB_ind:
                    dist = spl[DB_ind + 1]
                    if dist[0] == '<':
                        dist_lessthan = True
                    elif dist[0] == '>':
                        dist_lessthan = False
                    else:
                        raise ValueError('distance buried must be "<" or ">" a number, e.g. <0.5')
                    dist = float(dist[1:])
                else:
                    dist = np.nan
                    dist_lessthan = None

                CO_set = set()
                if CO_ind:
                    CO_set = set(spl[CO_ind + 1].split(','))

                groups.append(group)
                resnames.append(resname)
                names.append(name)
                cg_types.append(cg_type)
                cg_groups.append(cg_grs)
                contacts.append(CO_set)
                dists_buried.append(dist)
                dists_lessthan.append(dist_lessthan)
                hasvdms.append(hasvdm)
                hasvdms_inpose.append(hasvdmp)
        data = dict(cst_group=groups, lig_resname=resnames, lig_name=names, CG_type=cg_types,
                    CG_groups=cg_groups,
                    contact_type=contacts, Has_vdM=hasvdms, Has_vdM_in_pose=hasvdms_inpose, 
                    dist_buried=dists_buried,
                    dist_lessthan=dists_lessthan,
                    atom_dist_resnum=atom_dist_resnums,
                    atom_dist_chain=atom_dist_chains,
                    atom_dist_name=atom_dist_names,
                    atom_dist_lessthan=atom_dists_lessthan,
                    atom_dist_greaterthan=atom_dists_greaterthan)
        self.constraints = DataFrame(data)
        self.constraints_gr = self.constraints.groupby('cst_group')

    def _get_HasVdm_cst_filter(self):
        csts_has_vdm_grs = self.constraints[self.constraints['Has_vdM']].groupby('cst_group')
        wh_ = []
        for n, gr in csts_has_vdm_grs:
            if len(gr) > 1 and not gr.Has_vdM.all():
                continue
            for m, row in gr.iterrows():
                cg = row.CG_type
                wh = []
                for cg_group in row.CG_groups:
                    wh.append(np.array(list(map(np.any, self.ligand_neighbors_indices[cg][cg_group]))))
                wh = np.array(wh).any(axis=0)
                wh_.append(wh)
        if len(wh_) == 0:
            return None
        cst_filter = np.array(wh_).all(axis=0)
        return cst_filter

    def _get_dist_buried_cst_filter(self):
        # faster way would be to iterate thru ligands and apply all burial filters to each ligand,
        # rather than vice versa. Will adjust later.
        csts = self.constraints[(~self.constraints['dist_buried'].isna())
                                & (self.constraints['contact_type'] == set())
                                & (self.constraints['Has_vdM_in_pose'] == False)]
        csts_grs = csts.groupby('cst_group')
        cst_filters = []
        for n, gr in csts_grs:
            if len(gr) > 1 and not (~gr.dist_buried.isna()).all():
                continue
            gr_cst_filters = []
            for m, row in gr.iterrows():
                lig_resname = row.lig_resname
                lig_name = row.lig_name
                dist_buried = row.dist_buried
                lessthan = row.dist_lessthan
                cst_filter = []
                for lig_gr_name in self.ligand_dataframe_neighbors.values:
                    lig = self.ligand_dataframe_grs.get_group(tuple(lig_gr_name))
                    dist_to_hull = lig['dist_to_template_hull'][(lig['resname'] == lig_resname)
                                                       & (lig['name'] == lig_name)].iat[0]
                    if lessthan:
                        if dist_to_hull < dist_buried:
                            cst_filter.append(True)
                        else:
                            cst_filter.append(False)
                    else:
                        if dist_to_hull > dist_buried:
                            cst_filter.append(True)
                        else:
                            cst_filter.append(False)
                gr_cst_filters.append(cst_filter)
            cst_filters.append(np.array(gr_cst_filters).any(axis=0))

        if len(cst_filters) == 0:
            return None
        cst_filter = np.array(cst_filters).all(axis=0)
        return cst_filter

    def set_top_designable_percent(self, only_top_percent=0.1, min_poses=50, max_poses=None, 
                                    weight_dict=None, tamp_by_distance=True, exponential=False,
                                          log_logistic=False, gaussian=True, relu=False):

        if tamp_by_distance:
            if exponential:
                tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
            elif log_logistic:
                # middle ground between exponential and gaussian
                tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
            elif gaussian:
                tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
            elif relu:
                tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        else:
            tamp_func = lambda x: 1

        if weight_dict is None:
            weight_dict = dict()

        cg_cg_gr_dict = defaultdict(set)
        for cg, cg_gr in self.ligand_vdm_correspondence_grs.groups.keys():
            cg_cg_gr_dict[cg].add(cg_gr)

        all_nbr_inds = dict()
        all_nbr_dists = dict()
        for cg in cg_cg_gr_dict.keys():
            if cg not in self.cg_dict:
                continue

            if cg not in self.ligand_neighbors_indices:
                continue

            if cg not in weight_dict:
                weight_dict[cg] = 1

            nbr_inds = defaultdict(list)
            nbr_dists = defaultdict(list)
            for cg_group in cg_cg_gr_dict[cg]:
                if cg_group not in self.ligand_neighbors_indices[cg]:
                    continue
                if self.filtered_lig_indices is not None:
                    ligand_neighbors_indices = self.ligand_neighbors_indices[cg][cg_group][self.filtered_lig_indices]
                    ligand_neighbors_dists = self.ligand_neighbors_dists[cg][cg_group][self.filtered_lig_indices]
                else:
                    ligand_neighbors_indices = self.ligand_neighbors_indices[cg][cg_group]
                    ligand_neighbors_dists = self.ligand_neighbors_dists[cg][cg_group]
                len_ligand_neighbors_indices = len(ligand_neighbors_indices)
                for i, (neighbors, _dists) in enumerate(zip(ligand_neighbors_indices, ligand_neighbors_dists)):
                    if len(neighbors) == 0:
                        continue
                    nbr_inds[i].extend(list(neighbors))
                    nbr_dists[i].extend(list(_dists))

            all_nbr_inds[cg] = nbr_inds
            all_nbr_dists[cg] = nbr_dists

        lig_scores = np.zeros(len_ligand_neighbors_indices)
        for i in range(len_ligand_neighbors_indices):
            score_scr_dict = defaultdict(dict)
            score = 0
            for cg in all_nbr_inds.keys():
                if i not in all_nbr_inds[cg]:
                    continue
                nbr_inds = list(all_nbr_inds[cg][i])
                nbr_dists = np.array(list(all_nbr_dists[cg][i])).reshape(-1,1)
                if len(nbr_inds) == 0:
                    continue
                scr_cscores = self.cg_seg_chain_resnum_scores[cg][nbr_inds]
                scr_cscores = np.hstack((scr_cscores, nbr_dists))
                for scr in np.unique(scr_cscores[:, 0]):
                    scr_ = np.zeros(1, dtype=object)
                    scr_[0] = scr
                    _cscores = scr_cscores[scr_cscores[:, 0] == scr_, 1:]
                    _cscores = [sc * tamp_func(dist) for sc, dist in _cscores]
                    # _cscores = [sc * tamp_func(dist) for sc, dist in zip(_cscores, nbr_dists)]
                    score_scr_dict[scr][cg] = weight_dict[cg] * np.max(_cscores)
                    # score_scr_dict[scr][cg] = weight_dict[cg] * len(_cscores)
                    # score_scr_dict[scr][cg] = weight_dict[cg] * np.mean(_cscores) * np.log(len(_cscores))
            for scr in score_scr_dict.keys():
                score += max(score_scr_dict[scr].values())
                # score += np.log(sum(score_scr_dict[scr].values()))
            # max_scores = []
            # for scr in score_scr_dict.keys():
            #     max_score = max(score_scr_dict[scr].values())
            #     score += max_score
            #     max_scores.append(max_score)
            # if len(max_scores) > 0:
            #     score += max(max_scores)
            lig_scores[i] = score
  
        sorted_lig_indices = lig_scores.argsort()[::-1]
        if max_poses is not None:
            if max_poses > min_poses:
                last_index = min(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), max_poses)
                if last_index < min_poses:
                    last_index = min_poses
            else:
                last_index = max(min(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), max_poses), min_poses)
                if last_index > max_poses:
                    last_index = max_poses
        else:
            last_index = max(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), min_poses)
        top_percent = sorted_lig_indices[:last_index]
        # filter = np.in1d(np.arange(len(sorted_lig_indices)), top_percent)
        # self._apply_cst_filter(filter)
        if self.filtered_lig_indices is not None:
            self.leftover_filtered_lig_indices =self.filtered_lig_indices[sorted_lig_indices[last_index:]]
            self.filtered_lig_indices = self.filtered_lig_indices[top_percent]
        else:
            self.filtered_lig_indices = top_percent
            self.leftover_filtered_lig_indices = sorted_lig_indices[last_index:]

    # def set_top_designable_percent(self, only_top_percent=0.1, min_poses=50, max_poses=None, weight_dict=None):
    #     sum_info = []
    #     weights = []
    #     for cov_name, cov_gr in self.ligand_vdm_correspondence.groupby('CG_ligand_coverage'):
    #         arr = []
    #         for cg, cg_group in cov_gr[['CG_type', 'CG_group']].drop_duplicates().values:
    #             if cg not in self.cg_dict:
    #                 continue
    #             if self.filtered_lig_indices is not None:
    #                 ligand_neighbors_indices = self.ligand_neighbors_indices[cg][cg_group][self.filtered_lig_indices]
    #             else:
    #                 ligand_neighbors_indices = self.ligand_neighbors_indices[cg][cg_group]
    #             arr.append(np.array(list(map(len, ligand_neighbors_indices))))
    #             # arr.append(np.array(list(map(len, self.ligand_neighbors_indices[cg][cg_group]))))
    #         if len(arr) == 0:
    #             continue
    #         if weight_dict is not None:
    #             if cov_name not in weight_dict:
    #                 weights.append(1)
    #             else:
    #                 weights.append(weight_dict[cov_name])
    #         else:
    #             weights.append(1)
    #         sum_info.append(np.array(arr).sum(0) + 1)  # account for log(0)
    #     arr_sum_info = np.array(sum_info).T
    #     # sorted_lig_indices = np.log(arr_sum_info).sum(1).argsort()[::-1]
    #     sorted_lig_indices = np.dot(np.log(arr_sum_info), weights).argsort()[::-1]
    #     if max_poses is not None:
    #         if max_poses > min_poses:
    #             last_index = min(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), max_poses)
    #             if last_index < min_poses:
    #                 last_index = min_poses
    #         else:
    #             last_index = max(min(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), max_poses), min_poses)
    #             if last_index > max_poses:
    #                 last_index = max_poses
    #     else:
    #         last_index = max(int(np.ceil(len(sorted_lig_indices) * only_top_percent)), min_poses)
    #     top_percent = sorted_lig_indices[:last_index]
    #     # filter = np.in1d(np.arange(len(sorted_lig_indices)), top_percent)
    #     # self._apply_cst_filter(filter)
    #     if self.filtered_lig_indices is not None:
    #         self.leftover_filtered_lig_indices =self.filtered_lig_indices[sorted_lig_indices[last_index:]]
    #         self.filtered_lig_indices = self.filtered_lig_indices[top_percent]
    #     else:
    #         self.filtered_lig_indices = top_percent
    #         self.leftover_filtered_lig_indices = sorted_lig_indices[last_index:]

    def _apply_cst_filter(self, cst_filter):
        if self._ligand_neighbors_indices is None:
            self._ligand_neighbors_indices = self.ligand_neighbors_indices.copy()
        if self._ligand_neighbors_dists is None:
            self._ligand_neighbors_dists = self.ligand_neighbors_dists.copy()

        for key1 in self.ligand_neighbors_indices.keys():
            for key2 in self.ligand_neighbors_indices[key1].keys():
                self.ligand_neighbors_indices[key1][key2] = \
                    self.ligand_neighbors_indices[key1][key2][cst_filter]
                self.ligand_neighbors_dists[key1][key2] = \
                    self.ligand_neighbors_dists[key1][key2][cst_filter]

        if self._ligand_dataframe_neighbors is None:
            self._ligand_dataframe_neighbors = self.ligand_dataframe_neighbors.copy()
        self.ligand_dataframe_neighbors = self.ligand_dataframe_neighbors[cst_filter]

    def check_csts(self, pose, lig_index, template):
        if not self.constraints_gr:
            return True
        for n, cst_gr in self.constraints_gr:
            cst_gr_test = False
            for i, cst in cst_gr.iterrows():
                if cst['contact_type']:
                    resname = pose.ligand_contacts['resname_t'] == cst['lig_resname']
                    name = pose.ligand_contacts['name_t'] == cst['lig_name']
                    lig_atom = pose.ligand_contacts[resname & name]
                    if any(lig_atom['contact_type'].isin(cst['contact_type'])):
                        cst_gr_test = True
                        break
                if notnull(cst['Has_vdM_in_pose']):
                    if (pose.vdms['CG_ligand_coverage'] == int(cst['Has_vdM_in_pose'])).any():
                        cst_gr_test = True
                        break
                if notnull(cst['dist_buried']):
                    resname = pose.ligand['resname'] == cst['lig_resname']
                    name = pose.ligand['name'] == cst['lig_name']
                    lig_dist_buried = pose.ligand['dist_to_template_hull'][resname & name].iat[0]
                    if cst['dist_lessthan']:
                        if lig_dist_buried < cst['dist_buried']:
                            cst_gr_test = True
                            break
                    else:
                        if lig_dist_buried > cst['dist_buried']:
                            cst_gr_test = True
                            break
                if cst['Has_vdM']:
                    cg = cst.CG_type
                    wh = []
                    for cg_group in cst.CG_groups:
                        wh.append(np.any(self.ligand_neighbors_indices[cg][cg_group][lig_index]))
                    test = np.array(wh).any()
                    if test:
                        cst_gr_test = True
                        break
                if notnull(cst['atom_dist_resnum']):
                    df_t = template.dataframe
                    resname = pose.ligand['resname'] == cst['lig_resname']
                    name = pose.ligand['name'] == cst['lig_name']
                    lig_atom = pose.ligand[resname & name]
                    atom_dist_resnum = cst.atom_dist_resnum
                    atom_dist_chain = cst.atom_dist_chain
                    atom_dist_name = cst.atom_dist_name
                    df_template_atom_filter = (df_t.name == atom_dist_name) & (df_t.chain == atom_dist_chain) & (df_t.resnum == atom_dist_resnum)
                    df_lig_atom_coords = lig_atom[['c_x', 'c_y', 'c_z']].values
                    df_template_atom_coords = df_t[['c_x', 'c_y', 'c_z']][df_template_atom_filter].values
                    dists = cdist(df_lig_atom_coords, df_template_atom_coords)
                    atom_dist_filters = []
                    if not np.isnan(cst.atom_dist_lessthan):
                        atom_dist_filters.append(dists < cst.atom_dist_lessthan)
                    if not np.isnan(cst.atom_dist_greaterthan):
                        atom_dist_filters.append(dists > cst.atom_dist_greaterthan)
                    if np.array(atom_dist_filters).all():
                        cst_gr_test = True

            if not cst_gr_test:  # If any group cst fails, the function returns False
                return False
        return True

    def save(self, outpath='./', filename='sample.pkl', minimal_info_and_poses=False, minimal_info=False):
        if outpath[-1] != '/':
            outpath += '/'
        try:
            os.makedirs(outpath)
        except:
            pass

        if minimal_info_and_poses:
            minimal_info = True

        if minimal_info:
            sample = self.__copy__(no_vdms=True, minimal_info=minimal_info,
                                   include_poses_in_minimal_info=minimal_info_and_poses)
            with open(outpath + filename, 'wb') as outfile:
                dump(sample, outfile)
        else:
            with open(outpath + filename, 'wb') as outfile:
                dump(self, outfile)

    def cluster_ligands(self, rmsd_cutoff=0.5):
        num_ligs = len(self.ligand_dataframe_neighbors)
        ligs = merge(self.ligand_dataframe, self.ligand_dataframe_neighbors,
                     on=self.ligand_dataframe_neighbors.columns.to_list())
        example_lig = self.ligand_dataframe_grs.get_group(list(self.ligand_dataframe_grs.groups.keys())[0])
        num_atoms_ligand = len(example_lig[~example_lig['atom_type_label'].isin(['h_pol', 'h_alkyl', 'h_aro'])])
        ligs = ligs[~ligs['atom_type_label'].isin(['h_pol', 'h_alkyl', 'h_aro'])]
        lig_coords = ligs[['c_x', 'c_y', 'c_z']].values.reshape(num_ligs, 3*num_atoms_ligand)
        clu = Cluster(rmsd_cutoff=rmsd_cutoff)
        nbrs = NearestNeighbors(radius=np.sqrt(num_atoms_ligand)*rmsd_cutoff).fit(lig_coords)
        adj_mat = nbrs.radius_neighbors_graph(lig_coords)
        clu.adj_mat = adj_mat
        clu.fast_cluster()
        return clu

    # def filter_ligands_by_cluster_members(self, rmsd_cutoff=0.5, min_ligands_per_cluster=1,
    #                                       max_ligands_per_cluster=None,
    #                                       top_percent_per_cluster=None, weight_dict=None):
    #     clu = self.cluster_ligands(rmsd_cutoff=rmsd_cutoff)
    #     lig_indices = []
    #     for mems in clu.mems:
    #         sum_info = []
    #         weights = []
    #         for cov_name, cov_gr in self.ligand_vdm_correspondence.groupby('CG_ligand_coverage'):
    #             arr = []
    #             for cg, cg_group in cov_gr[['CG_type', 'CG_group']].drop_duplicates().values:
    #                 if cg not in self.cg_dict:
    #                     continue
    #                 arr.append(np.array(list(map(len, self.ligand_neighbors_indices[cg][cg_group][mems]))))
    #             if len(arr) == 0:
    #                 continue
    #             if weight_dict is not None:
    #                 if cov_name not in weight_dict:
    #                     weights.append(1)
    #                 else:
    #                     weights.append(weight_dict[cov_name])
    #             else:
    #                 weights.append(1)
    #             sum_info.append(np.array(arr).sum(0) + 1)  # account for log(0)
    #         arr_sum_info = np.array(sum_info).T
    #         # sorted_lig_indices = np.log(arr_sum_info).sum(1).argsort()[::-1]
    #         sorted_lig_indices = np.dot(np.log(arr_sum_info), weights).argsort()[::-1]
    #         if top_percent_per_cluster is not None:
    #             last_index = max(int(np.ceil(len(sorted_lig_indices) * top_percent_per_cluster)), min_ligands_per_cluster)
    #         else:
    #             last_index = 1
    #         if max_ligands_per_cluster is not None:
    #             last_index = min(last_index, max_ligands_per_cluster)
    #         top_percent = sorted_lig_indices[:last_index]
    #         lig_indices.extend(list(mems[top_percent]))
    #     # cst_filter = np.in1d(np.arange(len(self.ligand_dataframe_neighbors)), lig_indices)
    #     # self._apply_cst_filter(cst_filter)
    #     self.filtered_lig_indices = np.unique(lig_indices)

    def filter_ligands_by_cluster_members(self, rmsd_cutoff=0.5, min_ligands_per_cluster=1,
                                          max_ligands_per_cluster=None,
                                          top_percent_per_cluster=None, weight_dict=None,
                                          tamp_by_distance=True, exponential=False,
                                          log_logistic=False, gaussian=True, relu=False):

        if tamp_by_distance:
            if exponential:
                tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
            elif log_logistic:
                # middle ground between exponential and gaussian
                tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
            elif gaussian:
                tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
            elif relu:
                tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        else:
            tamp_func = lambda x: 1
    
        if weight_dict is None:
            weight_dict = dict()

        cg_cg_gr_dict = defaultdict(set)
        for cg, cg_gr in self.ligand_vdm_correspondence_grs.groups.keys():
            cg_cg_gr_dict[cg].add(cg_gr)
        
        clu = self.cluster_ligands(rmsd_cutoff=rmsd_cutoff)
        lig_indices = []
        for mems in clu.mems:
            if len(mems) <= min_ligands_per_cluster:
                lig_indices.extend(list(mems))
                continue

            all_nbr_inds = dict()
            all_nbr_dists = dict()
            for cg in cg_cg_gr_dict.keys():
                if cg not in self.cg_dict:
                    continue

                if cg not in self.ligand_neighbors_indices:
                    continue

                if cg not in weight_dict:
                    weight_dict[cg] = 1

                nbr_inds = defaultdict(list)
                nbr_dists = defaultdict(list)
                for cg_group in cg_cg_gr_dict[cg]:
                    if cg_group not in self.ligand_neighbors_indices[cg]:
                        continue
                    ligand_neighbors_indices = self.ligand_neighbors_indices[cg][cg_group][mems]
                    ligand_neighbors_dists = self.ligand_neighbors_dists[cg][cg_group][mems]
                    len_ligand_neighbors_indices = len(ligand_neighbors_indices)
                    for i, (neighbors, _dists) in enumerate(zip(ligand_neighbors_indices, ligand_neighbors_dists)):
                        if len(neighbors) == 0:
                            continue
                        nbr_inds[i].extend(list(neighbors))
                        nbr_dists[i].extend(list(_dists))

                all_nbr_inds[cg] = nbr_inds
                all_nbr_dists[cg] = nbr_dists

            lig_scores = np.zeros(len_ligand_neighbors_indices)
            for i in range(len_ligand_neighbors_indices):
                score_scr_dict = defaultdict(dict)
                score = 0
                for cg in all_nbr_inds.keys():
                    if i not in all_nbr_inds[cg]:
                        continue
                    nbr_inds = list(all_nbr_inds[cg][i])
                    nbr_dists = np.array(list(all_nbr_dists[cg][i])).reshape(-1,1)
                    if len(nbr_inds) == 0:
                        continue
                    scr_cscores = self.cg_seg_chain_resnum_scores[cg][nbr_inds]
                    scr_cscores = np.hstack((scr_cscores, nbr_dists))
                    for scr in np.unique(scr_cscores[:, 0]):
                        scr_ = np.zeros(1, dtype=object)
                        scr_[0] = scr
                        _cscores = scr_cscores[scr_cscores[:, 0] == scr_, 1:]
                        _cscores = [sc * tamp_func(dist) for sc, dist in _cscores]
                        # _cscores = [sc * tamp_func(dist) for sc, dist in zip(_cscores, nbr_dists)]
                        score_scr_dict[scr][cg] = weight_dict[cg] * np.max(_cscores)
                        # score_scr_dict[scr][cg] = weight_dict[cg] * len(_cscores)
                        # score_scr_dict[scr][cg] = weight_dict[cg] * np.mean(_cscores) * np.log(len(_cscores))
                        # score_scr_dict[scr][cg] = weight_dict[cg] * np.max(scr_cscores[scr_cscores[:, 0] == scr_, 1])
                for scr in score_scr_dict.keys():
                    score += max(score_scr_dict[scr].values())
                    # score += np.log(sum(score_scr_dict[scr].values()))
                    # score += sum(score_scr_dict[scr].values())
                # max_scores = []
                # for scr in score_scr_dict.keys():
                #     max_score = max(score_scr_dict[scr].values())
                #     score += max_score
                #     max_scores.append(max_score)
                # if len(max_scores) > 0:
                #     score += max(max_scores)
                lig_scores[i] = score

            sorted_lig_indices = lig_scores.argsort()[::-1]
            if top_percent_per_cluster is not None:
                last_index = max(int(np.ceil(len(sorted_lig_indices) * top_percent_per_cluster)), min_ligands_per_cluster)
            else:
                last_index = 1
            if max_ligands_per_cluster is not None:
                last_index = min(last_index, max_ligands_per_cluster)
            top_percent = sorted_lig_indices[:last_index]
            lig_indices.extend(list(mems[top_percent]))
        # cst_filter = np.in1d(np.arange(len(self.ligand_dataframe_neighbors)), lig_indices)
        # self._apply_cst_filter(cst_filter)
        self.filtered_lig_indices = np.unique(lig_indices)

    def set_cg_weights(self, weight_dict):
        self.cg_weights = weight_dict

    def find_poses(self, template=None, only_top_percent=None, min_poses=50, max_poses=None,
                   filter_ligands_by_cluster=False, lig_rmsd_cutoff=0.5, min_ligands_per_cluster=1,
                   max_ligands_per_cluster=None,
                   top_percent_per_cluster=None, weight_dict=None, max_ligands_to_search=None,
                   vdW_tolerance=0.1):

        self.set_ligand_rep()
        self.set_atom_cg_map_by_atomtype()

        if len(self.poses) > 0:
            self.poses = []

        if self.constraints is not None:
            #if lig-level csts, apply them now

            #Lig must have vdM for CG
            if self.constraints['Has_vdM'].any():
                cst_filter = self._get_HasVdm_cst_filter()
                if cst_filter is not None:
                    print('Applying HVM filter to', len(self.ligand_dataframe_neighbors), 'ligands...')
                    self._apply_cst_filter(cst_filter)
                    print('\t ', len(self.ligand_dataframe_neighbors), 'ligands remain...')

            ##Lig atom must be buried by certain distance
            # if ((~self.constraints['dist_buried'].isna())
            #     & (self.constraints['contact_type'] == set())
            #     & (self.constraints['Has_vdM_in_pose'].isna())).any():
            #     cst_filter = self._get_dist_buried_cst_filter()
            #     if cst_filter is not None:
            #         self._apply_cst_filter(cst_filter)

        if filter_ligands_by_cluster:
            print('Applying cluster filter to', len(self.ligand_dataframe_neighbors), 'ligands...')
            self.filter_ligands_by_cluster_members(rmsd_cutoff=lig_rmsd_cutoff,
                                                   min_ligands_per_cluster=min_ligands_per_cluster,
                                                   top_percent_per_cluster=top_percent_per_cluster,
                                                   weight_dict=weight_dict,
                                                   max_ligands_per_cluster=max_ligands_per_cluster)
            # print('\t ', len(self.ligand_dataframe_neighbors), 'ligands remain...')
            print('\t ', len(self.filtered_lig_indices), 'ligands remain...')

        if only_top_percent is not None:
            if self.filtered_lig_indices is not None:
                print('Applying top-percent filter to', len(self.filtered_lig_indices), 'ligands...')
            else:
                print('Applying top-percent filter to', len(self.ligand_dataframe_neighbors), 'ligands...')
            self.set_top_designable_percent(only_top_percent, min_poses=min_poses, max_poses=max_poses,
                                            weight_dict=weight_dict)
            # self.set_top_designable_percent(only_top_percent, min_poses=min_poses,
            #                                 max_poses=max_poses, weight_dict=weight_dict)
            # print('\t ', len(self.ligand_dataframe_neighbors), 'ligands remain...')
            print('\t ', len(self.filtered_lig_indices), 'ligands remain...')

        if self.filtered_lig_indices is not None:
            max_poses = min(max_poses, len(self.filtered_lig_indices))

        if self.constraints is not None and (self.constraints['contact_type'] != set()).any():
            contact_csts = True
        else:
            contact_csts = False

        pose_index = 0
        # for i, gr_name in enumerate(self.ligand_dataframe_neighbors.values):
        if self.filtered_lig_indices is not None:
            iterable = self.filtered_lig_indices
            if self.leftover_filtered_lig_indices is not None:
                iterable = np.concatenate((iterable, self.leftover_filtered_lig_indices))
        else:
            iterable = range(len(self.ligand_dataframe_neighbors))
        gr_names = self.ligand_dataframe_neighbors.values
        for k, i in enumerate(iterable):
            gr_name = gr_names[i]
            lig = self.ligand_dataframe_grs.get_group(tuple(gr_name))
            pose = Pose().set_ligand(lig)
            seen_vdms = set()
            for cg in self.ligand_neighbors_indices.keys():
                for cg_group in self.ligand_neighbors_indices[cg].keys():
                    indices = self.ligand_neighbors_indices[cg][cg_group][i]
                    dists = self.ligand_neighbors_dists[cg][cg_group][i]
                    if not self._max_dist_criterion:
                        dists = dists / np.sqrt(self.cg_num_atoms[cg])
                    vdm_names = self.cg_dataframe_neighbors[cg][indices]
                    un_vdm_names = set()
                    _vdm_names = []
                    _inds = []
                    for _i, vdm_name in enumerate(vdm_names):
                        tup_vdm_name = tuple(vdm_name)
                        if tup_vdm_name not in seen_vdms:
                            seen_vdms.add(tup_vdm_name)
                            if tup_vdm_name not in un_vdm_names:
                                un_vdm_names.add(tup_vdm_name)
                                _vdm_names.append(tup_vdm_name)
                                _inds.append(_i)

                    # _inds = [x for x, gr_name in enumerate(vdm_names)
                    #                 if tuple(gr_name) not in seen_vdms]
                    # vdm_names = np.array([tuple(vdm_names[x]) for x in _inds])
                    # un_vdm_names, un_inds = np.unique(vdm_names, axis=0, return_index=True)
                    # vdm_names = [tuple(un_vdm_name) for un_vdm_name in un_vdm_names]
                    # dists = dists[[_inds[un_ind] for un_ind in un_inds]]
                    # seen_vdms.update(vdm_names)

                    if len(_vdm_names) == 0:
                        continue
                    vdm_names = _vdm_names
                    dists = dists[_inds]

                    # cg_df_indices = list(itertools.chain(*[self.cg_dict_grs[cg].groups[tuple(gr_name)]
                    #                                        for gr_name in vdm_names]))
                    cg_df_indices = list(itertools.chain(*[self.cg_dict_grs[cg].groups[gr_name]
                                                          for gr_name in vdm_names]))

                    df = self.cg_dict[cg][self.cg_dict[cg].index.isin(cg_df_indices)]
                    df_info = DataFrame(vdm_names, columns=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                    df_info['dist_to_query'] = dists
                    df_info['CG_ligand_coverage'] = self.cg_ligand_coverage[cg][cg_group]
                    df_info['CG_group'] = cg_group
                    pose_vdms = merge(df, df_info, on=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                    if len(pose_vdms) > 0:
                        pose.vdms.append(pose_vdms)

            num_vdms = len(pose.vdms)
            if num_vdms == 0:
                print('no neighbors of ligand', gr_name)
                continue

            if num_vdms == 1:
                pose.vdms = pose.vdms[0]
            else:
                try:
                    pose.vdms = fast_concat(pose.vdms) # faster than pd.concat(pose.vdms)
                except ValueError:
                    print('\t', gr_name, 'has', num_vdms, 'vdms, but fast_concat failed')
                    for vdm in pose.vdms:
                        print(vdm.columns)
                        print(vdm.shape)
                        print(vdm)
                        print('resname', vdm.resname_rota.iat[0])
                        print(vdm.CG.iat[0], vdm.rota.iat[0], vdm.probe_name.iat[0], vdm.seg_chain_resnum.iat[0])
                    continue

            if self.constraints is not None:
                pose.set_nonclashing_vdms(copy=True, vdW_tolerance=vdW_tolerance,)
            else:
                pose.set_nonclashing_vdms(vdW_tolerance=vdW_tolerance)
            if len(pose.vdms) == 0:
                continue

            # if contact_csts:
            pose.set_ligand_contacts(template, vdW_tolerance=vdW_tolerance)

            set_pose = True
            if self.constraints is not None:
                if self.check_csts(pose, lig_index=i, template=template):
                    pose.set_vdms_satisfy_contact_csts(self.constraints)
                else:
                    set_pose = False
            if set_pose:
                pose.identifier = pose_index
                if self.cg_weights:
                    pose.cg_weights = self.cg_weights
                self.poses.append(pose)
                pose_index += 1
                if max_poses is not None:
                    if pose_index == max_poses:
                        break
            
            if max_ligands_to_search is not None:
                if k + 1 == max_ligands_to_search:
                    print('Reached maximum ligands to search for poses. Stopping.')
                    break

        print('A total of', len(self.poses), 'poses were found.')

    def add_vdms_to_poses(self, template, use_optimum_vdms_only=True, freeze_optimum_vdms=False,
                            vdW_tolerance=0.1):

        self.set_ligand_rep()
        self.set_atom_cg_map_by_atomtype()

        if len(self.poses) > 0:
            self.poses = []

        if len(self._poses) == 0:
            raise Exception('No poses have been loaded.  See load_poses().')

        # if self.constraints is not None:
        #     # if lig-level csts, apply them now
        #
        #     # Lig must have vdM for CG
        #     if self.constraints['Has_vdM'].any():
        #         cst_filter = self._get_HasVdm_cst_filter()
        #         if cst_filter is not None:
        #             print('Applying HVM filter to', len(self.ligand_dataframe_neighbors), 'ligands...')
        #             self._apply_cst_filter(cst_filter)
        #             print('\t ', len(self.ligand_dataframe_neighbors), 'ligands remain...')
        #
        #     # Lig atom must be buried by certain distance
        #     if ((~self.constraints['dist_buried'].isna())
        #         & (self.constraints['contact_type'] == set())
        #         & (self.constraints['Has_vdM_in_pose'].isna())).any():
        #         cst_filter = self._get_dist_buried_cst_filter()
        #         if cst_filter is not None:
        #             self._apply_cst_filter(cst_filter)
        #
        # if self.constraints is not None and (self.constraints['contact_type'] != set()).any():
        #     contact_csts = True
        # else:
        #     contact_csts = False

        # pose_index = 0
        for i, gr_name in enumerate(self.ligand_dataframe_neighbors.values):
            lig = self.ligand_dataframe_grs.get_group(tuple(gr_name))
            pose = Pose().set_ligand(lig)
            if use_optimum_vdms_only:
                seen_vdms = set(self._poses[i].opt_vdms_names)
            else:
                seen_vdms = set(self._poses[i].vdms_grs.groups.keys())
            for cg in self.ligand_neighbors_indices.keys():
                for cg_group in self.ligand_neighbors_indices[cg].keys():
                    indices = self.ligand_neighbors_indices[cg][cg_group][i]
                    dists = self.ligand_neighbors_dists[cg][cg_group][i]
                    if not self._max_dist_criterion:
                        dists = dists / np.sqrt(self.cg_num_atoms[cg])
                    vdm_names = self.cg_dataframe_neighbors[cg][indices]
                    _inds = [x for x, gr_name in enumerate(vdm_names)
                                    if tuple(gr_name) not in seen_vdms]
                    vdm_names = [tuple(vdm_names[x]) for x in _inds]
                    dists = dists[_inds]
                    seen_vdms.update(vdm_names)
                    # cg_df_indices = list(itertools.chain(*[self.cg_dict_grs[cg].groups[tuple(gr_name)]
                    #                                        for gr_name in vdm_names]))
                    cg_df_indices = list(itertools.chain(*[self.cg_dict_grs[cg].groups[gr_name]
                                                           for gr_name in vdm_names]))
                    df = self.cg_dict[cg][self.cg_dict[cg].index.isin(cg_df_indices)]
                    df_info = DataFrame(vdm_names, columns=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                    df_info['dist_to_query'] = dists
                    df_info['CG_ligand_coverage'] = self.cg_ligand_coverage[cg][cg_group]
                    df_info['CG_group'] = cg_group
                    pose_vdms = merge(df, df_info, on=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                    if len(pose_vdms) > 0:
                        pose.vdms.append(pose_vdms)

            num_vdms = len(pose.vdms)
            if num_vdms == 0:
                print('No new neighbors found for ligand in pose id', self._poses[i].identifier)
                pose = self._poses[i]
                # pose.identifier = pose_index
                # pose_index += 1
                pose._already_scored = True
                self.poses.append(pose)
                continue

            print('New neighbors found for ligand in pose id', self._poses[i].identifier)

            if num_vdms == 1:
                pose.vdms = pose.vdms[0]
            if num_vdms > 1:
                pose.vdms = fast_concat(pose.vdms)  # faster than pd.concat(pose.vdms)

            if self.constraints is not None:
                pose.set_nonclashing_vdms(copy=True, vdW_tolerance=vdW_tolerance)
            else:
                pose.set_nonclashing_vdms(vdW_tolerance=vdW_tolerance)

            if len(pose.vdms) > 0:
                print('A new vdM is not clashing with ligand.')
                if use_optimum_vdms_only:
                    if len(self._poses[i].opt_vdms) > 0:
                        old_pose_vdms = self._poses[i].opt_vdms.drop(columns='score')
                        old_pose_vdms_sidechains = self._poses[i].opt_vdms_sidechains.drop(columns='score')

                        if freeze_optimum_vdms:
                            print('Freezing previously optimum vdMs...')
                            frozen_sites = {n[-1]: n for n in self._poses[i].opt_vdms_names}
                            pose._frozen_sites = frozen_sites
                            pose._force_MC = True
                    else:
                        old_pose_vdms = pd.DataFrame()
                        old_pose_vdms_sidechains = pd.DataFrame()

                else:
                    old_pose_vdms = self._poses[i].vdms
                    old_pose_vdms_sidechains = self._poses[i].vdms_sidechains.drop(columns='score')

                pose.vdms = concat([old_pose_vdms, pose.vdms])
                pose.vdms_sidechains = concat([old_pose_vdms_sidechains, pose.vdms_sidechains], ignore_index=True)

            else:
                print('All new vdMs are clashing with ligand.')
                pose = self._poses[i]
                pose._already_scored = True

            # if contact_csts:
            pose.set_ligand_contacts(template, vdW_tolerance=vdW_tolerance)

            # set_pose = True
            # if self.constraints is not None:
            #     if self.check_csts(pose, lig_index=i):
            #         pose.set_vdms_satisfy_contact_csts(self.constraints)
            #     else:
            #         set_pose = False
            # if set_pose:
            pose.identifier = self._poses[i].identifier #pose_index
            # pose_index += 1
            self.poses.append(pose)

    def score_poses(self, template, poses=None, bbdep=True, use_hb_scores=False, C_score_threshold=0,
                    return_top_scoring_vdMs_only=False, store_pairwise=True, force_MC=False,
                    force_DEE=False, DEE_to_MC_switch_number=1000, 
                    compute_pairwise_contacts=False,
                    tamp_by_distance=False, pair_nbr_distance=0.7, 
                    exponential=False,
                    log_logistic=True, gaussian=False, relu=False,
                    knn_contacts=True, contact_distance_metric='rmsd',
                    use_same_rotamer_for_pairwise_contacts=True, use_same_rotamer_for_lig_contacts=True,
                    ignore_rmsd_column=(),
                    pairwise_contact_weight=0.5,
                    burial_threshold=0.5,
                    outer_shell_score_weight=0.5,
                    vdW_tolerance=0.1):
        
        if self.path_to_database is None:
            raise ValueError('No path to database provided.')

        if store_pairwise:
            pairwise_dict = self._pairwise_dict
        else:
            pairwise_dict = None

        if poses is None:
            poses = self.poses

        for pose in poses:
            pose.score(template, sample=self, bbdep=bbdep, use_hb_scores=use_hb_scores,
                       C_score_threshold=C_score_threshold, pairwise_dict=pairwise_dict,
                       force_MC=force_MC, force_DEE=force_DEE, 
                       DEE_to_MC_switch_number=DEE_to_MC_switch_number,
                       compute_pairwise_contacts=compute_pairwise_contacts,
                       path_to_database=self.path_to_database,
                       tamp_by_distance=tamp_by_distance,
                       pair_nbr_distance=pair_nbr_distance,
                       exponential=exponential,
                       log_logistic=log_logistic, gaussian=gaussian, 
                       relu=relu, knn_contacts=knn_contacts, 
                       contact_distance_metric=contact_distance_metric,
                       use_same_rotamer_for_pairwise_contacts=use_same_rotamer_for_pairwise_contacts, 
                       use_same_rotamer_for_lig_contacts=use_same_rotamer_for_lig_contacts,
                       ignore_rmsd_column=ignore_rmsd_column,
                       pairwise_contact_weight=pairwise_contact_weight,
                       burial_threshold=burial_threshold,
                       outer_shell_score_weight=outer_shell_score_weight,
                       vdW_tolerance=vdW_tolerance)
            if return_top_scoring_vdMs_only:
                pose.vdms = []

    def cluster_poses_by_ligand(self, rmsd_cutoff=0.8, max_dist=False):
        lig_coords = [pose.ligand[~pose.ligand['atom_type_label'].isin(['h_pol', 'h_alkyl', 'h_aro'])][['c_x', 'c_y', 'c_z']].values for pose in self.poses]
        clu = Cluster(rmsd_cutoff=rmsd_cutoff)
        clu.pdb_coords = np.array(lig_coords, dtype=np.float32)
        clu.make_pairwise_rmsd_mat(maxdist=max_dist, superpose=False)
        clu.make_square()
        clu.make_adj_mat()
        clu.cluster()
        self.pose_clusters_members = clu.mems
        self.pose_clusters_centroids = clu.cents

    def cluster_vdms(self, dists=None, cluster_cg_around_pts=None, radius_to_pts=5.0):
        # need to update this to account for symmetry-related CGs, e.g. coo and ph
        if cluster_cg_around_pts is not None:
            self.set_cg_com_neighbors()
            inds_around_pts = dict()
            for cg in self._cg_com_tree.keys():
                inds = set()
                for pt in cluster_cg_around_pts:
                    inds.update(list(self._cg_com_tree[cg].query_radius([pt], r=radius_to_pts)[0]))
                inds_around_pts[cg] = sorted(inds)

            for cg in self.cg_neighbors.keys():
                r = self.cg_neighbors[cg].radius
                if dists is not None and cg in dists:
                    r = dists[cg]
                _inds_around_pts_arr = np.array(inds_around_pts[cg])
                X = self.cg_neighbors[cg]._fit_X[_inds_around_pts_arr]
                nbrs = NearestNeighbors(radius=r, algorithm='ball_tree').fit(X)
                adj_mat = nbrs.radius_neighbors_graph(X, radius=r, mode='connectivity')
                clu = Cluster()
                clu.adj_mat = adj_mat
                clu.fast_cluster()
                self.vdm_clusters_members[cg] = [_inds_around_pts_arr[mems] for mems in clu.mems]
                self.vdm_clusters_centroids[cg] = [_inds_around_pts_arr[cent] for cent in clu.cents]
        else:
            for cg in self.cg_neighbors.keys():
                r = self.cg_neighbors[cg].radius
                if dists is not None and cg in dists:
                    r = dists[cg]
                adj_mat = self.cg_neighbors[cg].radius_neighbors_graph(self.cg_neighbors[cg]._fit_X,
                                                                    radius=r,
                                                                    mode='connectivity')
                clu = Cluster()
                clu.adj_mat = adj_mat
                clu.fast_cluster()
                self.vdm_clusters_members[cg] = clu.mems
                self.vdm_clusters_centroids[cg] = clu.cents

    def print_vdm_cluster(self, cg, cluster_number=1, outpath='./', tag='', prefix='',
                          centroids_only=False, b_factor_column=None):
        cluster_index = cluster_number - 1
        if cluster_index >= len(self.vdm_clusters_members[cg]):
            print('Cluster number', cluster_number, 'does not exist.')
            return
        mems = self.vdm_clusters_members[cg][cluster_index]
        cent = self.vdm_clusters_centroids[cg][cluster_index]
        gr_name_cent = tuple(self.cg_dataframe_neighbors[cg][cent])
        for gr_name in self.cg_dataframe_neighbors[cg][mems]:
            df_vdM = self.cg_dict_grs[cg].get_group(tuple(gr_name))
            if tuple(gr_name) == gr_name_cent:
                iscent = '_centroid'
            else:
                iscent = ''
            if centroids_only and iscent == '':
                continue
            Pose().print_vdM(df_vdM, outpath=outpath, tag=tag + iscent, 
                             prefix=prefix, b_factor_column=b_factor_column)

    @staticmethod
    def set_buried_unsatisfied(poses, template, burial_threshold=0.5, exclude_mc_hb=False):
        for pose in poses:
            pose.set_buried_unsatisfied(template, burial_threshold=burial_threshold, 
                                        exclude_mc_hb=exclude_mc_hb)

    def set_buried_unsatisfied_all_poses(self, template, burial_threshold=0.5, exclude_mc_hb=False):
        for pose in self.poses:
            pose.set_buried_unsatisfied(template, burial_threshold=burial_threshold,
                                        exclude_mc_hb=exclude_mc_hb)

    @staticmethod
    def store_buried_unsatisfied(poses):
        for pose in poses:
            pose.store_buried_unsatisfied()

    def set_poss_vdms_for_buried_unsatisfied(self, poses, template, allowed_amino_acids=None, 
                                   allowed_seg_chain_resnums=None):
        poss_vdms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
        vdm_pose_map = defaultdict(set)
        for i, pose in enumerate(poses):
            pose.set_poss_vdms_for_buried_unsatisfied(template, allowed_amino_acids=allowed_amino_acids,
                                            allowed_seg_chain_resnums=allowed_seg_chain_resnums)                                 
            d = pose.poss_vdms_for_buried_unsatisfied
            for cg in d.keys():
                for aa in d[cg].keys():
                    for label in d[cg][aa].keys():
                        for scrn in d[cg][aa][label].keys():
                            _vdms = d[cg][aa][label][scrn]
                            poss_vdms[cg][aa][label][scrn].update(_vdms)
                            for _vdm in _vdms:
                                for __vdm in _vdm:
                                    vdm_pose_map[__vdm].add(i)    
        for __vdm, indices in vdm_pose_map.items():
            self.vdm_pose_map[__vdm] = list(indices)
        self.poss_vdms_for_buried_unsatisfied =  poss_vdms

    def find_vdms_for_buried_unsatisfied(self, poses, template, path_to_nbrs_database, distance_metric='rmsd',
                                        rmsd=0.5, maxdist=0.65, specific_seg_chain_resnums_only=None,
                                        filter_by_phi_psi=False, filter_by_phi_psi_exclude_sc=True):
        """ 
        This function should be moved outside to Sample class. The vdms should contain a pose index in them.
        You should make a concatenated vdm dict of all poses.  Then run this function on that, so you only
        load the nbrs once. It's slow to do i/o so many times, I think.  Then once vdms are found,
        group all according to cg/aa, then load vdms (remove clashing) and add to appropriate pose indices.  
        Also need a way to track which buried unsats in a pose were already search for, so you don't try 
        satisfying them again in a recursive search.

        Could select best second shell irrespective of 3rd shell, then freeze that in opt_vdms for recursive search?
        Save 2nd shell possibilites in case can't find good 3rd shell for best 2nd shell. Choose 2nd shell with best
        score and least number of buried unsats?
        """
        # Might want to make a function that finds vdms for one residue only. Hack below
        vdms = self.poss_vdms_for_buried_unsatisfied
        vdms_buried_unsat = defaultdict(lambda: defaultdict(set))

        if specific_seg_chain_resnums_only is not None:
            vdms_old = vdms
            vdms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
            for cg in vdms_old.keys():
                for aa in vdms_old[cg].keys():
                    for label in vdms_old[cg][aa].keys():
                        for scrn in vdms_old[cg][aa][label]:
                            for vdm_name in vdms_old[cg][aa][label][scrn]:
                                if vdm_name[0][-1] in specific_seg_chain_resnums_only:
                                    vdms[cg][aa][label][scrn].add(vdm_name)

        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
        path_to_nbrs_database_scores = path_to_nbrs_database + 'vdMs_cg_nbrs_scores/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'
        print('finding vdM nbrs for buried unsatisfied polar atoms...')
        for cg in vdms.keys():
            if cg == 'ccoh' and 'coh' in vdms.keys():
                continue
            print('\t', cg)
            if cg not in os.listdir(path_to_nbrs_database_):
                print('\t\t CG not found in database. Skipping...')
                continue
            cg_df = cg_dfs[cg]
            for aa in vdms[cg].keys():
                print('\t\t', aa)
                if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                    print('\t\t\t AA not found in CG database. Skipping...')
                    continue
                if distance_metric == 'rmsd':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    num_cg_atoms = nbrs._fit_X.shape[1] / 3
                    radius = rmsd * np.sqrt(num_cg_atoms)
                elif distance_metric == 'maxdist':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    radius = maxdist

                scores = pd.read_parquet(path_to_nbrs_database_scores + cg + '/' + aa + '.parquet.gzip')
                score_col_dict = {colname: i for i, colname in enumerate(scores.columns)}
                scores = scores.values

                with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                    groupnames = pickle.load(f)

                for acc_or_don in vdms[cg][aa].keys():
                    for rota_info in vdms[cg][aa][acc_or_don]:
                        cg_info = vdms[cg][aa][acc_or_don][rota_info]
                        seg_aa, chain_aa, res_aa = rota_info
                        if (seg_aa, chain_aa, res_aa) not in template.phi_psi_dict:
                            df_targ_res = template.dataframe[template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)]
                            template_phi = df_targ_res['phi'].iat[0]
                            template_psi = df_targ_res['psi'].iat[0]
                            template.phi_psi_dict[(seg_aa, chain_aa, res_aa)] = (template_phi, template_psi)
                        else:
                            template_phi, template_psi = template.phi_psi_dict[(seg_aa, chain_aa, res_aa)]

                        if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                            df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                                    (template.dataframe['chain'] == chain_aa) &
                                                    (template.dataframe['resnum'] == res_aa)].copy()
                            m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                            for _name in ['N', 'CA', 'C']])
                            t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                            R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                            template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com)
                        
                        for vdms_cg in cg_info:
                            for vdm_cg in vdms_cg:
                                # print('finding nbrs for', vdm_cg)
                                # print('nbrs', cg, aa, acc_or_don, rota_info, vdm_cg)
                                pose_index = self.vdm_pose_map[vdm_cg][0] 
                                pose = poses[pose_index]
                                if pose.opt_vdms_grs is None:
                                    pose.opt_vdms_grs = pose.opt_vdms.groupby(pose.groupby)
                                vdm = pose.opt_vdms_grs.get_group(vdm_cg)
                                dfy = pd.merge(cg_df, vdm[vdm.chain=='X'], on=['name', 'resnum', 'resname'])
                                coords = dfy[['c_x', 'c_y', 'c_z']].values
                                # if vdm_cg == (4, 2, '6F7B_biomol_1_A_A', 'coh', ('', 'A', 49)):
                                #     print((seg_aa, chain_aa, res_aa))
                                #     print(vdm_cg)
                                #     print(coords)
                                #     print(vdm)
                                R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]
                                dfy[['c_x', 'c_y', 'c_z']] = apply_transform(R, mob_com, targ_com, coords)
                                coords = dfy[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)
                                # if vdm_cg == (6, 2, '1YOA_biomol_1_A_A', 'coh', ('', 'A', 52)):
                                #     print(vdm_cg, 'after')
                                #     print(coords)
                                try:
                                    dists, inds = nbrs.radius_neighbors(coords, radius=radius, return_distance=True)
                                except Exception as e:
                                    print(e)
                                    print('The pose possibily contains duplicate vdMs.  Perhaps \
                                        check the CG_ligand_coverage column of the ligand.txt file for \
                                        overlapping atoms in different CG_ligand_coverage groups.')
                                    print('culprit vdM:', cg, aa, vdm_cg)
                                    continue
                                dists, inds = dists[0], inds[0]
                                # if vdm_cg == (6, 2, '1YOA_biomol_1_A_A', 'coh', ('', 'A', 52)):
                                #     print('inds')
                                #     print(inds)
                                if cg in cgs_that_flip:
                                    dfy['chain'] = dfy['chain_x']
                                    coords = flip_cg_coords(dfy).reshape(1, -1)
                                    dists_flip, inds_flip = nbrs.radius_neighbors(coords, radius=radius, return_distance=True)
                                    dists_flip, inds_flip = dists_flip[0], inds_flip[0]
                                    if dists_flip.size > 0 and dists.size > 0:
                                        inds = np.concatenate((inds, inds_flip))
                                        inds, index = np.unique(inds, return_index=True)
                                        dists = np.concatenate((dists, dists_flip))[index]
                                    elif dists_flip.size > 0:
                                        dists = dists_flip
                                        inds = inds_flip
                                if dists.size > 0: # If there are nbrs...
                                    if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
                                        score_col_contact_type_ind = score_col_dict['contact_type']
                                        contact_types = scores[inds, score_col_contact_type_ind].astype(str)
                                        score_col_phi_ind = score_col_dict['phi']
                                        phis = scores[inds, score_col_phi_ind].astype(float)
                                        phi_diffs = get_angle_diff(phis, template_phi)
                                        score_col_psi_ind = score_col_dict['psi']
                                        psis = scores[inds, score_col_psi_ind].astype(float)
                                        psi_diffs = get_angle_diff(psis, template_psi)
                                        indices = np.arange(inds.size)
                                        passed_phi_psi_filter = np.zeros(indices.size, dtype=bool)
                                        for contact_type in phi_psi_dict.keys():
                                            if filter_by_phi_psi_exclude_sc:
                                                if contact_type == 'sc' and aa != 'GLY':
                                                    mask = contact_types == contact_type
                                                    passed_phi_psi_filter[mask] = True
                                                    continue
                                            mask = contact_types == contact_type
                                            if not mask.any():
                                                continue
                                            phi_diffs_masked = phi_diffs[mask]
                                            phi_mask = phi_diffs_masked <= 2 * phi_psi_dict[contact_type]['phi']
                                            psi_diffs_masked = psi_diffs[mask]
                                            psi_mask = psi_diffs_masked <= 2 * phi_psi_dict[contact_type]['psi']
                                            phi_psi_mask = phi_mask & psi_mask
                                            passed_phi_psi_filter[indices[mask][phi_psi_mask]] = True
                                        if not passed_phi_psi_filter.any():
                                            continue
                                                                        
                                        inds = inds[passed_phi_psi_filter]
                                        dists = dists[passed_phi_psi_filter]
                                        # if vdm_cg == (6, 2, '1YOA_biomol_1_A_A', 'coh', ('', 'A', 52)):
                                        #     print('inds phi/psi')
                                        #     print(inds)
                                        if len(inds) == 0:
                                            continue
                                    # acc_or_don == 'is_acceptor' or 'is_donor'
                                    is_acc_or_don = scores[inds, score_col_dict[acc_or_don]].astype(bool)
                                    inds = inds[is_acc_or_don]
                                    dists = dists[is_acc_or_don]
                                    if len(inds) == 0:
                                        continue
                                    if distance_metric == 'rmsd':
                                        nbr_groupnames = [(cg, aa, dist / np.sqrt(num_cg_atoms), groupnames[m]) for dist, m in zip(dists, inds)]
                                    else:
                                        nbr_groupnames = [(cg, aa, dist, groupnames[m]) for dist, m in zip(dists, inds)]
                                    print('\t\t\t', len(nbr_groupnames), 'nbrs found for', vdm_cg)
                                    vdms_buried_unsat[rota_info][vdm_cg].update(nbr_groupnames)
        self.vdms_buried_unsat = vdms_buried_unsat

    def reorganize_vdms_buried_unsat(self):
        vdms = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.pose_nbr_map = defaultdict(lambda: defaultdict(set))
        for scrn in self.vdms_buried_unsat.keys():
            for vdm in self.vdms_buried_unsat[scrn].keys():
                for nbr in self.vdms_buried_unsat[scrn][vdm]:
                    cg, aa, dist, nbr_vdm_gr_name = nbr
                    vdms[cg][aa][scrn].add(nbr_vdm_gr_name)
                    nbr_vdm_name = list(nbr_vdm_gr_name)
                    nbr_vdm_name.append(cg)
                    nbr_vdm_name.append(scrn)
                    self.nbr_vdm_map[tuple(nbr_vdm_name)][vdm] = dist
                    pose_indices = self.vdm_pose_map[vdm]
                    for pose_index in pose_indices:
                        self.pose_nbr_map[pose_index][cg].add(tuple(nbr_vdm_name))
        return vdms

    def load_vdms_buried_unsat(self, template, filter_by_phi_psi=False, 
                               filter_by_phi_psi_exclude_sc=True, vdW_tolerance=0.1):
        cg_vdm_gr_names = self.reorganize_vdms_buried_unsat()
        sample = self.__copy__(minimal_info=True, include_poses_in_minimal_info=False)
        residue_dict = defaultdict(lambda: defaultdict(list))
        for cg in cg_vdm_gr_names.keys():
            for aa in cg_vdm_gr_names[cg].keys():
                for scrn in cg_vdm_gr_names[cg][aa].keys():
                    res = Residue()
                    res.CG = cg
                    res.resname = aa
                    res.seg_chain_resnum = scrn
                    residue_dict[cg][aa].append(res)
        sample.residue_dict = residue_dict
        sample.load_vdms(template, filter_by_phi_psi=filter_by_phi_psi,
                filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                num_cpus=None, run_parallel=False, cg_vdm_gr_names=cg_vdm_gr_names,
                ignore_CG_for_clash_check=True, vdW_tolerance=vdW_tolerance)
        self._sample_buried_unsat = sample

    def dole_vdms_buried_unsat_to_poses(self, poses, template, use_optimum_vdms_only=True, freeze_optimum_vdms=True,
                                            vdW_tolerance=0.1):
        new_poses = []
        for i, old_pose in enumerate(poses):
            print('Doling new vdMs to pose', i)
            pose = Pose().set_ligand(old_pose.ligand)

            # carry over relevant attributes:
            pose.stored_buried_unsat_gr_names = old_pose.stored_buried_unsat_gr_names

            seen_vdms = set(old_pose.opt_vdms_names)
            for cg, nbr_vdm_names in self.pose_nbr_map[i].items():
                if cg not in self._sample_buried_unsat.cg_dict_grs:
                    continue
                df_cg_grs = self._sample_buried_unsat.cg_dict_grs[cg]
                cg_df_indices = list(itertools.chain(*[df_cg_grs.groups[gr_name]
                                        for gr_name in nbr_vdm_names if 
                                        (gr_name in df_cg_grs.groups and gr_name not in seen_vdms)]))
                df_cg = self._sample_buried_unsat.cg_dict[cg]
                df = df_cg[df_cg.index.isin(cg_df_indices)]
                nbr_dists = []
                _nbr_vdm_names =[]
                seg_ch_rns = []
                for nbr_vdm_name in nbr_vdm_names:
                    if not (nbr_vdm_name in df_cg_grs.groups and nbr_vdm_name not in seen_vdms):
                        continue
                    seen_vdms.add(nbr_vdm_name)
                    # print(nbr_vdm_name)
                    for vdm_name in old_pose.opt_vdms_names:
                        if vdm_name in self.nbr_vdm_map[nbr_vdm_name]:
                            dist = self.nbr_vdm_map[nbr_vdm_name][vdm_name]
                            if nbr_vdm_name in _nbr_vdm_names:
                                # print('nbr in _nbr_vdm_names', nbr_vdm_name)
                                index_nbr = _nbr_vdm_names.index(nbr_vdm_name)
                                if dist <  nbr_dists[index_nbr]:
                                    nbr_dists[index_nbr] = dist
                                    seg_ch_rns[index_nbr] = vdm_name[-1]
                            else:
                                nbr_dists.append(dist)
                                _nbr_vdm_names.append(nbr_vdm_name)
                                seg_ch_rns.append(vdm_name[-1])
                df_info = DataFrame(_nbr_vdm_names, columns=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                df_info['dist_to_query'] = nbr_dists
                df_info['CG_ligand_coverage'] = 1
                df_info['CG_group'] = 1
                df_info['ligand_type'] = 'sc'
                df_info['ligand_seg_chain_resnum'] = seg_ch_rns
                pose_vdms = merge(df, df_info, on=['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
                pose.vdms.append(pose_vdms)
            num_vdms = len(pose.vdms)
            if num_vdms == 0:
                pose = old_pose
                pose._already_scored = True
                new_poses.append(pose)
                continue
            print('Outer-shell vdM(s) found for vdMs in pose id', old_pose.identifier)
            if num_vdms == 1:
                pose.vdms = pose.vdms[0]
            if num_vdms > 1:
                pose.vdms = fast_concat(pose.vdms)  # faster than pd.concat(pose.vdms)
            
            if self.constraints is not None:
                pose.set_nonclashing_vdms(additional_vdms=old_pose.opt_vdms_sidechains, 
                                            vdW_tolerance=vdW_tolerance, copy=True)
            else:
                pose.set_nonclashing_vdms(additional_vdms=old_pose.opt_vdms_sidechains,
                                            vdW_tolerance=vdW_tolerance)
            if len(pose.vdms) > 0:
                print('New outer-shell vdM(s) not clashing with ligand or optimum vdMs.')
                if use_optimum_vdms_only:
                    if len(old_pose.opt_vdms) > 0:
                        old_pose_vdms = old_pose.opt_vdms.drop(columns='score')
                        old_pose_vdms_sidechains = old_pose.opt_vdms_sidechains.drop(columns='score')

                        if freeze_optimum_vdms:
                            print('Freezing previously optimum vdMs...')
                            frozen_sites = {n[-1]: n for n in old_pose.opt_vdms_names}
                            pose._frozen_sites = frozen_sites
                            pose._force_MC = True
                    else:
                        old_pose_vdms = pd.DataFrame()
                        old_pose_vdms_sidechains = pd.DataFrame()

                else:
                    old_pose_vdms = old_pose.vdms
                    old_pose_vdms_sidechains = old_pose.vdms_sidechains.drop(columns='score')
                    
                pose.vdms = concat([old_pose_vdms, pose.vdms])
                pose.vdms_sidechains = concat([old_pose_vdms_sidechains, pose.vdms_sidechains], ignore_index=True)

            else:
                print('All new vdMs are clashing with ligand or optimum vdMs.')
                pose = old_pose
                pose._already_scored = True

            # if contact_csts:
            pose.set_ligand_contacts(template, vdW_tolerance=vdW_tolerance)

            # set_pose = True
            # if self.constraints is not None:
            #     if self.check_csts(pose, lig_index=i):
            #         pose.set_vdms_satisfy_contact_csts(self.constraints)
            #     else:
            #         set_pose = False
            # if set_pose:
            pose.identifier = old_pose.identifier #pose_index
            # pose_index += 1
            new_poses.append(pose)
        return new_poses

    def _run_recursive_vdM_search(self, poses, template, bbdep=True, use_hb_scores=False, C_score_threshold=0,
                    return_top_scoring_vdMs_only=False, store_pairwise=True, force_MC=False,
                    force_DEE=False, DEE_to_MC_switch_number=1000, 
                    compute_pairwise_contacts=False,
                    tamp_by_distance=False, pair_nbr_distance=0.7, 
                    exponential=False,
                    log_logistic=True, gaussian=False, relu=False,
                    knn_contacts=True, contact_distance_metric='rmsd',
                    use_same_rotamer_for_pairwise_contacts=True, use_same_rotamer_for_lig_contacts=True,
                    ignore_rmsd_column=(),
                    pairwise_contact_weight=0.5,
                    filter_by_phi_psi=False, 
                    filter_by_phi_psi_exclude_sc=True,
                    specific_seg_chain_resnums_only=None,
                    rmsd=0.5,
                    maxdist=0.65,
                    distance_metric='rmsd',
                    allowed_amino_acids=None,
                    allowed_seg_chain_resnums=None,
                    burial_threshold=0.5,
                    use_optimum_vdms_only=True,
                    freeze_optimum_vdms=True,
                    path_to_database=None,
                    outer_shell_score_weight=0.5,
                    vdW_tolerance=0.1):
        self.set_buried_unsatisfied(poses, template, burial_threshold=burial_threshold, exclude_mc_hb=True)
        self.set_poss_vdms_for_buried_unsatisfied(poses, template, 
                                                allowed_amino_acids=allowed_amino_acids, 
                                                allowed_seg_chain_resnums=allowed_seg_chain_resnums)
        if len(self.poss_vdms_for_buried_unsatisfied) == 0:
            print('F 1')
            return poses, False

        if path_to_database is None:
            path_to_nbrs_database = self.path_to_database + '../nbrs/'
        else:
            path_to_nbrs_database = path_to_database + '../nbrs/'
        self.find_vdms_for_buried_unsatisfied(poses, template, 
                                            path_to_nbrs_database=path_to_nbrs_database, 
                                            distance_metric=distance_metric,
                                            rmsd=rmsd, maxdist=maxdist, 
                                            specific_seg_chain_resnums_only=specific_seg_chain_resnums_only,
                                            filter_by_phi_psi=filter_by_phi_psi, 
                                            filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc)
        if len(self.vdms_buried_unsat) == 0:
            print('F 2')
            return poses, False

        self.load_vdms_buried_unsat( template, filter_by_phi_psi=filter_by_phi_psi, 
                                    filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                    vdW_tolerance=vdW_tolerance)
        if len(self._sample_buried_unsat.cg_dict_grs) == 0:
            print('F 3')
            return poses, False

        poses = self.dole_vdms_buried_unsat_to_poses(poses, template, 
                                            use_optimum_vdms_only=use_optimum_vdms_only, 
                                            freeze_optimum_vdms=freeze_optimum_vdms)
        self.score_poses(template, poses=poses, bbdep=bbdep, use_hb_scores=use_hb_scores, 
                            C_score_threshold=C_score_threshold,
                            return_top_scoring_vdMs_only=return_top_scoring_vdMs_only, 
                            store_pairwise=store_pairwise, force_MC=force_MC,
                            force_DEE=force_DEE, DEE_to_MC_switch_number=DEE_to_MC_switch_number, 
                            compute_pairwise_contacts=compute_pairwise_contacts,
                            tamp_by_distance=tamp_by_distance, pair_nbr_distance=pair_nbr_distance, 
                            exponential=exponential,
                            log_logistic=log_logistic, gaussian=gaussian, relu=relu,
                            knn_contacts=knn_contacts, contact_distance_metric=contact_distance_metric,
                            use_same_rotamer_for_pairwise_contacts=use_same_rotamer_for_pairwise_contacts, 
                            use_same_rotamer_for_lig_contacts=use_same_rotamer_for_lig_contacts,
                            ignore_rmsd_column=ignore_rmsd_column,
                            pairwise_contact_weight=pairwise_contact_weight,
                            outer_shell_score_weight=outer_shell_score_weight,
                            burial_threshold=burial_threshold,
                            vdW_tolerance=vdW_tolerance)

        self.store_buried_unsatisfied(poses)
        return poses, True

    def run_recursive_vdM_search(self, poses, template, max_iter=4, bbdep=True, 
                                use_hb_scores=False, C_score_threshold=0,
                                return_top_scoring_vdMs_only=False, 
                                store_pairwise=False, force_MC=False,
                                force_DEE=False, DEE_to_MC_switch_number=1000, 
                                compute_pairwise_contacts=True,
                                tamp_by_distance=True, pair_nbr_distance=0.7, 
                                exponential=False,
                                log_logistic=False, gaussian=True, relu=False,
                                knn_contacts=True, contact_distance_metric='rmsd',
                                use_same_rotamer_for_pairwise_contacts=True, 
                                use_same_rotamer_for_lig_contacts=True,
                                ignore_rmsd_column=(),
                                pairwise_contact_weight=0.5,
                                filter_by_phi_psi=False, 
                                filter_by_phi_psi_exclude_sc=True,
                                specific_seg_chain_resnums_only=None,
                                rmsd=0.5,
                                maxdist=0.65,
                                distance_metric='rmsd',
                                allowed_amino_acids=None,
                                allowed_seg_chain_resnums=None,
                                burial_threshold=0.5,
                                use_optimum_vdms_only=True,
                                freeze_optimum_vdms=True,
                                outer_shell_score_weight=0.5,
                                vdW_tolerance=0.1
                                ):
        """will bb vdM (phi/psi) that are second-shell be included in pose 
        if sc of same res already a vdm for ligand?
        Also, is is_acceptor is_donor doing anything when finding nbrs?"""
        for i in range(max_iter):
            print('*******************')
            print('iteration:', i + 1)
            print('*******************')
            poses, keep_searching = self._run_recursive_vdM_search(poses, template,
                                burial_threshold=burial_threshold,
                                allowed_amino_acids=allowed_amino_acids, 
                                allowed_seg_chain_resnums=allowed_seg_chain_resnums,
                                distance_metric=distance_metric,
                                rmsd=rmsd, maxdist=maxdist, 
                                specific_seg_chain_resnums_only=specific_seg_chain_resnums_only,
                                filter_by_phi_psi=filter_by_phi_psi, 
                                filter_by_phi_psi_exclude_sc=filter_by_phi_psi_exclude_sc,
                                use_optimum_vdms_only=use_optimum_vdms_only, 
                                freeze_optimum_vdms=freeze_optimum_vdms,
                                bbdep=bbdep, use_hb_scores=use_hb_scores, 
                                C_score_threshold=C_score_threshold,
                                return_top_scoring_vdMs_only=return_top_scoring_vdMs_only, 
                                store_pairwise=store_pairwise, force_MC=force_MC,
                                force_DEE=force_DEE, DEE_to_MC_switch_number=DEE_to_MC_switch_number, 
                                compute_pairwise_contacts=compute_pairwise_contacts,
                                tamp_by_distance=tamp_by_distance, pair_nbr_distance=pair_nbr_distance, 
                                exponential=exponential,
                                log_logistic=log_logistic, gaussian=gaussian, relu=relu,
                                knn_contacts=knn_contacts, contact_distance_metric=contact_distance_metric,
                                use_same_rotamer_for_pairwise_contacts=use_same_rotamer_for_pairwise_contacts, 
                                use_same_rotamer_for_lig_contacts=use_same_rotamer_for_lig_contacts,
                                ignore_rmsd_column=ignore_rmsd_column,
                                pairwise_contact_weight=pairwise_contact_weight,
                                outer_shell_score_weight=outer_shell_score_weight,
                                vdW_tolerance=vdW_tolerance,
                                )
            if not keep_searching:
                break
        return poses

    def get_top_poses(self, poses=None, top=30, top_from_pose_group=True):
        if poses is None:
            poses = self.poses
        scores = [pose.opt_en for pose in poses]
        sorted_score_indices = sorted(list(range(len(scores))), key=lambda x: scores[x])
        
        if top_from_pose_group:
            if len(self.pose_groups) == 0:
                self.group_poses()
            inds_from_groups = set()
            top_poses = []
            j = 1
            for ind in sorted_score_indices:
                pose = poses[ind]
                if len(pose.opt_vdms) == 0:
                    continue
                if ind not in inds_from_groups:
                    inds_from_groups.update(self.pose_groups_dict[ind])
                    pose.rank = j
                    top_poses.append(pose)
                    j += 1
                    if j > top:
                        break
        else:
            top_vdm_names = set()
            top_poses = []
            j = 1
            for ind in sorted_score_indices:
                pose = poses[ind]
                if len(pose.opt_vdms) == 0:
                    continue
                opt_vdms_names = tuple(sorted(pose.opt_vdms_names))
                if opt_vdms_names in top_vdm_names:
                    continue
                top_vdm_names.add(opt_vdms_names)
                pose.rank = j
                top_poses.append(pose)
                j += 1
                if j > top:
                    break
        return top_poses

    @staticmethod
    def make_pose_data(poses):
        pose_data = []
        cols = ['seg_chain_resnum', 'CG_type', 
        'CG_ligand_coverage', 'score', 
        'rotamer', 'resname_rota']
        for pose in poses:
            if len(pose.opt_vdms_sidechains) == 0:
                data = make_empty_df(cols)
            else:
                data = pose.opt_vdms_sidechains[cols].drop_duplicates()
                data = data.sort_values(by=cols).reset_index(drop=True)
            pose_data.append(data)
        return pose_data
    
    def group_poses(self):
        pose_data = self.make_pose_data(self.poses)
        pose_groups = []
        in_group = set()
        for i in range(len(pose_data)):
            if i in in_group:
                continue
            pose_group = [i]
            for j in range(i + 1, len(pose_data)):
                if j in in_group:
                    continue
                if df_is_subset(pose_data[i], pose_data[j]) \
                 or df_is_subset(pose_data[j], pose_data[i]):
                    pose_group.append(j)
                    in_group.add(j)
            pose_groups.append(pose_group)
        self.pose_groups = pose_groups
        self.pose_groups_dict = {i: pose_group for pose_group in pose_groups 
                                 for i in pose_group}
        for i, pose_group in enumerate(pose_groups):
            for j in pose_group:
                self.poses[j].group_number = i


class Pose:
    def __init__(self, **kwargs):
        self.vdms = []
        self.ligand = None
        self.ligand_contacts = None
        self.C_score = 0
        self.ligand_burial = 0
        self.groupby = kwargs.get('groupby', ['CG', 'rota', 'probe_name', 'CG_type', 'seg_chain_resnum'])
        self.num_contact_csts = 0
        self.vdms_sidechains = None
        self.vdms_grs = None
        self.opt_en_sidechains = 0
        self.opt_en = 0
        self.opt_vdms_sidechains_names = []
        self.opt_vdms_names = []
        self.opt_vdms = []
        self.opt_vdms_sidechains = []
        self.opt_vdms_grs = None
        self.identifier = 0
        self.ligand_resname = None
        self.buried_unsat_sc_donor_atoms = []
        self.buried_unsat_sc_acceptor_atoms = []
        self.num_buried_unsat_donor_atoms = -1
        self.num_buried_unsat_acceptor_atoms = -1
        self.buried_unsat_lig_donor_atoms = []
        self.buried_unsat_lig_acceptor_atoms = []
        self.num_buried_unsat_lig_donor_atoms = -1
        self.num_buried_unsat_lig_acceptor_atoms = -1
        self.num_buried_unsat_sc_donor_atoms = -1
        self.num_buried_unsat_sc_acceptor_atoms = -1
        self.rank = 0
        self.vdm_stats = DataFrame()
        self.filename = ''
        self._already_scored = False
        self._frozen_sites = dict()
        self._force_MC = False
        self.group_number = -1
        self.first_shell_contact_vdms = dict()
        self.pairwise_scores = dict()
        self.cg_weights = dict()
        self.lig_additional_vdms = defaultdict(dict)
        self.poss_vdms_for_buried_unsatisfied = dict()
        self.stored_buried_unsat_gr_names = set()

    def set_ligand(self, ligand):
        self.ligand = ligand.copy()
        return self

    def score(self, template, sample, path_to_database, bbdep=True, use_hb_scores=False, C_score_threshold=0,
              pairwise_dict=None, force_MC=False, force_DEE=False, compute_pairwise_contacts=False,
              DEE_to_MC_switch_number=1000, tamp_by_distance=False, pair_nbr_distance=0.7,
              brute_force_no_DEE=False, print_DEE_time=False, exponential=False,
              log_logistic=True, gaussian=False, relu=False,
              knn_contacts=True, contact_distance_metric='rmsd',
              use_same_rotamer_for_pairwise_contacts=False, use_same_rotamer_for_lig_contacts=True,
              ignore_rmsd_column=(),
              pairwise_contact_weight=1.0,
              burial_threshold=0.5,
              outer_shell_score_weight=0.5,
              vdW_tolerance=0.1):

        if self.ligand_contacts is None:
            self.set_ligand_contacts(template, vdW_tolerance=vdW_tolerance)

        # For Future: for sidechain-sidechain interactions could look up scores via vdM centroids.
        self.find_opt(template, sample=sample, bbdep=bbdep, use_hb_scores=use_hb_scores,
                      C_score_threshold=C_score_threshold, pairwise_dict=pairwise_dict,
                      force_MC=force_MC, force_DEE=force_DEE, 
                      DEE_to_MC_switch_number=DEE_to_MC_switch_number,
                      compute_pairwise_contacts=compute_pairwise_contacts,
                      path_to_database=path_to_database,
                      tamp_by_distance=tamp_by_distance,
                      pair_nbr_distance=pair_nbr_distance,
                      brute_force_no_DEE=brute_force_no_DEE,
                      print_DEE_time=print_DEE_time, exponential=exponential,
                      log_logistic=log_logistic, gaussian=gaussian, 
                      relu=relu, knn_contacts=knn_contacts, contact_distance_metric=contact_distance_metric,
                      use_same_rotamer_for_pairwise_contacts=use_same_rotamer_for_pairwise_contacts,
                      use_same_rotamer_for_lig_contacts=use_same_rotamer_for_lig_contacts, 
                      ignore_rmsd_column=ignore_rmsd_column,
                      pairwise_contact_weight=pairwise_contact_weight,
                      burial_threshold=burial_threshold,
                      outer_shell_score_weight=outer_shell_score_weight,
                      vdW_tolerance=vdW_tolerance)
        # self.set_total_opt_en(template, bbdep=bbdep, use_hb_scores=use_hb_scores, C_score_threshold=C_score_threshold)

    def set_vdm_stats(self):
        grby = self.groupby.copy()
        grby.append('resname_rota')
        grby.append('dist_to_query')
        data = []
        for seg_chain_resnum, vdms in self.vdms.groupby('seg_chain_resnum'):
            for cg_type, vdms_cg_type in vdms.groupby('CG_type'):
                nr_vdms_cg_type = vdms_cg_type[grby].drop_duplicates()
                for resname, nr_vdms_cg_type_resname in nr_vdms_cg_type.groupby('resname_rota'):
                    min_dist = nr_vdms_cg_type_resname['dist_to_query'].min()
                    num_vdms = len(nr_vdms_cg_type_resname)
                    CG_ligand_coverage=set(nr_vdms_cg_type_resname.CG_ligand_coverage)
                    data.append((seg_chain_resnum, cg_type, resname, min_dist, num_vdms, CG_ligand_coverage))
        self.vdm_stats = DataFrame(data, columns=['seg_chain_resnum', 'CG_type',
                                                  'resname', 'min_CG_dist_to_query', 'number_vdMs',
                                                  'CG_ligand_coverage'])

    def print_vdm_stats(self):
        if len(self.vdm_stats) == 0:
            self.set_vdm_stats()
        for seg_chain_resnum, df1 in self.vdm_stats.groupby('seg_chain_resnum'):
            print(seg_chain_resnum)
            for cg_type, df2 in df1.groupby('CG_type'):
                print('    ', cg_type)
                print('        ', 'resname, min_CG_dist_to_query, number of vdMs, CG_ligand_coverage')
                for n, row in df2.iterrows():
                    print('        ', row['resname'], row['min_CG_dist_to_query'], row['number_vdMs'],
                          row['CG_ligand_coverage'])

    def set_nonclashing_vdms(self, additional_vdms=None, copy=False, vdW_tolerance=0.1):
        self.vdms.loc[:, 'unique_id'] = self.vdms['CG'].astype(str) + '_' + self.vdms['rota'].astype(str) \
                                 + '_' + self.vdms['probe_name'].astype(str) + '_' \
                                 + self.vdms['CG_type'] + '_' \
                                 + self.vdms['seg_chain_resnum'].astype('str')
        vdms_x = self.vdms[self.vdms.chain == 'X'].copy()
        if len(vdms_x) == 0:
            return
        if additional_vdms is not None:
            lig_and_additional = pd.concat([self.ligand, additional_vdms])
            cla = Clash(vdms_x, lig_and_additional, tol=vdW_tolerance)
        else:
            cla = Clash(vdms_x, self.ligand, tol=vdW_tolerance)
        cla.set_grouping('unique_id')
        cla.find(return_clash_free=True, return_clash=True)
        self.vdms = self.vdms[~self.vdms['unique_id'].isin(cla.dfq_clash['unique_id'])]
        # self.vdms.drop(columns='unique_id', inplace=True)
        self.vdms_sidechains = cla.dfq_clash_free
        if copy:
            self.vdms_sidechains = cla.dfq_clash_free.copy()
            self.vdms = self.vdms.copy()
        # self.vdms_sidechains.drop(columns='unique_id', inplace=True)
        # self.vdms_sidechains.reset_index(inplace=True, drop=True)

    # def set_nonclashing_vdms(self):
    #     vdms_x = self.vdms[self.vdms.chain == 'X'].copy()
    #     cla = Clash(vdms_x, self.ligand)
    #     cla.set_grouping(self.groupby)
    #     cla.find(return_clash_free=True, return_clash=True)
    #     self.vdms = outer_merge(self.vdms, cla.dfq_clash[self.groupby].drop_duplicates())
    #     self.vdms_sidechains = cla.dfq_clash_free

    def get_ligand_cg_coords_for_lookup(self, ligand_vdm_correspondence, CG_type, CG_group, cg_df, ignore_rmsd_column=()):
        if len(ignore_rmsd_column) > 0:
            dfs = []
            for (_cg, _cg_gr), g in ligand_vdm_correspondence.groupby(['CG_type', 'CG_group']):
                if (_cg, _cg_gr) in ignore_rmsd_column:
                    dfs.append(g)
                else:
                    dfs.append(g[g['rmsd'] == True])
            lig_vdm_corr = concat(dfs)
        elif 'rmsd' in ligand_vdm_correspondence.columns:
            lig_vdm_corr = ligand_vdm_correspondence[ligand_vdm_correspondence['rmsd'] == True]
        else:
            lig_vdm_corr = ligand_vdm_correspondence
        lig_cg_coords = dict()
        is_cg_gr = lig_vdm_corr['CG_group'] == CG_group
        is_cg_type = lig_vdm_corr['CG_type'] == CG_type
        vdm_lig_corr_cg_gr = lig_vdm_corr[is_cg_type & is_cg_gr][
                                ['resname', 'name', 'lig_resname', 'lig_name']].drop_duplicates()
        dfy = pd.merge(cg_df, vdm_lig_corr_cg_gr, on=['resname', 'name'])
        resnames = dfy.resname.unique()
        if len(resnames) > 1:
            dfy = dfy[dfy.resname == resnames[0]]
        dfy = dfy[['lig_resname', 'lig_name']]
        df_lig_cg = pd.merge(dfy, self.ligand, left_on=['lig_resname', 'lig_name'],
                                right_on=['resname', 'name'])
        lig_cg_coords = df_lig_cg[['c_x', 'c_y', 'c_z']].values
        return lig_cg_coords

    def set_ligand_contacts(self, template, vdW_tolerance=0.1):
        sc_bb = concat((self.vdms_sidechains, template.dataframe), sort=False)
        lig_con = Contact(sc_bb, self.ligand, **dict(tol=vdW_tolerance))
        lig_con.find()
        self.ligand_contacts = lig_con.df_contacts
        
    def set_vdms_satisfy_contact_csts(self, constraints):
        self.vdms['satisfies_cst'] = np.nan
        self.vdms_sidechains['satisfies_cst'] = np.nan
        for n, cst in constraints.groupby('cst_group'):
            if ((cst['Has_vdM_in_pose'].isna()) & (cst['contact_type'] == set())).any():
                # if a cst_group, each member must have contact csts or has_vdm_in_pose csts.
                continue
            for row_name, row in cst.iterrows():
                if not np.isnan(row['Has_vdM_in_pose']):
                    filt = self.vdms['CG_ligand_coverage'] == row['Has_vdM_in_pose']
                    if filt.any():
                        self.vdms.loc[filt, 'satisfies_cst'] = n
                    filt = self.vdms_sidechains['CG_ligand_coverage'] == row['Has_vdM_in_pose']
                    if filt.any():
                        self.vdms_sidechains.loc[filt, 'satisfies_cst'] = n
                if row['contact_type'] != set():
                    resname = self.ligand_contacts['resname_t'] == row['lig_resname']
                    name = self.ligand_contacts['name_t'] == row['lig_name']
                    lig_atom = self.ligand_contacts[resname & name]
                    lig_atom = lig_atom[lig_atom['contact_type'].isin(row['contact_type'])]
                    if len(lig_atom) == 0:
                        continue

                    vdms_key = self.vdms['CG'].astype(str) + '_' + self.vdms['rota'].astype(str) \
                               + '_' + self.vdms['probe_name'].astype(str) + '_' + self.vdms['seg_chain_resnum'].astype(str)
                    
                    lig_atom_key = lig_atom['CG_q'].astype(str) + '_' + lig_atom['rota_q'].astype(str) \
                               + '_' + lig_atom['probe_name_q'].astype(str) + '_' + lig_atom['seg_chain_resnum_q'].astype(str)

                    self.vdms.loc[vdms_key.isin(lig_atom_key).values, 'satisfies_cst'] = n
                    
                    vdms_sidechains_key = self.vdms_sidechains['CG'].astype(str) + '_' + self.vdms_sidechains['rota'].astype(str) \
                               + '_' + self.vdms_sidechains['probe_name'].astype(str) + '_' + self.vdms_sidechains['seg_chain_resnum'].astype(str)

                    self.vdms_sidechains.loc[vdms_sidechains_key.isin(lig_atom_key).values, 'satisfies_cst'] = n

            self.num_contact_csts += 1

    def score_vdms(self, vdms, template, bbdep=True, use_hb_scores=False, tamp_by_distance=False,
                   exponential=False, log_logistic=True, gaussian=False, relu=False, 
                   outer_shell_score_weight=0.5, burial_threshold=0.5,
                   ):
        score_threshold = 0 #-1 * C_score_threshold
        if tamp_by_distance:
            if exponential:
                tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
            elif log_logistic:
                # middle ground between exponential and gaussian
                tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
            elif gaussian:
                tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
            elif relu:
                tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        else:
            tamp_func = lambda x: 1
        for site, gr in vdms.groupby('seg_chain_resnum'):
            for vdm_name, subgroup in gr.groupby(self.groupby):
                hb = ''
                if use_hb_scores:
                    if subgroup['contact_hb'].any():
                        hb = '_hb'
                if bbdep:
                    abple = template.dataframe['ABPLE'][template.dataframe['seg_chain_resnum'] == site].iat[0]
                    score_col_name = 'C_score' + hb + '_ABPLE_' + abple
                else:
                    score_col_name = 'C_score' + hb + '_bb_ind'
                score = -1 * subgroup[score_col_name].iat[0]
                if np.isnan(score):
                    score = 10
                elif tamp_by_distance:
                    if score < score_threshold: 
                        dist = subgroup['dist_to_query'].iat[0]
                        score = score * tamp_func(dist)
                if self.cg_weights:
                    cg = subgroup['CG_type'].iat[0]
                    if cg in self.cg_weights:
                        score = score * self.cg_weights[cg]
                if 'ligand_type' in subgroup.columns:
                    if 'sc' in subgroup['ligand_type'].values:
                        subgroup_copy = subgroup.copy()
                        coords = subgroup_copy[['c_x', 'c_y', 'c_z']].values
                        subgroup_copy['dist_to_template_hull'] = template.alpha_hull.get_pnts_distance(coords)
                        w = outer_shell_score_weight
                        subgroup_copy_don = subgroup_copy[(~subgroup_copy.c_D_x.isna()) & (subgroup_copy.dist_to_template_hull > burial_threshold)]
                        if len(subgroup_copy_don) > 0:
                            subgroup_copy_don = subgroup_copy_don[~subgroup_copy_don.apply(get_heavy, axis=1)]
                        num_don = len(subgroup_copy_don)
                        subgroup_copy_acc = subgroup_copy[(~subgroup_copy.c_A1_x.isna()) & (subgroup_copy.dist_to_template_hull > burial_threshold)]
                        if len(subgroup_copy_acc) > 0:
                            subgroup_copy_acc = subgroup_copy_acc[subgroup_copy_acc.apply(get_heavy, axis=1)]
                            num_acc = subgroup_copy_acc.c_D_x.isna().sum() + len(subgroup_copy_acc) # if an acc is not also donating, count it twice.
                                                                                                # e.g. Asp num_acc=4, Ser num_acc=1
                        else:
                            num_acc = 0
                        # could add step to actually compute num_buns here!
                        total_num_polars = max(num_don + num_acc, 1) # so as not to divide by 0 below!
                        score = score * w * 1/total_num_polars # should favor solutions with few polar atoms
                for index in subgroup.index:
                    vdms.at[index, 'score'] = score
                # vdms.loc[subgroup.index, 'score'] = score

    def _get_contacting_cgs(self, all_lig_names, cg_alts, atom_cg_map_by_atomtype, cg_atom_map, ligand_atom_types):
        polars = {'h_pol', 'n', 'o', 's'}
        not_wc = self.ligand_contacts.contact_type != 'wc'
        q_cols = [col + '_q' for col in self.groupby]
        contacting_cgs = defaultdict(dict)
        contacting_cgs_vdm_info = dict()
        for n, g in self.ligand_contacts[not_wc].groupby(q_cols):
            v = self.vdms_grs.get_group(n)
            resn = v.resname_rota.iat[0]
            if resn not in ['ALA', 'GLY']:
                rotamer_df = rotamer_dfs[resn]
                df_rotamer_coords = merge(rotamer_df, v, on='name')
                rotamer_coords = df_rotamer_coords[['c_x', 'c_y', 'c_z']].values
            else:
                rotamer_coords = None

            cg = g.CG_type_q.iat[0]
            cg_gr = g.CG_group_q.iat[0]
            lig_names = set(g['name_t']) & all_lig_names
            num_contacts = len(lig_names)
            non_cg_contacts = lig_names - cg_atom_map[(cg, cg_gr)]
            cg_contacts = lig_names & cg_atom_map[(cg, cg_gr)]
            for name in non_cg_contacts:
                group = set()
                for cg2, cg_gr2 in atom_cg_map_by_atomtype[name] - {(cg, cg_gr)}:
                    if (cg2, cg_gr2) in cg_alts[(cg, cg_gr)]:
                        continue
                    # if cgj shares majority of contacting atoms with cgi and
                    # extra contacting atoms are minority and nonpolar, skip cgj
                    cg_2_contacts = cg_atom_map[(cg2, cg_gr2)] & lig_names
                    if len(cg_2_contacts - cg_contacts) / num_contacts < 0.5:
                        if ligand_atom_types[name] not in polars:
                            continue
                        else:
                            group.add((cg2, cg_gr2))
                    else:
                        group.add((cg2, cg_gr2))

                # remove cgs/cg_grs that are subsets of another cg/cg_gr
                # e.g if bb_cnh is a subset of conh2, but only want to
                # remove bb_cnh is there is a contact to the conh2 oxygen as well
                # as the nitrogen.  Hoping the code below does this...
                new_group = set()
                for _cg, _cg_gr in group.copy():
                    _cg_contacts = lig_names & cg_atom_map[(_cg, _cg_gr)]
                    len_cg_contacts = len(_cg_contacts)
                    do_add = True
                    for _cg2, _cg_gr2 in (group.copy() - {(_cg, _cg_gr)}):
                        _cg_2_contacts = cg_atom_map[(_cg2, _cg_gr2)] & lig_names
                        if len(_cg_2_contacts) > len_cg_contacts and len(_cg_contacts - _cg_2_contacts) == 0:
                            do_add = False
                            break
                    if do_add:
                        new_group.add((_cg, _cg_gr))

                # new_group = set()
                # contact_sets = []
                # for _cg, _cg_gr in group.copy():
                #     _non_cg_contacts = lig_names - cg_atom_map[(_cg, _cg_gr)]
                #     _cg_contacts = lig_names & cg_atom_map[(_cg, _cg_gr)]
                #     # _non_cg_contacts = _non_cg_contacts & non_cg_contacts
                #     # if len(_non_cg_contacts) == 0:
                #     #     new_group.add((_cg, _cg_gr))
                #     for _name in _non_cg_contacts:
                #         for _cg2, _cg_gr2 in (group.copy() - {(_cg, _cg_gr)}) & atom_cg_map_by_atomtype[_name]:
                #             _cg_2_contacts = cg_atom_map[(_cg2, _cg_gr2)] & lig_names
                #             if any([cs - _cg_2_contacts == set() for cs in contact_sets]):
                #                 continue
                #             if _cg_2_contacts == _cg_contacts:
                #                 new_group.add((_cg2, _cg_gr2))
                #                 contact_sets.append(_cg_2_contacts)
                #                 continue
                #             elif len(_cg_2_contacts - _cg_contacts) == 0:
                #                 if (_cg2, _cg_gr2) in new_group:
                #                     new_group.remove((_cg2, _cg_gr2))
                #                 continue
                #             cont = False
                #             for (alt_cg2, alt_cg_gr2) in cg_alts[(_cg2, _cg_gr2)]:
                #                 _cg_2_contacts_alt = cg_atom_map[(alt_cg2, alt_cg_gr2)] & lig_names
                #                 if _cg_2_contacts == _cg_contacts:
                #                     break
                #                 elif len(_cg_2_contacts_alt - _cg_contacts) == 0:
                #                     cont = True
                #                     break
                #             if cont:
                #                 if (_cg2, _cg_gr2) in new_group:
                #                     new_group.remove((_cg2, _cg_gr2))
                #                 continue
                #             # elif len(_cg_2_contacts - _cg_contacts) / num_contacts < 0.5:
                #             #     if ligand_atom_types[_name] not in polars:
                #             #         if (_cg2, _cg_gr2) in new_group:
                #             #             new_group.remove((_cg2, _cg_gr2))
                #             #         continue
                #             #     else:
                #             #         new_group.add((_cg2, _cg_gr2))
                #             #         contact_sets.append(_cg_2_contacts)
                #             # else:
                #             new_group.add((_cg2, _cg_gr2))
                #             contact_sets.append(_cg_2_contacts)
                # new_group = group
                if len(new_group) > 0:
                    contacting_cgs[n][name] = new_group
            contacting_cgs_vdm_info[n] = (g.resname_rota.iat[0], rotamer_coords)
        return contacting_cgs, contacting_cgs_vdm_info

    # def _get_contacting_cgs(self, all_lig_names, cg_alts, atom_cg_map_by_atomtype, cg_atom_map):
    #     not_wc = self.ligand_contacts.contact_type != 'wc'
    #     q_cols = [col + '_q' for col in self.groupby]
    #     contacting_cgs = dict()
    #     contacting_cgs_vdm_info = dict()
    #     for n, g in self.ligand_contacts[not_wc].groupby(q_cols):
    #         cg = g.CG_type_q.iat[0]
    #         cg_gr = g.CG_group_q.iat[0]
    #         lig_names = set(g['name_t']) & all_lig_names
    #         non_cg_contacts = lig_names - cg_atom_map[(cg, cg_gr)]
    #         lig_cgs = set()
    #         for name in non_cg_contacts:
    #             for cg2, cg_gr2 in atom_cg_map_by_atomtype[name] - {(cg, cg_gr)}:
    #                 if (cg2, cg_gr2) in cg_alts[(cg, cg_gr)]:
    #                     continue
    #                 else:
    #                     lig_cgs.add((cg2, cg_gr2))
    #         grouped_lig_cgs = []
    #         already_added = set()
    #         for tup in lig_cgs.copy():
    #             if tup in already_added:
    #                 continue
    #             tup_group = [tup]
    #             already_added.add(tup)
    #             for tup_alt in cg_alts[tup]:
    #                 if tup_alt in already_added:
    #                     continue
    #                 if tup_alt in lig_cgs:
    #                     tup_group.append(tup_alt)
    #                     already_added.add(tup_alt)
    #             grouped_lig_cgs.append(tup_group)
    #         contacting_cgs[n] = grouped_lig_cgs
    #         contacting_cgs_vdm_info[n] = (g.resname_rota.iat[0], g.rotamer.iat[0])
    #     return contacting_cgs, contacting_cgs_vdm_info

    # def _get_contacting_cgs(self, all_lig_names, cg_alts, atom_cg_map, cg_atom_map):
    #     not_wc = self.ligand_contacts.contact_type != 'wc'
    #     q_cols = [col + '_q' for col in self.groupby]
    #     contacting_cgs = dict()
    #     contacting_cgs_vdm_info = dict()
    #     for n, g in self.ligand_contacts[not_wc].groupby(q_cols):
    #         cg = g.CG_type_q.iat[0]
    #         cg_gr = g.CG_group_q.iat[0]
    #         lig_names = set(g['name_t']) & all_lig_names
    #         non_cg_contacts = lig_names - cg_atom_map[(cg, cg_gr)]
    #         lig_cgs = set()
    #         for name in non_cg_contacts:
    #             for cg2, cg_gr2 in atom_cg_map[name] - {(cg, cg_gr)}:
    #                 if (cg2, cg_gr2) in cg_alts[(cg, cg_gr)]:
    #                     continue
    #                 else:
    #                     lig_cgs.add((cg2, cg_gr2))
    #         grouped_lig_cgs = []
    #         already_added = set()
    #         for tup in lig_cgs.copy():
    #             if tup in already_added:
    #                 continue
    #             tup_group = [tup]
    #             already_added.add(tup)
    #             for tup_alt in cg_alts[tup]:
    #                 if tup_alt in already_added:
    #                     continue
    #                 if tup_alt in lig_cgs:
    #                     tup_group.append(tup_alt)
    #                     already_added.add(tup_alt)
    #             grouped_lig_cgs.append(tup_group)
    #         contacting_cgs[n] = grouped_lig_cgs
    #         contacting_cgs_vdm_info[n] = (g.resname_rota.iat[0], g.rotamer.iat[0])
    #     return contacting_cgs, contacting_cgs_vdm_info

    def find_contact_vdms(self, template, vdms, path_to_nbrs_database, distance_metric='rmsd', distance_cutoff=0.5,
                     same_rotamer=True, bb_dep=0, hb_only_score=False):
        contact_vdms = defaultdict(list)
        # if 'opt_vdms_grs' not in self.__dict__.keys() or self.opt_vdms_grs is None:
        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
        path_to_nbrs_database_scores = path_to_nbrs_database + 'vdMs_cg_nbrs_scores/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'
        for cg in vdms.keys():
            if cg == 'ccoh':
                continue
            print(cg)
            if cg not in os.listdir(path_to_nbrs_database_):
                print('\t CG not found in database. Skipping...')
                continue
            cg_df = cg_dfs[cg]
            for aa in vdms[cg].keys():
                print('\t', aa)
                if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                    print('\t\t AA not found in CG database. Skipping...')
                    continue
                if distance_metric == 'rmsd':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    num_cg_atoms = nbrs._fit_X.shape[1] / 3 
                    radius = np.sqrt(num_cg_atoms) * distance_cutoff
                elif distance_metric == 'maxdist':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    radius = distance_cutoff
                scores = pd.read_parquet(path_to_nbrs_database_scores + cg + '/' + aa + '.parquet.gzip')
                score_col_dict = {colname: i for i, colname in enumerate(scores.columns)}
                scores = scores.values
                with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                    groupnames = pickle.load(f)
                for rotamer, rota_info, cg_info in vdms[cg][aa]:
                    seg_aa, chain_aa, res_aa = rota_info[0][-1]
                    if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                        df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                                (template.dataframe['chain'] == chain_aa) &
                                                (template.dataframe['resnum'] == res_aa)].copy()
                        m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                        for _name in ['N', 'CA', 'C']])
                        t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                        R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                        template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com)
                    
                    for vdm_cg in cg_info:
                        vdm = self.vdms_grs.get_group(vdm_cg)
                        dfy = pd.merge(cg_df, vdm[vdm.chain=='X'], on=['name', 'resnum', 'resname'])
                        coords = dfy[['c_x', 'c_y', 'c_z']].values
                        R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]
                        dfy[['c_x', 'c_y', 'c_z']] = apply_transform(R, mob_com, targ_com, coords)
                        coords = dfy[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)
                        dists, inds = nbrs.radius_neighbors(coords, radius=radius)
                        dists, inds = dists[0], inds[0]
                        if cg in cgs_that_flip:
                            dfy['chain'] = dfy['chain_x']
                            coords = flip_cg_coords(dfy).reshape(1, -1)
                            dists_flip, inds_flip = nbrs.radius_neighbors(coords, radius=radius)
                            dists_flip, inds_flip = dists_flip[0], inds_flip[0]
                            if dists_flip.size > 0 and dists.size > 0:
                                inds = np.concatenate((inds, inds_flip))
                                inds, index = np.unique(inds, return_index=True)
                                dists = np.concatenate((dists, dists_flip))[index]
                            elif dists_flip.size > 0:
                                dists = dists_flip
                                inds = inds_flip
                        if dists.size > 0:
                            ss = ''
                            if bb_dep == 1:
                                res_filter = template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)
                                abple = template.dataframe[res_filter]['ABPLE'].iat[0]
                                ss = 'ABPLE_' + abple
                            elif bb_dep == 0:
                                ss = 'bb_ind'
                            score_col = 'C_score_' + ss
                            score_col_ind = score_col_dict[score_col]
                            scores_ = scores[inds, score_col_ind].astype(float)
                            not_na = ~np.isnan(scores_)
                            scores_ = scores_[not_na]
                            inds = inds[not_na]
                            dists = dists[not_na]
                            if len(scores_) == 0:
                                continue
                            if hb_only_score:
                                is_hb = scores[inds, score_col_dict['hbond']].astype(bool)
                                if ~is_hb.any():
                                    continue
                                inds = inds[is_hb]
                                dists = dists[is_hb]
                                score_col = 'C_score_hb_' + ss
                                score_col_ind = score_col_dict[score_col]
                                scores_ = scores[inds, score_col_ind]
                            if same_rotamer:
                                rotamers = scores[inds, score_col_dict['rotamer']]
                                rotamer_filter = rotamers == rotamer
                                scores_ = scores_[rotamer_filter]
                                inds = inds[rotamer_filter]
                                dists = dists[rotamer_filter]
                            if len(scores_) == 0:
                                continue
                            contact_type = scores[inds, score_col_dict['contact_type']]
                            ct_filter = contact_type == 'sc'
                            scores_ = scores_[ct_filter]
                            inds = inds[ct_filter]
                            dists = dists[ct_filter]
                            if len(scores_) == 0:
                                continue
                            ind_best_score = np.argmax(scores_)
                            dist_best_score = dists[ind_best_score]
                            if distance_metric == 'rmsd':
                                dist_best_score = dist_best_score / np.sqrt(num_cg_atoms)
                            best_score = scores_[ind_best_score]
                            best_groupname = groupnames[inds[ind_best_score]]
                            contact_vdms[(rota_info, cg_info)].append((best_score, dist_best_score, vdm_cg, (cg, aa, best_groupname)))
                        contact_vdms[(rota_info, cg_info)] = sorted(contact_vdms[(rota_info, cg_info)], key=lambda x: x[0], reverse=True)[0]
        best_contact_vdms = {}
        for key_ in contact_vdms.keys():
            best_contact_vdms[key_] = sorted(contact_vdms[key_], key=lambda x: x[0], reverse=True)[0]
        return best_contact_vdms 

    def find_contact_vdms_knn(self, template, vdms, rotamers, path_to_nbrs_database, distance_metric='rmsd',
                        same_rotamer=False, bb_dep=0, use_hb_scores=False):
        # print('Finding first-shell contact vdms...')
        contact_vdms = defaultdict(list)
        if same_rotamer:
            num_neighbors = 1000
        else:
            num_neighbors = 1000
        # if 'opt_vdms_grs' not in self.__dict__.keys() or self.opt_vdms_grs is None:
        #     self.opt_vdms_grs = self.opt_vdms.groupby(self.groupby)
        self.vdms_grs = self.vdms_sidechains.groupby(self.groupby)
        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
            path_to_rotamer_nbrs_database_ = path_to_nbrs_database + 'vdMs_rotamers_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
            path_to_rotamer_nbrs_database_ = path_to_nbrs_database + 'vdMs_rotamers_nbrs_maxdist/'
        path_to_nbrs_database_scores = path_to_nbrs_database + 'vdMs_cg_nbrs_scores/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'
        print('pairwise contacts...')
        for cg in vdms.keys():
            if cg == 'ccoh':
                continue
            print('\t', cg)
            if cg not in os.listdir(path_to_nbrs_database_):
                print('\t\t CG not found in database. Skipping...')
                continue
            cg_df = cg_dfs[cg]
            for aa in vdms[cg].keys():
                if aa in ['ALA', 'GLY']:
                    has_rotamer = False
                else:
                    has_rotamer = True
                if has_rotamer:
                    rotamer_df = rotamer_dfs[aa]
                print('\t\t', aa)
                if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                    print('\t\t\t AA not found in CG database. Skipping...')
                    continue
                if distance_metric == 'rmsd':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    num_cg_atoms = nbrs._fit_X.shape[1] / 3 
                    if same_rotamer and has_rotamer:
                        with open(path_to_rotamer_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            rotamer_nbrs = pickle.load(f)
                        num_rotamer_atoms = rotamer_nbrs._fit_X.shape[1] / 3 
                elif distance_metric == 'maxdist':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    if same_rotamer and has_rotamer:
                        with open(path_to_rotamer_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            rotamer_nbrs = pickle.load(f)

                _num_nbrs = min(nbrs._fit_X.shape[0], num_neighbors)
                scores = pd.read_parquet(path_to_nbrs_database_scores + cg + '/' + aa + '.parquet.gzip')
                score_col_dict = {colname: i for i, colname in enumerate(scores.columns)}
                scores = scores.values
                with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                    groupnames = pickle.load(f)
                for rota_info, cg_info in vdms[cg][aa]:
                    seg_aa, chain_aa, res_aa = rota_info[0][-1]
                    if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                        df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                                (template.dataframe['chain'] == chain_aa) &
                                                (template.dataframe['resnum'] == res_aa)].copy()
                        m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                        for _name in ['N', 'CA', 'C']])
                        t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                        R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                        template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com)
                    
                    for vdm_cg in cg_info:
                        vdm = self.vdms_grs.get_group(vdm_cg)
                        dfy = pd.merge(cg_df, vdm[vdm.chain=='X'], on=['name', 'resnum', 'resname'])
                        coords = dfy[['c_x', 'c_y', 'c_z']].values
                        R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]
                        dfy[['c_x', 'c_y', 'c_z']] = apply_transform(R, mob_com, targ_com, coords)
                        coords = dfy[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)
                        try:
                            dists, inds = nbrs.kneighbors(coords, n_neighbors=_num_nbrs, return_distance=True)
                        except Exception as e:
                            print(e)
                            print('The pose possibily contains duplicate vdMs.  Perhaps \
                                  check the CG_ligand_coverage column of the ligand.txt file for \
                                  overlapping atoms in different CG_ligand_coverage groups.')
                            print('culprit vdM:', cg, aa, vdm_cg)
                            continue
                        dists, inds = dists[0], inds[0]
                        if cg in cgs_that_flip:
                            dfy['chain'] = dfy['chain_x']
                            coords = flip_cg_coords(dfy).reshape(1, -1)
                            dists_flip, inds_flip = nbrs.kneighbors(coords, n_neighbors=_num_nbrs, return_distance=True)
                            dists_flip, inds_flip = dists_flip[0], inds_flip[0]
                            if dists_flip.size > 0 and dists.size > 0:
                                inds = np.concatenate((inds, inds_flip))
                                inds, index = np.unique(inds, return_index=True)
                                dists = np.concatenate((dists, dists_flip))[index]
                            elif dists_flip.size > 0:
                                dists = dists_flip
                                inds = inds_flip
                        if dists.size > 0: # There should always be at least one neighbor with kneighbors so this doesn't do anything...
                            contact_type = scores[inds, score_col_dict['contact_type']]
                            ct_filter = contact_type == 'sc'
                            inds = inds[ct_filter]
                            dists = dists[ct_filter]
                            if len(inds) == 0:  # only consider sidechain-only interactions
                                continue
                            ss = ''
                            if bb_dep == 1:
                                res_filter = template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)
                                abple = template.dataframe[res_filter]['ABPLE'].iat[0]
                                ss = 'ABPLE_' + abple
                            elif bb_dep == 0:
                                ss = 'bb_ind'
                            score_col = 'C_score_' + ss
                            score_col_ind = score_col_dict[score_col]
                            scores_ = scores[inds, score_col_ind].astype(float)
                            not_na = ~np.isnan(scores_)
                            scores_ = scores_[not_na]
                            inds = inds[not_na]
                            dists = dists[not_na]
                            if len(scores_) == 0:  # No ss score for vdm so give it the lowest ss score possible.
                                min_score = min(-1, scores[:, score_col_ind].min())
                                large_distance = 10
                                contact_vdms[(rota_info, cg_info)].append((min_score, large_distance, 'NA', (cg, aa, 'NA')))
                                continue
                            if use_hb_scores:
                                is_hb = scores[inds, score_col_dict['hbond']].astype(bool)
                                if is_hb.any():
                                #     continue
                                # inds = inds[is_hb]
                                # dists = dists[is_hb]
                                    score_col = 'C_score_hb_' + ss
                                    score_col_ind = score_col_dict[score_col]
                                    scores_[is_hb] = scores[inds[is_hb], score_col_ind]
                                    not_na = ~np.isnan(scores_)
                                    scores_ = scores_[not_na]
                                    inds = inds[not_na]
                                    dists = dists[not_na]
                                    if len(scores_) == 0:
                                        min_score = min(-1, scores[:, score_col_ind].min())
                                        large_distance = 10
                                        contact_vdms[(rota_info, cg_info)].append((min_score, large_distance, 'NA', (cg, aa, 'NA')))
                                        continue
                            # if same_rotamer:
                            #     rotamer_ind = score_col_dict['rotamer']
                            #     rotamers = scores[inds, rotamer_ind].astype(str)
                            #     rotamer_filter = rotamers == rotamer
                            #     scores_ = scores_[rotamer_filter]
                            #     inds = inds[rotamer_filter]
                            #     dists = dists[rotamer_filter]
                            #     if len(scores_) == 0:
                            #         min_score = min(-1, scores[:, score_col_ind].min())
                            #         large_distance = 10
                            #         contact_vdms[(rota_info, cg_info)].append((min_score, large_distance, 'NA', (cg, aa, 'NA')))
                            #         continue

                            if same_rotamer and has_rotamer:
                                # index_rotamer = groupnames.index(tuple(vdm_gr_name[:-2]))
                                # coords_rotamer = rotamer_nbrs._fit_X[index_rotamer].reshape(-1, 3)
                                coords_rotamer = rotamers[cg][aa][(rota_info, cg_info)]
                                coords_rotamer = apply_transform(R, mob_com, targ_com, coords_rotamer)
                                coords_rotamer = coords_rotamer.reshape(1, -1)
                                try:
                                    dists_rot, inds_rot = rotamer_nbrs.radius_neighbors(coords_rotamer, return_distance=True)
                                except Exception:
                                    print('The pose possibily contains duplicate vdMs. Problems with looking up rotamer.')
                                    print('culprit vdM:', cg, aa, vdm_cg)
                                    continue
                                dists_rot, inds_rot = dists_rot[0], inds_rot[0]

                                if aa in flip_dict:
                                    coords_rotamer = flip_coords_from_reference_df(coords_rotamer.reshape(-1,3), aa, rotamer_df)
                                    coords_rotamer = coords_rotamer.reshape(1, -1)
                                    try:
                                        dists_rot_flip, inds_rot_flip = rotamer_nbrs.radius_neighbors(coords_rotamer, return_distance=True)
                                    except Exception:
                                        print('The pose possibily contains duplicate vdMs. Problems with looking up rotamer.')
                                        print('culprit vdM:', cg, aa, vdm_cg)
                                        continue
                                    dists_rot_flip, inds_rot_flip = dists_rot_flip[0], inds_rot_flip[0]
                                    if len(inds_rot) > 0 and len(inds_rot_flip) > 0:
                                        inds_rot, uniq_inds = np.unique(np.concatenate((inds_rot, inds_rot_flip)), return_index=True)
                                        dists_rot = np.concatenate((dists_rot, dists_rot_flip))[uniq_inds]

                            if same_rotamer and has_rotamer:
                                if len(inds_rot) == 0:
                                    print('No rotamer nbrs found for', cg, aa, vdm_cg)
                                    min_score = min(-1, scores[:, score_col_ind].min())
                                    large_distance = 10
                                    contact_vdms[(rota_info, cg_info)].append((min_score, large_distance, 'NA', (cg, aa, 'NA')))
                                    continue
                                inds_tf = np.in1d(inds, inds_rot)
                                dists = dists[inds_tf]
                                inds = inds[inds_tf]
                                scores_ = scores_[inds_tf]
                                if len(inds) == 0:
                                    print('No rotamer nbrs found for', cg, aa, vdm_cg)
                                    min_score = min(-1, scores[:, score_col_ind].min())
                                    large_distance = 10
                                    contact_vdms[(rota_info, cg_info)].append((min_score, large_distance, 'NA', (cg, aa, 'NA')))
                                    continue

                            ind_lowest_dist = np.argmin(dists)
                            dist_lowest = dists[ind_lowest_dist]
                            if distance_metric == 'rmsd':
                                dist_lowest = dist_lowest / np.sqrt(num_cg_atoms)
                            score_lowest_dist = scores_[ind_lowest_dist]
                            best_groupname = groupnames[inds[ind_lowest_dist]]
                            contact_vdms[(rota_info, cg_info)].append((score_lowest_dist, dist_lowest, vdm_cg, (cg, aa, best_groupname)))
        best_contact_vdms = {}
        for key_ in contact_vdms.keys():
            # best_contact_vdms[key_] = sorted(contact_vdms[key_], key=lambda x: x[0], reverse=True)[0]
            best_contact_vdms[key_] = sorted(contact_vdms[key_], key=lambda x: x[1])[0]
        return best_contact_vdms  

    def get_pairwise(self, d, ep, key1, key2, add_cst_score, vdW_tolerance=0.1):
        if key1 in ep and key2 in ep[key1]:
            return ep[key1][key2]
        if key1[1] is None or key2[1] is None:
            ep[key1][key2] = 0
            ep[key2][key1] = 0
            return ep[key1][key2]
        if d[key1[0]][key1[1]]['resname_rota'].iat[0] == 'GLY' or \
                d[key2[0]][key2[1]]['resname_rota'].iat[0] == 'GLY':
            cla = Pose()
            cla.dfq_clash_free = [1]  # a convoluted way to get a "clash object" with dfq_clash_free of len 1
        else:
            cla = Clash(d[key1[0]][key1[1]], d[key2[0]][key2[1]], **dict(tol=vdW_tolerance))
            cla.set_grouping(self.groupby)
            cla.find()
        if add_cst_score:
            s1 = set(d[key1[0]][key1[1]]['satisfies_cst']) - {np.nan}
            s2 = set(d[key2[0]][key2[1]]['satisfies_cst']) - {np.nan}
            lens12 = len(s1 & s2)
        else:
            lens12 = 0
        if len(cla.dfq_clash_free) == 0:
            ep[key1][key2] = 10
            ep[key2][key1] = 10
        elif add_cst_score and lens12 > 0:
            ep[key1][key2] = 0 #100 * lens12
            ep[key2][key1] = 0 #100 * lens12
        else:
            ep[key1][key2] = 0
            ep[key2][key1] = 0
        return ep[key1][key2]

    def set_vdms_list(self, vdms, cg_alts, cg_atom_map, all_lig_names):
        rotamer_rmsd = 0.4
        vdms_list = defaultdict(list)
        sorted_vdmnames = set()
        if 'ligand_type' in vdms.columns:
            if 'sc' in vdms['ligand_type'].values:
                is_outer_shell = vdms['ligand_type'] == 'sc'
                vdms_sc = vdms[is_outer_shell]
                vdms = vdms[~is_outer_shell]
                _grs = vdms_sc.groupby(['seg_chain_resnum', 'resname_rota'], dropna=False)
                for (seg_chain_resnum, resname_rota), g1 in _grs:
                    g2 = g1.groupby(self.groupby)
                    if len(g2.groups.keys()) > 1:
                        rotamer_dict = defaultdict(set)
                        ligand_seg_ch_rn_dict = dict()
                        if resname_rota not in ['ALA', 'GLY']:
                            rotamer_df = rotamer_dfs[resname_rota]
                            num_atoms = len(rotamer_df)
                            all_coords = []
                            vdmnames = []
                            for vdmname, vdm in g2:
                                ligand_seg_ch_rn_dict[vdmname] = vdm['ligand_seg_chain_resnum'].iat[0]
                                vdmnames.append(vdmname)
                                _vdm = merge(rotamer_df, vdm, on='name')
                                coords = _vdm[['c_x', 'c_y', 'c_z']].values.flatten()
                                # print('resname', resname_rota, vdmname)
                                # print('coords1', coords)
                                # print(coords.shape)
                                all_coords.append(coords)
                                if resname_rota in flip_dict:
                                    coords = flip_coords_from_reference_df(coords.reshape(-1,3), resname_rota, rotamer_df)
                                    coords = coords.flatten()
                                    vdmnames.append(vdmname)
                                    # print('coords2', coords)
                                    # print(coords.shape)
                                    all_coords.append(coords)
                            try:
                                all_coords = np.array(all_coords)
                                nbrs = NearestNeighbors(radius=np.sqrt(num_atoms)*rotamer_rmsd).fit(all_coords)
                            except:
                                print('*************************************')
                                for vdmname, vdm in g2:
                                    print(vdmname)
                                    print(vdm)
                                    print('len vdm:', len(vdm))
                                print('*************************************')
                                continue

                            for vdmname, coords in zip(vdmnames, all_coords):
                                _, inds = nbrs.radius_neighbors(coords.reshape(1, -1))
                                for ind in inds[0]:
                                    if vdmnames[ind] != vdmname:
                                        rotamer_dict[vdmname].add(vdmnames[ind])
                        else:
                            for vdmname, vdm in g2:
                                ligand_seg_ch_rn_dict[vdmname] = vdm['ligand_seg_chain_resnum'].iat[0]
                            all_vdmnames = set(g2.groups.keys())
                            for vdmname in all_vdmnames:
                                rotamer_dict[vdmname] = all_vdmnames - {vdmname}
                        for vdmname, vdm in g2:
                            sorted_vdmnames.add(vdmname)
                            # works if sidechains are the same kind at each seg_chain_resnum
                            # so this works for frozen opt_vdms as "ligands"
                            seen_seg_ch_rns = dict()
                            scrn = ligand_seg_ch_rn_dict[vdmname]
                            for vdmname2 in rotamer_dict[vdmname]:
                                lig_scrn = ligand_seg_ch_rn_dict[vdmname2]
                                if lig_scrn != scrn:
                                    vdm2 = g2.get_group(vdmname2)
                                    if lig_scrn not in seen_seg_ch_rns:
                                        seen_seg_ch_rns[lig_scrn] = vdm2
                                    else:
                                        score_old = seen_seg_ch_rns[lig_scrn]['score'].iat[0]
                                        score_new = vdm2['score'].iat[0]
                                        if score_new < score_old:
                                            seen_seg_ch_rns[lig_scrn] = vdm2
                                        elif score_new == score_old:
                                            dist_old = seen_seg_ch_rns[lig_scrn]['dist_to_query'].iat[0]
                                            dist_new = vdm2['dist_to_query'].iat[0]
                                            if dist_new < dist_old:
                                                seen_seg_ch_rns[lig_scrn] = vdm2
                            if len(seen_seg_ch_rns.keys()) > 0:
                                _vdms = list(seen_seg_ch_rns.values())
                                _vdms.append(vdm)
                                vdm_concat = fast_concat(_vdms)
                                vdm_concat = vdm_concat.drop_duplicates()
                                vdms_list[seg_chain_resnum].append(vdm_concat)
                            else:
                                vdms_list[seg_chain_resnum].append(vdm)
                    else:
                        for vdmname, vdm in g2:
                            sorted_vdmnames.add(vdmname)
                            vdms_list[seg_chain_resnum].append(vdm)

        not_wc = self.ligand_contacts.contact_type != 'wc'
        q_cols = [col + '_q' for col in self.groupby]
        lig_contact_grs = self.ligand_contacts[not_wc].groupby(q_cols)
        _grs = vdms.groupby(['seg_chain_resnum', 'resname_rota'], dropna=False)
        for (seg_chain_resnum, resname_rota), g1 in _grs:
            # find out if which are the same rotamers
            # and, among these, find out which have independent cgs, cg_grs
            g2 = g1.groupby(self.groupby)

            if len(g2.groups.keys()) > 1:
                cg_cg_gr_dict = dict()
                for vdmname, vdm in g2:
                    cg = vdm.CG_type.iat[0]
                    cg_gr = vdm.CG_group.iat[0]
                    cg_cg_gr_dict[vdmname] = (cg, cg_gr)

                rotamer_dict = defaultdict(set)
                if resname_rota not in ['ALA', 'GLY']:
                    rotamer_df = rotamer_dfs[resname_rota]
                    num_atoms = len(rotamer_df)
                    all_coords = []
                    vdmnames = []
                    for vdmname, vdm in g2:
                        vdmnames.append(vdmname)
                        _vdm = merge(rotamer_df, vdm, on='name')
                        coords = _vdm[['c_x', 'c_y', 'c_z']].values.flatten()
                        # print('resname', resname_rota, vdmname)
                        # print('coords1', coords)
                        # print(coords.shape)
                        all_coords.append(coords)
                        if resname_rota in flip_dict:
                            try:
                                coords = flip_coords_from_reference_df(coords.reshape(-1,3), resname_rota, rotamer_df)
                                coords = coords.flatten()
                                vdmnames.append(vdmname)
                                # print('coords2', coords)
                                # print(coords.shape)
                                all_coords.append(coords)
                            except Exception as e:
                                print(e)
                                print(coords)
                    try:
                        all_coords = np.array(all_coords)
                        nbrs = NearestNeighbors(radius=np.sqrt(num_atoms)*rotamer_rmsd).fit(all_coords)
                    except Exception as e:
                        print(e)
                        print(all_coords)
                        continue

                    for vdmname, coords in zip(vdmnames, all_coords):
                        _, inds = nbrs.radius_neighbors(coords.reshape(1, -1))
                        for ind in inds[0]:
                            if vdmnames[ind] != vdmname:
                                rotamer_dict[vdmname].add(vdmnames[ind])
                else:
                    all_vdmnames = set(g2.groups.keys())
                    for vdmname in all_vdmnames:
                        rotamer_dict[vdmname] = all_vdmnames - {vdmname}

                for vdmname, vdm in g2:
                    if vdmname in sorted_vdmnames:
                        continue
                    seen_cg_cg_grs = dict()
                    cg, cg_gr = cg_cg_gr_dict[vdmname]
                    cg_atoms = cg_atom_map[(cg, cg_gr)]
                    if vdmname in lig_contact_grs.groups:
                        has_contacts1 = True
                    else:
                        has_contacts1 = False
                    if has_contacts1:
                        contacts1 = lig_contact_grs.get_group(vdmname)
                        lig_names1 = set(contacts1['name_t']) & all_lig_names
                        cg_contacts1 = lig_names1 & cg_atoms

                    if 'satisfies_cst' in vdms.columns:
                        cst_1 = vdm['satisfies_cst'].iat[0]

                    for vdmname2 in rotamer_dict[vdmname]: 
                        if vdmname in sorted_vdmnames:
                            continue
                        cg2, cg_gr2 = cg_cg_gr_dict[vdmname2]
                        if (cg, cg_gr) == (cg2, cg_gr2):
                            continue
                        
                        if (cg2, cg_gr2) in seen_cg_cg_grs:
                            continue
                        cg2_atoms = cg_atom_map[(cg2, cg_gr2)]
                        if len(cg2_atoms - cg_atoms) == 0:
                            continue
                        if len(cg_atoms - cg2_atoms) == 0:
                            continue
                        stop = False
                        for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                            if (cg, cg_gr) == (_cg2, _cg_gr2):
                                stop = True
                                break
                            _cg2_atoms = cg_atom_map[(_cg2, _cg_gr2)]
                            if len(cg_atoms - _cg2_atoms) == 0:
                                stop = True
                                break
                            if len(_cg2_atoms - cg_atoms) == 0:
                                stop = True
                                break
                            if (_cg2, _cg_gr2) in seen_cg_cg_grs:
                                stop = True
                                break
                        if stop:
                            continue

                        if has_contacts1: 
                            if vdmname2 in lig_contact_grs.groups:
                                contacts2 = lig_contact_grs.get_group(vdmname2)
                                lig_names2 = set(contacts2['name_t']) & all_lig_names
                                cg_contacts2 = lig_names2 & cg2_atoms
                                if len(cg_contacts1 - cg_contacts2) == 0:
                                    continue
                                if len(cg_contacts2 - cg_contacts1) == 0:
                                    continue
                            else:
                                continue # if no contacts, then don't combine vdms-- can't discern them enough to combine.
                        else:
                            continue # if no contacts, then don't combine vdms-- can't discern them enough to combine.

                        vdm2 = g2.get_group(vdmname2)
                        if (cg2, cg_gr2) not in seen_cg_cg_grs:
                            if all((_cg2, _cg_gr2) not in seen_cg_cg_grs 
                                    for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]):
                                seen_cg_cg_grs[(cg2, cg_gr2)] = vdm2
                                for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                                    seen_cg_cg_grs[(_cg2, _cg_gr2)] = vdm2
                                continue

                        if 'satisfies_cst' in vdms.columns:
                            cst_2 = vdm2['satisfies_cst'].iat[0]
                            satis_cst = seen_cg_cg_grs[(cg2, cg_gr2)]['satisfies_cst'].iat[0]
                            cst_is_nan = np.isnan(satis_cst)
                            if (cst_is_nan and (~np.isnan(cst_2))):
                                seen_cg_cg_grs[(cg2, cg_gr2)] = vdm2
                                for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                                    seen_cg_cg_grs[(_cg2, _cg_gr2)] = vdm2
                                continue

                            if (cst_is_nan and
                                    (~np.isnan(cst_2)) and (
                                            cst_2 == seen_cg_cg_grs[(cg2, cg_gr2)]['satisfies_cst'].iat[0])):
                                score = seen_cg_cg_grs[(cg2, cg_gr2)]['score'].iat[0]
                                vdm2_score = vdm2['score'].iat[0]
                                if vdm2_score < score:
                                    seen_cg_cg_grs[(cg2, cg_gr2)] = vdm2
                                    for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                                        seen_cg_cg_grs[(_cg2, _cg_gr2)] = vdm2
                                elif vdm2_score == score:
                                    if vdm2['dist_to_query'].iat[0] < \
                                            seen_cg_cg_grs[(cg2, cg_gr2)]['dist_to_query'].iat[0]:
                                        seen_cg_cg_grs[(cg2, cg_gr2)] = vdm2
                                        for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                                            seen_cg_cg_grs[(_cg2, _cg_gr2)] = vdm2
                                continue

                            if (~cst_is_nan and
                                    (~np.isnan(cst_2)) and (cst_1 == satis_cst) and cst_2 != cst_1):
                                seen_cg_cg_grs[(cg2, cg_gr2)] = vdm2
                                for _cg2, _cg_gr2 in cg_alts[(cg2, cg_gr2)]:
                                    seen_cg_cg_grs[(_cg2, _cg_gr2)] = vdm2
                                continue

                    if len(seen_cg_cg_grs.keys()) > 0:
                        _vdms = list(seen_cg_cg_grs.values())
                        _vdms.append(vdm)
                        vdm_concat = fast_concat(_vdms)
                        vdm_concat = vdm_concat.drop_duplicates()
                        vdms_list[seg_chain_resnum].append(vdm_concat)
                    else:
                        vdms_list[seg_chain_resnum].append(vdm)
            else:
                for vdmname, vdm in g2:
                    if vdmname in sorted_vdmnames:
                        continue
                    vdms_list[seg_chain_resnum].append(vdm)
        return vdms_list

    # def set_vdms_list(self, vdms):
    #     _grs = vdms.groupby(['seg_chain_resnum', 'resname_rota', 'rotamer'], dropna=False)
    #     vdms_list = defaultdict(list)
    #     for n1, g1 in _grs:
    #         if len(set(g1['CG_ligand_coverage'])) > 1:
    #             # print('Possible double-agent vdM!', n1, 'pose', self.identifier)
    #             g2 = g1.groupby(self.groupby)
    #             for vdmname, vdm in g2:
    #                 cg_lig_cov_1 = vdm['CG_ligand_coverage'].iat[0]
    #                 if 'satisfies_cst' in vdms.columns:
    #                     cst_1 = vdm['satisfies_cst'].iat[0]

    #                 vdm_dict_cg_lig_covs = dict()
    #                 # for vdmname2 in set(g2.groups.keys()) - {vdmname}:
    #                 # vdm2 = g2.get_group(vdmname2)
    #                 for vdmname2, vdm2 in g2:
    #                     if vdmname2 == vdmname:
    #                         continue
    #                     cg_lig_cov_2 = vdm2['CG_ligand_coverage'].iat[0]

    #                     if cg_lig_cov_2 == cg_lig_cov_1:
    #                         continue

    #                     if cg_lig_cov_2 not in vdm_dict_cg_lig_covs:
    #                         vdm_dict_cg_lig_covs[cg_lig_cov_2] = vdm2
    #                         continue

    #                     if 'satisfies_cst' in vdms.columns:
    #                         cst_2 = vdm2['satisfies_cst'].iat[0]
    #                         satis_cst = vdm_dict_cg_lig_covs[cg_lig_cov_2]['satisfies_cst'].iat[0]
    #                         cst_is_nan = np.isnan(satis_cst)
    #                         if (cst_is_nan and (~np.isnan(cst_2))):
    #                             vdm_dict_cg_lig_covs[cg_lig_cov_2] = vdm2
    #                             continue

    #                         if (cst_is_nan and
    #                                 (~np.isnan(cst_2)) and (
    #                                         cst_2 == vdm_dict_cg_lig_covs[cg_lig_cov_2]['satisfies_cst'].iat[0])):
    #                             score = vdm_dict_cg_lig_covs[cg_lig_cov_2]['score'].iat[0]
    #                             vdm2_score = vdm2['score'].iat[0]
    #                             if vdm2_score < score:
    #                                 vdm_dict_cg_lig_covs[cg_lig_cov_2] = vdm2
    #                             elif vdm2_score == score:
    #                                 if vdm2['dist_to_query'].iat[0] < \
    #                                         vdm_dict_cg_lig_covs[cg_lig_cov_2]['dist_to_query'].iat[0]:
    #                                     vdm_dict_cg_lig_covs[cg_lig_cov_2] = vdm2
    #                             continue

    #                         if (~cst_is_nan and
    #                                 (~np.isnan(cst_2)) and (cst_1 == satis_cst) and cst_2 != cst_1):
    #                             vdm_dict_cg_lig_covs[cg_lig_cov_2] = vdm2
    #                             continue

    #                 if len(vdm_dict_cg_lig_covs.keys()) > 0:
    #                     _vdms = list(vdm_dict_cg_lig_covs.values())
    #                     _vdms.append(vdm)
    #                     vdm_concat = fast_concat(_vdms)
    #                     vdms_list[n1[0]].append(vdm_concat)
    #                 else:
    #                     vdms_list[n1[0]].append(vdm)
    #         else:
    #             g2 = g1.groupby(self.groupby)
    #             for vdmname, vdm in g2:
    #                 vdms_list[n1[0]].append(vdm)
    #     return vdms_list

    # def set_site_ens(self, vdms_list, d, es, es_no_cst, add_cst_score):
    #     for site, vdm_list in vdms_list.items():
    #         for vdm in vdm_list:
    #             score = 0
    #             tuples = []
    #             for nn, gg in vdm.groupby(self.groupby):
    #                 score += gg['score'].iat[0]
    #                 tuples.append(nn)
    #             i = tuple(sorted(tuples))
    #             d[site][i] = vdm
    #             es_no_cst[(site, i)] = score
    #             if add_cst_score:
    #                 cst_set = set(vdm['satisfies_cst'][~vdm['satisfies_cst'].isna()])
    #                 score -= len(cst_set) * 5 #100
    #             es[(site, i)] = score

    def set_site_ens(self, vdms_list, d, es, es_no_cst, add_cst_score, all_lig_names, cg_alts, atom_cg_map, cg_atom_map, ligand_atom_types,
                     template, path_to_nbrs_database, ligand_vdm_correspondence, ligand_vdm_correspondence_grs,
                     distance_metric='rmsd', same_rotamer=False, bb_dep=0, use_hb_scores=False,
                     tamp_by_distance=False, ignore_rmsd_column=(),
                     exponential=False, log_logistic=True, gaussian=False, relu=False,
                     ):

        contacting_cgs, contacting_cgs_vdm_info = self._get_contacting_cgs(all_lig_names, cg_alts, atom_cg_map, cg_atom_map, ligand_atom_types)
        self.contacting_cgs = contacting_cgs
        self.contacting_cgs_vdm_info = contacting_cgs_vdm_info

        vdm_info = defaultdict(dict)
        for vdm_gr_name in contacting_cgs.keys():
            resname_rota, rotamer = contacting_cgs_vdm_info[vdm_gr_name]
            for name in contacting_cgs[vdm_gr_name].keys():
                for cg_type, cg_gr in contacting_cgs[vdm_gr_name][name]:
                    try:
                        vdm_info[cg_type][resname_rota].append((cg_gr, vdm_gr_name, rotamer, name))
                    except Exception:
                        vdm_info[cg_type][resname_rota] = [(cg_gr, vdm_gr_name, rotamer, name)]

        self.contacting_lig_vdms = self.find_lig_contact_vdms_knn(template, vdm_info, path_to_nbrs_database, 
                                                                ligand_vdm_correspondence, ligand_vdm_correspondence_grs,
                                                                ignore_rmsd_column=ignore_rmsd_column,
                                                                distance_metric=distance_metric, same_rotamer=same_rotamer, 
                                                                 bb_dep=bb_dep, use_hb_scores=use_hb_scores, 
                                                                tamp_by_distance=tamp_by_distance,
                                                                exponential=exponential, log_logistic=log_logistic, 
                                                                gaussian=gaussian, relu=relu,)

        # if tamp_by_distance:
        #     if exponential:
        #         tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
        #     elif log_logistic:
        #         # middle ground between exponential and gaussian
        #         tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
        #     elif gaussian:
        #         tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
        #     elif relu:
        #         tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        # else:
        tamp_func = lambda x: 1

        for site, vdm_list in vdms_list.items():
            vdm_info = defaultdict(dict)
            for vdm in vdm_list:
                score = 0
                vdm_gr_names = []
                cgs_cg_grs = set()
                vdm_grs = vdm.groupby(self.groupby)
                for nn, gg in vdm_grs:
                    score += gg['score'].iat[0]
                    cg_type = gg['CG_type'].iat[0]
                    cgs_cg_grs.add((cg_type, gg['CG_group'].iat[0]))
                    vdm_gr_names.append(nn)
                
                cgs_cg_grs_alts = {cg_alt for cg_cg_gr in cgs_cg_grs for cg_alt in cg_alts[cg_cg_gr]}

                added_cgs_cg_grs = set()
                for vdm_gr_name in vdm_gr_names:
                    if vdm_gr_name not in self.contacting_lig_vdms:
                        continue
                    for cg, cg_gr in self.contacting_lig_vdms[vdm_gr_name]:
                        if ((cg, cg_gr) in cgs_cg_grs) or ((cg, cg_gr) in added_cgs_cg_grs) or ((cg, cg_gr) in cgs_cg_grs_alts):
                            continue
                        cont = False
                        for (alt_cg, alt_cg_gr) in cg_alts[(cg, cg_gr)]:
                            if ((alt_cg, alt_cg_gr) in cgs_cg_grs) or ((alt_cg, alt_cg_gr) in added_cgs_cg_grs):
                                cont = True
                                break
                        if cont:
                            continue
                        added_cgs_cg_grs.add((cg, cg_gr))
                        s = -1 * self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)][0]
                        # print(vdm_gr_name, self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)][-1])
                        for _cg, _cg_gr in cgs_cg_grs:
                            tfs = []
                            tfs.append(len(cg_atom_map[(_cg, _cg_gr)] - cg_atom_map[(cg, cg_gr)]) == 0)
                            for __cg, __cg_gr in cg_alts[(_cg, _cg_gr)]:
                                tf = len(cg_atom_map[(__cg, __cg_gr)] - cg_atom_map[(cg, cg_gr)]) == 0
                                tfs.append(tf)
                            if any(tfs): # new cg covers old cg
                                _vdm = vdm_grs.get_group(vdm_gr_name)
                                score -= _vdm['score'].iat[0] # remove score contribution of old cg
                                gr_name = list(self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)][-1][-1])
                                cg_name = self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)][-1][0]
                                gr_name.append(cg_name)
                                gr_name.append(site)
                                gr_name = tuple(gr_name)
                                print('vdM', vdm_gr_name, 'replaced by', gr_name)
                                if gr_name in vdm_grs.groups:
                                    print(gr_name, 'in vdMs, penalizing score swap')
                                    score += 0.1 # make sure the true vdM is scored a little better than this swapped score
                        if s < 0:
                            dist = self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)][1]
                            _score = s * tamp_func(dist)
                            score += _score 
                        else:
                            _score = s
                            score += _score
                        self.lig_additional_vdms[vdm_gr_name][(cg, cg_gr)] = _score

                i = tuple(sorted(vdm_gr_names))
                d[site][i] = vdm
                es_no_cst[(site, i)] = score
                if add_cst_score:
                    cst_set = set(vdm['satisfies_cst'][~vdm['satisfies_cst'].isna()])
                    score -= len(cst_set) * 5 #100
                es[(site, i)] = score

    def find_lig_contact_vdms_knn(self, template, vdms, path_to_nbrs_database, ligand_vdm_correspondence, 
                        ligand_vdm_correspondence_grs, ignore_rmsd_column=(), distance_metric='rmsd',
                        same_rotamer=True, bb_dep=0, use_hb_scores=False, tamp_by_distance=False,
                        exponential=False, log_logistic=True, gaussian=False, relu=False):
        # print('Finding first-shell contact vdms...')

        if tamp_by_distance:
            if exponential:
                tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
            elif log_logistic:
                # middle ground between exponential and gaussian
                tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
            elif gaussian:
                tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
            elif relu:
                tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        else:
            tamp_func = lambda x: 1

        contact_vdms = defaultdict(dict)
        if same_rotamer:
            num_neighbors = 1000
        else:
            num_neighbors = 1000
        # if 'opt_vdms_grs' not in self.__dict__.keys() or self.opt_vdms_grs is None:
        #     self.opt_vdms_grs = self.opt_vdms.groupby(self.groupby)
        if self.vdms_grs is None:
            self.vdms_grs = self.vdms_sidechains.groupby(self.groupby)
        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
            path_to_rotamer_nbrs_database_ = path_to_nbrs_database + 'vdMs_rotamers_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
            path_to_rotamer_nbrs_database_ = path_to_nbrs_database + 'vdMs_rotamers_nbrs_maxdist/'
        path_to_nbrs_database_scores = path_to_nbrs_database + 'vdMs_cg_nbrs_scores/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'

        print('additional ligand contacts...')
        for cg in vdms.keys():
            if cg == 'ccoh' and 'coh' in vdms: # avoid double-counting with coh
                continue
            print('\t', cg)
            if cg not in os.listdir(path_to_nbrs_database_):
                print('\t\t CG not found in database. Skipping...')
                continue
            cg_df = cg_dfs[cg]
            for aa in vdms[cg].keys():
                if aa in ['ALA', 'GLY']:
                    has_rotamer = False
                else:
                    has_rotamer = True
                if has_rotamer:
                    rotamer_df = rotamer_dfs[aa]
                print('\t\t', aa)
                if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                    print('\t\t\t AA not found in CG database. Skipping...')
                    continue
                if distance_metric == 'rmsd':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    num_cg_atoms = nbrs._fit_X.shape[1] / 3 
                    if same_rotamer and has_rotamer:
                        with open(path_to_rotamer_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            rotamer_nbrs = pickle.load(f)
                        num_rotamer_atoms = rotamer_nbrs._fit_X.shape[1] / 3 
                elif distance_metric == 'maxdist':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    if same_rotamer and has_rotamer:
                        with open(path_to_rotamer_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                            rotamer_nbrs = pickle.load(f)

                _num_nbrs = min(nbrs._fit_X.shape[0], num_neighbors)
                scores = pd.read_parquet(path_to_nbrs_database_scores + cg + '/' + aa + '.parquet.gzip')
                score_col_dict = {colname: i for i, colname in enumerate(scores.columns)}
                scores = scores.values
                with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                    groupnames = pickle.load(f)

                for cg_gr, vdm_gr_name, coords_rotamer, lig_name in vdms[cg][aa]:
                    seg_aa, chain_aa, res_aa = vdm_gr_name[-1]
                    if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                        df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                                (template.dataframe['chain'] == chain_aa) &
                                                (template.dataframe['resnum'] == res_aa)].copy()
                        m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                        for _name in ['N', 'CA', 'C']])
                        t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                        R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                        template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com)
                    
                    coords = self.get_ligand_cg_coords_for_lookup(ligand_vdm_correspondence, cg, cg_gr, cg_df, 
                                                                  ignore_rmsd_column=ignore_rmsd_column) 
                    R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]
                    coords = apply_transform(R, mob_com, targ_com, coords)
                    coords = coords.reshape(1, -1)
                    try:
                        dists, inds = nbrs.kneighbors(coords, n_neighbors=_num_nbrs, return_distance=True)
                    except Exception:
                        print('The pose possibily contains duplicate vdMs.  Perhaps \
                                check the CG_ligand_coverage column of the ligand.txt file for \
                                overlapping atoms in different CG_ligand_coverage groups.')
                        print('culprit vdM:', cg, cg_gr, aa, vdm_gr_name)
                        continue
                    dists, inds = dists[0], inds[0]

                    if dists.size > 0: # There should always be at least one neighbor with kneighbors so this doesn't do anything...
                        contact_type = scores[inds, score_col_dict['contact_type']]
                        ct_filter = contact_type == 'sc'
                        inds = inds[ct_filter]
                        dists = dists[ct_filter]
                        if len(inds) == 0:  # only consider sidechain-only interactions
                            continue
                        ss = ''
                        if bb_dep == 1:
                            res_filter = template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)
                            abple = template.dataframe[res_filter]['ABPLE'].iat[0]
                            ss = 'ABPLE_' + abple
                        elif bb_dep == 0:
                            ss = 'bb_ind'
                        score_col = 'C_score_' + ss
                        score_col_ind = score_col_dict[score_col]
                        scores_ = scores[inds, score_col_ind].astype(float)
                        not_na = ~np.isnan(scores_)
                        scores_ = scores_[not_na]
                        inds = inds[not_na]
                        dists = dists[not_na]
                        if len(scores_) == 0:  # No ss score for vdm so give it the lowest ss score possible.
                            min_score = min(-1, scores[:, score_col_ind].min())
                            if cg in self.cg_weights:
                                min_score = min_score * self.cg_weights[cg]
                            large_distance = 10
                            contact_vdms[vdm_gr_name][(cg, cg_gr)] = (min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)
                            continue
                        
                        lig_vdm_corr = ligand_vdm_correspondence_grs.get_group((cg, cg_gr))
                        if lig_vdm_corr['is_not_acceptor'].any():
                            is_not_acc = ~(scores[inds, score_col_dict['is_acceptor']].astype(bool))
                            scores_ = scores_[is_not_acc]
                            inds = inds[is_not_acc]
                            dists = dists[is_not_acc]
                            if len(scores_) == 0:
                                min_score = min(-1, scores[:, score_col_ind].min())
                                if cg in self.cg_weights:
                                    min_score = min_score * self.cg_weights[cg]
                                large_distance = 10
                                try:
                                    contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                                except:
                                    contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                                continue
                        elif lig_vdm_corr['is_not_donor'].any():
                            is_not_don = ~(scores[inds, score_col_dict['is_donor']].astype(bool))
                            scores_ = scores_[is_not_don]
                            inds = inds[is_not_don]
                            dists = dists[is_not_don]
                            if len(scores_) == 0:
                                min_score = min(-1, scores[:, score_col_ind].min())
                                if cg in self.cg_weights:
                                    min_score = min_score * self.cg_weights[cg]
                                large_distance = 10
                                try:
                                    contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                                except:
                                    contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                                continue

                        if use_hb_scores:
                            is_hb = scores[inds, score_col_dict['hbond']].astype(bool)
                            if is_hb.any():
                            #     continue
                            # inds = inds[is_hb]
                            # dists = dists[is_hb]
                                score_col = 'C_score_hb_' + ss
                                score_col_ind = score_col_dict[score_col]
                                scores_[is_hb] = scores[inds[is_hb], score_col_ind]
                                not_na = ~np.isnan(scores_)
                                scores_ = scores_[not_na]
                                inds = inds[not_na]
                                dists = dists[not_na]
                                if len(scores_) == 0:
                                    min_score = min(-1, scores[:, score_col_ind].min())
                                    if cg in self.cg_weights:
                                        min_score = min_score * self.cg_weights[cg]
                                    large_distance = 10
                                    try:
                                        contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                                    except:
                                        contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                                    continue
                        if same_rotamer and has_rotamer:
                            # index_rotamer = groupnames.index(tuple(vdm_gr_name[:-2]))
                            # coords_rotamer = rotamer_nbrs._fit_X[index_rotamer].reshape(-1, 3)
                            coords_rotamer = apply_transform(R, mob_com, targ_com, coords_rotamer)
                            coords_rotamer = coords_rotamer.reshape(1, -1)
                            try:
                                dists_rot, inds_rot = rotamer_nbrs.radius_neighbors(coords_rotamer, return_distance=True)
                            except Exception:
                                print('The pose possibily contains duplicate vdMs. Problems with looking up rotamer.')
                                print('culprit vdM:', cg, cg_gr, aa, vdm_gr_name)
                                continue
                            dists_rot, inds_rot = dists_rot[0], inds_rot[0]

                            if aa in flip_dict:
                                coords_rotamer = flip_coords_from_reference_df(coords_rotamer.reshape(-1,3), aa, rotamer_df)
                                coords_rotamer = coords_rotamer.reshape(1, -1)
                                try:
                                    dists_rot_flip, inds_rot_flip = rotamer_nbrs.radius_neighbors(coords_rotamer, return_distance=True)
                                except Exception:
                                    print('The pose possibily contains duplicate vdMs. Problems with looking up rotamer.')
                                    print('culprit vdM:', cg, cg_gr, aa, vdm_gr_name)
                                    continue
                                dists_rot_flip, inds_rot_flip = dists_rot_flip[0], inds_rot_flip[0]
                                if len(inds_rot) > 0 and len(inds_rot_flip) > 0:
                                    inds_rot, uniq_inds = np.unique(np.concatenate((inds_rot, inds_rot_flip)), return_index=True)
                                    dists_rot = np.concatenate((dists_rot, dists_rot_flip))[uniq_inds]

                            if len(inds_rot) == 0:
                                print('No rotamer nbrs found for', cg, cg_gr, aa, vdm_gr_name)
                                min_score = min(-1, scores[:, score_col_ind].min())
                                if cg in self.cg_weights:
                                    min_score = min_score * self.cg_weights[cg]
                                large_distance = 10
                                try:
                                    contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                                except:
                                    contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                                continue
                            inds_tf = np.in1d(inds, inds_rot)
                            dists = dists[inds_tf]
                            inds = inds[inds_tf]
                            scores_ = scores_[inds_tf]
                            if len(inds) == 0:
                                print('No rotamer nbrs found for', cg, cg_gr, aa, vdm_gr_name)
                                min_score = min(-1, scores[:, score_col_ind].min())
                                if cg in self.cg_weights:
                                    min_score = min_score * self.cg_weights[cg]
                                large_distance = 10
                                try:
                                    contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                                except:
                                    contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                                continue
                        # if same_rotamer:
                        #     rotamer_ind = score_col_dict['rotamer']
                        #     rotamers = scores[inds, rotamer_ind].astype(str)
                        #     rotamer_filter = rotamers == rotamer
                        #     scores_ = scores_[rotamer_filter]
                        #     inds = inds[rotamer_filter]
                        #     dists = dists[rotamer_filter]
                            # if len(scores_) == 0:
                            #     min_score = min(-1, scores[:, score_col_ind].min())
                            #     large_distance = 10
                            #     try:
                            #         contact_vdms[vdm_gr_name][lig_name].append((min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),))
                            #     except:
                            #         contact_vdms[vdm_gr_name][lig_name] = [(min_score, large_distance, (cg, aa, 'NA'), (cg, cg_gr),)]
                            #     continue
                        ind_lowest_dist = np.argmin(dists)
                        dist_lowest = dists[ind_lowest_dist]
                        if distance_metric == 'rmsd':
                            dist_lowest = dist_lowest / np.sqrt(num_cg_atoms)
                        score_lowest_dist = scores_[ind_lowest_dist]
                        best_groupname = groupnames[inds[ind_lowest_dist]]
                        if score_lowest_dist > 0:
                            _score_lowest_dist = score_lowest_dist * tamp_func(dist_lowest)
                        else:
                            _score_lowest_dist = score_lowest_dist
                        if cg in self.cg_weights:
                            _score_lowest_dist = _score_lowest_dist * self.cg_weights[cg]
                        try:
                            contact_vdms[vdm_gr_name][lig_name].append((_score_lowest_dist, dist_lowest, (cg, aa, best_groupname), (cg, cg_gr),))
                        except:
                            contact_vdms[vdm_gr_name][lig_name] = [(_score_lowest_dist, dist_lowest, (cg, aa, best_groupname), (cg, cg_gr),)]

        filtered_contact_vdms = defaultdict(dict)
        for vdm_gr_name in contact_vdms.keys():
            for lig_name in contact_vdms[vdm_gr_name].keys():
                if len(contact_vdms[vdm_gr_name][lig_name]) == 0:
                    continue
                # contact_vdms[vdm_gr_name][lig_name].sort(key=lambda x: x[0], reverse=True)
                contact_vdms[vdm_gr_name][lig_name].sort(key=lambda x: x[1])
                best_contact = contact_vdms[vdm_gr_name][lig_name][0]
                filtered_contact_vdms[vdm_gr_name][best_contact[-1]] = best_contact[:-1]

        return filtered_contact_vdms

    def _get_sum_pairwise(self, sites, i, vdm_name_i, d, es, ep, add_cst_score, min_max, vdW_tolerance=0.1):
        p = 0
        for j in list(set(range(len(sites))) - {i}):
            _p = []
            for vdm_name_j in list(d[sites[j]].keys()):
                try:
                    _p.append(ep[(sites[i], vdm_name_i)][(sites[j], vdm_name_j)])
                except:
                    if d[sites[i]][vdm_name_i]['resname_rota'].iat[0] == 'GLY' or \
                            d[sites[j]][vdm_name_j]['resname_rota'].iat[0] == 'GLY':
                        cla = Pose()
                        cla.dfq_clash_free = [1]  # a convoluted way to get a "clash object" with dfq_clash_free of len 1
                    else:
                        cla = Clash(d[sites[i]][vdm_name_i], d[sites[j]][vdm_name_j],
                                    **dict(tol=vdW_tolerance))
                        cla.set_grouping(self.groupby)
                        cla.find()
                    if add_cst_score:
                        s1 = set(d[sites[i]][vdm_name_i]['satisfies_cst']) - {np.nan}
                        s2 = set(d[sites[j]][vdm_name_j]['satisfies_cst']) - {np.nan}
                        lens12 = len(s1 & s2)
                    else:
                        lens12 = 0
                    if len(cla.dfq_clash_free) == 0:
                        ep[(sites[i], vdm_name_i)][(sites[j], vdm_name_j)] = 10
                        ep[(sites[j], vdm_name_j)][(sites[i], vdm_name_i)] = 10
                        _p.append(10)
                    elif add_cst_score and lens12 > 0:
                        ep[(sites[i], vdm_name_i)][(sites[j], vdm_name_j)] = 0 #100 * lens12
                        ep[(sites[j], vdm_name_j)][(sites[i], vdm_name_i)] = 0 #100 * lens12
                        _p.append(0) # _p.append(100 * lens12)
                    else:
                        ep[(sites[i], vdm_name_i)][(sites[j], vdm_name_j)] = 0
                        ep[(sites[j], vdm_name_j)][(sites[i], vdm_name_i)] = 0
                        _p.append(0)
            if len(_p) > 0:
                if min_max == 'min':
                    p += np.min(_p)
                elif min_max == 'max':
                    p += np.max(_p)
        return p

    def dee(self, sites, d, es, ep, add_cst_score, print_time=False, vdW_tolerance=0.1):
        if print_time:
            t0 = time.time()

        for i in range(len(sites)):
            for vdm_name_i in list(d[sites[i]].keys()):
                si = es[(sites[i], vdm_name_i)]
                break_loop = False
                pmin = self._get_sum_pairwise(sites, i, vdm_name_i, d, es, ep, 
                                              add_cst_score, min_max='min', vdW_tolerance=vdW_tolerance)
                for vdm_name_k in list(d[sites[i]].keys()):
                    if vdm_name_k == vdm_name_i:
                        continue
                    sk = es[(sites[i], vdm_name_k)]
                    pmax = self._get_sum_pairwise(sites, i, vdm_name_k, d, es, ep, 
                                                  add_cst_score, min_max='max', vdW_tolerance=vdW_tolerance)
                    if si + pmin > sk + pmax:
                        # print('popping', vdm_name_i)
                        d[sites[i]].pop(vdm_name_i)
                        break_loop = True
                        break
                if break_loop:
                    break

        if print_time:
            print('Time to calculate dee: ', time.time() - t0)

    def make_pairwise_from_contacts(self, d, ep, clashing, contact_vdms, weight=1.0, 
                                    tamp_by_distance=False, exponential=False,
                                    log_logistic=False, gaussian=False, relu=False, C_score_threshold=0):
        # print('contact_vdms', contact_vdms)
        C_score_threshold = 0
        if tamp_by_distance:
            if exponential:
                tamp_func = lambda x: np.exp(2 * np.log(1/2) * x) # 0.71 at x = 0.25, 1/2 at x = 0.5, 1/4 at x = 1
            elif log_logistic:
                # middle ground between exponential and gaussian
                tamp_func = lambda x: 1 / (1 + (2 * x)**2) # 0.8 at x = 0.25, 1/2 at x = 0.5, 1/5 at x = 1
            elif gaussian:
                tamp_func = lambda x: np.exp(4 * np.log(1/2) * x**2) # 0.84 at x = 0.25, 1/2 at x = 0.5, 0.065 at x = 1
            elif relu:
                tamp_func = lambda x: max(1 - x, 0) # 0.75 at x = 0.25, 1/2 at x = 0.5, 0.0 at x = 1
        else:
            tamp_func = lambda x: 1
        for site_i in d.keys():
            for site_j in set(d.keys()) - {site_i}:
                for vdm_name_i in d[site_i].keys():
                    # print('i', vdm_name_i)
                    for vdm_name_j in d[site_j].keys():
                        # print('j', vdm_name_j)
                        if (vdm_name_i, vdm_name_j) in contact_vdms:
                            # print('covdms', contact_vdms[(vdm_name_i, vdm_name_j)])
                            # There could be more than one kind of CG between these two vdMs.
                            # Will need to account for this ion future.
                            score = contact_vdms[(vdm_name_i, vdm_name_j)][0]
                            if score > C_score_threshold:
                                dist = contact_vdms[(vdm_name_i, vdm_name_j)][1]
                                ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = -weight * tamp_func(dist) * score
                            else:
                                ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = -weight * score
                            if (vdm_name_j, vdm_name_i) not in contact_vdms:
                                ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)]
                        if (vdm_name_j, vdm_name_i) in contact_vdms:
                            # print('covdms', contact_vdms[(vdm_name_j, vdm_name_i)])
                            score = contact_vdms[(vdm_name_j, vdm_name_i)][0]
                            if score > C_score_threshold:
                                dist = contact_vdms[(vdm_name_j, vdm_name_i)][1]
                                ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = -weight * tamp_func(dist) * score
                            else:
                                ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = -weight * score
                            if (vdm_name_i, vdm_name_j) not in contact_vdms:
                                ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)]
                        if (vdm_name_i, vdm_name_j) in contact_vdms and (vdm_name_j, vdm_name_i) in contact_vdms:
                            avg = (ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] 
                                   + ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)]) / 2
                            ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = avg
                            ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = avg
                        elif (vdm_name_i, vdm_name_j) not in contact_vdms and (vdm_name_j, vdm_name_i) not in contact_vdms:
                            all_clashing = []
                            for vni in vdm_name_i:
                                if vni in clashing:
                                    for vnj in vdm_name_j:
                                        if vnj in clashing[vni]:
                                            all_clashing.append(True)
                                        else:
                                            all_clashing.append(False)
                                else:
                                    all_clashing.append(False)
                            if all(all_clashing):
                                ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = 10
                                ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = 10
                            else:
                                ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = 0
                                ep[(site_j, vdm_name_j)][(site_i, vdm_name_i)] = 0
        # print('ep', ep)
        # self.ep = ep

    def get_clashing_from_contacts(self, df_con):
        groupby_q = [s + '_q' for s in self.groupby]
        index_seg_chain_rn_q = groupby_q.index('seg_chain_resnum_q')
        groupby_t = [s + '_t' for s in self.groupby]
        index_seg_chain_rn_t = groupby_t.index('seg_chain_resnum_t')
        clashing = defaultdict(dict)
        _df_con = df_con[df_con.contact_type == 'cl']
        for n1, g1 in _df_con.groupby(groupby_q):
            for n2, g2 in g1.groupby(groupby_t):
                if n1[index_seg_chain_rn_q] == n2[index_seg_chain_rn_t]:
                    continue
                if 'cl' in g2.contact_type.values:
                    clashing[n1][n2] = True
                    clashing[n2][n1] = True
        return clashing

    def get_poss_vdms_from_contacts(self, df_con, d, clashing, skipping_number=0):

        name_map = {vn: vdm_names for site in d.keys() for vdm_names in d[site].keys() for vn in vdm_names}

        groupby_q = [s + '_q' for s in self.groupby]
        groupby_t = [s + '_t' for s in self.groupby]
        index_seg_chain_rn_t = groupby_t.index('seg_chain_resnum_t')
        vdms = defaultdict(dict)
        rotamers = defaultdict(dict)
        grs_sites_q = df_con.groupby(['seg_chain_resnum_q'])
        for seg_chain_rn_q, g in grs_sites_q:
            for q_vdm, vdm in g.groupby(groupby_q):
                if q_vdm not in name_map:
                    continue
               
                v = self.vdms_grs.get_group(q_vdm)
                resn = v.resname_rota.iat[0]
                if resn not in ['ALA', 'GLY']:
                    rotamer_df = rotamer_dfs[resn]
                    df_rotamer_coords = merge(rotamer_df, v, on='name')
                    rotamer_coords = df_rotamer_coords[['c_x', 'c_y', 'c_z']].values
                else:
                    rotamer_coords = None
                rota_resname = vdm.resname_rota_q.iat[0]
                
                qs = name_map[q_vdm]
                
                is_outer_shell = False
                if 'ligand_seg_chain_resnum' in vdm.columns:
                    ligand_seg_chain_resnum = vdm.ligand_seg_chain_resnum.iat[0]
                    if type(ligand_seg_chain_resnum) == tuple:
                        is_outer_shell = True
                elif 'ligand_seg_chain_resnum_q' in vdm.columns:
                    ligand_seg_chain_resnum = vdm.ligand_seg_chain_resnum_q.iat[0]
                    if type(ligand_seg_chain_resnum) == tuple:
                        is_outer_shell = True

                for t_vdm, vdm_cg in vdm.groupby(groupby_t):
                    
                    if t_vdm not in name_map:
                        continue
                    seg_chain_rn_t = t_vdm[index_seg_chain_rn_t]
                    if seg_chain_rn_t == seg_chain_rn_q:
                        continue
                    if t_vdm in clashing and q_vdm in clashing[t_vdm]:
                        continue
                    
                    #### Do not count outer-shell vdMs as possible new contacts... ####
                    if is_outer_shell:
                        if seg_chain_rn_t == ligand_seg_chain_resnum:
                            continue

                    is_outer_shell_cg = False
                    if 'ligand_seg_chain_resnum' in vdm_cg.columns:
                        ligand_seg_chain_resnum_cg = vdm_cg.ligand_seg_chain_resnum.iat[0]
                        if type(ligand_seg_chain_resnum_cg) == tuple:
                            is_outer_shell_cg = True
                    elif 'ligand_seg_chain_resnum_t' in vdm_cg.columns:
                        ligand_seg_chain_resnum_cg = vdm_cg.ligand_seg_chain_resnum_t.iat[0]
                        if type(ligand_seg_chain_resnum_cg) == tuple:
                            is_outer_shell_cg = True

                    if is_outer_shell_cg:
                        if seg_chain_rn_q == ligand_seg_chain_resnum_cg:
                            continue
                    ##################################################################

                    cg_resname = vdm_cg.resname_rota_t.iat[0]
                    cg_names = vdm_cg['name_t'].values
                    ts = name_map[t_vdm]
                    
                    for cg in cg_dicts.keys():
                        if cg in ['bb_cnh', 'bb_cco']:
                            continue
                        if cg_resname in cg_dicts[cg]:
                            for cg_name in cg_names:
                                if cg_name in set(cg_dicts[cg][cg_resname]):
                                    if ((seg_chain_rn_t[0] != seg_chain_rn_q[0]) or (seg_chain_rn_t[1] != seg_chain_rn_q[1]) \
                                    or np.abs(seg_chain_rn_t[-1] - seg_chain_rn_q[-1]) > skipping_number):
                                        try:
                                            vdms[cg][rota_resname].add((qs, ts))
                                        except:
                                            vdms[cg][rota_resname] = set()
                                            vdms[cg][rota_resname].add((qs, ts))
                                        try:
                                            rotamers[cg][rota_resname][(qs, ts)] = rotamer_coords
                                        except:
                                            rotamers[cg][rota_resname] = dict()
                                            rotamers[cg][rota_resname][(qs, ts)] = rotamer_coords
        return vdms, rotamers
                        

    def find_opt(self, template, path_to_database, sample,
                 bbdep=True, use_hb_scores=False, 
                 C_score_threshold=0,
                 pairwise_dict=None, force_MC=False, force_DEE=False, 
                 DEE_to_MC_switch_number=1000, print_DEE_time=False, 
                 compute_pairwise_contacts=False,
                 knn_contacts=True, contact_distance_metric='rmsd',
                 use_same_rotamer_for_pairwise_contacts=True, 
                 use_same_rotamer_for_lig_contacts=True,
                 ignore_rmsd_column=(),
                 pairwise_contact_weight=1.0, tamp_by_distance=False,
                 pair_nbr_distance=0.7, brute_force_no_DEE=False,
                 exponential=False, log_logistic=True, gaussian=False, relu=False, 
                 outer_shell_score_weight=0.5, 
                 burial_threshold=0.5,
                 vdW_tolerance=0.1,
                 ):

        if self._already_scored:
            print(self.identifier, self.opt_en, self.opt_vdms_names)
            return

        if len(self.vdms_sidechains) == 0:
            print(self.identifier, 'No vdms')
            return

        vdms = self.vdms_sidechains
        vdms['score'] = 0.0
        self.score_vdms(vdms, template, bbdep=bbdep, 
                        use_hb_scores=use_hb_scores,
                        tamp_by_distance=tamp_by_distance,
                        exponential=exponential,
                        log_logistic=log_logistic, gaussian=gaussian, 
                        relu=relu,
                        outer_shell_score_weight=outer_shell_score_weight,
                        burial_threshold=burial_threshold)

        self.vdms_grs = self.vdms_sidechains.groupby(self.groupby)

        es = dict()
        if pairwise_dict is None:
            ep = defaultdict(dict)
        else:
            ep = pairwise_dict
        d = defaultdict(dict)
        es_no_cst = dict()
        add_cst_score = False
        groupby = self.groupby.copy()
        groupby.append('dist_to_query')

        if 'satisfies_cst' in vdms.columns:
            add_cst_score = True

        if bbdep:
            bb_dep = 1
        else:
            bb_dep = 0

        if use_same_rotamer_for_lig_contacts:
            same_rot = True
        else:
            same_rot = False

        path_to_nbrs_database = path_to_database + '../nbrs/'

        all_lig_names = sample.ligand_names 
        cg_alts = sample.cg_alts
        # atom_cg_map = sample.atom_cg_map
        atom_cg_map_by_atomtype = sample.atom_cg_map_by_atomtype
        cg_atom_map = sample.cg_atom_map
        ligand_atom_types = sample.ligand_atom_types
        ligand_vdm_correspondence = sample.ligand_vdm_correspondence
        ligand_vdm_correspondence_grs = sample.ligand_vdm_correspondence_grs

        # print('Setting vdms...')
        vdms = vdms[vdms['score'] <= -1 * C_score_threshold]
        vdms_list = self.set_vdms_list(vdms, cg_alts, cg_atom_map, all_lig_names)

        # self.set_site_ens(vdms_list, d, es, es_no_cst, add_cst_score)
        self.set_site_ens(vdms_list, d, es, es_no_cst, add_cst_score, all_lig_names, cg_alts, 
                         atom_cg_map_by_atomtype, cg_atom_map,ligand_atom_types,
                     template, path_to_nbrs_database, ligand_vdm_correspondence, ligand_vdm_correspondence_grs,
                     ignore_rmsd_column=ignore_rmsd_column,
                     distance_metric=contact_distance_metric, same_rotamer=same_rot, bb_dep=bb_dep, 
                     use_hb_scores=use_hb_scores,
                     tamp_by_distance=tamp_by_distance,
                     exponential=exponential,
                     log_logistic=log_logistic, gaussian=gaussian, 
                     relu=relu,)
        
        print('Setting contacts...')
        # This code blows up memory for large rmsd (ie large numbers of contacts)
        # need to correct it in the future.
        sc1 = self.vdms_sidechains.copy()
        seg_chain_resnum_grs = sc1.groupby(['seg_chain_resnum'])
        seg_chain_resnums = seg_chain_resnum_grs.groups.keys()
        df_contacts = []
        keep_cols = [s for s in self.groupby]
        keep_cols.extend(['ligand_seg_chain_resnum', 'resname_rota', 'name'])
        for num_resnums, seg_chain_resnum_i in enumerate(seg_chain_resnums):
            g_i = seg_chain_resnum_grs.get_group(seg_chain_resnum_i).copy()
            g_not_i = sc1.drop(g_i.index)
            con = Contact(g_i, g_not_i, tol=vdW_tolerance)
            con.set_grouping(['seg_chain_resnum'])
            con.find(keep_columns=keep_cols)
            df_contacts.append(con.df_contacts)
            if num_resnums > 1 and num_resnums % 5 == 0:
                df_contacts = [pd.concat(df_contacts)]
        df_contacts = pd.concat(df_contacts)

        ##############################
        # self.df_contacts = df_contacts
        #############################

        print('Getting clashing...')
        clashing = self.get_clashing_from_contacts(df_contacts)
        # self.clashing = clashing

        contact_vdms = dict()
        if compute_pairwise_contacts:
            poss_vdms_from_contacts, poss_rotamers = self.get_poss_vdms_from_contacts(df_contacts, d, clashing)
            path_to_nbrs_database = path_to_database + '../nbrs/'

            # print('Finding contact nbrs...')
            if bbdep:
                bb_dep = 1
            else:
                bb_dep = 0

            if use_same_rotamer_for_pairwise_contacts:
                same_rot = True
            else:
                same_rot = False

            if knn_contacts:
                contact_vdms = self.find_contact_vdms_knn(template, poss_vdms_from_contacts, poss_rotamers, path_to_nbrs_database, 
                                    distance_metric=contact_distance_metric,
                                    same_rotamer=same_rot, bb_dep=bb_dep, use_hb_scores=use_hb_scores)
            else:
                contact_vdms = self.find_contact_vdms(template, poss_vdms_from_contacts, poss_rotamers, path_to_nbrs_database, 
                                    distance_metric=contact_distance_metric, distance_cutoff=pair_nbr_distance,
                                    same_rotamer=same_rot, bb_dep=bb_dep, use_hb_scores=use_hb_scores)

        # self.contact_vdms = contact_vdms
        # print('Making pairwise from contacts...')
        self.make_pairwise_from_contacts(d, ep, clashing, contact_vdms, 
                                        weight=pairwise_contact_weight,
                                        tamp_by_distance=tamp_by_distance,
                                        exponential=exponential,
                                        log_logistic=log_logistic, gaussian=gaussian, 
                                        relu=relu, C_score_threshold=C_score_threshold)

        #optimize this in future by creating pair-energy dict separately then using numba for DEE.
        num_combos = np.prod([len(d[site]) for site in d.keys()])
        do_DEE = False
        if force_DEE or (len(d.keys()) < 3 and num_combos <= DEE_to_MC_switch_number and not force_MC):
            if self._force_MC:
                pass
            else:
                do_DEE = True
                if brute_force_no_DEE:
                    print('brute force')
                else:
                    print('DEE + brute force')
        sites = list(d.keys())

        for key in d.keys():
            d[key][None] = -1 * C_score_threshold
            es[(key, None)] = -1 * C_score_threshold # for DEE/MC purposes
            es_no_cst[(key, None)] = 0 # absence of a vdM counts as 0 in final score
            # d[key][None] = 0
            # es[(key, None)] = 0 # for DEE/MC purposes
            # es_no_cst[(key, None)] = 0 # absence of a vdM counts as 0 in final score

        for key1 in list(ep.keys()):
            for key2 in list(ep[key1].keys()):
                ep[key1][(key2[0], None)] = 0
                ep[(key1[0], None)][(key2[0], None)] = 0
                ep[(key1[0], None)][key2] = 0
                ep[(key2[0], None)][(key1[0], None)] = 0
                ep[key2][(key1[0], None)] = 0

        if do_DEE and not brute_force_no_DEE:
            # print('Making pairwise energy dict...')
            self.dee(sites, d, es, ep, add_cst_score, print_time=print_DEE_time, vdW_tolerance=vdW_tolerance)

        # for key in d.keys():
        #     d[key][None] = 0
        #     es[(key, None)] = 0 # for DEE/MC purposes
        #     es_no_cst[(key, None)] = 0 # absence of a vdM counts as 0 in final score

        # for key1 in list(ep.keys()):
        #     for key2 in list(ep[key1].keys()):
        #         ep[key1][(key2[0], None)] = 0
        #         ep[(key1[0], None)][(key2[0], None)] = 0
        #         ep[(key1[0], None)][key2] = 0
        #         ep[(key2[0], None)][(key1[0], None)] = 0
        #         ep[key2][(key1[0], None)] = 0

        if do_DEE:
            opt_en = np.inf
            opt_vdm_names = list()
            keys = [list(zip([v]*len(k), k)) for v, k in d.items()]
            opt_vdm_names_sets = []
            for combo in itertools.product(*keys):
                c_en = sum(es[key_] for key_ in combo)
                c_en += sum(ep[key1_][key2_]
                               for key1_, key2_ in itertools.combinations(combo, 2))
                if c_en < opt_en:
                    opt_vdm_names_sets = []
                    opt_en = c_en
                    opt_vdm_names = combo
                    opt_vdm_names_sets.append(combo)
                if c_en == opt_en:
                    opt_vdm_names_sets.append(combo)

        else:
            print('MC')
            if len(self._frozen_sites) > 0:
                for site, frozen_vdm_name in self._frozen_sites.items():
                    for current_vdm_name in list(d[site].keys()):
                        if current_vdm_name is None:
                            d[site].pop(current_vdm_name)
                        elif frozen_vdm_name not in current_vdm_name:
                            d[site].pop(current_vdm_name)
                    if len(d[site].keys()) == 0:
                        d.pop(site)

            sites = list(d.keys())
            if len(sites) == 0:
                return

            sites_index = list(range(len(sites)))
            #initialize
            soln = []
            for i in sites_index:
                site = sites[i]
                best_en = np.inf
                best_vdm_name = None
                for vdm_name in d[site].keys():
                    en = es[(site, vdm_name)]
                    if en < best_en:
                        best_en = en
                        best_vdm_name = vdm_name
                soln_key = (site, best_vdm_name)
                if soln_key in es:
                    soln.append(soln_key)

            en_soln = sum(es[key_] for key_ in soln)
            en_soln += sum(self.get_pairwise(d, ep, key1_, key2_, add_cst_score, vdW_tolerance=vdW_tolerance)
                              for key1_, key2_ in itertools.combinations(soln, 2))

            kts = [10, 5, 2, 0.5, 0.1]
            en_solns = [en_soln]
            solns = [soln]
            consecutive_fails = 0
            min_tries = sum(len(d[site])*10 for site in d.keys()) # run at least 10 times the vdms at each site
            num_tries = max(DEE_to_MC_switch_number, max(len(d.keys()) * 500, min_tries)) # run at least 500 times per site or min_tries or DEE_to_MC (1000 times).
            # print('MC iterations:', num_tries)

            if len(self._frozen_sites) > 0:
                new_sites_index = []
                for si in sites_index:
                    if sites[si] not in self._frozen_sites:
                        new_sites_index.append(si)
                if len(new_sites_index) > 0:
                    sites_index = new_sites_index

            for kt in kts:
                for _ in range(num_tries):
                    site_index = random.choice(sites_index)
                    site = sites[site_index]
                    vdm_name = random.choice(list(d[site].keys()))
                    new_soln = soln.copy()
                    new_soln[site_index] = (site, vdm_name)
                    new_en = sum(es[key_] for key_ in new_soln)
                    new_en += sum(self.get_pairwise(d, ep, key1_, key2_, add_cst_score, vdW_tolerance=vdW_tolerance)
                                    for key1_, key2_ in itertools.combinations(new_soln, 2))
                    if new_en < en_soln:
                        en_soln = new_en
                        en_solns.append(en_soln)
                        soln = new_soln
                        solns.append(soln)
                        consecutive_fails = 0
                    elif new_en == en_soln:
                        consecutive_fails += 1
                    else:
                        if ~np.isnan(new_en):
                            p = np.exp((en_soln - new_en) / kt)
                            if p >= np.random.rand():
                                en_soln = new_en
                                en_solns.append(en_soln)
                                soln = new_soln
                                solns.append(soln)
                                consecutive_fails = 0
                            else:
                                consecutive_fails += 1
                        else:
                            consecutive_fails += 1
                    if consecutive_fails > 200:
                        print('    MC converged')
                        break
            best_soln_index = np.array(en_solns).argmin()
            opt_en = en_solns[best_soln_index]
            opt_soln = solns[best_soln_index]
            opt_vdm_names_sets = [tuple(solns[i]) for i in range(len(en_solns)) if en_solns[i] == opt_en]

            # for site_index in sites_index:
            #     site = sites[site_index]
            #     for vdm_name in d[site].keys():
            #         new_soln = opt_soln.copy()
            #         new_soln[site_index] = (site, vdm_name)
            #         new_en = sum(es[key_] for key_ in new_soln)
            #         new_en += sum(self.get_pairwise(d, ep, key1_, key2_, add_cst_score)
            #                          for key1_, key2_ in itertools.combinations(new_soln, 2))
            #         if new_en < opt_en:
            #             opt_en = new_en
            #             opt_soln = new_soln
            #
            # opt_vdm_names_sets = [tuple(opt_soln)]
            # for site_index in sites_index:
            #     site = sites[site_index]
            #     for vdm_name in d[site].keys():
            #         new_soln = opt_soln.copy()
            #         new_soln[site_index] = (site, vdm_name)
            #         new_en = sum(es[key_] for key_ in new_soln)
            #         new_en += sum(self.get_pairwise(d, ep, key1_, key2_, add_cst_score)
            #                          for key1_, key2_ in itertools.combinations(new_soln, 2))
            #         if new_en == opt_en:
            #             opt_vdm_names_sets.append(tuple(new_soln))

            opt_vdm_names_sets = list(set(opt_vdm_names_sets))
            opt_vdm_names = opt_soln

        if len(opt_vdm_names_sets) > 1:
            # select vdm set that minimizes average distance to CGs.
            if len(set(opt_vdm_names_sets)) > 1:
                avg_dists = []
                for names_set in opt_vdm_names_sets:
                    dists = []
                    for vdmname in names_set:
                        if vdmname[1] is None:
                            continue
                        dists.append(d[vdmname[0]][vdmname[1]]['dist_to_query'].iat[0])
                    avg_dists.append(np.mean(dists))
                index_lowest_dist = sorted(list(range(len(opt_vdm_names_sets))), key=lambda x: avg_dists[x])[0]
                opt_vdm_names = opt_vdm_names_sets[index_lowest_dist]

        # self.opt_en_sidechains = np.sum(es_no_cst[key_] for key_ in opt_vdm_names)
        self.opt_en = sum(es_no_cst[key_] for key_ in opt_vdm_names)
        self.opt_en += sum(ep[key1_][key2_] 
                           for key1_, key2_ in itertools.combinations(opt_vdm_names, 2))

        # opt_vdm_names = list(opt_vdm_names)
        # new_opt_en = self.opt_en
        # old_opt_vdm_names = opt_vdm_names.copy()
        # for vdmname in old_opt_vdm_names:
        #     if vdmname[1] is None:
        #         poss_additions = []
        #         for v in d[vdmname[0]].keys():
        #             if v is None:
        #                 continue
        #             new_opt_vdm_names = old_opt_vdm_names.copy()
        #             new_opt_vdm_name = (vdmname[0], v)
        #             new_opt_vdm_names.append(new_opt_vdm_name)
        #             new_en = sum(es[key_] for key_ in new_opt_vdm_names)
        #             new_en += sum(self.get_pairwise(d, ep, key1_, key2_, add_cst_score)
        #                             for key1_, key2_ in itertools.combinations(new_opt_vdm_names, 2))
        #             if new_en < self.opt_en - C_score_threshold:
        #                 poss_additions.append((new_en, new_opt_vdm_name))
        #         if len(poss_additions) > 0:
        #             best_addition = sorted(poss_additions)[0]
        #             print('Adding', best_addition[1], 'to opt vdMs')
        #             opt_vdm_names.append(best_addition[1])
        #             new_opt_en += (best_addition[0] - self.opt_en)
        # self.opt_en = new_opt_en

        if len(contact_vdms) > 0:
            for key1_, key2_ in itertools.combinations(opt_vdm_names, 2):
                # ep[(site_i, vdm_name_i)][(site_j, vdm_name_j)] = -1 * contact_vdms[(vdm_name_j, vdm_name_i)]
                if (key1_[1], key2_[1]) in contact_vdms:
                    self.first_shell_contact_vdms[(key1_[1], key2_[1])] = contact_vdms[(key1_[1], key2_[1])]
                    self.pairwise_scores[(key1_[1], key2_[1])] = ep[key1_][key2_]
                if (key2_[1], key1_[1]) in contact_vdms:
                    self.first_shell_contact_vdms[(key2_[1], key1_[1])] = contact_vdms[(key2_[1], key1_[1])]
                    self.pairwise_scores[(key2_[1], key1_[1])] = ep[key2_][key1_]

        dfs = []
        for vdmname in opt_vdm_names:
            if vdmname[1] is None:
                continue
            dfs.append(d[vdmname[0]][vdmname[1]])
        if len(dfs) > 0:
            vdm_df = fast_concat(dfs)
            # self.opt_vdms_sidechains_names = list(vdm_df.groupby(self.groupby).groups.keys())
            self.opt_vdms_sidechains = vdm_df
            self.opt_vdms_names = sorted(vdm_df.groupby(self.groupby).groups.keys())
            cols = self.groupby.copy()
            cols.append('score')
            # if 'score' in self.vdms.columns:
            #     cols_no_score = self.vdms.columns.to_list()
            #     score_index = self.vdms.columns.get_loc('score')
            #     cols_no_score.pop(score_index)
            #     self.vdms = self.vdms[cols_no_score]
            self.opt_vdms = merge(self.vdms, vdm_df[cols].drop_duplicates(), on=self.groupby)
            print(self.identifier, self.opt_en, self.opt_vdms_names)
        else:
            print(self.identifier, 'No vdMs worth keeping.', opt_vdm_names)

        self.es = es
        self.ep = ep
        self.es_no_cst = es_no_cst
        self.opt_vdm_names = opt_vdm_names

    @staticmethod
    def _get_buns(df1, df2):
        con = Contact(df1.copy(), df2.copy())
        con.find()
        df_con_hb = con.df_contacts[con.df_contacts.contact_type == 'hb'][
            ['resname_q', 'name_q', 'seg_chain_resnum_q']].drop_duplicates()
        merged = merge(df1, df_con_hb, left_on=['resname', 'name', 'seg_chain_resnum'],
                          right_on=['resname_q', 'name_q', 'seg_chain_resnum_q'], how='outer',
                          indicator=True)
        # bun_atoms = merged[merged._merge == 'left_only'].drop('_merge', axis=1)
        bun_atoms = merged[merged._merge == 'left_only']
        bun_atoms = bun_atoms.drop(columns=['resname_q', 'name_q', 'seg_chain_resnum_q', '_merge'])
        return bun_atoms

    def _set_buns(self, template, burial_threshold=1, exclude_mc_hb=True):
        if self.ligand_resname is None:
            self.ligand_resname = self.ligand.resname.iat[0]

        if len(self.opt_vdms_sidechains) > 0:
            df = concat((self.ligand, self.opt_vdms_sidechains))
            df['dist_to_template_hull'] = template.alpha_hull.get_pnts_distance(df[['c_x', 'c_y', 'c_z']].values)
        else:
            df = self.ligand
        if exclude_mc_hb:
            df_wt = df
        else:
            df_wt = concat((df, template.dataframe))
        df_wt = df_wt[(~df_wt.c_A1_x.isna()) | (~df_wt.c_D_x.isna())]
        df_don = df[(~df.c_D_x.isna()) & (df.dist_to_template_hull > burial_threshold)]
        if len(df_don) > 0:
            df_don = df_don[~df_don.apply(get_heavy, axis=1)]
            bun_dons = self._get_buns(df_don, df_wt)
            self.buried_unsat_sc_donor_atoms = bun_dons[bun_dons.resname != self.ligand_resname]
            self.buried_unsat_lig_donor_atoms = bun_dons[bun_dons.resname == self.ligand_resname]
        else:
            self.buried_unsat_sc_donor_atoms = pd.DataFrame(columns=self.groupby)
            self.buried_unsat_lig_donor_atoms = pd.DataFrame(columns=self.groupby)
        
        df_acc = df[(~df.c_A1_x.isna()) & (df.dist_to_template_hull > burial_threshold)]
        if len(df_acc) > 0:
            df_acc = df_acc[df_acc.apply(get_heavy, axis=1)]
            df_acc[['c_D_x', 'c_D_y', 'c_D_z']] = np.nan
            bun_accs = self._get_buns(df_acc, df_wt)
            self.buried_unsat_sc_acceptor_atoms = bun_accs[bun_accs.resname != self.ligand_resname]
            self.buried_unsat_lig_acceptor_atoms = bun_accs[bun_accs.resname == self.ligand_resname]
        else:
            self.buried_unsat_sc_acceptor_atoms = pd.DataFrame(columns=self.groupby)
            self.buried_unsat_lig_acceptor_atoms = pd.DataFrame(columns=self.groupby)

    def set_buried_unsatisfied(self, template, burial_threshold=0.5, exclude_mc_hb=False):
        self._set_buns(template, burial_threshold=burial_threshold, exclude_mc_hb=exclude_mc_hb)
        self.num_buried_unsat_sc_donor_atoms = len(self.buried_unsat_sc_donor_atoms)
        self.num_buried_unsat_lig_donor_atoms = len(self.buried_unsat_lig_donor_atoms)
        self.num_buried_unsat_sc_acceptor_atoms = len(self.buried_unsat_sc_acceptor_atoms)
        self.num_buried_unsat_lig_acceptor_atoms = len(self.buried_unsat_lig_acceptor_atoms)
        self.num_buried_unsat_donor_atoms = self.num_buried_unsat_sc_donor_atoms + self.num_buried_unsat_lig_donor_atoms
        self.num_buried_unsat_acceptor_atoms = self.num_buried_unsat_sc_acceptor_atoms + self.num_buried_unsat_lig_acceptor_atoms

    def store_buried_unsatisfied(self):
        if len(self.buried_unsat_sc_acceptor_atoms) > 0:
            acc_gr_names = self.buried_unsat_sc_acceptor_atoms.groupby(self.groupby).groups.keys()
            self.stored_buried_unsat_gr_names.update(acc_gr_names)
        if len(self.buried_unsat_sc_donor_atoms) > 0:
            don_gr_names = self.buried_unsat_sc_donor_atoms.groupby(self.groupby).groups.keys()
            self.stored_buried_unsat_gr_names.update(don_gr_names)

    def set_poss_vdms_for_buried_unsatisfied(self, template, allowed_amino_acids='hb_set', allowed_seg_chain_resnums=None):
        # return vdms[cg][aa][is_acceptor or is_donor] = [list of (rota_info, cg_info)]
        # rota_info = seg_aa, chain_aa, res_aa
        # cg_info = vdm_group_name in pose with cg that coords will be extracted from to lookup nbrs.
        # Might want to add an argument allowing specification for which residues to allow at buried, intermediate, exposed positions.
        if allowed_amino_acids is None:
            # will include more hydrophobics for bb h-bonds
            allowed_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        elif allowed_amino_acids == 'hb_set':
            allowed_amino_acids = set('ADEGHKMNQRSTWY')
        template_ca = template.dataframe[template.dataframe['name'] == 'CA']
        ca_coords = template_ca[['c_x', 'c_y', 'c_z']].values
        ca_seg_chain_resnums = template_ca['seg_chain_resnum'].values
        if allowed_seg_chain_resnums is None:
            allowed_seg_chain_resnums = set(ca_seg_chain_resnums)
        nbrs_template_ca = NearestNeighbors(radius=15).fit(ca_coords)
        bun_grs = [self.buried_unsat_sc_acceptor_atoms.groupby(self.groupby),
               self.buried_unsat_sc_donor_atoms.groupby(self.groupby),]
        labels = ['is_acceptor', 'is_donor']
        opt_seg_chain_resnums = {ovn[-1] for ovn in self.opt_vdms_names}
        vdms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
        for bun_grs_, label in zip(bun_grs, labels):
            for n, df in bun_grs_:
                if n in self.stored_buried_unsat_gr_names:
                    # print('skipping because have seen before', n)
                    continue
                for _, row in df.iterrows():
                    cg_resname = row['resname']
                    cg_name = row['name']
                    coords = row[['c_x', 'c_y', 'c_z']].values.reshape(1,-1)
                    ind_neighbors = nbrs_template_ca.radius_neighbors(coords, return_distance=False)
                    scrns = [ca_seg_chain_resnums[j] for j in ind_neighbors[0] 
                            # if ca_seg_chain_resnums[j] != row['seg_chain_resnum']]
                            if ca_seg_chain_resnums[j] not in opt_seg_chain_resnums]
                    for cg in cg_dicts.keys():
                        if cg in ['bb_cnh', 'bb_cco']:
                            continue
                        if cg_resname in cg_dicts[cg]:
                            if cg_name in set(cg_dicts[cg][cg_resname]):
                                for aa in allowed_amino_acids:
                                    for scrn in scrns:
                                        if scrn in allowed_seg_chain_resnums:
                                            vdms[cg][inv_one_letter_code[aa]][label][scrn].add((n,))
        self.poss_vdms_for_buried_unsatisfied =  vdms
    
    def cleanup(self):
        self.poss_vdms_for_buried_unsatisfied = None

    def find_vdms_for_buried_unsatisfied(self, template, vdms, path_to_nbrs_database, distance_metric='rmsd',
                                        rmsd=0.5, maxdist=0.65, specific_seg_chain_resnums_only=None,
                                        filter_by_phi_psi=False, filter_by_phi_psi_exclude_sc=True,
                                        ):
        """ 
        This function should be moved outside to Sample class. The vdms should contain a pose index in them.
        You should make a concatenated vdm dict of all poses.  Then run this function on that, so you only
        load the nbrs once. It's slow to do i/o so many times, I think.  Then once vdms are found,
        group all according to cg/aa, then load vdms (remove clashing) and add to appropriate pose indices.  
        Also need a way to track which buried unsats in a pose were already search for, so you don't try 
        satisfying them again in a recursive search.

        Could select best second shell irrespective of 3rd shell, then freeze that in opt_vdms for recursive search?
        Save 2nd shell possibilites in case can't find good 3rd shell for best 2nd shell. Choose 2nd shell with best
        score and least number of buried unsats?
        """
        # Might want to make a function that finds vdms for one residue only. Hack below
        if specific_seg_chain_resnums_only:
            vdms_old = vdms
            vdms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for cg in vdms_old.keys():
                for aa in vdms_old[cg].keys():
                    for label in vdms_old[cg][aa].keys():
                        for (scrn, vdm_names) in vdms_old[cg][aa][label]:
                            if vdm_names[0][-1] in specific_seg_chain_resnums_only:
                                vdms[cg][aa][label].append((scrn, vdm_names))

        contact_vdms = defaultdict(set)
        if self.opt_vdms_grs is None:
            self.opt_vdms_grs = self.opt_vdms.groupby(self.groupby)
        # self.vdms_grs = self.vdms_sidechains.groupby(self.groupby)
        if distance_metric == 'rmsd':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_rmsd/'
        elif distance_metric == 'maxdist':
            path_to_nbrs_database_ = path_to_nbrs_database + 'vdMs_cg_nbrs_maxdist/'
        path_to_nbrs_database_scores = path_to_nbrs_database + 'vdMs_cg_nbrs_scores/'
        path_to_nbrs_database_groupnames = path_to_nbrs_database + 'vdMs_cg_nbrs_groupnames/'
        print('finding vdM nbrs for buried unsatisfied polar atoms in first shell...')
        for cg in vdms.keys():
            if cg == 'ccoh' and 'coh' in vdms.keys():
                continue
            print('\t', cg)
            if cg not in os.listdir(path_to_nbrs_database_):
                print('\t\t CG not found in database. Skipping...')
                continue
            cg_df = cg_dfs[cg]
            for aa in vdms[cg].keys():
                print('\t\t', aa)
                if aa + '.pkl' not in os.listdir(path_to_nbrs_database_ + cg):
                    print('\t\t\t AA not found in CG database. Skipping...')
                    continue
                if distance_metric == 'rmsd':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    num_cg_atoms = nbrs._fit_X.shape[1] / 3
                    radius = rmsd * np.sqrt(num_cg_atoms)
                elif distance_metric == 'maxdist':
                    with open(path_to_nbrs_database_ + cg + '/' + aa + '.pkl', 'rb') as f:
                        nbrs = pickle.load(f)
                    radius = maxdist

                scores = pd.read_parquet(path_to_nbrs_database_scores + cg + '/' + aa + '.parquet.gzip')
                score_col_dict = {colname: i for i, colname in enumerate(scores.columns)}
                scores = scores.values

                with open(path_to_nbrs_database_groupnames + cg + '/' + aa + '.pkl', 'rb') as f:
                    groupnames = pickle.load(f)

                for acc_or_don in vdms[cg][aa].keys():
                    for rota_info, cg_info in vdms[cg][aa][acc_or_don]:
                        seg_aa, chain_aa, res_aa = rota_info
                        if (seg_aa, chain_aa, res_aa) not in template.phi_psi_dict:
                            df_targ_res = template.dataframe[template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)]
                            template_phi = df_targ_res['phi'].iat[0]
                            template_psi = df_targ_res['psi'].iat[0]
                            template.phi_psi_dict[(seg_aa, chain_aa, res_aa)] = (template_phi, template_psi)
                        else:
                            template_phi, template_psi = template.phi_psi_dict[(seg_aa, chain_aa, res_aa)]

                        if (seg_aa, chain_aa, res_aa) not in template.transformations_to_ideal_ala:
                            df_aa = template.dataframe[(template.dataframe['segment'] == seg_aa) &
                                                    (template.dataframe['chain'] == chain_aa) &
                                                    (template.dataframe['resnum'] == res_aa)].copy()
                            m_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                            for _name in ['N', 'CA', 'C']])
                            t_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']].values[:3]
                            R, mob_com, targ_com = get_rot_trans(m_coords, t_coords)
                            template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)] = (R, mob_com, targ_com)
                        
                        for vdm_cg in cg_info:
                            vdm = self.opt_vdms_grs.get_group(vdm_cg)
                            dfy = pd.merge(cg_df, vdm[vdm.chain=='X'], on=['name', 'resnum', 'resname'])
                            coords = dfy[['c_x', 'c_y', 'c_z']].values
                            R, mob_com, targ_com = template.transformations_to_ideal_ala[(seg_aa, chain_aa, res_aa)]
                            dfy[['c_x', 'c_y', 'c_z']] = apply_transform(R, mob_com, targ_com, coords)
                            coords = dfy[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)
                            try:
                                dists, inds = nbrs.radius_neighbors(coords, radius=radius, return_distance=True)
                            except Exception as e:
                                print(e)
                                print('The pose possibily contains duplicate vdMs.  Perhaps \
                                    check the CG_ligand_coverage column of the ligand.txt file for \
                                    overlapping atoms in different CG_ligand_coverage groups.')
                                print('culprit vdM:', cg, aa, vdm_cg)
                                continue
                            dists, inds = dists[0], inds[0]
                            if cg in cgs_that_flip:
                                dfy['chain'] = dfy['chain_x']
                                coords = flip_cg_coords(dfy).reshape(1, -1)
                                dists_flip, inds_flip = nbrs.radius_neighbors(coords, radius=radius, return_distance=True)
                                dists_flip, inds_flip = dists_flip[0], inds_flip[0]
                                if dists_flip.size > 0 and dists.size > 0:
                                    inds = np.concatenate((inds, inds_flip))
                                    inds, index = np.unique(inds, return_index=True)
                                    dists = np.concatenate((dists, dists_flip))[index]
                                elif dists_flip.size > 0:
                                    dists = dists_flip
                                    inds = inds_flip
                            if dists.size > 0: # If there are nbrs...
                                if filter_by_phi_psi or filter_by_phi_psi_exclude_sc:
                                    score_col_contact_type_ind = score_col_dict['contact_type']
                                    contact_types = scores[inds, score_col_contact_type_ind].astype(str)
                                    score_col_phi_ind = score_col_dict['phi']
                                    phis = scores[inds, score_col_phi_ind].astype(float)
                                    phi_diffs = get_angle_diff(phis, template_phi)
                                    score_col_psi_ind = score_col_dict['psi']
                                    psis = scores[inds, score_col_psi_ind].astype(float)
                                    psi_diffs = get_angle_diff(psis, template_psi)
                                    indices = np.arange(inds.size)
                                    passed_phi_psi_filter = np.zeros(indices.size, dtype=bool)
                                    for contact_type in phi_psi_dict.keys():
                                        if filter_by_phi_psi_exclude_sc:
                                            if contact_type == 'sc' and aa != 'GLY':
                                                mask = contact_types == contact_type
                                                passed_phi_psi_filter[mask] = True
                                                continue
                                        mask = contact_types == contact_type
                                        if not mask.any():
                                            continue
                                        phi_diffs_masked = phi_diffs[mask]
                                        phi_mask = phi_diffs_masked <= 2 * phi_psi_dict[contact_type]['phi']
                                        psi_diffs_masked = psi_diffs[mask]
                                        psi_mask = psi_diffs_masked <= 2 * phi_psi_dict[contact_type]['psi']
                                        phi_psi_mask = phi_mask & psi_mask
                                        passed_phi_psi_filter[indices[mask][phi_psi_mask]] = True
                                    if not passed_phi_psi_filter.any():
                                        continue
                                                                    
                                    inds = inds[passed_phi_psi_filter]
                                    dists = dists[passed_phi_psi_filter]
                                # acc_or_don == 'is_acceptor' or 'is_donor'
                                is_acc_or_don = scores[inds, score_col_dict[acc_or_don]].astype(bool)
                                inds = inds[is_acc_or_don]
                                dists = dists[is_acc_or_don]
                                if len(inds) == 0:
                                    continue

                                # ss = ''
                                # if bb_dep == 1:
                                #     res_filter = template.dataframe['seg_chain_resnum'] == (seg_aa, chain_aa, res_aa)
                                #     abple = template.dataframe[res_filter]['ABPLE'].iat[0]
                                #     ss = 'ABPLE_' + abple
                                # elif bb_dep == 0:
                                #     ss = 'bb_ind'
                                # score_col = 'C_score_' + ss
                                # score_col_ind = score_col_dict[score_col]
                                # scores_ = scores[inds, score_col_ind].astype(float)
                                # not_na = ~np.isnan(scores_)
                                # scores_ = scores_[not_na]
                                # inds = inds[not_na]
                                # dists = dists[not_na]
                                # if len(scores_) == 0:  # No ss score for vdm so give it the lowest ss score possible.
                                #     continue
                                # if use_hb_scores:
                                #     is_hb = scores[inds, score_col_dict['hbond']].astype(bool)
                                #     if is_hb.any():
                                #         score_col = 'C_score_hb_' + ss
                                #         score_col_ind = score_col_dict[score_col]
                                #         scores_[is_hb] = scores[inds[is_hb], score_col_ind]
                                #         not_na = ~np.isnan(scores_)
                                #         scores_ = scores_[not_na]
                                #         inds = inds[not_na]
                                #         dists = dists[not_na]
                                #         if len(scores_) == 0:
                                #             continue
                                #     else:
                                #         continue

                                # ind_lowest_dist = np.argmin(dists)
                                # dist_lowest = dists[ind_lowest_dist]
                                # if distance_metric == 'rmsd':
                                #     dist_lowest = dist_lowest / np.sqrt(num_cg_atoms)
                                # score_lowest_dist = scores_[ind_lowest_dist]
                                # best_groupname = groupnames[inds[ind_lowest_dist]]
                                if distance_metric == 'rmsd':
                                    nbr_groupnames = [(cg, aa, dist / np.sqrt(num_cg_atoms), groupnames[m]) for dist, m in zip(dists, inds)]
                                else:
                                    nbr_groupnames = [(cg, aa, dist, groupnames[m]) for dist, m in zip(dists, inds)]
                                contact_vdms[(rota_info, cg_info)].update(nbr_groupnames)
        # best_contact_vdms = {}
        # for key_ in contact_vdms.keys():
        #     # best_contact_vdms[key_] = sorted(contact_vdms[key_], key=lambda x: x[0], reverse=True)[0]
        #     best_contact_vdms[key_] = sorted(contact_vdms[key_], key=lambda x: x[1])[0]
        return contact_vdms

    def print_to_energy_table(self, outdir='./', filename=None, filename_tag='', tag=None):
        if outdir[-1] != '/':
            outdir += '/'

        try:
            os.makedirs(outdir)
        except:
            pass

        if tag is None:
            tag = self.filename

        if filename is None:
            filename = 'energy_table' + filename_tag + '.csv'

        write_header = True
        if filename in os.listdir(outdir):
            write_header = False

        f = open(outdir + filename, 'a')
        if write_header:
            f.write(','.join(['score', 'num_covered_ligand_CGs', 'num_ligand_apolar_heavy_atoms_buried',
                             'frac_ligand_apolar_heavy_atoms_buried',
                              'ligand_mean_dist_buried', 'ligand_std_dist_buried', 
                              'ligand_num_bun_donor_atoms', 'ligand_num_bun_acceptor_atoms',
                              'sidechain_num_bun_donor_atoms', 'sidechain_num_bun_acceptor_atoms',
                              'total_num_bun_donor_atoms',
                             'total_num_bun_acceptor_atoms', 'total_num_bun_atoms', 'pose_identifier',
                              'pose_rank', 'tag']) + '\n')
        score = np.round(self.opt_en, 6)
        num_covered_ligand_CGs = len(set(self.opt_vdms.CG_ligand_coverage))
        num_ligand_apolar_heavy_atoms_buried = self.ligand['heavy_atoms_buried'].iat[0]
        frac_ligand_apolar_heavy_atoms_buried = self.ligand['frac_heavy_atoms_buried'].iat[0]
        lig_dists_to_hull = self.ligand['dist_to_template_hull'][self.ligand.apply(get_heavy, axis=1)]
        mean_lig_dist = np.round(lig_dists_to_hull.mean(), 5)
        std_lig_dist = np.round(lig_dists_to_hull.std(), 5)
        lig_num_bun_don = self.num_buried_unsat_lig_donor_atoms
        lig_num_bun_acc = self.num_buried_unsat_lig_acceptor_atoms
        sc_num_bun_don = self.num_buried_unsat_sc_donor_atoms
        sc_num_bun_acc = self.num_buried_unsat_sc_acceptor_atoms
        num_bun_don = self.num_buried_unsat_donor_atoms
        num_bun_acc = self.num_buried_unsat_acceptor_atoms
        total_num_bun = num_bun_don + num_bun_acc
        vars = [score, num_covered_ligand_CGs, num_ligand_apolar_heavy_atoms_buried,
                frac_ligand_apolar_heavy_atoms_buried,
                mean_lig_dist, std_lig_dist, lig_num_bun_don, lig_num_bun_acc, sc_num_bun_don, sc_num_bun_acc,
                num_bun_don, num_bun_acc, total_num_bun, self.identifier,
                self.rank, tag]
        f.write(','.join([str(s) for s in vars]) + '\n')
        f.close()

    def print_opt_pdb(self, template, outdir='./', filename=None, tag='', include_CG=False,
                      include_template=True, label_vdM_segment_X=False,
                      print_double_agents=False, include_ligand=True, keep_template_H=True, exclude_backbone=False,
                      vdW_tolerance=0.1):

        #In future, write code that puts path to the input combs script that generated the pdb file in the header
        if outdir[-1] != '/':
            outdir += '/'

        try:
            os.makedirs(outdir)
        except:
            pass

        if filename is None:
            filename = str(self.rank) + '_' + 'pose' + str(self.identifier) + tag + '.pdb'

        self.filename = filename

        outpath = outdir + filename

        template = template.dataframe
        template['score'] = 0
        template['resname_rota'] = template['resname']

        if len(self.opt_vdms) == 0:
            vdms = DataFrame(columns=['c_x', 'c_y', 'c_z', 'name', 'resname_rota',
                                      'resnum', 'resname', 'chain', 'segment', 'score'])
        else:
            vdms = self.opt_vdms.copy()
            if not print_double_agents:
                _vdms = []
                vdms_grs = vdms.groupby('seg_chain_resnum')
                num_sites = len(vdms_grs)
                for n_segchres, vdm_segchres in vdms_grs:
                    num_vdms = len(vdm_segchres[self.groupby].drop_duplicates())
                    if num_sites > 1 and num_vdms > 1:
                        cla = Clash(vdm_segchres[vdm_segchres.chain=='X'].copy(),
                                    vdms[(vdms.chain == 'X') & (vdms.seg_chain_resnum != n_segchres)].copy(),
                                    **dict(tol=vdW_tolerance))
                        cla.set_grouping(self.groupby)
                        cla.find()
                        if len(cla.dfq_clash_free) == 0:
                            print('Pose', self.rank, '(', 'Index', self.identifier, ')', 'contains clashing sidechains...')
                        else:
                            vdm_segchres_Y = vdm_segchres[vdm_segchres.chain=='Y']
                            # vdm_segchres = merge(vdm_segchres, cla.dfq_clash_free[self.groupby].drop_duplicates(),
                            #                      on=self.groupby)
                            # cla.dfq_clash_free.drop(columns='num_tag', inplace=True)
                            vdm_segchres = fast_concat([cla.dfq_clash_free, vdm_segchres_Y])
                            vdm_segchres = vdm_segchres.drop_duplicates()
                    if include_CG:
                        cgs = vdm_segchres[vdm_segchres.chain == 'Y']
                        if len(cgs) > 0:
                            _vdms.append(cgs)
                    old_sc = 100000
                    best_vdm = None
                    # sidechain_present = 'X' in vdm_segchres['chain']
                    vdm_segchres_X = vdm_segchres[vdm_segchres.chain=='X']
                    for n_, vdm_vdm in vdm_segchres_X.groupby(self.groupby):
                        sc = vdm_vdm['score'].iat[0]
                        if sc < old_sc:
                            # if sidechain_present:
                            #     if 'X' in vdm_vdm['chain']:
                            #         best_vdm = vdm_vdm
                            # else:
                            best_vdm = vdm_vdm
                    if best_vdm is not None:
                        _vdms.append(best_vdm)
                vdms = fast_concat(_vdms)
                vdms = vdms.drop_duplicates()
            vdms = merge(vdms, template[['resnum', 'chain', 'segment', 'seg_chain_resnum']].drop_duplicates(),
                        on='seg_chain_resnum', suffixes=['', '_t'])

        if label_vdM_segment_X:
            vdms['segment_t'] = 'X'
        else:
            vdms['segment_t'] = vdms['segment']
        vdms.loc[vdms.chain == 'Y', 'segment_t'] = 'Y'
        vdms.loc[vdms.chain == 'Y', 'chain_t'] = 'Y'

        if not include_CG:
            vdms = vdms[vdms.chain == 'X']

        seg_chain_resnums = set(vdms.seg_chain_resnum)

        if keep_template_H:
            template = template[~template.apply(get_HA3, args=(seg_chain_resnums,), axis=1)].copy()
            template.loc[template.apply(get_HA2, args=(seg_chain_resnums,), axis=1), 'name'] = 'HA'
            template = template[~((template.seg_chain_resnum.isin(set(vdms[vdms.resname_rota=='GLY'].seg_chain_resnum))) &
                                (template['name'] == 'HA'))].copy()
        else:
            template = template[template.apply(get_heavy, axis=1)].copy()

        if not include_template and not exclude_backbone:
            template = merge(template, vdms['seg_chain_resnum'].drop_duplicates(), on='seg_chain_resnum')
        elif not include_template and exclude_backbone:
            template = template[:0]
        template['resnum_t'] = template['resnum']
        template['chain_t'] = template['chain']
        template['segment_t'] = template['segment']
        ligand = self.ligand.copy()
        ligand['resname_rota'] = ligand['resname']
        ligand['resnum_t'] = ligand['resnum']
        ligand['chain_t'] = 'L'
        ligand['segment_t'] = 'L'
        ligand['score'] = self.opt_en

        cols = ['c_x', 'c_y', 'c_z', 'name', 'resnum_t', 'chain_t', 'segment_t', 'seg_chain_resnum', 'resname', 'score']
        vdms = concat([vdms, template[cols]])
        vdms.reset_index(drop=True, inplace=True)
        for n, g in vdms.groupby('seg_chain_resnum'):
            g_ = g['resname_rota'][~g.resname_rota.isna()]
            if len(g_) > 0:
                resname = g_.iat[0]
            else:
                resname = g['resname'].iat[0]
            vdms.loc[g.index, 'resname_rota'] = resname

        cols = ['c_x', 'c_y', 'c_z', 'name', 'resname_rota', 'resnum_t', 'chain_t', 'segment_t', 'score']

        if include_ligand:
            vdms = concat([vdms, ligand[cols]])

        vdms = vdms.sort_values(['seg_chain_resnum', 'CG', 'rota', 'name'])
        scores = vdms['score'].values

        num_set = set('0123456789')
        atom_names = []
        four_spaces = '    '
        three_spaces = '   '
        for atom_name in vdms['name'].values:
            len_atom_name = len(atom_name)
            if len_atom_name < 4:
                if atom_name[0] in num_set:
                    atom_names.append(atom_name + four_spaces[len_atom_name:])
                else:
                    atom_names.append(' ' + atom_name + three_spaces[len_atom_name:])
            else:
                atom_names.append(atom_name)
        vdms['name'] = atom_names

        ag = AtomGroup()
        ag.setCoords(vdms[['c_x', 'c_y', 'c_z']].values)
        ag.setResnums(vdms['resnum_t'].values)
        ag.setResnames(vdms['resname_rota'].values)
        ag.setNames(vdms['name'].values)
        ag.setChids(vdms['chain_t'].values)
        ag.setSegnames(vdms['segment_t'].values)
        ag.setBetas(scores)

        heteroflags = ag.getSegnames() == 'L'
        ag.setFlags('hetatm', heteroflags)

        occ = np.ones(len(vdms))
        writePDB(outpath, ag, occupancy=occ)

        vars = ['score', 'CG_ligand_coverage', 'dist_to_query', 'segment', 'chain', 'resnum',
                'resname', 'CG_type', 'contact_type',
                'CG', 'rota', 'probe_name']

        # if outpath.split('.')[-1] == 'pdb':
        #     f = open(outpath, 'a')
        # elif outpath.split('.')[-1] == 'gz':
        #     f = gzip.open(outpath, 'ab')

        f = open(outpath, 'a')
        f.write('# COMBS info \n')
        f.write('# Total_COMBS_Energy= ' + str(np.round(self.opt_en, 6)) + ' \n')
        f.write('# Number of covered ligand CGs= ' + str(len(set(self.opt_vdms.CG_ligand_coverage))) + ' \n')
        f.write('# Number of apolar ligand heavy atoms buried= ' + str(self.ligand['heavy_atoms_buried'].iat[0]) + ' \n')
        f.write('# Fraction of apolar ligand heavy atoms buried= ' + str(self.ligand['frac_heavy_atoms_buried'].iat[0]) + ' \n')
        lig_dists_to_hull = self.ligand['dist_to_template_hull'][self.ligand.apply(get_heavy, axis=1)]
        f.write('# Mean distance buried of all ligand heavy atoms= '
                + str(np.round(lig_dists_to_hull.mean(), 5))
                + ' +- ' + str(np.round(lig_dists_to_hull.std(), 5)) + ' \n')
        f.write('# Number buried unsatisfied ligand donor atoms= ' + str(self.num_buried_unsat_lig_donor_atoms) + ' \n')
        f.write('# Number buried unsatisfied ligand acceptor atoms= ' + str(self.num_buried_unsat_lig_acceptor_atoms) + ' \n')
        f.write('# Number buried unsatisfied sidechain donor atoms= ' + str(self.num_buried_unsat_sc_donor_atoms) + ' \n')
        f.write('# Number buried unsatisfied sidechain acceptor atoms= ' + str(self.num_buried_unsat_sc_acceptor_atoms) + ' \n')
        f.write('# Total number buried unsatisfied donor atoms= ' + str(self.num_buried_unsat_donor_atoms) + ' \n')
        f.write('# Total number buried unsatisfied acceptor atoms= ' + str(self.num_buried_unsat_acceptor_atoms) + ' \n')
        f.write('# Total number buried unsatisfied polar atoms= ' + str(self.num_buried_unsat_acceptor_atoms
                                                                        + self.num_buried_unsat_donor_atoms) + ' \n')
        f.write('# BEGIN_COMBS_ENERGIES_TABLE ' + outpath.split('/')[-1] + ' \n')
        f.write(' '.join(vars) + ' \n')

        for n, v in self.opt_vdms.groupby(self.groupby):
            _score = v['score'].iat[0]
            score = str(np.round(_score, 4))
            CG_lig_coverage = str(v['CG_ligand_coverage'].iat[0])
            _dist_to_query = v['dist_to_query'].iat[0]
            dist_to_query = str(np.round(_dist_to_query, 4))
            seg_chain_resnum = v['seg_chain_resnum'].iat[0]
            resnum = str(seg_chain_resnum[2])
            chain = seg_chain_resnum[1]
            segment = seg_chain_resnum[0]
            resname = v['resname_rota'].iat[0]
            CG_type = v['CG_type'].iat[0]
            contact_type = v['contact_type'].iat[0]
            CG = str(v['CG'].iat[0])
            rota = str(v['rota'].iat[0])
            probe_name = v['probe_name'].iat[0]
            _vars = [score, CG_lig_coverage, dist_to_query, segment, chain, resnum,
               resname, CG_type, contact_type,
                CG, rota, probe_name]
            f.write(' '.join(_vars) + ' \n')
        f.write('# END_COMBS_ENERGIES_TABLE ' + outpath.split('/')[-1] + ' \n')

        if len(set(self.lig_additional_vdms.keys()) & set(self.opt_vdms_names)) > 0:
            # contact_vdms[(rota_info, cg_info)].append((score_lowest_dist, dist_lowest, vdm_cg, (cg, aa, best_groupname)))
            vars = ['score', 'dist_to_query', 'vdM_1', 'CG_type', 'CG_group', 'vdM_match']
            f.write('# BEGIN_COMBS_ENERGIES_TABLE_ADDITIONAL_LIGAND ' + outpath.split('/')[-1] + ' \n')
            f.write(', '.join(vars) + ' \n')
            for vdm_gr_name in self.lig_additional_vdms.keys():
                if vdm_gr_name in self.opt_vdms_names:
                    for (cg, cg_gr), _score in self.lig_additional_vdms[vdm_gr_name].items():
                        v = self.contacting_lig_vdms[vdm_gr_name][(cg, cg_gr)]
                        score = str(np.round(_score, 4))
                        _dist_to_query = v[1]
                        dist_to_query = str(np.round(_dist_to_query, 4))
                        vdm_match = v[2]
                        _vars = [score, dist_to_query, vdm_gr_name, cg, cg_gr, vdm_match]
                        f.write(', '.join(str(v) for v in _vars) + ' \n')
            f.write('# END_COMBS_ENERGIES_TABLE_ADDITIONAL_LIGAND ' + outpath.split('/')[-1] + ' \n')

        if len(self.first_shell_contact_vdms) > 0:
            # contact_vdms[(rota_info, cg_info)].append((score_lowest_dist, dist_lowest, vdm_cg, (cg, aa, best_groupname)))
            vars = ['score', 'dist_to_query', 'vdM_1', 'vdM_2', 'vdM_2_specific', 'CG_type', 'resname', 'vdM_match']
            f.write('# BEGIN_COMBS_ENERGIES_TABLE_FIRST_SHELL_PAIRWISE ' + outpath.split('/')[-1] + ' \n')
            f.write(', '.join(vars) + ' \n')
            for (rota_info, cg_info), val in self.first_shell_contact_vdms.items():
                vdm_cg = val[2]
                score_lowest_dist = self.pairwise_scores[(rota_info, cg_info)] #-1 * val[0]
                dist_lowest = val[1]
                cg = val[-1][0]
                aa = val[-1][1]
                best_groupname = val[-1][2]
                _vars = [score_lowest_dist, dist_lowest, rota_info, cg_info, vdm_cg, cg, aa, best_groupname]
                f.write(', '.join(str(v) for v in _vars) + ' \n')
            f.write('# END_COMBS_ENERGIES_TABLE_FIRST_SHELL_PAIRWISE ' + outpath.split('/')[-1] + ' \n')
        f.close()

    @staticmethod
    def print_vdM(df_vdM, outpath='./', tag='', prefix='', b_factor_column=None):
        if len(set(df_vdM['CG_type'])) != 1:
            raise Exception('vdM has more than one CG_type. More than 1 vdM present in df_vdM?')
        tag_ = '_' + df_vdM['CG_type'].iat[0]
        print_dataframe(df_vdM, outpath=outpath, tag=tag_+tag, 
                        prefix=prefix, b_factor_column=b_factor_column)

    def print_all_vdMs(self, outpath='./', tag='', prefix='', b_factor_column=None):
        if type(self.vdms) == list:
            print('No vdMs in pose.')
            return
        for n, vdm in self.vdms.groupby(self.groupby):
            tag_ = '_' + n[-2] # cg_type
            print_dataframe(vdm, outpath=outpath, tag=tag_+tag, 
                            prefix=prefix, b_factor_column=b_factor_column)
