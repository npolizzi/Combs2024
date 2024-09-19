from ..design.constants import cgs_that_flip, resnames_aa_20_join
from ..design.functions import flip_cg_coords, add_slash, df_gly, df_ideal_ala_sc
from ..design.cluster import _make_pairwise_maxdist_mat, _make_pairwise_rmsd_mat
from ..design.probe import parse_probe, _probe
from ..design.transformation import superpose_df
from ..design.template import Template
from ..design._sample import Sample
from ..design.functions import make_lig_atom_type_dict
from ..parse.comb import cg_dicts
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from prody import parsePDB


def parse_vdms_from_probe(df_probe, skipping_number=6):
    vdms = defaultdict(dict)
    bb_names = ['O', 'H', 'C', 'CA', 'N']
    for n, g in df_probe.groupby(['chain1', 'resnum1']):
        rota_resname = g.resname1.iat[0]
        for (chain2, resnum2, cg_resname, cg_name), g_cg in g.groupby(['chain2', 'resnum2', 'resname2', 'name2']):
            m = (chain2, resnum2)
            if cg_name in bb_names:
                if cg_name in ['O', 'C']:
                    cgs = ['bb_cco']
                elif cg_name in ['N', 'H']:
                    cgs = ['bb_cnh']
                else:
                    cgs = ['bb_cnh', 'bb_cco']
                if m[0] != n[0] or ((m[0] == n[0]) and np.abs(m[1] - n[1]) > skipping_number):
                    for cg in cgs:
                        try:
                            vdms[cg][rota_resname].add((n, m))
                        except:
                            vdms[cg][rota_resname] = set()
                            vdms[cg][rota_resname].add((n, m))
            else:
                for cg in cg_dicts.keys():
                    if cg in ['bb_cnh', 'bb_cco']:
                        continue
                    if cg_resname in cg_dicts[cg]:
                        if cg_name in set(cg_dicts[cg][cg_resname]):
                            if (m[0] != n[0]) or np.abs(m[1] - n[1]) > skipping_number:
                                try:
                                    vdms[cg][rota_resname].add((n, m))
                                except:
                                    vdms[cg][rota_resname] = set()
                                    vdms[cg][rota_resname].add((n, m))
    return vdms


def parse_ligand_vdms_from_probe(df_probe, ligand_vdm_correspondence):
    vdms = defaultdict(dict)
    lig_resnames = set(ligand_vdm_correspondence.lig_resname)
    df_probe = df_probe[df_probe.resname2.isin(lig_resnames)]
    if len(df_probe) == 0:
        return
    for n, g in df_probe.groupby(['chain1', 'resnum1']):
        rota_resname = g.resname1.iat[0]
        for (chain2, resnum2, lig_resname, lig_name), g_cg in g.groupby(['chain2', 'resnum2', 'resname2', 'name2']):
            filt = ((ligand_vdm_correspondence['lig_name'] == lig_name) &
                    (ligand_vdm_correspondence['lig_resname'] == lig_resname))
            lig_cg_map = ligand_vdm_correspondence[filt][['CG_type', 'CG_group']].drop_duplicates()
            if len(lig_cg_map) == 0:
                continue
            for cg, cg_gr in lig_cg_map.values:
                try:
                    vdms[cg][rota_resname].add((n, (chain2, resnum2, cg_gr)))
                except:
                    vdms[cg][rota_resname] = set()
                    vdms[cg][rota_resname].add((n, (chain2, resnum2, cg_gr)))
    return vdms


def map_vdms_to_structure(template, vdms, path_to_database, distance_cutoff=0.5,
                          max_distance_cutoff=0.8, strict=False):
    info = []
    for cg in vdms.keys():
        print(cg)
        if cg not in os.listdir(path_to_database):
            print('\t CG not found in database. Skipping...')
            continue
        for aa in vdms[cg].keys():
            print('\t', aa)
            if aa + '.parquet.gzip' not in os.listdir(path_to_database + cg):
                print('\t\t AA not found in CG database. Skipping...')
                continue
            df = pd.read_parquet(path_to_database + cg + '/' + aa + '.parquet.gzip')
            df_cluster_atom_info = df[['chain', 'resname', 'name', 'cluster_order']][
                df['cluster_atom']].drop_duplicates().sort_values(by='cluster_order') #.drop(columns='cluster_order')
            df_centroids = df[['c_x', 'c_y', 'c_z', 'chain', 'resname', 'name', 'cluster_number', 'cluster_order']][
                df['cluster_atom'] & df['centroid']].sort_values(by='cluster_order')
            grs_centroids = df_centroids.groupby('cluster_number')
            for (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg) in vdms[cg][aa]:
                print('\t\t', str((chain_res_aa, resnum_res_aa)), str((chain_res_cg, resnum_res_cg)))
                no_match_found = False
                df_aa = template.dataframe[(template.dataframe['chain'] == chain_res_aa) & (
                            template.dataframe['resnum'] == resnum_res_aa)].copy()
                df_cg = template.dataframe[(template.dataframe['chain'] == chain_res_cg) & (
                            template.dataframe['resnum'] == resnum_res_cg)].copy()
                if aa == 'GLY' and 'HA3' not in df_aa['name'].values:
                    m_coords = df_gly[['c_x', 'c_y', 'c_z']].values[:3]
                    t_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                 for _name in ['N', 'CA', 'C']])
                    df_gly_super = superpose_df(m_coords, t_coords, df_gly, copy_df_mob=True)
                    df_aa = df_aa.append(df_gly_super[df_gly_super['name'] == 'HA3'])
                if aa != 'GLY' and 'CB' not in df_aa['name'].values:
                    m_coords = df_ideal_ala_sc[['c_x', 'c_y', 'c_z']].values[:3]
                    t_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                 for _name in ['N', 'CA', 'C']])
                    df_ala_super = superpose_df(m_coords, t_coords, df_ideal_ala_sc, copy_df_mob=True)
                    df_aa = df_aa.append(df_ala_super[df_ala_super['name'] == 'CB'])
                df_aa['chain'] = 'X'
                df_cg['chain'] = 'Y'
                df_concat = pd.concat([df_aa, df_cg])
                if cg in ['bb_cco', 'bb_cnh']:
                    df_merged = pd.merge(df_cluster_atom_info[['chain', 'name']].drop_duplicates(), df_concat,
                                         on=['chain', 'name'])
                    # df_cluster_atom_info_x = df_cluster_atom_info[df_cluster_atom_info['chain'] == 'X']
                    # df_merged_x = pd.merge(df_cluster_atom_info_x[['chain', 'name', 'cluster_order']].drop_duplicates(), df_aa,
                    #     on=['chain', 'name'])
                    # df_cluster_atom_info_y = df_cluster_atom_info[df_cluster_atom_info['chain'] == 'Y']
                    # df_merged_y = pd.merge(
                    #     df_cluster_atom_info_y[['chain', 'resname', 'name', 'cluster_order']].drop_duplicates(),
                    #     df_cg, on=['chain', 'resname', 'name'])
                    # df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
                else:
                    # df_merged = pd.merge(df_cluster_atom_info, df_concat, on=['chain', 'resname', 'name'])
                    df_cluster_atom_info_x = df_cluster_atom_info[df_cluster_atom_info['chain'] == 'X']
                    df_merged_x = pd.merge(df_cluster_atom_info_x, df_aa, on=['chain', 'name'])
                    df_cluster_atom_info_y = df_cluster_atom_info[df_cluster_atom_info['chain'] == 'Y']
                    df_merged_y = pd.merge(
                        df_cluster_atom_info_y[['chain', 'resname', 'name', 'cluster_order']].drop_duplicates(),
                        df_cg, on=['chain', 'resname', 'name'])
                    df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
                mob_coords = df_merged[['c_x', 'c_y', 'c_z']].values
                cluster_number_match = None
                cluster_maxdists = []
                cluster_real_maxdists = []
                for cluster_number in sorted(grs_centroids.groups.keys()):
                    df_clu = grs_centroids.get_group(cluster_number)

                    all_clu_coords = [df_clu.values[:, :3]]

                    if cg in cgs_that_flip:
                        all_clu_coords.append(flip_cg_coords(df_clu))

                    maxdist = distance_cutoff + 1
                    real_maxdist = max_distance_cutoff + 1
                    _clu_max_dists = []
                    _clu_real_max_dists = []
                    for clu_coords in all_clu_coords:

                        try:
                            coords = np.array([mob_coords, clu_coords], dtype=np.float32)
                        except ValueError:
                            continue

                        maxdist_mat = _make_pairwise_rmsd_mat(coords)  #RMSD
                        maxdist = maxdist_mat[0, 1]
                        real_maxdist_mat = _make_pairwise_maxdist_mat(coords)
                        real_maxdist = real_maxdist_mat[0, 1]
                        if maxdist <= distance_cutoff and real_maxdist <= max_distance_cutoff:
                            cluster_number_match = cluster_number
                            break
                        _clu_max_dists.append(maxdist)
                        _clu_real_max_dists.append(real_maxdist)
                    if maxdist <= distance_cutoff and real_maxdist <= max_distance_cutoff:
                        break
                    if len(_clu_max_dists) > 0:
                        cluster_maxdists.append((min(_clu_max_dists), cluster_number))
                        argmin = np.argmin(np.array(_clu_max_dists))
                        cluster_real_maxdists.append(_clu_real_max_dists[argmin])

                if cluster_number_match is None and len(cluster_maxdists) == 0:
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, None, None, None, None,
                             None, 'none', None, None]
                    no_match_found = True
                elif cluster_number_match is None:
                    if strict:
                        best_index = sorted(range(len(cluster_real_maxdists)), key=lambda x: cluster_real_maxdists[x])[0]
                    else:
                        best_index = sorted(range(len(cluster_maxdists)), key=lambda x: cluster_maxdists[x])[0]
                        # maxdist, cluster_number_match = sorted(cluster_maxdists)[0]
                    maxdist, cluster_number_match = cluster_maxdists[best_index]
                    real_maxdist = cluster_real_maxdists[best_index]
                    abple = df_aa.ABPLE.iat[0]
                    try:
                        c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_abple = None
                    try:
                        c_score_hb_abple = \
                        df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_hb_abple = None
                    c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
                    try:
                        c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
                    except:
                        c_score_hb_bb_ind = None
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cluster_number_match,
                             c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'closest', maxdist, real_maxdist]
                else:
                    abple = df_aa.ABPLE.iat[0]
                    try:
                        c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_abple = None
                    try:
                        c_score_hb_abple = \
                        df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_hb_abple = None
                    c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
                    try:
                        c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
                    except:
                        c_score_hb_bb_ind = None
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cluster_number_match,
                             c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'exact', maxdist, real_maxdist]
                info.append(_info)
                if no_match_found:
                    print('No match for', (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg))
                    print('\t Perhaps residues are missing atoms? If Gly only, check Gly has HA3 atoms.')
    df_info = pd.DataFrame(info, columns=['chain_rota', 'resnum_rota', 'resname_rota', 'chain_CG', 'resnum_CG', 'CG_type',
                                          'cluster_number', 'C_score_bb_ind', 'C_score_hb_bb_ind', 'C_score_ABPLE',
                                          'C_score_hb_ABPLE', 'match_type', 'rmsd_to_centroid', 'max_dist_to_centroid'])
    return df_info

def map_lig_vdms_to_structure(template, vdms, path_to_database, ligand_vdm_correspondence,
                              distance_cutoff=0.5, max_distance_cutoff=0.8, strict=False,
                              ):
    info = []
    for cg in vdms.keys():
        print(cg)
        if cg not in os.listdir(path_to_database):
            print('\t CG not found in database. Skipping...')
            continue
        for aa in vdms[cg].keys():
            print('\t', aa)
            if aa + '.parquet.gzip' not in os.listdir(path_to_database + cg):
                print('\t\t AA not found in CG database. Skipping...')
                continue
            df = pd.read_parquet(path_to_database + cg + '/' + aa + '.parquet.gzip')
            df_cluster_atom_info = df[['chain', 'resname', 'name', 'cluster_order']][
                df['cluster_atom']].drop_duplicates().sort_values(by='cluster_order')

            df_centroids = df[['c_x', 'c_y', 'c_z', 'chain', 'resname', 'name', 'cluster_number', 'cluster_order']][
                df['cluster_atom'] & df['centroid']].sort_values(by='cluster_order')
            grs_centroids = df_centroids.groupby('cluster_number')
            for (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg, cg_group) in vdms[cg][aa]:
                print('\t\t', str((chain_res_aa, resnum_res_aa)), str((chain_res_cg, resnum_res_cg, cg_group)))
                no_match_found = False
                df_aa = template.dataframe[(template.dataframe['chain'] == chain_res_aa) & (
                            template.dataframe['resnum'] == resnum_res_aa)].copy()
                df_cg = template.dataframe[(template.dataframe['chain'] == chain_res_cg) & (
                            template.dataframe['resnum'] == resnum_res_cg)].copy()
                if aa == 'GLY' and 'HA3' not in df_aa['name'].values:
                    m_coords = df_gly[['c_x', 'c_y', 'c_z']].values[:3]
                    t_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                 for _name in ['N', 'CA', 'C']])
                    df_gly_super = superpose_df(m_coords, t_coords, df_gly, copy_df_mob=True)
                    df_aa = df_aa.append(df_gly_super[df_gly_super['name'] == 'HA3'])
                if aa != 'GLY' and 'CB' not in df_aa['name'].values:
                    m_coords = df_ideal_ala_sc[['c_x', 'c_y', 'c_z']].values[:3]
                    t_coords = np.array([df_aa[['c_x', 'c_y', 'c_z']][df_aa['name'] == _name].values[0] 
                                 for _name in ['N', 'CA', 'C']])
                    df_ala_super = superpose_df(m_coords, t_coords, df_ideal_ala_sc, copy_df_mob=True)
                    df_aa = df_aa.append(df_ala_super[df_ala_super['name'] == 'CB'])
                df_aa['chain'] = 'X'
                df_cg['chain'] = 'Y'
                _lig_vdm_corr = ligand_vdm_correspondence[(ligand_vdm_correspondence.CG_type==cg) &
                                                          (ligand_vdm_correspondence.CG_group==cg_group)]
                _df_clu_info = df_cluster_atom_info[df_cluster_atom_info['chain']=='Y']
                df_lig_cluster_atom_info = pd.merge(_df_clu_info, _lig_vdm_corr,
                                                    on=['resname', 'name']).sort_values(by='cluster_order')
                for lig_clu_name, lig_clu_group in df_lig_cluster_atom_info.groupby('resname'):
                    df_lig_cluster_atom_info = lig_clu_group
                    break
                if cg in ['bb_cco', 'bb_cnh']:
                    df_merged_x = pd.merge(df_cluster_atom_info[['chain', 'name', 'cluster_order']].drop_duplicates(), df_aa,
                                         on=['chain', 'name'])
                    df_merged_y = pd.merge(
                        df_lig_cluster_atom_info[['lig_resname', 'lig_name', 'cluster_order']].drop_duplicates(),
                        df_cg, left_on=['lig_resname', 'lig_name'], right_on=['resname', 'name'])
                    df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
                else:
                    df_merged_x = pd.merge(df_cluster_atom_info, df_aa, on=['chain', 'name'])
                    df_merged_y = pd.merge(
                        df_lig_cluster_atom_info[['lig_resname', 'lig_name', 'cluster_order']].drop_duplicates(),
                        df_cg, left_on=['lig_resname', 'lig_name'], right_on=['resname', 'name'])
                    df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
                mob_coords = df_merged[['c_x', 'c_y', 'c_z']].values
                cluster_number_match = None
                cluster_maxdists = []
                cluster_real_maxdists = []
                for cluster_number in sorted(grs_centroids.groups.keys()):
                    df_clu = grs_centroids.get_group(cluster_number)

                    all_clu_coords = [df_clu.values[:, :3]]

                    if cg in cgs_that_flip:
                        all_clu_coords.append(flip_cg_coords(df_clu))

                    maxdist = distance_cutoff + 1
                    real_maxdist = max_distance_cutoff + 1
                    _clu_max_dists = []
                    _clu_real_max_dists = []
                    for clu_coords in all_clu_coords:

                        try:
                            coords = np.array([mob_coords, clu_coords], dtype=np.float32)
                        except ValueError:
                            continue

                        maxdist_mat = _make_pairwise_rmsd_mat(coords)  # RMSD
                        maxdist = maxdist_mat[0, 1]
                        real_maxdist_mat = _make_pairwise_maxdist_mat(coords)
                        real_maxdist = real_maxdist_mat[0, 1]
                        if maxdist <= distance_cutoff and real_maxdist <= max_distance_cutoff:
                            cluster_number_match = cluster_number
                            break
                        _clu_max_dists.append(maxdist)
                        _clu_real_max_dists.append(real_maxdist)
                    if maxdist <= distance_cutoff and real_maxdist <= max_distance_cutoff:
                        break
                    if len(_clu_max_dists) > 0:
                        cluster_maxdists.append((min(_clu_max_dists), cluster_number))
                        argmin = np.argmin(np.array(_clu_max_dists))
                        cluster_real_maxdists.append(_clu_real_max_dists[argmin])
                cg_lig_cov = _lig_vdm_corr['CG_ligand_coverage'].iat[0]

                if cluster_number_match is None and len(cluster_maxdists) == 0:
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cg_group, cg_lig_cov, None, None, None, None,
                             None, 'none', None, None]
                    no_match_found = True
                elif cluster_number_match is None:
                    if strict:
                        best_index = sorted(range(len(cluster_real_maxdists)), key=lambda x: cluster_real_maxdists[x])[
                            0]
                    else:
                        best_index = sorted(range(len(cluster_maxdists)), key=lambda x: cluster_maxdists[x])[0]
                        # maxdist, cluster_number_match = sorted(cluster_maxdists)[0]
                    maxdist, cluster_number_match = cluster_maxdists[best_index]
                    real_maxdist = cluster_real_maxdists[best_index]
                    abple = df_aa.ABPLE.iat[0]
                    try:
                        c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_abple = None
                    try:
                        c_score_hb_abple = \
                            df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_hb_abple = None
                    c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
                    try:
                        c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
                    except:
                        c_score_hb_bb_ind = None
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg,  cg_group, cg_lig_cov, cluster_number_match,
                             c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'closest', maxdist,
                             real_maxdist]
                else:
                    abple = df_aa.ABPLE.iat[0]
                    try:
                        c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_abple = None
                    try:
                        c_score_hb_abple = \
                            df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
                    except KeyError:
                        c_score_hb_abple = None
                    c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
                    try:
                        c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
                    except:
                        c_score_hb_bb_ind = None
                    _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cg_group, cg_lig_cov, cluster_number_match,
                             c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'exact', maxdist,
                             real_maxdist]
                info.append(_info)
                if no_match_found:
                    print('No match for', (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg, cg_group))
                    print('\t Perhaps the ligand txt file contains an error? If Gly only, check Gly has HA3 atoms.')
    df_info = pd.DataFrame(info, columns=['chain_rota', 'resnum_rota', 'resname_rota', 'chain_CG', 'resnum_CG',
                                          'CG_type', 'CG_group', 'CG_ligand_coverage',
                                          'cluster_number', 'C_score_bb_ind', 'C_score_hb_bb_ind',
                                          'C_score_ABPLE',
                                          'C_score_hb_ABPLE', 'match_type', 'rmsd_to_centroid',
                                          'max_dist_to_centroid'])
    return df_info


# def map_lig_vdms_to_structure(template, vdms, path_to_database, ligand_vdm_correspondence, distance_cutoff=0.65):
#     info = []
#     for cg in vdms.keys():
#         print(cg)
#         if cg not in os.listdir(path_to_database):
#             print('\t CG not found in database. Skipping...')
#             continue
#         for aa in vdms[cg].keys():
#             print('\t', aa)
#             if aa + '.parquet.gzip' not in os.listdir(path_to_database + cg):
#                 print('\t\t AA not found in CG database. Skipping...')
#                 continue
#             df = pd.read_parquet(path_to_database + cg + '/' + aa + '.parquet.gzip')
#             df_cluster_atom_info = df[['chain', 'resname', 'name', 'cluster_order']][
#                 df['cluster_atom']].drop_duplicates().sort_values(by='cluster_order')
#
#             df_centroids = df[['c_x', 'c_y', 'c_z', 'chain', 'resname', 'name', 'cluster_number', 'cluster_order']][
#                 df['cluster_atom'] & df['centroid']].sort_values(by='cluster_order')
#             grs_centroids = df_centroids.groupby('cluster_number')
#             for (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg, cg_group) in vdms[cg][aa]:
#                 print('\t\t', str((chain_res_aa, resnum_res_aa)), str((chain_res_cg, resnum_res_cg, cg_group)))
#                 no_match_found = False
#                 df_aa = template.dataframe[(template.dataframe['chain'] == chain_res_aa) & (
#                             template.dataframe['resnum'] == resnum_res_aa)].copy()
#                 df_cg = template.dataframe[(template.dataframe['chain'] == chain_res_cg) & (
#                             template.dataframe['resnum'] == resnum_res_cg)].copy()
#                 df_aa['chain'] = 'X'
#                 df_cg['chain'] = 'Y'
#                 _lig_vdm_corr = ligand_vdm_correspondence[(ligand_vdm_correspondence.CG_type==cg) &
#                                                           (ligand_vdm_correspondence.CG_group==cg_group)]
#                 _df_clu_info = df_cluster_atom_info[df_cluster_atom_info['chain']=='Y']
#                 df_lig_cluster_atom_info = pd.merge(_df_clu_info, _lig_vdm_corr,
#                                                     on=['resname', 'name']).sort_values(by='cluster_order')
#                 for lig_clu_name, lig_clu_group in df_lig_cluster_atom_info.groupby('resname'):
#                     df_lig_cluster_atom_info = lig_clu_group
#                     break
#                 if cg in ['bb_cco', 'bb_cnh']:
#                     df_merged_x = pd.merge(df_cluster_atom_info[['chain', 'name', 'cluster_order']].drop_duplicates(), df_aa,
#                                          on=['chain', 'name'])
#                     df_merged_y = pd.merge(
#                         df_lig_cluster_atom_info[['lig_resname', 'lig_name', 'cluster_order']].drop_duplicates(),
#                         df_cg, left_on=['lig_resname', 'lig_name'], right_on=['resname', 'name'])
#                     df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
#                 else:
#                     df_merged_x = pd.merge(df_cluster_atom_info, df_aa, on=['chain', 'resname', 'name'])
#                     df_merged_y = pd.merge(
#                         df_lig_cluster_atom_info[['lig_resname', 'lig_name', 'cluster_order']].drop_duplicates(),
#                         df_cg, left_on=['lig_resname', 'lig_name'], right_on=['resname', 'name'])
#                     df_merged = pd.concat([df_merged_x, df_merged_y]).sort_values(by='cluster_order')
#                 mob_coords = df_merged[['c_x', 'c_y', 'c_z']].values
#                 cluster_number_match = None
#                 cluster_maxdists = []
#                 for cluster_number in sorted(grs_centroids.groups.keys()):
#                     df_clu = grs_centroids.get_group(cluster_number)
#
#                     all_clu_coords = [df_clu.values[:, :3]]
#
#                     if cg in cgs_that_flip:
#                         all_clu_coords.append(flip_cg_coords(df_clu))
#
#                     maxdist = distance_cutoff + 1
#                     _clu_max_dists = []
#                     for clu_coords in all_clu_coords:
#
#                         try:
#                             coords = np.array([mob_coords, clu_coords], dtype=np.float32)
#                         except ValueError:
#                             continue
#                         maxdist_mat = _make_pairwise_rmsd_mat(coords) #RMSD
#                         maxdist = maxdist_mat[0, 1]
#                         # maxdist_mat = _make_pairwise_maxdist_mat(coords)
#                         # maxdist = maxdist_mat[0, 1]
#                         if maxdist <= distance_cutoff:
#                             cluster_number_match = cluster_number
#                             break
#                         _clu_max_dists.append(maxdist)
#                     if maxdist <= distance_cutoff:
#                         break
#                     if len(_clu_max_dists) > 0:
#                         cluster_maxdists.append((min(_clu_max_dists), cluster_number))
#
#                 if cluster_number_match is None and len(cluster_maxdists) == 0:
#                     _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cg_group,
#                              None, None, None, None, None, 'none', None]
#                     no_match_found = True
#                 elif cluster_number_match is None:
#                     maxdist, cluster_number_match = sorted(cluster_maxdists)[0]
#                     abple = df_aa.ABPLE.iat[0]
#                     try:
#                         c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
#                     except KeyError:
#                         c_score_abple = None
#                     try:
#                         c_score_hb_abple = \
#                         df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
#                     except KeyError:
#                         c_score_hb_abple = None
#                     c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
#                     try:
#                         c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
#                     except:
#                         c_score_hb_bb_ind = None
#                     _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cg_group, cluster_number_match,
#                              c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'closest', maxdist]
#                 else:
#                     abple = df_aa.ABPLE.iat[0]
#                     try:
#                         c_score_abple = df[df.cluster_number == cluster_number_match]['C_score_ABPLE_' + abple].iat[0]
#                     except KeyError:
#                         c_score_abple = None
#                     try:
#                         c_score_hb_abple = \
#                         df[df.cluster_number == cluster_number_match]['C_score_hb_ABPLE_' + abple].iat[0]
#                     except KeyError:
#                         c_score_hb_abple = None
#                     c_score_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_bb_ind'].iat[0]
#                     try:
#                         c_score_hb_bb_ind = df[df.cluster_number == cluster_number_match]['C_score_hb_bb_ind'].iat[0]
#                     except:
#                         c_score_hb_bb_ind = None
#                     _info = [chain_res_aa, resnum_res_aa, aa, chain_res_cg, resnum_res_cg, cg, cg_group, cluster_number_match,
#                              c_score_bb_ind, c_score_hb_bb_ind, c_score_abple, c_score_hb_abple, 'exact', maxdist]
#                 info.append(_info)
#                 if no_match_found:
#                     print('No match for', (chain_res_aa, resnum_res_aa), (chain_res_cg, resnum_res_cg, cg_group))
#                     print('\t Perhaps the ligand txt file contains an error? If Gly only, check Gly has HA3 atoms.')
#
#     df_info = pd.DataFrame(info, columns=['chain_rota', 'resnum_rota', 'resname_rota', 'chain_CG', 'resnum_CG', 'CG_type',
#                                           'CG_group', 'cluster_number', 'C_score_bb_ind', 'C_score_hb_bb_ind',
#                                           'C_score_ABPLE', 'C_score_hb_ABPLE', 'match_type', 'max_dist_to_centroid'])
#     return df_info


def run_lookup(path_to_pdb, path_to_database, outdir, filename='', skipping_number=6, distance_cutoff=0.6,
               probe_segname1='', probe_chain1='', probe_resnum1='',
               probe_segname2='', probe_chain2='', probe_resnum2='',
               probe_sel_criteria='NOT METAL', maxbonded=4, explicit_H=True, include_mc_mc=False,
               include_wc=True, ignore_bo=True, strict=False,
               path_to_probe=_probe, max_distance_cutoff=0.8):

    outdir = add_slash(outdir)

    try:
        os.makedirs(outdir)
    except:
        pass

    if filename == '':
        filename = path_to_pdb.split('/')[-1].split('.')[0]

    df_probe = parse_probe(path_to_pdb, segname1=probe_segname1, chain1=probe_chain1, resnum1=probe_resnum1,
                           segname2=probe_segname2, chain2=probe_chain2, resnum2=probe_resnum2,
                           probe_sel_criteria=probe_sel_criteria, maxbonded=maxbonded, explicit_H=explicit_H,
                           include_mc_mc=include_mc_mc, include_wc=include_wc, outdir=None,
                           path_to_probe=path_to_probe, ignore_bo=ignore_bo)

    vdms = parse_vdms_from_probe(df_probe, skipping_number=skipping_number)
    pdb = parsePDB(path_to_pdb)
    pdb = pdb.select('resname ' + resnames_aa_20_join)
    template = Template(pdb)
    df_vdms = map_vdms_to_structure(template, vdms, path_to_database,
                                    distance_cutoff=distance_cutoff,
                                    max_distance_cutoff=max_distance_cutoff, strict=strict)
    df_vdms.to_csv(outdir + filename + '.csv', index=False)


def run_lookup_ligand(path_to_pdb, path_to_database, outdir, path_to_ligand_file,
                      path_to_ligand_params=None,
                      ligand_atom_type_dict=None, filename='', distance_cutoff=0.6,
                      probe_segname1='', probe_chain1='', probe_resnum1='',
                      probe_segname2='', probe_chain2='', probe_resnum2='',
                      probe_sel_criteria='NOT METAL', maxbonded=4, explicit_H=True,
                      include_mc_mc=False, include_wc=True, ignore_bo=True, strict=False,
                      path_to_probe=_probe,  max_distance_cutoff=0.8):

    outdir = add_slash(outdir)

    try:
        os.makedirs(outdir)
    except:
        pass

    if filename == '':
        filename = path_to_pdb.split('/')[-1].split('.')[0]

    s = Sample()
    s.set_ligand_vdm_correspondence(path_to_ligand_file)
    lig_resname = s.ligand_vdm_correspondence.lig_resname.iat[0]

    df_probe = parse_probe(path_to_pdb, segname1=probe_segname1, chain1=probe_chain1, resnum1=probe_resnum1,
                           segname2=probe_segname2, chain2=probe_chain2, resnum2=probe_resnum2,
                           resname2='RES' + lig_resname,
                           probe_sel_criteria=probe_sel_criteria, maxbonded=maxbonded, explicit_H=explicit_H,
                           include_mc_mc=include_mc_mc, include_wc=include_wc, outdir=None,
                           path_to_probe=path_to_probe, ignore_bo=ignore_bo)

    vdms = parse_ligand_vdms_from_probe(df_probe, s.ligand_vdm_correspondence)

    if path_to_ligand_params is not None:
        ligand_atom_type_dict = make_lig_atom_type_dict(lig_resname, path_to_ligand_params)
    elif ligand_atom_type_dict is None:
        raise Exception('ligand_atom_type_dict or path_to_ligand_params must be set')
    pdb = parsePDB(path_to_pdb)
    pdb = pdb.select('resname ' + resnames_aa_20_join + ' ' + lig_resname)
    template = Template(pdb, lig_atom_types_dict=ligand_atom_type_dict)
    df_vdms = map_lig_vdms_to_structure(template, vdms, path_to_database,
                                        ligand_vdm_correspondence=s.ligand_vdm_correspondence,
                                        distance_cutoff=distance_cutoff,
                                        max_distance_cutoff=max_distance_cutoff, strict=strict)
    df_vdms.to_csv(outdir + filename + '_' + lig_resname + '.csv', index=False)


