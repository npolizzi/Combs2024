from ..design.constants import resnames_aa_20, cg_flip_dict, keep_H, residue_sc_names
from ..design.functions import flip_cg_coords
from ..design.cluster import Cluster
import pandas as pd
import numpy as np
import os


def make_dict_coord_sort(cg_dict, aa, cg_H_name_dict=None):
    dict_coord_sort = dict()
    for cg_resname in cg_dict.keys():
        if aa == 'GLY':
            dfx = pd.DataFrame({'name': ['N', 'CA', 'C', 'HA3'],
                                'resnum': [10, 10, 10, 10],
                                'chain': ['X', 'X', 'X', 'X']})
        else:
            dfx = pd.DataFrame({'name': ['N', 'CA', 'C', 'CB'],
                                'resnum': [10, 10, 10, 10],
                                'chain': ['X', 'X', 'X', 'X']})

        if cg_H_name_dict is None:
            cg_names = [n for n in cg_dict[cg_resname] if n[0] != 'H']
        else:
            cg_names = [n for n in cg_dict[cg_resname] if (n in cg_H_name_dict[cg_resname] or n[0] != 'H')]
        dfy = pd.DataFrame(dict(name=cg_names))
        dfy['resnum'] = 10
        dfy['chain'] = 'Y'
        df = pd.concat((dfx, dfy))
        dict_coord_sort[cg_resname] = df
    return dict_coord_sort


def run_cluster(cg_dict, inpath_comb_nr, outpath, ind, rmsd_cutoff=0.5, path_to_cg_H_names=None,
                min_cluster_size=2, maxdist=False):

    cg_aa_in_flip_keys = all([aa_cg in cg_flip_dict.keys() for aa_cg in cg_dict.keys()])
    if cg_aa_in_flip_keys:
        cg_names_in_flip_names = all([len(cg_flip_dict[aa_cg] - set(names)) == 0 for aa_cg, names in cg_dict.items()])
    else:
        cg_names_in_flip_names = False

    aa = resnames_aa_20[ind]
    f = aa + '.parquet.gzip'

    num_atoms_sc = len(residue_sc_names[aa])
    if aa in keep_H:
        num_atoms_sc += 1

    for key, val in cg_dict.items():
        num_atoms_cg = len(val)
        break

    try:
        os.makedirs(outpath)
    except:
        pass

    df = pd.read_parquet(inpath_comb_nr + f)

    #filter vdm from df if the vdm doesn't contain correct number of sidechain atoms
    if aa != 'GLY':
        if aa in keep_H.keys():
            df_noH = df[df.apply(lambda x: x['name'][0] != 'H', axis=1) |
                          ((df.chain == 'X') & (df.resnum == 10) & (df['name'] == keep_H[aa]))]
        else:
            df_noH = df[df.apply(lambda x: x['name'][0] != 'H', axis=1)]
        dfx = df_noH[(df_noH.chain == 'X') & (df_noH.resnum == 10) & (~df_noH['name'].isin({'O', 'N', 'CA', 'C'}))]
        dfx = dfx.groupby(['CG', 'rota', 'probe_name']).filter(lambda x: len(x) == num_atoms_sc)
        df = pd.merge(df, dfx[['CG', 'rota', 'probe_name']].drop_duplicates(), on=['CG', 'rota', 'probe_name'])
    # filter vdm from df if the vdm doesn't contain correct number of CG atoms
    df_cgs = df[df.chain == 'Y']
    df_cgs = df_cgs.groupby(['CG', 'rota', 'probe_name']).filter(lambda x: len(x) == num_atoms_cg)
    df = pd.merge(df, df_cgs[['CG', 'rota', 'probe_name']].drop_duplicates(), on=['CG', 'rota', 'probe_name'])

    cg_H_name_dict = None
    if path_to_cg_H_names is not None:
        cg_H_name_dict=dict()
        with open(path_to_cg_H_names, 'r') as infile:
            for line in infile:
                spl = line.split()
                cg_H_name_dict[spl[0]] = [s.strip() for s in spl[1:]]

    dict_coord_sort = make_dict_coord_sort(cg_dict, aa, cg_H_name_dict=cg_H_name_dict)

    dfs = []
    for cg_resname in dict_coord_sort.keys():
        dfy_ = df[['CG', 'rota', 'probe_name']][(df.chain=='Y') & (df.resname == cg_resname)].drop_duplicates()
        df_ = pd.merge(df, dfy_, on=['CG', 'rota', 'probe_name'])
        df_ = pd.merge(dict_coord_sort[cg_resname], df_, on=['name', 'resnum', 'chain'])
        dfs.append(df_)
    df_ = pd.concat(dfs)

    grs = df_.groupby(['CG', 'rota', 'probe_name'])
    group_names = []
    coords = []
    for n, gr in grs:
        try:
            coords.append(gr[['c_x', 'c_y', 'c_z']].values.astype('float32'))
            group_names.append(n)
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                coords.append(flip_cg_coords(gr))
                group_names.append(n)
        except Exception:
            pass

    coords_ = []
    group_names_ = []
    keys = list(dict_coord_sort.keys())
    len_clu_coords = len(dict_coord_sort[keys[0]])
    for n, c in zip(group_names, coords):
        if len(c) != len_clu_coords:
            continue
        coords_.append(c)
        group_names_.append(n)

    coords_arr = np.array(coords_, dtype='float32')
    clu = Cluster()
    clu.rmsd_cutoff = rmsd_cutoff
    clu.pdb_coords = coords_arr
    clu.make_pairwise_rmsd_mat(maxdist=maxdist)
    clu.make_square()
    clu.make_adj_mat()
    if cg_aa_in_flip_keys and cg_names_in_flip_names:
        index_labels = group_names_
    else:
        index_labels = None
    clu.cluster(min_cluster_size=min_cluster_size, index_labels=index_labels)

    clu_sizes = np.array([len(mems) for mems in clu.mems])
    avg_clu_size = np.mean(clu_sizes)
    cluster_nums = []
    cluster_sizes = []
    cluster_scores = []
    cgs = []
    rotas = []
    probe_names = []
    centroids = []
    dists_to_cent = []
    j=1
    cluster_sets = set()
    for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            # if i % 2 != 0:  # Skip duplicate clusters (i.e. keep only odd cluster numbers, e.g, 1,3,5, etc)
            #     continue
            cluster_set = tuple(sorted([group_names_[mem] for mem in mems]))
            if cluster_set in cluster_sets:
                continue
            else:
                cluster_sets.add(cluster_set)
        cluster_size = len(mems)
        cluster_score = np.log(cluster_size / avg_clu_size)
        for mem in mems:
            gr = grs.get_group(group_names_[mem])
            if mem == cent:
                centroids.append(True)
            else:
                centroids.append(False)
            cluster_nums.append(j)
            cluster_sizes.append(cluster_size)
            cluster_scores.append(cluster_score)
            dists_to_cent.append(clu.rmsd_mat[cent, mem])
            cgs.append(gr.CG.iat[0])
            rotas.append(gr.rota.iat[0])
            probe_names.append(gr.probe_name.iat[0])
        j += 1
    df_ = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, centroid=centroids,
                            cluster_number=cluster_nums, cluster_size=cluster_sizes,
                            C_score_bb_ind=cluster_scores, maxdist_to_centroid=dists_to_cent))
    df = pd.merge(df, df_, on=['CG', 'rota', 'probe_name'])

    dict_coord_sort_ca = dict()
    for key in dict_coord_sort.keys():
        dict_coord_sort_ca[key] = dict_coord_sort[key].copy()
        dict_coord_sort_ca[key]['cluster_atom'] = True
        dict_coord_sort_ca[key]['cluster_order'] = list(range(len_clu_coords))

    dfs = []
    for cg_resname in dict_coord_sort.keys():
        dfy_ = df[['CG', 'rota', 'probe_name']][(df.chain=='Y') & (df.resname == cg_resname)].drop_duplicates()
        df_ = pd.merge(df, dfy_, on=['CG', 'rota', 'probe_name'])
        df_ = pd.merge(df_, dict_coord_sort_ca[cg_resname], on=['name', 'resnum', 'chain'], how='left')
        dfs.append(df_)
    df = pd.concat(dfs)
    df.loc[df.cluster_atom.isna(), 'cluster_atom'] = False

    for abple in set('ABPLE'):  # For any ABPLE not present in dataset
        df['centroid_ABPLE_' + abple] = False
    cols = ['ABPLE', 'cluster_number', 'maxdist_to_centroid', 'rota', 'CG', 'probe_name']
    df_abple = df[cols][(df.chain=='X') & (df.resnum==10)].drop_duplicates()
    for n, g in df_abple.groupby('cluster_number'):
        for abple, g_abple in g.groupby('ABPLE'):
            g_abple_rep = g_abple[g_abple.maxdist_to_centroid == g_abple.maxdist_to_centroid.min()]
            cg = g_abple_rep['CG'].iat[0]
            rota = g_abple_rep['rota'].iat[0]
            pn = g_abple_rep['probe_name'].iat[0]
            filters = (df['CG'] == cg) & (df['rota'] == rota) & (df['probe_name'] == pn)
            df.loc[filters, 'centroid_ABPLE_' + abple] = True

    for dssp in set('GHITESBSC'):  # For any dssp not present in dataset
        df['centroid_dssp_' + dssp] = False
    cols = ['dssp', 'cluster_number', 'maxdist_to_centroid', 'rota', 'CG', 'probe_name']
    df_dssp = df[cols][(df.chain=='X') & (df.resnum==10)].drop_duplicates()
    for n, g in df_dssp.groupby('cluster_number'):
        for dssp, g_dssp in g.groupby('dssp'):
            g_dssp_rep = g_dssp[g_dssp.maxdist_to_centroid == g_dssp.maxdist_to_centroid.min()]
            cg = g_dssp_rep['CG'].iat[0]
            rota = g_dssp_rep['rota'].iat[0]
            pn = g_dssp_rep['probe_name'].iat[0]
            filters = (df['CG'] == cg) & (df['rota'] == rota) & (df['probe_name'] == pn)
            df.loc[filters, 'centroid_dssp_' + dssp] = True
            
    df['centroid_hb_bb_ind'] = False
    cols = ['cluster_number', 'maxdist_to_centroid', 'rota', 'CG', 'probe_name']
    df_hb = df[cols][(df.chain=='X') & (df.resnum==10) & (~df.contact_hb.isna())].drop_duplicates()
    for n, g in df_hb.groupby('cluster_number'):
        g_rep = g[g.maxdist_to_centroid == g.maxdist_to_centroid.min()]
        cg = g_rep['CG'].iat[0]
        rota = g_rep['rota'].iat[0]
        pn = g_rep['probe_name'].iat[0]
        filters = (df['CG'] == cg) & (df['rota'] == rota) & (df['probe_name'] == pn)
        df.loc[filters, 'centroid_hb_bb_ind'] = True


    #########################################################################
    # SCORE THE CLUSTERS

    dfx = df[(df.chain == 'X') & (df.resnum == 10)][['rota', 'CG', 'probe_name',
                                                     'cluster_number', 'cluster_size', 'dssp',
                                                     'ABPLE']].drop_duplicates()

    dfxcopy = dfx.copy()
    for ss in set(dfx.ABPLE):
        dfx_ss = dfx[dfx.ABPLE == ss]
        s_ss = np.log(dfx_ss.groupby('cluster_number').size() / dfx_ss.groupby('cluster_number').size().mean())
        df_ss = pd.DataFrame(zip(s_ss.index.values, s_ss.values),
                             columns=['cluster_number', 'C_score_ABPLE_' + ss]).drop_duplicates()
        df_ss = df_ss.sort_values('C_score_ABPLE_' + ss, ascending=False)
        df_ss['cluster_rank_ABPLE_' + ss] = list(range(1, len(df_ss) + 1))
        dfxcopy = pd.merge(dfxcopy, df_ss, on='cluster_number', how='left')

    for ss in set(dfx.dssp):
        dfx_ss = dfx[dfx.dssp == ss]
        s_ss = np.log(dfx_ss.groupby('cluster_number').size() / dfx_ss.groupby('cluster_number').size().mean())
        df_ss = pd.DataFrame(zip(s_ss.index.values, s_ss.values),
                             columns=['cluster_number', 'C_score_dssp_' + ss]).drop_duplicates()
        df_ss = df_ss.sort_values('C_score_dssp_' + ss, ascending=False)
        df_ss['cluster_rank_dssp_' + ss] = list(range(1, len(df_ss) + 1))
        dfxcopy = pd.merge(dfxcopy, df_ss, on='cluster_number', how='left')

    # Get h-bonding dfx
    dfx = df[(df.chain == 'X') & (df.resnum == 10) & (~df.contact_hb.isna())][['rota', 'CG', 'probe_name',
                                                                               'cluster_number', 'cluster_size', 'dssp',
                                                                               'ABPLE']].drop_duplicates()

    if len(dfx) != 0:
        # bb independent hb cluster scores
        s_ss = np.log(dfx.groupby('cluster_number').size() / dfx.groupby('cluster_number').size().mean())
        df_ss = pd.DataFrame(zip(s_ss.index.values, s_ss.values),
                             columns=['cluster_number', 'C_score_hb_bb_ind']).drop_duplicates()
        df_ss = df_ss.sort_values('C_score_hb_bb_ind', ascending=False)
        df_ss['cluster_rank_hb_bb_ind'] = list(range(1, len(df_ss) + 1))
        dfxcopy = pd.merge(dfxcopy, df_ss, on='cluster_number', how='left')

        for ss in set(dfx.ABPLE):
            dfx_ss = dfx[dfx.ABPLE == ss]
            s_ss = np.log(dfx_ss.groupby('cluster_number').size() / dfx_ss.groupby('cluster_number').size().mean())
            df_ss = pd.DataFrame(zip(s_ss.index.values, s_ss.values),
                                 columns=['cluster_number', 'C_score_hb_ABPLE_' + ss]).drop_duplicates()
            df_ss = df_ss.sort_values('C_score_hb_ABPLE_' + ss, ascending=False)
            df_ss['cluster_rank_hb_ABPLE_' + ss] = list(range(1, len(df_ss) + 1))
            dfxcopy = pd.merge(dfxcopy, df_ss, on='cluster_number', how='left')

        for ss in set(dfx.dssp):
            dfx_ss = dfx[dfx.dssp == ss]
            s_ss = np.log(dfx_ss.groupby('cluster_number').size() / dfx_ss.groupby('cluster_number').size().mean())
            df_ss = pd.DataFrame(zip(s_ss.index.values, s_ss.values),
                                 columns=['cluster_number', 'C_score_hb_dssp_' + ss]).drop_duplicates()
            df_ss = df_ss.sort_values('C_score_hb_dssp_' + ss, ascending=False)
            df_ss['cluster_rank_hb_dssp_' + ss] = list(range(1, len(df_ss) + 1))
            dfxcopy = pd.merge(dfxcopy, df_ss, on='cluster_number', how='left')

    cols = ['rota', 'CG', 'probe_name', 'C_score_hb_bb_ind', 'C_score_ABPLE_P', 'C_score_ABPLE_B', 'C_score_ABPLE_L',
            'C_score_ABPLE_A', 'C_score_ABPLE_E', 'C_score_dssp_S', 'C_score_dssp_T',
            'C_score_dssp_C', 'C_score_dssp_B', 'C_score_dssp_I', 'C_score_dssp_E',
            'C_score_dssp_G', 'C_score_dssp_H', 'C_score_hb_ABPLE_P', 'C_score_hb_ABPLE_B', 'C_score_hb_ABPLE_L',
            'C_score_hb_ABPLE_A', 'C_score_hb_ABPLE_E', 'C_score_hb_dssp_S', 'C_score_hb_dssp_T',
            'C_score_hb_dssp_C', 'C_score_hb_dssp_B', 'C_score_hb_dssp_I', 'C_score_hb_dssp_E',
            'C_score_hb_dssp_G', 'C_score_hb_dssp_H', 'cluster_rank_hb_bb_ind', 'cluster_rank_ABPLE_A',
            'cluster_rank_ABPLE_P',
            'cluster_rank_ABPLE_B', 'cluster_rank_ABPLE_L', 'cluster_rank_ABPLE_E', 'cluster_rank_dssp_S',
            'cluster_rank_dssp_T', 'cluster_rank_dssp_C', 'cluster_rank_dssp_B', 'cluster_rank_dssp_I',
            'cluster_rank_dssp_E', 'cluster_rank_dssp_G', 'cluster_rank_dssp_H',
            'cluster_rank_hb_ABPLE_A', 'cluster_rank_hb_ABPLE_P',
            'cluster_rank_hb_ABPLE_B', 'cluster_rank_hb_ABPLE_L', 'cluster_rank_hb_ABPLE_E', 'cluster_rank_hb_dssp_S',
            'cluster_rank_hb_dssp_T', 'cluster_rank_hb_dssp_C', 'cluster_rank_hb_dssp_B', 'cluster_rank_hb_dssp_I',
            'cluster_rank_hb_dssp_E', 'cluster_rank_hb_dssp_G', 'cluster_rank_hb_dssp_H',
            ]

    cols = sorted(set(dfxcopy.columns) & set(cols))

    df = pd.merge(df, dfxcopy[cols].drop_duplicates(), on=['rota', 'CG', 'probe_name'])

    df.to_parquet(outpath + f, engine='pyarrow',
                  compression='gzip')