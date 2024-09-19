from ..design.constants import resnames_aa_20, cg_flip_dict, coords_cols, residue_sc_names, keep_H
from ..design.functions import flip_cg_coords, df_ideal_ala, flip_dict, flip_cg_x_coords, flip_x_coords
from ..design.transformation import get_rot_trans
from ..design.cluster import Cluster
from .comb import set_contacts
import pandas as pd
import numpy as np
import os
import traceback


def run_reps(cg_dict, inpath_comb_nr_cluster, outpath, ind):
    
    cg_aa_in_flip_keys = all([aa_cg in cg_flip_dict.keys() for aa_cg in cg_dict.keys()])
    if cg_aa_in_flip_keys:
        cg_names_in_flip_names = all([len(cg_flip_dict[aa_cg] - set(names)) == 0 for aa_cg, names in cg_dict.items()])
    else:
        cg_names_in_flip_names = False

    aa = resnames_aa_20[ind]
    f = aa + '.parquet.gzip'

    try:
        os.makedirs(outpath)
    except:
        pass

    df_orig = pd.read_parquet(inpath_comb_nr_cluster + f)

    print('len_df', len(df_orig))

    num_atoms = len(residue_sc_names[aa])
    if aa in keep_H:
        num_atoms += 1
    for key, val in cg_dict.items():
        num_atoms += len(val)
        break

    print('num_atoms', num_atoms)
    print('Making reps for', f, '...')

    grs_orig = df_orig.groupby(['rota', 'CG', 'probe_name'])

    df_10 = pd.merge(df_ideal_ala['name'][df_ideal_ala.name.isin({'N', 'CA', 'C'})],
                     df_orig[(df_orig.resnum == 10) & (df_orig.chain == 'X')], on='name')

    targ_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']][df_ideal_ala.name.isin({'N', 'CA', 'C'})].values

    df_reps = []
    for abple, df in df_10.groupby('ABPLE'):
        if abple == 'n':
            continue
        print('...Making reps for', abple, '...')
        grs = df.groupby(['rota', 'CG', 'probe_name'])
        print(len(grs))
        print('len_df', abple, len(df))

        transformed_dfs = []
        print('transforming coordinates to ideal alanine ref frame...')
        for n, g in grs:
            try:
                mob_coords = g[['c_x', 'c_y', 'c_z']].values
                R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
                orig_g = grs_orig.get_group(n).copy()
                for i in range(0, len(coords_cols), 3):
                    orig_g[coords_cols[i:i + 3]] = np.dot(orig_g[coords_cols[i:i + 3]] - m_com, R) + t_com
                transformed_dfs.append(orig_g)
            except:
                traceback.print_exc()
                print('error in', n)
                print(g)

        df_tr = pd.concat(transformed_dfs)
        print('len_df_tr', len(df_tr))
        grs_df_tr = df_tr.groupby(['rota', 'CG', 'probe_name'])

        df_cgs = df_tr[df_tr.chain == 'Y'][['name', 'resname']].drop_duplicates()
        df_cgs_interim = []
        for rn, names in cg_dict.items():
            for name in names:
                df_cgs_interim.append(df_cgs[(df_cgs.resname == rn) & (df_cgs['name'] == name)])
        df_cgs = pd.concat(df_cgs_interim)

        df_tr_y = pd.merge(df_cgs, df_tr[df_tr.chain == 'Y'], on=['name', 'resname']) # df_cgs has to be listed
        # first to preserve atom order!
        grs_y = df_tr_y.groupby(['rota', 'CG', 'probe_name'])

        groups = []
        coords = []
        for n, g in grs_y:
            groups.append(n)
            coords.append(g[['c_x', 'c_y', 'c_z']].values)
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                coords.append(flip_cg_coords(g))
                groups.append(n)
        coords_arr = np.array(coords, dtype=np.float32)
        print('clustering CGs...')
        clu = Cluster()
        clu.pdb_coords = coords_arr
        clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
        clu.rmsd_cutoff = 2
        clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        j = 1
        cluster_sets = set()
        print('num members', len(set([m for mems in clu.mems for m in mems])))
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)
            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    centroids.append(True)
                else:
                    centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                cluster_nums.append(j)
                cluster_sizes.append(cluster_size)
                cluster_scores.append(cluster_score)
                cgs.append(gr.CG.iat[0])
                rotas.append(gr.rota.iat[0])
                probe_names.append(gr.probe_name.iat[0])
            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_coarse_ABPLE=centroids,
                                   CG_rep_cluster_number_coarse_ABPLE=cluster_nums, CG_rep_cluster_size_coarse_ABPLE=cluster_sizes,
                                   CG_rep_cluster_score_coarse_ABPLE=cluster_scores)).drop_duplicates()
        print('df_clu', len(df_clu))
        print('df_clu_dropdup', len(df_clu[['CG', 'rota', 'probe_name']].drop_duplicates()))
        df__ = pd.merge(df_tr, df_clu, on=['rota', 'CG', 'probe_name'])
        print('1 len df__', len(df__[['CG', 'rota', 'probe_name']].drop_duplicates()))
        print('len_df__', len(df__))

        ################
        clu.rmsd_cutoff = 0.5
        # clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        clu.cluster()

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        reps = []
        ABPLE_score_col = 'C_score_ABPLE_' + abple
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:

                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)

            _cluster_nums = []
            _cluster_sizes = []
            _cluster_scores = []
            _cgs = []
            _rotas = []
            _probe_names = []
            _centroids = []
            _reps = []
            _score_col = []

            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    _centroids.append(True)
                else:
                    _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                _cluster_nums.append(j)
                _cluster_sizes.append(cluster_size)
                _cluster_scores.append(cluster_score)
                _cgs.append(gr.CG.iat[0])
                _rotas.append(gr.rota.iat[0])
                _probe_names.append(gr.probe_name.iat[0])
                _reps.append(False)
                _score_col.append(gr[ABPLE_score_col].iat[0])
            rep_ind = np.argmax(_score_col)
            _reps[rep_ind] = True

            cluster_nums.extend(_cluster_nums)
            cluster_sizes.extend(_cluster_sizes)
            cluster_scores.extend(_cluster_scores)
            cgs.extend(_cgs)
            rotas.extend(_rotas)
            probe_names.extend(_probe_names)
            centroids.extend(_centroids)
            reps.extend(_reps)

            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_medium_ABPLE=centroids,
                                   CG_rep_cluster_number_medium_ABPLE=cluster_nums,
                                   CG_rep_cluster_size_medium_ABPLE=cluster_sizes,
                                   CG_rep_cluster_score_medium_ABPLE=cluster_scores,
                                   CG_rep_medium_ABPLE=reps)).drop_duplicates()
        df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])
        print('1 len df__', len(df__[['CG', 'rota', 'probe_name']].drop_duplicates()))
        print('len_df__', len(df__))
        ##################
        clu.rmsd_cutoff = 0.15
        # clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        reps = []
        ABPLE_score_col = 'C_score_ABPLE_' + abple
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)

            _cluster_nums = []
            _cluster_sizes = []
            _cluster_scores = []
            _cgs = []
            _rotas = []
            _probe_names = []
            _centroids = []
            _reps = []
            _score_col = []

            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    _centroids.append(True)
                else:
                    _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                _cluster_nums.append(j)
                _cluster_sizes.append(cluster_size)
                _cluster_scores.append(cluster_score)
                _cgs.append(gr.CG.iat[0])
                _rotas.append(gr.rota.iat[0])
                _probe_names.append(gr.probe_name.iat[0])
                _reps.append(False)
                _score_col.append(gr[ABPLE_score_col].iat[0])
            rep_ind = np.argmax(_score_col)
            _reps[rep_ind] = True

            cluster_nums.extend(_cluster_nums)
            cluster_sizes.extend(_cluster_sizes)
            cluster_scores.extend(_cluster_scores)
            cgs.extend(_cgs)
            rotas.extend(_rotas)
            probe_names.extend(_probe_names)
            centroids.extend(_centroids)
            reps.extend(_reps)

            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_fine_ABPLE=centroids,
                                   CG_rep_cluster_number_fine_ABPLE=cluster_nums, CG_rep_cluster_size_fine_ABPLE=cluster_sizes,
                                   CG_rep_cluster_score_fine_ABPLE=cluster_scores, CG_rep_fine_ABPLE=reps)).drop_duplicates()

        df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])
        print('3 len df__', len(df__[['CG', 'rota', 'probe_name']].drop_duplicates()))
        print('len_df__', len(df__))

        grs_df_ = df__.groupby(['rota', 'CG', 'probe_name'])

        print('Making sub-clusters...')
        if aa in keep_H.keys():
            df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1) |
                          ((df__.chain == 'X') & (df__.resnum == 10) & (df__.name == keep_H[aa]))]
        else:
            df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1)]
        dfx = df_noH[(df_noH.chain == 'X') & (df_noH.resnum == 10) & (~df_noH.name.isin({'O', 'N', 'CA', 'C'}))]
        dfx = dfx.sort_values('name')
        df_y__ = pd.merge(df__, df_tr_y[['name', 'resname', 'chain']].drop_duplicates(), on=['name', 'resname', 'chain'])
        df_noHx = pd.concat([dfx, df_y__])

        print('1 len df_noHx', len(df_noHx[['CG', 'rota', 'probe_name']].drop_duplicates()))
        print('len_df_noHx', len(df_noHx))

        dfs_subrep = []
        for qq, dc in df_noHx.groupby('CG_rep_cluster_number_fine_ABPLE'):
            groups = []
            coords = []
            for n, g in dc.groupby(['rota', 'CG', 'probe_name']):
                groups.append(n)
                coords.append(g[['c_x', 'c_y', 'c_z']].values)
                if cg_aa_in_flip_keys and cg_names_in_flip_names:
                    coords.append(flip_cg_coords(g))
                    groups.append(n)
                if aa in flip_dict.keys():
                    coords.append(flip_x_coords(g))
                    groups.append(n)
                if cg_aa_in_flip_keys and cg_names_in_flip_names and aa in flip_dict.keys():
                    coords.append(flip_cg_x_coords(g))
                    groups.append(n)

            coords_ = []
            groups_ = []
            for n, c in zip(groups, coords):
                if len(c) != num_atoms:
                    print('filtering num atoms')
                    print(n)
                    print(c)
                    continue
                coords_.append(c)
                groups_.append(n)
            coords = coords_
            groups = groups_

            if len(coords) == 0:
                print('coords is 0')
                continue

            coords_arr = np.array(coords, dtype=np.float32)

            clu = Cluster()
            clu.pdb_coords = coords_arr
            clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
            clu.rmsd_cutoff = 0.25
            clu.make_square()
            clu.make_adj_mat()
            # clu.make_adj_mat_no_superpose()
            if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
                index_labels = groups
            else:
                index_labels = None
            clu.cluster(index_labels=index_labels)

            clu_sizes = np.array([len(mems) for mems in clu.mems])
            avg_clu_size = np.mean(clu_sizes)
            cluster_nums = []
            cluster_sizes = []
            cluster_scores = []
            cgs = []
            rotas = []
            probe_names = []
            centroids = []
            reps = []
            ABPLE_score_col = 'C_score_ABPLE_' + abple
            j = 1
            cluster_sets = set()
            seen_groups = set()
            for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
                if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
                    cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                    if cluster_set in cluster_sets:
                        continue
                    else:
                        cluster_sets.add(cluster_set)
                cluster_size = len(mems)
                cluster_score = np.log(cluster_size / avg_clu_size)

                _cluster_nums = []
                _cluster_sizes = []
                _cluster_scores = []
                _cgs = []
                _rotas = []
                _probe_names = []
                _centroids = []
                _reps = []
                _score_col = []

                for mem in mems:
                    gname = groups[mem]
                    if gname in seen_groups:
                        print('seen member', mem, groups[mem])
                        continue
                    seen_groups.add(gname)
                    gr = grs_df_.get_group(groups[mem])
                    if mem == cent:
                        _centroids.append(True)
                    else:
                        _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                    _cluster_nums.append(j)
                    _cluster_sizes.append(cluster_size)
                    _cluster_scores.append(cluster_score)
                    _cgs.append(gr.CG.iat[0])
                    _rotas.append(gr.rota.iat[0])
                    _probe_names.append(gr.probe_name.iat[0])
                    _reps.append(False)
                    _score_col.append(gr[ABPLE_score_col].iat[0])
                rep_ind = np.argmax(_score_col)
                _reps[rep_ind] = True

                cluster_nums.extend(_cluster_nums)
                cluster_sizes.extend(_cluster_sizes)
                cluster_scores.extend(_cluster_scores)
                cgs.extend(_cgs)
                rotas.extend(_rotas)
                probe_names.extend(_probe_names)
                centroids.extend(_centroids)
                reps.extend(_reps)
                j += 1

            df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, sc_rep_centroid_fine_ABPLE=centroids,
                                       sc_rep_cluster_number_fine_ABPLE=cluster_nums, sc_rep_cluster_size_fine_ABPLE=cluster_sizes,
                                       sc_rep_cluster_score_fine_ABPLE=cluster_scores,
                                       sc_rep_fine_ABPLE=reps)).drop_duplicates()

            df_subrep = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])
            dfs_subrep.append(df_subrep)

        if len(dfs_subrep) > 0:
            df___ = pd.concat(dfs_subrep)
            df_reps.append(df___)

    df_rep = pd.concat(df_reps)
    df_rep.reset_index(inplace=True, drop=True)


######
# DSSP

    cg_aa_in_flip_keys = all([aa_cg in cg_flip_dict.keys() for aa_cg in cg_dict.keys()])
    if cg_aa_in_flip_keys:
        cg_names_in_flip_names = all([len(cg_flip_dict[aa_cg] - set(names)) == 0 for aa_cg, names in cg_dict.items()])
    else:
        cg_names_in_flip_names = False

    aa = resnames_aa_20[ind]
    f = aa + '.parquet.gzip'

    try:
        os.makedirs(outpath)
    except:
        pass

    df_orig = pd.read_parquet(inpath_comb_nr_cluster + f)

    num_atoms = len(residue_sc_names[aa])
    if aa in keep_H:
        num_atoms += 1
    for key, val in cg_dict.items():
        num_atoms += len(val)
        break

    print('num_atoms', num_atoms)
    print('Making reps for', f, '...')

    grs_orig = df_orig.groupby(['rota', 'CG', 'probe_name'])

    df_10 = pd.merge(df_ideal_ala['name'][df_ideal_ala.name.isin({'N', 'CA', 'C'})],
                     df_orig[(df_orig.resnum == 10) & (df_orig.chain == 'X')], on='name')

    targ_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']][df_ideal_ala.name.isin({'N', 'CA', 'C'})].values

    df_reps = []
    for dssp, df in df_10.groupby('dssp'):
        if dssp == 'n':
            continue
        print('...Making reps for', dssp, '...')
        grs = df.groupby(['rota', 'CG', 'probe_name'])
        print(len(grs))

        transformed_dfs = []
        print('transforming coordinates to ideal alanine ref frame...')
        for n, g in grs:
            try:
                mob_coords = g[['c_x', 'c_y', 'c_z']].values
                R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
                orig_g = grs_orig.get_group(n).copy()
                for i in range(0, len(coords_cols), 3):
                    orig_g[coords_cols[i:i + 3]] = np.dot(orig_g[coords_cols[i:i + 3]] - m_com, R) + t_com
                transformed_dfs.append(orig_g)
            except:
                traceback.print_exc()
                print('error in', n)
                print(g)

        df_tr = pd.concat(transformed_dfs)
        grs_df_tr = df_tr.groupby(['rota', 'CG', 'probe_name'])

        df_cgs = df_tr[df_tr.chain == 'Y'][['name', 'resname']].drop_duplicates()
        df_cgs_interim = []
        for rn, names in cg_dict.items():
            for name in names:
                df_cgs_interim.append(df_cgs[(df_cgs.resname == rn) & (df_cgs['name'] == name)])
        df_cgs = pd.concat(df_cgs_interim)

        df_tr_y = pd.merge(df_cgs, df_tr[df_tr.chain == 'Y'], on=['name', 'resname']) # df_cgs has to be listed
        # first to preserve atom order!
        grs_y = df_tr_y.groupby(['rota', 'CG', 'probe_name'])

        groups = []
        coords = []
        for n, g in grs_y:
            groups.append(n)
            coords.append(g[['c_x', 'c_y', 'c_z']].values)
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                coords.append(flip_cg_coords(g))
                groups.append(n)
        coords_arr = np.array(coords, dtype=np.float32)
        print('clustering CGs...')
        clu = Cluster()
        clu.pdb_coords = coords_arr
        clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
        clu.rmsd_cutoff = 2
        clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)
            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    centroids.append(True)
                else:
                    centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                cluster_nums.append(j)
                cluster_sizes.append(cluster_size)
                cluster_scores.append(cluster_score)
                cgs.append(gr.CG.iat[0])
                rotas.append(gr.rota.iat[0])
                probe_names.append(gr.probe_name.iat[0])
            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_coarse_dssp=centroids,
                                   CG_rep_cluster_number_coarse_dssp=cluster_nums, CG_rep_cluster_size_coarse_dssp=cluster_sizes,
                                   CG_rep_cluster_score_coarse_dssp=cluster_scores)).drop_duplicates()
        df__ = pd.merge(df_tr, df_clu, on=['rota', 'CG', 'probe_name'])

        ################
        clu.rmsd_cutoff = 0.5
        # clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        reps = []
        dssp_score_col = 'C_score_dssp_' + dssp
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)

            _cluster_nums = []
            _cluster_sizes = []
            _cluster_scores = []
            _cgs = []
            _rotas = []
            _probe_names = []
            _centroids = []
            _reps = []
            _score_col = []

            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    _centroids.append(True)
                else:
                    _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                _cluster_nums.append(j)
                _cluster_sizes.append(cluster_size)
                _cluster_scores.append(cluster_score)
                _cgs.append(gr.CG.iat[0])
                _rotas.append(gr.rota.iat[0])
                _probe_names.append(gr.probe_name.iat[0])
                _reps.append(False)
                _score_col.append(gr[dssp_score_col].iat[0])
            rep_ind = np.argmax(_score_col)
            _reps[rep_ind] = True

            cluster_nums.extend(_cluster_nums)
            cluster_sizes.extend(_cluster_sizes)
            cluster_scores.extend(_cluster_scores)
            cgs.extend(_cgs)
            rotas.extend(_rotas)
            probe_names.extend(_probe_names)
            centroids.extend(_centroids)
            reps.extend(_reps)

            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_medium_dssp=centroids,
                                   CG_rep_cluster_number_medium_dssp=cluster_nums,
                                   CG_rep_cluster_size_medium_dssp=cluster_sizes,
                                   CG_rep_cluster_score_medium_dssp=cluster_scores,
                                   CG_rep_medium_dssp=reps)).drop_duplicates()

        df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])

        ##################
        clu.rmsd_cutoff = 0.15
        # clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        reps = []
        dssp_score_col = 'C_score_dssp_' + dssp
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                # if i % 2 != 0:  # Skip duplicate clusters (i.e. keep only odd cluster numbers, e.g, 1,3,5, etc)
                #     continue
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)

            _cluster_nums = []
            _cluster_sizes = []
            _cluster_scores = []
            _cgs = []
            _rotas = []
            _probe_names = []
            _centroids = []
            _reps = []
            _score_col = []

            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_tr.get_group(groups[mem])
                if mem == cent:
                    _centroids.append(True)
                else:
                    _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                _cluster_nums.append(j)
                _cluster_sizes.append(cluster_size)
                _cluster_scores.append(cluster_score)
                _cgs.append(gr.CG.iat[0])
                _rotas.append(gr.rota.iat[0])
                _probe_names.append(gr.probe_name.iat[0])
                _reps.append(False)
                _score_col.append(gr[dssp_score_col].iat[0])
            rep_ind = np.argmax(_score_col)
            _reps[rep_ind] = True

            cluster_nums.extend(_cluster_nums)
            cluster_sizes.extend(_cluster_sizes)
            cluster_scores.extend(_cluster_scores)
            cgs.extend(_cgs)
            rotas.extend(_rotas)
            probe_names.extend(_probe_names)
            centroids.extend(_centroids)
            reps.extend(_reps)

            j += 1
        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_fine_dssp=centroids,
                                   CG_rep_cluster_number_fine_dssp=cluster_nums, CG_rep_cluster_size_fine_dssp=cluster_sizes,
                                   CG_rep_cluster_score_fine_dssp=cluster_scores, CG_rep_fine_dssp=reps)).drop_duplicates()

        df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])


        grs_df_ = df__.groupby(['rota', 'CG', 'probe_name'])

        print('Making sub-clusters...')
        if aa in keep_H.keys():
            df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1) |
                          ((df__.chain == 'X') & (df__.resnum == 10) & (df__.name == keep_H[aa]))]
        else:
            df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1)]
        dfx = df_noH[(df_noH.chain == 'X') & (df_noH.resnum == 10) & (~df_noH.name.isin({'O', 'N', 'CA', 'C'}))]
        dfx = dfx.sort_values('name')
        df_y__ = pd.merge(df__, df_tr_y[['name', 'resname', 'chain']].drop_duplicates(), on=['name', 'resname', 'chain'])
        df_noHx = pd.concat([dfx, df_y__])

        dfs_subrep = []
        for qq, dc in df_noHx.groupby('CG_rep_cluster_number_fine_dssp'):
            groups = []
            coords = []
            for n, g in dc.groupby(['rota', 'CG', 'probe_name']):
                groups.append(n)
                coords.append(g[['c_x', 'c_y', 'c_z']].values)
                if cg_aa_in_flip_keys and cg_names_in_flip_names:
                    coords.append(flip_cg_coords(g))
                    groups.append(n)
                if aa in flip_dict.keys():
                    coords.append(flip_x_coords(g))
                    groups.append(n)
                if cg_aa_in_flip_keys and cg_names_in_flip_names and aa in flip_dict.keys():
                    coords.append(flip_cg_x_coords(g))
                    groups.append(n)

            coords_ = []
            groups_ = []
            for n, c in zip(groups, coords):
                if len(c) != num_atoms:
                    continue
                coords_.append(c)
                groups_.append(n)
            coords = coords_
            groups = groups_

            if len(coords) == 0:
                continue

            coords_arr = np.array(coords, dtype=np.float32)

            clu = Cluster()
            clu.pdb_coords = coords_arr
            clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
            clu.rmsd_cutoff = 0.25
            clu.make_square()
            clu.make_adj_mat()
            # clu.make_adj_mat_no_superpose()
            if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
                index_labels = groups
            else:
                index_labels = None
            clu.cluster(index_labels=index_labels)

            clu_sizes = np.array([len(mems) for mems in clu.mems])
            avg_clu_size = np.mean(clu_sizes)
            cluster_nums = []
            cluster_sizes = []
            cluster_scores = []
            cgs = []
            rotas = []
            probe_names = []
            centroids = []
            reps = []
            dssp_score_col = 'C_score_dssp_' + dssp
            j = 1
            cluster_sets = set()
            seen_groups = set()
            for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
                if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
                    cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                    if cluster_set in cluster_sets:
                        continue
                    else:
                        cluster_sets.add(cluster_set)
                cluster_size = len(mems)
                cluster_score = np.log(cluster_size / avg_clu_size)

                _cluster_nums = []
                _cluster_sizes = []
                _cluster_scores = []
                _cgs = []
                _rotas = []
                _probe_names = []
                _centroids = []
                _reps = []
                _score_col = []

                for mem in mems:
                    gname = groups[mem]
                    if gname in seen_groups:
                        print('seen member', mem, groups[mem])
                        continue
                    seen_groups.add(gname)
                    gr = grs_df_.get_group(groups[mem])
                    if mem == cent:
                        _centroids.append(True)
                    else:
                        _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                    _cluster_nums.append(j)
                    _cluster_sizes.append(cluster_size)
                    _cluster_scores.append(cluster_score)
                    _cgs.append(gr.CG.iat[0])
                    _rotas.append(gr.rota.iat[0])
                    _probe_names.append(gr.probe_name.iat[0])
                    _reps.append(False)
                    _score_col.append(gr[dssp_score_col].iat[0])
                rep_ind = np.argmax(_score_col)
                _reps[rep_ind] = True

                cluster_nums.extend(_cluster_nums)
                cluster_sizes.extend(_cluster_sizes)
                cluster_scores.extend(_cluster_scores)
                cgs.extend(_cgs)
                rotas.extend(_rotas)
                probe_names.extend(_probe_names)
                centroids.extend(_centroids)
                reps.extend(_reps)
                j += 1

            df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, sc_rep_centroid_fine_dssp=centroids,
                                       sc_rep_cluster_number_fine_dssp=cluster_nums, sc_rep_cluster_size_fine_dssp=cluster_sizes,
                                       sc_rep_cluster_score_fine_dssp=cluster_scores,
                                       sc_rep_fine_dssp=reps)).drop_duplicates()

            df_subrep = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])
            dfs_subrep.append(df_subrep)

        if len(dfs_subrep) > 0:
            df___ = pd.concat(dfs_subrep)
            df_reps.append(df___)

    df_rep_dssp = pd.concat(df_reps)
    df_rep_dssp.reset_index(inplace=True, drop=True)
    cols_dssp_only = list(set(df_rep_dssp.columns) - set(df_rep.columns))
    cols_dssp_only.extend(['rota', 'CG', 'probe_name'])
    df_rep = pd.merge(df_rep, df_rep_dssp[cols_dssp_only].drop_duplicates(), on=['rota', 'CG', 'probe_name'])


##bbind

    cg_aa_in_flip_keys = all([aa_cg in cg_flip_dict.keys() for aa_cg in cg_dict.keys()])
    if cg_aa_in_flip_keys:
        cg_names_in_flip_names = all([len(cg_flip_dict[aa_cg] - set(names)) == 0 for aa_cg, names in cg_dict.items()])
    else:
        cg_names_in_flip_names = False

    aa = resnames_aa_20[ind]
    f = aa + '.parquet.gzip'

    try:
        os.makedirs(outpath)
    except:
        pass

    df_orig = pd.read_parquet(inpath_comb_nr_cluster + f)

    num_atoms = len(residue_sc_names[aa])
    if aa in keep_H:
        num_atoms += 1
    for key, val in cg_dict.items():
        num_atoms += len(val)
        break

    print('num_atoms', num_atoms)
    print('Making reps for', f, '...')

    grs_orig = df_orig.groupby(['rota', 'CG', 'probe_name'])

    df_10 = pd.merge(df_ideal_ala['name'][df_ideal_ala.name.isin({'N', 'CA', 'C'})],
                     df_orig[(df_orig.resnum == 10) & (df_orig.chain == 'X')], on='name')

    targ_coords = df_ideal_ala[['c_x', 'c_y', 'c_z']][df_ideal_ala.name.isin({'N', 'CA', 'C'})].values

    df_reps = []
    df = df_10

    print('...Making reps for bb ind...')
    grs = df.groupby(['rota', 'CG', 'probe_name'])
    print(len(grs))

    transformed_dfs = []
    print('transforming coordinates to ideal alanine ref frame...')
    for n, g in grs:
        try:
            mob_coords = g[['c_x', 'c_y', 'c_z']].values
            R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
            orig_g = grs_orig.get_group(n).copy()
            for i in range(0, len(coords_cols), 3):
                orig_g[coords_cols[i:i + 3]] = np.dot(orig_g[coords_cols[i:i + 3]] - m_com, R) + t_com
            transformed_dfs.append(orig_g)
        except:
            traceback.print_exc()
            print('error in', n)
            print(g)

    df_tr = pd.concat(transformed_dfs)
    grs_df_tr = df_tr.groupby(['rota', 'CG', 'probe_name'])

    df_cgs = df_tr[df_tr.chain == 'Y'][['name', 'resname']].drop_duplicates()
    df_cgs_interim = []
    for rn, names in cg_dict.items():
        for name in names:
            df_cgs_interim.append(df_cgs[(df_cgs.resname == rn) & (df_cgs['name'] == name)])
    df_cgs = pd.concat(df_cgs_interim)

    df_tr_y = pd.merge(df_cgs, df_tr[df_tr.chain == 'Y'], on=['name', 'resname']) # df_cgs has to be listed
    # first to preserve atom order!
    grs_y = df_tr_y.groupby(['rota', 'CG', 'probe_name'])

    groups = []
    coords = []
    for n, g in grs_y:
        groups.append(n)
        coords.append(g[['c_x', 'c_y', 'c_z']].values)
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            coords.append(flip_cg_coords(g))
            groups.append(n)
    coords_arr = np.array(coords, dtype=np.float32)
    print('clustering CGs...')
    clu = Cluster()
    clu.pdb_coords = coords_arr
    clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
    clu.rmsd_cutoff = 2
    clu.make_square()
    clu.make_adj_mat()
    # clu.make_adj_mat_no_superpose()
    if cg_aa_in_flip_keys and cg_names_in_flip_names:
        index_labels = groups
    else:
        index_labels = None
    clu.cluster(index_labels=index_labels)

    clu_sizes = np.array([len(mems) for mems in clu.mems])
    avg_clu_size = np.mean(clu_sizes)
    cluster_nums = []
    cluster_sizes = []
    cluster_scores = []
    cgs = []
    rotas = []
    probe_names = []
    centroids = []
    j = 1
    cluster_sets = set()
    seen_groups = set()
    for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            # if i % 2 != 0:  # Skip duplicate clusters (i.e. keep only odd cluster numbers, e.g, 1,3,5, etc)
            #     continue
            cluster_set = tuple(sorted([groups[mem] for mem in mems]))
            if cluster_set in cluster_sets:
                continue
            else:
                cluster_sets.add(cluster_set)
        cluster_size = len(mems)
        cluster_score = np.log(cluster_size / avg_clu_size)
        for mem in mems:
            gname = groups[mem]
            if gname in seen_groups:
                print('seen member', mem, groups[mem])
                continue
            seen_groups.add(gname)
            gr = grs_df_tr.get_group(groups[mem])
            if mem == cent:
                centroids.append(True)
            else:
                centroids.append(False)  # Later I will filter these out to create nr set for sampling.
            cluster_nums.append(j)
            cluster_sizes.append(cluster_size)
            cluster_scores.append(cluster_score)
            cgs.append(gr.CG.iat[0])
            rotas.append(gr.rota.iat[0])
            probe_names.append(gr.probe_name.iat[0])
        j += 1
    df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_coarse_bb_ind=centroids,
                               CG_rep_cluster_number_coarse_bb_ind=cluster_nums, CG_rep_cluster_size_coarse_bb_ind=cluster_sizes,
                               CG_rep_cluster_score_coarse_bb_ind=cluster_scores)).drop_duplicates()
    df__ = pd.merge(df_tr, df_clu, on=['rota', 'CG', 'probe_name'])

    ################
    clu.rmsd_cutoff = 0.5
    # clu.make_square()
    clu.make_adj_mat()
    # clu.make_adj_mat_no_superpose()
    if cg_aa_in_flip_keys and cg_names_in_flip_names:
        index_labels = groups
    else:
        index_labels = None
    clu.cluster(index_labels=index_labels)

    clu_sizes = np.array([len(mems) for mems in clu.mems])
    avg_clu_size = np.mean(clu_sizes)
    cluster_nums = []
    cluster_sizes = []
    cluster_scores = []
    cgs = []
    rotas = []
    probe_names = []
    centroids = []
    reps = []
    bb_ind_score_col = 'C_score_bb_ind'
    j = 1
    cluster_sets = set()
    seen_groups = set()
    for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            cluster_set = tuple(sorted([groups[mem] for mem in mems]))
            if cluster_set in cluster_sets:
                continue
            else:
                cluster_sets.add(cluster_set)
        cluster_size = len(mems)
        cluster_score = np.log(cluster_size / avg_clu_size)

        _cluster_nums = []
        _cluster_sizes = []
        _cluster_scores = []
        _cgs = []
        _rotas = []
        _probe_names = []
        _centroids = []
        _reps = []
        _score_col = []

        for mem in mems:
            gname = groups[mem]
            if gname in seen_groups:
                print('seen member', mem, groups[mem])
                continue
            seen_groups.add(gname)
            gr = grs_df_tr.get_group(groups[mem])
            if mem == cent:
                _centroids.append(True)
            else:
                _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
            _cluster_nums.append(j)
            _cluster_sizes.append(cluster_size)
            _cluster_scores.append(cluster_score)
            _cgs.append(gr.CG.iat[0])
            _rotas.append(gr.rota.iat[0])
            _probe_names.append(gr.probe_name.iat[0])
            _reps.append(False)
            _score_col.append(gr[bb_ind_score_col].iat[0])
        rep_ind = np.argmax(_score_col)
        _reps[rep_ind] = True

        cluster_nums.extend(_cluster_nums)
        cluster_sizes.extend(_cluster_sizes)
        cluster_scores.extend(_cluster_scores)
        cgs.extend(_cgs)
        rotas.extend(_rotas)
        probe_names.extend(_probe_names)
        centroids.extend(_centroids)
        reps.extend(_reps)

        j += 1
    df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_medium_bb_ind=centroids,
                               CG_rep_cluster_number_medium_bb_ind=cluster_nums,
                               CG_rep_cluster_size_medium_bb_ind=cluster_sizes,
                               CG_rep_cluster_score_medium_bb_ind=cluster_scores,
                               CG_rep_medium_bb_ind=reps)).drop_duplicates()
    df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])

    ##################
    clu.rmsd_cutoff = 0.15
    # clu.make_square()
    clu.make_adj_mat()
    # clu.make_adj_mat_no_superpose()
    if cg_aa_in_flip_keys and cg_names_in_flip_names:
        index_labels = groups
    else:
        index_labels = None
    clu.cluster(index_labels=index_labels)

    clu_sizes = np.array([len(mems) for mems in clu.mems])
    avg_clu_size = np.mean(clu_sizes)
    cluster_nums = []
    cluster_sizes = []
    cluster_scores = []
    cgs = []
    rotas = []
    probe_names = []
    centroids = []
    reps = []
    bb_ind_score_col = 'C_score_bb_ind' 
    j = 1
    cluster_sets = set()
    seen_groups = set()
    for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):
        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            cluster_set = tuple(sorted([groups[mem] for mem in mems]))
            if cluster_set in cluster_sets:
                continue
            else:
                cluster_sets.add(cluster_set)
        cluster_size = len(mems)
        cluster_score = np.log(cluster_size / avg_clu_size)

        _cluster_nums = []
        _cluster_sizes = []
        _cluster_scores = []
        _cgs = []
        _rotas = []
        _probe_names = []
        _centroids = []
        _reps = []
        _score_col = []

        for mem in mems:
            gname = groups[mem]
            if gname in seen_groups:
                print('seen member', mem, groups[mem])
                continue
            seen_groups.add(gname)
            gr = grs_df_tr.get_group(groups[mem])
            if mem == cent:
                _centroids.append(True)
            else:
                _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
            _cluster_nums.append(j)
            _cluster_sizes.append(cluster_size)
            _cluster_scores.append(cluster_score)
            _cgs.append(gr.CG.iat[0])
            _rotas.append(gr.rota.iat[0])
            _probe_names.append(gr.probe_name.iat[0])
            _reps.append(False)
            _score_col.append(gr[bb_ind_score_col].iat[0])
        rep_ind = np.argmax(_score_col)
        _reps[rep_ind] = True

        cluster_nums.extend(_cluster_nums)
        cluster_sizes.extend(_cluster_sizes)
        cluster_scores.extend(_cluster_scores)
        cgs.extend(_cgs)
        rotas.extend(_rotas)
        probe_names.extend(_probe_names)
        centroids.extend(_centroids)
        reps.extend(_reps)

        j += 1
    df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, CG_rep_centroid_fine_bb_ind=centroids,
                               CG_rep_cluster_number_fine_bb_ind=cluster_nums, CG_rep_cluster_size_fine_bb_ind=cluster_sizes,
                               CG_rep_cluster_score_fine_bb_ind=cluster_scores, CG_rep_fine_bb_ind=reps)).drop_duplicates()

    df__ = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])


    grs_df_ = df__.groupby(['rota', 'CG', 'probe_name'])

    print('Making sub-clusters...')
    if aa in keep_H.keys():
        df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1) |
                      ((df__.chain == 'X') & (df__.resnum == 10) & (df__.name == keep_H[aa]))]
    else:
        df_noH = df__[df__.apply(lambda x: x['name'][0] != 'H', axis=1)]
    dfx = df_noH[(df_noH.chain == 'X') & (df_noH.resnum == 10) & (~df_noH.name.isin({'O', 'N', 'CA', 'C'}))]
    dfx = dfx.sort_values('name')
    df_y__ = pd.merge(df__, df_tr_y[['name', 'resname', 'chain']].drop_duplicates(), on=['name', 'resname', 'chain'])
    df_noHx = pd.concat([dfx, df_y__])

    dfs_subrep = []
    for qq, dc in df_noHx.groupby('CG_rep_cluster_number_fine_bb_ind'):
        groups = []
        coords = []
        for n, g in dc.groupby(['rota', 'CG', 'probe_name']):
            groups.append(n)
            coords.append(g[['c_x', 'c_y', 'c_z']].values)
            if cg_aa_in_flip_keys and cg_names_in_flip_names:
                coords.append(flip_cg_coords(g))
                groups.append(n)
            if aa in flip_dict.keys():
                coords.append(flip_x_coords(g))
                groups.append(n)
            if cg_aa_in_flip_keys and cg_names_in_flip_names and aa in flip_dict.keys():
                coords.append(flip_cg_x_coords(g))
                groups.append(n)

        coords_ = []
        groups_ = []
        for n, c in zip(groups, coords):
            if len(c) != num_atoms:
                continue
            coords_.append(c)
            groups_.append(n)
        coords = coords_
        groups = groups_

        if len(coords) == 0:
            continue

        coords_arr = np.array(coords, dtype=np.float32)

        clu = Cluster()
        clu.pdb_coords = coords_arr
        clu.make_pairwise_rmsd_mat(maxdist=True, superpose=False)
        clu.rmsd_cutoff = 0.25
        clu.make_square()
        clu.make_adj_mat()
        # clu.make_adj_mat_no_superpose()
        if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
            index_labels = groups
        else:
            index_labels = None
        clu.cluster(index_labels=index_labels)

        clu_sizes = np.array([len(mems) for mems in clu.mems])
        avg_clu_size = np.mean(clu_sizes)
        cluster_nums = []
        cluster_sizes = []
        cluster_scores = []
        cgs = []
        rotas = []
        probe_names = []
        centroids = []
        reps = []
        bb_ind_score_col = 'C_score_bb_ind' 
        j = 1
        cluster_sets = set()
        seen_groups = set()
        for i, (cent, mems) in enumerate(zip(clu.cents, clu.mems)):

            if (cg_aa_in_flip_keys and cg_names_in_flip_names) or aa in flip_dict.keys():
                cluster_set = tuple(sorted([groups[mem] for mem in mems]))
                if cluster_set in cluster_sets:
                    continue
                else:
                    cluster_sets.add(cluster_set)
            cluster_size = len(mems)
            cluster_score = np.log(cluster_size / avg_clu_size)

            _cluster_nums = []
            _cluster_sizes = []
            _cluster_scores = []
            _cgs = []
            _rotas = []
            _probe_names = []
            _centroids = []
            _reps = []
            _score_col = []

            for mem in mems:
                gname = groups[mem]
                if gname in seen_groups:
                    print('seen member', mem, groups[mem])
                    continue
                seen_groups.add(gname)
                gr = grs_df_.get_group(groups[mem])
                if mem == cent:
                    _centroids.append(True)
                else:
                    _centroids.append(False)  # Later I will filter these out to create nr set for sampling.
                _cluster_nums.append(j)
                _cluster_sizes.append(cluster_size)
                _cluster_scores.append(cluster_score)
                _cgs.append(gr.CG.iat[0])
                _rotas.append(gr.rota.iat[0])
                _probe_names.append(gr.probe_name.iat[0])
                _reps.append(False)
                _score_col.append(gr[bb_ind_score_col].iat[0])
            rep_ind = np.argmax(_score_col)
            _reps[rep_ind] = True

            cluster_nums.extend(_cluster_nums)
            cluster_sizes.extend(_cluster_sizes)
            cluster_scores.extend(_cluster_scores)
            cgs.extend(_cgs)
            rotas.extend(_rotas)
            probe_names.extend(_probe_names)
            centroids.extend(_centroids)
            reps.extend(_reps)
            j += 1

        df_clu = pd.DataFrame(dict(CG=cgs, rota=rotas, probe_name=probe_names, sc_rep_centroid_fine_bb_ind=centroids,
                                   sc_rep_cluster_number_fine_bb_ind=cluster_nums, sc_rep_cluster_size_fine_bb_ind=cluster_sizes,
                                   sc_rep_cluster_score_fine_bb_ind=cluster_scores,
                                   sc_rep_fine_bb_ind=reps)).drop_duplicates()

        df_subrep = pd.merge(df__, df_clu, on=['rota', 'CG', 'probe_name'])
        dfs_subrep.append(df_subrep)

    if len(dfs_subrep) > 0:
        df___ = pd.concat(dfs_subrep)
        df_reps.append(df___)

    df_rep_bb_ind = pd.concat(df_reps)
    df_rep_bb_ind.reset_index(inplace=True, drop=True)

    cols_bb_ind_only = list(set(df_rep_bb_ind.columns) - set(df_rep.columns))
    cols_bb_ind_only.extend(['rota', 'CG', 'probe_name'])
    df_rep = pd.merge(df_rep, df_rep_bb_ind[cols_bb_ind_only].drop_duplicates(), on=['rota', 'CG', 'probe_name'])

    df_rep.reset_index(inplace=True, drop=True)
    # print('len dfrep 3', len(df_rep[['CG', 'rota', 'probe_name']].drop_duplicates()))

    df_rep = set_contacts(df_rep) #This resets the contact_type column,
                                  # which may or not have been set correctly beforehand

    df_rep.to_parquet(outpath + f, engine='pyarrow', compression='gzip')

