from ..design.functions import get_mat, get_nr_reps
from ..design.cluster import Cluster
import pickle
import pandas as pd
import numpy as np
import os
import traceback


def run_nr(cg_dict, path_to_probe_paths, outpath, path_to_comb_output):

    with open(path_to_probe_paths, 'rb') as infile:
        probe_paths = pickle.load(infile)

    try:
        os.makedirs(outpath)
    except:
        pass

    filepath_to_probe_paths = '/'.join(probe_paths[0].split('/')[:-1])

    dfs = []
    for p in probe_paths:
        f = p.split('/')[-1]
        try:
            dfs.append(pd.read_parquet(path_to_comb_output + f.split('.')[0] + '.parquet.gzip'))
        except:
            print('Problem with probe path', p)
            traceback.print_exc()

    df = pd.concat(dfs)
    df_x_10 = df[['CG', 'rota', 'probe_name', 'chain', 'resname', 'resnum']][(df.chain == 'X') & (df.resnum == 10)]
    df_x_10 = df_x_10[['CG', 'rota', 'probe_name', 'chain', 'resname']].drop_duplicates()
    dfy = df[df.chain == 'Y'][['CG', 'rota', 'probe_name', 'chain', 'resname', 'seq']].drop_duplicates()
    for resn in set(df_x_10.resname):
        dfs_nr_reps = []
        df_resn_x = df_x_10[df_x_10.resname == resn]
        df_resn_y = pd.merge(dfy, df_resn_x[['CG', 'rota', 'probe_name']], on=['CG', 'rota', 'probe_name'])
        for resn_cg in cg_dict.keys():
            df_resn_y_ = df_resn_y[df_resn_y.resname == resn_cg]
            df_resn_y_ = df_resn_y_[['rota', 'CG', 'probe_name', 'seq']].drop_duplicates()
            seqs = []
            groups = []
            for rota, cg, pn, s in df_resn_y_.values:
                if len(s) != 27:
                    continue
                groups.append((rota, cg, pn))
                seqs.append(np.array(list(s[:6] + s[7:13] + s[14:20] + s[21:27])))
            seqs = np.array(seqs)

            p_mat = get_mat(seqs)
            clu_seq = Cluster()
            clu_seq.rmsd_cutoff = 0.24
            clu_seq.rmsd_mat = p_mat
            clu_seq.make_square()
            clu_seq.make_adj_mat()
            clu_seq.cluster()

            nr_reps = get_nr_reps(clu_seq, probe_paths, groups, filepath_to_probe_paths)
            d = pd.DataFrame(nr_reps, columns=['rota', 'CG', 'probe_name'])
            dfs_nr_reps.append(d)

        df_nr_reps_all = pd.concat(dfs_nr_reps)
        df_resn = pd.merge(df, df_nr_reps_all, on=['CG', 'rota', 'probe_name'])
        df_resn['resname_rota'] = resn
        df_resn.to_parquet(outpath + resn + '.parquet.gzip', engine='pyarrow',
                      compression='gzip')