import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2
from combs2.design.functions import add_slash
import pandas as pd
from collections import defaultdict



def _listdir(path):
    return [d for d in os.listdir(path) if d[0] != '.']


def combine(path_to_cg_folders):
    path_to_cg_folders = add_slash(path_to_cg_folders)
    for cg in _listdir(path_to_cg_folders):
        for cg_group in _listdir(path_to_cg_folders + cg):
            dfs = defaultdict(list)
            cg_gr_path = path_to_cg_folders + cg + '/' + cg_group + '/'
            for abple in _listdir(cg_gr_path):
                abple_path = cg_gr_path + abple + '/'
                for file in _listdir(abple_path):
                    resname = file.split('.')[0]
                    dfs[resname].append(pd.read_parquet(abple_path + file))
                os.system('rm ' + abple_path + '*')
                os.system('rmdir ' + abple_path)
            for resname, dfs_rn in dfs.items():
                df = pd.concat(dfs_rn).drop_duplicates()
                df.to_parquet(cg_gr_path + resname + '.parquet.gzip',
                              engine='pyarrow', compression='gzip')


def main():
    par = argparse.ArgumentParser()
    par.add_argument('--path_to_lig_cg_folders', default=None, help='path to superimposed-ligand CG folders')
    args = par.parse_args()

    combine(args.path_to_lig_cg_folders)


if __name__ == '__main__':
    main()