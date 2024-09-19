import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
from combs2.design.functions import add_slash, listdir_mac
import pandas as pd


def retype_df(df):
    dts = df.dtypes.copy()
    cat_cols = ['chain',
                'resname',
                'name',
                'atom_type_label',
                'evaluation',
                'rotamer',
                'rama',
                'pdb_chain',
                'pdb_segment',
                'dssp',
                'ABPLE',
                'probe_name',
                'pdb_name',
                'contact_type']
    dts[dts.index.isin(cat_cols)] = 'category'
    dts[dts == 'float64'] = 'float32'
    dts[dts == 'int64'] = 'int32'
    return df.astype(dts)


def main():
    par = argparse.ArgumentParser()
    par.add_argument('--db', required=True, help='path to vdM databases')
    args = par.parse_args()
    db_path = add_slash(args.db)
    for cg in listdir_mac(db_path):
        print(cg)
        path = db_path + cg + '/'
        for f in listdir_mac(path):
            print('\t', f)
            df = pd.read_parquet(path + f)
            df = retype_df(df)
            df.to_parquet(path + f, engine='pyarrow', compression='gzip')


if __name__ == '__main__':
    main()