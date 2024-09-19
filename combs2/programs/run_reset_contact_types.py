import combs2
import pandas as pd
import os
from multiprocessing import Pool


base_path = '/Volumes/disk1/Combs2/database/20211005/vdMs/'
filepaths = [base_path + d + '/' + f for d in [d for d in os.listdir(base_path) if d[0] != '.']
             for f in [f for f in os.listdir(base_path + d) if f[0] != '.']]


def set_contacts(df):
    df = df.drop(columns='contact_type')
    contacts = []
    for n, g in df.groupby(['CG', 'rota', 'probe_name']):
        contact_info = list(n)
        contact_info.append(combs2.parse.comb.get_contact_type(g))
        contacts.append(contact_info)
    df_contact_type = pd.DataFrame(contacts, columns=['CG', 'rota', 'probe_name', 'contact_type']).drop_duplicates()
    df_contact_type = df_contact_type.astype(dict(contact_type='category'))
    df = pd.merge(df, df_contact_type, on=['CG', 'rota', 'probe_name'])
    df = df.sort_values(['CG', 'rota', 'probe_name'])
    df.reset_index(drop=True, inplace=True)
    return df


def run_reset_contacts(filepath):
    print(filepath)
    df = pd.read_parquet(filepath)
    df = set_contacts(df)
    df.to_parquet(filepath, engine='pyarrow', compression='gzip')


if __name__ == '__main__':
    with Pool(7) as p:
        p.map(run_reset_contacts, filepaths)