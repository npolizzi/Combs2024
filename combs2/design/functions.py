__all__ = ['writePDBStream', 'find_buried_unsatisfied_hbonds', 'read_lig_txt',
           'make_lig_hbond_dict', 'make_lig_atom_type_dict', 'get_ABPLE', 'df_ideal_ala']

from numba import jit, prange
import numpy as np
import prody as pr
import os
from .dataframe import make_df_from_prody
from prody import parsePDB, AtomGroup, writePDB
from .convex_hull import AlphaHull, partition_res_by_burial
from pandas import concat, DataFrame, merge, read_pickle, read_parquet, Index, MultiIndex
from collections import defaultdict
import pickle
from .transformation import get_rot_trans
from .cluster import _make_pairwise_maxdist_mat, get_max
from .constants import one_letter_code, flip_dict, cg_flip_dict, atom_types_sortkey, cgs, residue_sc_names
from itertools import chain
from sklearn.neighbors import NearestNeighbors
try:
    import pickle5
except:
    pass


_dir = os.path.dirname(__file__)
path_to_rosetta_polar_atom_props = os.path.join(_dir, '../files/rosetta_polar_atom_types.pkl')
path_to_rosetta_atom_types = os.path.join(_dir, '../files/rosetta_types_to_atom_types.pkl')
path_to_abple_dict = os.path.join(_dir, '../files/abple_dict.pkl')
path_to_ideal_ala = os.path.join(_dir, '../files/ideal_alanine_bb_only.pkl')
path_to_ideal_ala_sc = os.path.join(_dir, '../files/ideal_alanine.pkl')
path_to_gly = os.path.join(_dir, '../files/gly.pkl')
path_to_propensities_folder = os.path.join(_dir, '../files/propensities/')
path_to_files_folder = os.path.join(_dir, '../files/')

try:
    cg_dicts = read_pickle(path_to_files_folder + 'cg_dicts.pkl')
    abple_dict = read_pickle(path_to_abple_dict)
    df_ideal_ala = read_pickle(path_to_ideal_ala)
    df_ideal_ala_sc = read_pickle(path_to_ideal_ala_sc)
    df_gly = read_pickle(path_to_gly)
    cg_dfs = read_pickle(path_to_files_folder + 'cg_dataframes.pkl')
    rotamer_dfs = read_pickle(path_to_files_folder + 'rotamer_dataframes.pkl')
except Exception:
    with open(path_to_files_folder + 'cg_dicts.pkl', 'rb') as infile:
        cg_dicts = pickle5.load(infile)
    with open(path_to_abple_dict, 'rb') as infile:
        abple_dict = pickle5.load(infile)
    with open(path_to_ideal_ala, 'rb') as infile:
        df_ideal_ala = pickle5.load(infile)
    with open(path_to_ideal_ala_sc, 'rb') as infile:
        df_ideal_ala_sc = pickle5.load(infile)
    with open(path_to_gly, 'rb') as infile:
        df_gly = pickle5.load(infile)
    with open(path_to_files_folder + 'cg_dataframes.pkl', 'rb') as infile:
        cg_dfs = pickle5.load(infile)
    with open(path_to_files_folder + 'rotamer_dataframes.pkl', 'rb') as infile:
        cg_dfs = pickle5.load(infile)

# def make_rosetta_atom_props(path_to_rosetta_atom_props):
#     # path_to_rosetta_atom_props = \
#     #     '~/rosetta_bin_mac_2020.08.61146_bundle/main/database/chemical/atom_type_sets/fa_standard/atom_properties.txt'
#     with open(path_to_rosetta_atom_props, 'r') as infile:
#         infile.readline()
#         data = []
#         for line in infile:
#             if line[:2] == '#Z':
#                 continue
#             if line[0] == '#':
#                 break
#             spl = line.split()
#             name = spl[0]
#             if 'ACCEPTOR' in spl:
#                 acc = True
#             else:
#                 acc = False
#             if 'DONOR' in spl:
#                 don = True
#             else:
#                 don = False
#             if 'POLAR_HYDROGEN' in spl:
#                 don = True
#             if (not don) and (not acc):
#                 continue
#             data.append((name, acc, don))
#     rosetta_polar_atom_types = DataFrame(data, columns=['rosetta_atom_type', 'acceptor', 'donor'])
#     return rosetta_polar_atom_types


def make_atom_type_df(path_to_lig_params_file):
    with open(path_to_lig_params_file, 'r') as infile:
        data = []
        for line in infile:
            spl = line.split()
            if spl[0] == 'ATOM':
                data.append((spl[1], spl[2]))
    atom_type_df = DataFrame(data, columns=['name', 'rosetta_atom_type'])
    return atom_type_df


def make_bond_type_df(path_to_lig_params_file):
    bt = defaultdict(list)
    with open(path_to_lig_params_file, 'r') as infile:
        for line in infile:
            spl = line.split()
            if spl[0] == 'BOND_TYPE':
                bt[spl[1]].append(spl[2])
                bt[spl[2]].append(spl[1])
    bond_type_df = DataFrame([(k,v) for k,v in bt.items()], columns=['name', 'bonded_to'])
    return bond_type_df


def make_lig_hbond_dict(lig_resname, path_to_lig_params_file):
    """

    Parameters
    ----------
    lig_resname : str
    path_to_lig_params_file : str

    Returns
    -------
    dict

    """
    atom_type_df = make_atom_type_df(path_to_lig_params_file)
    bond_type_df = make_bond_type_df(path_to_lig_params_file)
    rosetta_polar_atom_types = read_pickle(path_to_rosetta_polar_atom_props)
    dfm = merge(atom_type_df, rosetta_polar_atom_types, on='rosetta_atom_type')
    dfmm = merge(dfm, bond_type_df, on='name')

    hbond_dict = defaultdict(dict)
    for n, row in dfmm.iterrows():
        if row['acceptor']:
            acc_list = [row['name']]
            bt = row['bonded_to']
            if len(row['bonded_to']) == 1:
                bt.append(bt[0])
            [acc_list.append(bn) for bn in bt]
            hbond_dict[row['name']]['acceptor'] = tuple(acc_list)
        if row['donor']:
            name = row['name']
            bt = row['bonded_to']
            don_list = []
            if name[0] != 'H':
                bt = [b for b in bt if b[0] == 'H']
                for b in bt:
                    don_list.append((b, name))
            else:
                for b in bt:
                    don_list.append((name, b))
            if len(don_list) > 0:
                hbond_dict[name]['donor'] = don_list
    hbond_dict = {lig_resname: hbond_dict}
    return hbond_dict


def make_lig_atom_type_dict(lig_resname, path_to_lig_params_file):
    """

    Parameters
    ----------
    lig_resname : str
    path_to_lig_params_file : str

    Returns
    -------
    dict

    """
    atom_type_df = make_atom_type_df(path_to_lig_params_file)
    num_atoms = len(atom_type_df)
    rosetta_atom_types = read_pickle(path_to_rosetta_atom_types)
    df = merge(atom_type_df, rosetta_atom_types, on='rosetta_atom_type')
    if len(df) != num_atoms:
        raise Exception('Missing rosetta atom types in files/rosetta_types_to_atom_types.pkl')
    dict_types = {r['name']: r['atom_type'] for n, r in df.iterrows()}
    atom_type_dict = {lig_resname: dict_types}
    return atom_type_dict


def get_heavy(row):
    name = row['name']
    if name[0] == 'H':
        return False
    if name[:2] in {'1H', '2H', '3H', '4H'}:
        return False
    return True


def get_HA3(row, seg_chain_resnums):
    name = row['name']
    seg_chain_resnum = row['seg_chain_resnum']
    if name in ['HA3', '2HA'] and seg_chain_resnum in seg_chain_resnums:
        return True
    return False


def get_HA2(row, seg_chain_resnums):
    name = row['name']
    seg_chain_resnum = row['seg_chain_resnum']
    if name in ['HA2', '1HA'] and seg_chain_resnum in seg_chain_resnums:
        return True
    return False


def read_lig_txt(path_to_txt):
    """

    Parameters
    ----------
    path_to_txt : str

    Returns
    -------
    `~pandas.DataFrame`
        Dataframe containing ligand to amino-acid correspondence.

    """
    dtype_dict = {'lig_resname': str,
                  'lig_name': str,
                  'resname': str,
                  'name': str,
                  'CG_type': str,
                  'CG_group': int,
                  'CG_ligand_coverage': int,
                  'rmsd': bool,
                  'is_acceptor': bool, 
                  'is_donor': bool, 
                  'is_not_acceptor': bool, 
                  'is_not_donor': bool,
                  }
    data = []
    with open(path_to_txt, 'r') as infile:
        for line in infile:
            if line[0] == '#':
                continue
            # try:
            #     spl = line.strip().split()
            #     if len(spl) != 7:
            #         continue
            #     data.append(spl)
            # except Exception:
            #     pass
            try:
                # spl = line.strip().split()
                # if len(spl) == 7:
                #     spl.append(True)
                # elif len(spl) == 8 and 'no_rmsd' in line:
                #     spl[-1] = False
                # else:
                #     continue
                spl = line.strip().split()
                if len(spl) >= 7 and len(spl) <= 9:
                    new_spl = np.zeros(12, dtype=object)
                    new_spl[:7] = spl[:7]
                    if 'no_rmsd' not in line:
                        new_spl[7] = True
                    if 'is_acceptor' in line:
                        new_spl[8] = True
                    if 'is_donor' in line:
                        new_spl[9] = True
                    if 'is_not_acceptor' in line:
                        new_spl[10] = True
                    if 'is_not_donor' in line:
                        new_spl[11] = True
                    spl.append(True)
                else:
                    continue
                data.append(new_spl)
            except Exception:
                pass
    df = DataFrame(data, columns=['lig_resname', 'lig_name', 'resname', 'name', 'CG_type', 
                                  'CG_group', 'CG_ligand_coverage', 'rmsd', 'is_acceptor', 
                                  'is_donor', 'is_not_acceptor', 'is_not_donor'])
    return df.astype(dtype=dtype_dict)


def print_dataframe(df, outpath='./', filename='', tag='', prefix='', b_factor_column=None):
    df = df.copy()
    try:
        os.makedirs(outpath)
    except:
        pass
    if filename == '':
        cg = df.CG.iat[0]
        rota = df.rota.iat[0]
        probe_name = df.probe_name.iat[0]
        filename = 'CG_'+ str(cg) + '_rota_' + str(rota) + '_' + probe_name
    if 'chain' not in df.columns:
        df['chain'] = 'A'
    ag = AtomGroup()
    ag.setCoords(df[['c_x','c_y','c_z']].values)
    ag.setResnums(df['resnum'].values)
    ag.setResnames(df['resname'].values)
    ag.setNames(df['name'].values)
    ag.setChids(df['chain'].values)
    ag.setSegnames(df['chain'].values)
    heteroflags = ag.getSegnames() == 'L'
    ag.setFlags('hetatm', heteroflags)
    if 'beta' in df.columns and b_factor_column is None:
        ag.setBetas(df['beta'].values)
    elif b_factor_column is not None:
        ag.setBetas(df[b_factor_column].values)
    if 'occ' not in df.columns:   
        df['occ'] = 1
    writePDB(outpath + prefix + filename + tag + '.pdb.gz', ag, occupancy=df['occ'].values)


def get_extended(gr, path_to_prody='./', res_before=0, res_after=0):
    with open(path_to_prody + '_'.join(gr.probe_name.iat[0].split('_')[:3]) + '.pkl', 'rb') as infile:
        prody_pdb = pickle.load(infile)
    gr10x = gr[(gr.chain == 'X') & (gr.resnum == 10)]
    resnum10x = gr10x.pdb_resnum.iat[0]
    chain10x = gr10x.pdb_chain.iat[0]
    sel_string_x = 'chain ' + chain10x + ' resnum `' + str(resnum10x - res_before) + 'to' + str(resnum10x + res_after) + '`'
    selx = prody_pdb.select(sel_string_x)
    gry = gr[(gr.chain == 'Y')]
    resnumy = gry.pdb_resnum.iat[0]
    chainy = gry.pdb_chain.iat[0]
    namesy = set(gry.name)
    sel_string_y = 'chain ' + chainy + ' resnum `' + str(resnumy) + '` name ' + ' '.join(namesy)
    sely = prody_pdb.select(sel_string_y)
    sel = selx | sely
    gr_ = make_df_from_prody(sel, include_betas_occupancies=True)
    gr_['CG'] = gr.CG.iat[0]
    gr_['rota'] = gr.rota.iat[0]
    gr_['probe_name'] = gr.probe_name.iat[0]
    gr = gr_
    gr['pdb_chain'] = gr.chain
    gr['pdb_resnum'] = gr.resnum
    cg_sel = (gr.pdb_chain == chainy) & (gr.pdb_resnum == resnumy)
    gr.loc[cg_sel, 'chain'] = 'Y'
    gr.loc[cg_sel, 'resnum'] = 10
    rota_sel = ~cg_sel
    gr.loc[rota_sel, 'chain'] = 'X'
    gr.loc[rota_sel, 'resnum'] = gr.loc[rota_sel, 'resnum'] - resnum10x + 10
    return gr


def get_centroid(df_cluster):
    centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
    centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
    gr_name = list(centroid_grs.groups.keys())[0]
    df_c = centroid_grs.get_group(gr_name)
    return df_c


def print_cluster(df, cluster_num, max_members=0, outpath='./', tag='',
                  ABPLE=None, dssp=None, hbonly=False, use_bb_ind_centroid=True,
                  print_bb_ind_centroid=False, print_resnum_10_only=False,
                  res_before=0, res_after=0, path_to_prody='./',
                  align_centroids=False, centroid_alignment_cluster=1, ignore_ss_hb_rankings=False,
                  ):
    """
    Note that the centroids of the specialty clusters (ABPLE, dssp, hbonly) are chosen such that they
    are the closest to the centroid of the bb independent cluster.  The speciality clusters are printed
    such that they are aligned to the specialty cluster centroid, which means the the maximum distance from
    the specialty centroid might be larger than for what defines the bb-independent cluster (max atom distance
    of 0.65 Angstroms). I am including an option that the centroid can be defined as the real centroid
    (bb-independent) of the cluster, such that the members of the specialty cluster can be aligned to the
    "real centroid", even if the "real centroid" isn't a member of the specialty cluster (e.g. if the
    "real centroid" has an ABPLE of B but the specialty cluster is for ABPLE of A).

    Parameters
    ----------
    df
    cluster_num
    max_members
    outpath
    tag
    ABPLE
    dssp
    hbonly
    use_bb_ind_centroid
    print_bb_ind_centroid
    res_before
    res_after
    path_to_prody
    align_centroids
    centroid_alignment_cluster

    Returns
    -------

    """
    if print_resnum_10_only:
        df = df[df.resnum==10]

    cluster_num_for_print = cluster_num

    df_resname_names_y = df[['resname', 'name']][df.chain=='Y'].drop_duplicates()

    cg_aa_in_flip_keys = all([aa_cg in cg_flip_dict.keys() for aa_cg in set(df_resname_names_y['resname'])])
    if cg_aa_in_flip_keys:
        cg_names_in_flip_names = all([len(cg_flip_dict[aa_cg] - set(gr['name'])) == 0 for aa_cg, gr in df_resname_names_y.groupby('resname')])
    else:
        cg_names_in_flip_names = False

    if hbonly:
        hb = 'hb_'
    else:
        hb = ''

    prefix = 'cluster_'

    if ABPLE is None and dssp is None and not hbonly:
        cluster_col = 'cluster_number'
        centroid = 'centroid'
        df_cluster = df[(df[cluster_col] == cluster_num)]
        df_c = df_cluster[df_cluster[centroid]]
        df_c_print = df_c
        df_nc = df_cluster[~df_cluster[centroid]]
    elif ABPLE is None and dssp is None and hbonly:
        if ignore_ss_hb_rankings:
            cluster_col = 'cluster_number'
        else:
            cluster_col = 'cluster_rank_hb_bb_ind'
        # there is currently no column corresponding to hb_bb_ind centroid,
                                # so need to calculate on the fly
        prefix += hb
        df_cluster = df[(df[cluster_col] == cluster_num)]
        dfx = df_cluster[['CG', 'rota', 'probe_name']][(df_cluster.chain == 'X') & (df_cluster.resnum == 10) & (~df_cluster.contact_hb.isna())].drop_duplicates()
        df_cluster = merge(df_cluster, dfx, on=['CG', 'rota', 'probe_name'])
        if use_bb_ind_centroid:
            cluster_num = df_cluster.cluster_number.iat[0]
            df_c = df[(df.cluster_number==cluster_num) & (df.centroid)]
            if print_bb_ind_centroid:
                df_c_print = df_c
        else:
            centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
            centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
            gr_name = list(centroid_grs.groups.keys())[0]
            df_c = centroid_grs.get_group(gr_name)
        if not print_bb_ind_centroid:
            centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
            centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
            gr_name = list(centroid_grs.groups.keys())[0]
            df_c_print = centroid_grs.get_group(gr_name)
        df_nc = df_cluster[~df_cluster.index.isin(df_c_print.index)]
    elif ABPLE is not None and dssp is None:
        if ignore_ss_hb_rankings:
            cluster_col = 'cluster_number'
        else:
            cluster_col = 'cluster_rank_' + hb + 'ABPLE_' + ABPLE
        centroid = 'centroid_ABPLE_' + ABPLE
        prefix += hb + 'ABPLE_' + ABPLE + '_'
        df_cluster = df[(df[cluster_col] == cluster_num)]
        dfx = df_cluster[['CG', 'rota', 'probe_name']][(df_cluster.chain == 'X') & (df_cluster.resnum == 10) & (
            df_cluster.ABPLE == ABPLE)].drop_duplicates()
        df_cluster = merge(df_cluster, dfx, on=['CG', 'rota', 'probe_name'])
        if hbonly:
            dfx = df_cluster[['CG', 'rota', 'probe_name']][(df_cluster.chain == 'X') & (df_cluster.resnum == 10) & (
                ~df_cluster.contact_hb.isna())].drop_duplicates()
            df_cluster = merge(df_cluster, dfx, on=['CG', 'rota', 'probe_name'])

        if use_bb_ind_centroid:
            cluster_num = df_cluster.cluster_number.iat[0]
            df_c = df[(df.cluster_number == cluster_num) & (df.centroid)]
            # print('cluster_number', cluster_num)
            # print('df_c')
            # print(df_c)
            if print_bb_ind_centroid:
                df_c_print = df_c
        else:
            df_c = df_cluster[df_cluster[centroid]]
            # df_nc = df_cluster[~df_cluster[centroid]]
            if len(df_c) == 0:
                centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
                centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
                gr_name = list(centroid_grs.groups.keys())[0]
                df_c = centroid_grs.get_group(gr_name)
        if not print_bb_ind_centroid:
            df_c_print = df_cluster[df_cluster[centroid]]
            # df_nc = df_cluster[~df_cluster[centroid]]
            if len(df_c_print) == 0:
                centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
                centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
                gr_name = list(centroid_grs.groups.keys())[0]
                df_c_print = centroid_grs.get_group(gr_name)
        df_nc = df_cluster[~df_cluster.index.isin(df_c_print.index)]
    elif dssp is not None and ABPLE is None:
        if ignore_ss_hb_rankings:
            cluster_col = 'cluster_number'
        else:
            cluster_col = 'cluster_rank_' + hb + 'dssp_' + dssp
        centroid = 'centroid_dssp_' + dssp
        prefix += hb + 'dssp_' + dssp + '_'
        df_cluster = df[(df[cluster_col] == cluster_num)]
        dfx = df_cluster[['CG', 'rota', 'probe_name']][(df_cluster.chain == 'X') & (df_cluster.resnum == 10) & (
            df_cluster.dssp == dssp)].drop_duplicates()
        df_cluster = merge(df_cluster, dfx, on=['CG', 'rota', 'probe_name'])
        if hbonly:
            dfx = df_cluster[['CG', 'rota', 'probe_name']][(df_cluster.chain == 'X') & (df_cluster.resnum == 10) & (
                ~df_cluster.contact_hb.isna())].drop_duplicates()
            df_cluster = merge(df_cluster, dfx, on=['CG', 'rota', 'probe_name'])
        if use_bb_ind_centroid:
            cluster_num = df_cluster.cluster_number.iat[0]
            df_c = df[(df.cluster_number == cluster_num) & (df.centroid)]
            if print_bb_ind_centroid:
                df_c_print = df_c
        else:
            df_c = df_cluster[df_cluster[centroid]]
            # df_nc = df_cluster[~df_cluster[centroid]]
            if len(df_c) == 0:
                centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
                centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
                gr_name = list(centroid_grs.groups.keys())[0]
                df_c = centroid_grs.get_group(gr_name)
        if not print_bb_ind_centroid:
            df_c_print = df_cluster[df_cluster[centroid]]
            # df_nc = df_cluster[~df_cluster[centroid]]
            if len(df_c_print) == 0:
                centroid = df_cluster[df_cluster.maxdist_to_centroid == df_cluster.maxdist_to_centroid.min()]
                centroid_grs = centroid.groupby(['CG', 'rota', 'probe_name'])
                gr_name = list(centroid_grs.groups.keys())[0]
                df_c_print = centroid_grs.get_group(gr_name)
        df_nc = df_cluster[~df_cluster.index.isin(df_c.index)]
        
    if align_centroids:
        df_cluster_0 = df[(df[cluster_col] == centroid_alignment_cluster)]
        df_c0 = get_centroid(df_cluster_0)
        df_c0_bb10 = df_c0[(df_c0.chain == 'X') & (df_c0.resnum == 10) & (df_c0.name.isin({'CA', 'N', 'C', 'CB'}))]
        df_c0_bb10_coords = df_c0_bb10[['c_x', 'c_y', 'c_z']].values
        # df_c = df[(df.cluster_number == cluster_num) & df.centroid]
        df_c_bb10 = merge(df_c0_bb10[['chain', 'resnum', 'name']], df_c, on=['chain', 'resnum', 'name'])
        df_c_bb10_coords = df_c_bb10[['c_x', 'c_y', 'c_z']].values
        # Align backbone of cluster centroid to backbone of centroid of largest cluster.
        R, m_com, t_com = get_rot_trans(df_c_bb10_coords, df_c0_bb10_coords)
        cent_coords = np.dot((df_c[['c_x', 'c_y', 'c_z']].values - m_com), R) + t_com
        df_c = df_c.copy()
        df_c[['c_x', 'c_y', 'c_z']] = cent_coords

    if (res_before != 0) or (res_after != 0):
        df_c_ext = get_extended(df_c_print, path_to_prody, res_before, res_after)
        df_c_ext_bb10 = df_c_ext[(df_c_ext.chain == 'X') & (df_c_ext.resnum == 10) & (df_c_ext.name.isin({'CA', 'N', 'C', 'CB'}))]
        df_c_ext_bb10_coords = df_c_ext_bb10[['c_x', 'c_y', 'c_z']].values
        # df_c = df[(df.cluster_number == cluster_num) & df.centroid]
        df_c_bb10 = merge(df_c_ext_bb10[['chain', 'resnum', 'name']], df_c, on=['chain', 'resnum', 'name'])
        df_c_bb10_coords = df_c_bb10[['c_x', 'c_y', 'c_z']].values
        # Align backbone of cluster centroid to backbone of centroid of largest cluster.
        R, m_com, t_com = get_rot_trans(df_c_ext_bb10_coords, df_c_bb10_coords)
        cent_coords = np.dot((df_c_ext[['c_x', 'c_y', 'c_z']].values - m_com), R) + t_com
        df_c_ext[['c_x', 'c_y', 'c_z']] = cent_coords
        print_dataframe(df_c_ext, outpath, tag='_centroid_1' + tag, prefix=prefix + str(cluster_num_for_print) + '_')
    else:
        print_dataframe(df_c_print, outpath, tag='_centroid_1' + tag, prefix=prefix + str(cluster_num_for_print) + '_')
    max_members -= 1
    if max_members == 0:
        return
    # df_nc = df[(df.cluster_number == cluster_num) & ~df.centroid]
    grs = df_nc.groupby(['CG', 'rota', 'probe_name'])
    num_grs = len(grs)
    if max_members < 0 or max_members > num_grs:
        max_members = num_grs
    df_c_cluster_atoms = df_c[df_c.cluster_atom].sort_values('cluster_order')
    dfc_coords = df_c_cluster_atoms[['c_x', 'c_y', 'c_z']].values

    for i, (n, gr) in enumerate(grs):
        gr_ = gr[gr.cluster_atom].sort_values('cluster_order')
        df_m_cluster_atoms = gr_[['c_x','c_y','c_z']].values
        all_df_m_cluster_atoms = [df_m_cluster_atoms]

        if cg_aa_in_flip_keys and cg_names_in_flip_names:
            all_df_m_cluster_atoms.append(flip_cg_coords(gr_))

        max_dists = []
        transforms = []
        for coords_ in all_df_m_cluster_atoms:
            # print('coords_')
            # print(coords_)
            # print('dfc_coords')
            # print(dfc_coords)
            R, m_com, t_com = get_rot_trans(coords_, dfc_coords)
            super_coords = np.dot((coords_ - m_com), R) + t_com
            arr = np.array([super_coords, dfc_coords], dtype=np.float32)
            max_dist_mat = _make_pairwise_maxdist_mat(arr)
            max_dist_mat = max_dist_mat + max_dist_mat.T
            max_dist = max_dist_mat[0,1]
            max_dists.append(max_dist)
            transforms.append((R, m_com, t_com))

        min_ind = np.argmin(max_dists)
        R, m_com, t_com = transforms[min_ind]

        #this won't work for get_extended. need to fix.
        if (res_before != 0) or (res_after != 0):
            gr = get_extended(gr, path_to_prody, res_before, res_after)
        coords_transformed = np.dot((gr[['c_x','c_y','c_z']].values - m_com), R) + t_com
        df_m = gr.copy()
        df_m[['c_x','c_y','c_z']] = coords_transformed
        print_dataframe(df_m, outpath, tag='_' + str(i + 2) + tag, prefix=prefix + str(cluster_num_for_print) + '_')
        if i + 2 > max_members:
            return


def get_ABPLE(resn, phi, psi):
    try:
        psi = int(np.ceil(psi / 10.0)) * 10
        phi = int(np.ceil(phi / 10.0)) * 10
        if psi == -180:
            psi = -170
        if phi == -180:
            phi = -170
        return abple_dict[resn][psi][phi]
    except ValueError:
        return 'n'
    except KeyError:
        return 'n'


def get_ABPLE_from_sel(sel):
    phi = sel.getData('dssp_phi')[0]
    psi = sel.getData('dssp_psi')[0]
    resname = sel.getResnames()[0]
    return get_ABPLE(resname, phi, psi)


def load_propensities(CG, use_abple=True, use_dssp=False):
    ss = 'bb_ind_'
    if use_abple:
        ss = 'abple_'
    if use_dssp:
        ss = 'dssp_'
    propensity_filename = path_to_propensities_folder + 'aa_propensities_' + ss + CG + '.pkl'
    with open(propensity_filename, 'rb') as infile:
        propensities = pickle.load(infile)
    return propensities


def get_aas_with_enriched_propensity(CG, use_abple=True, use_dssp=False, propensity_threshold=1.0):
    prop = load_propensities(CG=CG, use_abple=use_abple, use_dssp=use_dssp)
    prop_dict = dict()
    if use_abple or use_dssp:
        for ss in prop.keys():
            aas = []
            for aa, p in prop[ss].items():
                if p > propensity_threshold:
                    aas.append(one_letter_code[aa])
            prop_dict[ss] = ''.join(aas)
    else:  #backbone independent
        aas = []
        for aa, p in prop.items():
            if p > propensity_threshold:
                aas.append(one_letter_code[aa])
        prop_dict['bb_ind'] = ''.join(aas)
    return prop_dict


def check_string_overlap(hb_only_residues, all_contact_residues, cg_name='All CGs'):
    if hb_only_residues == '' and all_contact_residues == '':
        all_contact_residues = 'ACDEFGHIKLMNPQRSTVWY'
    elif hb_only_residues == '' and all_contact_residues != '':
        aas = set('ACDEFGHIKLMNPQRSTVWY')
        hb_only_residues = ''.join(aas - set(all_contact_residues))
    elif hb_only_residues != '' and all_contact_residues == '':
        aas = set('ACDEFGHIKLMNPQRSTVWY')
        all_contact_residues = ''.join(aas - set(hb_only_residues))

    overlap_set = set(hb_only_residues) & set(all_contact_residues)
    if len(overlap_set) > 0:
        print('*** WARNING: hb_only_residues and all_contact_residues overlap ***')
        print('*** Overlapping residues will be assigned as all_contact_residues ***')
        hb_only_residues = ''.join(set(hb_only_residues) - overlap_set)
    return hb_only_residues, all_contact_residues


def write_resfile(template, CGs, outpath='./', filename='resfile', tag='', resindices=None, segs_chains_resnums=None,
                  pikaa_dict=None, bb_dep=1,
                 use_enriched_vdMs=True, CA_burial_distance=None, exclude_exposed=True, exclude_intermed=False,
                 exclude_buried=False, top_exposed=None, top_intermed=None, top_buried=None, alpha_hull_radius=9,
                  use_propensities=True, dist_from_CoM=15,
                  propensity_threshold=1.0, use_abple=True, use_dssp=False, path_to_pdb_for_dssp=None,
                  allowed_exposed='ADEGHKMNPQRSTWY', allowed_intermed='AFGHILMNPQSTVWY',
                  allowed_buried='ACFGHILMPSTVWY', rotamers=None,
                  hb_only_residues='', all_contact_residues='', cg_is_hb_donor=[], cg_is_hb_acceptor=[],
                  cg_is_not_hb_donor=[], cg_is_not_hb_acceptor=[],
                  pikaa_override=None, CG_specific_residues=None):

    """
    cg_is_hb_donor=[], etc is an all or nothing list because we do not load CGs by CG group number.
    So if a CG has groups that are mixtures of donors and acceptors, need to load all of them.
    If all of the CG groups are donors only, then set this option.
    """
    try:
        os.makedirs(outpath)
    except:
        pass
    prody_pdb = template.pdb_w_CB.copy()
    # if hb_only_residues == '' and all_contact_residues == '':
    #     all_contact_residues = 'ACDEFGHIKLMNPQRSTVWY'
    # elif hb_only_residues == '' and all_contact_residues != '':
    #     aas = set('ACDEFGHIKLMNPQRSTVWY')
    #     hb_only_residues = ''.join(aas - set(all_contact_residues))
    # elif hb_only_residues != '' and all_contact_residues == '':
    #     aas = set('ACDEFGHIKLMNPQRSTVWY')
    #     all_contact_residues = ''.join(aas - set(hb_only_residues))

    # overlap_set = set(hb_only_residues) & set(all_contact_residues)
    # if len(overlap_set) > 0:
    #     print('*** WARNING: hb_only_residues and all_contact_residues overlap ***')
    #     print('*** Overlapping residues will be assigned as all_contact_residues ***')
    #     hb_only_residues = ''.join(set(hb_only_residues) - overlap_set)

    if type(hb_only_residues) == str and type(all_contact_residues) == str:
        hb_only_residues, all_contact_residues = check_string_overlap(hb_only_residues, 
                                                                      all_contact_residues, 
                                                                      cg_name='All CGs')
    elif type(hb_only_residues) == dict and type(all_contact_residues) == dict:
        for cg_name in hb_only_residues.keys():
            if cg_name in all_contact_residues.keys():
                hb_only_residues[cg_name], all_contact_residues[cg_name] = \
                    check_string_overlap(hb_only_residues[cg_name], 
                                        all_contact_residues[cg_name], 
                                        cg_name=cg_name)
    elif type(hb_only_residues) == dict and type(all_contact_residues) == str:
        all_contact_residues_str = all_contact_residues
        all_contact_residues = dict()
        for cg_name in hb_only_residues.keys():
            all_contact_residues[cg_name] = all_contact_residues_str
            hb_only_residues[cg_name], all_contact_residues[cg_name] = \
                check_string_overlap(hb_only_residues[cg_name], 
                                    all_contact_residues[cg_name], 
                                    cg_name=cg_name)
    elif type(hb_only_residues) == str and type(all_contact_residues) == dict:
        hb_only_residues_str = hb_only_residues
        hb_only_residues = dict()
        for cg_name in all_contact_residues.keys():
            hb_only_residues[cg_name] = hb_only_residues_str
            hb_only_residues[cg_name], all_contact_residues[cg_name] = \
                check_string_overlap(hb_only_residues[cg_name], 
                                    all_contact_residues[cg_name], 
                                    cg_name=cg_name)

    if resindices is None and segs_chains_resnums is None:
        ris = set(prody_pdb.select('within ' + str(dist_from_CoM) + ' of pt', pt=pr.calcCenter(prody_pdb.ca)).getResindices())
    elif resindices is not None and segs_chains_resnums is None:
        ris = resindices
    elif resindices is None and segs_chains_resnums is not None:
        ris = set([prody_pdb.select('segment ' + (seg if seg != '' else '_') + ' chain ' + ch + ' resnum ' + str(rn)).getResindices()[0]
                  for seg, ch, rn in segs_chains_resnums])

    ri_ex, ri_int, ri_bur = partition_res_by_burial(prody_pdb, alpha=alpha_hull_radius)

    if pikaa_dict is None and not use_propensities:
        pikaa_dict = dict()
        for cg in CGs:  #use a generic pikaa dict
            pikaa_dict[cg] = dict(bb_ind=dict(ex=dict(hb='KRDENQSTMAGP', all_contacts='HYWF'),
                                inter=dict(hb='NQSTCMAGPVIL', all_contacts='HYWF'),
                                bur=dict(hb='AGSTMCPVIL', all_contacts='HYWF')))
    else:
        if use_dssp:
            if 'dssp' not in template.dataframe.columns:
                template.set_dssp(path_to_pdb_for_dssp)

        if use_propensities:
            prop_dict = dict()
            for cg in CGs:
                prop_dict[cg] = get_aas_with_enriched_propensity(cg, use_abple=use_abple,
                                                                 use_dssp=use_dssp,
                                                                 propensity_threshold=propensity_threshold)

    top_dict = dict(ex=top_exposed, inter=top_intermed, bur=top_buried)
    resinds_dict = dict(ex=ri_ex, inter=ri_int, bur=ri_bur)
    if exclude_exposed:
        resinds_dict.pop('ex')
    if exclude_intermed:
        resinds_dict.pop('inter')
    if exclude_buried:
        resinds_dict.pop('bur')
    pdb_ala = prody_pdb

    # aas_dict_hb_vs_all = dict(hb=set(hb_only_residues), all_contacts=set(all_contact_residues))

    aas_dict_hb_vs_all = defaultdict(dict)
    if type(hb_only_residues) == str and type(all_contact_residues) == str:
        for cg in CGs:
            aas_dict_hb_vs_all[cg] = dict(hb=set(hb_only_residues), all_contacts=set(all_contact_residues))
    elif type(hb_only_residues) == dict and type(all_contact_residues) == dict:
        for cg in CGs:
            aas_dict_hb_vs_all[cg] = dict(hb=set(hb_only_residues[cg]), all_contacts=set(all_contact_residues[cg]))

    allowed_exposed_dict = dict()
    if type(allowed_exposed) == str:    
        for cg in CGs:
             allowed_exposed_dict[cg] = set(allowed_exposed)
    elif type(allowed_exposed) == dict:
        allowed_exposed_dict = allowed_exposed

    allowed_intermed_dict = dict()
    if type(allowed_intermed) == str:    
        for cg in CGs:
             allowed_intermed_dict[cg] = set(allowed_intermed)
    elif type(allowed_intermed) == dict:
        allowed_intermed_dict = allowed_intermed

    allowed_buried_dict = dict()
    if type(allowed_buried) == str:    
        for cg in CGs:
             allowed_buried_dict[cg] = set(allowed_buried)
    elif type(allowed_buried) == dict:
        allowed_buried_dict = allowed_buried

    with open(outpath + filename + tag + '.txt', 'w') as outfile:
        for key, resinds in resinds_dict.items():
            # if key == 'ex':
            #     allowed_aa_set = set(allowed_exposed)
            # elif key == 'inter':
            #     allowed_aa_set = set(allowed_intermed)
            # else:
            #     allowed_aa_set = set(allowed_buried)
            for ri in resinds:
                if ri not in ris:
                    continue
                if ri == pdb_ala.ca.getResindices().min():
                    continue
                if ri == pdb_ala.ca.getResindices().max():
                    continue
                sel = pdb_ala.ca.select('resindex ' + str(ri))
                seg = sel.getSegnames()[0]
                # if seg == '':
                #     seg = '_'
                chain = sel.getChids()[0]
                resnum = sel.getResnums()[0]

                set_rotamer = False
                if rotamers is not None:
                    resname = sel.getResnames()[0]
                    if rotamers == 'all':
                        set_rotamer = True
                    if (seg, chain, resnum) in rotamers:
                        set_rotamer = True

                if use_abple:
                    ss = template.dataframe[template.dataframe['seg_chain_resnum']
                                            == (seg, chain, resnum)]['ABPLE'].iat[0]
                elif use_dssp:
                    ss = template.dataframe[template.dataframe['seg_chain_resnum']
                                            == (seg, chain, resnum)]['dssp'].iat[0]

                else:
                    ss = 'bb_ind'

                for cg in CGs:
                    if CG_specific_residues is not None:
                        if cg in CG_specific_residues:
                            if (seg, chain, resnum) not in CG_specific_residues[cg]:
                                continue
                    if key == 'ex':
                        allowed_aa_set = set(allowed_exposed_dict[cg])
                    elif key == 'inter':
                        allowed_aa_set = set(allowed_intermed_dict[cg])
                    else:
                        allowed_aa_set = set(allowed_buried_dict[cg])
                    for type_ in ['hb', 'all_contacts']:
                        if set_rotamer:
                            pikaas = ''.join(set(one_letter_code[resname]) & aas_dict_hb_vs_all[cg][type_])
                        elif pikaa_dict is not None:
                            pikaas = pikaa_dict[cg][ss][key][type_]
                        else:
                            pikaas = ''.join(set(prop_dict[cg][ss]) & allowed_aa_set & aas_dict_hb_vs_all[cg][type_])
                        if len(pikaas) == 0:
                            continue
                        line_list = []
                        line_list.append(' '.join([str(resnum), chain, seg if seg != '' else '_']))
                        if type(pikaas) is dict:
                            line_list.append('PIKAA ' + pikaas[(seg, chain, resnum)])
                        else:
                            line_list.append('PIKAA ' + pikaas)
                        if pikaa_override is not None:
                            if (seg, chain, resnum) in pikaa_override:
                                 line_list[-1] = 'PIKAA ' + pikaa_override[(seg, chain, resnum)]
                        line_list.append('CG ' + cg)
                        line_list.append('bbdep ' + str(bb_dep))
                        if set_rotamer:
                            line_list.append('rotamer')
                        if use_dssp:
                            line_list.append('dssp')
                        if top_dict[key] is not None:
                            line_list.append('top ' + str(top_dict[key]))
                        if use_enriched_vdMs:
                            line_list.append('enriched')
                        if CA_burial_distance is not None:
                            line_list.append('CA_burial ' + str(CA_burial_distance))
                        if type_ == 'hb':
                            line_list.append('hbond_only')
                            if cg in cg_is_hb_acceptor:
                                line_list.append('cg_is_hb_acceptor')
                            elif cg in cg_is_hb_donor:
                                line_list.append('cg_is_hb_donor')
                        if cg in cg_is_not_hb_acceptor:
                            line_list.append('cg_is_not_hb_acceptor')
                        if cg in cg_is_not_hb_donor:
                            line_list.append('cg_is_not_hb_donor')
                        outfile.write(', '.join(line_list) + ' \n')


def find_buried_unsatisfied_hbonds(pdb, lig_resname=None, lig_can_hbond_dict=None,
                                   lig_atom_type_dict=None, append_to_file=False,
                                   outdir='./', ignore=set(), alpha=9):
    try:
        if isinstance(pdb, str):
            pdb_name = pdb.split('/')[-1].split('.')[0]
            pdb = parsePDB(pdb)
        else:
            pdb_name = str(pdb).split()[1]
    except:
        raise TypeError('*pdb* must be a pdb file or prody object')

    if set(pdb.getSegnames()) == {''}:
        pdb.setSegnames('A')

    ahull = AlphaHull(alpha=alpha)
    ahull.set_coords(pdb)
    ahull.calc_hull()

    if (lig_resname is not None) and (lig_can_hbond_dict is not None) \
            and (lig_atom_type_dict is not None):
        dflig = make_df_from_prody(pdb.select('resname ' + lig_resname),
                                       **{'lig_atom_types_dict': lig_atom_type_dict,
                                          'can_hbond_dict': lig_can_hbond_dict})
        columns_to_file = ['resnum_q', 'chain_q', 'segment_q', 'resname_q',
                           'name_q', 'resnum_t', 'chain_t', 'segment_t',
                           'resname_t', 'name_t', 'lig_resname', 'lig_name',
                           'contact_type']
    else:
        columns_to_file = ['resnum_q', 'chain_q', 'segment_q', 'resname_q',
                           'name_q', 'resnum_t', 'chain_t', 'segment_t',
                           'resname_t', 'name_t', 'contact_type']

    if append_to_file:
        status = 'a'
        pdb_filename = pdb_name + '.pdb'
    else:
        status = 'w'
        pdb_filename = 'buried_unsat_hb_' + pdb_name + '.txt'

    if outdir[-1] != '/':
        outdir += '/'

    with open(outdir + pdb_filename, status) as outfile:
        outfile.write('\n')
        num_bur_unsatisfied = 0
        num_bur_unsatisfied_ignore = 0
        df_pdb = make_df_from_prody(pdb.select('protein and sidechain'))
        df_pdb_polar = df_pdb[df_pdb.atom_type_label.isin({'h_pol', 'o', 'n'})]
        for n, row in df_pdb_polar.iterrows():
            if (row.atom_type_label == 'n') and ~np.isnan(row.c_D_x):
                continue
            seg, ch, rn = row.seg_chain_resnum
            q_sel = pdb.select('resnum ' + str(rn) + ' chain ' + ch + ' segment ' + seg + ' and sidechain')
            if q_sel is None:
                continue
            if q_sel.select('element O N') is None:
                continue
            name = row['name']
            q_sel_atom = q_sel.select('name ' + name)
            is_buried = ahull.get_pnt_distance(q_sel_atom.getCoords().flatten()) > 1
            if is_buried:
                dfq = make_df_from_prody(q_sel)
                t_sel = pdb.select(
                    'protein and not (resnum ' + str(rn) + ' chain ' + ch + ' segment ' + seg + ' and sidechain)')
                dft = make_df_from_prody(t_sel)
                if lig_resname is not None:
                    dft = concat((dft, dflig), sort=False)
                con = Contact(dfq=dfq, dft=dft)
                con.find()
                isON = con.df_contacts.name_q == name
                df = con.df_contacts[isON]
                if not any(df.contact_type == 'hb'):
                    outfile.write('(' + seg + ', ' + ch + ', ' + str(rn) + ', ' + name + '), ' + df[columns_to_file].to_string() + ' \n')
                    num_bur_unsatisfied += 1
                    if (seg, ch, rn, name) not in ignore:
                        num_bur_unsatisfied_ignore += 1
        outfile.write('\n')
        outfile.write('Total_buried_unsatisfied_hbonds: ' + str(num_bur_unsatisfied) + ' \n')
        outfile.write('Total_buried_unsatisfied_hbonds_ignore: ' + str(num_bur_unsatisfied_ignore) + ' \n')


def listdir_mac(path):
    return [f for f in os.listdir(path) if f[0] != '.']


def writePDBStream(stream, atoms, csets=None, **kwargs):
    """Write *atoms* in PDB format to a *stream*.

    Needs selection with bfactors to be printed as atom indices.

    :arg stream: anything that implements a :meth:`write` method (e.g. file,
        buffer, stdout)"""

    # remark = str(atoms)
    PDBLINE = ('{0:6s}{1:5d} {2:4s}{3:1s}'
               '{4:4s}{5:1s}{6:4d}{7:1s}   '
               '{8:8.3f}{9:8.3f}{10:8.3f}'
               '{11:6.2f}{12:6.2f}      '
               '{13:4s}{14:2s}\n')

    PDBLINE_LT100K = ('%-6s%5d %-4s%1s%-4s%1s%4d%1s   '
                      '%8.3f%8.3f%8.3f%6.2f%6.2f      '
                      '%4s%2s\n')

    PDBLINE_GE100K = ('%-6s%5x %-4s%1s%-4s%1s%4d%1s   '
                      '%8.3f%8.3f%8.3f%6.2f%6.2f      '
                      '%4s%2s\n')

    try:
        coordsets = atoms._getCoordsets(csets)
    except AttributeError:
        try:
            coordsets = atoms._getCoords()
        except AttributeError:
            raise TypeError('atoms must be an object with coordinate sets')
        if coordsets is not None:
            coordsets = [coordsets]
    else:
        if coordsets.ndim == 2:
            coordsets = [coordsets]
    if coordsets is None:
        raise ValueError('atoms does not have any coordinate sets')

    try:
        acsi = atoms.getACSIndex()
    except AttributeError:
        try:
            atoms = atoms.getAtoms()
        except AttributeError:
            raise TypeError('atoms must be an Atomic instance or an object '
                            'with `getAtoms` method')
        else:
            if atoms is None:
                raise ValueError('atoms is not associated with an Atomic '
                                 'instance')
            try:
                acsi = atoms.getACSIndex()
            except AttributeError:
                raise TypeError('atoms does not have a valid type')

    try:
        atoms.getIndex()
    except AttributeError:
        pass
    else:
        atoms = atoms.select('all')

    n_atoms = atoms.numAtoms()

    # indices = atoms._getIndices()

    occupancy = kwargs.get('occupancy')
    if occupancy is None:
        occupancies = atoms._getOccupancies()
        if occupancies is None:
            occupancies = np.zeros(n_atoms, float)
    else:
        occupancies = np.array(occupancy)
        if len(occupancies) != n_atoms:
            raise ValueError('len(occupancy) must be equal to number of atoms')

    beta = kwargs.get('beta')
    if beta is None:
        bfactors = atoms._getBetas()
        if bfactors is None:
            bfactors = np.zeros(n_atoms, float)
    else:
        bfactors = np.array(beta)
        if len(bfactors) != n_atoms:
            raise ValueError('len(beta) must be equal to number of atoms')

    atomnames = atoms.getNames()
    if atomnames is None:
        raise ValueError('atom names are not set')
    for i, an in enumerate(atomnames):
        if len(an) < 4:
            atomnames[i] = ' ' + an

    s_or_u = np.array(['a']).dtype.char

    altlocs = atoms._getAltlocs()
    if altlocs is None:
        altlocs = np.zeros(n_atoms, s_or_u + '1')

    resnames = atoms._getResnames()
    if resnames is None:
        resnames = ['UNK'] * n_atoms

    chainids = atoms._getChids()
    if chainids is None:
        chainids = np.zeros(n_atoms, s_or_u + '1')

    resnums = atoms._getResnums()
    if resnums is None:
        resnums = np.ones(n_atoms, int)

    icodes = atoms._getIcodes()
    if icodes is None:
        icodes = np.zeros(n_atoms, s_or_u + '1')

    hetero = ['ATOM'] * n_atoms
    heteroflags = atoms._getFlags('hetatm')
    if heteroflags is None:
        heteroflags = atoms._getFlags('hetero')
    if heteroflags is not None:
        hetero = np.array(hetero, s_or_u + '6')
        hetero[heteroflags] = 'HETATM'

    elements = atoms._getElements()
    if elements is None:
        elements = np.zeros(n_atoms, s_or_u + '1')
    else:
        elements = np.char.rjust(elements, 2)

    segments = atoms._getSegnames()
    if segments is None:
        segments = np.zeros(n_atoms, s_or_u + '6')

    # stream.write('REMARK {0}\n'.format(remark))

    multi = len(coordsets) > 1
    write = stream.write
    for m, coords in enumerate(coordsets):
        pdbline = PDBLINE_LT100K
        if multi:
            write('MODEL{0:9d}\n'.format(m + 1))
        for i, xyz in enumerate(coords):
            if i == 99999:
                pdbline = PDBLINE_GE100K
            write(pdbline % (hetero[i], bfactors[i],
                             atomnames[i], altlocs[i],
                             resnames[i], chainids[i], resnums[i],
                             icodes[i],
                             xyz[0], xyz[1], xyz[2],
                             occupancies[i], 0,
                             segments[i], elements[i]))
        if multi:
            write('ENDMDL\n')
            altlocs = np.zeros(n_atoms, s_or_u + '1')


def make_pdb_wholebb_noligand(poi, outdir, vdm):
    """prints pdbs of the top number of hotspots (number) to the output directory (outdir)."""
    if outdir[-1] != '/':
        outdir += '/'
    poi = poi.copy()
    # resnum_chid = tuple(self._all_resnum_chid[mem])
    # type_ = self._all_type[mem]
    # typestr = type_
    # resn = self._all_resn[mem]
    # vdm_tags = self._all_vdm_tags[mem]

    sc = vdm.select('not element H and chain X and resnum 10 and sidechain')
    resnum = str(vdm).split()[-1].split('_')[3][:-1]
    chid = str(vdm).split()[-1].split('_')[3][-1]
    resnum_chid = [int(resnum), chid]
    resn = vdm.select('chain X and resnum 10 and name CA').getResnames()[0]

    bb_first_resnum = np.min(poi.select('backbone').getResnums())
    bb_last_resnum = np.max(poi.select('backbone').getResnums())


    bb1 = poi.select('not element H and resnum ' + str(bb_first_resnum) + 'to' + str(resnum_chid[0]))
    num_ind = len(bb1.getIndices())
    start = 1
    finish = start + num_ind
    bb1.setBetas(list(range(start, finish)))
    poi.select('resnum ' + str(resnum_chid[0])).setResnames(resn)
    # bb1.setResnames('GLY')

    # if typestr == 'PHI_PSI':
    #     typestr = 'PHI_PSI/' + self.rel_vdm_phipsi_bin[resnum_chid]
    # pdbpath = self.rel_vdm_path + typestr + '/pdbs/' + resn + '/'
    # filename = 'iFG_' + str(vdm_tags[0]) + '_vdM_' + str(vdm_tags[1]) + '_iFlip_' \
    #            + str(vdm_tags[2]) + '_' + self.name + '_' + 'oriented.pdb.gz'
    # pdb = pr.parsePDB(pdbpath + filename)
    # old_coords = pdb.getCoords()
    # new_coords = \
    #     np.dot((old_coords - self._rois_rot_trans[resnum_chid][type_][resn][1]),
    #            self._rois_rot_trans[resnum_chid][type_][resn][0]) \
    #     + self._rois_rot_trans[resnum_chid][type_][resn][2]
    # pdb.setCoords(new_coords)
    # newfile_path = outdir
    # newfile_name = str(counter) + '_mem_' + str(mem) + '_' + self.name + '_' + label + '_' \
    #                + ''.join(str(x) for x in resnum_chid) + '_' + type_ + '_' + filename[:-3]
    #
    # sc = pdb.select('sidechain and chain X and resnum 10')

    sc.setResnums(str(resnum_chid[0]))
    sc.setChids(str(resnum_chid[1]))
    num_ind = len(sc.getIndices())
    start = finish
    finish = start + num_ind
    sc.setBetas(list(range(start, finish)))

    bb2 = poi.select('not element H and resnum ' + str(resnum_chid[0] + 1) + 'to' + str(bb_last_resnum))
    num_ind = len(bb2.getIndices())
    start = finish
    finish = start + num_ind
    bb2.setBetas(list(range(start, finish)))

    try:
        os.makedirs(outdir)
    except:
        pass
    with open(outdir + 'pose_with_bb_template.pdb', 'w') as outfile:
        writePDBStream(outfile, bb1)
        writePDBStream(outfile, sc)
        writePDBStream(outfile, bb2)


def make_pdb_wholebb_ligand(poi, outdir, vdm, ligand):
    """prints pdbs of the top number of hotspots (number) to the output directory (outdir)."""
    if outdir[-1] != '/':
        outdir += '/'
    poi = poi.copy()
    # resnum_chid = tuple(self._all_resnum_chid[mem])
    # type_ = self._all_type[mem]
    # typestr = type_
    # resn = self._all_resn[mem]
    # vdm_tags = self._all_vdm_tags[mem]

    sc = vdm.select('not element H and chain X and resnum 10 and sidechain')
    resnum = str(vdm).split()[-1].split('_')[3][:-1]
    chid = str(vdm).split()[-1].split('_')[3][-1]
    resnum_chid = [int(resnum), chid]
    resn = vdm.select('chain X and resnum 10 and name CA').getResnames()[0]

    bb_first_resnum = np.min(poi.select('backbone').getResnums())
    bb_last_resnum = np.max(poi.select('backbone').getResnums())


    bb1 = poi.select('not element H and resnum ' + str(bb_first_resnum) + 'to' + str(resnum_chid[0]))
    num_ind = len(bb1.getIndices())
    start = 1
    finish = start + num_ind
    bb1.setBetas(list(range(start, finish)))
    poi.select('resnum ' + str(resnum_chid[0])).setResnames(resn)
    # bb1.setResnames('GLY')

    # if typestr == 'PHI_PSI':
    #     typestr = 'PHI_PSI/' + self.rel_vdm_phipsi_bin[resnum_chid]
    # pdbpath = self.rel_vdm_path + typestr + '/pdbs/' + resn + '/'
    # filename = 'iFG_' + str(vdm_tags[0]) + '_vdM_' + str(vdm_tags[1]) + '_iFlip_' \
    #            + str(vdm_tags[2]) + '_' + self.name + '_' + 'oriented.pdb.gz'
    # pdb = pr.parsePDB(pdbpath + filename)
    # old_coords = pdb.getCoords()
    # new_coords = \
    #     np.dot((old_coords - self._rois_rot_trans[resnum_chid][type_][resn][1]),
    #            self._rois_rot_trans[resnum_chid][type_][resn][0]) \
    #     + self._rois_rot_trans[resnum_chid][type_][resn][2]
    # pdb.setCoords(new_coords)
    # newfile_path = outdir
    # newfile_name = str(counter) + '_mem_' + str(mem) + '_' + self.name + '_' + label + '_' \
    #                + ''.join(str(x) for x in resnum_chid) + '_' + type_ + '_' + filename[:-3]
    #
    # sc = pdb.select('sidechain and chain X and resnum 10')
    sc.setResnums(str(resnum_chid[0]))
    sc.setChids(str(resnum_chid[1]))
    num_ind = len(sc.getIndices())
    start = finish
    finish = start + num_ind
    sc.setBetas(list(range(start, finish)))


    bb2 = poi.select('not element H and resnum ' + str(resnum_chid[0] + 1) + 'to' + str(bb_last_resnum))
    num_ind = len(bb2.getIndices())
    start = finish
    finish = start + num_ind
    bb2.setBetas(list(range(start, finish)))

    num_ind = len(ligand.select('all').getIndices())
    start = finish
    finish = start + num_ind
    ligand.setBetas(list(range(start, finish)))

    try:
        os.makedirs(outdir)
    except:
        pass
    with open(outdir + 'pose_with_bb_template.pdb', 'w') as outfile:
        writePDBStream(outfile, bb1)
        writePDBStream(outfile, sc)
        writePDBStream(outfile, bb2)
        writePDBStream(outfile, ligand)



def make_protein_from_dee_pose(indir, outdir, path_to_bb_template):
    if indir[-1] != '/':
        indir = indir + '/'
    if outdir[-1] != '/':
        outdir = outdir + '/'
    bb_template = pr.parsePDB(path_to_bb_template)
    ligand = pr.parsePDB(indir + 'lig.pdb')
    num_fns = len([fn for fn in os.listdir(indir) if fn[-2:] == 'gz'])
    for i, f in enumerate([fn for fn in os.listdir(indir) if fn[-2:] == 'gz']):
        vdm = pr.parsePDB(indir + f)
        if i + 1 < num_fns:
            if i == 0:
                make_pdb_wholebb_noligand(bb_template, outdir, vdm)
            else:
                bb_template = pr.parsePDB(outdir + 'pose_with_bb_template.pdb')
                make_pdb_wholebb_noligand(bb_template, outdir, vdm)
        if i + 1 == num_fns:
            bb_template = pr.parsePDB(outdir + 'pose_with_bb_template.pdb')
            make_pdb_wholebb_ligand(bb_template, outdir, vdm, ligand)


def make_resfile(path_to_pdb, outdir):
    if outdir[-1] != '/':
        outdir = outdir + '/'
    pdb = pr.parsePDB(path_to_pdb)
    sel = pdb.select('name CA and not resname GLY')
    resnums = sel.getResnums()
    chains = sel.getChids()

    with open(outdir + 'resfile.txt', 'w') as resfile:
        resfile.write('ALLAAxc \n')
        resfile.write('USE_INPUT_SC \n')
        resfile.write('EX 1 LEVEL 1 \n')
        resfile.write('start \n')
        resfile.write('\n')
        for resnum, chid in zip(resnums, chains):
            resfile.write(str(resnum) + ' ' + chid + ' NATRO \n')
        resfile.write('1 X NATRO \n')


@jit(nopython=True)
def get_mat(seqs):
    rmsd_mat = np.zeros((len(seqs), len(seqs)), dtype=np.float32)
    l = len(seqs[0])
    for i in np.arange(len(seqs) - 1):
        si = seqs[i]
        for j in np.arange(i + 1, len(seqs)):
            sj = seqs[j]
            p = 1 - ((si == sj).sum() / l)
            rmsd_mat[i,j] = p
    return rmsd_mat


def get_nr_reps(clu_seq, probe_paths, groups, path_to_probe_paths):
    if path_to_probe_paths[-1] != '/':
        path_to_probe_paths += '/'
    nr_reps = []
    for j in range(len(clu_seq.mems)):
        inds = []
        for rot, cg, pn in [groups[i] for i in clu_seq.mems[j]]:
            inds.append((probe_paths.index(path_to_probe_paths + pn + '.pkl'), rot, cg, pn))
        ind = sorted(inds)[0]
        nr_reps.append(ind[1:])
    return nr_reps


def add_slash(string):
    if string[-1] != '/':
        return string + '/'
    else:
        return string


# def flip_cg_coords(g):
#     gc = g.copy()
#     cg_rn = gc[gc.chain == 'Y'].resname.iat[0]
#     for name1, name2 in flip_dict[cg_rn].items():
#         sele = (gc.chain == 'Y') & (gc.name.isin([name1, name2]))
#         cs = gc[sele][['c_x', 'c_y', 'c_z']].values
#         if len(cs) != 2:
#             raise Exception('flip_cg_coords group contains duplicates?')
#         cs[[0, 1]] = cs[[1, 0]]
#         gc.loc[sele, ['c_x', 'c_y', 'c_z']] = cs
#     return gc[['c_x', 'c_y', 'c_z']].values.astype('float32')


def flip_coords_from_reference_df(coords, resn, df_names):
    for name1, name2 in flip_dict[resn].items():
        sele = df_names['name'].isin([name1, name2])
        cs = coords[sele]
        cs[[0, 1]] = cs[[1, 0]]
        coords[sele] = cs
    return coords


def flip_cg_coords(gc):
    coords = gc[['c_x', 'c_y', 'c_z']].values.astype('float32')
    sel_y = gc.chain == 'Y'
    cg_rn = gc[sel_y].resname.iat[0]
    for name1, name2 in flip_dict[cg_rn].items():
        sele = sel_y & (gc.name.isin([name1, name2]))
        cs = coords[sele]
        if len(cs) != 2:
            raise Exception('flip_cg_coords group contains duplicates?')
        cs[[0, 1]] = cs[[1, 0]]
        coords[sele] = cs
    return coords


def flip_x_coords(g):
    gc = g.copy()
    x_rn = gc[gc.chain == 'X'].resname.iat[0]
    for name1, name2 in flip_dict[x_rn].items():
        sele = (gc.chain == 'X') & (gc.name.isin([name1, name2]))
        cs = gc[sele][['c_x', 'c_y', 'c_z']].values
        cs[[0, 1]] = cs[[1, 0]]
        gc.loc[sele, ['c_x', 'c_y', 'c_z']] = cs
    return gc[['c_x', 'c_y', 'c_z']].values.astype('float32')


def flip_cg_x_coords(g):
    gc = g.copy()
    cg_rn = gc[gc.chain == 'Y'].resname.iat[0]
    for name1, name2 in flip_dict[cg_rn].items():
        sele = (gc.chain == 'Y') & (gc.name.isin([name1, name2]))
        cs = gc[sele][['c_x', 'c_y', 'c_z']].values
        cs[[0, 1]] = cs[[1, 0]]
        gc.loc[sele, ['c_x', 'c_y', 'c_z']] = cs

    x_rn = gc[gc.chain == 'X'].resname.iat[0]
    for name1, name2 in flip_dict[x_rn].items():
        sele = (gc.chain == 'X') & (gc.name.isin([name1, name2]))
        cs = gc[sele][['c_x', 'c_y', 'c_z']].values
        cs[[0, 1]] = cs[[1, 0]]
        gc.loc[sele, ['c_x', 'c_y', 'c_z']] = cs
    return gc[['c_x', 'c_y', 'c_z']].values.astype('float32')


def write_cg_dicts_txt_file(cg_dicts):
    with open(path_to_files_folder + 'cg_dicts.txt', 'w') as outfile:
        for cg_name in cg_dicts.keys():
            outfile.write(cg_name + ' \n')
            for resn, atoms in cg_dicts[cg_name].items():
                outfile.write('\t ' + resn + ' \t' + ', '.join(atoms) + ' \n')


def _make_cg_propensities(cg, path_to_db):

    with open(path_to_files_folder + 'aa_bkgrd_frequencies_abple.pkl', 'rb') as outfile:
        freq_res_dict_abple = pickle.load(outfile)
    with open(path_to_files_folder + 'aa_bkgrd_frequencies_dssp.pkl', 'rb') as outfile:
        freq_res_dict_dssp = pickle.load(outfile)
    with open(path_to_files_folder + 'aa_bkgrd_frequencies.pkl', 'rb') as outfile:
        freq_res_dict_bb_ind = pickle.load(outfile)

    path_to_db = add_slash(path_to_db)
    d = path_to_db + cg + '/'

    pdict_abple = defaultdict(dict)
    pdict_dssp = defaultdict(dict)
    pdict_bb_ind = defaultdict(int)
    for f in [f for f in os.listdir(d) if f[0] != '.']:
        df = read_parquet(d + f)
        df = df[(df['chain'] == 'X') & (df['resnum'] == 10)]
        for abple in set('ABPLEn'):
            pdict_abple[abple][f.split('.')[0]] = len(
                df[['CG', 'rota', 'probe_name']][df['ABPLE'] == abple].drop_duplicates())
        for dssp in set(df['dssp']):
            pdict_dssp[dssp][f.split('.')[0]] = len(
                df[['CG', 'rota', 'probe_name']][df['dssp'] == dssp].drop_duplicates())
    for abple in pdict_abple.keys():
        for resname in pdict_abple[abple].keys():
            pdict_bb_ind[resname] += pdict_abple[abple][resname]

    # bb independent props
    total_ = sum(pdict_bb_ind.values())
    freq_pdict_bb_ind = {aa: v / total_ if total_ != 0 else 0 for aa, v in pdict_bb_ind.items()}

    aa_props_bb_ind = dict()
    for res in freq_pdict_bb_ind.keys():
        try:
            aa_props_bb_ind[res] = freq_pdict_bb_ind[res] / freq_res_dict_bb_ind[res]
        except:
            aa_props_bb_ind[res] = 0

    with open(path_to_propensities_folder + 'aa_propensities_bb_ind_' + cg + '.pkl',
              'wb') as outfile:
        pickle.dump(aa_props_bb_ind, outfile)

    # abple props
    freq_pdict_abple = pdict_abple.copy()
    for abple in pdict_abple.keys():
        total_ = sum(pdict_abple[abple].values())
        aa_freq = {aa: v / total_ if total_ != 0 else 0 for aa, v in pdict_abple[abple].items()}
        freq_pdict_abple[abple] = aa_freq

    aa_props_abple = defaultdict(dict)
    for abple in freq_pdict_abple.keys():
        for res in freq_pdict_abple[abple].keys():
            try:
                aa_props_abple[abple][res] = freq_pdict_abple[abple][res] / freq_res_dict_abple[abple][res]
            except:
                aa_props_abple[abple][res] = 0

    with open(path_to_propensities_folder + 'aa_propensities_abple_' + cg + '.pkl',
              'wb') as outfile:
        pickle.dump(aa_props_abple, outfile)

    # dssp props
    freq_pdict_dssp = pdict_dssp.copy()
    for dssp in pdict_dssp.keys():
        total_ = sum(pdict_dssp[dssp].values())
        aa_freq = {aa: v / total_ if total_ != 0 else 0 for aa, v in pdict_dssp[dssp].items()}
        freq_pdict_dssp[dssp] = aa_freq

    aa_props_dssp = defaultdict(dict)
    for dssp in freq_pdict_dssp.keys():
        for res in freq_pdict_dssp[dssp].keys():
            try:
                aa_props_dssp[dssp][res] = freq_pdict_dssp[dssp][res] / freq_res_dict_dssp[dssp][res]
            except:
                aa_props_dssp[dssp][res] = 0

    with open(path_to_propensities_folder + 'aa_propensities_dssp_' + cg + '.pkl',
              'wb') as outfile:
        pickle.dump(aa_props_dssp, outfile)


def make_cg_propensities(path_to_db):
    for cg in cg_dicts.keys():
        _make_cg_propensities(cg, path_to_db)

## Not good if frames have different columns, falling back on pd.concat for now (below)
## There is also a dtype error that I don't understand.
# def fast_concat(frames):
#     if len(frames) == 0:
#         return DataFrame()
#     columns = frames[0].columns
#     dtypes = {col: dt.name for col, dt in frames[0].dtypes.iteritems()}
#     stacked_vals = np.vstack([frame[columns] for frame in frames if len(frame) != 0])
#     return DataFrame(stacked_vals, columns=columns).astype(dtypes)


def fast_concat(frames):
    if len(frames) == 0:
        return DataFrame()
    return concat(frames)


# def fast_concat(frames, keep_index=False, keep_dtypes=True, sort_columns=True):
#     columns = frames[0].columns
#     dtypes = frames[0].dtypes
#     if sort_columns:
#         stacked_vals = np.vstack([frame[columns] for frame in frames if len(frame) != 0])
#     else:
#         stacked_vals = np.vstack([frame for frame in frames if len(frame) != 0])
#     if keep_index:
#         index_names = frames[0].index.names
#         stacked_inds = list(chain(*[frame.index for frame in frames if len(frame) != 0]))
#         if len(index_names) == 1:
#             index = Index(stacked_inds, name=index_names)
#         else:
#             index = MultiIndex.from_tuples(stacked_inds, names=index_names)
#         if keep_dtypes:
#             df = DataFrame(stacked_vals, columns=columns, index=index).astype(dtypes)
#         else:
#             df = DataFrame(stacked_vals, columns=columns, index=index)
#         df.index.names = index_names
#     else:
#         if keep_dtypes:
#             df = DataFrame(stacked_vals, columns=columns).astype(dtypes)
#         else:
#             df = DataFrame(stacked_vals, columns=columns)
#     return df


# def fast_flatten(input_list):
#     return list(chain.from_iterable(input_list))
#
#
# def fast_concat(frames):
#     if len(frames) == 0:
#         return DataFrame()
#
#     COLUMN_NAMES = frames[0].columns
#     dtypes = frames[0].dtypes  # slows fn down by factor of 3
#     #Now, construct a dictionary from the column names:
#
#     df_dict = dict.fromkeys(COLUMN_NAMES, [])
#     #Iterate though the columns:
#
#     for col in COLUMN_NAMES:
#         extracted = (frame[col] for frame in frames if len(frame) != 0) # (frame[col] for frame in frames)
#
#         # Flatten and save to df_dict
#         df_dict[col] = fast_flatten(extracted)
#
#     #Lastly use the from_dict method to produce the combined DataFrame:
#
#     return DataFrame.from_dict(df_dict)[COLUMN_NAMES].astype(dtypes)


# def fast_flatten(input_list, N):
#     a = list(chain.from_iterable(input_list))
#     a += [False] * (N - len(a)) # collating logical arrays - missing values are replaced with False
#     return list(a)
#
#
# def fast_concat(frames, reduced_memory=False):
#     COLUMN_NAMES = [frames[i].columns for i in range(len(frames))]
#     N = sum([len(frames[i]) for i in range(len(frames))])
#     COL_NAMES=list(set(list(chain(*COLUMN_NAMES))))
#     df_dict = dict.fromkeys(COL_NAMES, [])
#     for col in COL_NAMES:
#         extracted = (frame[col] for frame in frames if (len(frame) != 0 and col in frame.columns))
#         df_dict[col] = fast_flatten(extracted, N)
#         if reduced_memory:
#             [frame.drop(columns=col, inplace=True) for frame in frames
#              if (len(frame) != 0 and col in frame.columns)]
#     return DataFrame.from_dict(df_dict)[COL_NAMES]


def fast_merge(df_big, df_small, columns=None):
    if columns is None:
        columns = df_small.columns
    return df_big[df_big.set_index(columns.to_list()).index.isin(df_small.set_index(columns.to_list()).index)]


def outer_merge(df_big, df_small, columns=None):
    if columns is None:
        columns = df_small.columns
    return df_big[~df_big.set_index(columns.to_list()).index.isin(df_small.set_index(columns.to_list()).index)]


def atom_types_sort(val):
    try:
        return atom_types_sortkey.index(val)
    except:
        return len(atom_types_sortkey)


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def df_is_subset(df1, df2):
    """
    Example:
    >>> df1 = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df2 = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df_is_subset(df1, df2)
    True
    >>> df2 = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    >>> df_is_subset(df1, df2)
    True
    >>> df2 = DataFrame({'a': [1, 2, 3], 'b': [5, 6, 7]})
    >>> df_is_subset(df1, df2)
    False

    parameters
    ----------
    df1: pandas.DataFrame
    df2: pandas.DataFrame

    returns
    -------
    bool: True if df1 is a subset of df2, False otherwise
    """

    if df1.shape[0] > df2.shape[0]:
        return False
    tfs = []
    for col in df1.columns:
        tfs.append(np.in1d(df2[col], df1[col]))
    return np.all(tfs)


def make_empty_df(columns):
    return DataFrame(columns=columns)


# @jit(parallel=True, nopython=True)
# def isin(b, s):
#     o = np.zeros(b.shape[0])
#     r = s.shape[0]
#     q = s.shape[1]
#     for i in prange(r):
#         t = np.ones(b.shape[0]).astype(np.bool_)
#         for j in range(q):
#             t = t & (b[:, j] == s[i, j])
#         o[t] = 1
#     return o.astype(np.bool_)
#
#
# def fast_isin(b, s):
#     d = dict()
#     o = np.zeros(b.shape[0])
#     r = s.shape[0]
#     q = s.shape[1]
#     for i in range(r):
#         t = np.ones(b.shape[0]).astype(np.bool_)
#         for j in range(q):
#             arr = (j, s[i, j])
#             if arr in d:
#                 t = t & d[arr]
#             else:
#                 d[arr] = (b[:, j] == s[i, j])
#                 t = t & d[arr]
#         o[t] = 1
#     return o.astype(np.bool_)
#
# def isin(b, s):
#     o = np.zeros(b.shape[0])
#     r = s.shape[0]
#     for i in range(r):
#         o[(b == s[i, :]).all(axis=1)] = 1
#     return o.astype(np.bool_)
#
# def fast_merge(df_big, df_small, columns=None):
#     if columns is None:
#         columns = df_small.columns.to_list()
#     big_arr = df_big[columns].values
#     small_arr = df_small[columns].values
#     return df_big[isin(big_arr, small_arr)]
#
# def fast_merge_hash(df1, df2):
#     return df1[np.in1d(df1['hash_id'].values, df2['hash_id'].drop_duplicates().values)]

    # global_inds = self.dfq_atom_type[atom_type_q]['num_tag'].values
    # all_qs = np.array(list(chain(*q_inds)))
    # _ts = []
    # _qs = []
    # _ds = []
    # mask = ~np.in1d(global_inds[all_qs], list(self.clash_indices))
    

def make_cg_ligand_txt():                                                    
    with open(path_to_files_folder + 'cg_ligand.txt', 'w') as outfile: 
        for cg in cgs: 
            outfile.write('\n') 
            for aa in cg_dicts[cg].keys(): 
                for atom in cg_dicts[cg][aa]: 
                    skip_H = True 
                    if cg in ['bb_cnh', 'coh', 'csh']: 
                        if atom in ['HG', 'HG1', 'H']: 
                            skip_H = False 
                    if skip_H and atom[0] == 'H': 
                        continue 
                    outfile.write(' '.join((aa, atom, aa, atom, cg, '1', '1', '\n'))) 


def make_group_indices(vdm_path, indices_path):
    if vdm_path[-1] != '/':
        vdm_path += '/'
    if indices_path[-1] != '/':
        indices_path += '/'
    for cg in os.listdir(vdm_path):
        if cg[0] == '.':
            continue
        indices_path_cg = indices_path + cg + '/'
        os.makedirs(indices_path_cg, exist_ok=True)
        vdm_path_cg = vdm_path + cg + '/'
        for f in os.listdir(vdm_path_cg):
            if f[0] == '.':
                continue
            df = read_parquet(vdm_path_cg + f)
            grs_ = df.groupby(['CG', 'rota', 'probe_name'])
            with open(indices_path_cg + f.split('.')[0] + '.pkl', 'wb') as f:
                pickle.dump(grs_.indices, f)


def make_cg_dict_coord_sort(cg_dict, cg_H_name_dict=None):
    dict_coord_sort = []
    for cg_resname in cg_dict.keys():
        if cg_H_name_dict is None:
            cg_names = [n for n in cg_dict[cg_resname] if n[0] != 'H']
        else:
            cg_names = [n for n in cg_dict[cg_resname] if (n in cg_H_name_dict[cg_resname] or n[0] != 'H')]
        dfy = DataFrame(dict(name=cg_names))
        dfy['resnum'] = 10
        dfy['chain'] = 'Y'
        dfy['resname'] = cg_resname
        dict_coord_sort.append(dfy)
    return concat(dict_coord_sort)


def make_cg_dataframes():
    cg_dfs = dict()
    for cg in cg_dicts.keys():
        if cg == 'coh':
            cg_H_name_dict = {'SER': ['HG'], 'THR': ['HG1']}
        elif cg == 'csh':
            cg_H_name_dict = {'CYS': ['HG']}
        elif cg == 'bb_cnh':
            cg_H_name_dict = {'GLY': ['H'], 'ALA': ['H'], 'LYS': ['H']}
        else:
            cg_H_name_dict=None
        cg_dict = make_cg_dict_coord_sort(cg_dicts[cg], cg_H_name_dict=cg_H_name_dict)
        cg_dfs[cg] = cg_dict
    with open(path_to_files_folder + 'cg_dataframes.pkl', 'wb') as f:
        pickle.dump(cg_dfs, f)


# vdm_path = '/Volumes/disk1/Combs2/database/20211005/vdMs/'
# nbrs_path_rmsd = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_rmsd/' + cg + '/'
# nbrs_path_maxdist = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_maxdist/' + cg + '/'
# nbrs_path_groupnames = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_groupnames/' + cg + '/'
# nbrs_path_scores = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_scores/' + cg + '/'
def make_cg_nbrs(vdm_path, cg_dfs, path_rmsd, path_maxdist, path_groupnames, path_scores):
    if vdm_path[-1] != '/':
        vdm_path += '/'
    if path_rmsd[-1] != '/':
        path_rmsd += '/'
    if path_maxdist[-1] != '/':
        path_maxdist += '/'
    if path_groupnames[-1] != '/':
        path_groupnames += '/'
    score_cols = None
    for cg in os.listdir(vdm_path):
        if cg[0] == '.':
            continue
        print(cg)
        nbrs_path_rmsd = path_rmsd + cg + '/'
        nbrs_path_maxdist = path_maxdist + cg + '/'
        nbrs_path_groupnames = path_groupnames + cg + '/'
        nbrs_path_scores = path_scores + cg + '/'
        cg_df = cg_dfs[cg]
        os.makedirs(nbrs_path_rmsd, exist_ok=True)
        os.makedirs(nbrs_path_maxdist, exist_ok=True)
        os.makedirs(nbrs_path_groupnames, exist_ok=True)
        os.makedirs(nbrs_path_scores, exist_ok=True)
        vdm_path_cg = vdm_path + cg + '/'
        for f in os.listdir(vdm_path_cg):
            if f[0] == '.':
                continue
            print(f)
            df = read_parquet(vdm_path_cg + f)
            if score_cols is None:
                score_cols = [c for c in df.columns if 'C_score' in c]
                score_cols.extend(['rotamer', 'phi', 'psi', 'contact_type', 'ABPLE', 'ABPLE_3mer'])

            for c in score_cols:
                if c not in df.columns:
                    df[c] = np.nan

            dfx10ca = df[(df.chain=='X') & (df.resnum==10) & (df['name']=='CA')]
            grs_dfx10ca = dfx10ca.groupby(['CG', 'rota', 'probe_name'])

            dfy_all = df[(df.chain=='Y') & (df.resnum==10)]

            dfy = merge(cg_df, dfy_all, on=['name', 'resnum', 'chain', 'resname'])
            grs = dfy.groupby(['CG', 'rota', 'probe_name'])
            grs_all = dfy_all.groupby(['CG', 'rota', 'probe_name'])

            coords, gr_names, scores, ishbond, isacceptor, isdonor = [], [], [], [], [], []
            for n, g in grs:
                coords.append(g[['c_x', 'c_y', 'c_z']].values)
                gr_names.append(n)
                grx = grs_dfx10ca.get_group(n)
                scores.append(grx[score_cols])
                g_for_hb = grs_all.get_group(n)
                _ishb = (g_for_hb['contact_hb'] == True).any()
                ishbond.append(_ishb)
                if _ishb:
                    ghb = g_for_hb[g_for_hb['contact_hb'] == True].copy()
                    startswithH = ghb['partners_hb'].str.startswith('H')
                    _isacc = startswithH.any()
                    if _isacc:
                        isacceptor.append(True)
                    else:
                        isacceptor.append(False)
                    _isdon = (~startswithH).any()
                    if _isdon:
                        isdonor.append(True)
                    else:
                        isdonor.append(False)
                else:
                    isacceptor.append(False)
                    isdonor.append(False)

                coords, gr_names, scores, ishbond, isacceptor, isdonor = [], [], [], [], [], []
                for n, g in grs:
                    coords.append(g[['c_x', 'c_y', 'c_z']].values)
                    gr_names.append(n)
                    grx = grs_dfx10ca.get_group(n)
                    scores.append(grx[score_cols])
                    ishbond.append((g['contact_hb'] == True).any())
                coords = np.array(coords, dtype=np.float32)
                a, b, c = coords.shape
                coords = coords.reshape(a, b*c)
                scores = concat(scores, ignore_index=True)
                scores['hbond'] = ishbond
                scores['is_acceptor'] = isacceptor
                scores['is_donor'] = isdonor

                rmsd = 0.5
                nbrs_rmsd = NearestNeighbors(algorithm='ball_tree', radius=np.sqrt(b) * rmsd).fit(coords)

                maxdist = 0.65
                nbrs_maxdist = NearestNeighbors(algorithm='ball_tree', radius=maxdist, metric=get_max).fit(coords)

                with open(nbrs_path_rmsd + f.split('.')[0] + '.pkl', 'wb') as outfile:
                    pickle.dump(nbrs_rmsd, outfile)

                with open(nbrs_path_maxdist + f.split('.')[0] + '.pkl', 'wb') as outfile:
                    pickle.dump(nbrs_maxdist, outfile)

                with open(nbrs_path_groupnames + f.split('.')[0] + '.pkl', 'wb') as outfile:
                    pickle.dump(gr_names, outfile)

                scores.to_parquet(nbrs_path_scores + f.split('.')[0] + '.parquet.gzip', compression='gzip', engine='pyarrow')


def make_cg_nbrs_hb():
    with open('/Users/npolizzi/Projects/design/Combs2/combs2/files/cg_dataframes.pkl', 'rb') as infile:
        cg_dfs = pickle.load(infile)
    vdm_path = '/Volumes/disk1/Combs2/database/20211005/vdMs/'
    score_cols = None
    for cg in os.listdir(vdm_path):
        print(cg)
        if cg[0] == '.':
            continue
        nbrs_path_rmsd = '/Volumes/disk1/Combs2/database/20211005/hbond_only/vdMs_cg_nbrs_rmsd/' + cg + '/'
        nbrs_path_maxdist = '/Volumes/disk1/Combs2/database/20211005/hbond_only/vdMs_cg_nbrs_maxdist/' + cg + '/'
        nbrs_path_groupnames = '/Volumes/disk1/Combs2/database/20211005/hbond_only/vdMs_cg_nbrs_groupnames/' + cg + '/'
        nbrs_path_scores = '/Volumes/disk1/Combs2/database/20211005/hbond_only/vdMs_cg_nbrs_scores/' + cg + '/'
        cg_df = cg_dfs[cg]
        os.makedirs(nbrs_path_rmsd, exist_ok=True)
        os.makedirs(nbrs_path_maxdist, exist_ok=True)
        os.makedirs(nbrs_path_groupnames, exist_ok=True)
        os.makedirs(nbrs_path_scores, exist_ok=True)
        vdm_path_cg = vdm_path + cg + '/'
        for f in os.listdir(vdm_path_cg):
            if f[0] == '.':
                continue
            print(f)
            df = read_parquet(vdm_path_cg + f)
            if score_cols is None:
                score_cols = [c for c in df.columns if 'C_score' in c]
                score_cols.extend(['rotamer', 'phi', 'psi', 'contact_type', 'ABPLE', 'ABPLE_3mer'])

            for c in score_cols:
                if c not in df.columns:
                    df[c] = np.nan
            
            dfx10ca = df[(df.chain=='X') & (df.resnum==10) & (df['name']=='CA')]
            grs_dfx10ca = dfx10ca.groupby(['CG', 'rota', 'probe_name'])
                    
            dfy = df[(df.chain=='Y') & (df.resnum==10)]
            hb_filter = dfy['contact_hb'] == True
            _dfy = dfy[hb_filter]

            if len(_dfy) == 0:
                print('no vdms', cg, f, 'due to no h-bonding')
                continue 
            dfy = merge(dfy, _dfy[['CG', 'rota', 'probe_name']].drop_duplicates(),
                        on=['CG', 'rota', 'probe_name'])
            dfy = merge(cg_df, dfy, on=['name', 'resnum', 'chain', 'resname'])
            grs = dfy.groupby(['CG', 'rota', 'probe_name'])

            coords, gr_names, scores = [], [], []
            for n, g in grs:
                coords.append(g[['c_x', 'c_y', 'c_z']].values)
                gr_names.append(n)
                grx = grs_dfx10ca.get_group(n)
                scores.append(grx[score_cols])
            coords = np.array(coords, dtype=np.float32)
            a, b, c = coords.shape
            coords = coords.reshape(a, b*c)
            scores = concat(scores, ignore_index=True)

            rmsd = 0.5
            nbrs_rmsd = NearestNeighbors(algorithm='ball_tree', radius=np.sqrt(b) * rmsd).fit(coords)

            maxdist = 0.65
            nbrs_maxdist = NearestNeighbors(algorithm='ball_tree', radius=maxdist, metric=get_max).fit(coords)

            with open(nbrs_path_rmsd + f.split('.')[0] + '.pkl', 'wb') as outfile:
                pickle.dump(nbrs_rmsd, outfile)

            with open(nbrs_path_maxdist + f.split('.')[0] + '.pkl', 'wb') as outfile:
                pickle.dump(nbrs_maxdist, outfile)

            with open(nbrs_path_groupnames + f.split('.')[0] + '.pkl', 'wb') as outfile:
                pickle.dump(gr_names, outfile)

            scores.to_parquet(nbrs_path_scores + f.split('.')[0] + '.parquet.gzip', compression='gzip', engine='pyarrow')
            

def make_rotamer_dfs():
    rotamer_dfs = dict()
    for resn, names in residue_sc_names.items():
        if resn in ['ALA', 'GLY']:
            continue
        _dict = dict(name=names)
        df = DataFrame(_dict)
        rotamer_dfs[resn] = df
    with open('/Users/npolizzi/Projects/design/Combs2/combs2/files/rotamer_dataframes.pkl', 'wb') as outfile:
        pickle.dump(rotamer_dfs, outfile)


def make_rotamer_nbrs():
    with open('/Users/npolizzi/Projects/design/Combs2/combs2/files/rotamer_dataframes.pkl', 'rb') as infile:
        rotamer_dfs = pickle.load(infile)
    vdm_path = '/Volumes/disk1/Combs2/database/20211005/vdMs/'
    for cg in os.listdir(vdm_path):
        if cg[0] == '.':
            continue
        print(cg)
        rotamers_path_rmsd = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_rotamers_nbrs_rmsd/' + cg + '/'
        rotamers_path_maxdist = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_rotamers_nbrs_maxdist/' + cg + '/'

        os.makedirs(rotamers_path_rmsd, exist_ok=True)
        os.makedirs(rotamers_path_maxdist, exist_ok=True)

        vdm_path_cg = vdm_path + cg + '/'
        for f in os.listdir(vdm_path_cg):
            if f[0] == '.':
                continue
            print(f)
            df = read_parquet(vdm_path_cg + f)
            resn = f.split('.')[0]
            if resn in ['ALA', 'GLY']:
                continue
            
            rotamer_df = rotamer_dfs[resn]

            dfx10 = df[(df.chain=='X') & (df.resnum==10)]
            dfx10 = merge(rotamer_df, dfx10, on=['name'])
            grs_dfx10 = dfx10.groupby(['CG', 'rota', 'probe_name'])

            coords, gr_names = [], []
            for n, g in grs_dfx10:
                coords.append(g[['c_x', 'c_y', 'c_z']].values)
                # gr_names.append(n)
                
            coords = np.array(coords, dtype=np.float32)
            a, b, c = coords.shape
            coords = coords.reshape(a, b*c)

            rmsd = 0.4
            rotamers_rmsd = NearestNeighbors(algorithm='ball_tree', radius=np.sqrt(b) * rmsd).fit(coords)

            maxdist = 0.6
            rotamers_maxdist = NearestNeighbors(algorithm='ball_tree', radius=maxdist, metric=get_max).fit(coords)

            with open(rotamers_path_rmsd + f.split('.')[0] + '.pkl', 'wb') as outfile:
                pickle.dump(rotamers_rmsd, outfile)

            with open(rotamers_path_maxdist + f.split('.')[0] + '.pkl', 'wb') as outfile:
                pickle.dump(rotamers_maxdist, outfile)


def get_angle_diff(ang1, ang2):
    """ Computes the difference in the angles ang1 and ang2 under period conditions.
        ang1 is a float, list, or array of angles in degrees.
        ang2 is a float in degrees."""
    if type(ang1) == list:
        ang1 = np.array(ang1)
    ang1 = ang1 * np.pi / 180
    ang2 = ang2 * np.pi / 180
    if type(ang1) == np.ndarray:
        vec1 = np.array([np.sin(ang1), np.cos(ang1)]).T
    elif type(ang1) == float:
        vec1 = [np.sin(ang1), np.cos(ang1)]
    else:
        raise Exception('ang1 is not a float, list, or array')
    vec2 = [np.sin(ang2), np.cos(ang2)]
    dp = np.dot(vec1,vec2)
    dp[dp > 1] = 1
    dp[dp < -1] = -1
    return np.arccos(dp) * 180 / np.pi


def add_hbond_info_to_dataframes_from_nbrs():
    """
    I used this function to add the hbond, is_donor, and is_acceptor columns to the vdM dataframes.
    """
    nbr_path = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_scores/'
    groupname_path = '/Volumes/disk1/Combs2/database/20211005/nbrs/vdMs_cg_nbrs_groupnames/'
    for d in os.listdir():
        if d[0] == '.':
            continue
        print(d)
        for aa in os.listdir(d):
            if aa[0] == '.':
                continue
            print('   ', aa)
            vdmpath = d + '/' + aa
            df_vdm = pd.read_parquet(vdmpath)
            df_nbr = pd.read_parquet(nbr_path + d + '/' + aa)
            with open(groupname_path + d + '/' + aa.split('.')[0] + '.pkl', 'rb') as infile:
                grnames = pickle.load(infile)
            df_grnames = pd.DataFrame(grnames, columns=['CG', 'rota', 'probe_name'])
            df_nbr = df_nbr.join(df_grnames)
            df_nbr = df_nbr[['CG', 'rota', 'probe_name', 'hbond', 'is_acceptor', 'is_donor']].drop_duplicates()
            df_vdm = pd.merge(df_vdm, df_nbr, on=['CG', 'rota', 'probe_name'])
            df_vdm.to_parquet(vdmpath, compression='gzip', engine='pyarrow')


#write a function that takes 3 angles and produces a rotation matrix
def rotmat(ang1, ang2, ang3):
    rotmat = np.array([[np.cos(ang1) * np.cos(ang2), 
                        np.cos(ang1) * np.sin(ang2) * np.sin(ang3) - np.sin(ang1) * np.cos(ang3), 
                        np.cos(ang1) * np.sin(ang2) * np.cos(ang3) + np.sin(ang1) * np.sin(ang3)],
                       [np.sin(ang1) * np.cos(ang2), 
                       np.sin(ang1) * np.sin(ang2) * np.sin(ang3) + np.cos(ang1) * np.cos(ang3), 
                       np.sin(ang1) * np.sin(ang2) * np.cos(ang3) - np.cos(ang1) * np.sin(ang3)],
                       [-np.sin(ang2),
                        np.cos(ang2) * np.sin(ang3), 
                        np.cos(ang2) * np.cos(ang3)]])
    return rotmat


#write a function that take a rotation matrix and computes the 3 angles that define it
def rotmat_to_angles(rotmat):
    ang1 = np.arctan2(rotmat[1,0], rotmat[0,0])
    ang2 = np.arctan2(-rotmat[2,0], np.sqrt(rotmat[2,1]**2 + rotmat[2,2]**2))
    ang3 = np.arctan2(rotmat[2,1], rotmat[2,2])
    return ang1, ang2, ang3


#write a function that takes the sin and cos of an angle and returns the angle
def sin_cos_to_angle(sin, cos):
    return np.arctan2(sin, cos)
