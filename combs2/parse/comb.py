from ..design.constants import aa_heavy_atom_names
from ..design.convex_hull import partition_res_by_burial, AlphaHull
from ..design.functions import get_ABPLE, get_ABPLE_from_sel
from ..design.dataframe import make_df_from_prody
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import traceback


"""Note that there seems to be an issue with chain numbering of biounits like large viruses.
Perhaps this is only for a few instances, e.g. with pdb 4OQ8.  Perhaps it is a general
issue with the way prody generates biomolecules.  Thus, I am skipping the 
problematic biounits with a try: except clause below."""


_dir = os.path.dirname(__file__)
path_to_cg_dicts = os.path.join(_dir, '../files/cg_dicts.pkl')
cg_dicts = pd.read_pickle(path_to_cg_dicts)


def run_comb(cg_dict, inpath_prody_dir,
             inpath_rotamer_dir, outdir, path_to_probe_paths, ind, HIS_option=None):

    try:
        os.makedirs(outdir)
    except:
        pass

    with open(path_to_probe_paths, 'rb') as infile:
        probe_paths = pickle.load(infile)

    probe_path = probe_paths[ind]
    print(probe_path)
    pdb_name = '_'.join(probe_path.split('/')[-1].split('_')[:3])
    prody_pdb_path = inpath_prody_dir + pdb_name + '.pkl'

    with open(probe_path, 'rb') as infile:
        probe = pickle.load(infile)

    with open(prody_pdb_path, 'rb') as infile:
        prody_pdb = pickle.load(infile)

    avg_bb_beta = prody_pdb.select('name N CA C O').getBetas().mean()
    sigma_bb_beta = prody_pdb.select('name N CA C O').getBetas().std()

    alpha_hull = AlphaHull(9)
    alpha_hull.set_coords(prody_pdb)
    alpha_hull.calc_hull()
    ri_ex, ri_int, ri_bur = partition_res_by_burial(prody_pdb, alpha=9, ahull_cb=alpha_hull)

    with open(inpath_rotamer_dir + pdb_name + '.pkl', 'rb') as infile:
        df_rot = pickle.load(infile)

    probe = probe[(~probe.phi1.isna()) & (~probe.psi1.isna())]
    probe = probe[
        (~probe.clash1) & (probe.rscc1 >= 0.8) & (probe.rsr1 <= 0.4) & (probe.rsrz1 <= 2)]
    chain1 = probe.chain1.iat[0]
    probe_same_chain = probe[probe.chain2 == chain1]
    probe_same_chain = probe_same_chain[np.abs(probe_same_chain.resnum1 - probe_same_chain.resnum2) > 8]
    probe = pd.concat([probe[probe.chain2 != probe.chain1.iat[0]], probe_same_chain])
    df_ = pd.concat([probe[(probe.resname1 == resname) & probe.name1.isin(atom_names)]
            for resname, atom_names in cg_dict.items()])
    if len(df_) == 0:
        return
    cg_grs = df_.groupby('resnum1')
    d = dict()
    for n, cg_gr in cg_grs:
        rot_grs = cg_gr.groupby(['chain2', 'resnum2'])
        d[n] = rot_grs

    i = 1
    dfs = []
    probe1 = probe[
        ['chain1', 'resnum1', 'resname1', 'rscc1', 'rsr1', 'rsrz1', 'phi1', 'psi1', 'rama1']].drop_duplicates()
    probe_colnames = ['chain', 'resnum', 'resname', 'rscc', 'rsr', 'rsrz', 'phi', 'psi', 'rama']
    probe1.columns = probe_colnames
    probe2 = probe[
        ['chain2', 'resnum2', 'resname2', 'rscc2', 'rsr2', 'rsrz2', 'phi2', 'psi2', 'rama2']].drop_duplicates()
    probe2.columns = probe_colnames
    for resnum1, rots in d.items():
        try:
            old_len_dfs = len(dfs)
            sel1 = prody_pdb.select('chain ' + chain1 + ' and resnum ' + str(resnum1))
            resname1 = sel1.getResnames()[0]
            names1 = set(sel1.getNames())
            if len(set(cg_dict[resname1]) - names1) != 0:
                print('resnum1 does not contain all atoms in CG', resnum1)
                continue
            if HIS_option is not None:
                if HIS_option == 'hid':
                    if 'HE2' in names1:
                        continue
                if HIS_option == 'hie':
                    if 'HD1' in names1:
                        continue
                if HIS_option == 'hip':
                    if 'HE2' not in names1 or 'HD1' not in names1:
                        continue
            if (sel1.getBetas() > 60).any() or (sel1.getBetas() < 1).any():
                print('betas', resnum1)
                continue
            if (sel1.getOccupancies() < 0.99).any():
                print('occs', resnum1)
                continue
            sel1 = sel1.select('name ' + ' '.join(cg_dict[resname1]))
            df_sel1 = make_df_from_prody(sel1, include_betas_occupancies=True)
            if 'GLY' in cg_dict or 'ALA' in cg_dict:
                df_sel1_G_A = df_sel1[df_sel1.resname.isin({'GLY', 'ALA'})]
                df_sel1_rots = pd.merge(df_sel1, df_rot[
                    ['chain', 'resnum', 'resname', 'chi1', 'chi2', 'chi3', 'chi4', 'evaluation', 'rotamer']],
                                        on=['chain', 'resnum', 'resname'])
                df_sel1 = pd.concat([df_sel1_G_A, df_sel1_rots])
            else:
                df_sel1 = pd.merge(df_sel1, df_rot[
                    ['chain', 'resnum', 'resname', 'chi1', 'chi2', 'chi3', 'chi4', 'evaluation', 'rotamer']],
                                   on=['chain', 'resnum', 'resname'], how='left')
            df_sel1 = pd.merge(df_sel1, probe1, on=['chain', 'resnum', 'resname'])
            df_sel1['pdb_chain'] = df_sel1['chain']
            df_sel1['chain'] = 'Y'
            df_sel1['pdb_segment'] = df_sel1['segment']
            df_sel1 = df_sel1.drop(columns=['segment', 'seg_chain_resnum'])
            df_sel1['pdb_resnum'] = df_sel1['resnum']
            df_sel1['resnum'] = 10
            df_sel1['dssp'] = sel1.getData('secondary')[0]
            df_sel1['dssp_acc'] = sel1.getData('dssp_acc')[0]
            seq_sel1 = ''
            dsspseq_sel1 = ''
            abpleseq_sel1 = ''
            for k in range(1, 8):
                s = prody_pdb.select('chain ' + chain1 + ' and resnum `' + str(resnum1 - 7 + k) + '` and name CA')
                if s is None:
                    seq_sel1 += '_'
                    dsspseq_sel1 += '_'
                    abpleseq_sel1 += '_'
                else:
                    seq_sel1 += s.getSequence()
                    dsspseq_sel1 += s.getData('secondary')[0]
                    phi = s.getData('dssp_phi')[0]
                    psi = s.getData('dssp_psi')[0]
                    resname = s.getResnames()[0]
                    abpleseq_sel1 += get_ABPLE(resname, phi, psi)
            for k in range(1, 7):
                s = prody_pdb.select('chain ' + chain1 + ' and resnum `' + str(resnum1 + k) + '` and name CA')
                if s is None:
                    seq_sel1 += '_'
                    dsspseq_sel1 += '_'
                    abpleseq_sel1 += '_'
                else:
                    seq_sel1 += s.getSequence()
                    dsspseq_sel1 += s.getData('secondary')[0]
                    phi = s.getData('dssp_phi')[0]
                    psi = s.getData('dssp_psi')[0]
                    resname = s.getResnames()[0]
                    abpleseq_sel1 += get_ABPLE(resname, phi, psi)
            df_sel1['dssp_seq'] = dsspseq_sel1
            df_sel1['ABPLE_seq'] = abpleseq_sel1
            df_sel1['ABPLE'] = abpleseq_sel1[6]
            df_sel1['ABPLE_3mer'] = abpleseq_sel1[5:8]
            df_sel1['dssp_3mer'] = dsspseq_sel1[5:8]
            j = 1
            for (chain2, resnum2), rot in rots:
                # print(rot.columns)
                try:
                    if (rot.clash2.iat[0]) | (rot.rscc2.iat[0] < 0.8) | (rot.rsr2.iat[0] > 0.4) | (rot.rsrz2.iat[0] > 2):
                        print(chain2, resnum2, 'rsr, rsrz, rscc, clash')
                        continue
                    sel2 = prody_pdb.select('chain ' + chain2 + ' and resnum ' + str(resnum2))
                    sel2_ = sel2.copy()
                    if (sel2.getBetas() > 60).any() or (sel2.getBetas() < 1).any():
                        print(chain2, resnum2, 'betas')
                        continue
                    if (sel2.getOccupancies() < 0.99).any():
                        print(chain2, resnum2, 'occs')
                        continue
                    dssp_sel2 = sel2.getData('secondary')[0]
                    resname2 = sel2.getResnames()[0]
                    names2 = set(sel2.getNames())
                    if len(aa_heavy_atom_names[resname2] - names2) != 0:
                        print(1, 'chain2,resnum2', chain2, resnum2)
                        continue
                    sel2m1 = prody_pdb.select('chain ' + chain2 + ' and resnum ' + str(resnum2 - 1) + ' and name CA C O N')
                    # sel2m1 = prody_pdb.select(
                    #     'chain ' + chain2 + ' and resnum ' + str(resnum2 - 1) + ' and name CA C O')
                    if sel2m1 is None:
                        print(2, 'chain2,resnum2', chain2, resnum2)
                        continue
                    sel2p1 = prody_pdb.select('chain ' + chain2 + ' and resnum ' + str(resnum2 + 1) + ' and name CA N C O')
                    # sel2p1 = prody_pdb.select(
                    #     'chain ' + chain2 + ' and resnum ' + str(resnum2 + 1) + ' and name CA N')
                    if sel2p1 is None:
                        print(3, 'chain2,resnum2', chain2, resnum2)
                        continue
                    # if len({'CA', 'C', 'O'} - set(sel2m1.getNames())) != 0:
                    #     print(4, 'chain2,resnum2', chain2, resnum2)
                    #     continue
                    # if len({'CA', 'N'} - set(sel2p1.getNames())) != 0:
                    #     print(5, 'chain2,resnum2', chain2, resnum2)
                    #     continue
                    if len({'CA', 'C', 'O', 'N'} - set(sel2m1.getNames())) != 0:
                        print(4, 'chain2,resnum2', chain2, resnum2)
                        continue
                    if len({'CA', 'N', 'C', 'O'} - set(sel2p1.getNames())) != 0:
                        print(5, 'chain2,resnum2', chain2, resnum2)
                        continue
                    seq_sel2 = ''
                    dsspseq_sel2 = ''
                    abpleseq_sel2 = ''
                    for k in range(1, 8):
                        s = prody_pdb.select('chain ' + chain2 + ' and resnum `' + str(resnum2 - 7 + k) + '` and name CA')
                        if s is None:
                            seq_sel2 += '_'
                            dsspseq_sel2 += '_'
                            abpleseq_sel2 += '_'
                        else:
                            seq_sel2 += s.getSequence()
                            dsspseq_sel2 += s.getData('secondary')[0]
                            phi = s.getData('dssp_phi')[0]
                            psi = s.getData('dssp_psi')[0]
                            resname = s.getResnames()[0]
                            abpleseq_sel2 += get_ABPLE(resname, phi, psi)

                    for k in range(1, 7):
                        s = prody_pdb.select('chain ' + chain2 + ' and resnum `' + str(resnum2 + k) + '` and name CA')
                        if s is None:
                            seq_sel2 += '_'
                            dsspseq_sel2 += '_'
                            abpleseq_sel2 += '_'
                        else:
                            seq_sel2 += s.getSequence()
                            dsspseq_sel2 += s.getData('secondary')[0]
                            phi = s.getData('dssp_phi')[0]
                            psi = s.getData('dssp_psi')[0]
                            resname = s.getResnames()[0]
                            abpleseq_sel2 += get_ABPLE(resname, phi, psi)
                    sel2 = sel2 | sel2m1 | sel2p1
                    df_sel2 = make_df_from_prody(sel2, include_betas_occupancies=True)
                    df_sel2 = pd.merge(df_sel2, df_rot[
                        ['chain', 'resnum', 'resname', 'chi1', 'chi2', 'chi3', 'chi4', 'evaluation', 'rotamer']],
                                            on=['chain', 'resnum', 'resname'], how='left')
                    df_sel2 = pd.merge(df_sel2, probe2, on=['chain', 'resnum', 'resname'], how='left')
                    cols = ['name', 'contact_hb', 'contact_wh', 'contact_cc', 'contact_so', 'partners_hb', 'partners_wh',
                            'partners_cc', 'partners_so']
                    ggrs = rot.groupby(['resname2', 'name2'])
                    contact_dict = defaultdict(list)
                    for g_name, ggr in ggrs:
                        interactions = set(ggr.interaction)
                        contact_dict['name'].append(g_name[1])
                        for inter in interactions:
                            contact_dict['contact_' + inter].append(True)
                            contact_dict['partners_' + inter].append(','.join(set(ggr[ggr.interaction == inter].name1)))
                        for inter in {'wh', 'hb', 'cc', 'so'} - interactions:
                            contact_dict['contact_' + inter].append(np.nan)
                            contact_dict['partners_' + inter].append(np.nan)
                    df_int_2 = pd.DataFrame(contact_dict, columns=cols)
                    old_df_sel1 = df_sel1.copy()
                    df_sel2_not10 = df_sel2[df_sel2.resnum != resnum2]
                    df_sel2_10 = df_sel2[df_sel2.resnum == resnum2]
                    df_sel2_10 = pd.merge(df_sel2_10, df_int_2, on='name', how='left')
                    df_sel2 = pd.concat([df_sel2_10, df_sel2_not10])

                    ggrs = rot.groupby(['resname1', 'name1'])
                    contact_dict = defaultdict(list)
                    for g_name, ggr in ggrs:
                        interactions = set(ggr.interaction)
                        contact_dict['name'].append(g_name[1])
                        for inter in interactions:
                            contact_dict['contact_' + inter].append(True)
                            contact_dict['partners_' + inter].append(','.join(set(ggr[ggr.interaction == inter].name2)))
                        for inter in {'wh', 'hb', 'cc', 'so'} - interactions:
                            contact_dict['contact_' + inter].append(np.nan)
                            contact_dict['partners_' + inter].append(np.nan)
                    df_int_1 = pd.DataFrame(contact_dict, columns=cols)
                    df_sel1 = pd.merge(df_sel1, df_int_1, on='name', how='left')

                    df_sel2['pdb_chain'] = df_sel2['chain']
                    df_sel2['pdb_segment'] = df_sel2['segment']
                    df_sel2['pdb_resnum'] = df_sel2['resnum']
                    df_sel2['dssp'] = dssp_sel2
                    df_sel2 = df_sel2.drop(columns=['segment', 'seg_chain_resnum'])
                    df_sel2['chain'] = 'X'
                    df_sel2['resnum'] = df_sel2['pdb_resnum'] - resnum2 + 10
                    abplem1 = get_ABPLE_from_sel(sel2m1)
                    abplep1 = get_ABPLE_from_sel(sel2p1)
                    abple = get_ABPLE_from_sel(sel2_)
                    dsspm1 = sel2m1.getData('secondary')[0]
                    dsspp1 = sel2p1.getData('secondary')[0]
                    df_sel2.loc[df_sel2.resnum == 9, 'dssp'] = dsspm1
                    df_sel2.loc[df_sel2.resnum == 11, 'dssp'] = dsspp1
                    df_sel2.loc[df_sel2.resnum == 10, 'dssp_acc'] = sel2_.getData('dssp_acc')[0]
                    df_sel2.loc[df_sel2.resnum == 9, 'dssp_acc'] = sel2m1.getData('dssp_acc')[0]
                    df_sel2.loc[df_sel2.resnum == 11, 'dssp_acc'] = sel2p1.getData('dssp_acc')[0]
                    df_sel2.loc[df_sel2.resnum == 9, 'ABPLE'] = abplem1
                    df_sel2.loc[df_sel2.resnum == 11, 'ABPLE'] = abplep1
                    df_sel2.loc[df_sel2.resnum == 10, 'ABPLE'] = abple
                    df_sel2.loc[:, 'ABPLE_3mer'] = abplem1 + abple + abplep1
                    df_sel2.loc[:, 'dssp_3mer'] = dsspm1 + dssp_sel2 + dsspp1
                    df_sel2['dssp_seq'] = dsspseq_sel2
                    df_sel2['ABPLE_seq'] = abpleseq_sel2
                    df_sels = pd.concat([df_sel1, df_sel2])
                    df_sels['rota'] = j
                    df_sels['CG'] = i
                    df_sels['seq'] = seq_sel1 + '-' + seq_sel2
                    dfs.append(df_sels)
                    j += 1
                    df_sel1 = old_df_sel1
                except Exception:
                    print('Exception within')
                    traceback.print_exc()
            if len(dfs) > old_len_dfs:
                i += 1
        except Exception:
            print('Exception')
            traceback.print_exc()
    if len(dfs) == 0:
        return
    df = pd.concat(dfs)

    df_chain_resnum = df['pdb_chain'] + '_' + df['pdb_resnum'].astype(str)
    set_status(df, prody_pdb, ri_ex, df_chain_resnum, 'exposed')
    set_status(df, prody_pdb, ri_int, df_chain_resnum, 'intermed')
    set_status(df, prody_pdb, ri_bur, df_chain_resnum, 'buried')

    df['dist_to_hull'] = alpha_hull.get_pnts_distance(df[['c_x','c_y','c_z']].values)
    df['pdb_name'] = pdb_name
    df['score_index'] = ind
    df['probe_name'] = probe_path.split('/')[-1].split('.')[0]
    df['avg_bb_beta'] = avg_bb_beta
    df['sigma_bb_beta'] = sigma_bb_beta
    df = df[df.apply(lambda x: '_' not in x.seq, axis=1)]
    if len(df) == 0:
        return
    grs = [set_contact_type(gr) for n, gr in df.groupby(['CG', 'rota', 'probe_name'])]
    df = pd.concat(grs)
    df = df.sort_values(['CG', 'rota', 'probe_name'])
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(outdir + probe_path.split('/')[-1].split('.')[0] + '.parquet.gzip', engine='pyarrow', compression='gzip')
    # df.to_pickle(outdir + probe_path.split('/')[-1])


def set_status(df, prody_pdb, resindices, df_chain_resnum, hull_status):
    sel = prody_pdb.ca.select('resindex ' + ' '.join([str(ri) for ri in resindices]))
    chids = sel.getChids()
    resnums = sel.getResnums()
    chain_resnum = {ch + '_' + str(rn) for ch, rn in zip(chids, resnums)}
    df.loc[df_chain_resnum.isin(chain_resnum), 'hull_status'] = hull_status


def get_contact_names_set(g):
    s = set()
    for partners in ['partners_hb', 'partners_cc', 'partners_wh', 'partners_so']:
        s |= {x for v in g[partners][~g[partners].isna()].apply(lambda x: x.split(',')).values for x in v}
    return s


def set_contact_type(gr):
    gr = gr.copy()
    bb_atoms = {'N', 'H', 'C', 'O', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}
    gr_cg = gr[gr['chain'] == 'Y']
    gr_cg_contact_atoms = get_contact_names_set(gr_cg)
    if len(gr_cg_contact_atoms & bb_atoms) > 0:
        set_bb_contact_type(gr, gr_cg_contact_atoms)
    else:
        gr['contact_type'] = 'sc'
    return gr


def set_bb_contact_type(gr, gr_cg_contact_atoms):
    if len(gr_cg_contact_atoms - {None, 'N', 'H', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}) == 0:
        gr['contact_type'] = 'phi'
    elif len(gr_cg_contact_atoms - {None, 'C', 'O', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}) == 0:
        gr['contact_type'] = 'psi'
    else:
        gr['contact_type'] = 'phi_psi'


def get_contact_type(gr):
    bb_atoms = {'N', 'H', 'C', 'O', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}
    gr_cg = gr[gr['chain'] == 'Y']
    gr_cg_contact_atoms = get_contact_names_set(gr_cg)
    if len(gr_cg_contact_atoms & bb_atoms) > 0:
        return get_bb_contact_type(gr_cg_contact_atoms)
    else:
        return 'sc'


def get_bb_contact_type(gr_cg_contact_atoms):
    if len(gr_cg_contact_atoms - {None, 'N', 'H', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}) == 0:
        return 'phi'
    elif len(gr_cg_contact_atoms - {None, 'C', 'O', 'CA', 'HA', 'HA1', 'HA2', 'HA3'}) == 0:
        return 'psi'
    else:
        return 'phi_psi'


def set_contacts(df):
    df = df.drop(columns='contact_type')
    contacts = []
    for n, g in df.groupby(['CG', 'rota', 'probe_name']):
        contact_info = list(n)
        contact_info.append(get_contact_type(g))
        contacts.append(contact_info)
    df_contact_type = pd.DataFrame(contacts, columns=['CG', 'rota', 'probe_name', 'contact_type']).drop_duplicates()
    df = pd.merge(df, df_contact_type, on=['CG', 'rota', 'probe_name'])
    df = df.sort_values(['CG', 'rota', 'probe_name'])
    df.reset_index(drop=True, inplace=True)
    return df

