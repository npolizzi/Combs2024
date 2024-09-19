__all__ = ['Template']

import pandas as pd
from .dataframe import make_df_from_prody
from .functions import get_ABPLE
from prody.measure.measure import getDihedral
from prody import parsePDB, parseDSSP, execDSSP
from .convex_hull import AlphaHull
from numpy import nan, diff, ndarray
from .rotalyze import parse_rotalyze
import os
from pickle import dump
from collections import Counter


_dir = os.path.dirname(__file__)
path_to_rotalyze = os.path.join(_dir, '../programs/phenix.rotalyze')


class Template:

    def __init__(self, pdb, **kwargs):
        if type(pdb) == str:
            self.pdb = parsePDB(pdb)
        else:
            self.pdb = pdb  # pdb should be prody object poly-gly with CA hydrogens for design.
        self.dataframe = make_df_from_prody(self.pdb, **kwargs)
        self.alpha_hull = None
        self.set_phi_psi_abple()
        self.transformations_to_ideal_ala = dict()
        self.phi_psi_dict = dict()

    @staticmethod
    def calc_phi(res_bef, res):
        """
        res_bef and res are pre-sorted by N, Ca, C

        Parameters
        ----------
        res_bef
        res

        Returns
        -------

        """
        cm1 = res_bef[res_bef['name'] == 'C'][['c_x', 'c_y', 'c_z']].values
        c = res[['c_x', 'c_y', 'c_z']].values
        return getDihedral(cm1, c[0 ,:], c[1 ,:], c[2 ,:], radian=False)[0]

    @staticmethod
    def calc_psi(res, res_aft):
        """
        res_aft and res are pre-sorted by N, Ca, C

        Parameters
        ----------
        res
        res_aft

        Returns
        -------

        """
        cp1 = res_aft[res_aft['name'] == 'N'][['c_x', 'c_y', 'c_z']].values
        c = res[['c_x', 'c_y', 'c_z']].values
        return getDihedral(c[0 ,:], c[1 ,:], c[2 ,:], cp1, radian=False)[0]

    def set_phi_psi_abple(self):
        """

        Returns
        -------

        """
        cols = ['segment', 'chain', 'resnum', 'resname', 'name', 'c_x', 'c_y', 'c_z']
        df = self.dataframe[cols]
        df_name_order = pd.DataFrame(dict(name=['N', 'CA', 'C']))
        df = pd.merge(df_name_order, df, on='name')  # sorts df by N, Ca, C atoms
        data = []
        for seg, g_seg in df.groupby('segment'):
            for ch, g_ch in g_seg.groupby('chain'):
                resnums = sorted(set(g_ch.resnum))
                gs_rn = g_ch.groupby('resnum')
                chain_data = []
                n_indices = []
                for i, rn in enumerate(resnums):
                    g_rn = gs_rn.get_group(rn)
                    if rn - 1 in resnums:
                        try:
                            phi = self.calc_phi(gs_rn.get_group(rn - 1), g_rn)
                        except ValueError:
                            phi = nan
                    else:
                        phi = nan
                    if rn + 1 in resnums:
                        try:
                            psi = self.calc_psi(g_rn, gs_rn.get_group(rn + 1))
                        except ValueError:
                            psi = nan
                    else:
                        psi = nan
                    resname = g_rn.resname.iat[0]
                    abple = get_ABPLE(resname, phi, psi)
                    if abple == 'n':
                        n_indices.append(i)
                    chain_data.append([seg, ch, rn, phi, psi, abple])
                total_resnums = len(chain_data)
                if len(n_indices) > 0:
                    # replace n label with abple vote from nbr residues
                    for n_index in n_indices:
                        prev_ind = max(0, n_index - 2)
                        post_ind = n_index + 3
                        abples = [c[-1] for c in chain_data[prev_ind:n_index]]
                        post_abples = [c[-1] for c in chain_data[n_index + 1:post_ind]]
                        abples.extend(post_abples)
                        c = Counter(abples)
                        common = c.most_common(2)
                        for abple, count in common:
                            if abple != 'n':
                                chain_data[n_index][-1] = abple
                                break
                        
                for j in range(total_resnums):
                    if j == 0 and j < total_resnums - 2 and (diff(resnums[j:j+2])==1).all():
                        # abple3mer = 'n' + chain_data[j][-1] + chain_data[j+1][-1]
                        # instead of n label, label the terminus with abple of nbr residue
                        abple3mer = chain_data[j][-1] + chain_data[j][-1] + chain_data[j+1][-1]
                    elif j > 0 and j < total_resnums - 1 and (diff(resnums[j-1:j+2])==1).all():
                        abple3mer = chain_data[j-1][-1] + chain_data[j][-1] + chain_data[j+1][-1]
                    elif j == total_resnums - 1 and (diff(resnums[j-1:])==1).all():
                        # abple3mer = chain_data[j - 1][-1] + chain_data[j][-1] + 'n'
                        abple3mer = chain_data[j - 1][-1] + chain_data[j][-1] + chain_data[j][-1]
                    else:
                        # abple3mer = 'nnn'
                        abple3mer = 'EEE'
                    _data = chain_data[j].copy()
                    _data.append(abple3mer)
                    data.append(_data)
        cols_phipsi = ['segment', 'chain', 'resnum', 'phi', 'psi', 'ABPLE', 'ABPLE_3mer']
        df_phipsi = pd.DataFrame(data, columns=cols_phipsi)
        self.dataframe = pd.merge(self.dataframe, df_phipsi, on=['segment', 'chain', 'resnum'], how='left')

    def rotalyze(self, pdb_path, path_to_phenix_rotalyze=path_to_rotalyze):
        """

        Parameters
        ----------
        pdb_path
        path_to_phenix_rotalyze

        Returns
        -------

        """
        rot_cols = ['chain', 'resnum', 'chi1', 'chi2',
                    'chi3', 'chi4', 'evaluation', 'rotamer']
        df_rot = parse_rotalyze(pdb_path, path_to_phenix_rotalyze)
        old_df = self.dataframe.copy()
        self.dataframe = pd.merge(self.dataframe, df_rot[rot_cols],
                                  on=['chain', 'resnum'])  # needs unique chains (no segments in merge)
        if len(self.dataframe) < len(old_df):
            mer = pd.merge(old_df, self.dataframe[['chain', 'resnum']].drop_duplicates(),
                           on=['chain', 'resnum'], how='outer', indicator=True)
            df = mer[mer['_merge'] == 'left_only'].drop(columns='_merge')
            self.dataframe = pd.concat((self.dataframe, df)).sort_values(by=['segment', 'chain', 'resnum', 'name'])

    def set_dssp(self, path_to_pdb_for_dssp='./'):
        if type(self.pdb.getData('secondary')) != ndarray:
            try:
                parseDSSP(execDSSP(path_to_pdb_for_dssp), self.pdb)
            except Exception:
                raise ('PDB does not contain dssp info. Use path_to_pdb_for_dssp to set it.')
        seg_chain_resnums = list((s, c, r) for s, c, r in zip(self.pdb.ca.getSegnames(),
                                                             self.pdb.ca.getChids(),
                                                             self.pdb.ca.getResnums()))

        segs = self.pdb.ca.getSegnames()
        chains = self.pdb.ca.getChids()
        resnums = self.pdb.ca.getResnums()
        
        df_dssp = pd.DataFrame(dict(dssp=self.pdb.ca.getData('secondary'), segment=segs, chain=chains, resnum=resnums))
        if '' in df_dssp.dssp.values:
            df_dssp.loc[df_dssp.dssp == '', 'dssp'] = 'C'

        data = []
        for seg, g_seg in df_dssp.groupby('segment'):
            for ch, g_ch in g_seg.groupby('chain'):
                resnums = sorted(set(g_ch.resnum))
                gs_rn = g_ch.groupby('resnum')
                chain_data = []
                for i, rn in enumerate(resnums):
                    g_rn = gs_rn.get_group(rn)
                    chain_data.append([seg, ch, rn, g_rn['dssp'].iat[0]])
                total_resnums = len(chain_data)
                for j in range(total_resnums):
                    if j == 0 and j < total_resnums - 2 and (diff(resnums[j:j + 2]) == 1).all():
                        dssp3mer = 'n' + chain_data[j][-1] + chain_data[j + 1][-1]
                    elif j > 0 and j < total_resnums - 1 and (diff(resnums[j - 1:j + 2]) == 1).all():
                        dssp3mer = chain_data[j - 1][-1] + chain_data[j][-1] + chain_data[j + 1][-1]
                    elif j == total_resnums - 1 and (diff(resnums[j - 1:]) == 1).all():
                        dssp3mer = chain_data[j - 1][-1] + chain_data[j][-1] + 'n'
                    else:
                        dssp3mer = 'nnn'
                    _data = chain_data[j][:-1]
                    _data.append(dssp3mer)
                    data.append(_data)
        
        df_dssp_3mer = pd.DataFrame(data, columns=['segment', 'chain', 'resnum', 'dssp_3mer'])
    
        df_dssp = pd.merge(df_dssp_3mer, df_dssp, on=['segment', 'chain', 'resnum'])
        self.dataframe = pd.merge(self.dataframe, df_dssp, on=['segment', 'chain', 'resnum'])

    def set_alpha_hull(self, pdb_w_CB, alpha=9):
        """

        Parameters
        ----------
        pdb_w_CB
        alpha

        Returns
        -------

        """
        self.pdb_w_CB = pdb_w_CB
        self.alpha_hull = AlphaHull(alpha)
        self.alpha_hull.set_coords(pdb_w_CB)
        self.alpha_hull.calc_hull()
        df_coords = self.dataframe[['c_x', 'c_y', 'c_z']].values
        self.dataframe['dist_to_hull'] = self.alpha_hull.get_pnts_distance(df_coords)

    def save(self, outpath='./', filename='template.pkl'):
        if outpath[-1] != '/':
            outpath += '/'
        try:
            os.makedirs(outpath)
        except:
            pass

        with open(outpath + filename, 'wb') as outfile:
            dump(self, outfile)