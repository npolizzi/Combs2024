__all__ = ['SuperposeLig', 'make_df_corr']

from prody import writePDB, parsePDB
from pandas import DataFrame, merge, concat, read_pickle, read_parquet
from .contacts import Clash
from .constants import coords_cols
from .transformation import get_rot_trans
import numpy as np
from .functions import make_lig_atom_type_dict, make_lig_hbond_dict, read_lig_txt, fast_concat
from .dataframe import make_df_from_prody
from multiprocessing import Pool
from os import listdir, makedirs


def make_df_corr(dict_corr):
    """Example of dict_corr for apixaban and carboxamide:
    dict_corr = dict(APX=dict(GLN=dict(NE2='N3', CD='C11', OE1='O1', CG='C10'),
                              ASN=dict(ND2='N3',CG='C11',OD1='O1',CB='C10')))"""
    names = list()
    lig_names = list()
    resnames = list()
    lig_resnames = list()
    for lig_resname, lig_dict in dict_corr.items():
        for resname, names_dict in lig_dict.items():
            names.extend(names_dict.keys())
            lig_names.extend(names_dict.values())
            len_ = len(names_dict.keys())
            resnames.extend([resname] * len_)
            lig_resnames.extend([lig_resname] * len_)
    return DataFrame(list(zip(names, resnames, lig_names, lig_resnames)),
                     columns=['name', 'resname', 'lig_name', 'lig_resname'])


class _SuperposeLig:
    """This really only needs to be the coords of the ligand,
    superposed onto a vdm, but it need not include the vdm itself, right?"""

    def __init__(self, df_reps, df_lig, df_corr, **kwargs):
        self.df_reps = df_reps
        self.groupby = kwargs.get('groupby', ['rota', 'CG', 'probe_name'])
        self.df_reps_gr = self.df_reps.groupby(self.groupby)
        self.df_lig = df_lig
        self.df_corr = df_corr
        self.df_nonclashing_lig = DataFrame()
        self._nonclashing_lig = list()
        self.lig_coords = None
        self.ligand_iFG_corr_sorted = None
        self.df_corr_sorted = None
        self.truncate_lig_atoms = kwargs.get('truncate_lig_atoms', [])  # list of (lig_resname, lig_name) tuples

    def set_sorted_lig_corr(self):
        self.ligand_iFG_corr_sorted = self.df_corr.sort_values(by=['lig_resname', 'lig_name'])
        self.df_corr_sorted = self.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates()

    def get_lig_coords(self):
        df_corr = self.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates()
        self.lig_coords = merge(df_corr, self.df_lig,
                                on=['lig_resname', 'lig_name'], sort=False)[['c_x', 'c_y', 'c_z']].values

    def _get_cg_coords(self, rep):
        df_cg = rep[rep.chain == 'Y']
        df_cg_c = merge(self.ligand_iFG_corr_sorted[['resname', 'name']], df_cg,
                         how='inner', on=['resname', 'name'], sort=False)
        return df_cg_c[['c_x', 'c_y', 'c_z']].values

    @staticmethod
    def _get_rep_info(rep):
        rota = rep.rota.iat[0]
        CG = rep.CG.iat[0]
        probe_name = rep.probe_name.iat[0]
        return rota, CG, probe_name

    @staticmethod
    def _set_info(df, rota, CG, probe_name):
        df['rota'] = rota
        df['CG'] = CG
        df['probe_name'] = probe_name

    def _find(self, rep):
        cg_coords = self._get_cg_coords(rep)
        R, m_com, t_com = get_rot_trans(self.lig_coords, cg_coords)
        df_lig = self.df_lig.copy()
        for i in range(0, len(coords_cols), 3):
            df_lig[coords_cols[i:i + 3]] = np.dot(df_lig[coords_cols[i:i + 3]] - m_com, R) + t_com
        df_rep_vdm = rep[rep.chain == 'X']
        rota, CG, probe_name = self._get_rep_info(rep)
        self._set_info(df_lig, rota, CG, probe_name)
        df_lig_trunc = merge(df_lig, self.df_corr, on=['lig_resname', 'lig_name'], sort=False, how='outer', indicator=True)
        df_lig_trunc = df_lig_trunc[df_lig_trunc._merge == 'left_only'].drop(columns='_merge')

        if self.truncate_lig_atoms:
            df_trunc = DataFrame(self.truncate_lig_atoms, columns=['lig_resname', 'lig_name'])
            df_lig_trunc = merge(df_lig_trunc, df_trunc, on=['lig_resname', 'lig_name'], sort=False, how='outer',
                                 indicator=True)
            df_lig_trunc = df_lig_trunc[df_lig_trunc._merge == 'left_only'].drop(columns='_merge')

        clash = Clash(df_lig_trunc, df_rep_vdm)
        clash.set_grouping(['resnum'])
        clash.find()
        if len(clash.dfq_clash_free) > 0:
            self._nonclashing_lig.append(df_lig)

    def find(self):
        if self.ligand_iFG_corr_sorted is None:
            self.set_sorted_lig_corr()

        if self.lig_coords is None:
            self.get_lig_coords()

        for name, rep in self.df_reps_gr:
            self._find(rep)

        if len(self._nonclashing_lig) > 0:
            self.df_nonclashing_lig = fast_concat(self._nonclashing_lig)

    def print_lig(self, lig, rota, CG, probe_name, out_path, out_name):
        df_lig = self.df_nonclashing_lig[(self.df_nonclashing_lig['rota'] == rota)
                                         & (self.df_nonclashing_lig['CG'] == CG)
                                         & (self.df_nonclashing_lig['probe_name'] == probe_name)]
        coords = np.stack(df_lig[df_lig['name'] == n][['c_x', 'c_y', 'c_z']].item() for n in lig.getNames())
        lig_copy = lig.copy()
        lig_copy.setCoords(coords)
        writePDB(out_path + out_name, lig_copy)


class SuperposeLig:
    """
    Superimposes a ligand *path_to_lig_pdb* onto vdM instances and removes clashing ligands.
    Outputs :class:`pandas.DataFrame` files (pickled).

    Example usage:
        s = SuperposeLig(**kwargs)
        s.setup()
        s.sys_argv = int(sys.argv[1]) - 1  # if running on a cluster
        s.run()

    Parameters
    ----------
    lig_resname : str
        residue name of the ligand

    path_to_lig_params : str
        path to Rosetta params file for the ligand

    remove_from_hbond_dict : list
        list of atom names in the ligand to be removed from the H-bonding dictionary.

    path_to_lig_pdb : str
        path to ligand pdb file

    path_to_lig_txt : str
        path to ligand txt file that lists correspondence of ligand atoms
        with atoms in a vdM.

    path_to_dataframe_files : str
        path to vdM database files

    dataframe_file_extension : str
        file extension of dataframe files, default='.parquet.gzip'

    superpose_type : str
        category of vdMs on which the ligand will be superposed. Allowed
        values are: 'top_X' (where X is any integer), 'all_enriched',
        or 'all'.

    hb_only : bool
        only superpose onto H-bonding vdMs. default=False

    resnames_for_superpose : dict
        dictionary containing keys that are CG names (e.g. conh2, bb_cco, ...)
        and values that are a list of allowed amino acid residue names,
        e.g. ['ALA', 'SER', 'GLU'].
        Example:
            resnames_for_superpose=dict(conh2=['ALA', 'SER', 'GLU'],
                                        bb_cco=['THR', 'ASN', 'LYS'])

    path_to_outdir : str
        path to output directory. It will be created if it doesn't exist.

    num_cpus : int, default=1
        number of cpus to use for the superposition job.

    ind : int, default=None
        index of *_params* on which to operate.  Used in conjunction with a
        compute cluster.
    """
    def __init__(self, **kwargs):
        self._params = list()
        self.lig_resname = kwargs.get('lig_resname')
        self.path_to_lig_params = kwargs.get('path_to_lig_params')
        self.remove_from_hbond_dict = kwargs.get('remove_from_hbond_dict', list())
        self.path_to_lig_pdb = kwargs.get('path_to_lig_pdb')
        self.path_to_lig_txt = kwargs.get('path_to_lig_txt')
        self.lig_df = None
        self.path_to_dataframe_files = kwargs.get('path_to_dataframe_files')
        self.dataframe_file_extension = kwargs.get('dataframe_file_extension', '.parquet.gzip')
        self.superpose_type = kwargs.get('superpose_type', '')
        self.hb_only = kwargs.get('hb_only', False)
        self.resnames_for_superpose = kwargs.get('resnames_for_superpose', dict())
        self.path_to_outdir = kwargs.get('path_to_outdir')
        self.num_cpus = kwargs.get('num_cpus', 1)
        self.ind = kwargs.get('ind', None)

    def setup(self):
        """

        Returns
        -------

        """
        atom_type_dict = make_lig_atom_type_dict(self.lig_resname, self.path_to_lig_params)
        hbond_dict = make_lig_hbond_dict(self.lig_resname, self.path_to_lig_params)
        [hbond_dict[self.lig_resname].pop(n) for n in self.remove_from_hbond_dict]
        lig = parsePDB(self.path_to_lig_pdb)
        self.lig_df = make_df_from_prody(lig, can_hbond_dict=hbond_dict,
                                         lig_atom_types_dict=atom_type_dict)
        df_corr = read_lig_txt(self.path_to_lig_txt)
        for (CG_type, CG_group_number), df_corr_CG_type in df_corr.groupby(['CG_type', 'CG_group']):
            path_to_cg_dataframe_files = self.path_to_dataframe_files + CG_type + '/'
            for abple in set('ABPLE'):
                resnames_in_dir = listdir(path_to_cg_dataframe_files)
                if len(self.resnames_for_superpose) > 0 and CG_type in self.resnames_for_superpose:
                    resnames = [r for r in self.resnames_for_superpose[CG_type]
                                    if r + self.dataframe_file_extension in resnames_in_dir]
                else:
                    resnames = [r.split('.')[0] for r in resnames_in_dir]
                for resname in resnames:
                    self._params.append((path_to_cg_dataframe_files, resname,
                                        df_corr_CG_type, CG_type, CG_group_number, abple))
        self._params = sorted(self._params, key=lambda x: [x[0], x[1], x[3], x[4], x[5]])

    def _run(self, path_to_cg_dataframe_files, resname,
            df_corr_CG_type, CG_type, CG_group_number, abple):
        """

        Parameters
        ----------
        path_to_cg_dataframe_files
        resname
        df_corr_CG_type
        CG_type
        CG_group_number
        abple

        Returns
        -------

        """
        if self.dataframe_file_extension == '.parquet.gzip':
            resn_df = read_parquet(path_to_cg_dataframe_files + resname + self.dataframe_file_extension)
        elif self.dataframe_file_extension == '.pkl':
            resn_df = read_pickle(path_to_cg_dataframe_files + resname + self.dataframe_file_extension)
        else:
            raise Exception('Dataframe filetype is not recognized...')

        print(path_to_cg_dataframe_files, resname,
              df_corr_CG_type, CG_type, CG_group_number, abple)

        resn_df = resn_df[(resn_df['resnum'] == 10)]
        resn_df_x_abple = resn_df[(resn_df['chain'] == 'X') & (resn_df['ABPLE'] == abple)]
        if len(resn_df_x_abple) == 0:
            return
        if self.hb_only:
            score_col = 'C_score_hb_ABPLE_' + abple
            rank_col = 'cluster_rank_hb_ABPLE_' + abple
        else:
            score_col = 'C_score_ABPLE_' + abple
            rank_col = 'cluster_rank_ABPLE_' + abple
        if 'top' in self.superpose_type:
            rank = int(self.superpose_type.split('_')[-1])
            resn_df_x_abple = resn_df_x_abple[resn_df_x_abple[rank_col] <= rank]  # note these might not be enriched
        if 'enriched' in self.superpose_type:
            resn_df_x_abple = resn_df_x_abple[resn_df_x_abple[score_col] > 0]  # enriched
        resn_df = merge(resn_df, resn_df_x_abple[['CG', 'rota', 'probe_name']].drop_duplicates(),
                           on=['CG', 'rota', 'probe_name'])

        s = _SuperposeLig(resn_df, self.lig_df, df_corr_CG_type)
        s.find()
        if len(s.df_nonclashing_lig) > 0:
            s.df_nonclashing_lig.drop(columns='seg_chain_resnum', inplace=True)
            outpath = '/'.join([self.path_to_outdir, self.lig_resname, self.superpose_type,
                                CG_type, str(CG_group_number), abple])
            outpath += '/'
            try:
                makedirs(outpath)
            except:
                pass
            if self.dataframe_file_extension == '.parquet.gzip':
                s.df_nonclashing_lig.to_parquet(outpath + resname + self.dataframe_file_extension,
                                               engine='pyarrow', compression='gzip')
            elif self.dataframe_file_extension == '.pkl':
                s.df_nonclashing_lig.to_pickle(outpath + resname + self.dataframe_file_extension)

    def run(self):
        """

        Returns
        -------

        """
        if self.num_cpus > 1:
            with Pool(self.num_cpus) as pool:
                pool.starmap(self._run, self._params)

        elif self.ind is not None:
            ind = int(self.ind) - 1  # account for indexing on wynton cluster
            self._run(*self._params[ind])
            # number of task ids for wynton = (number of CG groups in df_corr) x 5 ABPLE x 20 aas

        else:
            for ind in range(len(self._params)):
                self._run(*self._params[ind])
