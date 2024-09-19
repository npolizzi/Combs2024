""""""
__all__ = ['parse_rotalyze']

import os
import pandas as pd
import numpy as np


_rotalyze = '/Users/npolizzi/Applications/MolProbity/build/bin/phenix.rotalyze'


#dtypes of dataframe columns in :func:`parse_rotalyze`
rota_dtype_dict = {'chain': 'category',
                    'resnum': int,
                    'icode': 'category',
                    'altloc': 'category',
                    'resname': 'category',
                    'occupancy': float,
                    'chi1': float,
                    'chi2': float,
                    'chi3': float,
                    'chi4': float,
                    'evaluation': 'category',
                    'rotamer': 'category'}


# dataframe columns in :func:`parse_rotalyze`
rota_col_names = list(rota_dtype_dict.keys())


def _parse_rota_line(line):
    """ """
    spl = line.split(':')
    CHAIN = spl[0][:2].strip()
    RESNUM = spl[0][2:6]
    ICODE = spl[0][6]
    ALTLOC = spl[0][7]
    RESNAME = spl[0][8:].strip()
    OCCU = spl[1]
    CHI1 = np.nan if spl[3] == '' else spl[3]
    CHI2 = np.nan if spl[4] == '' else spl[4]
    CHI3 = np.nan if spl[5] == '' else spl[5]
    CHI4 = np.nan if spl[6] == '' else spl[6]
    EVAL = spl[7]
    ROTA = spl[8].strip()
    return (CHAIN, RESNUM, ICODE, ALTLOC, RESNAME,
            OCCU, CHI1, CHI2, CHI3, CHI4, EVAL, ROTA)


def parse_rotalyze(pdb_file, path_to_phenix_rotalyze='', outdir=None):
    """ Creates a :class:`pandas.DataFrame` of rotamer
    chi angles, occupancy, and rotamer classification.

    Parameters
    ----------
    pdb_file : str
        path to pdb file
    outdir : str, optional
        path to output directory for rotalyze txt file

    Returns
    -------
    pandas.DataFrame
        rotamer info per residue
    """
    if path_to_phenix_rotalyze == '':
        path_to_phenix_rotalyze = _rotalyze

    cmd = ' '.join([path_to_phenix_rotalyze, pdb_file])
    rota_data = []
    with os.popen(cmd) as rotafile:
        for line in rotafile:
            if 'residue:occupancy' in line:
                break
        for line in rotafile:
            if line[:7] == 'SUMMARY':
                continue
            rota_data.append(_parse_rota_line(line))
    rota_df = pd.DataFrame(rota_data, columns=rota_col_names).astype(dtype=rota_dtype_dict)
    if outdir is not None:
        if outdir[-1] != '/':
            outdir += '/'
        try:
            os.makedirs(outdir)
        except:
            pass
        rota_df.to_pickle(outdir + pdb_file.split('/')[-1][:-3] + 'pkl')
    else:
        return rota_df.astype(dtype=rota_dtype_dict)

