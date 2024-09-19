"""I changed the order of cc/so in parse_probe so that so comes before cc now. - 2021/5/8"""
__all__ = ['parse_probe']

import os
import pandas as pd
from collections import defaultdict


_dir = os.path.dirname(__file__)
_probe = os.path.join(_dir, '../programs/probe')
# Note that the github probe (probe-master) needed to be modified
# to recognize lower case chain letters.


#dtypes of dataframe columns in :func:`parse_probe`
probe_dtype_dict = {'interaction': 'category',
                    'chain1': 'category',
                    'resnum1': int,
                    'resname1': 'category',
                    'name1': 'category',
                    'atomtype1': 'category',
                    'chain2': 'category',
                    'resnum2': int,
                    'resname2': 'category',
                    'name2': 'category',
                    'atomtype2': 'category'}


# dataframe columns in :func:`parse_probe`
probe_col_names = list(probe_dtype_dict.keys())


def _make_segname(segname):
    return 'SEG' + '____'[:-len(segname)] + segname


def _make_chainname(chainname):
    return 'CHAIN' + '__'[:-len(chainname)] + chainname


def _make_cmd(pdb_file,
              segname1='', chain1='', resnum1='',
              segname2='', chain2='', resnum2='', resname2='',
              probe_sel_criteria='NOT METAL', maxbonded=4, # default for probe is maxbonded=4
              include_mc_mc=False, explicit_H=True, path_to_probe=_probe, outdir=None):
    """ """
    if include_mc_mc:
        mc = '-MC'
    else:
        mc = ''
    if explicit_H:
        Hs = '-Explicit'
    else:
        Hs = '-Implicit'

    if not all([segname1 == '', chain1 == '', resnum1 == '',
                segname2 == '', chain2 == '', resnum2 == '', resname2 == '',]):
        if segname1 != '':
            segname1 = _make_segname(segname1)
        if chain1 != '':
            chain1 = _make_chainname(chain1)
        if segname2 != '':
            segname2 = _make_segname(segname2)
        if chain2 != '':
            chain2 = _make_chainname(chain2)

        probe_action = '-ON'

        cmd = [path_to_probe,'-U -SEGID -CON -NOFACE', Hs, mc, '-WEAKH -DE32', '-' + str(maxbonded),
               probe_action, '"', segname1, chain1, str(resnum1), probe_sel_criteria,
               '"', '"', segname2, chain2, str(resnum2), resname2, probe_sel_criteria, '"']
    else:
        probe_action = '-SE'

        cmd = [path_to_probe,'-U -SEGID -CON -NOFACE', Hs, mc, '-WEAKH -DE32', '-' + str(maxbonded),
               probe_action, '"', 'ALL', probe_sel_criteria, '"']

    cmd.append(pdb_file)

    if outdir is not None:
        if outdir[-1] != '/':
            outdir += '/'
        outfile = ''.join([outdir, pdb_file.split('/')[-1], '.probe'])
        cmd.extend(['>', outfile])

    return ' '.join(cmd)


def _parse_probe_line(line):
    """ """
        
    spl = line.split(':')[1:]
    INTERACTION = spl[1]
    CHAIN1 = spl[2][:2].strip()
    RESNUM1 = int(spl[2][2:6])
    RESNAME1 = spl[2][6:10].strip()
    NAME1 = spl[2][10:15].strip()
    ATOMTYPE1 = spl[12]
    CHAIN2 = spl[3][:2].strip()
    RESNUM2 = int(spl[3][2:6])
    RESNAME2 = spl[3][6:10].strip()
    NAME2 = spl[3][10:15].strip()
    ATOMTYPE2 = spl[13]
    if RESNAME1 == 'HOH':
        NAME1 = 'O'
    if RESNAME2 == 'HOH':
        NAME2 = 'O'
    return (INTERACTION, CHAIN1,
            RESNUM1, RESNAME1, NAME1, ATOMTYPE1,
            CHAIN2, RESNUM2, RESNAME2,
            NAME2, ATOMTYPE2)


def parse_probe(pdb_file,
                segname1='', chain1='', resnum1='',
                segname2='', chain2='', resnum2='', resname2='',
                probe_sel_criteria='NOT METAL', maxbonded=4, # default for probe is Maxbonded=4
                include_mc_mc=False, explicit_H=True, include_wc=False, path_to_probe=_probe,
                outdir=None, ignore_bo=False):
    """ Creates a :class:`pandas.DataFrame` of Probe
    contacts for *segname1* and *chain1*.
    Only one atom-atom contact per unique pair is reported, in
    order of *hb* (H-bond), then *wh* (weak H-bond), then *cc* (close contact).  Contacts
    labeled *so* (small overlap), *bo* (bad overlap), *wc* (wide contact)
    are not reported.  Metal contacts are not reported accurately, so
    the default behavior is to exlude them. For metal coordination,
    see :func:`~.pdbheader.parse_metal_contacts`.

    Parameters
    ----------
    pdb_file : str
        path to pdb file, preferably a biological unit
    segname1 : str
    chain1 : str
    resnum1 : int or str
    segname2 : str
    chain2 : str
    resnum2 : int or str
    probe_sel_criteria : str, optional
        filter Probe output by selection
    maxbonded: int
        number of bonds between two atoms after which probe detects contacts.
    path_to_probe : str
        path to probe program
    outdir : str, optional
        path to output directory for Probe txt file

    Returns
    -------
    pandas.DataFrame
        Probe contact info for unique atom-atom pairs
    """
    cmd = _make_cmd(pdb_file=pdb_file,
              segname1=segname1, chain1=chain1, resnum1=resnum1,
              segname2=segname2, chain2=chain2, resnum2=resnum2, resname2=resname2,
              probe_sel_criteria=probe_sel_criteria, maxbonded=maxbonded,
              include_mc_mc=include_mc_mc, explicit_H=explicit_H,
                    path_to_probe=path_to_probe, outdir=outdir)
    probe_dict_interactions = defaultdict(list)
    probe_dict_info = defaultdict(list)
    probe_data = []
    print(cmd)
    with os.popen(cmd) as probefile:
        for line in probefile:
            line_data = _parse_probe_line(line)
            probe_dict_interactions[line_data[1:4]].append(line_data[0])
            probe_dict_info[line_data[1:4]].append(line_data)
    for residue, interactions in probe_dict_interactions.items():
        if (not ignore_bo) and 'bo' in interactions:
            continue
        probe_dict = defaultdict(list)
        for line_data in probe_dict_info[residue]:
            probe_dict[line_data[1:]].append(line_data[0])
        for info, interactions in probe_dict.items():
            if 'hb' in interactions:
                interaction = 'hb'
            elif 'wh' in interactions:
                interaction = 'wh'
            elif 'so' in interactions:
                interaction = 'so'
            elif 'cc' in interactions:
                interaction = 'cc'
            elif include_wc and 'wc' in interactions:
                interaction = 'wc'
            else:
                continue
            data = [interaction]
            data.extend(info)
            probe_data.append(data)
    probe_df = pd.DataFrame(probe_data, columns=probe_col_names)
    return probe_df.astype(dtype=probe_dtype_dict)

