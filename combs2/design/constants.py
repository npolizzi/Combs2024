__all__ = ['one_letter_code', 'resnames_aa_20', 'resnames_aa_20_join', 'can_hbond', 'atom_type_dict', 'coords_cols',
           'atom_types_sortkey']

from collections import defaultdict


one_letter_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                   'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                   'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                   'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
                   'MSE': 'm', 'ANY': '.', 'FE': 'fe', 'ZN': 'zn', 'HEM': 'h'}

inv_one_letter_code = {
    'C': 'CYS',
    'D': 'ASP',
    'S': 'SER',
    'Q': 'GLN',
    'K': 'LYS',
    'I': 'ILE',
    'P': 'PRO',
    'T': 'THR',
    'F': 'PHE',
    'N': 'ASN',
    'G': 'GLY',
    'H': 'HIS',
    'L': 'LEU',
    'R': 'ARG',
    'W': 'TRP',
    'A': 'ALA',
    'V': 'VAL',
    'E': 'GLU',
    'Y': 'TYR',
    'M': 'MET',
    'm': 'MSE',
    '.': 'ANY',
    'fe': 'FE',
    'zn': 'ZN',
    'h': 'HEM'}

resnames_aa_20 = ['CYS', 'ASP', 'SER', 'GLN', 'LYS',
                   'ILE', 'PRO', 'THR', 'PHE', 'ASN',
                   'GLY', 'HIS', 'LEU', 'ARG', 'TRP',
                   'ALA', 'VAL', 'GLU', 'TYR', 'MET',
                   'MSE']

resnames_aa_20_join = 'CYS ASP SER GLN LYS ILE PRO THR PHE ASN GLY ' \
                      'HIS LEU ARG TRP ALA VAL GLU TYR MET MSE'

aa_heavy_atom_names = {
    'CYS': {'C', 'CA', 'CB', 'N', 'O', 'SG'},
    'ASP': {'C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'},
    'SER': {'C', 'CA', 'CB', 'N', 'O', 'OG'},
    'GLN': {'C', 'CA', 'CB', 'CD', 'CG', 'N', 'NE2', 'O', 'OE1'},
    'LYS': {'C', 'CA', 'CB', 'CD', 'CE', 'CG', 'N', 'NZ', 'O'},
    'ILE': {'C', 'CA', 'CB', 'CD1', 'CG1', 'CG2', 'N', 'O'},
    'PRO': {'C', 'CA', 'CB', 'CD', 'CG', 'N', 'O'},
    'THR': {'C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'},
    'PHE': {'C', 'CA', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'N', 'O'},
    'ASN': {'C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'},
    'GLY': {'C', 'CA', 'N', 'O'},
    'HIS': {'C', 'CA', 'CB', 'CD2', 'CE1', 'CG', 'N', 'ND1', 'NE2', 'O'},
    'LEU': {'C', 'CA', 'CB', 'CD1', 'CD2', 'CG', 'N', 'O'},
    'ARG': {'C', 'CA', 'CB', 'CD', 'CG', 'CZ', 'N', 'NE', 'NH1', 'NH2', 'O'},
    'TRP': {'C',
            'CA',
            'CB',
            'CD1',
            'CD2',
            'CE2',
            'CE3',
            'CG',
            'CH2',
            'CZ2',
            'CZ3',
            'N',
            'NE1',
            'O'},
    'ALA': {'C', 'CA', 'CB', 'N', 'O'},
    'VAL': {'C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O'},
    'GLU': {'C', 'CA', 'CB', 'CD', 'CG', 'N', 'O', 'OE1', 'OE2'},
    'TYR': {'C',
            'CA',
            'CB',
            'CD1',
            'CD2',
            'CE1',
            'CE2',
            'CG',
            'CZ',
            'N',
            'O',
            'OH'},
    'MET': {'C', 'CA', 'CB', 'CE', 'CG', 'N', 'O', 'SD'}
}


atom_type_dict = defaultdict(dict,
            {'ARG': {'1HD': 'h_alkyl',
              '1HG': 'h_alkyl',
              '1HH1': 'h_pol',
              '1HH2': 'h_pol',
              '2HD': 'h_alkyl',
              '3HD': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HG': 'h_alkyl',
              '2HH1': 'h_pol',
              '2HH2': 'h_pol',
              'CD': 'c_alkyl',
              'CG': 'c_alkyl',
              'CZ': 'co',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HE': 'h_pol',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'HH11': 'h_pol',
              'HH12': 'h_pol',
              'HH21': 'h_pol',
              'HH22': 'h_pol',
              'NE': 'n',
              'NH1': 'n',
              'NH2': 'n'},
             'ASN': {'1HD2': 'h_pol',
              '2HD2': 'h_pol',
              'CG': 'co',
              'HD21': 'h_pol',
              'HD22': 'h_pol',
              'ND2': 'n',
              'OD1': 'o'},
             'ASP': {'CG': 'co', 'OD1': 'o', 'OD2': 'o'},
             'CYS': {'HG': 'h_pol', 'SG': 's'},
             'GLN': {'1HE2': 'h_pol',
              '1HG': 'h_alkyl',
              '2HE2': 'h_pol',
              '2HG': 'h_alkyl',
              '3HG': 'h_alkyl',
              'CD': 'co',
              'CG': 'c_alkyl',
              'HE21': 'h_pol',
              'HE22': 'h_pol',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'NE2': 'n',
              'OE1': 'o'},
             'GLU': {'1HG': 'h_alkyl',
              '2HG': 'h_alkyl',
              'CD': 'co',
              'CG': 'c_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'OE1': 'o',
              'OE2': 'o'},
             'HIS': {'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CG': 'c_aro',
              'HD1': 'h_pol',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_pol',
              'ND1': 'n',
              'NE2': 'n'},
             'ILE': {'1HD1': 'h_alkyl',
              '1HG1': 'h_alkyl',
              '1HG2': 'h_alkyl',
              '2HD1': 'h_alkyl',
              '2HG1': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HD1': 'h_alkyl',
              '3HG1': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CD1': 'c_alkyl',
              'CG1': 'c_alkyl',
              'CG2': 'c_alkyl',
              'HD11': 'h_alkyl',
              'HD12': 'h_alkyl',
              'HD13': 'h_alkyl',
              'HG12': 'h_alkyl',
              'HG13': 'h_alkyl',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl'},
             'LEU': {'1H': 'h_alkyl',
              '1HD1': 'h_alkyl',
              '1HD2': 'h_alkyl',
              '2H': 'h_alkyl',
              '2HD1': 'h_alkyl',
              '2HD2': 'h_alkyl',
              '3H': 'h_alkyl',
              '3HD1': 'h_alkyl',
              '3HD2': 'h_alkyl',
              'CD1': 'c_alkyl',
              'CD2': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD11': 'h_alkyl',
              'HD12': 'h_alkyl',
              'HD13': 'h_alkyl',
              'HD21': 'h_alkyl',
              'HD22': 'h_alkyl',
              'HD23': 'h_alkyl',
              'HG': 'h_alkyl'},
             'LYS': {'1HD': 'h_alkyl',
              '1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '1HZ': 'h_pol',
              '2HD': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '2HZ': 'h_pol',
              '3HZ': 'h_pol',
              'CD': 'c_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'HZ1': 'h_pol',
              'HZ2': 'h_pol',
              'HZ3': 'h_pol',
              'NZ': 'n'},
             'MET': {'1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HE': 'h_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HE1': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'SD': 's'},
             'MSE': {'1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HE': 'h_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HE1': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'SE': 's'},
             'PHE': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CE2': 'c_aro',
              'CG': 'c_aro',
              'CZ': 'c_aro',
              'HD1': 'h_aro',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_aro',
              'HZ': 'h_aro'},
             'PRO': {'1HD': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HD': 'h_alkyl',
              '2HG': 'h_alkyl',
              'CD': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              '3HG': 'h_alkyl',
              '3HD': 'h_alkyl'},
             'SER': {'HG': 'h_pol', 'OG': 'o'},
             'THR': {'1HG2': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CG2': 'c_alkyl',
              'HG1': 'h_pol',
              '1HG': 'h_pol',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl',
              'OG1': 'o'},
             'TRP': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE2': 'c_aro',
              'CE3': 'c_aro',
              'CG': 'c_aro',
              'CH2': 'c_aro',
              'CZ2': 'c_aro',
              'CZ3': 'c_aro',
              'HD1': 'h_aro',
              'HE1': 'h_pol',
              'HE3': 'h_aro',
              'HH2': 'h_aro',
              'HZ2': 'h_aro',
              'HZ3': 'h_aro',
              'NE1': 'n'},
             'TYR': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CE2': 'c_aro',
              'CG': 'c_aro',
              'CZ': 'c_aro',
              'HD1': 'h_aro',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_aro',
              'HH': 'h_pol',
              'OH': 'o'},
             'VAL': {'1HG1': 'h_alkyl',
              '1HG2': 'h_alkyl',
              '2HG1': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HG1': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CG1': 'c_alkyl',
              'CG2': 'c_alkyl',
              'HG11': 'h_alkyl',
              'HG12': 'h_alkyl',
              'HG13': 'h_alkyl',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl'},
             'HEM': {'FE': 'fe',
                     'NA': 'n',
                     'NB': 'n',
                     }
             })


for aa in resnames_aa_20:
    atom_type_dict[aa]['H'] = 'h_pol'
    atom_type_dict[aa]['1H'] = 'h_pol'
    atom_type_dict[aa]['2H'] = 'h_pol'
    atom_type_dict[aa]['3H'] = 'h_pol'
    atom_type_dict[aa]['H1'] = 'h_pol'
    atom_type_dict[aa]['H2'] = 'h_pol'
    atom_type_dict[aa]['H3'] = 'h_pol'
    atom_type_dict[aa]['N'] = 'n'
    atom_type_dict[aa]['O'] = 'o'
    atom_type_dict[aa]['OXT'] = 'o'
    atom_type_dict[aa]['1HA'] = 'h_alkyl'
    atom_type_dict[aa]['2HA'] = 'h_alkyl'
    atom_type_dict[aa]['3HA'] = 'h_alkyl'
    atom_type_dict[aa]['HA'] = 'h_alkyl'
    atom_type_dict[aa]['HA2'] = 'h_alkyl'
    atom_type_dict[aa]['HA3'] = 'h_alkyl'
    atom_type_dict[aa]['HB'] = 'h_alkyl'
    atom_type_dict[aa]['HB1'] = 'h_alkyl'
    atom_type_dict[aa]['HB2'] = 'h_alkyl'
    atom_type_dict[aa]['HB3'] = 'h_alkyl'
    atom_type_dict[aa]['1HB'] = 'h_alkyl'
    atom_type_dict[aa]['2HB'] = 'h_alkyl'
    atom_type_dict[aa]['3HB'] = 'h_alkyl'
    atom_type_dict[aa]['CB'] = 'c_alkyl'
    atom_type_dict[aa]['CA'] = 'c_alkyl'
    atom_type_dict[aa]['C'] = 'co'

# METALS
atom_type_dict['NA']['NA'] = 'na'


def rec_dd():
    """returns a recursive dictionary"""
    return defaultdict(rec_dd)

# make dictionary of (resname, name, donor/acceptor) keys and their h-bond atom vectors.
can_hbond = rec_dd()
for aa in resnames_aa_20:
    can_hbond[aa]['H']['donor'] = [('H', 'N')]
    can_hbond[aa]['1H']['donor'] = [('1H', 'N')]
    can_hbond[aa]['2H']['donor'] = [('2H', 'N')]
    can_hbond[aa]['3H']['donor'] = [('3H', 'N')]
    can_hbond[aa]['H1']['donor'] = [('H1', 'N')]
    can_hbond[aa]['H2']['donor'] = [('H2', 'N')]
    can_hbond[aa]['H3']['donor'] = [('H3', 'N')]
    can_hbond[aa]['N']['donor'] = [('H', 'N')]
    can_hbond[aa]['O']['acceptor'] = ('O', 'C', 'C')
    can_hbond[aa]['OXT']['acceptor'] = ('OXT', 'C', 'C')
can_hbond['N_term']['N']['donor'] = [('H1 1H', 'N'), ('H2 2H', 'N'), ('H3 3H', 'N')]
can_hbond['SER']['OG']['donor'] = [('HG', 'OG')]
can_hbond['SER']['HG']['donor'] = [('HG', 'OG')]
can_hbond['SER']['OG']['acceptor'] = ('OG', 'HG', 'CB')
can_hbond['THR']['OG1']['donor'] = [('HG1', 'OG1')]
can_hbond['THR']['HG1']['donor'] = [('HG1', 'OG1')]
can_hbond['THR']['OG1']['acceptor'] = ('OG1', 'HG1', 'CB')
can_hbond['TRP']['NE1']['donor'] = [('HE1', 'NE1')]
can_hbond['TRP']['HE1']['donor'] = [('HE1', 'NE1')]
can_hbond['TYR']['OH']['donor'] = [('HH', 'OH')]
can_hbond['TYR']['HH']['donor'] = [('HH', 'OH')]
can_hbond['TYR']['OH']['acceptor'] = ('OH','HH','CZ')
can_hbond['MET']['SD']['acceptor'] = ('SD','CE','CG')
can_hbond['MSE']['SE']['acceptor'] = ('SE','CE','CG')
can_hbond['CYS']['SG']['donor'] = [('HG', 'SG')]
can_hbond['CYS']['HG']['donor'] = [('HG', 'SG')]
can_hbond['CYS']['SG']['acceptor'] = ('SG', 'HG', 'CB')
can_hbond['GLN']['NE2']['donor'] = [('1HE2 HE21', 'NE2'), ('2HE2 HE22', 'NE2')]
can_hbond['GLN']['1HE2']['donor'] = [('1HE2 HE21', 'NE2')]
can_hbond['GLN']['HE21']['donor'] = [('1HE2 HE21', 'NE2')]
can_hbond['GLN']['2HE2']['donor'] = [('2HE2 HE22', 'NE2')]
can_hbond['GLN']['HE22']['donor'] = [('2HE2 HE22', 'NE2')]
can_hbond['GLN']['OE1']['acceptor'] = ('OE1', 'CD', 'CD')
can_hbond['ASN']['ND2']['donor'] = [('1HD2 HD21', 'ND2'), ('2HD2 HD22', 'ND2')]
can_hbond['ASN']['1HD2']['donor'] = [('1HD2 HD21', 'ND2')]
can_hbond['ASN']['HD21']['donor'] = [('1HD2 HD21', 'ND2')]
can_hbond['ASN']['2HD2']['donor'] = [('2HD2 HD22', 'ND2')]
can_hbond['ASN']['HD22']['donor'] = [('2HD2 HD22', 'ND2')]
can_hbond['ASN']['OD1']['acceptor'] = ('OD1', 'CG', 'CG')
can_hbond['GLU']['OE1']['acceptor'] = ('OE1', 'CD', 'CD')
can_hbond['GLU']['OE2']['acceptor'] = ('OE2', 'CD', 'CD')
can_hbond['ASP']['OD1']['acceptor'] = ('OD1', 'CG', 'CG')
can_hbond['ASP']['OD2']['acceptor'] = ('OD2', 'CG', 'CG')
can_hbond['LYS']['NZ']['donor'] = [('1HZ HZ1', 'NZ'), ('2HZ HZ2', 'NZ'), ('3HZ HZ3', 'NZ')]
can_hbond['LYS']['1HZ']['donor'] = [('1HZ HZ1', 'NZ')]
can_hbond['LYS']['HZ1']['donor'] = [('1HZ HZ1', 'NZ')]
can_hbond['LYS']['2HZ']['donor'] = [('2HZ HZ2', 'NZ')]
can_hbond['LYS']['HZ2']['donor'] = [('2HZ HZ2', 'NZ')]
can_hbond['LYS']['3HZ']['donor'] = [('3HZ HZ3', 'NZ')]
can_hbond['LYS']['HZ3']['donor'] = [('3HZ HZ3', 'NZ')]
can_hbond['HIS']['ND1']['donor'] = [('HD1', 'ND1')]
can_hbond['HIS']['HD1']['donor'] = [('HD1', 'ND1')]
can_hbond['HIS']['ND1']['acceptor'] = ('ND1', 'CG', 'CD2')
can_hbond['HIS']['NE2']['donor'] = [('HE2', 'NE2')]
can_hbond['HIS']['HE2']['donor'] = [('HE2', 'NE2')]
can_hbond['HIS']['NE2']['acceptor'] = ('NE2', 'CD2', 'CE1')
can_hbond['ARG']['NE']['donor'] = [('HE', 'NE')]
can_hbond['ARG']['HE']['donor'] = [('HE', 'NE')]
can_hbond['ARG']['NH1']['donor'] = [('HH11 1HH1', 'NH1'), ('HH12 2HH1', 'NH1')]
can_hbond['ARG']['HH11']['donor'] = [('HH11 1HH1', 'NH1')]
can_hbond['ARG']['1HH1']['donor'] = [('HH11 1HH1', 'NH1')]
can_hbond['ARG']['HH12']['donor'] = [('HH12 2HH1', 'NH1')]
can_hbond['ARG']['2HH1']['donor'] = [('HH12 2HH1', 'NH1')]
can_hbond['ARG']['NH2']['donor'] = [('HH21 1HH2', 'NH2'), ('HH22 2HH2', 'NH2')]
can_hbond['ARG']['HH21']['donor'] = [('HH21 1HH2', 'NH2')]
can_hbond['ARG']['1HH2']['donor'] = [('HH21 1HH2', 'NH2')]
can_hbond['ARG']['HH22']['donor'] = [('HH22 2HH2', 'NH2')]
can_hbond['ARG']['2HH2']['donor'] = [('HH22 2HH2', 'NH2')]

hbond_types = {'h_pol', 'o', 'n', 's'}
hbond_donor_types = {'h_pol', 'o', 'n', 's'}
hbond_acceptor_types = {'o', 'n', 's', 'f'}
charged_atoms = {('LYS', 'NZ'),
                 ('LYS', '1HZ'),
                 ('LYS', 'HZ1'),
                 ('LYS', '2HZ'),
                 ('LYS', 'HZ2'),
                 ('LYS', '3HZ'),
                 ('LYS', 'HZ3'),
                 ('GLU', 'OE1'),
                 ('GLU', 'OE2'),
                 ('ASP', 'OD1'),
                 ('ASP', 'OD2'),
                 ('ARG', 'NE'),
                 ('ARG', 'HE'),
                 ('ARG', 'NH1'),
                 ('ARG', 'HH11'),
                 ('ARG', '1HH1'),
                 ('ARG', 'HH12'),
                 ('ARG', '2HH1'),
                 ('ARG', 'NH2'),
                 ('ARG', 'HH21'),
                 ('ARG', '1HH2'),
                 ('ARG', 'HH22'),
                 ('ARG', '2HH2')}


num_sc_atoms_residue = dict(ALA=4, GLY=0, ASP=6, ASN=8, GLN=11, GLU=9, ARG=18,
                            ILE=13, TYR=15, THR=8, PHE=14, LYS=16, LEU=13, SER=5,
                            VAL=10, TRP=18, MET=11, MSE=11, HIS=11, CYS=5, PRO=9)

num_sc_atoms_residue_deprotonated = dict(TYR=14, THR=7, SER=4, HIS=10, CYS=4)


rel_coords_dict = dict(SC=['CA', 'N', 'C'], HNCA=['N', 'H', 'CA'],
                       CO=['C', 'O', 'CA'], PHI_PSI=['CA', 'N', 'C'])


backbone_str = 'name CA 1H 2H 3H H1 H2 H3 HA HA1 HA2 HA3 1HA 2HA 3HA C O N H OXT'


dict_ = {'LYS': ['CE', 'CD', 'NZ'],
          'ASP': ['CG', 'OD1', 'OD2'],
          'PHE': ['CZ', 'CE1', 'CE2'],
          'ASN': ['CG', 'OD1', 'ND2'],
          'GLN': ['CD', 'OE1', 'NE2'],
          'ALA': ['C', 'CA', 'CB'],
          'ARG': ['CZ', 'NH2', 'NH1'],
          'THR': ['OG1', 'CB', 'CG2'],
          'GLY': ['CA', 'N', 'C'],
          'TYR': ['CZ', 'CE1', 'OH'],
          'LEU': ['CG', 'CD1', 'CD2'],
          'VAL': ['CB', 'CG1', 'CG2'],
          'GLU': ['CD', 'OE1', 'OE2'],
          'PRO': ['CB', 'CG', 'CD'],
          'SER': ['CB', 'OG', 'CA'],
          'CYS': ['CB', 'SG', 'CA'],
          'MET': ['SD', 'CG', 'CE'],
          'TRP': ['NE1', 'CD1', 'CE2'],
          'ILE': ['CG1', 'CD1', 'CG2'],
          }


flip_names = {'PHE': [('CE1', 'CE2'), ('CD1', 'CD2')],
              'ASP': [('OD1', 'OD2')],
              'GLU': [('OE1', 'OE2')],
              'ARG': [('NH1', 'NH2')],
              'TYR': [('CE1', 'CE2'), ('CD1', 'CD2')]
              }

flip_residues = ['PHE', 'ASP', 'GLU', 'ARG', 'TYR']

flip_sets = [{'OD1', 'OD2'}, {'CE1', 'CE2'}, {'NH2', 'NH1'}, {'OE1', 'OE2'}, {'CD1', 'CD2'}]

bb_type_dict = {'N_CA': ['N', 'H', 'CA'], 'C_O': ['C', 'O', 'CA'], 'SC': ['CA', 'N', 'C'],
                'PHI_PSI': ['CA', 'N', 'C']}

residue_sc_names = {'ALA': ['CB'], 'CYS': ['CB', 'SG'], 'ASP': ['CB', 'CG', 'OD1', 'OD2'],
                    'ASN': ['CB', 'CG', 'OD1', 'ND2'], 'VAL': ['CB', 'CG1', 'CG2'],
                    'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'], 'LEU': ['CB', 'CG', 'CD1', 'CD2'],
                    'HIS': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
                    'ILE': ['CB', 'CG2', 'CG1', 'CD1'], 'MET': ['CB', 'CG', 'SD', 'CE'], 'MSE': ['CB', 'CG', 'SE', 'CE'],
                    'TRP': ['CB', 'CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'],
                    'SER': ['CB', 'OG'], 'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
                    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'PRO': ['CB', 'CG', 'CD'],
                    'GLY': [], 'THR': ['CB', 'OG1', 'CG2'],
                    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                    'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
                    'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']}


coords_cols = ['c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
            'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
            'c_H2_x', 'c_H2_y', 'c_H2_z',
            'c_H3_x', 'c_H3_y', 'c_H3_z',
            'c_H4_x', 'c_H4_y', 'c_H4_z',
            'c_A1_x', 'c_A1_y', 'c_A1_z',
            'c_A2_x', 'c_A2_y', 'c_A2_z']


atom_types_sortkey = ['c_alkyl', 'c_aro', 'h_alkyl', 'h_aro', 'co', 's', 'n', 'o', 'h_pol', 'f']


flip_dict = dict()
flip_dict['ASP'] = dict(OD1='OD2')
flip_dict['GLU'] = dict(OE1='OE2')
flip_dict['TYR'] = dict(CE1='CE2', CD1='CD2')
flip_dict['PHE'] = dict(CE1='CE2', CD1='CD2')
flip_dict['VAL'] = dict(CG1='CG2')
flip_dict['LEU'] = dict(CD1='CD2')


cg_flip_dict = dict()
cg_flip_dict['ASP'] = {'OD1', 'OD2'}
cg_flip_dict['GLU'] = {'OE1', 'OE2'}
cg_flip_dict['TYR'] = {'CE1', 'CE2', 'CD1', 'CD2'}
cg_flip_dict['PHE'] = {'CE1', 'CE2', 'CD1', 'CD2'}
cg_flip_dict['VAL'] = {'CG1', 'CG2'}
cg_flip_dict['LEU'] = {'CD1', 'CD2'}
# cg_flip_dict['PRO'] = {'CB', 'CD'}


# cg_flip_defaultdict = defaultdict(dict)
# cg_flip_defaultdict['coo']['ASP'] = dict(OD1='OD2')
# cg_flip_defaultdict['coo']['GLU'] = dict(OE1='OE2')
# cg_flip_defaultdict['coo']['ANY'] = dict(O='OXT')
# cg_flip_defaultdict['phenol']['TYR'] = dict(CE1='CE2', CD1='CD2')
# cg_flip_defaultdict['ph']['PHE'] = dict(CE1='CE2', CD1='CD2')
# cg_flip_defaultdict['isopropyl']['VAL'] = dict(CG1='CG2')
# cg_flip_defaultdict['isopropyl']['LEU'] = dict(CD1='CD2')
# cg_flip_defaultdict['pro']['PRO'] = dict(CB='CD')

cgs_that_flip = ['coo', 'ph', 'phenol', 'isopropyl']
# cgs_that_flip = ['coo', 'ph', 'phenol', 'isopropyl', 'pro']


keep_H = dict()
keep_H['SER'] = 'HG'
keep_H['THR'] = 'HG1'
keep_H['CYS'] = 'HG'
keep_H['TYR'] = 'HH'

load_columns = [
    'resnum',
    'chain',
    'resname',
    'name',
    'c_x',
    'c_y',
    'c_z',
    'c_D_x',
    'c_D_y',
    'c_D_z',
    'c_H1_x',
    'c_H1_y',
    'c_H1_z',
    'c_H2_x',
    'c_H2_y',
    'c_H2_z',
    'c_H3_x',
    'c_H3_y',
    'c_H3_z',
    'c_H4_x',
    'c_H4_y',
    'c_H4_z',
    'c_A1_x',
    'c_A1_y',
    'c_A1_z',
    'c_A2_x',
    'c_A2_y',
    'c_A2_z',
    'atom_type_label',
    'rotamer',
    'phi',
    'psi',
    'dssp',
    'dssp_acc',
    'ABPLE',
    'ABPLE_3mer',
    'dssp_3mer',
    'contact_hb',
    'contact_wh',
    'contact_cc',
    'contact_so',
    'partners_hb',
    'partners_wh',
    'partners_cc',
    'partners_so',
    'rota',
    'CG',
    'hull_status',
    'dist_to_hull',
    'probe_name',
    'resname_rota',
    'centroid',
    'cluster_number',
    'C_score_bb_ind',
    'maxdist_to_centroid',
    'centroid_ABPLE_A',
    'centroid_ABPLE_B',
    'centroid_ABPLE_P',
    'centroid_ABPLE_E',
    'centroid_ABPLE_L',
    'centroid_dssp_S',
    'centroid_dssp_I',
    'centroid_dssp_C',
    'centroid_dssp_H',
    'centroid_dssp_E',
    'centroid_dssp_B',
    'centroid_dssp_G',
    'centroid_dssp_T',
    'centroid_hb_bb_ind',
    'C_score_ABPLE_A',
    'C_score_ABPLE_B',
    'C_score_ABPLE_E',
    'C_score_ABPLE_L',
    'C_score_ABPLE_P',
    'C_score_dssp_B',
    'C_score_dssp_C',
    'C_score_dssp_E',
    'C_score_dssp_G',
    'C_score_dssp_H',
    'C_score_dssp_I',
    'C_score_dssp_S',
    'C_score_dssp_T',
    'C_score_hb_ABPLE_A',
    'C_score_hb_ABPLE_B',
    'C_score_hb_ABPLE_E',
    'C_score_hb_ABPLE_L',
    'C_score_hb_ABPLE_P',
    'C_score_hb_bb_ind',
    'C_score_hb_dssp_B',
    'C_score_hb_dssp_C',
    'C_score_hb_dssp_E',
    'C_score_hb_dssp_G',
    'C_score_hb_dssp_H',
    'C_score_hb_dssp_I',
    'C_score_hb_dssp_S',
    'C_score_hb_dssp_T',
    'cluster_rank_ABPLE_A',
    'cluster_rank_ABPLE_B',
    'cluster_rank_ABPLE_E',
    'cluster_rank_ABPLE_L',
    'cluster_rank_ABPLE_P',
    'cluster_rank_dssp_B',
    'cluster_rank_dssp_C',
    'cluster_rank_dssp_E',
    'cluster_rank_dssp_G',
    'cluster_rank_dssp_H',
    'cluster_rank_dssp_I',
    'cluster_rank_dssp_S',
    'cluster_rank_dssp_T',
    'cluster_rank_hb_ABPLE_A',
    'cluster_rank_hb_ABPLE_B',
    'cluster_rank_hb_ABPLE_E',
    'cluster_rank_hb_ABPLE_L',
    'cluster_rank_hb_ABPLE_P',
    'cluster_rank_hb_bb_ind',
    'cluster_rank_hb_dssp_B',
    'cluster_rank_hb_dssp_C',
    'cluster_rank_hb_dssp_E',
    'cluster_rank_hb_dssp_G',
    'cluster_rank_hb_dssp_H',
    'cluster_rank_hb_dssp_I',
    'cluster_rank_hb_dssp_S',
    'cluster_rank_hb_dssp_T',
    'CG_rep_fine_ABPLE',
    'sc_rep_fine_ABPLE',
    'sc_rep_fine_dssp',
    'CG_rep_fine_dssp',
    'CG_rep_fine_bb_ind',
    'sc_rep_fine_bb_ind',
    'contact_type',
    'hbond',
    'is_acceptor',
    'is_donor',
]


hb_cols = ['c_D_x', 'c_D_y', 'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z', 'c_H2_x',
           'c_H2_y', 'c_H2_z', 'c_H3_x', 'c_H3_y', 'c_H3_z', 'c_H4_x', 'c_H4_y',
           'c_H4_z', 'c_A1_x', 'c_A1_y', 'c_A1_z', 'c_A2_x', 'c_A2_y', 'c_A2_z']


cgs = ['bb_cco', 'bb_cnh', 'coo', 'csc', 'conh2', 'hid', 'hie', 'hip', 'coh', 
       'ccoh', 'gn', 'csh', 'phenol', 'indole', 'ccn', 'pro', 'ph', 'isopropyl']  