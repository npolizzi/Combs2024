import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2


path_to_probe = combs2.design.probe._probe


def main():

    par = argparse.ArgumentParser()
    par.add_argument('--pdb', required=True, help='path to PDB file')
    par.add_argument('--db', required=True, help='path to vdM database')
    par.add_argument('--probe', default=path_to_probe, help='path to Probe program')
    par.add_argument('--o', required=True, help='output path')
    par.add_argument('--lig', required=True, help='path to ligand txt file')
    par.add_argument('--lig_params', default=None, help='path to ligand params file')
    par.add_argument('--lig_atom_types', default=None, help='path to ligand atom-type dictionary')
    par.add_argument('--filename', default='', help='output filename')
    par.add_argument('--skip', default=6, help='skipping number for defining vdMs in input PDB (default: %(default)s)')
    par.add_argument('--dist_cut', default=0.5, help='rmsd cutoff to centroid '
                                                     'for inclusion of vdM in cluster (default: %(default)s)')
    par.add_argument('--max_dist_cut', default=0.8, help='maximum atom distance cutoff to centroid '
                                                     'for inclusion of vdM in cluster (default: %(default)s)')
    par.add_argument('--include_mc_mc', default=False, help='include mainchain-mainchain vdMs in analysis (default: %(default)s)')
    par.add_argument('--seg1', default='', help='segment1 name for Probe')
    par.add_argument('--seg2', default='', help='segment2 name for Probe')
    par.add_argument('--chain1', default='', help='chain1 name for Probe')
    par.add_argument('--chain2', default='', help='chain2 name for Probe')
    par.add_argument('--resnum1', default='', help='resnum1 name for Probe')
    par.add_argument('--resnum2', default='', help='resnum2 name for Probe')
    par.add_argument('--sel_criteria', default='NOT METAL', help='contact selection criteria for Probe (default: %(default)s)')
    par.add_argument('--max_bonded', default=4, help='max number of bonds between atoms '
                                                     'before Probe detects interaction (default: %(default)s)')
    par.add_argument('--explicit_H', default=False,
                     help='tell Probe to explicitly use hydrogens (default: %(default)s)')
    par.add_argument('--wc', default=True,
                     help='parse wide contacts from Probe (default: %(default)s)')
    par.add_argument('--strict', action='store_true', help='set distance criteria to strict '
                                                            'matching of vdM clusters (max atom-pair '
                                                            'distance = 0.65 A)')
    args = par.parse_args()

    _outdir = args.o
    filename = args.filename
    _path_to_probe = args.probe or path_to_probe
    _path_to_pdb = args.pdb
    _path_to_database = args.db
    path_to_ligand_file = args.lig
    path_to_ligand_params = args.lig_params
    path_to_ligand_atypes = args.lig_atom_types
    if args.strict:
        distcut = 0.65
        maxdistcut = 0.65
    else:
        distcut = float(args.dist_cut)
        maxdistcut = float(args.max_dist_cut)
    probe_segname1 = args.seg1
    probe_segname2 = args.seg2
    probe_chain1 = args.chain1
    probe_chain2 = args.chain2
    probe_resnum1 = args.resnum1
    probe_resnum2 = args.resnum2
    sel_criteria = args.sel_criteria
    max_bonded = int(args.max_bonded)
    explicit_H = args.explicit_H
    if explicit_H in [False, 'False', 0, '0', 'F']:
        explicit_H = False
    else:
        explicit_H = True
    wc = args.wc
    if wc in [False, 'False', 0, '0', 'F']:
        wc = False
    else:
        wc = True
    include_mc_mc = args.include_mc_mc
    if include_mc_mc in [False, 'False', 0, '0', 'F']:
        include_mc_mc = False
    else:
        include_mc_mc = True


    combs2.validation.lookup.run_lookup_ligand(_path_to_pdb, _path_to_database, _outdir, path_to_ligand_file,
                                               path_to_ligand_params=path_to_ligand_params,
                                               ligand_atom_type_dict=path_to_ligand_atypes,
                                               filename=filename, distance_cutoff=distcut,
                                               probe_segname1=probe_segname1, probe_chain1=probe_chain1,
                                               probe_resnum1=probe_resnum1,
                                               probe_segname2=probe_segname2, probe_chain2=probe_chain2,
                                               probe_resnum2=probe_resnum2,
                                               probe_sel_criteria=sel_criteria, maxbonded=max_bonded,
                                               explicit_H=explicit_H,
                                               include_mc_mc=include_mc_mc, include_wc=wc, strict=args.strict,
                                               path_to_probe=_path_to_probe, max_distance_cutoff=maxdistcut)


if __name__ == '__main__':
    main()