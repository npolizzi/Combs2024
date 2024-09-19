import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2
from combs2.design.functions import add_slash
from combs2.design.constants import inv_one_letter_code


path_to_database='/wynton/scratch/nick.polizzi/vdMs/'
outdir='/wynton/scratch/nick.polizzi/lig_vdMs/'


def main():
    par = argparse.ArgumentParser()
    par.add_argument('--d', default=None, help='path to vdM database')
    par.add_argument('--lig_txt', default=None, help='path to ligand txt file')
    par.add_argument('--lig_pdb', default=None, help='path to ligand pdb file')
    par.add_argument('--lig_params', default=None, help='path to ligand params file')
    par.add_argument('--lig_resname', default=None, help='ligand residue name')
    par.add_argument('--remove_atoms_from_hb_dict', default=None, help='txt file listing atom names to remove from the '
                                                                       'auto-generated H-bonding dictionary '
                                                                       'of the ligand')
    par.add_argument('--hb_only', default=False, help='Use only H-bonding vdMs')
    par.add_argument('--superpose_type', default='enriched', help='category of vdMs on which to superpose ligand. '
                                                                  'examples: all, enriched, top_x (x is an integer)')
    par.add_argument('--superpose_resnames_file', default=None, help='Use only vdMs of these residue names')
    par.add_argument('--o', default=None, help='output path for ligand-superimposed vdM files')
    par.add_argument('--i', default=None, help='index for parallel computing')
    args = par.parse_args()

    _path_to_vdms = args.d or path_to_database
    _path_to_vdms = add_slash(_path_to_vdms)
    lig_txt_path = args.lig_txt
    lig_pdb_path = args.lig_pdb
    lig_params_path = args.lig_params
    lig_resname = args.lig_resname

    remove_hb_atoms = []
    if args.remove_atoms_from_hb_dict is not None:
        with open(args.remove_atoms_from_hb_dict) as infile:
            for line in infile:
                spl = line.strip().split()
                remove_hb_atoms.extend(spl)

    if args.hb_only in [1, 'True', 'T', 'TRUE']:
        hbonly = True
    else:
        hbonly = False
    superpose_type = args.superpose_type

    superpose_resnames_dict = dict()
    if args.superpose_resnames_file is not None:
        superpose_resnames_dict = dict()
        with open(args.superpose_resnames_file) as infile:
            for line in infile:
                spl = line.strip().split()
                superpose_resnames_dict[spl[0]] = [inv_one_letter_code[aa] for aa in set(spl[1])]

    _outdir = args.o or outdir + lig_resname + '/'

    kwargs = dict(
        num_cpus=1,  # one cpu per function call
        path_to_outdir=_outdir,
        path_to_lig_txt=lig_txt_path,
        path_to_lig_pdb=lig_pdb_path,
        path_to_lig_params=lig_params_path,
        lig_resname=lig_resname,
        remove_from_hbond_dict=remove_hb_atoms,
        hb_only=hbonly,
        superpose_type=superpose_type,  # or top_30, top_60, top_200, top_all_enriched, all
        resnames_for_superpose=superpose_resnames_dict,
        path_to_dataframe_files=_path_to_vdms,
        dataframe_file_extension='.parquet.gzip',
        ind=args.i,
    )

    s = combs2.design.superpose_ligand.SuperposeLig(**kwargs)
    s.setup()
    s.run()


if __name__ == '__main__':
    main()