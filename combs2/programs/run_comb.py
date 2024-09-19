import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2
from combs2.design.functions import add_slash
import traceback


inpath_prody_dir = '/wynton/scratch/nick.polizzi/db_2p8A_0p35rfree_reduced_reps_biounits_prody/'
outdir = '/wynton/scratch/nick.polizzi/comb/'
path_to_probe_paths = '/wynton/home/degradolab/nick.polizzi/Combs2/designs/probe_paths_list_wynton.pkl'
inpath_rotamer_dir = '/wynton/scratch/nick.polizzi/db_2p8A_0p35rfree_reduced_reps_biounits_pdb_rotamers/'


def main():

    par = argparse.ArgumentParser()
    par.add_argument('--prody_paths', default=inpath_prody_dir, help='input path for ProDy files')
    par.add_argument('--rotamer_paths', default=inpath_rotamer_dir, help='input path for rotamer files')
    par.add_argument('--o', default=None, help='output path for COMB files')
    par.add_argument('--probe_paths', default=path_to_probe_paths, help='input path for pkl file containing Probe file paths')
    par.add_argument('--cg', required=True, help='Chemical group to be COMBed, found in cg_dicts')
    par.add_argument('--i_first', required=True, help='first index in Probe file paths to be COMBed')
    par.add_argument('--i_last', default=None, help='first index in Probe file paths to be COMBed')
    par.add_argument('--his', default=None, help='If COMBing HIS residue, filters by protonation state.')
    args = par.parse_args()

    cg_dict = combs2.parse.comb.cg_dicts[args.cg]
    ind_first = int(args.i_first) - 1 # account for task id starting at 1.
    ind_last = args.i_last
    _inpath_prody_dir = add_slash(args.prody_paths)
    _inpath_rotamer_dir = add_slash(args.rotamer_paths)
    _outdir = args.o or outdir + args.cg + '/'
    _outdir = add_slash(_outdir)
    _path_to_probe_paths = args.probe_paths
    his_option = args.his

    if ind_last is not None:  # Run through multiple pdbs
        for ind in range(ind_first, int(ind_last) + 1):
            try:
                combs2.parse.comb.run_comb(cg_dict, _inpath_prody_dir,
                                           _inpath_rotamer_dir,
                                           _outdir, _path_to_probe_paths, ind, his_option)
            except Exception:
                print('Exception at index', ind)
                traceback.print_exc()

    else:  # Run through only one pdb
        try:
            combs2.parse.comb.run_comb(cg_dict, _inpath_prody_dir, _inpath_rotamer_dir,
                                       _outdir, _path_to_probe_paths, ind_first, his_option)
        except Exception:
            print('Exception at index', ind_first)
            traceback.print_exc()


if __name__ == '__main__':
    main()