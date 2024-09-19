import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2


inpath_prody_dir = '/wynton/scratch/nick.polizzi/db_2p8A_0p35rfree_reduced_reps_biounits_prody/'
comb_dir = '/wynton/scratch/nick.polizzi/comb/'
outdir = '/wynton/scratch/nick.polizzi/comb/'
path_to_probe_paths = '/wynton/home/degradolab/nick.polizzi/Combs2/designs/probe_paths_list_wynton.pkl'


def main():

    par = argparse.ArgumentParser()
    par.add_argument('--comb_paths', default=None, help='path to COMB output files')
    par.add_argument('--probe_paths', default=path_to_probe_paths, help='input path for pkl file containing Probe file paths')
    par.add_argument('--o', default=None, help='output path for COMB files')
    par.add_argument('--cg', required=True, help='Chemical group to be COMBed, found in cg_dicts')
    args = par.parse_args()

    cg_dict = combs2.parse.comb.cg_dicts[args.cg]
    _outdir = args.o or outdir + args.cg + '_nr/'
    _path_to_probe_paths = args.probe_paths
    _path_to_comb_output = args.comb_paths or comb_dir + args.cg + '/'

    combs2.parse.nr.run_nr(cg_dict, _path_to_probe_paths, _outdir, _path_to_comb_output)


if __name__ == '__main__':
    main()