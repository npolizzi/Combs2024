import sys
import argparse
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2
from combs2.design.functions import add_slash


outdir = '/wynton/scratch/nick.polizzi/comb/'


def main():


    par = argparse.ArgumentParser()
    par.add_argument('--i', required=True, help='index of 20 aa to be clustered. Must be 0 through 19.')
    par.add_argument('--comb_nr_clu_paths', default=None, help='path to non-redundant clustered COMB output files')
    par.add_argument('--o', default=None, help='output path for clustered nrCOMB files with representatives')
    par.add_argument('--cg', required=True, help='Chemical group to be COMBed, found in cg_dicts')

    args = par.parse_args()

    cg_dict = combs2.parse.comb.cg_dicts[args.cg]
    _outdir = args.o or outdir + args.cg + '_nr_cluster_reps/'
    _outdir = add_slash(_outdir)
    _path_to_comb_nr_output = args.comb_nr_clu_paths or outdir + args.cg + '_nr_cluster/'
    _path_to_comb_nr_output = add_slash(_path_to_comb_nr_output)
    ind = int(args.i)


    combs2.parse.reps.run_reps(cg_dict, _path_to_comb_nr_output, _outdir, ind)


if __name__ == '__main__':
    main()