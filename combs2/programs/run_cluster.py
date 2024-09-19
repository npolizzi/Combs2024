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
    par.add_argument('--comb_nr_paths', default=None, help='path to non-redundant COMB output files')
    par.add_argument('--o', default=None, help='output path for clustered nrCOMB files')
    par.add_argument('--cg', required=True, help='Chemical group to be COMBed, found in cg_dicts')
    par.add_argument('--rmsdcut', default=0.65, help='rmsd cutoff that defines clusters')
    par.add_argument('--min_cluster_size', default=2, help='size of clusters less than which all entries are singletons.')
    par.add_argument('--cgh_name_path', default=None, help='path to txt file containing lines of cg resnames and '
                                                           'H names to include in clustering')
    args = par.parse_args()

    cg_dict = combs2.parse.comb.cg_dicts[args.cg]
    _outdir = args.o or outdir + args.cg + '_nr_cluster/'
    _outdir = add_slash(_outdir)
    _path_to_comb_nr_output = args.comb_nr_paths or outdir + args.cg + '_nr/'
    _path_to_comb_nr_output = add_slash(_path_to_comb_nr_output)
    ind = int(args.i)
    rc = args.rmsdcut
    cghnp = args.cgh_name_path
    min_clu_size = int(args.min_cluster_size)

    combs2.parse.cluster.run_cluster(cg_dict, _path_to_comb_nr_output, _outdir, ind,
                                     rmsd_cutoff=rc, path_to_cg_H_names=cghnp, min_cluster_size=min_clu_size)


if __name__ == '__main__':
    main()