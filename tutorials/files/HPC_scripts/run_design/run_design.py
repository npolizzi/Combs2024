import sys
import os
_dir = os.path.dirname(__file__)
path_to_combs2 = '/'.join(_dir.split('/')[:-2])
sys.path.insert(0, path_to_combs2)
import combs2
import prody as pr


path_to_glys = '/wynton/home/degradolab/nick.polizzi/Combs2/tutorials/files/HPC_scripts/run_design/parametric_bundles/param_gly/'
path_to_alas = '/wynton/home/degradolab/nick.polizzi/Combs2/tutorials/files/HPC_scripts/run_design/parametric_bundles/param_ala/'
outpath = '/wynton/scratch/nick.polizzi/apixaban/parametric_bundles/output/'
path_to_ligand_database = '/wynton/scratch/nick.polizzi/comb/lig_vdMs/apixaban/GG2/enriched/'
path_to_database = '/wynton/scratch/nick.polizzi/comb/vdMs/'
path_to_ligand_txt = '/wynton/home/degradolab/nick.polizzi/Combs2/tutorials/files/HPC_scripts/run_design/ligand.txt'
path_to_csts = '/wynton/home/degradolab/nick.polizzi/Combs2/tutorials/files/HPC_scripts/run_design/lig_csts.txt'
outpath_sample = outpath
cgs = ['bb_cco', 'conh2']
cg_max_dists = dict(conh2=0.8, bb_cco=0.8)


def main():
    files = sorted([fp for fp in os.listdir(path_to_glys) if fp[0] != '.'])
    ind = int(sys.argv[1]) - 1
    gly_file = files[ind]
    pdb_gly_path = path_to_glys + gly_file
    ala_file = gly_file.replace('gly', 'ala')
    pdb_ala_path = path_to_alas + ala_file
    run_comb(pdb_gly_path, pdb_ala_path, outpath + gly_file.split('.')[0] + '/')


def run_comb(pdb_gly_path, pdb_ala_path, outpath):
    pdb_gly = pr.parsePDB(pdb_gly_path)
    pdb_ala = pr.parsePDB(pdb_ala_path)
    template = combs2.design.template.Template(pdb_gly)
    template.set_alpha_hull(pdb_ala, alpha=9)

    combs2.design.functions.write_resfile(template, CGs=cgs,
                                          outpath=outpath,
                                          filename='resfile', tag='',
                                          resindices=None, segs_chains_resnums=None,
                                          pikaa_dict=None, bb_dep=1,
                                          use_enriched_vdMs=True, CA_burial_distance=None, exclude_exposed=False,
                                          exclude_intermed=False,
                                          exclude_buried=False, top_exposed=3, top_intermed=None, top_buried=None,
                                          alpha_hull_radius=9,
                                          use_propensities=True,
                                          propensity_threshold=0.9, use_abple=True, use_dssp=False,
                                          path_to_pdb_for_dssp=None,
                                          allowed_exposed='KRDENQSTMAGP', allowed_intermed='NQSTCMAGPVIL',
                                          allowed_buried='AGSTMCPVILHFWY',
                                          hb_only_residues='', all_contact_residues='GHFWY')
    s = combs2.design._sample.Sample(**dict(path_to_resfile=outpath + 'resfile.txt',
                                            path_to_database=path_to_database))
    s.read_resfile()
    s.load_vdms(template, filter_by_phi_psi=False, run_parallel=False)
    s.load_ligands(template, use_ligs_of_loaded_vdms_only=True,
                        frac_non_hb_heavy_buried=0.5,
                        path_to_ligand_database=path_to_ligand_database,
                        hull_tolerance=0,
                        run_parallel=False)
    s.set_ligand_vdm_correspondence(path_to_ligand_txt)
    s.set_constraints(path_to_csts)
    s.set_cg_neighbors(max_dist_criterion=True, cg_max_dists=cg_max_dists)
    print('Finding neighbors of Ligands CGs...')
    s.find_ligand_cg_neighbors()
    print('Finding poses...')
    s.find_poses(template)
    print('Scoring poses...')
    s.score_poses(template, bbdep=True, use_hb_scores=False,
                  return_top_scoring_vdMs_only=False)
    print('Saving Sample object...')
    param_name = pdb_gly_path.split('/')[-1].split('.')[0]
    outfile_name = 'sample_' + param_name + '.pkl'
    s.save(outpath=outpath_sample, filename=outfile_name)
    print('Getting top poses...')
    poses = s.get_top_poses(top=50)
    print('Finding buried unsatisfied polar atoms of top poses...')
    s.set_buried_unsatisfied(poses, template)
    print('Printing top poses...')
    [pose.print_opt_pdb(template, outdir=outpath, tag='_' + param_name,
                        include_CG=True, label_vdM_segment_X=True)
     for pose in poses]
    [pose.print_to_energy_table(outdir=outpath, filename_tag='_' + param_name)
     for pose in poses]


if __name__ == '__main__':
    main()
