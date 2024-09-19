import combs2
import prody as pr
import pickle
import os


"""

This script runs recursive COMBS to search for a biotin binding site in 
streptavidin (pdb 3ry2) using a limited set of amino acids and residue positions. 
The script takes about 1 hr to run on a single cpu core using about 3 GB of RAM.
It produces ranked PDB output files, with the top-ranked file(s) closely matching
the binding site position and sequence in the co-crystal structure.  It also 
produces a csv file with each row being a separate output PDB and with columns such 
as vdM "energies" (lower is better) and other attributes about the binding site, 
e.g. fraction of atoms that are ligand buried and total number of buried, 
"unsatisfied" polar atoms (includes ligand + sidechains).

It can be run from the command line by typing:

    conda activate env_combs  # activate the COMBS conda environment
    python run_design.py      # run the script

"""


# Change input_dir and output_dir as desired
input_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
print('Input directory:', input_dir)
output_dir = input_dir + 'output/'
print('Output directory:', output_dir)

def main():

    gly_pdb_name = '3ry2_A_H_gly.pdb'
    ala_pdb_name = '3ry2_A_H_ala.pdb'
    pdb_gly_path = input_dir + gly_pdb_name
    pdb_ala_path = input_dir + ala_pdb_name

    pdb_gly = pr.parsePDB(pdb_gly_path)
    pdb_ala = pr.parsePDB(pdb_ala_path)

    template = combs2.design.template.Template(pdb_gly)
    template.set_alpha_hull(pdb_ala, alpha=9)

    ## Can save the template for later use:
    # template.save(output_dir, 'template.pkl')

    ## If you want to design only on buried or intermediate residues:
    # exposed, intermed, buried = combs2.design.convex_hull.partition_res_by_burial(pdb_ala, alpha=9, 
    #                                                             assign_intermediate_by_distance=True,
    #                                                             distance_threshold=-1.0,)
    # designable_resindices = intermed | buried
    # resnums = pdb_ala.select('name CA and resindex ' + ' '.join(str(j) 
    #                           for j in designable_resindices)).getResnums()

    # Choosing the residues to design on by hand:
    resnums = [23, 25, 27, 43, 45, 49, 88, 90, 128]
    chains = ['A'] * len(resnums)
    segs = [''] * len(resnums)
    segs_chains_resnums = zip(segs, chains, resnums)

    # Choose the chemical groups for which to find vdMs (must be defined in the ligand.txt file):
    CGs = ['bb_cco', 'bb_cnh', 'coo', 'csc', 'conh2']

    # Choose the chemical groups and chemical-group numbers on which to superimpose the ligand
    # (can be a subset of the chemical groups defined in the ligand.txt file):
    superpose_to_cgs = dict(bb_cco=[1,2], bb_cnh=[1,2,3,4], coo=[1,2], csc=[1,2])

    # Choose allowed AAs for each chemical group in the ligand.txt file:
    allowed_exposed = dict(conh2='DNTSYLG',
                           bb_cnh='DNTSYLG',
                           coo='NTSYLG', # exclude D
                           bb_cco='NTSYLG', # exclude D
                           csc='NTSYLG') # exclude D

    allowed_intermed = dict(conh2='DNTSYLG',
                           bb_cnh='DNTSYLG',
                           coo='NTSYLG', # exclude D
                           bb_cco='NTSYLG', # exclude D
                           csc='NTSYLG') # exclude D

    allowed_buried = dict(conh2='DNTSYLG',
                           bb_cnh='DNTSYLG',
                           coo='NTSYLG', # exclude D
                           bb_cco='NTSYLG', # exclude D
                           csc='NTSYLG') # exclude D

    # Choose which AAs are h-bonding only vs all-contacts for each chemical-group:
    hb_only_residues = dict(conh2='DNTSY',
                           bb_cnh='DNTSY',
                           coo='NTSY', 
                           bb_cco='NTSY', 
                           csc='NTSY') 
    
    all_contact_residues = dict(conh2='GL',
                                bb_cnh='GL',
                                coo='GL', 
                                bb_cco='GL', 
                                csc='GL') 

    outpath_resfile = input_dir

    combs2.design.functions.write_resfile(template,
                                          CGs=CGs,
                                          outpath=outpath_resfile,
                                          filename='resfile',
                                          tag='',
                                          resindices=None,
                                          segs_chains_resnums=segs_chains_resnums, 
                                          pikaa_dict=None,
                                          bb_dep=0,
                                          use_enriched_vdMs=True,
                                          CA_burial_distance=None,
                                          exclude_exposed=False,
                                          exclude_intermed=False,
                                          exclude_buried=False,
                                          top_exposed=None,
                                          top_intermed=None,
                                          top_buried=None,
                                          alpha_hull_radius=9,
                                          use_propensities=True,
                                          propensity_threshold=0.5,
                                          use_abple=False,
                                          use_dssp=False,
                                          path_to_pdb_for_dssp=None,
                                          allowed_exposed=allowed_exposed,
                                          allowed_intermed=allowed_intermed,
                                          allowed_buried=allowed_buried,
                                          hb_only_residues=hb_only_residues,
                                          all_contact_residues=all_contact_residues)

    #path to COMBS residue file for design
    path_to_resfile= input_dir + 'resfile.txt'

    # path to vdM databases
    path_to_database='/Volumes/disk1/Combs2/database/20211005/vdMs/'

    s = combs2.design._sample.Sample(**dict(path_to_resfile=path_to_resfile,
                                            path_to_database=path_to_database))

    # Read the resfile into the Sample object
    s.read_resfile()

    path_to_ligand_file = input_dir + 'ligand.txt'
    s.set_ligand_vdm_correspondence(path_to_ligand_file)

    rmsd_dict = {cg: 0.7 for cg in s.ligand_vdm_correspondence.CG_type.unique()}

    ## Set CG scoring weights if desired:
    # CG_weights = {cg: 1 for cg in CGs}
    # CG_weights['bb_cco'] = 2 # increase scoring weight for bb_cco CG
    # s.set_cg_weights(CG_weights)

    # Set ligand constraints (Comment out if not using constraints)
    path_to_constraint_file = input_dir + 'lig_csts.txt'
    s.set_constraints(path_to_constraint_file)

    paths_to_lig_pdbs = [input_dir + 'conformers/BTN_0001.pdb',] # can list multiple pdbs here
    path_to_lig_params = input_dir + 'conformers/BTN.params'
    lig_resname = 'BTN'
    ignore_rmsd_column_for_extra_lig_interactions = ()
    # The variable directly above tells COMBS to ignore any no_rmsd flags in the ligand.txt file
    # This is sometimes necessary when the chemical group has fewer atoms in the ligand.txt
    # file than in the pre-computed Neighbors files.  For example, for imidazole,
    # the atom names are CG, ND1, CD2, CE1, NE2, but you might only use 3 of these atoms 
    # in the ligand.txt file (e.g. CG, ND1, CE1).  In this case, you can set ignore_no_rmsd
    # to a list with items (chemical-group type, chemical-group number) whose rmsd flag should be
    # ignored.
    # Example:
    #   ignore_rmsd_column_for_extra_lig_interactions = [k for k in s.ligand_vdm_correspondence_grs.groups.keys() 
    #                                                    if 'hid' in k or 'hie' in k]

    s.load_ligand_conformers(paths_to_lig_pdbs=paths_to_lig_pdbs, path_to_lig_params=path_to_lig_params,
                            lig_resname=lig_resname, remove_atom_from_hb_dict=None, ligand_dataframes=None)

    s.load_vdms_ligands_low_mem(template, path_to_database=path_to_database, 
                                superpose_to_cgs=superpose_to_cgs, 
                                residue_chunk_size=10, 
                                lig_chunk_size=10000,
                                frac_non_hb_heavy_buried=0.8, # Change for different ligands
                                hull_tolerance=0,
                                filter_by_phi_psi=False, 
                                filter_by_phi_psi_exclude_sc=True, 
                                distance_metric='rmsd',
                                cg_rmsds=rmsd_dict, 
                                cg_max_dists=None, 
                                max_dist_criterion=False,
                                ignore_rmsd_column=ignore_rmsd_column_for_extra_lig_interactions,
                                vdW_tolerance_vdms=0.1,
                                vdW_tolerance_ligands=0.0)

    s.set_cg_neighbors(cg_rmsds=rmsd_dict)

    print('Finding ligand neighbors')
    s.find_ligand_cg_neighbors()

    print('Finding poses...')
    ## For heuristic ranking of poses, you can define weights for 
    ## "Ligand coverage number" in ligand.txt file (last column).
    ## The commented lines below would weight the bb_cco group (Lig cov # == 3)
    ## 10x higher than the other Lig cov #s.
    # vals = [1, 1, 10, 1, 1]
    # weights = {i: vals[i-1] for i in range(1, 6)}
    s.find_poses(template, only_top_percent=0.1, min_poses=100, max_poses=150, 
                 filter_ligands_by_cluster=True,
                 lig_rmsd_cutoff=1.5, min_ligands_per_cluster=3, 
                 top_percent_per_cluster=0.25, max_ligands_per_cluster=8, 
                 max_ligands_to_search=1000,
                 vdW_tolerance=0.1,)
                # weight_dict=weights)

    print('Scoring poses...')
    s.score_poses(template, bbdep=True, use_hb_scores=True,
                  return_top_scoring_vdMs_only=False, store_pairwise=False, 
                  force_MC=False, C_score_threshold=-0.5, 
                  compute_pairwise_contacts=True, tamp_by_distance=True, 
                  log_logistic=False, gaussian=True, exponential=False, 
                  pairwise_contact_weight=0.5,
                  vdW_tolerance=0.1,
                  ignore_rmsd_column=ignore_rmsd_column_for_extra_lig_interactions)

    ## Can save the sample object for later use if desired:
    # s.save(outpath=output_dir, filename='sample.pkl', minimal_info_and_poses=True)
    running_recursive = False

    ####################### Run Recursive COMBS ######################
    running_recursive = True
    poses = s.get_top_poses(top=100, top_from_pose_group=True)
    s.set_buried_unsatisfied(poses, template)

    poses = s.run_recursive_vdM_search(poses, template, max_iter=4, bbdep=True, 
                                use_hb_scores=True, C_score_threshold=-0.5,
                                return_top_scoring_vdMs_only=False, 
                                store_pairwise=True, force_MC=False,
                                force_DEE=False, DEE_to_MC_switch_number=1000, 
                                compute_pairwise_contacts=True,
                                tamp_by_distance=True, pair_nbr_distance=0.7, 
                                exponential=False,
                                log_logistic=False, gaussian=True, relu=False,
                                knn_contacts=True, contact_distance_metric='rmsd',
                                use_same_rotamer_for_pairwise_contacts=True, 
                                use_same_rotamer_for_lig_contacts=True,
                                ignore_rmsd_column=ignore_rmsd_column_for_extra_lig_interactions,
                                pairwise_contact_weight=0.5,
                                filter_by_phi_psi=False, 
                                filter_by_phi_psi_exclude_sc=True,
                                specific_seg_chain_resnums_only=None,
                                rmsd=0.6,
                                maxdist=0.7,
                                distance_metric='rmsd',
                                allowed_amino_acids='hb_set', # 'hb_set' = 'ADEGHKMNQRSTWY' 
                                allowed_seg_chain_resnums=None, # if None, allow all residues in protein to be "designable"
                                burial_threshold=0.5, # distance (Angstroms) of atom to protein surface
                                use_optimum_vdms_only=True,
                                freeze_optimum_vdms=True,
                                outer_shell_score_weight=0.5,
                                vdW_tolerance=0.1,
                                )
    ##########################################################################
    # Comment out the above block if recursive COMBS is not desired

    if not running_recursive:
        poses = None

    poses = s.get_top_poses(poses=poses, top=30, top_from_pose_group=True)
    s.set_buried_unsatisfied(poses, template, burial_threshold=0.5, 
                             exclude_mc_hb=False)

    for pose in poses:
        pose.print_opt_pdb(template, outdir=output_dir + 'opt_poses/',
                           include_CG=True, label_vdM_segment_X=True)

    poses[0].print_opt_pdb(template, outdir=output_dir + 'opt_poses_no_CGs/',
                           include_CG=False, label_vdM_segment_X=False, tag='_no_CG')

    for pose in poses:
        pose.print_to_energy_table(outdir=output_dir, filename_tag='_' + gly_pdb_name)
        pose.cleanup()

    # # Can save the top poses for later use if desired:
    # with open(output_dir + 'top_poses.pkl', 'wb') as outfile:
    #     pickle.dump(poses, outfile)


if __name__ == '__main__':
    main()



