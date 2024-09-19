#!/bin/bash

# parameters
path_to_lookup=~/Projects/design/Combs2/combs2/programs/run_lookup_ligand.py
path_to_lig_txt=ligand.txt
path_to_pdb=output/1_pose1_no_CG.pdb
path_to_db=/Volumes/disk1/Combs/probe_Qbits_2p8/20211005/vdMs/
path_to_lig_params=../HPC_scripts/superpose_ligand/GG2.params
outpath=output/
path_to_probe=~/Projects/design/Combs2/combs2/programs/probe

# before running this script, activate combs python environment by:
# > conda activate env_combs

# run the program
python $path_to_lookup --lig=$path_to_lig_txt --pdb=$path_to_pdb --db=$path_to_db --o=$outpath --lig_params=$path_to_lig_params --probe=$path_to_probe
