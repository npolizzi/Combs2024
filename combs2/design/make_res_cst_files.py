## This is an example of code to generate resfiles and cst files

import os
import sys
sys.path.append('/Users/npolizzi/Projects/combs/src/')
import combs
import prody as pr
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

os.mkdir('/Users/npolizzi/Projects/combs/src/runs/apixaban/20190317/utopian_pts/apx3/top')
pdb = pr.parsePDB('/Users/npolizzi/Projects/combs/src/runs/apixaban/20190317/utopian_pts/apx3/apx3_w_res_0001.pdb')
pdb_ala = pr.parsePDB('/Users/npolizzi/Projects/combs/src/runs/apixaban/20190313/ala/apx_traj_cent_1_trans_ala_0001.pdb')
cons_set = set(list(zip(pdb.select('not resname GLY and name CA').getResnames(), pdb.select('not resname GLY and name CA').getResnums())))
d = dict(ASP=-1, GLU=-1, LYS=1, ARG=1)
constrained_rns = []
constrained_rns_vals = []
for resname, resnum in cons_set:
    constrained_rns.append(resnum)
    if resname in d.keys():
        constrained_rns_vals.append(d[resname])
    else:
        constrained_rns_vals.append(0)
top = combs.apps.topology.Topology(**dict(constrained_rns=constrained_rns, constrained_rns_vals=constrained_rns_vals))
top.load_pdb(pdb_ala, selection='resnum 1to40 44to78 83to118 126to160')
top.load_pdb_ala(pdb_ala, selection='resnum 1to40 44to78 83to118 126to160')
top.set_topologies(outdir='/Users/npolizzi/Projects/combs/src/runs/apixaban/20190317/utopian_pts/apx3/top/')
top.set_surface_res()
top.set_contacts()
top.run_mc(num_iterations=100000)
top.find_pareto_front()
top.find_nearest_utopian_pt(weight_en_f=0.75, weight_seq_rep_len=0.5)
top.map_seq_resnums_to_pdb(pdb_ala)
top.set_charge_groups(top.seqs[top.nearest_utopian_pt])
top.print_charge_groups()
top.save_sequence(top.seqs[top.nearest_utopian_pt], outdir='/Users/npolizzi/Projects/combs/src/runs/apixaban/20190317/utopian_pts/apx3/top/')
with open('resfile.txt', 'w') as resfile:
    resfile.write('start \n')
    for aa, resnum in cons_set:
        aa_single = combs.apps.terms.one_letter_code[aa]
        resfile.write(str(resnum) + ' A NATRO \n')
dssp = combs.apps.terms.parse_dssp('/Users/npolizzi/Projects/combs/src/runs/apixaban/20190313/dssp/apx_traj_cent_1_trans_gly_0001.dssp')
def rec_dd():
    """returns a recursive dictionary"""
    return defaultdict(rec_dd)
with open('/Users/npolizzi/Projects/combs/database/master_biounit/biounits/ss_burial_propensity_label_dict_20180902.pkl', 'rb') as infile:
    ss_bur_prop_label_dict = pickle.load(infile)
polar_dict = {}
polar_dict[-1] = set('ED')
polar_dict[1] = set('KR')
polar_dict[0] = polar_dict[-1] | polar_dict[1]

with open('top/surface_sequence.txt','r') as infile:
    infile.readline()
    surface_seq = {}
    for line in infile:
        line = line.split()
        surface_seq[int(line[0])] = int(line[1])

with open('resfile.txt', 'r') as infile:
    resfile = {}
    infile.readline()
    for line in infile:
        try:
            line = line.strip().split()
            resfile[(int(line[0]), line[1])] = line[2:]
        except:
            pass

pdb_ala = pr.parsePDB('/Users/npolizzi/Projects/combs/src/runs/apixaban/20190313/ala/apx_traj_cent_1_trans_ala_0001.pdb')
resind_exp, resind_int, resind_bur = combs.apps.convex_hull.partition_res_by_burial(pdb_ala, alpha=9)
resnum_exp = set(pdb_ala.select('resindex ' + ' '.join([str(r) for r in resind_exp])).getResnums())
resnum_int = set(pdb_ala.select('resindex ' + ' '.join([str(r) for r in resind_int])).getResnums())
resnum_bur = set(pdb_ala.select('resindex ' + ' '.join([str(r) for r in resind_bur])).getResnums())

dontallow_global = {'C'}
add_to_all = {'AV'}

res_burial_map = {}
for res in resnum_exp:
    res_burial_map[res] = 'e'
for res in resnum_int:
    res_burial_map[res] = 'i'
for res in resnum_bur:
    res_burial_map[res] = 'b'

with open('resfile.txt', 'a') as outfile:
    for (res, ch), ss in dssp.items():
        if (res, ch) in resfile.keys():
            continue
        bur = res_burial_map[res]
        aa_set = {aa for aa, prop in ss_bur_prop_label_dict[ss][bur].items() if np.round(prop, 1) >= 0.9} - dontallow_global
        aa_set |= add_to_all
        if res in surface_seq.keys():
            q = surface_seq[res]
            aa_set = aa_set - polar_dict[-1*q]
        line = str(res) + ' ' + ch + ' PIKAA ' + ''.join(aa_set) + ' \n'
        outfile.write(line)
with open('/Users/npolizzi/Projects/combs/src/runs/apixaban/apx_can_hbond_dict.pkl', 'rb') as infile:
    apx_can_hbond_dict = pickle.load(infile)
with open('/Users/npolizzi/Projects/combs/src/runs/apixaban/apx_atom_type_dict.pkl', 'rb') as infile:
    apx_atom_type_dict = pickle.load(infile)
pdb_df = combs.apps.clashfilter.make_pose_df(pdb.select('not resname APX'))
lig_df = combs.apps.clashfilter.make_lig_df(pose=pdb.select('resname APX'), **dict(can_hbond_dict=apx_can_hbond_dict, lig_atom_types_dict=apx_atom_type_dict))
con = combs.apps.clashfilter.Contact(lig_df, pdb_df.copy())
con.find()
hb_cons = con.df_contacts[con.df_contacts.contact_type == 'hb']
from scipy.spatial.distance import cdist
def find_more(hb_cons, ligname, cstfile, previous_sels, sels):
    for n, row in hb_cons[['segment_t', 'chain_t', 'resnum_t']].drop_duplicates().iterrows():
        seg = row.segment_t
        chain = row.chain_t
        resnum = row.resnum_t
        sel = '(sidechain and chain ' + chain + ' segment ' + seg + ' resnum ' + str(resnum) + ')'
        if sel in sels:
            continue
        sels.append(sel)
        try:
            if len(previous_sels) > 0:
                print(previous_sels)
                print(sel)
                pdb_df = combs.apps.clashfilter.make_pose_df(pdb.select('(not ' + previous_sels + ') and (not resname ' + ligname + ') and not ' + sel))
                pdb_df2 = combs.apps.clashfilter.make_pose_df(pdb.select('(not ' + previous_sels + ') and (not resname ' + ligname + ') and ' + sel))
            else:
                pdb_df = combs.apps.clashfilter.make_pose_df(pdb.select('(not resname ' + ligname + ') and not ' + sel))
                pdb_df2 = combs.apps.clashfilter.make_pose_df(pdb.select('(not resname ' + ligname + ') and ' + sel))
        except:
            continue
        con = combs.apps.clashfilter.Contact(pdb_df.copy(), pdb_df2.copy())
        con.find()
        hb_cons2 = con.df_contacts[con.df_contacts.contact_type == 'hb']
        print(hb_cons2)
        if len(hb_cons2) > 0:
            previous_sels = '(' + previous_sels + ' ' + sel + ')'
            write_csts(hb_cons2, ligname, cstfile, previous_sels, sels)
def write_csts(hb_cons, ligname, cstfile, previous_sels='', sels=[]):
    for n, row in hb_cons.iterrows():
        name_1 = row.name_t
        name_2 = row.name_q
        chain_1 = row.chain_t
        chain_2 = row.chain_q
        resnum_1 = str(row.resnum_t)
        resnum_2 = str(row.resnum_q)
        dist = str(cdist(row[['c_x_t', 'c_y_t', 'c_z_t']].values.reshape(1,-1), row[['c_x_q', 'c_y_q', 'c_z_q']].values.reshape(1,-1)).flatten()[0].round(1))
        cstfile.write('AtomPair ' + name_1 + ' ' + resnum_1 + chain_1 + ' ' + name_2 + ' ' + resnum_2 + chain_2 + ' HARMONIC ' + dist + ' 0.3 \n')
    find_more(hb_cons, ligname, cstfile, previous_sels, sels)
with open('apx3.cst', 'w') as cstfile:
    write_csts(hb_cons, 'APX', cstfile)