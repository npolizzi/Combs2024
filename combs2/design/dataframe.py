__all__ = ['get_hbond_atom_coords', 'make_df_from_prody']


from pandas import DataFrame
from .constants import atom_type_dict, can_hbond, resnames_aa_20
import numpy as np
from prody import calcCenter


def get_hbond_atom_coords(name, resname, selection):
    """Gets the hydrogen bond vectors from a 1-residue prody selection, returns
    donor vectors as a list of coord arrays, returns acc vectors as a coord array."""

    assert len(set(selection.getResindices())) == 1, 'input must be only 1 residue'

    ev = np.empty(3)
    ev[:] = np.nan
    h_don_coords = [ev, ev, ev, ev]
    don_coords = ev
    try:
        for i, (name1, name2) in enumerate(can_hbond[resname][name]['donor']):
            v1 = selection.select('name ' + name1).getCoords()[0]
            v2 = selection.select('name ' + name2).getCoords()[0]
            h_don_coords[i] = v1
            don_coords = v2
    except (AttributeError, ValueError, KeyError):
        pass

    if name == 'N':
        try:
            for i, (name1, name2) in enumerate(can_hbond['N_term'][name]['donor']):
                v1 = selection.select('name ' + name1).getCoords()[0]
                v2 = selection.select('name ' + name2).getCoords()[0]
                h_don_coords[i] = v1
                don_coords = v2
        except (AttributeError, ValueError, KeyError):
            pass


    # Need to account for special case of histidine
    # If a His N is a donor it cannot be an acceptor.
    if resname == 'HIS' and ~np.isnan(don_coords[0]):
        acc1_coords = ev
        acc2_coords = ev
        return don_coords, h_don_coords, acc1_coords, acc2_coords

    try:
        name1, name2, name3 = can_hbond[resname][name]['acceptor']
        acc1_coords = selection.select('name ' + name1).getCoords()[0]
        sel = selection.select('name ' + name2 + ' ' + name3)
        acc2_coords = calcCenter(sel).flatten()
    except (AttributeError, ValueError, KeyError):
        acc1_coords = ev
        acc2_coords = ev

    return don_coords, h_don_coords, acc1_coords, acc2_coords


def make_df_from_prody(prody_obj, include_betas_occupancies=False, **kwargs):
    """Returns a dataframe with rows of every atom in
    every residue of a prody object or selection"""

    lig_atom_type_dict = kwargs.get('lig_atom_types_dict')
    can_hbond_dict = kwargs.get('can_hbond_dict')
    if can_hbond_dict is not None:
        can_hbond.update(can_hbond_dict)

    resnums = list()
    chains = list()
    segments = list()
    names = list()
    resnames = list()
    betas = list()
    occs = list()
    c_x = list()
    c_y = list()
    c_z = list()
    cd_x = list()
    cd_y = list()
    cd_z = list()
    ch1_x = list()
    ch1_y = list()
    ch1_z = list()
    ch2_x = list()
    ch2_y = list()
    ch2_z = list()
    ch3_x = list()
    ch3_y = list()
    ch3_z = list()
    ch4_x = list()
    ch4_y = list()
    ch4_z = list()
    ca1_x = list()
    ca1_y = list()
    ca1_z = list()
    ca2_x = list()
    ca2_y = list()
    ca2_z = list()
    atom_type_labels = list()
    seg_chain_resnums = list()

    for res_sel in prody_obj.getHierView().iterResidues():
        resname = set(res_sel.getResnames()).pop()
        for atom in res_sel:
            resnames.append(resname)
            name = atom.getName()
            names.append(name)
            if lig_atom_type_dict and resname not in resnames_aa_20:
                atom_type_labels.append(lig_atom_type_dict[resname][name])
            else:
                atom_type_labels.append(atom_type_dict[resname][name])
            resnum = atom.getResnum()
            resnums.append(resnum)
            chain = atom.getChid()
            chains.append(chain)
            segment = atom.getSegname()
            segments.append(segment)
            betas.append(atom.getBeta().astype('float32'))
            occs.append(atom.getOccupancy().astype('float32'))
            seg_chain_resnums.append((segment, chain, resnum))
            don_coords, h_don_coords, acc1_coords, acc2_coords = get_hbond_atom_coords(name, resname, res_sel)
            cd_x.append(don_coords[0].astype('float32'))
            cd_y.append(don_coords[1].astype('float32'))
            cd_z.append(don_coords[2].astype('float32'))
            ch1_x.append(h_don_coords[0][0].astype('float32'))
            ch1_y.append(h_don_coords[0][1].astype('float32'))
            ch1_z.append(h_don_coords[0][2].astype('float32'))
            ch2_x.append(h_don_coords[1][0].astype('float32'))
            ch2_y.append(h_don_coords[1][1].astype('float32'))
            ch2_z.append(h_don_coords[1][2].astype('float32'))
            ch3_x.append(h_don_coords[2][0].astype('float32'))
            ch3_y.append(h_don_coords[2][1].astype('float32'))
            ch3_z.append(h_don_coords[2][2].astype('float32'))
            ch4_x.append(h_don_coords[3][0].astype('float32'))
            ch4_y.append(h_don_coords[3][1].astype('float32'))
            ch4_z.append(h_don_coords[3][2].astype('float32'))
            ca1_x.append(acc1_coords[0].astype('float32'))
            ca1_y.append(acc1_coords[1].astype('float32'))
            ca1_z.append(acc1_coords[2].astype('float32'))
            ca2_x.append(acc2_coords[0].astype('float32'))
            ca2_y.append(acc2_coords[1].astype('float32'))
            ca2_z.append(acc2_coords[2].astype('float32'))
            c = atom.getCoords()
            c_x.append(c[0].astype('float32'))
            c_y.append(c[1].astype('float32'))
            c_z.append(c[2].astype('float32'))

    if include_betas_occupancies:
        df = DataFrame(list(zip(resnums, chains, segments, resnames, names, betas, occs,
                                c_x, c_y, c_z, cd_x, cd_y, cd_z,
                                ch1_x, ch1_y, ch1_z, ch2_x, ch2_y, ch2_z,
                                ch3_x, ch3_y, ch3_z, ch4_x, ch4_y, ch4_z,
                                ca1_x, ca1_y, ca1_z, ca2_x, ca2_y, ca2_z,
                                atom_type_labels, seg_chain_resnums)),
                       columns=['resnum', 'chain', 'segment', 'resname', 'name', 'beta', 'occ',
                                'c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
                                'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
                                'c_H2_x', 'c_H2_y', 'c_H2_z',
                                'c_H3_x', 'c_H3_y', 'c_H3_z',
                                'c_H4_x', 'c_H4_y', 'c_H4_z',
                                'c_A1_x', 'c_A1_y', 'c_A1_z',
                                'c_A2_x', 'c_A2_y', 'c_A2_z', 'atom_type_label', 'seg_chain_resnum'])
        dtype_dict = dict(beta='float32', occ='float32',
                          c_x='float32', c_y='float32', c_z='float32', c_D_x='float32', c_D_y='float32',
                          c_D_z='float32', c_H1_x='float32', c_H1_y='float32', c_H1_z='float32',
                          c_H2_x='float32', c_H2_y='float32', c_H2_z='float32',
                          c_H3_x='float32', c_H3_y='float32', c_H3_z='float32',
                          c_H4_x='float32', c_H4_y='float32', c_H4_z='float32',
                          c_A1_x='float32', c_A1_y='float32', c_A1_z='float32',
                          c_A2_x='float32', c_A2_y='float32', c_A2_z='float32',
                          )
    else:
        df = DataFrame(list(zip(resnums, chains, segments, resnames, names,
                                c_x, c_y, c_z, cd_x, cd_y, cd_z,
                                ch1_x, ch1_y, ch1_z, ch2_x, ch2_y, ch2_z,
                                ch3_x, ch3_y, ch3_z, ch4_x, ch4_y, ch4_z,
                                ca1_x, ca1_y, ca1_z, ca2_x, ca2_y, ca2_z,
                                atom_type_labels, seg_chain_resnums)),
                       columns=['resnum', 'chain', 'segment', 'resname', 'name',
                                'c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
                                'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
                                'c_H2_x', 'c_H2_y', 'c_H2_z',
                                'c_H3_x', 'c_H3_y', 'c_H3_z',
                                'c_H4_x', 'c_H4_y', 'c_H4_z',
                                'c_A1_x', 'c_A1_y', 'c_A1_z',
                                'c_A2_x', 'c_A2_y', 'c_A2_z', 'atom_type_label', 'seg_chain_resnum'])
        dtype_dict = dict(
                          c_x='float32', c_y='float32', c_z='float32', c_D_x='float32', c_D_y='float32',
                          c_D_z='float32', c_H1_x='float32', c_H1_y='float32', c_H1_z='float32',
                          c_H2_x='float32', c_H2_y='float32', c_H2_z='float32',
                          c_H3_x='float32', c_H3_y='float32', c_H3_z='float32',
                          c_H4_x='float32', c_H4_y='float32', c_H4_z='float32',
                          c_A1_x='float32', c_A1_y='float32', c_A1_z='float32',
                          c_A2_x='float32', c_A2_y='float32', c_A2_z='float32',
                          )

    if can_hbond_dict is not None and lig_atom_type_dict is not None:
        df['lig_resname'] = df['resname']
        df['lig_name'] =  df['name']

    return df.astype(dtype_dict)









