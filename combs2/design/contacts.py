__all__ = ['Clash', 'Contact']


from pandas import concat
import numpy as np
from .hbond import is_hbond, is_hbond_S_acceptor
from sklearn.neighbors import BallTree
from .constants import coords_cols, hbond_donor_types, hbond_acceptor_types
from .functions import atom_types_sort
from itertools import chain
import pickle


class Clash:
    """Will find members of dataframe dfq that do not clash with dft.
    Appears slow because it does not whittle down the vdms if an atom type is clashing...
    Might want to improve this...

    ...slightly improved in _find_clash_indices

    cla = Clash(dfq, dft)
    cla.set_grouping(['CG', 'rota', 'probe_name'])
    cla.find()
    df_clash_free = cla.dfq_clash_free

    """

    def __init__(self, dfq, dft, **kwargs):
        self.q_grouping = kwargs.get('q_grouping')
        dfq.loc[:, 'num_tag'] = np.arange(len(dfq))
        self.dfq = dfq
        self.dft = dft
        self.atom_types_dfq = None
        self.atom_types_dft = None
        self.dfq_atom_type = dict()
        self.dft_atom_type = dict()
        self._balltrees = dict()
        self.clash_indices = set()
        self.dfq_clash_free = []
        self.overlap_hb = kwargs.get('overlap_hb', 0.7)
        self.overlap_hb_heavy_nn = kwargs.get('overlap_hb_heavy_nn', 0.6)
        self.overlap_hb_heavy_no = kwargs.get('overlap_hb_heavy_no', 0.45)
        self.overlap_hb_heavy_oo = kwargs.get('overlap_hb_heavy_oo', 0.3)
        self.tol = kwargs.get('tol', 0.1)
        self.tol_h_alkyl = kwargs.get('tol_h_alkyl', 0.1)
        # Using default e-cloud vdW params from D and J Richardson's Probe program.
        self.vdw_radii = dict(co=kwargs.get('r_co', 1.65),
                              c_alkyl=kwargs.get('r_c_alkyl', 1.70),
                              c_aro=kwargs.get('r_c_aro', 1.75),
                              c_aro_met=kwargs.get('r_c_aro_met', 1.2),
                              n=kwargs.get('r_n', 1.55),
                              n_met=kwargs.get('r_n_met', 1.0),
                              f=kwargs.get('r_f', 1.30),
                              o=kwargs.get('r_o', 1.40),
                              s=kwargs.get('r_s', 1.80),
                              p=kwargs.get('r_p', 1.80),
                              h_pol=kwargs.get('r_h_pol', 1.05),
                              h_aro=kwargs.get('r_h_aro', 1.05),
                              h_alkyl=kwargs.get('r_h_alkyl', 1.22),
                              cl=kwargs.get('r_cl', 1.77),
                              na=kwargs.get('r_na', 0.95),
                              # fe=kwargs.get('r_fe', 0.74),
                              fe=kwargs.get('r_fe', 0.6),
                              zn=kwargs.get('r_zn', 0.71))

    def set_atom_types(self):
        self.atom_types_dfq = sorted(set(self.dfq.atom_type_label), key=lambda x: atom_types_sort(x))
        self.atom_types_dft = sorted(set(self.dft.atom_type_label), key=lambda x: atom_types_sort(x))

    def set_grouping(self, grouping):
        self.q_grouping = grouping

    def set_index(self):
        self.dfq.set_index(self.q_grouping, inplace=True, drop=False)

    def split_dfs_to_atom_types(self):
        for atom_type in self.atom_types_dfq:
            self.dfq_atom_type[atom_type] = self.dfq[self.dfq.atom_type_label == atom_type]
        for atom_type in self.atom_types_dft:
            self.dft_atom_type[atom_type] = self.dft[self.dft.atom_type_label == atom_type]

    @staticmethod
    def make_tree(dfq_):
        return BallTree(dfq_[['c_x', 'c_y', 'c_z']].values)

    def make_trees(self):
        for atom_type, dfq_ in self.dfq_atom_type.items():
            self._balltrees[atom_type] = self.make_tree(dfq_)

    @staticmethod
    def prune_empty(dists, inds):
        t_inds = []
        dists_ = []
        q_inds = []
        for t_ind, (dist, q_ind) in enumerate(zip(dists, inds)):
            if dist.size > 0:
                t_inds.append([t_ind])
                dists_.append(dist)
                q_inds.append(q_ind)
        return t_inds, q_inds, dists_

    @staticmethod
    def get_clashes_hb_hard_cutoff(dists, q_inds, t_inds, hb_hard_cutoff):
        q_inds_clashes = []
        q_inds_poss_hbonds = []
        t_inds_poss_hbonds = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < hb_hard_cutoff
            clashes = i_q[clashing]
            poss_hbonds = i_q[~clashing]
            if clashes.size > 0:
                q_inds_clashes.extend(clashes)
            if poss_hbonds.size > 0:
                q_inds_poss_hbonds.append(poss_hbonds)
                t_inds_poss_hbonds.append(i_t)
        return q_inds_clashes, q_inds_poss_hbonds, t_inds_poss_hbonds

    def _angle_test(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft[:, 3])
        t_is_acc = ~np.isnan(dft[:, 18])
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds)

        q_is_don = ~np.isnan(dfq[:, 3])
        q_is_acc = ~np.isnan(dfq[:, 18])
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds)

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            #q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            d_arr = dfq[q_is_don, 3:18]
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[:, 18:], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            #q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            a_arr = dfq[q_is_acc, 18:]
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[:, 3:18], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing)

    def _angle_test_S_acceptor(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft[:, 3])
        t_is_acc = ~np.isnan(dft[:, 18])
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds)

        q_is_don = ~np.isnan(dfq[:, 3])
        q_is_acc = ~np.isnan(dfq[:, 18])
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds)

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            #q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            d_arr = dfq[q_is_don, 3:18]
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[:, 18:], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            #q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            a_arr = dfq[q_is_acc, 18:]
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[:, 3:18], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing)

    def angle_test(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        clashes = []
        for q_inds_poss_hbond, t_inds_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_[coords_cols].values[t_inds_poss_hbond]
            df_poss_hb_q = dfq_[coords_cols].values[q_inds_poss_hbond]
            clashes.extend(self._angle_test(df_poss_hb_q, df_poss_hb_t,
                                            q_inds_poss_hbond))
        return clashes

    def angle_test_S_acceptor(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        clashes = []
        for q_inds_poss_hbond, t_inds_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_[coords_cols].values[t_inds_poss_hbond]
            df_poss_hb_q = dfq_[coords_cols].values[q_inds_poss_hbond]
            clashes.extend(self._angle_test_S_acceptor(df_poss_hb_q, df_poss_hb_t,
                                            q_inds_poss_hbond))
        return clashes

    def _find_clash_indices(self, atom_type_q, atom_type_t):
        tree = self._balltrees[atom_type_q]
        dft_ = self.dft_atom_type[atom_type_t]
        if (atom_type_q == 'h_alkyl') or (atom_type_t == 'h_alkyl'):
            cutoff = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t] - self.tol - self.tol_h_alkyl
        else:
            cutoff = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t] - self.tol
        i, d = tree.query_radius(dft_[['c_x', 'c_y', 'c_z']].values,
                                 r=cutoff, return_distance=True)
        t_inds, q_inds, dists = self.prune_empty(d, i)

        if t_inds:
            D_q = atom_type_q in hbond_donor_types
            A_t = atom_type_t in hbond_acceptor_types
            A_q = atom_type_q in hbond_acceptor_types
            D_t = atom_type_t in hbond_donor_types
            if not ((D_q and A_t) or (D_t and A_q)):
                return [j for i in q_inds for j in i]

            if (atom_type_q in {'n', 'p', 's'}) and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_nn
            elif (atom_type_q == 'o') and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_no
            elif (atom_type_t == 'o') and (atom_type_q in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_no
            elif (atom_type_q == 'o') and (atom_type_t == 'o'):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_oo
            else:
                hb_hard_cutoff = cutoff - self.overlap_hb

            # whittle down so not performing H-bond check on known clashers.
            global_inds = self.dfq_atom_type[atom_type_q]['num_tag'].values
            all_qs = np.array(list(chain(*q_inds)))
            mask = ~np.in1d(global_inds[all_qs], list(self.clash_indices))
            _ts = []
            _qs = []
            _ds = []
            i = 0
            for _t, _q, _d in zip(t_inds, q_inds, dists):
                j = len(_q)
                local_mask = mask[i:i+j]
                if local_mask.any():
                    _ts.append(_t)
                    _qs.append(np.array(_q)[local_mask])
                    _ds.append(np.array(_d)[local_mask])
                i += j
            if len(_ts) == 0:
                return list()
            q_inds_hard_clashes, q_inds_poss_hbonds, t_inds_poss_hbonds = \
                self.get_clashes_hb_hard_cutoff(_ds, _qs, _ts, hb_hard_cutoff)

            # q_inds_hard_clashes, q_inds_poss_hbonds, t_inds_poss_hbonds = \
            #     self.get_clashes_hb_hard_cutoff(dists, q_inds, t_inds, hb_hard_cutoff)

            if q_inds_poss_hbonds:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q == 's' or atom_type_t == 's':
                    q_inds_soft_clashes = self.angle_test_S_acceptor(q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_)
                else:
                    q_inds_soft_clashes = self.angle_test(q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_)
                q_inds_hard_clashes.extend(q_inds_soft_clashes)
            return q_inds_hard_clashes
        else:
            return list()

    def find_clash_indices(self):
        for atom_type_q in self.atom_types_dfq:
            for atom_type_t in self.atom_types_dft:
                local_clash_inds = self._find_clash_indices(atom_type_q, atom_type_t)
                if len(local_clash_inds) > 0:
                    global_clash_inds = self.dfq_atom_type[atom_type_q]['num_tag'].values[local_clash_inds]
                    self.clash_indices.update(global_clash_inds)

    def drop(self, return_clash_free=True, return_clash=False):
        if len(self.clash_indices) == 0:
            mask = np.zeros(len(self.dfq), dtype=bool)
        else:
            cind = self.dfq.index[list(self.clash_indices)]
            qind = self.dfq.index
            mask = qind.isin(cind)
        self.dfq.reset_index(drop=True, inplace=True)
        self.dfq.pop('num_tag')
        if return_clash_free:
            self.dfq_clash_free = self.dfq.loc[~mask] #.reset_index(drop=True)
            # self.dfq_clash_free.drop(columns='num_tag', inplace=True)
        if return_clash:
            self.dfq_clash = self.dfq.loc[mask] #.reset_index(drop=True)
            # self.dfq_clash.drop(columns='num_tag', inplace=True)

    def find(self, return_clash_free=True, return_clash=False):
        self.set_index()

        if self.atom_types_dfq is None:
            self.set_atom_types()

        if not self.dfq_atom_type:
            self.split_dfs_to_atom_types()

        if not self._balltrees:
            self.make_trees()

        self.find_clash_indices()
        self.drop(return_clash_free, return_clash)


class Contact:
    """Will find contacts between dataframe dfq and dft"""

    def __init__(self, dfq, dft, **kwargs):
        dfq.loc[:, 'num_tag'] = np.arange(len(dfq))
        dft.loc[:, 'num_tag'] = np.arange(len(dft))
        self.dfq = dfq
        self.dft = dft
        self.grouping = None
        self.atom_types_dfq = None
        self.atom_types_dft = None
        self.dfq_atom_type = dict()
        self.dft_atom_type = dict()
        self._balltrees = dict()
        self.q_global_indices = list()
        self.t_global_indices = list()
        self.contact_types = list()
        self.df_contacts = None
        self.dfq_clash_free = None
        self.gap_close_contact = kwargs.get('gap_close_contact', 0.3)
        self.gap_wide_contact = kwargs.get('gap_wide_contact', 0.5)
        self.overlap_hb = kwargs.get('overlap_hb', 0.7)
        self.overlap_hb_heavy_nn = kwargs.get('overlap_hb_heavy_nn', 0.6)
        self.overlap_hb_heavy_no = kwargs.get('overlap_hb_heavy_no', 0.45)
        self.overlap_hb_heavy_oo = kwargs.get('overlap_hb_heavy_oo', 0.3)
        # self.angle_hb_cutoff = kwargs.get('angle_hb_cutoff', 1.174)  # 100 degrees
        self.tol = kwargs.get('tol', 0.1)
        self.tol_h_alkyl = kwargs.get('tol_h_alkyl', 0.1)
        # Using default e-cloud vdW params from D and J Richardson's Probe program.
        self.vdw_radii = dict(co=kwargs.get('r_co', 1.65),
                              c_alkyl=kwargs.get('r_c_alkyl', 1.70),
                              c_aro=kwargs.get('r_c_aro', 1.75),
                              c_aro_met=kwargs.get('r_c_aro_met', 1.2),
                              n=kwargs.get('r_n', 1.55),
                              n_met=kwargs.get('r_n_met', 1.0),
                              f=kwargs.get('r_f', 1.30),
                              o=kwargs.get('r_o', 1.40),
                              s=kwargs.get('r_s', 1.80),
                              p=kwargs.get('r_p', 1.80),
                              h_pol=kwargs.get('r_h_pol', 1.05),
                              h_aro=kwargs.get('r_h_aro', 1.05),
                              h_alkyl=kwargs.get('r_h_alkyl', 1.22),
                              cl=kwargs.get('r_cl', 1.77),
                              na=kwargs.get('r_na', 0.95),
                              # fe=kwargs.get('r_fe', 0.74),
                              fe=kwargs.get('r_fe', 0.6),
                              zn=kwargs.get('r_zn', 0.71))

    def set_atom_types(self):
        self.atom_types_dfq = sorted(set(self.dfq.atom_type_label), key=lambda x: atom_types_sort(x))
        self.atom_types_dft = sorted(set(self.dft.atom_type_label), key=lambda x: atom_types_sort(x))

    def split_dfs_to_atom_types(self):
        for atom_type in self.atom_types_dfq:
            self.dfq_atom_type[atom_type] = self.dfq[self.dfq['atom_type_label'] == atom_type]
        for atom_type in self.atom_types_dft:
            self.dft_atom_type[atom_type] = self.dft[self.dft['atom_type_label'] == atom_type]

    @staticmethod
    def make_tree(dfq_):
        return BallTree(dfq_[['c_x', 'c_y', 'c_z']].values)

    def make_trees(self):
        for atom_type, dfq_ in self.dfq_atom_type.items():
            self._balltrees[atom_type] = self.make_tree(dfq_)
    
    def set_grouping(self, grouping):
        self.grouping = grouping

    @staticmethod
    def prune_empty(dists, inds):
        t_inds = []
        dists_ = []
        q_inds = []
        for t_ind, (dist, q_ind) in enumerate(zip(dists, inds)):
            if dist.size > 0:
                t_inds.append([t_ind])
                dists_.append(dist)
                q_inds.append(q_ind)
        return t_inds, q_inds, dists_

    @staticmethod
    def partition_contacts_hb_hard_cutoff(dists, q_inds, t_inds,
                                          cc_low, cc_low_hb, cc_high, wc_high,
                                          hb_hard_cutoff):
        q_inds_clashes = []
        t_inds_clashes = []
        q_inds_cc = []
        t_inds_cc = []
        q_inds_wc = []
        t_inds_wc = []
        q_inds_poss_hbonds_cl = []
        t_inds_poss_hbonds_cl = []
        q_inds_poss_hbonds_cc = []
        t_inds_poss_hbonds_cc = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < hb_hard_cutoff
            poss_hbonds_cl_test = (d >= hb_hard_cutoff) & (d < cc_low)
            poss_hbonds_cc_test = (d >= cc_low) & (d < cc_low_hb)
            clashes = i_q[clashing]
            cc_test = (d >= cc_low_hb) & (d < cc_high)
            wc_test = (d >= cc_high) & (d < wc_high)
            ccs = i_q[cc_test]
            wcs = i_q[wc_test]
            poss_hbonds_cl = i_q[poss_hbonds_cl_test]
            poss_hbonds_cc = i_q[poss_hbonds_cc_test]
            if clashes.size > 0:
                q_inds_clashes.append(clashes)
                t_inds_clashes.append(i_t)
            if ccs.size > 0:
                q_inds_cc.append(ccs)
                t_inds_cc.append(i_t)
            if wcs.size > 0:
                q_inds_wc.append(wcs)
                t_inds_wc.append(i_t)
            if poss_hbonds_cl.size > 0:
                q_inds_poss_hbonds_cl.append(poss_hbonds_cl)
                t_inds_poss_hbonds_cl.append(i_t)
            if poss_hbonds_cc.size > 0:
                q_inds_poss_hbonds_cc.append(poss_hbonds_cc)
                t_inds_poss_hbonds_cc.append(i_t)
        return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
               q_inds_wc, t_inds_wc, q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, \
               q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc

    @staticmethod
    def partition_contacts_no_hb(dists, q_inds, t_inds, cc_low, cc_high, wc_high):
        q_inds_clashes = []
        t_inds_clashes = []
        q_inds_cc = []
        t_inds_cc = []
        q_inds_wc = []
        t_inds_wc = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < cc_low
            clashes = i_q[clashing]
            cc_test = (d >= cc_low) & (d < cc_high)
            wc_test = (d >= cc_high) & (d < wc_high)
            ccs = i_q[cc_test]
            wcs = i_q[wc_test]
            if clashes.size > 0:
                q_inds_clashes.append(clashes)
                t_inds_clashes.append(i_t)
            if ccs.size > 0:
                q_inds_cc.append(ccs)
                t_inds_cc.append(i_t)
            if wcs.size > 0:
                q_inds_wc.append(wcs)
                t_inds_wc.append(i_t)
        return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
               q_inds_wc, t_inds_wc, [], []

    def _angle_test(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft[:, 3])
        t_is_acc = ~np.isnan(dft[:, 18])
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds), []

        q_is_don = ~np.isnan(dfq[:, 3])
        q_is_acc = ~np.isnan(dfq[:, 18])
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds), []

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            # q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            d_arr = dfq[q_is_don, 3:18]
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[:, 18:], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            # q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            a_arr = dfq[q_is_acc, 18:]
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[:, 3:18], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing), list(hbonds)

    def _angle_test_S_acceptor(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft[:, 3])
        t_is_acc = ~np.isnan(dft[:, 18])
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds), []

        q_is_don = ~np.isnan(dfq[:, 3])
        q_is_acc = ~np.isnan(dfq[:, 18])
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds), []

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            # q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            d_arr = dfq[q_is_don, 3:18]
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[:, 18:], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            # q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            a_arr = dfq[q_is_acc, 18:]
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[:, 3:18], (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing), list(hbonds)

    def angle_test(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        q_inds_clash = []
        t_inds_clash = []
        q_inds_hbond = []
        t_inds_hbond = []
        for q_inds_poss_hbond, t_ind_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_[coords_cols].values[t_ind_poss_hbond]
            df_poss_hb_q = dfq_[coords_cols].values[q_inds_poss_hbond]

            q_inds_clash_, q_inds_hbond_ = self._angle_test(df_poss_hb_q, df_poss_hb_t,
                                                            q_inds_poss_hbond)
            if q_inds_clash_:
                q_inds_clash.append(q_inds_clash_)
                t_inds_clash.append(t_ind_poss_hbond)
            if q_inds_hbond_:
                q_inds_hbond.append(q_inds_hbond_)
                t_inds_hbond.append(t_ind_poss_hbond)
        return q_inds_clash, t_inds_clash, q_inds_hbond, t_inds_hbond

    def angle_test_S_acceptor(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        q_inds_clash = []
        t_inds_clash = []
        q_inds_hbond = []
        t_inds_hbond = []
        for q_inds_poss_hbond, t_ind_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_[coords_cols].values[t_ind_poss_hbond]
            df_poss_hb_q = dfq_[coords_cols].values[q_inds_poss_hbond]

            q_inds_clash_, q_inds_hbond_ = self._angle_test_S_acceptor(df_poss_hb_q, df_poss_hb_t,
                                                                       q_inds_poss_hbond)
            if q_inds_clash_:
                q_inds_clash.append(q_inds_clash_)
                t_inds_clash.append(t_ind_poss_hbond)
            if q_inds_hbond_:
                q_inds_hbond.append(q_inds_hbond_)
                t_inds_hbond.append(t_ind_poss_hbond)
        return q_inds_clash, t_inds_clash, q_inds_hbond, t_inds_hbond

    def _find_contact_indices(self, atom_type_q, atom_type_t):
        tree = self._balltrees[atom_type_q]
        dft_ = self.dft_atom_type[atom_type_t]
        if (atom_type_q == 'h_alkyl') or (atom_type_t == 'h_alkyl'):
            vdw_sum = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t] 
            cc_high = vdw_sum + self.gap_close_contact
            cutoff = wc_high = vdw_sum + self.gap_wide_contact
            cc_low = vdw_sum - self.tol - self.tol_h_alkyl
        else:
            vdw_sum = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t]
            cc_high = vdw_sum + self.gap_close_contact
            cutoff = wc_high = vdw_sum + self.gap_wide_contact
            cc_low = vdw_sum - self.tol 
        
        i, d = tree.query_radius(dft_[['c_x', 'c_y', 'c_z']].values,
                                 r=cutoff, return_distance=True)
        t_inds, q_inds, dists = self.prune_empty(d, i)
        if t_inds:
            # if not {atom_type_q, atom_type_t}.issubset(hbond_types):
            D_q = atom_type_q in hbond_donor_types
            A_t = atom_type_t in hbond_acceptor_types
            A_q = atom_type_q in hbond_acceptor_types
            D_t = atom_type_t in hbond_donor_types
            if not ((D_q and A_t) or (D_t and A_q)):
                return self.partition_contacts_no_hb(dists, q_inds, t_inds, cc_low, cc_high, wc_high)
                # This returns q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc,
                # q_inds_wc, t_inds_wc, q_inds_hb, t_inds_hb (hb are empty lists)

            if (atom_type_q in {'n', 'p', 's'}) and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_nn
            elif (atom_type_q in {'o', 'f'}) and (atom_type_t in {'f', 'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_no
            elif (atom_type_t in {'o', 'f'}) and (atom_type_q in {'f', 'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_no
            elif (atom_type_q in {'o', 'f'}) and (atom_type_t in {'o', 'f'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_oo
            if (atom_type_q in {'n', 'o', 'p', 's', 'f'}) and (atom_type_t in {'n', 'o', 'p', 's', 'f'}):
                cc_low_hb = max(3.3, cc_low)
                cc_high = max(3.6, cc_low + 0.4)
                wc_high = max(4.0, cc_high + 0.4)
            else:
                hb_hard_cutoff = cc_low - self.overlap_hb
                cc_low_hb = max(2.6, cc_low)
                cc_high = max(2.8, cc_low + 0.4)
                wc_high = max(3.2, cc_high + 0.4)
            q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
            q_inds_wc, t_inds_wc, q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, \
            q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc = \
                self.partition_contacts_hb_hard_cutoff(dists, q_inds, t_inds, cc_low,
                                                       cc_low_hb, cc_high, wc_high,
                                                       hb_hard_cutoff)

            q_inds_hbond = []
            t_inds_hbond = []
            if q_inds_poss_hbonds_cl:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q == 's' or atom_type_t == 's':
                    q_inds_clash, t_inds_clash, q_inds_hbond_cl, t_inds_hbond_cl \
                        = self.angle_test_S_acceptor(q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, dfq_, dft_)
                else:
                    q_inds_clash, t_inds_clash, q_inds_hbond_cl, t_inds_hbond_cl \
                        = self.angle_test(q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, dfq_, dft_)
                q_inds_clashes.extend(q_inds_clash)
                t_inds_clashes.extend(t_inds_clash)
                q_inds_hbond.extend(q_inds_hbond_cl)
                t_inds_hbond.extend(t_inds_hbond_cl)
            if q_inds_poss_hbonds_cc:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q == 's' or atom_type_t == 's':
                    q_inds_cc_, t_inds_cc_, q_inds_hbond_cc, t_inds_hbond_cc \
                        = self.angle_test_S_acceptor(q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc, dfq_, dft_)
                else:
                    q_inds_cc_, t_inds_cc_, q_inds_hbond_cc, t_inds_hbond_cc \
                        = self.angle_test(q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc, dfq_, dft_)
                q_inds_cc.extend(q_inds_cc_)
                t_inds_cc.extend(t_inds_cc_)
                q_inds_hbond.extend(q_inds_hbond_cc)
                t_inds_hbond.extend(t_inds_hbond_cc)
                return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                       q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond
            else:
                return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                       q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond
        else:
            return [[]] * 8

    def extend_global_indices(self, atom_type_q, atom_type_t, q_inds, t_inds, contact_type):
        q_global_inds = list()
        t_global_inds = list()
        dfq = self.dfq_atom_type[atom_type_q]
        dft = self.dft_atom_type[atom_type_t]
        for q_inds_, t_ind in zip(q_inds, t_inds):
            q_global_inds.extend(dfq['num_tag'].values[q_inds_])
            t_global_inds.extend([dft['num_tag'].iat[t_ind[0]]] * len(q_inds_))

        self.q_global_indices.extend(q_global_inds)
        self.t_global_indices.extend(t_global_inds)
        self.contact_types.extend([contact_type] * len(q_global_inds))

    def _find(self, exclude_self_contacts=False, keep_columns=None):
        for atom_type_q in self.atom_types_dfq:
            for atom_type_t in self.atom_types_dft:
                q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond \
                    = self._find_contact_indices(atom_type_q, atom_type_t)
                if q_inds_clashes:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_clashes, t_inds_clashes, 'cl')
                if q_inds_cc:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_cc, t_inds_cc, 'cc')
                if q_inds_wc:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_wc, t_inds_wc, 'wc')
                if q_inds_hbond:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_hbond, t_inds_hbond, 'hb')
        # print('Doing iloc...')
        dfq_ = self.dfq.iloc[self.q_global_indices]
        dft_ = self.dft.iloc[self.t_global_indices]
        # print('len dfq_', len(dfq_), 'len dft_', len(dft_))
        # print('Doing outer join...')
        if keep_columns is not None:
            _keep_cols = []
            for col in keep_columns:
                if col[-2:] in ['_q', '_t']:
                    _keep_cols.append(col[:-2])
                else:
                    _keep_cols.append(col)
            dfq_cols = list(set(dfq_.columns) & set(_keep_cols))
            dfq_ = dfq_.loc[:, dfq_cols]
            dft_cols = list(set(dft_.columns) & set(_keep_cols))
            dft_ = dft_.loc[:, dft_cols]
        df = dfq_.reset_index(drop=True).join(dft_.reset_index(drop=True),
                                              how='outer', lsuffix='_q', rsuffix='_t')
        df['contact_type'] = self.contact_types
        ## exclude self-contacts hogs too much memory. Omitting for now.
        # if exclude_self_contacts:
        #     if self.grouping is None:
        #         assert Exception('grouping is None. Set grouping first with set_grouping().')
        #     # print('Doing self filter...')
        #     grouping_left = [c + '_q' for c in self.grouping]
        #     grouping_right = [c + '_t' for c in self.grouping]
        #     self_filter = (df[grouping_left].values == df[grouping_right].values).all(axis=1)
        #     # print('Filtering...')
        #     df = df[~self_filter]
        self.df_contacts = df
    
    @staticmethod
    def self_contact(row, left_on, right_on):
        return (row[left_on].values == row[right_on].values).all()

    def find(self, exclude_self_contacts=False, keep_columns=None):

        if self.atom_types_dfq is None:
            self.set_atom_types()

        if not self.dfq_atom_type:
            self.split_dfs_to_atom_types()

        if not self._balltrees:
            self.make_trees()

        self._find(exclude_self_contacts=exclude_self_contacts, keep_columns=keep_columns)


class ClashVDM:
    '''Do Cbeta, do non-Cbeta, combine results'''
    def __init__(self, dfq, dft):
        self.dfq = dfq
        self.dft = dft
        self.exclude = None
        self.q_grouping = None
        self.dfq_cb_clash_free = None
        self.dfq_non_cb_clash_free = None
        self.dfq_clash_free = None
        self.dft_for_non_cb = None
        self.dft_for_cb = None
        self.dfq_non_cb = None
        self.dfq_cb = None
        self.resname_q = dfq.resname_rota.iat[0]

    def set_grouping(self, grouping):
        self.q_grouping = grouping

    def set_exclude(self, exclude):
        self.exclude = exclude

    def setup(self):
        df = self.dft
        resnum = self.exclude[2]
        chain = self.exclude[1]
        seg = self.exclude[0]

        res = ((df['resnum'] == resnum) &
               (df['chain'] == chain) &
               (df['segment'] == seg))

        if self.resname_q == 'PRO':
            print('            setting special PRO case for clashing')
            gen_exclude = (res & ~(df['name'] == 'O'))

            gen_exclude = gen_exclude | ( ( (df['resnum'] == resnum - 1) &
                                            (df['chain'] == chain) &
                                            (df['segment'] == seg) &
                                            (df['name'].isin({'C', 'O', 'CA', 'HA', 'HA2', '2HA'})) )) #|
                                            # (((df['resnum'] == resnum - 2) &
                                            # (df['chain'] == chain) &
                                            # (df['segment'] == seg) &
                                            # (df['name'] == 'C'))))

        else:
            gen_exclude = (res & ~(df['name'].isin(['H', 'O'])))

        # cbeta_exclude is for a 4-bond distance condition to start
        # measuring clashes between atoms.  For Cbeta atoms, this means
        # excluding the i-1 residue's C atom and the i+1 residue's N atom.

        resm1 = ((df['resnum'] == resnum - 1) &
                (df['chain'] == chain) &
                (df['segment'] == seg) &
                (df['name'] == 'C'))

        resp1 = ((df['resnum'] == resnum + 1) &
                 (df['chain'] == chain) &
                 (df['segment'] == seg) &
                 (df['name'] == 'N'))

        if resm1.any() and resp1.any():
            cbeta_exclude = (resm1 | res | resp1)
        elif resm1.any():
            cbeta_exclude = (resm1 | res)
        else:
            cbeta_exclude = (resp1 | res)
        self.dft_for_non_cb = df[~gen_exclude].copy()
        self.dft_for_cb = df[~cbeta_exclude].copy()

        cb_crit = (self.dfq.name == 'CB') & (self.dfq.chain == 'X')
        self.dfq_non_cb = self.dfq[~cb_crit].copy()
        self.dfq_cb = self.dfq[cb_crit].copy()

    def set_index(self, df):
        df.set_index(self.q_grouping, inplace=True, drop=False)

    def find_cb_clash(self, **kwargs):
        cla = Clash(self.dfq_cb, self.dft_for_cb, **kwargs)
        cla.set_grouping(self.q_grouping)
        cla.find(return_clash_free=True, return_clash=True)
        self.dfq_cb_clash = cla.dfq_clash
        self.dfq_cb_clash_free = cla.dfq_clash_free

    def find_non_cb_clash(self, **kwargs):
        cla = Clash(self.dfq_non_cb, self.dft_for_non_cb, **kwargs)
        cla.set_grouping(self.q_grouping)
        cla.find(return_clash_free=True, return_clash=True)
        self.dfq_non_cb_clash = cla.dfq_clash
        self.dfq_non_cb_clash_free = cla.dfq_clash_free

    def find(self, **kwargs):
        self.find_cb_clash(**kwargs)
        self.find_non_cb_clash(**kwargs)
        self.set_index(self.dfq_non_cb_clash_free)
        self.set_index(self.dfq_cb_clash)
        self.set_index(self.dfq_cb_clash_free)
        self.set_index(self.dfq_non_cb_clash)
        isin1 = self.dfq_non_cb_clash_free.index.isin(self.dfq_cb_clash.index)
        # df1 = self.dfq_non_cb_clash_free.loc[~isin1]
        df1 = self.dfq_non_cb_clash_free[~isin1]
        isin2 = self.dfq_cb_clash_free.index.isin(self.dfq_non_cb_clash.index)
        # df2 = self.dfq_cb_clash_free.loc[~isin2]
        df2 = self.dfq_cb_clash_free[~isin2]
        self.dfq_clash_free = concat((df1, df2), sort=False, ignore_index=True)


