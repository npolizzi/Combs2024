import numpy as np
from scipy.sparse import csr_matrix
from numba import jit, types, int32
from numba.typed import List
from os import makedirs
from sklearn.neighbors import NearestNeighbors


class Cluster:

    def __init__(self, **kwargs):

        self.bb_sel = kwargs.get('bb_sel', [(10, ['N', 'CA', 'C'])])
        self.cluster_type = kwargs.get('cluster_type', 'unlabeled')
        self.ifg_dict = kwargs.get('ifg_dict', None)
        self.num_atoms_ifg = len([v for v in self.ifg_dict.values()][0]) \
            if self.ifg_dict else None
        self.rmsd_cutoff = kwargs.get('rmsd_cutoff', 0.5)
        od = kwargs.get('rmsd_mat_outdir', '.')
        self.rmsd_mat_outdir = od if od[-1] == '/' else od + '/'
        od = kwargs.get('clusters_outdir', '.')
        self.clusters_outdir = od if od[-1] == '/' else od + '/'
        od = kwargs.get('pickle_file_outdir', '.')
        self.pickle_file_outdir = od if od[-1] == '/' else od + '/'
        self.contacts_to_query = list()
        self.contacts_to_full_query = list()
        self.diversity = list()
        self.sequences = list()
        self.pdbs = None
        self.resname = None
        self.pdb_coords = list()
        self.pdbs_errorfree = list()
        self.rmsd_mat = None
        self.adj_mat = None
        self.mems = None
        self.cents = None
        self._square = False
        self._adj_mat = False
        self._data_set = False
        self._ifg_count = list()
        self._vdm_count = list()
        self._query_name = list()
        self._centroid = list()
        self._cluster_num = list()
        self._cluster_size = list()
        self._rmsd_from_centroid = list()
        self._resname = list()
        self.__resname = list()
        self._cluster_type = list()
        self.df = None
        self.query_names_errorfree = list()
        self.query_names = kwargs.get('query_names')

    def make_pairwise_rmsd_mat(self, maxdist=False, superpose=True):
        """Uses C-compiled numba code for fast pairwise superposition
        and RMSD calculation of coords (all against all)."""
        assert isinstance(self.pdb_coords, np.ndarray), 'PDB coords must be ' \
                                                       'numpy array'
        assert self.pdb_coords.dtype == 'float32', 'PDB coords must ' \
                                                   'be dtype of float32'
        if superpose:
            if maxdist:
                self.rmsd_mat = _make_pairwise_maxdist_mat(self.pdb_coords)
            else:
                self.rmsd_mat = _make_pairwise_rmsd_mat(self.pdb_coords)
        else:
            if maxdist:
                self.rmsd_mat = _make_pairwise_maxdist_mat_no_superpose(self.pdb_coords)
            else:
                self.rmsd_mat = _make_pairwise_rmsd_mat_no_superpose(self.pdb_coords)

    def save_rmsd_mat(self):
        """Saves RMSD matrix (lower triangular) as a numpy array.  This also
        saves a text file that lists the file names in the order of the matrix
        indices."""
        outpath = self.rmsd_mat_outdir + self.cluster_type + '/' + self.resname + '/'
        try:
            makedirs(outpath)
        except FileExistsError:
            pass

        np.save(outpath + 'rmsd_mat_half_'
                + self.resname, self.rmsd_mat)

        with open(outpath + 'filenames.txt', 'w') as outfile:
            outfile.write('bb_sel = ' + ' '.join(str(t) for t in self.bb_sel) + '\n')
            outfile.write('ifg_dict = ' + ', '.join(key + ': ' + str(val) for key, val
                                                    in self.ifg_dict.items()) + '\n')
            for pdb in self.pdbs_errorfree:
                outfile.write(str(pdb).split()[-1] + '\n')

    @staticmethod
    def greedy(adj_mat, min_cluster_size=2, index_labels=None):
        """Takes an adjacency matrix as input.
            All values of adj_mat are 1 or 0:  1 if <= to cutoff, 0 if > cutoff.
            Can generate adj_mat from data in column format with:
            sklearn.neighbors.NearestNeighbors(metric='euclidean',
            radius=cutoff).fit(data).radius_neighbors_graph(data)"""

        if not isinstance(adj_mat, csr_matrix):
            try:
                adj_mat = csr_matrix(adj_mat)
            except:
                print('adj_mat distance matrix must be scipy csr_matrix '
                      '(or able to convert to one)')
                return

        assert adj_mat.shape[0] == adj_mat.shape[1], 'Distance matrix is not square.'

        all_mems = []
        cents = []
        indices = np.arange(adj_mat.shape[0])

        if index_labels is not None:
            d = {il: i for i, il in enumerate(set(index_labels))}
            index_labels = [d[il] for il in index_labels]
            index_labels = np.array(index_labels)
            _index_labels = index_labels.copy()

        try:
            while adj_mat.shape[0] > 0:

                if index_labels is not None:
                    adj_mat_sum = adj_mat.sum(axis=1).flatten()
                    possible_cents = adj_mat_sum.argsort()
                    for i_cent in range(1, possible_cents.shape[1] + 1):
                        cent = possible_cents[0, -i_cent]
                        row = adj_mat.getrow(cent)
                        tf = ~row.toarray().astype(bool)[0]
                        mems = indices[~tf]
                        labels = index_labels[mems]
                        u, i = np.unique(labels, return_index=True)
                        if len(u) == adj_mat_sum[0, cent]:
                            break
                        else:
                            test = True
                            for j in range(i_cent + 1, possible_cents.shape[1] + 1):
                                _cent = possible_cents[0, -j]
                                _row = adj_mat.getrow(_cent)
                                _tf = ~_row.toarray().astype(bool)[0]
                                _mems = indices[~_tf]
                                _labels = index_labels[_mems]
                                _u, _i = np.unique(_labels, return_index=True)
                                if len(u) < len(_u):
                                    test = False
                                    break
                            if test:
                                break

                    mems = mems[i]
                    _cent_ = indices[cent]
                    if _cent_ not in mems:
                        label = index_labels[_cent_]
                        __cent__ = mems[u == label][0]
                        cent = indices.tolist().index(__cent__)

                    tf = ~np.in1d(_index_labels, u)
                    _index_labels = _index_labels[tf]

                else:
                    cent = adj_mat.sum(axis=1).argmax()
                    row = adj_mat.getrow(cent)
                    tf = ~row.toarray().astype(bool)[0]
                    mems = indices[~tf]

                if len(mems) < min_cluster_size:

                    if index_labels is not None:
                        labels = index_labels[indices]
                        u, i = np.unique(labels, return_index=True)
                        indices = indices[i]

                    [cents.append(i) for i in indices]
                    [all_mems.append(np.array([i])) for i in indices]
                    break

                cents.append(indices[cent])
                all_mems.append(mems)

                indices = indices[tf]
                adj_mat = adj_mat[tf][:, tf]
        except KeyboardInterrupt:
            pass

        return all_mems, cents

    @staticmethod
    def fast_greedy(adj_mat, min_cluster_size=2):
        """Takes an adjacency matrix as input.
            All values of adj_mat are 1 or 0:  1 if <= to cutoff, 0 if > cutoff.
            Can generate adj_mat from data in column format with:
            sklearn.neighbors.NearestNeighbors(metric='euclidean',
            radius=cutoff).fit(data).radius_neighbors_graph(data)"""

        if not isinstance(adj_mat, csr_matrix):
            try:
                adj_mat = csr_matrix(adj_mat)
            except:
                print('adj_mat distance matrix must be scipy csr_matrix '
                      '(or able to convert to one)')
                return

        assert adj_mat.shape[0] == adj_mat.shape[1], 'Distance matrix is not square.'

        all_mems = []
        cents = []
        indices = np.arange(adj_mat.shape[0])
        filter = np.ones(adj_mat.shape[0], dtype=bool)
        try:
            while filter.any():
                adj_mat_sum = adj_mat.sum(axis=0)
                adj_mat_sum[:, ~filter] = 0
                cent = adj_mat_sum.argmax()
                mems = adj_mat.indices[adj_mat.indptr[cent]:adj_mat.indptr[cent + 1]]
                mems = mems[filter[mems]]

                if len(mems) < min_cluster_size:
                    [cents.append(i) for i in indices[filter]]
                    [all_mems.append(np.array([i])) for i in indices[filter]]
                    break

                # for m in mems:
                #     adj_mat.data[adj_mat.indptr[m]:adj_mat.indptr[m + 1]] = 0
                ## replace with change_data below
                change_data(mems, adj_mat.data, adj_mat.indptr)
                cents.append(cent)
                all_mems.append(mems)
                filter[mems] = False

        except KeyboardInterrupt:
            pass

        return all_mems, cents

    def make_square(self):
        self.rmsd_mat = self.rmsd_mat.T + self.rmsd_mat
        self._square = True

    def make_adj_mat_no_superpose(self):
        num_atoms = len(self.pdb_coords[0])
        nbrs = NearestNeighbors(radius=self.rmsd_cutoff * np.sqrt(num_atoms))
        nbrs_coords = np.array([s.getCoords().flatten() for s in self.pdb_coords])
        nbrs.fit(nbrs_coords)
        self.adj_mat = nbrs.radius_neighbors_graph(nbrs_coords)
        self._adj_mat = True

    def make_adj_mat(self):
        """Makes an adjacency matrix from the RMSD matrix"""
        self.adj_mat = np.zeros(self.rmsd_mat.shape)
        self.adj_mat[self.rmsd_mat <= self.rmsd_cutoff] = 1
        self.adj_mat = csr_matrix(self.adj_mat)
        self._adj_mat = True

    def cluster(self, min_cluster_size=2, index_labels=None):
        """Performs greedy clustering of the RMSD matrix with a given
        RMSD cutoff (Default cutoff is 0.5 A)."""
        self.mems, self.cents = self.greedy(self.adj_mat, min_cluster_size, index_labels)

    def fast_cluster(self, min_cluster_size=2):
        """Performs greedy clustering of the RMSD matrix with a given
        RMSD cutoff (Default cutoff is 0.5 A)."""
        self.mems, self.cents = self.fast_greedy(self.adj_mat, min_cluster_size)


@jit(nopython=True, cache=True)
def change_data(mems, data, indptr):
    num_mems = mems.shape[0]
    for i in range(num_mems):
        data[indptr[mems[i]]: indptr[mems[i] + 1]] = 0


@jit("f4[:,:](f4[:,:,:])", nopython=True, cache=True)
def _make_pairwise_rmsd_mat(X):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=np.float32)
    m_com = np.zeros(O, dtype=np.float32)
    t_com = np.zeros(O, dtype=np.float32)
    m = np.zeros((N, O), dtype=np.float32)
    mtrans = np.zeros((O, N), dtype=np.float32)
    mtr = np.zeros((N, O), dtype=np.float32)
    t = np.zeros((N, O), dtype=np.float32)
    c = np.zeros((O, O), dtype=np.float32)
    U = np.zeros((O, O), dtype=np.float32)
    S = np.zeros(O, dtype=np.float32)
    Wt = np.zeros((O, O), dtype=np.float32)
    R = np.zeros((O, O), dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    sub = np.zeros(N * O, dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            for k in range(O):
                m_com[k] = np.mean(X[i, :, k])
                t_com[k] = np.mean(X[j, :, k])
            m = np.subtract(X[i, :, :], m_com)
            for a in range(N):
                for b in range(O):
                    mtrans[b, a] = m[a, b]
            t = np.subtract(X[j, :, :], t_com)
            c = np.dot(mtrans, t)
            U, S, Wt = np.linalg.svd(c)
            R = np.dot(U, Wt)
            if np.linalg.det(R) < 0.0:
                Wt[-1, :] *= -1.0
                R = np.dot(U, Wt)
            mtr = np.add(np.dot(m, R), t_com)
            q = 0
            for a in range(N):
                for b in range(O):
                    mtr_re[q] = mtr[a, b]
                    t_re[q] = X[j, :, :][a, b]
                    q += 1
            sub = np.subtract(mtr_re, t_re)
            D[i, j] = np.sqrt(1.0 / N * np.dot(sub, sub))
    return D


@jit("f4[:,:](f4[:,:,:])", nopython=True, cache=True)
def _make_pairwise_maxdist_mat(X):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=np.float32)
    m_com = np.zeros(O, dtype=np.float32)
    t_com = np.zeros(O, dtype=np.float32)
    m = np.zeros((N, O), dtype=np.float32)
    mtrans = np.zeros((O, N), dtype=np.float32)
    mtr = np.zeros((N, O), dtype=np.float32)
    t = np.zeros((N, O), dtype=np.float32)
    c = np.zeros((O, O), dtype=np.float32)
    U = np.zeros((O, O), dtype=np.float32)
    S = np.zeros(O, dtype=np.float32)
    Wt = np.zeros((O, O), dtype=np.float32)
    R = np.zeros((O, O), dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    sub = np.zeros(N * O, dtype=np.float32)
    dists = np.zeros(N, dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            for k in range(O):
                m_com[k] = np.mean(X[i, :, k])
                t_com[k] = np.mean(X[j, :, k])
            m = np.subtract(X[i, :, :], m_com)
            for a in range(N):
                for b in range(O):
                    mtrans[b, a] = m[a, b]
            t = np.subtract(X[j, :, :], t_com)
            c = np.dot(mtrans, t)
            U, S, Wt = np.linalg.svd(c)
            R = np.dot(U, Wt)
            if np.linalg.det(R) < 0.0:
                Wt[-1, :] *= -1.0
                R = np.dot(U, Wt)
            mtr = np.add(np.dot(m, R), t_com)
            pp = 0
            for a in range(N):
                q = 0
                for b in range(O):
                    mtr_re[q] = mtr[a, b]
                    t_re[q] = X[j, :, :][a, b]
                    q += 1
                sub = np.subtract(mtr_re, t_re)
                dists[pp] = np.sqrt(np.dot(sub, sub))
                pp += 1
            D[i, j] = np.max(dists)
    return D


@jit("f4[:,:](f4[:,:,:])", nopython=True, cache=True)
def _make_pairwise_rmsd_mat_no_superpose(X):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            q = 0
            for a in range(N):
                for b in range(O):
                    mtr_re[q] = X[i, :, :][a, b]
                    t_re[q] = X[j, :, :][a, b]
                    q += 1
            sub = np.subtract(mtr_re, t_re)
            D[i, j] = np.sqrt(1.0 / N * np.dot(sub, sub))
    return D + D.T


@jit("f4[:,:](f4[:,:,:])", nopython=True, cache=True)
def _make_pairwise_maxdist_mat_no_superpose(X):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    sub = np.zeros(N * O, dtype=np.float32)
    dists = np.zeros(N, dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            pp = 0
            for a in range(N):
                q = 0
                for b in range(O):
                    mtr_re[q] = X[i, :, :][a, b]
                    t_re[q] = X[j, :, :][a, b]
                    q += 1
                sub = np.subtract(mtr_re, t_re)
                dists[pp] = np.sqrt(np.dot(sub, sub))
                pp += 1
            D[i, j] = np.max(dists)
    return D


@jit("b1[:,:](f4[:,:,:], f8)", nopython=True, cache=True)
def _make_adj_mat(X, rmsd_cutoff):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=types.bool_)
    msd_cutoff = rmsd_cutoff ** 2
    for i in range(M):
        D[i, i] = True
        for j in range(i + 1, M):
            msd = 0
            for a in range(N):
                for b in range(O):
                    msd += (X[i, :, :][a, b] - X[j, :, :][a, b]) ** 2
            if 1.0 / N * msd < msd_cutoff:
                D[i, j] = True
                D[j, i] = True
    return D


@jit("b1[:,:](f4[:,:,:], i8[:,:], f8)", nopython=True, cache=True)
def _make_adj_mat_no_self_adj(X, ind_arr, rmsd_cutoff):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=types.bool_)
    L = ind_arr.shape[0]
    msd_cutoff = rmsd_cutoff ** 2
    for k in range(L):
        low = ind_arr[k, 0]
        high = ind_arr[k, 1]
        for i in range(low, high + 1):
            D[i, i] = True
            for j in range(high + 1, M):
                msd = 0
                for a in range(N):
                    for b in range(O):
                        msd += (X[i, :, :][a, b] - X[j, :, :][a, b]) ** 2
                if 1.0 / N * msd < msd_cutoff:
                    D[i, j] = True
                    D[j, i] = True
    return D


# @jit("i4[:,:](f4[:,:,:], i8[:,:], f8)", nopython=True, cache=True, parallel=True)
# def _make_adj_mat_no_self_adj_parallel(X, ind_arr, rmsd_cutoff):
#     M = X.shape[0]
#     N = X.shape[1]
#     O = X.shape[2]
#     D = np.zeros((M, M), dtype=types.bool_)
#     L = ind_arr.shape[0]
#     msd_cutoff = rmsd_cutoff ** 2
#     for k in prange(L):
#         low = ind_arr[k, 0]
#         high = ind_arr[k, 1]
#         for i in prange(low, high + 1):
#             D[i, i] = True
#             for j in prange(high + 1, M):
#                 msd = 0
#                 for a in prange(N):
#                     for b in prange(O):
#                         msd += (X[i, :, :][a, b] - X[j, :, :][a, b]) ** 2
#                 if 1.0 / N * msd < msd_cutoff:
#                     D[i, j] = True
#                     D[j, i] = True
#     D = np.where(D)
#     A = np.zeros((D[0].size, 2), dtype=np.int32)
#     A[:, 0] = D[0]
#     A[:, 1] = D[1]
#     return A


@jit(nopython=True, cache=True)
def _make_adj_mat_no_self_adj_sparse(X, ind_arr, rmsd_cutoff):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    # D = List.empty_list(List[types.f4])
    D1 = List.empty_list(types.i4)
    D2 = List.empty_list(types.i4)
    L = ind_arr.shape[0]
    msd_cutoff = rmsd_cutoff ** 2
    for k in range(L):
        low = ind_arr[k, 0]
        high = ind_arr[k, 1]
        for i in np.arange(low, high + 1, dtype=int32):
            D1.append(i)
            D2.append(i)
            for j in np.arange(high + 1, M, dtype=int32):
                msd = 0
                for a in range(N):
                    for b in range(O):
                        msd += (X[i, :, :][a, b] - X[j, :, :][a, b]) ** 2
                if 1.0 / N * msd < msd_cutoff:
                    D1.append(i)
                    D2.append(j)
                    D1.append(j)
                    D2.append(i)
    F = np.zeros((len(D1), 2), dtype=np.int32)
    for i in range(len(D1)):
        F[i, :] = D1[i], D2[i]
    return F


# @jit(nopython=True, cache=True)
# def get_max(x,y):
#     x = x.reshape(-1,3)
#     y = y.reshape(-1,3)
#     N = x.shape[0]
#     dists = np.zeros(N)
#     pp = 0
#     for a in range(N):
#         sub = np.subtract(x[a], y[a])
#         dists[pp] = np.sqrt(np.sum(np.square(sub)))
#         pp += 1
#     return np.max(dists)


@jit(nopython=True, fastmath=True, cache=True)
def get_max(x, y):
    x = x.reshape(-1, 3)
    y = y.reshape(-1, 3)
    N = x.shape[0]
    max_ = 0
    for a in range(N):
        dist = 0
        for i in range(3):
            dist += (x[a, i] - y[a, i])**2
        if dist > max_:
            max_ = dist
    return np.sqrt(max_)


@jit(nopython=True, fastmath=True, cache=True)
def get_rmsd(x, y):
    N = int(x.shape[0] / 3)
    O = 3
    x = x.reshape(N, O)
    y = y.reshape(N, O)
    m_com = np.zeros(O, dtype=np.float32)
    t_com = np.zeros(O, dtype=np.float32)
    m = np.zeros((N, O), dtype=np.float32)
    mtrans = np.zeros((O, N), dtype=np.float32)
    mtr = np.zeros((N, O), dtype=np.float32)
    t = np.zeros((N, O), dtype=np.float32)
    c = np.zeros((O, O), dtype=np.float32)
    U = np.zeros((O, O), dtype=np.float32)
    S = np.zeros(O, dtype=np.float32)
    Wt = np.zeros((O, O), dtype=np.float32)
    R = np.zeros((O, O), dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    sub = np.zeros(N * O, dtype=np.float32)
    for k in range(O):
        m_com[k] = np.mean(x[:, k])
        t_com[k] = np.mean(y[:, k])
    m = np.subtract(x, m_com)
    for a in range(N):
        for b in range(O):
            mtrans[b, a] = m[a, b]
    t = np.subtract(y, t_com)
    c = np.dot(mtrans, t)
    U, S, Wt = np.linalg.svd(c)
    R = np.dot(U, Wt)
    if np.linalg.det(R) < 0.0:
        Wt[-1, :] *= -1.0
        R = np.dot(U, Wt)
    mtr = np.add(np.dot(m, R), t_com)
    q = 0
    for a in range(N):
        for b in range(O):
            mtr_re[q] = mtr[a, b]
            t_re[q] = y[a, b]
            q += 1
    sub = np.subtract(mtr_re, t_re)
    D = np.sqrt(1.0 / N * np.dot(sub, sub))
    return D