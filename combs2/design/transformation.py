__all__ = ['get_rot_trans']


import numpy as np
from numba import jit
from .constants import coords_cols


# @jit(nopython=True)
# def mean_numba(a):
#     N = a.shape[1]
#     res = np.zeros(N)
#     for i in range(N):
#         res[i] = a[:, i].mean()
#     return res
#
# @jit(nopython=True)
# def get_rot_trans(mob_coords, targ_coords):
#     mob_coords_com = mean_numba(mob_coords)
#     targ_coords_com = mean_numba(targ_coords)
#     mob_coords_cen = mob_coords - mob_coords_com
#     targ_coords_cen = targ_coords - targ_coords_com
#     cov_matrix = np.dot(mob_coords_cen.T, targ_coords_cen)
#     U, S, Wt = np.linalg.svd(cov_matrix)
#     R = np.dot(U, Wt)
#     if np.linalg.det(R) < 0.:
#         Wt[-1] *= -1
#         R = np.dot(U, Wt)
#     return R, mob_coords_com, targ_coords_com

@jit(nopython=True)
def mean_numba(a, weights=None):
    """weight should be normalized"""
    N = a.shape[1]
    M = a.shape[0]
    if weights is None:
        weights = np.ones(M) / M
    if weights.shape[0] != M:
        raise Exception('Weights must be same size as coordinates')
    res = np.zeros(N)
    for i in range(N):
        res[i] = 0
        for j in range(M):
            res[i] += a[j, i]*weights[j]
    return res


@jit(nopython=True, cache=True)
def get_rot_trans(mob_coords, targ_coords, weights=None):
    mob_coords_com = mean_numba(mob_coords, weights=weights)
    targ_coords_com = mean_numba(targ_coords, weights=weights)
    mob_coords_cen = mob_coords - mob_coords_com
    targ_coords_cen = targ_coords - targ_coords_com
    M = mob_coords.shape[0]
    if weights is None:
        weights = np.ones(M) / M
    for i in range(M):
        mob_coords_cen[i, :] = mob_coords_cen[i, :] * weights[i]
        targ_coords_cen[i, :] = targ_coords_cen[i, :] * weights[i]
    cov_matrix = np.dot(mob_coords_cen.T, targ_coords_cen)
    U, S, Wt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Wt)
    if np.linalg.det(R) < 0.:
        Wt[-1] *= -1
        R = np.dot(U, Wt)
    return R, mob_coords_com, targ_coords_com


def apply_transform(R, mob_coords_com, targ_coords_com, coords):
    return np.dot(coords - mob_coords_com, R) + targ_coords_com


@jit(nopython=True, cache=True)
def apply_transform_to_coords_cols(R, mob_coords_com, targ_coords_com, coords_cols):
    new_coords_cols = np.zeros(coords_cols.shape)
    for i in range(0, coords_cols.shape[1], 3):
        new_coords_cols[:, i:i + 3] = np.dot(coords_cols[:, i:i + 3] - mob_coords_com, R) + targ_coords_com
    return new_coords_cols


def superpose_df(mob_coords, targ_coords, df_mob, weights=None, copy_df_mob=False):
    if copy_df_mob:
        df_mob = df_mob.copy()
    R, mob_coords_com, targ_coords_com = get_rot_trans(mob_coords, targ_coords, weights=weights)
    for i in range(0, len(coords_cols), 3):
        df_mob[coords_cols[i:i + 3]] = np.dot(df_mob[coords_cols[i:i + 3]] - mob_coords_com, R) + targ_coords_com
    return df_mob


def superpose_coords(mob_coords, targ_coords, weights=None, apply_to=None):
    R, mob_coords_com, targ_coords_com = get_rot_trans(mob_coords, targ_coords, weights=weights)
    if apply_to is None:
        return np.dot(mob_coords - mob_coords_com, R) + targ_coords_com
    else:
        return np.dot(apply_to - mob_coords_com, R) + targ_coords_com


# def get_rot_trans(mob_coords, targ_coords):
#     mob_coords_com = mob_coords.mean(0)
#     targ_coords_com = targ_coords.mean(0)
#     mob_coords_cen = mob_coords - mob_coords_com
#     targ_coords_cen = targ_coords - targ_coords_com
#     cov_matrix = np.dot(mob_coords_cen.T, targ_coords_cen)
#     U, S, Wt = np.linalg.svd(cov_matrix)
#     R = np.dot(U, Wt)
#     if np.linalg.det(R) < 0.:
#         Wt[-1] *= -1
#         R = np.dot(U, Wt)
#     return R, mob_coords_com, targ_coords_com


def get_rot(mob_coords, targ_coords):
    cov_matrix = np.dot(mob_coords.T, targ_coords)
    U, S, Wt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Wt)
    if np.linalg.det(R) < 0.:
        Wt[-1] *= -1
        R = np.dot(U, Wt)
    return R


def get_rotation_axis(R, normalized=False):
    """rotation matrix R"""
    u = np.zeros(3)
    u[0] = R[2, 1] - R[1, 2]
    u[1] = R[0, 2] - R[2, 0]
    u[2] = R[1, 0] - R[0, 1]
    if normalized:
        u = u / np.linalg.norm(u)
    return u


def get_rotation_angle(R):
    """rotation matrix R"""
    return np.arccos((np.trace(R) - 1) / 2)


def rotate_by_angle(u, angle):
    """Takes rotation axis u and angle to rotate about this axis, returns rotation matrix."""
    norm = np.linalg.norm(u)
    if norm != 1:
        u = u / norm
    R = np.zeros((3, 3))
    ux, uy, uz = u
    cos = np.cos
    sin = np.sin
    R[0, 0] = cos(angle) + ux**2 * (1 - cos(angle))
    R[0, 1] = ux*uy*(1-cos(angle)) - uz*sin(angle)
    R[0, 2] = ux*uz*(1-cos(angle)) + uy*sin(angle)
    R[1, 0] = uy*ux*(1-cos(angle)) + uz*sin(angle)
    R[1, 1] = cos(angle) + uy**2*(1 - cos(angle))
    R[1, 2] = uy*uz*(1-cos(angle)) - ux*sin(angle)
    R[2, 0] = uz*ux*(1-cos(angle)) - uy*sin(angle)
    R[2, 1] = uz*uy*(1-cos(angle)) + ux*sin(angle)
    R[2, 2] = cos(angle) + uz**2*(1 - cos(angle))
    return R

