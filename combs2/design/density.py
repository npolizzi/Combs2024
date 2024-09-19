import numpy as np
from numba import jit

# def rigid_body_highdim(mob_coord, x):
#     dim = int(mob_coord.shape[1]/3)
#     return np.dot(mob_coord + np.matlib.repmat(np.array([x[3], x[4], x[5]]), 1, dim),
#                   np.kron(np.eye(dim), R(x[0], x[1], x[2])).T)


def rigid_body_highdim(mob_coord, x):
    dim = int(mob_coord.shape[0]/3)
    return np.dot(mob_coord + np.matlib.repmat(np.array([x[3], x[4], x[5]]), 1, dim),
                  np.kron(np.eye(dim), R(x[0], x[1], x[2])).T)


@jit(nopython=True, cache=True)
def rigid_body(mob_coord, x):
    return np.dot(mob_coord + x[3:], R(x[0], x[1], x[2]).T)


@jit(nopython=True, cache=True)
def apply_rigid_body_to_coords_cols(coords_cols, x):
    new_coords_cols = np.zeros(coords_cols.shape)
    for i in range(0, coords_cols.shape[1], 3):
        new_coords_cols[:, i:i + 3] = rigid_body(coords_cols[:, i:i + 3], x)
    return new_coords_cols

# def score_fit(mob_coords_densities, x):
#     return sum(-1 * density.logpdf(rigid_body_highdim(mob_coord, x))
#                   for mob_coord, density in mob_coords_densities)


def score_fit(mob_coords_densities, x):
    return sum(-1 * density.score(rigid_body_highdim(mob_coord, x))
                  for mob_coord, density in mob_coords_densities)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


@jit(nopython=True, cache=True)
def R(phi, thet, psi):
    return np.array([[-np.sin(phi) * np.cos(thet) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                      np.cos(phi) * np.cos(thet) * np.sin(psi) + np.sin(phi) * np.cos(psi), 
                      np.sin(thet) * np.sin(psi)],
                     [-np.sin(phi) * np.cos(thet) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                      np.cos(phi) * np.cos(thet) * np.cos(psi) - np.sin(phi) * np.sin(psi), 
                      np.sin(thet) * np.cos(psi)],
                     [np.sin(phi) * np.sin(thet), 
                     -np.cos(phi) * np.sin(thet), 
                     np.cos(thet)]])


