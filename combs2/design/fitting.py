__all__ = ['score_fit', 'rigid_body', 'score_fit_1']

import numpy as np
# from numba import jit


# @jit('f8(f8[:,:], f8[:])')
def rigid_body_highdim(mob_coord, x):
    dim = int(mob_coord.shape[1]/3)
    return np.dot(mob_coord + np.matlib.repmat(np.array([x[3], x[4], x[5]]), 1, dim),
                  np.kron(np.eye(dim), R(x[0], x[1], x[2])).T)


# def rigid_body_highdim(mob_coord, x):
#     dim = int(mob_coord.shape[1]/3)
#     RR = R
#     repmat = np.matlib.repmat
#     array = np.array
#     eye = np.eye
#     kron = np.kron
#     dot = np.dot
#     return dot(mob_coord + repmat(array([x[3], x[4], x[5]]), 1, dim),
#                kron(eye(dim), RR(x[0], x[1], x[2])).T)

def rigid_body(mob_coord, x):
    return np.dot(mob_coord + x[3:], R(x[0], x[1], x[2]).T)


# def score_fit(mob_coords, densities, x):
#         return np.sum(-1 * density.score_samples(rigid_body_highdim(mob_coord, x))
#                       for mob_coord, density in zip(mob_coords, densities))


# def score_fit(mob_coords, densities):
#     return np.sum(-1 * density.score_samples(mob_coord)
#                   for mob_coord, density in zip(mob_coords, densities))

#can test this case by dropping the for loop?
# def score_fit(mob_coords_densities, x):
#     return np.sum(-1 * density.score_samples(rigid_body_highdim(mob_coord, x))
#                   for mob_coord, density in mob_coords_densities)
def score_fit(mob_coords_densities, x):
    return np.sum(-1 * density.logpdf(rigid_body_highdim(mob_coord, x))
                  for mob_coord, density in mob_coords_densities)
# def score_fit(mob_coords_densities, x):
#     return np.sum([-1 * mob_coords_densities[0][1].score_samples(rigid_body_highdim(mob_coords_densities[0][0], x)),
#                    -1 * mob_coords_densities[1][1].scsore_samples(rigid_body_highdim(mob_coords_densities[1][0], x)),
#                    -1 * mob_coords_densities[2][1].score_samples(rigid_body_highdim(mob_coords_densities[2][0], x))])

# def score_fit(mob_coords_densities, x):
#     return np.sum([-1 * mob_coords_densities[0][1].score_samples(rigid_body_highdim(mob_coords_densities[0][0], x)),
#                    -1 * mob_coords_densities[1][1].score_samples(rigid_body_highdim(mob_coords_densities[1][0], x))
#                    ])
#
# def score_fit_1(mob_coords_densities, x):
#     return -1 * mob_coords_densities[1].score_samples(rigid_body_highdim(mob_coords_densities[0], x))

# def score_fit(mob_coords_densities, x):
#     return np.sum([-1 * mob_coords_densities[0][1].logpdf(rigid_body_highdim(mob_coords_densities[0][0], x)),
#                    -1 * mob_coords_densities[1][1].logpdf(rigid_body_highdim(mob_coords_densities[1][0], x))
#                    ])

def score_fit_1(mob_coords_densities, x):
    return -1 * mob_coords_densities[1].logpdf(rigid_body_highdim(mob_coords_densities[0], x))

# def score_fit(len_mob_coords_densities, x):
#     return np.sum([-len_mob_coords_densities[0][0] * len_mob_coords_densities[0][2].score_samples(rigid_body_highdim(len_mob_coords_densities[0][1], x)),
#                    -len_mob_coords_densities[1][0] * len_mob_coords_densities[1][2].score_samples(rigid_body_highdim(len_mob_coords_densities[1][1], x)),
#                    -len_mob_coords_densities[2][0] * len_mob_coords_densities[2][2].score_samples(rigid_body_highdim(len_mob_coords_densities[2][1], x))])

# @jit
def sin(x):
    return np.sin(x)

# @jit
def cos(x):
    return np.cos(x)


# @jit
def R(phi, thet, psi):
    return np.array([[-sin(phi) * cos(thet) * sin(psi) + cos(phi) * cos(psi),
                      cos(phi) * cos(thet) * sin(psi) + sin(phi) * cos(psi), sin(thet) * sin(psi)],
                     [-sin(phi) * cos(thet) * cos(psi) - cos(phi) * sin(psi),
                      cos(phi) * cos(thet) * cos(psi) - sin(phi) * sin(psi), sin(thet) * cos(psi)],
                     [sin(phi) * sin(thet), -cos(phi) * sin(thet), cos(thet)]])

def R_vec(x):
    return np.array([[-sin(x[:, 0]) * cos(x[:, 1]) * sin(x[:, 2]) + cos(x[:, 0]) * cos(x[:, 2]),
                      cos(x[:, 0]) * cos(x[:, 1]) * sin(x[:, 2]) + sin(x[:, 0]) * cos(x[:, 2]), sin(x[:, 1]) * sin(x[:, 2])],
                     [-sin(x[:, 0]) * cos(x[:, 1]) * cos(x[:, 2]) - cos(x[:, 0]) * sin(x[:, 2]),
                      cos(x[:, 0]) * cos(x[:, 1]) * cos(x[:, 2]) - sin(x[:, 0]) * sin(x[:, 2]), sin(x[:, 1]) * cos(x[:, 2])],
                     [sin(x[:, 0]) * sin(x[:, 1]), -cos(x[:, 0]) * sin(x[:, 1]), cos(x[:, 1])]])


# def t(x, y, z, num_atoms):
#     #num_atoms = c.shape[1]
#     return np.matlib.repmat(np.array([x, y, z]), 1, num_atoms)
#
# def K(x, y, z, ph, th, ps, c):
#     # c is query coords matrix
#     return np.dot( (c + t(x, y, z)), R(ph, th, ps).T )
#
# def fit(w, kde, x, y, z, ph, th, ps, c):
#     return np.dot(kde.score_samples(K(x, y, z, ph, th, ps, c)), w) + np.sum(w)