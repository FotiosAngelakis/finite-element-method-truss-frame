"""solvers.py - linear static solution of the four structures.

Each solver: assemble K -> penalty boundary conditions -> solve K u = F ->
reactions R = K_unrestrained u -> deformed lengths, axial strains and stresses.
"""
import numpy as np

import fem_core


def _truss(cord, BC, F, conn, area, elast, penalty, mode):
    K_stiff, lengths = fem_core.truss_global_stiffness(cord, conn, area, elast)
    K = fem_core.apply_penalty(K_stiff, BC, dof=3, value=penalty, mode=mode)
    fem_core.check_invertible(K)
    u = np.linalg.solve(K, F.reshape(-1))
    Rs = K_stiff @ u
    U = u.reshape(-1, 3)
    newcord = cord + U
    new_len, strain, stress = fem_core.strains_stresses(newcord, conn, lengths, elast)
    R = Rs.reshape(-1, 3)
    Reactions = np.column_stack([R, np.sqrt((R ** 2).sum(axis=1))])
    return dict(U=U, newcord=newcord, lengths=lengths, new_len=new_len,
                strain=strain, stress=stress, Reactions=Reactions, K=K)


def solve_truss_cube(cord, BC, F, conn, area, elast):
    return _truss(cord, BC, F, conn, area, elast, penalty=1e8, mode="multiply")


def solve_truss_crane(cord, BC, F, conn, area, elast):
    return _truss(cord, BC, F, conn, area, elast, penalty=1e7, mode="add")


def _beam(cord, BC, F, conn, area, elast, truss_supports, v=0.3):
    """The last node is the orientation reference node: it is not part of K."""
    nn = cord.shape[0]
    ne = len(conn)
    K_stiff = np.zeros(((nn - 1) * 6, (nn - 1) * 6))
    lengths = np.zeros(ne)

    for i in range(ne):
        a, b, c = (int(x) - 1 for x in conn[i, :3])
        lengths[i] = np.linalg.norm(cord[b, :] - cord[a, :])
        Ke = fem_core.beam_element_stiffness(cord[a], cord[b], cord[c],
                                             area[i], elast[i], lengths[i], v)
        fem_core.assemble(K_stiff, Ke, a, b, 6)

    if truss_supports:   # crane: the 4 support members also get a bar stiffness
        for i in range(ne - 4, ne):
            a, b = int(conn[i, 0]) - 1, int(conn[i, 1]) - 1
            Ke = fem_core.truss6_element_stiffness(cord[a], cord[b], area[i], elast[i],
                                                   lengths[i])
            fem_core.assemble(K_stiff, Ke, a, b, 6)

    K = fem_core.apply_penalty(K_stiff, BC, dof=6, value=1e7, mode="add")
    fem_core.check_invertible(K)
    u = np.linalg.solve(K, F[:nn - 1, :].reshape(-1))
    Rs = K_stiff @ u

    U = np.vstack([u.reshape(-1, 6), np.zeros((1, 6))])
    newcord = np.zeros((nn, 6))
    newcord[:nn - 1] = cord[:nn - 1] + U[:nn - 1]
    new_len, strain, stress = fem_core.strains_stresses(newcord, conn, lengths, elast)

    R = Rs.reshape(-1, 6)
    Reactions = np.column_stack([R, np.sqrt((R[:, 0:3] ** 2).sum(axis=1)),
                                 np.sqrt((R[:, 3:6] ** 2).sum(axis=1))])
    return dict(U=U, newcord=newcord, lengths=lengths, new_len=new_len,
                strain=strain, stress=stress, Reactions=Reactions, K=K)


def solve_beam_cube(cord, BC, F, conn, area, elast):
    return _beam(cord, BC, F, conn, area, elast, truss_supports=False)


def solve_beam_crane(cord, BC, F, conn, area, elast):
    return _beam(cord, BC, F, conn, area, elast, truss_supports=True)
