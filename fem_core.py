"""fem_core.py - element stiffness matrices, assembly, boundary conditions, post-processing.

Node / element numbers are 1-based (as in the model tables) and converted to 0-based
only when indexing arrays.
"""
import numpy as np


# --------------------------------------------------------------------------- truss (3 dof/node)
def truss_global_stiffness(cord, conn, area, elast):
    """Global stiffness matrix of a space truss. Returns (K, element lengths)."""
    nn = cord.shape[0]
    K = np.zeros((3 * nn, 3 * nn))
    lengths = np.zeros(len(conn))
    for i in range(len(conn)):
        a, b = int(conn[i, 0]) - 1, int(conn[i, 1]) - 1
        Le = np.linalg.norm(cord[b, :] - cord[a, :])
        lengths[i] = Le
        l, m, n = (cord[a, :3] - cord[b, :3]) / Le          # direction cosines
        ke = (area[i] * elast[i] / Le) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        T = np.array([[l, m, n, 0, 0, 0],
                      [0, 0, 0, l, m, n]])
        Ke = T.T @ ke @ T
        assemble(K, Ke, a, b, 3)
    return K, lengths


# --------------------------------------------------------------------------- beam (6 dof/node)
def beam_element_stiffness(p1, p2, p3, A, E, Le, v=0.3):
    """12x12 global stiffness of a 3D Euler-Bernoulli beam with a solid circular section.
    p3 is the reference (3rd) node that defines the local z axis."""
    X1, Y1, Z1 = p1[:3]
    X2, Y2, Z2 = p2[:3]
    X3, Y3, Z3 = p3[:3]

    A123 = np.sqrt(((Y2 - Y1) * (Z3 - Z1) - (Y3 - Y1) * (Z2 - Z1)) ** 2
                   + ((Z2 - Z1) * (X3 - X1) - (Z3 - Z1) * (X2 - X1)) ** 2
                   + ((X2 - X1) * (Y3 - Y1) - (X3 - X1) * (Y2 - Y1)) ** 2)

    # local x: along the element
    lx = (X2 - X1) / Le
    mx = (Y2 - Y1) / Le
    nx = (Z2 - Z1) / Le
    # local z: normal to the plane of nodes 1, 2, 3
    lz = ((Y2 - Y1) * (Z3 - Z1) - (Y3 - Y1) * (Z2 - Z1)) / A123
    mz = ((Z2 - Z1) * (X3 - X1) - (Z3 - Z1) * (X2 - X1)) / A123
    nz = ((X2 - X1) * (Y3 - Y1) - (X3 - X1) * (Y2 - Y1)) / A123
    # local y = z x x
    ly = mz * nx - nz * mx
    my = nz * lx - lz * nx
    ny = lz * mx - mz * lx

    T3 = np.array([[lx, mx, nx], [ly, my, ny], [lz, mz, nz]])
    T = np.kron(np.eye(4), T3)

    a = Le / 2
    Iy = (np.pi * (np.sqrt(A / np.pi)) ** 4) / 4   # solid circular section
    Iz = Iy
    J = 2 * Iy
    G = E / (2 * (1 + v))

    c1 = (A * E) / (2 * a)
    c2 = (3 * E * Iz) / (2 * a ** 3)
    c3 = (3 * E * Iz) / (2 * a ** 2)
    c4 = (3 * E * Iy) / (2 * a ** 3)
    c5 = (3 * E * Iy) / (2 * a ** 2)
    c6 = (G * J) / (2 * a)
    c7 = (E * Iy) / a
    c8 = (E * Iz) / a

    ke = np.array([
        [c1, 0, 0, 0, 0, 0, -c1, 0, 0, 0, 0, 0],
        [0, c2, 0, 0, 0, c3, 0, -c2, 0, 0, 0, c3],
        [0, 0, c4, 0, -c5, 0, 0, 0, -c4, 0, -c5, 0],
        [0, 0, 0, c6, 0, 0, 0, 0, 0, -c6, 0, 0],
        [0, 0, -c5, 0, 2 * c7, 0, 0, 0, c5, 0, c7, 0],
        [0, c3, 0, 0, 0, 2 * c8, 0, -c3, 0, 0, 0, c8],
        [-c1, 0, 0, 0, 0, 0, c1, 0, 0, 0, 0, 0],
        [0, -c2, 0, 0, 0, -c3, 0, c2, 0, 0, 0, -c3],
        [0, 0, -c4, 0, c5, 0, 0, 0, c4, 0, c5, 0],
        [0, 0, 0, -c6, 0, 0, 0, 0, 0, c6, 0, 0],
        [0, 0, -c5, 0, c7, 0, 0, 0, c5, 0, 2 * c7, 0],
        [0, c3, 0, 0, 0, c8, 0, -c3, 0, 0, 0, 2 * c8]])
    return T.T @ ke @ T


def truss6_element_stiffness(p1, p2, A, E, Le):
    """Truss (bar) element written in 6-dof form, used for the crane's support members.
    The unit terms on the rotational dofs are scaled by EA/L as well."""
    l = (p2[0] - p1[0]) / Le
    m = (p2[1] - p1[1]) / Le
    n = (p2[2] - p1[2]) / Le
    M = np.array([
        [l*l, l*m, l*n, 0, 0, 0, -l*l, -l*m, -l*n, 0, 0, 0],
        [l*m, m*m, m*n, 0, 0, 0, -l*m, -m*m, -m*n, 0, 0, 0],
        [l*n, m*n, n*n, 0, 0, 0, -l*n, -m*n, -n*n, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [-l*l, -l*m, -l*n, 0, 0, 0, l*l, l*m, l*n, 0, 0, 0],
        [-l*m, -m*m, -m*n, 0, 0, 0, l*m, m*m, m*n, 0, 0, 0],
        [-l*n, -m*n, -n*n, 0, 0, 0, l*n, m*n, n*n, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
    return (E * A / Le) * M


# --------------------------------------------------------------------------- helpers
def assemble(K, Ke, a, b, dof):
    """Add the element matrix Ke (2*dof x 2*dof) of nodes a, b (0-based) to K."""
    j, k = dof * a, dof * b
    K[j:j+dof, j:j+dof] += Ke[0:dof, 0:dof]
    K[j:j+dof, k:k+dof] += Ke[0:dof, dof:2*dof]
    K[k:k+dof, j:j+dof] += Ke[dof:2*dof, 0:dof]
    K[k:k+dof, k:k+dof] += Ke[dof:2*dof, dof:2*dof]


def apply_penalty(K_stiff, BC, dof, value, mode):
    """Penalty method: modify the diagonal term of every restrained dof.
    mode='multiply' -> K(i,i) * value,  mode='add' -> K(i,i) + value."""
    K = K_stiff.copy()
    for i in range(BC.shape[0]):
        for d in range(dof):
            if BC[i, d] == 1:
                idx = dof * i + d
                if idx >= K.shape[0]:   # beam reference node is not part of K
                    continue
                if mode == "multiply":
                    K[idx, idx] = K[idx, idx] * value
                else:
                    K[idx, idx] = K[idx, idx] + value
    return K


def check_invertible(K):
    sign, _ = np.linalg.slogdet(K)
    if sign == 0:
        raise np.linalg.LinAlgError("Stiffness matrix not invertible - check boundary conditions")


def strains_stresses(newcord, conn, lengths, elast):
    """Deformed length, axial strain (L_new - L)/L and axial stress E*strain of every element."""
    ne = len(conn)
    new_len = np.zeros(ne)
    strain = np.zeros(ne)
    stress = np.zeros(ne)
    for i in range(ne):
        a, b = int(conn[i, 0]) - 1, int(conn[i, 1]) - 1
        new_len[i] = np.linalg.norm(newcord[b, :] - newcord[a, :])
        strain[i] = (new_len[i] - lengths[i]) / lengths[i]
        stress[i] = elast[i] * strain[i]
    return new_len, strain, stress
