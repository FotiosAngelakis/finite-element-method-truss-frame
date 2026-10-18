"""models.py - geometry, supports, default loads and element tables of the four structures.

Every build function returns a dict with
    cord  (n_nodes x 3 or 6)  node coordinates [mm]
    BC    (n_nodes x 3 or 6)  1 = restrained dof
    F     (n_nodes x 3 or 6)  nodal forces [N] / moments [N*mm]
    conn  (n_el x 2 or 3)     element nodes (1-based); 3rd column = beam reference node
    area, elast               cross-section [mm^2] and Young's modulus [MPa] per element
data = (F, E, A, B, angle, A0, L, Ad, Ay); if omitted the default values are used.

Inside the crane builders arrays get one extra row so the 1-based node numbers can be
used directly; row 0 is dropped before returning.
"""
import numpy as np

from calc_data import calc_data


def _data(data):
    return data if data is not None else calc_data()


def _cube_coordinates(cord, L):
    """8 corners of a cube of side L, origin at node 1."""
    for i in range(1, 9):
        cord[i - 1, 0] = L if i % 2 == 0 else 0
        cord[i - 1, 1] = 0 if i in (1, 2, 5, 6) else L
        cord[i - 1, 2] = 0 if i <= 4 else L


def _cube_supports(BC):
    BC[0, 0:3] = 1      # node 1: x, y, z
    BC[1, 1:3] = 1      # node 2: y, z
    BC[2, 0] = 1        # node 3: x, z
    BC[2, 2] = 1
    BC[3, 2] = 1        # node 4: z


# =============================================================================== TRUSS CUBE
def build_truss_cube(data=None):
    F, E, A, B, angle, A0, L, Ad, Ay = _data(data)
    cord = np.zeros((8, 3))
    _cube_coordinates(cord, L)
    BC = np.zeros((8, 3))
    _cube_supports(BC)
    Fn = np.zeros((8, 3))
    Fn[6, 2] = -2000                       # default load: node 7, -z

    # 24 bars: 12 edges (Ay) + 12 diagonals (Ad). Row 0 unused (1-based).
    N = np.zeros((25, 2), dtype=int)
    Ar = np.zeros(25)
    for i in range(1, 8):
        if i == 1:
            for j in range(1, 7):
                N[j] = (i, j + 1)
                Ar[j] = Ay if N[j, 1] in (2, 3, 5) else Ad
        elif i == 2:
            for j in range(7, 12):
                N[j] = (i, j - 4)
                if j == 11:
                    N[j, 1] = j - 3
                Ar[j] = Ay if N[j, 1] in (4, 6) else Ad
        elif i == 3:
            for j in range(12, 16):
                N[j] = (i, j - 8) if j in (12, 13) else (i, j - 7)
                Ar[j] = Ay if N[j, 1] in (4, 7) else Ad
        elif i == 4:
            for j in range(16, 19):
                N[j] = (i, j - 10)
                Ar[j] = Ay if N[j, 1] == 8 else Ad
        elif i == 5:
            for j in range(19, 22):
                N[j] = (i, j - 13)
                Ar[j] = Ay if N[j, 1] in (6, 7) else Ad
        elif i == 6:
            N[22] = (6, 7); Ar[22] = Ad
            N[23] = (6, 8); Ar[23] = Ay
        else:
            N[24] = (7, 8); Ar[24] = Ay

    return dict(cord=cord, BC=BC, F=Fn, conn=N[1:], area=Ar[1:], elast=E * np.ones(24),
                Ay=Ay)


# =============================================================================== BEAM CUBE
def build_beam_cube(data=None):
    F, E, A, B, angle, A0, L, Ad, Ay = _data(data)
    cord = np.zeros((9, 6))
    _cube_coordinates(cord, L)
    cord[8, 0:3] = (0, -L, 2 * L)          # node 9: orientation reference node
    BC = np.zeros((9, 6))
    _cube_supports(BC)
    Fn = np.zeros((9, 6))
    Fn[6, 2] = -2000

    # 12 edge beams; the diagonals are removed and their area is smeared into the edges
    ly = np.linalg.norm(cord[0] - cord[1])
    ld = np.linalg.norm(cord[0] - cord[3])
    AY = (Ad * ld + Ay * ly) / ly
    conn = np.zeros((12, 3), dtype=int)
    C = np.array([1, 2, 4, 3])
    for i in range(0, 4):                  # lower plane
        conn[i, 0:2] = (C[0], C[1])
        C = np.roll(C, -1)
    for i in range(4, 8):                  # vertical
        conn[i, 0:2] = (C[0], C[0] + 4)
        C = np.roll(C, -1)
    C = C + 4
    for i in range(8, 12):                 # upper plane
        conn[i, 0:2] = (C[0], C[1])
        C = np.roll(C, -1)
    conn[:, 2] = 9

    return dict(cord=cord, BC=BC, F=Fn, conn=conn, area=AY * np.ones(12),
                elast=E * np.ones(12), AY=AY)


# =============================================================================== CRANE helpers
def _crane_main_nodes(cord, L, HP):
    """Base, main body (HP square planes) and head nodes. cord is 1-based."""
    cord[1, 0:3] = 0
    cord[2, 0:3] = 0
    cord[2, 1] = L
    C = np.array([3, 4, 5, 6])
    b = 0
    Lz = 0
    for _ in range(HP):
        cord[C + b, 0] = -L / 2
        cord[C[0:2] + b, 1] = 0
        cord[C[1:3] + b, 0] = L / 2
        cord[C[2:4] + b, 1] = L
        cord[C + b, 2] = L + Lz
        Lz += L
        b += 4
    cord[C[0] + b, 0:3] = cord[C[0] + b - 3, 0:3]
    cord[C[0] + b, 2] = cord[C[0] + b - 3, 2] + L
    cord[C[1] + b, 0:3] = cord[C[1] + b - 3, 0:3]
    cord[C[1] + b, 2] = cord[C[1] + b - 3, 2] + L
    cord[C[2] + b, 0] = 3 * L / 2
    cord[C[2] + b, 1] = L / 2
    cord[C[2] + b, 2] = cord[b, 2] + L / 2


def _rotate(cord, first, last, angle):
    """Tilt nodes first..last (1-based, inclusive) about the y axis."""
    c, s = np.cos(np.pi / 2 - angle), np.sin(np.pi / 2 - angle)
    for i in range(first, last + 1):
        X = cord[i, 0] * c + cord[i, 2] * s
        Z = cord[i, 2] * c - cord[i, 0] * s
        cord[i, 0] = X
        cord[i, 2] = Z


def _truss_crane_elements(n_nodes, HP, Ad, Ay):
    """Bar connectivity & areas of the truss crane (1-based)."""
    n_el = 18 * (HP - 1) + 32
    N = np.zeros((n_el + 1, 2), dtype=int)
    Ar = np.zeros(n_el + 1)

    # base
    C = np.array([3, 4, 5, 6])
    b = 0
    N[1] = (1, 2); Ar[1] = Ay
    for i in (1, 2):
        for j in range(2, 6):
            N[j + b] = (i, C[j - 2])
            Ar[j + b] = Ad if j >= 4 else Ay
        C = np.roll(C, -2)
        b += 4

    # main body
    C = np.array([3, 4, 5, 6])
    b = 0
    for _ in range(HP - 1):
        for i in range(10 + b, 14 + b):            # plane edges
            N[i] = (C[0], C[1]); Ar[i] = Ay
            C = np.roll(C, -1)
        a = 0
        for i in range(14 + b, 16 + b):            # plane diagonals
            N[i] = (C[0 + a], C[2 + a]); Ar[i] = Ad
            a += 1
        for i in range(16 + b, 20 + b):            # verticals
            N[i] = (C[0], C[0] + 4); Ar[i] = Ay
            C = np.roll(C, -1)
        for i in range(20 + b, 24 + b):            # side diagonals
            N[i] = (C[0], C[0] + 1) if i == 23 + b else (C[0], C[0] + 5)
            Ar[i] = Ad
            C = np.roll(C, -1)
        for i in range(24 + b, 28 + b):
            N[i] = (C[0], C[0] + 7) if i == 24 + b else (C[0], C[0] + 3)
            Ar[i] = Ad
            C = np.roll(C, -1)
        C = C + 4
        b += 18

    # top plane
    for i in range(10 + b, 14 + b):
        N[i] = (C[0], C[1]); Ar[i] = Ay
        C = np.roll(C, -1)
    a = 0
    for i in range(14 + b, 16 + b):
        N[i] = (C[0 + a], C[2 + a]); Ar[i] = Ad
        a += 1

    # head
    for i in range(16 + b, 20 + b):
        N[i] = (C[0], n_nodes - 5)
        Ar[i] = Ad if i >= 18 + b else Ay
        C = np.roll(C, -1)
    for i in range(20 + b, 24 + b):
        N[i] = (C[0], n_nodes - 4)
        Ar[i] = Ay if i >= 22 + b else Ad
        C = np.roll(C, -1)
    N[24 + b] = (n_nodes - 5, n_nodes - 4); Ar[24 + b] = Ay
    C = np.array([24, 25, 27, 28]) + (HP - 6) * 4
    for i in range(25 + b, 29 + b):
        N[i] = (C[0], n_nodes - 3); Ar[i] = Ay
        C = np.roll(C, -1)

    # support members
    N[29 + b] = (n_nodes - 2, 11); Ar[29 + b] = Ad
    N[30 + b] = (n_nodes, 14); Ar[30 + b] = Ad
    N[31 + b] = (n_nodes - 1, C[0] - 1)
    N[32 + b] = (n_nodes - 1, C[2] - 1)
    Ar[31 + b:33 + b] = Ad
    return N[1:], Ar[1:]


# =============================================================================== TRUSS CRANE
def build_truss_crane(HP=6, data=None):
    F, E, A, B, angle, A0, L, Ad, Ay = _data(data)
    n_nodes = 8 + HP * 4

    cord = np.zeros((n_nodes + 1, 3))
    _crane_main_nodes(cord, L, HP)
    cord[n_nodes - 2:n_nodes + 1, 0] = -A          # 3 support nodes
    cord[n_nodes - 2:n_nodes + 1, 2] = B
    Ly = 0
    for i in range(n_nodes - 2, n_nodes + 1):
        cord[i, 1] = Ly
        Ly += L / 2
    _rotate(cord, 3, n_nodes - 3, angle)

    BC = np.zeros((n_nodes + 1, 3))
    BC[1, :] = 1
    BC[2, 0:3] = 1
    BC[2, 1] = 0
    BC[n_nodes - 2:n_nodes + 1, :] = 1

    Fn = np.zeros((n_nodes + 1, 3))
    Fn[n_nodes - 3, 2] = -F                         # load at the crane tip

    conn, area = _truss_crane_elements(n_nodes, HP, Ad, Ay)
    return dict(cord=cord[1:], BC=BC[1:], F=Fn[1:], conn=conn, area=area,
                elast=E * np.ones(len(conn)), Ay=Ay)


# =============================================================================== BEAM CRANE
def build_beam_crane(HP=6, data=None):
    F, E, A, B, angle, A0, L, Ad, Ay = _data(data)
    n_nodes = 9 + HP * 4
    nn = n_nodes

    cord = np.zeros((nn + 1, 6))
    _crane_main_nodes(cord, L, HP)
    cord[nn - 3:nn, 0] = -A                         # 3 support nodes
    cord[nn - 3:nn, 2] = B
    Ly = 0
    for i in range(nn - 3, nn):
        cord[i, 1] = Ly
        Ly += L / 2
    cord[nn, 0:2] = L / 4                           # orientation reference node
    cord[nn, 2] = 5 * L
    _rotate(cord, 3, nn - 4, angle)

    BC = np.zeros((nn + 1, 6))
    BC[1, 0:3] = 1
    BC[2, 0:3] = 1
    BC[2, 1] = 0
    BC[nn - 3:nn, 0:3] = 1

    Fn = np.zeros((nn + 1, 6))
    Fn[nn - 4, 2] = -F

    # equivalent beam section: diagonal length is smeared into the main members
    tconn, tarea = _truss_crane_elements(nn, HP, Ad, Ay)
    sum_Ad = 0.0
    sum_Ay = 0.0
    for i in range(len(tconn) - 4):
        Lg = np.linalg.norm(cord[tconn[i, 1]] - cord[tconn[i, 0]])
        if tarea[i] == Ad:
            sum_Ad += Lg
        else:
            sum_Ay += Lg
    AY = Ad * sum_Ad / sum_Ay + Ay

    n_el = 8 * (HP - 1) + 22
    N = np.zeros((n_el + 1, 3), dtype=int)
    Ar = np.zeros(n_el + 1)

    C = np.array([3, 4, 5, 6])                      # base
    b = 0
    N[1] = (1, 2, nn); Ar[1] = AY
    for i in (1, 2):
        for j in (2, 3):
            N[j + b] = (i, C[j - 2], nn); Ar[j + b] = AY
        C = np.roll(C, -2)
        b += 2

    C = np.array([3, 4, 5, 6])                      # main body
    b = 0
    for _ in range(HP - 1):
        for i in range(6 + b, 10 + b):
            N[i] = (C[0], C[1], nn); Ar[i] = AY
            C = np.roll(C, -1)
        for i in range(10 + b, 14 + b):
            N[i] = (C[0], C[0] + 4, nn); Ar[i] = AY
            C = np.roll(C, -1)
        C = C + 4
        b += 8
    for i in range(6 + b, 10 + b):                  # top plane
        N[i] = (C[0], C[1], nn); Ar[i] = AY
        C = np.roll(C, -1)

    for i in range(10 + b, 12 + b):                 # head
        N[i] = (C[0], nn - 6, nn); Ar[i] = AY
        C = np.roll(C, -1)
    for i in range(12 + b, 14 + b):
        N[i] = (C[0], nn - 5, nn); Ar[i] = AY
        C = np.roll(C, -1)
    N[14 + b] = (nn - 6, nn - 5, nn); Ar[14 + b] = AY
    C = np.array([24, 25, 27, 28]) + (HP - 6) * 4
    for i in range(15 + b, 19 + b):
        N[i] = (C[0], nn - 4, nn); Ar[i] = AY
        C = np.roll(C, -1)

    # support members (last 4 elements, also given truss stiffness by the solver)
    N[19 + b] = (nn - 3, 11, nn); Ar[19 + b] = Ad
    N[20 + b] = (nn - 1, 14, nn); Ar[20 + b] = Ad
    N[21 + b] = (nn - 2, C[0] - 1, nn)
    N[22 + b] = (nn - 2, C[2] - 1, nn)
    Ar[21 + b:23 + b] = Ad

    return dict(cord=cord[1:], BC=BC[1:], F=Fn[1:], conn=N[1:], area=Ar[1:],
                elast=E * np.ones(n_el), AY=AY)
