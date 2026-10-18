"""calc_data.py - default problem data.

Units: lengths in mm, forces in N, stresses in N/mm^2 (MPa).
All values can be changed by the user in the web app.
"""
import math

F = 20000.0                 # crane tip load [N]
E = 210000.0                # Young's modulus [MPa] (steel)
A = 1848.0                  # tie-support position, x = -A [mm]
B = 1460.0                  # tie-support position, z = +B [mm]
ANGLE = math.radians(64.6)  # crane inclination from horizontal [rad]
A0 = 696.0                  # reference cross-section [mm^2]
AD = 0.5 * A0               # diagonal members [mm^2]  (348)
AY = 1.5 * A0               # main members [mm^2]      (1044)
L = 2490.0                  # member length [mm]


def calc_data():
    """Return the default (F, E, A, B, angle, A0, L, Ad, Ay)."""
    return F, E, A, B, ANGLE, A0, L, AD, AY
