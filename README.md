# 3D Truss & Frame Finite Element Analysis

An interactive web application for the **linear static finite element analysis of 3D
structures**. Four structural models – a cube and a lattice crane, each modelled either as a
**space truss** or as a **3D frame (beam)** – can be configured in the browser. The app solves
them with the direct stiffness method and shows displacements, axial strains, axial stresses
and support reactions as interactive 3D plots and tables.

> **Live demo:** (https://finite-element-method-truss-frame.streamlit.app/)

---

## Contents

1. [Features](#features)
2. [The structural models](#the-structural-models)
3. [Finite element formulation](#finite-element-formulation)
4. [Code structure](#code-structure)
5. [Using the app](#using-the-app)
6. [Running locally](#running-locally)
7. [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
8. [Checking the results](#checking-the-results)
9. [Assumptions and limitations](#assumptions-and-limitations)
10. [Possible extensions](#possible-extensions)

---

## Features

- Four parametric structures: **Truss Cube, Beam Cube, Truss Crane, Beam Crane**
- Editable geometry and material: member length, crane inclination, support positions,
  cross-section areas, Young's modulus, crane height (number of panels)
- Editable **loads** (any number of nodal forces and, for beam models, moments)
- Editable **boundary conditions** (fix x / y / z at any node)
- Interactive 3D plots (rotate, zoom, hover):
  - undeformed structure with supports and loads
  - undeformed vs. deformed shape (adjustable deformation scale)
  - elements coloured by axial strain and by axial stress
  - elements coloured by nodal displacement (Ux, Uy, Uz, total), interpolated along each member
- Summary metrics: maximum displacement, maximum tensile and compressive stress,
  vertical equilibrium check (sum of reactions vs. applied load)
- Result tables for nodes and elements, downloadable as CSV
- Detection of unsupported (mechanism) structures

Units throughout: **mm, N, N/mm² (MPa)**; moments in N·mm.

---

## The structural models

The default dimensions and section properties are defined in `calc_data.py`.
Every value can be changed in the app.

| Parameter | Symbol | Default | Used in |
|---|---|---|---|
| Member length (cube side / crane panel size) | L | 2490 mm | all |
| Area of main members (edges, chords, verticals) | Ay = 1.5·A0 | 1044 mm² | all |
| Area of diagonal members | Ad = 0.5·A0 | 348 mm² | all |
| Young's modulus | E | 210 000 MPa (steel) | all |
| Poisson's ratio | ν | 0.3 (fixed) | beam models |
| Crane inclination (from horizontal) | θ | 64.6° | cranes |
| Support position of the tie members | A, B | 1848 mm, 1460 mm | cranes |
| Load at the crane tip | F | 20 000 N (downward) | cranes |
| Load on the cube | – | 2 000 N downward at node 7 | cubes |

### Truss Cube
- 8 nodes at the corners of an L × L × L cube (origin at node 1).
- 24 pin-jointed bars: the 12 edges (area Ay) and two crossing diagonals on each of the
  6 faces (area Ad).
- Supports: node 1 fixed in x, y, z; node 2 in y, z; node 3 in x, z; node 4 in z
  (six restraints, enough to prevent rigid-body motion).
- Load: 2 000 N downward at top corner node 7.

### Beam Cube
- The same cube built from 12 **rigidly connected beams** (the edges only).
- The face diagonals are removed and their material is added to the edges, giving an
  equivalent area  AY = (Ad·L√2 + Ay·L) / L.
- A 9th node at (0, −L, 2L) is used only to orient the beam cross-sections
  (it carries no degrees of freedom).
- Same supports and load as the truss cube.

### Truss Crane
- A lattice mast of **HP square panels** (default 6), each L × L in plan and L high, on a
  two-node base, topped by a head that ends in a cantilevered tip.
- The whole mast is tilted about the y axis to the inclination θ.
- Each panel has edge members and vertical members (Ay) plus plan and side diagonals (Ad).
- Three tie supports at x = −A, z = B (y = 0, L/2, L) are connected to the mast by four
  bars that brace it.
- Supports: base node 1 fixed in x, y, z; base node 2 fixed in x, z (free to slide in y);
  the three tie nodes fully fixed.
- Load: F downward at the crane tip.
- Size: 8 + 4·HP nodes and 18·(HP − 1) + 32 bars (32 nodes / 122 bars for HP = 6).

### Beam Crane
- The same crane as a 3D frame: the diagonals are removed and the remaining members are
  beams with an equivalent area AY = Ay + Ad·(ΣL_diagonals / ΣL_main).
- The four tie members are treated as bars (axial members).
- An extra node above the crane, at (L/4, L/4, 5L), orients all beam cross-sections.
- Size: 9 + 4·HP nodes and 8·(HP − 1) + 22 elements (33 nodes / 62 elements for HP = 6).

---

## Finite element formulation

### Space truss element (3 dof per node)

Each bar carries axial force only. In local coordinates

$$
\mathbf{k}_e = \frac{EA}{L}\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}
$$

and with the direction cosines $(l, m, n)$ of the bar axis the transformation is

$$
\mathbf{T} = \begin{bmatrix} l & m & n & 0 & 0 & 0\\ 0 & 0 & 0 & l & m & n \end{bmatrix},
\qquad \mathbf{K}_e = \mathbf{T}^\mathsf{T}\mathbf{k}_e\mathbf{T}\;(6\times6).
$$

### 3D beam element (6 dof per node)

Two-node Euler–Bernoulli frame element with dofs $(u, v, w, \theta_x, \theta_y, \theta_z)$ at
each end. The 12×12 local stiffness matrix combines

- axial stiffness $EA/L$,
- torsional stiffness $GJ/L$, with $G = E / 2(1+\nu)$,
- bending about the local y and z axes ($12EI/L^3$, $6EI/L^2$, $4EI/L$, $2EI/L$ terms).

Cross-sections are **solid circles**, so $I_y = I_z = A^2 / 4\pi$ and $J = 2I$.

The local axes are built from three points: local x runs from node 1 to node 2, local z is
normal to the plane through nodes 1, 2 and the reference node, and local y completes the
right-handed system. The 3×3 matrix of direction cosines is repeated four times on the
diagonal of $\mathbf{T}$ (12×12) and $\mathbf{K}_e = \mathbf{T}^\mathsf{T}\mathbf{k}_e\mathbf{T}$.

### Assembly and boundary conditions

Element matrices are added into the global stiffness matrix $\mathbf{K}$ by node numbers
(direct stiffness method). Supports are applied with the **penalty method**: the diagonal term
of every restrained dof is made very large (×10⁸ for the truss cube, +10⁷ for the other
models), which forces that displacement to (almost) zero without removing rows and columns.

### Solution and post-processing

1. Solve $\mathbf{K}_{\text{penalty}}\,\mathbf{u} = \mathbf{F}$ for the nodal displacements.
2. Reactions: $\mathbf{R} = \mathbf{K}\,\mathbf{u}$ with the **unmodified** stiffness matrix.
   At free nodes this returns the applied loads; at supported nodes it gives the reactions.
3. Deformed coordinates $\mathbf{x}' = \mathbf{x} + \mathbf{u}$ and deformed member lengths $L'$.
4. Axial strain $\varepsilon = (L' - L)/L$ and axial stress $\sigma = E\,\varepsilon$
   (positive = tension, negative = compression).

---

## Code structure

```
├── app.py            Streamlit user interface: inputs, solution call, Plotly 3D plots, tables
├── models.py         Geometry, connectivity, sections, supports and default loads of the 4 structures
├── solvers.py        Truss and beam solvers: assembly, penalty BCs, solution, strains, stresses, reactions
├── fem_core.py       Element stiffness matrices (truss, 3D beam), assembly, penalty method, strain/stress
├── calc_data.py      Default problem data (dimensions, areas, material, loads)
└── requirements.txt  Python dependencies
```

Data flow:

```
sidebar inputs ─► models.build_*()  ─► node / element arrays
                                        │
                                        ▼
                  solvers.solve_*() ─► K, u, reactions, strains, stresses
                                        │
                                        ▼
                  app.py plots & tables (Plotly, pandas)
```

The FEM part (`models.py`, `solvers.py`, `fem_core.py`) depends only on NumPy and can be used
without the web interface, for example:

```python
import models, solvers

m = models.build_truss_crane(HP=6)
r = solvers.solve_truss_crane(m["cord"], m["BC"], m["F"], m["conn"], m["area"], m["elast"])
print(r["U"].max(), r["stress"].min())
```

---

## Using the app

1. **Structure** – choose one of the four models (and the number of panels for the cranes).
2. **Geometry & material** – change L, areas, E and, for the cranes, inclination, tie-support
   position and tip load.
3. **Loads** – edit the table; add a row per load (node, component, value) or delete rows.
4. **Supports** – open *Edit boundary conditions* and tick the fixed directions of each node.
5. **Display** – set the deformation scale factor and optionally show node numbers.
6. Switch between the views: *Structure*, *Strains & Stresses*, *Displacements*,
   *Results tables* (only the selected view is drawn, to keep the 3D graphics responsive).
   Hover over a member in the strain/stress plots to see its value.

If the supports cannot hold the structure (a mechanism), the app shows a warning instead
of results.

---

## Running locally

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. **Create app** → select the repository, branch `main`, main file `app.py` → **Deploy**.

Free apps go to sleep after a period without visitors and take a few seconds to wake up.

---

## Checking the results

- **Equilibrium:** the *Sum of vertical reactions* metric should equal minus the applied
  vertical load (shown in its tooltip).
- **Supports:** displacements at restrained dofs should be much smaller than the largest
  displacement (they are not exactly zero because of the penalty method).
- **Mesh consistency:** changing the number of crane panels or the member length should
  change the results smoothly. Sudden jumps point to an input or support problem.
- **Truss vs. beam:** comparing the two versions of the same structure shows how much the
  diagonal bracing and the joint type (pinned vs. rigid) affect stiffness and member forces.

---

## Assumptions and limitations

**Physical model**
- Linear elastic material, small displacements and small strains (first-order, linear static
  analysis). No geometric or material nonlinearity.
- **No buckling check.** Slender compression members (e.g. the crane mast) may fail by
  buckling at loads well below what the stresses suggest.
- No self-weight, no distributed loads, no thermal loads. Only nodal forces and,
  in the beam models, nodal moments.
- Static loads only (no dynamics, fatigue or wind).
- Ideal joints: perfect pins in the truss models, fully rigid joints in the beam models.

**Beam models**
- Only the **axial** stress σ = E·ε is reported. Bending and torsional stresses are not
  computed, so the peak fibre stress in a beam can be considerably higher than shown.
- Solid circular sections only (I = A²/4π). Hollow tubes or other profiles would need a
  different I and J.
- The "equivalent area" that replaces the diagonals is a simplification. Diagonals provide
  shear (racking) stiffness, which a thicker chord cannot reproduce. The beam and truss models
  are therefore different structures, not two descriptions of the same one.
- In the beam crane, the four tie members get both a beam stiffness and a bar stiffness.
  The bar formulation also adds an EA/L term on their rotational dofs. This keeps the model
  stable but slightly overestimates their stiffness.

**Numerical method**
- Supports use the penalty method, so restrained dofs are not exactly zero and reactions
  are approximate. The accuracy depends on the penalty value relative to the member
  stiffness: a weaker penalty lets supports move slightly, a stronger one worsens
  the matrix conditioning.
- Strains come from the difference between deformed and original member lengths. This is
  accurate for the small displacements assumed here but subtracts two nearly equal numbers.
- Dense matrices and a direct solver are used. This is ideal for these small models
  (≤ about 60 nodes) but not efficient for large structures.
- Rotational boundary conditions cannot be set from the interface: beam supports are pinned
  (translations only).

**Geometry generator**
- The crane connectivity assumes at least 3 panels (HP ≥ 3). The interface allows 3–12.
- The crane topology (tie connections, head shape) is fixed. Only its dimensions, height
  and inclination can change.

---

## Possible extensions

- Bending and combined (axial + bending) stresses for beam elements, and stress utilisation
  against a yield strength
- Euler buckling check for compression members, or a linear buckling (eigenvalue) analysis
- Self-weight and distributed loads
- Exact elimination of restrained dofs instead of the penalty method
- Sparse matrices for larger structures
- Modal (natural frequency) analysis
- Import of arbitrary node / element tables (CSV) to analyse any truss or frame

---

## Author

_Fotios Angelakis – MSc in Mechanical Engineering, Aristotle University of Thessaloniki_

## License

MIT
