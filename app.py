"""app.py - Streamlit web app: 3D truss & frame finite element analysis.

Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calc_data import calc_data
import models
import solvers

st.set_page_config(page_title="3D Truss & Frame FEM", layout="wide")

MODELS = {
    "Truss Cube":  dict(build=models.build_truss_cube,  solve=solvers.solve_truss_cube,
                        crane=False, beam=False, scale=20000.0, offset=1000),
    "Truss Crane": dict(build=models.build_truss_crane, solve=solvers.solve_truss_crane,
                        crane=True,  beam=False, scale=50.0,    offset=-2500),
    "Beam Cube":   dict(build=models.build_beam_cube,   solve=solvers.solve_beam_cube,
                        crane=False, beam=True,  scale=10000.0, offset=1000),
    "Beam Crane":  dict(build=models.build_beam_crane,  solve=solvers.solve_beam_crane,
                        crane=True,  beam=True,  scale=10.0,    offset=-2500),
}
TRUSS_DOFS = ["Fx", "Fy", "Fz"]
BEAM_DOFS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# default camera: azimuth -13°, elevation 11°, orthographic projection
_az, _el = np.radians(-13), np.radians(11)
CAMERA = dict(eye=dict(x=2.0 * np.sin(_az) * np.cos(_el),
                       y=-2.0 * np.cos(_az) * np.cos(_el),
                       z=2.0 * np.sin(_el)),
              projection=dict(type="orthographic"))


# =============================================================================== plotting helpers
def _segments(P, conn, idx):
    x, y, z = [], [], []
    for e in idx:
        a, b = conn[e, 0] - 1, conn[e, 1] - 1
        x += [P[a, 0], P[b, 0], None]
        y += [P[a, 1], P[b, 1], None]
        z += [P[a, 2], P[b, 2], None]
    return x, y, z


def add_lines(fig, P, conn, idx, width, color, dash="solid", name=None):
    if len(idx) == 0:
        return
    x, y, z = _segments(P, conn, idx)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name,
                               showlegend=name is not None, hoverinfo="skip",
                               line=dict(width=width, color=color, dash=dash)))


def add_colored_lines(fig, P, conn, idx, width, va, vb, cmin, cmax, title, showscale):
    """Each element coloured from value va[e] (node 1) to vb[e] (node 2), jet colour map."""
    if len(idx) == 0:
        return
    x, y, z = _segments(P, conn, idx)
    c = []
    for e in idx:
        c += [va[e], vb[e], vb[e]]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(width=width, color=c, colorscale="Jet", cmin=cmin, cmax=cmax,
                  showscale=showscale, colorbar=dict(title=title, len=0.75))))


def add_element_hover(fig, P, conn, text):
    mid = (P[conn[:, 0] - 1, :3] + P[conn[:, 1] - 1, :3]) / 2
    fig.add_trace(go.Scatter3d(x=mid[:, 0], y=mid[:, 1], z=mid[:, 2], mode="markers",
                               marker=dict(size=3, color="rgba(0,0,0,0)"),
                               hovertext=text, hoverinfo="text", showlegend=False))


def add_nodes(fig, P, n, color, labels=False, name=None):
    fig.add_trace(go.Scatter3d(
        x=P[:n, 0], y=P[:n, 1], z=P[:n, 2],
        mode="markers+text" if labels else "markers",
        text=[str(i + 1) for i in range(n)] if labels else None,
        textposition="top center", textfont=dict(size=11),
        marker=dict(size=4, color=color), name=name, showlegend=name is not None,
        hovertext=[f"Node {i + 1}" for i in range(n)], hoverinfo="text"))


def add_supports(fig, P, BC, n):
    idx = [i for i in range(n) if BC[i, :3].any()]
    if not idx:
        return
    fig.add_trace(go.Scatter3d(
        x=P[idx, 0], y=P[idx, 1], z=P[idx, 2], mode="markers", name="Supports",
        marker=dict(size=7, color="green", symbol="square"),
        hovertext=[f"Node {i + 1}: fixed " +
                   "".join(d for d, f in zip("xyz", BC[i, :3]) if f) for i in idx],
        hoverinfo="text"))


def add_forces(fig, P, F, offset):
    """Arrow of length |offset| mm along each non-zero force component."""
    first = True
    h = 0.25 * abs(offset)
    for i in range(F.shape[0]):
        for j in range(3):
            if F[i, j] == 0:
                continue
            p1 = P[i, :3].astype(float)
            p2 = p1.copy()
            p2[j] += offset
            start, end = (p2, p1) if np.sign(F[i, j]) == -np.sign(offset) else (p1, p2)
            d = (end - start) / np.linalg.norm(end - start)
            perp1 = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(d, perp1)
            x, y, z = [start[0], end[0], None], [start[1], end[1], None], [start[2], end[2], None]
            for p in (perp1, -perp1, perp2, -perp2):
                q = end - h * d + 0.4 * h * p
                x += [end[0], q[0], None]
                y += [end[1], q[1], None]
                z += [end[2], q[2], None]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", line=dict(width=6, color="black"),
                name="Force", showlegend=first, hoverinfo="text",
                hovertext=f"{TRUSS_DOFS[j]} = {F[i, j]:g} N at node {i + 1}"))
            first = False


def finish(fig, title, height=600):
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0, y=1),
        scene=dict(aspectmode="data", camera=CAMERA,
                   xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]"))
    return fig


# =============================================================================== sidebar inputs
st.sidebar.title("Inputs")
name = st.sidebar.selectbox("Structure", list(MODELS))
cfg = MODELS[name]
HP = st.sidebar.slider("Horizontal planes (main body)", 3, 12, 6) if cfg["crane"] else None

# ---- geometry & material (default values)
D_F, D_E, D_A, D_B, D_angle, D_A0, D_L, D_Ad, D_Ay = calc_data()
st.sidebar.subheader("Geometry & material")
L = st.sidebar.number_input("Member length L [mm]", 100.0, 20000.0, float(D_L), step=100.0,
                            help="Cube side / crane panel size")
if cfg["crane"]:
    angle_deg = st.sidebar.number_input("Crane inclination [° from horizontal]", 30.0, 90.0,
                                        round(float(np.degrees(D_angle)), 2), step=1.0,
                                        help="90° = vertical mast")
    A_sup = st.sidebar.number_input("Support position A [mm]", 0.0, 20000.0, float(D_A),
                                    step=100.0, help="Support nodes at x = −A")
    B_sup = st.sidebar.number_input("Support position B [mm]", 0.0, 20000.0, float(D_B),
                                    step=100.0, help="Support nodes at z = +B")
    F_def = st.sidebar.number_input("Tip load F [N]", 0.0, 1e7, float(D_F), step=1000.0,
                                    help="Default downward load at the crane tip")
else:
    angle_deg, A_sup, B_sup, F_def = float(np.degrees(D_angle)), D_A, D_B, D_F
Ay = st.sidebar.number_input("Main member area Ay [mm²]", 1.0, 1e6, float(D_Ay), step=50.0)
Ad = st.sidebar.number_input("Diagonal member area Ad [mm²]", 1.0, 1e6, float(D_Ad), step=50.0,
                             help="In the beam models the diagonals are removed and their "
                                  "area is added to an equivalent beam cross-section")
E = st.sidebar.number_input("Young's modulus E [MPa]", 1000.0, 1e7, float(D_E), step=1000.0)

data = (F_def, E, A_sup, B_sup, np.radians(angle_deg), Ay / 1.5, L, Ad, Ay)
m = cfg["build"](HP=HP, data=data) if cfg["crane"] else cfg["build"](data=data)
cord, conn, area, elast = m["cord"], m["conn"], m["area"], m["elast"]
n_nodes = cord.shape[0]
n_edit = n_nodes - 1 if cfg["beam"] else n_nodes   # beams: last node = orientation reference
dofs = BEAM_DOFS if cfg["beam"] else TRUSS_DOFS
thick = np.isclose(area, m["AY"] if cfg["beam"] else m["Ay"])
if name == "Beam Cube":
    thick[:] = True
model_key = f"{name}_{HP}"

# ---- loads
st.sidebar.subheader("Loads")
st.sidebar.caption("Forces in N, moments in N·mm. Add or delete rows to change the loads.")
def_rows = [dict(Node=i + 1, Component=dofs[j], Value=float(m["F"][i, j]))
            for i in range(n_edit) for j in range(len(dofs)) if m["F"][i, j] != 0]
loads = st.sidebar.data_editor(
    pd.DataFrame(def_rows, columns=["Node", "Component", "Value"]),
    num_rows="dynamic", key=f"loads_{model_key}_{F_def}",
    column_config={
        "Node": st.column_config.NumberColumn(min_value=1, max_value=n_edit, step=1),
        "Component": st.column_config.SelectboxColumn(options=dofs),
        "Value": st.column_config.NumberColumn(format="%.1f")})

F = np.zeros_like(m["F"])
for _, row in loads.iterrows():
    if pd.notna(row["Node"]) and pd.notna(row["Value"]) and row["Component"] in dofs:
        node = int(row["Node"])
        if 1 <= node <= n_edit:
            F[node - 1, dofs.index(row["Component"])] += float(row["Value"])

# ---- boundary conditions
st.sidebar.subheader("Supports")
with st.sidebar.expander("Edit boundary conditions (tick = fixed)"):
    bc_df = pd.DataFrame({"Node": np.arange(1, n_edit + 1),
                          "x": m["BC"][:n_edit, 0].astype(bool),
                          "y": m["BC"][:n_edit, 1].astype(bool),
                          "z": m["BC"][:n_edit, 2].astype(bool)})
    bc_df = st.data_editor(bc_df, hide_index=True, disabled=["Node"],
                           key=f"bc_{model_key}", height=300)
BC = m["BC"].copy()
BC[:n_edit, 0:3] = bc_df[["x", "y", "z"]].to_numpy().astype(float)

# ---- display
st.sidebar.subheader("Display")
s = st.sidebar.number_input("Deformation scale factor", min_value=0.0,
                            value=cfg["scale"], key=f"scale_{name}")
labels = st.sidebar.checkbox("Show node numbers", value=False)

# =============================================================================== solve
st.title("3D Truss & Frame Finite Element Analysis")
st.caption(f"{name}" + (f"  ·  {HP} horizontal planes" if HP else "") +
           f"  ·  {n_nodes} nodes, {len(conn)} elements  ·  units: mm, N, MPa")

try:
    r = cfg["solve"](cord, BC, F, conn, area, elast)
    # an unsupported (mechanism) structure gives absurd or non-finite displacements
    if not np.all(np.isfinite(r["U"])) or np.abs(r["U"][:, :3]).max() > 1e6:
        raise np.linalg.LinAlgError("singular")
except np.linalg.LinAlgError:
    st.error("The stiffness matrix is singular: the structure is not properly supported. "
             "Add more boundary conditions.")
    st.stop()

U, strain, stress, R = r["U"], r["strain"], r["stress"], r["Reactions"]
Ut = np.sqrt((U[:, :3] ** 2).sum(axis=1))
P0 = cord[:, :3]
Pd = P0 + s * U[:, :3]
all_idx = np.arange(len(conn))
thick_idx, thin_idx = all_idx[thick], all_idx[~thick]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max total displacement", f"{Ut.max():.4g} mm", help=f"at node {Ut.argmax() + 1}")
c2.metric("Max tensile stress", f"{stress.max():.4g} MPa",
          help=f"element {stress.argmax() + 1}")
c3.metric("Max compressive stress", f"{stress.min():.4g} MPa",
          help=f"element {stress.argmin() + 1}")
sup = [i for i in range(R.shape[0]) if BC[i].any()]
c4.metric("Sum of vertical reactions", f"{R[sup, 2].sum():.4g} N",
          help=f"Applied vertical load: {F[:, 2].sum():.4g} N")

# Only the selected view is drawn: every 3D plot uses a browser WebGL context and browsers
# allow only ~16, so drawing all plots on every rerun makes the oldest ones go blank.
view = st.radio("View", ["Structure", "Strains & Stresses", "Displacements", "Results tables"],
                horizontal=True, key="view", label_visibility="collapsed")

# ---- Structure
if view == "Structure":
    a, b = st.columns(2)
    fig = go.Figure()
    add_lines(fig, P0, conn, thick_idx, 6, "#0072BD", name="Elements")
    add_lines(fig, P0, conn, thin_idx, 2, "#0072BD")
    add_nodes(fig, P0, n_edit, "red", labels, name="Nodes")
    add_supports(fig, P0, BC, n_edit)
    add_forces(fig, P0, F, cfg["offset"])
    a.plotly_chart(finish(fig, "Undeformed Structure"), use_container_width=True,
                   key="plot_undeformed")

    fig = go.Figure()
    add_lines(fig, P0, conn, all_idx, 2, "black", dash="dash", name="Undeformed")
    add_lines(fig, Pd, conn, thick_idx, 6, "red", name="Deformed")
    add_lines(fig, Pd, conn, thin_idx, 2, "red")
    add_nodes(fig, Pd, n_edit, "red", labels)
    add_forces(fig, Pd, F, cfg["offset"])
    b.plotly_chart(finish(fig, f"Undeformed - Deformed Structure (scale {s:g})"),
                   use_container_width=True, key="plot_deformed")

# ---- Strains & Stresses
elif view == "Strains & Stresses":
    a, b = st.columns(2)
    for col, vals, title, unit in [(a, strain, "Axial Strain", ""),
                                   (b, stress, "Axial Stress", " MPa")]:
        fig = go.Figure()
        add_lines(fig, P0, conn, all_idx, 2, "black", dash="dash")
        cmin, cmax = vals.min(), vals.max()
        add_colored_lines(fig, Pd, conn, thick_idx, 7, vals, vals, cmin, cmax, title, True)
        add_colored_lines(fig, Pd, conn, thin_idx, 4, vals, vals, cmin, cmax, title,
                          len(thick_idx) == 0)
        add_nodes(fig, Pd, n_edit, "blue", labels)
        add_element_hover(fig, Pd, conn,
                          [f"Element {e + 1}: {v:.4g}{unit}" for e, v in enumerate(vals)])
        add_forces(fig, Pd, F, cfg["offset"])
        col.plotly_chart(finish(fig, f"Deformed Structure with {title}s"),
                         use_container_width=True, key=f"plot_{title}")

# ---- Displacements
elif view == "Displacements":
    comp = st.radio("Component", ["Ux", "Uy", "Uz", "Utotal"], horizontal=True, index=3)
    vals = {"Ux": U[:, 0], "Uy": U[:, 1], "Uz": U[:, 2], "Utotal": Ut}[comp]
    va, vb = vals[conn[:, 0] - 1], vals[conn[:, 1] - 1]
    fig = go.Figure()
    add_lines(fig, P0, conn, all_idx, 2, "black", dash="dash")
    add_colored_lines(fig, Pd, conn, thick_idx, 7, va, vb, vals.min(), vals.max(),
                      f"{comp} [mm]", True)
    add_colored_lines(fig, Pd, conn, thin_idx, 3, va, vb, vals.min(), vals.max(),
                      f"{comp} [mm]", len(thick_idx) == 0)
    add_nodes(fig, Pd, n_edit, "black", labels)
    add_forces(fig, Pd, F, cfg["offset"])
    st.plotly_chart(finish(fig, f"Deformed Structure on {comp}", height=700),
                    use_container_width=True, key="plot_displacement")

# ---- Tables
else:
    nd = pd.DataFrame({"Node": np.arange(1, n_edit + 1),
                       "x": P0[:n_edit, 0], "y": P0[:n_edit, 1], "z": P0[:n_edit, 2],
                       "Ux [mm]": U[:n_edit, 0], "Uy [mm]": U[:n_edit, 1],
                       "Uz [mm]": U[:n_edit, 2], "Utotal [mm]": Ut[:n_edit],
                       "Rx [N]": R[:n_edit, 0], "Ry [N]": R[:n_edit, 1],
                       "Rz [N]": R[:n_edit, 2]})
    el = pd.DataFrame({"Element": np.arange(1, len(conn) + 1),
                       "N1": conn[:, 0], "N2": conn[:, 1], "Area [mm²]": area,
                       "L [mm]": r["lengths"], "New L [mm]": r["new_len"],
                       "Strain": strain, "Stress [MPa]": stress})
    st.subheader("Nodes")
    st.dataframe(nd, hide_index=True, use_container_width=True)
    st.download_button("Download nodes CSV", nd.to_csv(index=False), f"nodes_{name}.csv")
    st.subheader("Elements")
    st.dataframe(el, hide_index=True, use_container_width=True)
    st.download_button("Download elements CSV", el.to_csv(index=False),
                       f"elements_{name}.csv")

with st.expander("About this app"):
    st.markdown(
        "Linear static finite element analysis of space trusses (3 dof/node) and "
        "3D Euler–Bernoulli frames (6 dof/node) with solid circular cross-sections. "
        "Supports are applied with the penalty method; axial strain = (L_new − L)/L, "
        "stress = E·strain. See the README on GitHub for details and limitations.")
