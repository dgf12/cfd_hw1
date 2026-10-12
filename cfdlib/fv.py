"""Finite-volume functions for Section 2: grid, sparse operators, boundary conditions, potential-flow solver, post-processing.

Cell-centred FV on the rectangle [0, x_max] x [0, y_max]; unknowns ordered row by row, k = j * nx + i.
Each 2D operator is the Kronecker product of 1D operators, so no loop is used to assemble, apply or solve.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from .common import solve_linear

# ============================================================================
# Grid and geometry
# ============================================================================
def faces_1d(length, n, kind, first):
    """Return the n+1 face coordinates on [0, length]: 'uniform', or 'geometric' (first cell `first`, constant ratio)."""
    if kind == "uniform":
        return np.linspace(0.0, length, n + 1)
    if kind == "geometric":
        ratio = brentq(lambda r: first * (r**n - 1.0) / (r - 1.0) - length, 1.0 + 1e-10, 2.0)
        return np.concatenate([[0.0], np.cumsum(first * ratio ** np.arange(n))])
    raise ValueError(f"Unknown grid kind '{kind}'. Options: 'uniform', 'geometric'.")


def make_grid(domain, grid):
    """Return the cell-centred grid (faces, centres, widths) of the rectangle defined in `domain` and `grid`."""
    xf = faces_1d(domain["x_max"], grid["nx"], grid["kind_x"], grid["first_x"])
    yf = faces_1d(domain["y_max"], grid["ny"], grid["kind_y"], grid["first_y"])
    return {"xf": xf, "yf": yf, "xc": 0.5 * (xf[1:] + xf[:-1]), "yc": 0.5 * (yf[1:] + yf[:-1]),
            "dx": np.diff(xf), "dy": np.diff(yf), "nx": grid["nx"], "ny": grid["ny"]}


def diamond_height(x, airfoil):
    """Return the diamond-airfoil surface y_s(x) (zero outside the chord)."""
    xi = (x - airfoil["x_le"]) / airfoil["chord"]
    return np.where((xi > 0.0) & (xi < 1.0), 2.0 * airfoil["t_half"] * np.minimum(xi, 1.0 - xi), 0.0)


def wall_slope(xf, airfoil):
    """Return the mean surface slope dy_s/dx over each bottom cell, (y_s(x_e) - y_s(x_w)) / dx (exact for the diamond)."""
    return np.diff(diamond_height(xf, airfoil)) / np.diff(xf)


# ============================================================================
# 1D building blocks (sparse)
# ============================================================================
def _csr(rows, cols, vals, shape):
    """Return a CSR matrix from coordinate lists."""
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape)


def face_gradient_1d(xc, xf, bc_start, bc_end):
    """Sparse (n+1) x n operator: dphi/dx on every face from the cell values, homogeneous boundary conditions.

    Boundary face options: 'dirichlet' (phi = 0 on the face), 'neumann' (dphi/dx = 0, value added elsewhere),
    'extrapolate' (gradient copied from the neighbouring interior face: no condition imposed).
    """
    n = len(xc)
    f = np.arange(1, n)                                         # interior faces
    d = xc[1:] - xc[:-1]
    rows, cols, vals = [f, f], [f, f - 1], [1.0 / d, -1.0 / d]
    if bc_start == "dirichlet":
        rows, cols, vals = rows + [[0]], cols + [[0]], vals + [[1.0 / (xc[0] - xf[0])]]
    elif bc_start != "neumann":
        raise ValueError(f"Unknown start boundary '{bc_start}'.")
    if bc_end == "dirichlet":
        rows, cols, vals = rows + [[n]], cols + [[n - 1]], vals + [[-1.0 / (xf[-1] - xc[-1])]]
    elif bc_end == "extrapolate":
        rows, cols, vals = rows + [[n, n]], cols + [[n - 1, n - 2]], vals + [[1.0 / d[-1], -1.0 / d[-1]]]
    elif bc_end != "neumann":
        raise ValueError(f"Unknown end boundary '{bc_end}'.")
    return _csr(rows, cols, vals, (n + 1, n))


def central_divergence_1d(xf):
    """Sparse n x (n+1) operator (G_e - G_w) / dx_i: cell balance of face values (central scheme)."""
    n, dx, i = len(xf) - 1, np.diff(xf), np.arange(len(xf) - 1)
    return _csr([i, i], [i + 1, i], [1.0 / dx, -1.0 / dx], (n, n + 1))


def backward_divergence_1d(xf, ghost_gradient_zero):
    """Sparse n x (n+1) operator (G_w - G_ww) / dx_(i-1): balance shifted one cell upstream (backward scheme).

    Cell 0: G_ww = 0 if `ghost_gradient_zero` (dphi/dx = 0 upstream of A), otherwise the central balance is used.
    """
    n, dx, i = len(xf) - 1, np.diff(xf), np.arange(1, len(xf) - 1)
    rows, cols, vals = [i, i], [i, i - 1], [1.0 / dx[i - 1], -1.0 / dx[i - 1]]
    if ghost_gradient_zero:
        rows, cols, vals = rows + [[0]], cols + [[0]], vals + [[1.0 / dx[0]]]
    else:
        rows, cols, vals = rows + [[0, 0]], cols + [[1, 0]], vals + [[1.0 / dx[0], -1.0 / dx[0]]]
    return _csr(rows, cols, vals, (n, n + 1))


def face_to_cell_1d(n, kind):
    """Sparse n x (n+1) operator giving a cell value from face values: 'average' of both faces or 'west' face."""
    i = np.arange(n)
    if kind == "average":
        return _csr([i, i], [i, i + 1], [np.full(n, 0.5), np.full(n, 0.5)], (n, n + 1))
    return _csr([i], [i], [np.ones(n)], (n, n + 1))


# ============================================================================
# 2D operators and boundary conditions
# ============================================================================
BC_X_START = {"phi": "dirichlet", "phi+phix": "dirichlet"}            # boundary A
BC_X_END = {"neumann": "neumann", "dirichlet": "dirichlet", "none": "extrapolate"}   # boundary C
BC_Y_END = {"neumann": "neumann", "dirichlet": "dirichlet"}           # boundary B


def build_operators(g, bc_inflow, bc_outflow, bc_farfield):
    """Return the sparse 2D operators phi_xx (central, backward), phi_yy and cell gradients for the named BCs."""
    for name, value, options in (("bc_inflow", bc_inflow, BC_X_START), ("bc_outflow", bc_outflow, BC_X_END),
                                 ("bc_farfield", bc_farfield, BC_Y_END)):
        if value not in options:
            raise ValueError(f"Unknown {name} '{value}'. Options: {list(options)}")
    nx, ny = g["nx"], g["ny"]
    Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    Gx = face_gradient_1d(g["xc"], g["xf"], BC_X_START[bc_inflow], BC_X_END[bc_outflow])
    Gy = face_gradient_1d(g["yc"], g["yf"], "neumann", BC_Y_END[bc_farfield])   # south face: wall D/E (wall_terms)
    Db = backward_divergence_1d(g["xf"], ghost_gradient_zero=(bc_inflow == "phi+phix"))
    gx = sp.kron(Iy, Gx, format="csr")                                          # u on the x-faces
    return {
        "gx": gx,
        "div_central": sp.kron(Iy, central_divergence_1d(g["xf"]), format="csr"),
        "div_backward": sp.kron(Iy, Db, format="csr"),
        "yy": sp.kron(central_divergence_1d(g["yf"]) @ Gy, Ix, format="csr"),
        "avg": sp.kron(Iy, face_to_cell_1d(nx, "average"), format="csr"),
        "west": sp.kron(Iy, face_to_cell_1d(nx, "west"), format="csr"),
        "y": sp.kron(face_to_cell_1d(ny, "average") @ Gy, Ix, format="csr"),
    }


def wall_terms(g, airfoil, U_inf):
    """Return the contributions of dphi/dy = U dy_s/dx on y = 0 (D/E) to phi_yy and to the cell value of dphi/dy."""
    flux = U_inf * wall_slope(g["xf"], airfoil)               # gradient on the south faces of row j = 0
    yy, y = np.zeros(g["nx"] * g["ny"]), np.zeros(g["nx"] * g["ny"])
    yy[:g["nx"]] = -flux / g["dy"][0]                         # (G_n - G_s)/dy with G_s = flux
    y[:g["nx"]] = 0.5 * flux                                  # cell gradient = average of the two faces
    return yy, y


# ============================================================================
# Equation, schemes and solver
# ============================================================================
def freestream(M_inf, gas):
    """Return U_inf = M sqrt(gamma R T_inf) and rho_inf = p_inf / (R T_inf)."""
    return M_inf * np.sqrt(gas["gamma"] * gas["R"] * gas["T_inf"]), gas["p_inf"] / (gas["R"] * gas["T_inf"])


def x_flux(u, M_inf, U_inf, gamma, equation):
    """Return the x-flux F(u) and A = dF/du: F = (1-M^2) u ('linear', eq. 3a) or - (gamma+1) M^2 u^2/(2U) added ('tsd', 3b)."""
    if equation not in ("linear", "tsd"):
        raise ValueError(f"Unknown equation '{equation}'. Options: 'linear', 'tsd'.")
    kappa = (gamma + 1.0) * M_inf**2 / U_inf if equation == "tsd" else 0.0
    return (1.0 - M_inf**2) * u - 0.5 * kappa * u**2, (1.0 - M_inf**2) - kappa * u


def x_scheme_masks(A_cell, scheme_x, nx, ny, upstream_supersonic):
    """Return per-cell weights (central, backward) of the x-flux balance: 'central', 'backward' or 'switch'.

    'switch' (Murman, conservative): mu = 1 where A < 0; cell i uses (1 - mu_i) x central balance
    + mu_(i-1) x backward balance, so the shock point gets both and the sonic point neither.
    """
    if scheme_x == "central":
        return np.ones(nx * ny), np.zeros(nx * ny)
    if scheme_x == "backward":
        return np.zeros(nx * ny), np.ones(nx * ny)
    if scheme_x != "switch":
        raise ValueError(f"Unknown scheme_x '{scheme_x}'. Options: 'central', 'backward', 'switch'.")
    mu = (A_cell < 0.0).astype(float).reshape(ny, nx)
    mu_upstream = np.hstack([np.full((ny, 1), float(upstream_supersonic)), mu[:, :-1]])
    return 1.0 - mu.ravel(), mu_upstream.ravel()


def residual_and_jacobian(phi, ops, rhs, case, U_inf, gamma):
    """Return the discrete residual R(phi) = X F(u_faces) + phi_yy - b, its Jacobian dR/dphi, A per cell and the masks."""
    M_inf, nx, ny = case["M_inf"], case["nx"], case["ny"]
    u_face = ops["gx"] @ phi
    F, dF = x_flux(u_face, M_inf, U_inf, gamma, case["equation"])
    _, A_cell = x_flux(ops["avg"] @ u_face, M_inf, U_inf, gamma, case["equation"])   # switch on A at the cell
    masks = x_scheme_masks(A_cell, case["scheme_x"], nx, ny, M_inf > 1.0)
    X = sp.diags(masks[0]) @ ops["div_central"] + sp.diags(masks[1]) @ ops["div_backward"]
    R = X @ F + ops["yy"] @ phi - rhs
    J = X @ sp.diags(dF) @ ops["gx"] + ops["yy"]
    return R, J, A_cell, masks


def newton(phi, ops, rhs, case, U_inf, gamma, solver, nonlinear):
    """Newton iteration phi <- phi - relax J^-1 R until ||R|| / ||b|| < tol (one step for the linear equation)."""
    history, lin_history = [], []
    b_norm = np.linalg.norm(rhs)
    for _ in range(nonlinear["max_iter"]):
        R, J, A_cell, masks = residual_and_jacobian(phi, ops, rhs, case, U_inf, gamma)
        history.append(np.linalg.norm(R) / b_norm)
        if history[-1] < nonlinear["tol"] or not np.isfinite(history[-1]):
            break
        dphi, lin_history = solve_linear(J, R, solver, np.zeros_like(phi))
        phi = phi - nonlinear["relax"] * dphi
    else:
        print(f"  WARNING: Newton stopped at max_iter = {nonlinear['max_iter']}, residual {history[-1]:.2e}")
    return phi, A_cell, masks, history, lin_history


def solve_potential_flow(g, case, gas, airfoil, nonlinear):
    """Solve eq. (3a) or (3b) for one case (optionally with continuation in M_inf); return phi and diagnostics."""
    if case["scheme_y"] != "central":
        raise ValueError("Only scheme_y = 'central' is implemented.")
    case = dict(case, nx=g["nx"], ny=g["ny"])
    ops = build_operators(g, case["bc_inflow"], case["bc_outflow"], case["bc_farfield"])
    steps = [m for m in nonlinear["continuation"] if m < case["M_inf"]] if case["equation"] == "tsd" else []
    phi, U_prev, histories = np.zeros(g["nx"] * g["ny"]), None, {}
    for M in steps + [case["M_inf"]]:                         # continuation in Mach number (eq. 3b only)
        U_inf, _ = freestream(M, gas)
        phi = phi if U_prev is None else phi * U_inf / U_prev  # phi scales with U_inf
        wall_yy, wall_y = wall_terms(g, airfoil, U_inf)
        phi, A_cell, masks, hist, lin_hist = newton(phi, ops, -wall_yy, dict(case, M_inf=M), U_inf,
                                                    gas["gamma"], case["solver"], nonlinear)
        histories[M], U_prev = hist, U_inf
        print(f"  M = {M:.3f}: {len(hist) - 1} Newton step(s), residual {hist[-1]:.1e}, "
              f"{int((A_cell < 0).sum())} supersonic cells")
    return {"phi": phi, "A": A_cell, "masks": masks, "ops": ops, "wall_y": wall_y, "U_inf": U_inf,
            "history": histories[case["M_inf"]], "all_histories": histories, "linear_history": lin_hist}


# ============================================================================
# Post-processing
# ============================================================================
def flow_fields(sol, g, case, gas, airfoil):
    """Return phi/(U c), u, v, M, T/T_inf and rho/rho_inf as (ny, nx) arrays at the cell centres (eq. 4)."""
    U, M_inf, gam = sol["U_inf"], case["M_inf"], gas["gamma"]
    u_face = sol["ops"]["gx"] @ sol["phi"]                     # central cells: face average; supersonic: west face
    u = sol["masks"][0] * (sol["ops"]["avg"] @ u_face) + (1.0 - sol["masks"][0]) * (sol["ops"]["west"] @ u_face)
    v = sol["ops"]["y"] @ sol["phi"] + sol["wall_y"]
    q2 = (U + u) ** 2 + v**2
    T = 1.0 - 0.5 * (gam - 1.0) * M_inf**2 * (q2 - U**2) / U**2
    shape = (g["ny"], g["nx"])
    fields = {"phi": sol["phi"] / (U * airfoil["chord"]), "u": u, "v": v, "T": T,
              "rho": np.maximum(T, 0.0) ** (1.0 / (gam - 1.0)),
              "mach": np.sqrt(q2) / np.sqrt(gam * gas["R"] * gas["T_inf"] * np.maximum(T, 1e-12))}
    return {k: f.reshape(shape) for k, f in fields.items()}


def mach_wave_angle(g, u, threshold, y_range):
    """Angle (deg) of the leading-edge Mach wave: first x where |u| > threshold x (row max of |u|), fitted vs y."""
    rows = np.where((g["yc"] >= y_range[0]) & (g["yc"] <= y_range[1]))[0]
    au = np.abs(u[rows])
    first = np.argmax(au > threshold * au.max(axis=1, keepdims=True), axis=1)
    slope = np.polyfit(g["yc"][rows], g["xc"][first], 1)[0]    # dx/dy along the wave
    return np.degrees(np.arctan(1.0 / slope))


def discretisation_table(cases):
    """Return a text table with, per regime: equation, x/y schemes, switch, boundary conditions A-E and solver."""
    text = {"linear": "(3a) linear", "tsd": "(3b) TSD"}
    sw = {"central": "none (central)", "backward": "none (backward)", "switch": "Murman: A>0 central/A<0 backward"}
    lines = [f"{'':14s}" + "".join(f"{name:>34s}" for name in cases)]
    rows = [("equation", lambda c: text[c["equation"]]),
            ("flux x", lambda c: c["scheme_x"]), ("flux y", lambda c: c["scheme_y"]),
            ("switch", lambda c: sw[c["scheme_x"]]),
            ("A (x=0)", lambda c: {"phi": "phi=0", "phi+phix": "phi=0, phi_x=0"}[c["bc_inflow"]]),
            ("B (y=ymax)", lambda c: {"neumann": "phi_y=0", "dirichlet": "phi=0"}[c["bc_farfield"]]),
            ("C (x=xmax)", lambda c: {"neumann": "phi_x=0", "dirichlet": "phi=0", "none": "none"}[c["bc_outflow"]]),
            ("D (y=0)", lambda c: "phi_y=0"), ("E (airfoil)", lambda c: "phi_y=U dys/dx"),
            ("solver", lambda c: c["solver"]["name"] + (" + Newton" if c["equation"] == "tsd" else ""))]
    for label, fn in rows:
        lines.append(f"{label:14s}" + "".join(f"{fn(c):>34s}" for c in cases.values()))
    return "\n".join(lines)
