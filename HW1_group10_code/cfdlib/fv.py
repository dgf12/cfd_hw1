"""Finite-volume functions for Section 2: grid, sparse operators, boundary conditions, potential-flow solver, post-processing.

Cell-centred FV on the rectangle [0, x_max] x [0, y_max]; unknowns ordered k = j * nx + i.
Each 2D operator is the Kronecker product of 1D operators.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

from .common import solve_linear, local_orders

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
    raise ValueError(f"Unknown grid kind '{kind}'. Options: 'uniform', 'geometric' (x also 'clustered').")


def stretched_widths(length, first, ratio):
    """Return cell widths starting at `first` and growing geometrically (ratio adjusted exactly) to fill `length`."""
    if length <= 1e-12:
        return np.zeros(0)
    n = max(1, int(round(np.log(1.0 + length * (ratio - 1.0) / first) / np.log(ratio))))
    if n * first >= length:                                    
        return np.full(max(1, int(round(length / first))), length / max(1, int(round(length / first))))
    r = brentq(lambda q: first * (q**n - 1.0) / (q - 1.0) - length, 1.0 + 1e-12, 10.0)
    return first * r ** np.arange(n)


def faces_clustered(length, cluster, first, ratio):
    """Return x-faces on [0, length]: uniform cells of size ~first on cluster = (x0, x1), geometric stretching outside."""
    x0, x1 = cluster
    n = max(1, int(round((x1 - x0) / first)))
    widths = np.concatenate([stretched_widths(x0, first, ratio)[::-1],     # towards the cluster (reversed)
                             np.full(n, (x1 - x0) / n),
                             stretched_widths(length - x1, first, ratio)])  # away from the cluster
    xf = np.concatenate([[0.0], np.cumsum(widths)])
    xf[-1] = length                                            
    return xf


def make_grid(domain, grid):
    """Return the cell-centred grid (faces, centres, widths) of the rectangle defined in `domain` and `grid`."""
    if grid["kind_x"] == "clustered":                          # nx follows from first_x, ratio_x and cluster_x
        xf = faces_clustered(domain["x_max"], grid["cluster_x"], grid["first_x"], grid["ratio_x"])
    else:
        xf = faces_1d(domain["x_max"], grid["nx"], grid["kind_x"], grid["first_x"])
    yf = faces_1d(domain["y_max"], grid["ny"], grid["kind_y"], grid["first_y"])
    return {"xf": xf, "yf": yf, "xc": 0.5 * (xf[1:] + xf[:-1]), "yc": 0.5 * (yf[1:] + yf[:-1]),
            "dx": np.diff(xf), "dy": np.diff(yf), "nx": len(xf) - 1, "ny": len(yf) - 1}


def diamond_height(x, airfoil):
    """Return the diamond-airfoil surface y_s(x) (zero outside the chord)."""
    xi = (x - airfoil["x_le"]) / airfoil["chord"]
    return np.where((xi > 0.0) & (xi < 1.0), 2.0 * airfoil["t_half"] * np.minimum(xi, 1.0 - xi), 0.0)


def wall_slope(xf, airfoil):
    """Return the mean surface slope dy_s/dx over each bottom cell, (y_s(x_e) - y_s(x_w)) / dx (exact for the diamond)."""
    return np.diff(diamond_height(xf, airfoil)) / np.diff(xf)


# ============================================================================
# 1D building blocks 
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
BC_X_START = {"phi": "dirichlet", "phi+phix": "dirichlet"}            # Boundary A
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
    flux = U_inf * wall_slope(g["xf"], airfoil)               # gradiente on the south faces of row j = 0
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


def picard(phi, ops, rhs, case, U_inf, gamma, solver, nonlinear):
    """Picard iteration: freeze the secant coefficient a = F(u)/u and the switch, solve, phi <- phi + relax (phi* - phi)."""
    # The fluxo is written F(u) = a(u) u with the secant a = (1-M^2) - (gamma+1) M^2 u/(2U) = ((1-M^2) + A)/2.
    # Freezing a (and the masks) at the current phi gives the linear system K phi* = b, K = X diag(a) gx + phi_yy;
    # at convergence K(phi) phi = b is exactly R(phi) = 0. step is stopped early if the residual diverges.
    history, lin_history = [], []
    b_norm = np.linalg.norm(rhs)
    for _ in range(nonlinear["max_iter"]):
        R, _, A_cell, masks = residual_and_jacobian(phi, ops, rhs, case, U_inf, gamma)
        history.append(np.linalg.norm(R) / b_norm)
        if history[-1] < nonlinear["tol"] or not history[-1] < nonlinear.get("diverged", 1e6):
            break
        _, A_face = x_flux(ops["gx"] @ phi, case["M_inf"], U_inf, gamma, case["equation"])
        a_secant = 0.5 * ((1.0 - case["M_inf"] ** 2) + A_face)
        X = sp.diags(masks[0]) @ ops["div_central"] + sp.diags(masks[1]) @ ops["div_backward"]
        K = X @ sp.diags(a_secant) @ ops["gx"] + ops["yy"]
        phi_new, lin_history = solve_linear(K, rhs, solver, phi)
        phi = phi + nonlinear["relax"] * (phi_new - phi)
    else:
        print(f"  WARNING: Picard stopped at max_iter = {nonlinear['max_iter']}, residual {history[-1]:.2e}")
    return phi, A_cell, masks, history, lin_history


NONLINEAR_METHODS = {"newton": newton, "picard": picard}


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
        method = NONLINEAR_METHODS[nonlinear.get("method", "newton")]  # 'newton' (default) or 'picard'
        phi, A_cell, masks, hist, lin_hist = method(phi, ops, -wall_yy, dict(case, M_inf=M), U_inf,
                                                    gas["gamma"], case["solver"], nonlinear)
        histories[M], U_prev = hist, U_inf
        print(f"  M = {M:.3f}: {len(hist) - 1} {nonlinear.get('method', 'newton')} step(s), residual {hist[-1]:.1e}, "
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
    slope = np.polyfit(g["yc"][rows], g["xc"][first], 1)[0]    
    return np.degrees(np.arctan(1.0 / slope))


def discretisation_table(cases, nonlinear=None):
    """Return a text table with, per regime: equation, x/y schemes, switch, boundary conditions A-E and solver."""
    method = (nonlinear or {}).get("method", "newton").capitalize()
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
            ("solver", lambda c: c["solver"]["name"] + (f" + {method}" if c["equation"] == "tsd" else ""))]
    for label, fn in rows:
        lines.append(f"{label:14s}" + "".join(f"{fn(c):>34s}" for c in cases.values()))
    return "\n".join(lines)


# ============================================================================
# Mach-wave angle (supersonic check against beta = arcsin(1/M))
# ============================================================================
def mach_wave_angle_steepest(g, u, y_range):
    """Angle (deg) of the leading-edge Mach wave: per row the steepest compression (min du/dx) upstream of the
    mid-chord expansion (max du/dx), located to sub-cell accuracy by a parabola, then a straight-line fit x(y)."""
    rows = np.where((g["yc"] >= y_range[0]) & (g["yc"] <= y_range[1]))[0]
    x_wave = []
    for j in rows:
        dudx = np.gradient(u[j], g["xc"])
        k = np.argmin(dudx[:np.argmax(dudx)])                  # LE compression lies upstream of the expansion
        a, b, c = dudx[max(k - 1, 0)], dudx[k], dudx[min(k + 1, len(dudx) - 1)]
        curv = a - 2.0 * b + c
        shift = 0.5 * (a - c) / curv if 0 < k < len(dudx) - 1 and curv != 0 else 0.0
        x_wave.append(g["xc"][k] + shift * g["dx"][k])
    slope = np.polyfit(g["yc"][rows], np.array(x_wave), 1)[0]  # dx/dy along the wave
    return np.degrees(np.arctan(1.0 / slope))


def measure_mach_angle(g, u, options):
    """Return the Mach-wave angle (deg) with the named method: 'threshold' (first |u| > threshold max|u|) or 'steepest'."""
    if options["method"] == "threshold":
        return mach_wave_angle(g, u, options["threshold"], options["y_range"])
    if options["method"] == "steepest":
        return mach_wave_angle_steepest(g, u, options["y_range"])
    raise ValueError(f"Unknown Mach-angle method '{options['method']}'. Options: 'threshold', 'steepest'.")


def refine_grid(grid, factor):
    """Return the grid parameters with `factor` times more cells in x and y (first cell sizes divided by factor)."""
    return dict(grid, nx=None if grid["nx"] is None else int(round(grid["nx"] * factor)),
                ny=int(round(grid["ny"] * factor)),
                first_x=None if grid["first_x"] is None else grid["first_x"] / factor,
                first_y=None if grid["first_y"] is None else grid["first_y"] / factor)


def mach_angle_refinement(domain, grid, factors, case, gas, airfoil, nonlinear, methods):
    """Return a text table of the measured Mach-wave angle against grid refinement, one column per method."""
    beta = np.degrees(np.arcsin(1.0 / case["M_inf"]))
    lines = [f"Mach-wave angle, M = {case['M_inf']}: arcsin(1/M) = {beta:.2f} deg",
             f"{'nx x ny':>12s}" + "".join(f"{name:>16s}" for name in methods)]
    for factor in factors:
        gr = refine_grid(grid, factor)
        g = make_grid(domain, gr)
        fields = flow_fields(solve_potential_flow(g, case, gas, airfoil, nonlinear), g, case, gas, airfoil)
        lines.append(f"{gr['nx']:>6d} x {gr['ny']:<4d}" +
                     "".join(f"{measure_mach_angle(g, fields['u'], opt):>16.2f}" for opt in methods.values()))
    return "\n".join(lines)


# ============================================================================
# Verification: method of manufactured solutions (MMS)
# ============================================================================
def mms_solution(x, y, domain, kind):
    """Return phi_m, d2phi_m/dx2, d2phi_m/dy2 and dphi_m/dy of the manufactured solution phi_m = X(x) Y(y).

    X satisfies the homogeneous conditions of the airfoil problem: 'elliptic' X = sin(pi x/2L) (phi = 0 at A,
    phi_x = 0 at C); 'hyperbolic' X = sin^2(pi x/2L) (phi = phi_x = 0 at A). Y = cos(pi y/H) + (1 - y/H)^3 has
    phi_y = 0 at B and a non-zero flux at y = 0 (plays the role of the airfoil condition E).
    """
    L, H = domain["x_max"], domain["y_max"]
    k = np.pi / (2.0 * L)
    if kind == "elliptic":
        X, Xxx = np.sin(k * x), -k**2 * np.sin(k * x)
    elif kind == "hyperbolic":
        X, Xxx = np.sin(k * x) ** 2, 2.0 * k**2 * np.cos(2.0 * k * x)
    else:
        raise ValueError(f"Unknown MMS kind '{kind}'. Options: 'elliptic', 'hyperbolic'.")
    s = 1.0 - y / H
    Y = np.cos(np.pi * y / H) + s**3
    Yy = -np.pi / H * np.sin(np.pi * y / H) - 3.0 / H * s**2
    Yyy = -(np.pi / H) ** 2 * np.cos(np.pi * y / H) + 6.0 / H**2 * s
    return X * Y, Xxx * Y, X * Yyy, X * Yy


def mms_solve(g, case, domain, kind):
    """Solve (1 - M^2) phi_xx + phi_yy = f with the manufactured source and wall flux; return phi and phi_m (ny x nx)."""
    xc, yc = np.meshgrid(g["xc"], g["yc"])
    phi_m, phi_xx, phi_yy, _ = mms_solution(xc, yc, domain, kind)
    _, _, _, wall = mms_solution(g["xc"], 0.0, domain, kind)            # dphi_m/dy on the south faces (y = 0)
    rhs = ((1.0 - case["M_inf"] ** 2) * phi_xx + phi_yy).ravel()
    rhs[:g["nx"]] += wall / g["dy"][0]                                 # wall flux moved to the right-hand side
    ops = build_operators(g, case["bc_inflow"], case["bc_outflow"], case["bc_farfield"])
    nonlinear = {"tol": 1e-12, "max_iter": 3, "relax": 1.0}            # linear equation: one Newton step
    phi, *_ = newton(np.zeros(g["nx"] * g["ny"]), ops, rhs, dict(case, nx=g["nx"], ny=g["ny"]), 1.0, 1.4,
                     case["solver"], nonlinear)
    return phi.reshape(g["ny"], g["nx"]), phi_m


def mms_study(n_list, case, domain, kind):
    """Return h and the MMS errors (interior L2, interior L_inf, boundary-cell L_inf) on uniform n x n grids."""
    out = {"h": [], "L2_int": [], "Linf_int": [], "Linf_bnd": []}
    for n in n_list:
        g = make_grid(domain, {"nx": n, "ny": n, "kind_x": "uniform", "kind_y": "uniform",
                               "first_x": None, "first_y": None})
        phi, phi_m = mms_solve(g, case, domain, kind)
        err = np.abs(phi - phi_m)
        bnd = np.zeros_like(err, dtype=bool)                           # cells touching a boundary
        bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
        out["h"].append(domain["x_max"] / n)
        out["L2_int"].append(np.sqrt(np.mean(err[~bnd] ** 2)))
        out["Linf_int"].append(err[~bnd].max())
        out["Linf_bnd"].append(err[bnd].max())
    return {key: np.array(val) for key, val in out.items()}


def mms_order_table(results, n_list):
    """Return a text table of the MMS errors and observed orders p = log(e_(k-1)/e_k) / log(h_(k-1)/h_k)."""
    keys = ("L2_int", "Linf_int", "Linf_bnd")
    lines = []
    for name, r in results.items():
        lines += [name, f"{'N':>6s} {'h':>10s}" + "".join(f" {key:>10s} {'p':>5s}" for key in keys)]
        orders = {key: local_orders(r["h"], r[key]) for key in keys}
        for k in range(len(n_list)):
            row = f"{n_list[k]:>6d} {r['h'][k]:10.3e}"
            for key in keys:
                p = "--" if k == 0 else f"{orders[key][k]:.2f}"
                row += f" {r[key][k]:10.3e} {p:>5s}"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Exploration: treatment of the non-linearity (Newton vs Picard)
# ============================================================================
def nonlinear_study(g, case, gas, airfoil, runs):
    """Solve one case with each named non-linear setting; return the residual histories and a text summary table."""
    histories = {}
    lines = [f"{'run':36s} {'iterations':>10s} {'CPU [s]':>8s} {'final res.':>11s}  outcome"]
    for label, nonlinear in runs.items():
        t0 = time.perf_counter()
        sol = solve_potential_flow(g, case, gas, airfoil, nonlinear)
        cpu = time.perf_counter() - t0
        reached = [m for m, h in sol["all_histories"].items() if h[-1] < nonlinear["tol"]]
        if sol["history"][-1] < nonlinear["tol"]:
            outcome = "converged"
        else:
            outcome = f"diverged after M = {max(reached):.2f}" if reached else "diverged"
        histories[label] = np.concatenate(list(sol["all_histories"].values()))
        lines.append(f"{label:36s} {len(histories[label]):>10d} {cpu:>8.2f} {sol['history'][-1]:>11.1e}  {outcome}")
    return histories, "\n".join(lines)


def secant_type_conflicts(sol, case, gas):
    """Return the faces used by backward balances, and how many have A < 0 but secant a > 0 (wrong type for Picard)."""
    _, A = x_flux(sol["ops"]["gx"] @ sol["phi"], case["M_inf"], sol["U_inf"], gas["gamma"], case["equation"])
    a_secant = 0.5 * ((1.0 - case["M_inf"] ** 2) + A)
    used = np.abs(sol["ops"]["div_backward"]).T @ sol["masks"][1] > 0
    return int(used.sum()), int((used & (A < 0) & (a_secant > 0)).sum())


# ============================================================================
# Exploration: uniform vs non-uniform grids (accuracy against cost)
# ============================================================================
def wall_cp(sol, g):
    """Return x and the linearised wall pressure coefficient Cp = -2 u/U on y = 0 (phi extrapolated to the wall)."""
    phi_wall = sol["phi"][:g["nx"]] - sol["wall_y"][:g["nx"]] * g["dy"][0]   # phi_P - flux dy/2 (wall_y = flux/2)
    return g["xc"], -2.0 * np.gradient(phi_wall, g["xc"]) / sol["U_inf"]


def wave_drag(sol, g, airfoil):
    """Return the drag coefficient Cd = 2 int Cp dy_s/dx dx/c of the airfoil (upper and lower sides)."""
    _, cp = wall_cp(sol, g)
    return 2.0 * np.sum(cp * wall_slope(g["xf"], airfoil) * g["dx"]) / airfoil["chord"]


def grid_study(domain, families, factors, reference, case, gas, airfoil, nonlinear):
    """Solve one case on refined grids of each family; return cells, CPU time, Cd and wall-Cp errors against a
    reference grid (family, factor), plus a text table."""
    def run(grid):
        g = make_grid(domain, grid)
        t0 = time.perf_counter()
        sol = solve_potential_flow(g, case, gas, airfoil, nonlinear)
        cpu = time.perf_counter() - t0
        x, cp = wall_cp(sol, g)
        return {"cells": g["nx"] * g["ny"], "nx": g["nx"], "ny": g["ny"], "cpu": cpu,
                "cd": wave_drag(sol, g, airfoil), "x": x, "cp": cp, "res": sol["history"][-1]}

    ref = run(refine_grid(families[reference[0]], reference[1]))
    if not ref["res"] < nonlinear["tol"]:
        raise RuntimeError(f"Reference solution did not converge (residual {ref['res']:.1e}): "
                           "use a longer continuation or another reference grid.")
    on = (ref["x"] > airfoil["x_le"]) & (ref["x"] < airfoil["x_le"] + airfoil["chord"])
    results = {}
    lines = [f"reference: {reference[0]} x{reference[1]} ({ref['nx']} x {ref['ny']} = {ref['cells']} cells), "
             f"Cd = {ref['cd']:.5f}, CPU = {ref['cpu']:.1f} s",
             f"{'family':30s} {'factor':>6s} {'nx x ny':>11s} {'cells':>7s} {'CPU [s]':>8s} {'Cd':>9s} "
             f"{'|dCd|/Cd':>9s} {'Cp L1 err':>10s}"]
    for name, grid in families.items():
        r = {key: [] for key in ("cells", "cpu", "cd", "cd_err", "cp_err")}
        for factor in factors:
            s = run(refine_grid(grid, factor))
            if s["res"] < nonlinear["tol"]:
                cp_on_ref = np.interp(ref["x"][on], s["x"], s["cp"])  # Cp error on the chord, reference points
                s["cd_err"] = abs(s["cd"] - ref["cd"]) / abs(ref["cd"])
                s["cp_err"] = np.mean(np.abs(cp_on_ref - ref["cp"][on]))
                values = f"{s['cd']:>9.5f} {s['cd_err']:>9.2e} {s['cp_err']:>10.2e}"
            else:                                                    # not converged: left out of the plots
                s["cd"] = s["cd_err"] = s["cp_err"] = np.nan
                values = f"  not converged (residual {s['res']:.1e})"
            for key in r:
                r[key].append(s[key])
            lines.append(f"{name:30s} {factor:>6g} {s['nx']:>5d} x {s['ny']:<4d} {s['cells']:>7d} {s['cpu']:>8.2f} "
                         + values)
        results[name] = {key: np.array(val) for key, val in r.items()}
    return results, ref, "\n".join(lines)
