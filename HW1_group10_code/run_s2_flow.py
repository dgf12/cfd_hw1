"""exercise 2 (fv) - Linearised (3a) and transonic small-disturbance (3b) potential flow over a diamond airfoil ."""

# Grupo 10: David Ferreira, Martim Pestana, Francisco Boavida, Pedro Pereira

import os
import sys

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      
    HERE = os.getcwd()
sys.path.insert(0, HERE)

import numpy as np
from cfdlib.fv import (make_grid, diamond_height, solve_potential_flow, flow_fields, discretisation_table,
                       measure_mach_angle, mach_angle_refinement, mms_study, mms_order_table,
                       nonlinear_study, secant_type_conflicts, refine_grid, grid_study)
from cfdlib.common import save_figure, save_text
from cfdlib.plots import (plot_field, plot_residuals, colour_limits, plot_mms_errors, plot_meshes,
                          plot_grid_study, finish_figures)

# ============================== PARAMETERS ==============================
GAS     = {"gamma": 1.4, "R": 287.0, "T_inf": 288.0, "p_inf": 101325.0}
AIRFOIL = {"chord": 1.0, "t_half": 0.03, "x_le": 3.0}           # t/2 = 0.03 c, leading edge at x_airfoil
DOMAIN  = {"x_max": 8.0, "y_max": 20.0}                         # [m] (c = 1 m)
GRID    = {"nx": 240, "kind_x": "uniform", "first_x": None,     # 'uniform' | 'geometric' (first cell, constant ratio)
           "ny": 90,  "kind_y": "geometric", "first_y": 0.02}
SOLVER  = {"name": "direct", "tol": 1e-10, "max_iter": 20000, "omega": 1.9}   # 'direct' | 'jacobi' | 'gauss-seidel' | 'sor'
CASES   = {                                                     # regime -> choices (selected by name)
    "subsonic":   {"M_inf": 0.30, "equation": "linear", "scheme_x": "central",  "scheme_y": "central",
                   "bc_inflow": "phi",      "bc_outflow": "neumann", "bc_farfield": "neumann", "solver": SOLVER},
    "transonic":  {"M_inf": 0.91, "equation": "tsd",    "scheme_x": "switch",   "scheme_y": "central",
                   "bc_inflow": "phi",      "bc_outflow": "neumann", "bc_farfield": "neumann", "solver": SOLVER},
    "supersonic": {"M_inf": 2.20, "equation": "linear", "scheme_x": "backward", "scheme_y": "central",
                   "bc_inflow": "phi+phix", "bc_outflow": "none",    "bc_farfield": "neumann", "solver": SOLVER},
}
# options -> equation: 'linear' (3a) | 'tsd' (3b);  scheme_x: 'central' | 'backward' | 'switch' (Murman, on sign of A)
#            bc_inflow: 'phi' | 'phi+phix';  bc_outflow: 'neumann' | 'dirichlet' | 'none';  bc_farfield: 'neumann' | 'dirichlet'
NONLINEAR = {"method": "newton",                                # 'newton': phi <- phi - relax J^-1 R
             "tol": 1e-10, "max_iter": 30, "relax": 1.0,        # 'picard': phi <- phi + relax (phi* - phi)
             "continuation": [0.80, 0.85, 0.88, 0.90]}          # M_inf solved first for eq. (3b) ([] = none)
FIELDS  = {"phi": r"$\phi/(U_\infty c)$", "mach": "$M$", "rho": r"$\rho/\rho_\infty$"}
WINDOW  = {"x": (1.5, 6.5), "y": (0.0, 2.5)}                    # plotted region
COLOUR_SCALE = "span"                                           # 'common' (same range) | 'span' (same width) | 'own' (own plot for each case)
MACH_ANGLE = {"method": "steepest", "threshold": 0.5,           # 'steepest' | 'threshold' (first |u| > threshold max|u|)
              "y_range": (0.05, 0.5)}                           # rows used for the fit [m]
FIG_DIR = os.path.join(HERE, "figures")
SHOW    = True                                                  #Show the plots or just save them

FULL    = False                                                 # True: also the grid-refinement studies (slower)
REFINE  = [1, 2, 4]                                             # grid factors for the Mach-angle refinement (FULL)

VERIFY  = True                                                  # manufactured-solution verification
MMS_DOMAIN = {"x_max": 1.0, "y_max": 1.0}
MMS_N   = [8, 16, 32, 64, 128, 256] if FULL else [8, 16, 32, 64]
MMS_CASES = {                                                   # label -> (case, manufactured solution kind)
    "central, M = 0.3":  (dict(CASES["subsonic"]), "elliptic"),
    "backward, M = 2.2": (dict(CASES["supersonic"]), "hyperbolic"),
}

EXPLORE_NONLINEAR = True                                        # exploration: Newton vs Picard on eq. (3b)
NONLINEAR_RUNS = {                                              # label -> non-linear settings
    "Newton, continuation":            dict(NONLINEAR),
    "Newton, no continuation":         dict(NONLINEAR, continuation=[]),
    "Picard relax 1.0, continuation":  dict(NONLINEAR, method="picard", relax=1.0, max_iter=300),
    "Picard relax 0.5, continuation":  dict(NONLINEAR, method="picard", relax=0.5, max_iter=300),
    "Picard relax 0.2, continuation":  dict(NONLINEAR, method="picard", relax=0.2, max_iter=300),
    "Picard relax 0.1, no continuation": dict(NONLINEAR, method="picard", relax=0.1, max_iter=300, continuation=[]),
}
RESIDUAL_YLIM = (1e-14, 1e8)                                    # residual axis of the exploration figure

EXPLORE_GRIDS = True                                            # exploration: uniform vs non-uniform grids
GRID_CASE = "transonic"                                         # regime used for the grid study
GRID_FAMILIES = {                                               # label -> grid parameters at factor 1
    "U: uniform x, uniform y":     {"nx": 240, "kind_x": "uniform", "first_x": None,
                                    "ny": 90, "kind_y": "uniform", "first_y": None},
    "G: uniform x, geometric y":   dict(GRID),
    "C: clustered x, geometric y": {"nx": None, "kind_x": "clustered", "first_x": 1.0 / 30.0, "ratio_x": 1.05,
                                    "cluster_x": (AIRFOIL["x_le"] - 0.5, AIRFOIL["x_le"] + AIRFOIL["chord"] + 0.5),
                                    "ny": 90, "kind_y": "geometric", "first_y": 0.02},
}
GRID_FACTORS = [0.5, 1, 2] if FULL else [0.5, 1]                # refinement factors (cells x factor per direction)
GRID_REFERENCE = ("C: clustered x, geometric y", 4 if FULL else 2)   
GRID_NONLINEAR = dict(NONLINEAR, continuation=[0.5, 0.6, 0.7, 0.75] + # finer grids need a longer continuation
                      [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90])
MESH_SCHEME_FACTOR = 0.25                                       
MESH_ZOOM = {"x": (AIRFOIL["x_le"] - 1.0, AIRFOIL["x_le"] + 2.0), "y": (0.0, 1.5)}
# ======================== END OF PARAMETERS ================================================

g = make_grid(DOMAIN, GRID)
print(f"Grid {GRID['nx']} x {GRID['ny']}: dx = {g['dx'].min():.4f}, dy from {g['dy'].min():.4f} to {g['dy'].max():.3f}")
save_text(discretisation_table(CASES, NONLINEAR), "s2_flow_table", FIG_DIR)

solutions, fields = {}, {}
for regime, case in CASES.items():
    print(f"\n{regime} (M = {case['M_inf']}):")
    solutions[regime] = solve_potential_flow(g, case, GAS, AIRFOIL, NONLINEAR)
    fields[regime] = flow_fields(solutions[regime], g, case, GAS, AIRFOIL)

if "supersonic" in CASES:
    beta = measure_mach_angle(g, fields["supersonic"]["u"], MACH_ANGLE)
    beta_th = np.degrees(np.arcsin(1.0 / CASES["supersonic"]["M_inf"]))
    print(f"\nSupersonic Mach-wave angle ({MACH_ANGLE['method']}): measured {beta:.2f} deg, "
          f"arcsin(1/M) = {beta_th:.2f} deg")
    if FULL:                                                    # angle against grid refinement, both methods
        methods = {name: dict(MACH_ANGLE, method=name) for name in ("threshold", "steepest")}
        save_text(mach_angle_refinement(DOMAIN, GRID, REFINE, CASES["supersonic"], GAS, AIRFOIL, NONLINEAR,
                                        methods), "s2_mach_angle", FIG_DIR)

x_air = np.linspace(AIRFOIL["x_le"], AIRFOIL["x_le"] + AIRFOIL["chord"], 101)
for key, label in FIELDS.items():
    for regime, f in fields.items():
        vmin, vmax = colour_limits(fields, key, regime, g, WINDOW, COLOUR_SCALE)
        fig = plot_field(g, f[key], vmin, vmax, label, f"{regime}, $M_\\infty = {CASES[regime]['M_inf']}$",
                         x_air, diamond_height(x_air, AIRFOIL), WINDOW)
        save_figure(fig, f"s2_{regime}_{key}", FIG_DIR)

nonlinear = {f"{r} (M = {m})": h for r, s in solutions.items() if CASES[r]["equation"] == "tsd"
             for m, h in s["all_histories"].items()}
if nonlinear:
    save_figure(plot_residuals(nonlinear, f"{NONLINEAR['method'].capitalize()} convergence, eq. (3b)"),
                "s2_transonic_newton", FIG_DIR)

if VERIFY:                                                      
    print("\nManufactured-solution verification:")
    mms = {name: mms_study(MMS_N, case, MMS_DOMAIN, kind) for name, (case, kind) in MMS_CASES.items()}
    save_text(mms_order_table(mms, MMS_N), "s2_verification_table", FIG_DIR)
    save_figure(plot_mms_errors(mms), "s2_verification_error", FIG_DIR)

if EXPLORE_NONLINEAR:                                           # exploration: executes non-linearity when set to True
    print("\nExploration: Newton vs Picard (transonic):")
    histories, table = nonlinear_study(g, CASES["transonic"], GAS, AIRFOIL, NONLINEAR_RUNS)
    n_backward, n_conflict = secant_type_conflicts(solutions["transonic"], CASES["transonic"], GAS)
    table += (f"\nconverged solution: {n_backward} faces in backward balances, {n_conflict} of them with A < 0 "
              f"but secant a > 0 (Picard freezes a wrong-type operator there)")
    save_text(table, "s2_explore_nonlinear", FIG_DIR)
    save_figure(plot_residuals(histories, "Transonic: Newton vs Picard (all continuation steps)", RESIDUAL_YLIM),
                "s2_explore_nonlinear", FIG_DIR)

if EXPLORE_GRIDS:                                               # exploration: grid families, accuracy vs cost, executes when True
    meshes = {name: make_grid(DOMAIN, refine_grid(grid, MESH_SCHEME_FACTOR)) for name, grid in GRID_FAMILIES.items()}
    save_figure(plot_meshes(meshes, x_air, diamond_height(x_air, AIRFOIL), MESH_ZOOM), "s2_explore_meshes", FIG_DIR)
    print(f"\nExploration: grid families ({GRID_CASE}):")
    grid_results, grid_ref, table = grid_study(DOMAIN, GRID_FAMILIES, GRID_FACTORS, GRID_REFERENCE,
                                               CASES[GRID_CASE], GAS, AIRFOIL, GRID_NONLINEAR)
    save_text(table, "s2_explore_grids", FIG_DIR)
    save_figure(plot_grid_study(grid_results, grid_ref["cells"]), "s2_explore_grids", FIG_DIR)

finish_figures(SHOW)
