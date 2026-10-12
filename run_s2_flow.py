"""Section 2 - Linearised (3a) and transonic small-disturbance (3b) potential flow over a diamond airfoil (cell-centred FV)."""

import os
import sys

# Make the cfdlib folder (next to this script) importable, whatever the working directory is
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running cell-by-cell in Spyder/IPython
    HERE = os.getcwd()
sys.path.insert(0, HERE)

import numpy as np
from cfdlib.fv import make_grid, diamond_height, solve_potential_flow, flow_fields, mach_wave_angle, discretisation_table
from cfdlib.common import save_figure, save_text
from cfdlib.plots import plot_field, plot_residuals, field_limits, finish_figures

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
NONLINEAR = {"tol": 1e-10, "max_iter": 30, "relax": 1.0,        # Newton: phi <- phi - relax J^-1 R
             "continuation": [0.80, 0.85, 0.88, 0.90]}          # M_inf solved first for eq. (3b) ([] = none)
FIELDS  = {"phi": r"$\phi/(U_\infty c)$", "mach": "$M$", "rho": r"$\rho/\rho_\infty$"}
WINDOW  = {"x": (1.5, 6.5), "y": (0.0, 2.5)}                    # plotted region
COMMON_SCALE = True                                             # True: same colour scale for the 3 regimes
MACH_ANGLE = {"threshold": 0.5, "y_range": (0.3, 1.5)}          # Mach-wave detection (supersonic)
FIG_DIR = os.path.join(HERE, "figures")
SHOW    = True
# ========================================================================

g = make_grid(DOMAIN, GRID)
print(f"Grid {GRID['nx']} x {GRID['ny']}: dx = {g['dx'].min():.4f}, dy from {g['dy'].min():.4f} to {g['dy'].max():.3f}")
save_text(discretisation_table(CASES), "s2_flow_table", FIG_DIR)

solutions, fields = {}, {}
for regime, case in CASES.items():
    print(f"\n{regime} (M = {case['M_inf']}):")
    solutions[regime] = solve_potential_flow(g, case, GAS, AIRFOIL, NONLINEAR)
    fields[regime] = flow_fields(solutions[regime], g, case, GAS, AIRFOIL)

if "supersonic" in CASES:
    beta = mach_wave_angle(g, fields["supersonic"]["u"], MACH_ANGLE["threshold"], MACH_ANGLE["y_range"])
    beta_th = np.degrees(np.arcsin(1.0 / CASES["supersonic"]["M_inf"]))
    print(f"\nSupersonic Mach-wave angle: measured {beta:.1f} deg, arcsin(1/M) = {beta_th:.1f} deg")

x_air = np.linspace(AIRFOIL["x_le"], AIRFOIL["x_le"] + AIRFOIL["chord"], 101)
for key, label in FIELDS.items():
    for regime, f in fields.items():
        vmin, vmax = field_limits(fields if COMMON_SCALE else {regime: f}, key, g, WINDOW)
        fig = plot_field(g, f[key], vmin, vmax, label, f"{regime}, $M_\\infty = {CASES[regime]['M_inf']}$",
                         x_air, diamond_height(x_air, AIRFOIL), WINDOW)
        save_figure(fig, f"s2_{regime}_{key}", FIG_DIR)

nonlinear = {f"{r} (M = {m})": h for r, s in solutions.items() if CASES[r]["equation"] == "tsd"
             for m, h in s["all_histories"].items()}
if nonlinear:
    save_figure(plot_residuals(nonlinear, "Newton convergence, eq. (3b)"), "s2_transonic_newton", FIG_DIR)

finish_figures(SHOW)
