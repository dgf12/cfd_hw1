"""Section 1.2 - Steady convection a du/dx = s(x) on [0, 40], u(0) = 1, exact solution u = f2(x)."""

import os
import sys

# Make the cfdlib folder (next to this script) importable, whatever the working directory is
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running cell-by-cell in Spyder/IPython
    HERE = os.getcwd()
sys.path.insert(0, HERE)

from cfdlib.fd import fd_formula_text, uniform_grid_1d, solve_convection_1d, convection_convergence, truncation_term
from cfdlib.common import test_function, observed_order, save_figure
from cfdlib.plots import plot_solution_1d, plot_error_1d, plot_norms, finish_figures

# ============================== PARAMETERS ==============================
A_CONV          = 1.0            # convection velocity a
L               = 40.0           # domain length
K               = 0.2            # wavenumber of f2 (source s = a f2'(x))
U0              = 1.0            # Dirichlet condition u(0)
N_PLOT          = 40             # grid used for the solution/error figures
SCHEME          = "central-2"    # interior scheme (see cfdlib.fd.STENCILS)
SCHEME_BOUNDARY = "backward-2"   # scheme where SCHEME does not fit (x = 40)
SOLVER          = {"name": "direct"}  # linear solver ('jacobi'/'gauss-seidel' do not converge: D is not diagonally dominant)
FULL            = True           # run the grid-refinement study
N_LIST          = [10, 20, 40, 80, 160, 320, 640, 1280]
N_FINE          = 1000           # points used to draw the exact solution
FIG_DIR         = os.path.join(HERE, "figures")
SHOW            = True
# ========================================================================

print("Interior: " + fd_formula_text(SCHEME))
print("Boundary: " + fd_formula_text(SCHEME_BOUNDARY))

# --- Solution and error for N = N_PLOT ---
x, u, h = solve_convection_1d(N_PLOT, L, A_CONV, K, U0, SCHEME, SCHEME_BOUNDARY, SOLVER)
x_fine, _ = uniform_grid_1d(0.0, L, N_FINE)
fig = plot_solution_1d(x, u, x_fine, test_function(x_fine, K), f"{SCHEME} + {SCHEME_BOUNDARY} at $x=L$, $N={N_PLOT}$")
save_figure(fig, "s1_convection_solution", FIG_DIR)
fig = plot_error_1d(x, u - test_function(x, K), f"Error, {SCHEME} + {SCHEME_BOUNDARY}, $N={N_PLOT}$")
save_figure(fig, "s1_convection_error", FIG_DIR)

# --- Grid-refinement study (L2 and L_inf norms) ---
if FULL:
    hs, L2, Linf = convection_convergence(N_LIST, L, A_CONV, K, U0, SCHEME, SCHEME_BOUNDARY, SOLVER)
    p2, pinf = observed_order(hs, L2), observed_order(hs, Linf)
    print(f"Observed order: L2 = {p2:.3f}, L_inf = {pinf:.3f}")
    fig = plot_norms(hs, L2, Linf, p2, pinf, truncation_term(SCHEME)[0], f"Error norms, {SCHEME} + {SCHEME_BOUNDARY}")
    save_figure(fig, "s1_convection_norms", FIG_DIR)

finish_figures(SHOW)
