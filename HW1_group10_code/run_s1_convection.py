"""Section 1.2 - Steady convection a du/dx = a f2'(x), u(0) = 1 on [0, 40]; exact solution u = f2(x).

Produces in figures/: s1_convection_solution.pdf, s1_convection_error.pdf, s1_convection_norms.pdf
and s1_convection_table.txt. Only parameters and calls to cfdlib.
"""

# Grupo 10: David Ferreira, Martim Pestana, Francisco Boavida, Pedro Pereira

import os
import sys

# Make the cfdlib folder (next to this script) importable, whatever the working directory is
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running cell-by-cell in Spyder/IPython
    HERE = os.getcwd()
sys.path.insert(0, HERE)

from cfdlib.fd import (fd_formula_text, uniform_grid_1d, predicted_error_1d, convection_solutions,
                       convection_refinement, convection_table)
from cfdlib.common import test_function, save_figure, save_text
from cfdlib.plots import (set_plot_style, plot_convection_solution, plot_convection_error, plot_convection_norms,
                          finish_figures)

# ============================== PARAMETERS ==============================
A_CONV  = 1.0                     # convection velocity a
L       = 40.0                    # domain [0, L]
K       = 0.2                     # f2(x) = exp(sin(K x)); source s = a f2'(x)
U0      = 1.0                     # Dirichlet condition u(0)
N_PLOT  = 40                      # grid of the solution and error figures
CASES   = {                       # label: (interior scheme, scheme where it does not fit, i.e. x = L)
    "backward-2": ("central-2", "backward-2"),
}                                 # schemes: see cfdlib.fd.STENCILS (e.g. add "backward-1": ("backward-1", "backward-1"))
SOLVER  = {"name": "direct"}      # sparse LU ('jacobi'/'gauss-seidel' do not converge: D is not diagonally dominant)
FULL    = True                    # True: adds the two finest grids to the refinement study
N_LIST  = [10, 20, 40, 80, 160, 320, 640] + ([1280, 2560] if FULL else [])
N_FIT   = 4                       # number of finest grids used to fit the order
REF_ORDERS = [2]                  # reference slopes in the norms figure
SHOW_PREDICTION = True            # dashed predicted error in the error figure
N_FINE  = 1000                    # points used to draw the exact solution and the prediction
FONT_SIZE = 8
FIG_SIZE  = (3.3, 2.6)            # inches; three figures side by side on a page
FIG_DIR   = os.path.join(HERE, "figures")
SHOW      = True                  # show the figures on screen
# ========================================================================

for label, (scheme, boundary) in CASES.items():
    print(f"{label}: interior {fd_formula_text(scheme)}\n{'':{len(label) + 2}}boundary {fd_formula_text(boundary)}")

set_plot_style(FONT_SIZE)
solutions = convection_solutions(CASES, N_PLOT, L, A_CONV, K, U0, SOLVER)
results = convection_refinement(CASES, N_LIST, L, A_CONV, K, U0, SOLVER, N_FIT)

x_fine, _ = uniform_grid_1d(0.0, L, N_FINE)
errors = {lab: u - test_function(x, K) for lab, (x, u, h, _) in solutions.items()}
predictions = {lab: predicted_error_1d(x_fine, K, h, sch) if SHOW_PREDICTION else None
               for lab, (x, u, h, sch) in solutions.items()}

save_figure(plot_convection_solution(solutions, x_fine, test_function(x_fine, K), FIG_SIZE),
            "s1_convection_solution", FIG_DIR)
save_figure(plot_convection_error(solutions, errors, x_fine, predictions, FIG_SIZE), "s1_convection_error", FIG_DIR)
save_figure(plot_convection_norms(results, REF_ORDERS, FIG_SIZE), "s1_convection_norms", FIG_DIR)
save_text(convection_table(results, N_FIT), "s1_convection_table", FIG_DIR)

finish_figures(SHOW)
