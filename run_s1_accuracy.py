"""Section 1.1 - Accuracy of FD first-derivative formulas for f = exp(sin(k x)) at x = 40."""

import os
import sys

# Make the cfdlib folder (next to this script) importable, whatever the working directory is
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running cell-by-cell in Spyder/IPython
    HERE = os.getcwd()
sys.path.insert(0, HERE)

from cfdlib.fd import fd_formula_text, truncation_term, accuracy_h_values, accuracy_study
from cfdlib.common import observed_order, save_figure
from cfdlib.plots import plot_accuracy, finish_figures

# ============================== PARAMETERS ==============================
X0        = 40.0                        # point where f'(x) is evaluated
FUNCTIONS = {1: 0.02, 2: 0.2}           # function index n of f_n -> wavenumber k
SCHEMES   = ["central-2", "backward-1"] # options: see cfdlib.fd.STENCILS
H_MIN     = 1e-8                        # finest h (coarsest is 2*pi/k)
N_H       = 300                         # number of h values
H_FIT     = (1e-3, 1e-1)                # h-range used to measure the slope
XLIM      = (1e-8, 1e3)                 # same axes for both figures
YLIM      = (1e-16, 1e1)
FIG_DIR   = os.path.join(HERE, "figures")
SHOW      = True                        # show figures on screen
# ========================================================================

print("Finite-difference formulas and leading truncation error:")
for scheme in SCHEMES:
    print("  " + fd_formula_text(scheme))

for n, k in FUNCTIONS.items():
    h = accuracy_h_values(k, H_MIN, N_H)
    results = accuracy_study(X0, k, SCHEMES, h)
    orders = {s: observed_order(h, err, H_FIT) for s, (err, _) in results.items()}
    for s in SCHEMES:
        print(f"f{n} (k={k}): {s:<11s} analytical order {truncation_term(s)[0]}, observed slope {orders[s]:.3f}")
    fig = plot_accuracy(h, results, orders, f"$f_{n}$, $k={k}$", XLIM, YLIM)
    save_figure(fig, f"s1_accuracy_f{n}", FIG_DIR)
    print(os.path.abspath(FIG_DIR), os.listdir(FIG_DIR))

finish_figures(SHOW)
