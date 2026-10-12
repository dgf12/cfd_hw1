"""Shared utilities: analytical test function, linear solvers, error norms, figure output."""

import os
from math import comb

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# Analytical test function f(x) = exp(sin(k x))
# ----------------------------------------------------------------------------
def test_function(x, k):
    """Return f(x) = exp(sin(k x))."""
    return np.exp(np.sin(k * x))


def test_function_derivatives(x, k, n_max):
    """Return [f, f', ..., f^(n_max)] of f = exp(sin(k x)) (recursion f^(n+1) = sum C(n,j) g^(j+1) f^(n-j))."""
    # g = sin(k x)  ->  g^(m) = k^m sin(k x + m pi/2);  f' = g' f
    g = [k**m * np.sin(k * x + m * np.pi / 2) for m in range(n_max + 1)]
    f = [test_function(x, k)]
    for n in range(n_max):
        f.append(sum(comb(n, j) * g[j + 1] * f[n - j] for j in range(n + 1)))
    return f


# ----------------------------------------------------------------------------
# Linear solvers (selected by name)
# ----------------------------------------------------------------------------
def solve_linear(A, b, solver, x0):
    """Solve A x = b with solver['name'] ('direct', 'jacobi', 'gauss-seidel', 'sor'); return x and the residual history."""
    name = solver["name"]
    if name == "direct":                                        # Gauss elimination (sparse LU)
        return spla.spsolve(A.tocsc(), b), []
    if name not in ("jacobi", "gauss-seidel", "sor"):
        raise ValueError(f"Unknown solver '{name}'. Options: 'direct', 'jacobi', 'gauss-seidel', 'sor'.")
    A = A.tocsr()
    d = A.diagonal()
    if name == "jacobi":                                        # M = D
        M = sp.diags(d).tocsr()
    else:                                                       # M = D/omega + L  (omega = 1: Gauss-Seidel)
        omega = 1.0 if name == "gauss-seidel" else solver["omega"]
        M = (sp.tril(A, k=-1) + sp.diags(d / omega)).tocsr()
    x = x0.copy()
    b_norm = np.linalg.norm(b) if np.linalg.norm(b) > 0 else 1.0
    history = []
    for _ in range(solver["max_iter"]):                         # iteration loop (matrix-vector operations only)
        r = b - A @ x
        history.append(np.linalg.norm(r) / b_norm)
        if history[-1] < solver["tol"]:
            break
        x = x + (r / d if name == "jacobi" else spla.spsolve_triangular(M, r, lower=True))
    else:
        print(f"  WARNING: {name} stopped at max_iter = {solver['max_iter']}, residual {history[-1]:.2e}")
    return x, history


# ----------------------------------------------------------------------------
# Error analysis
# ----------------------------------------------------------------------------
def error_norms(e, h):
    """Return the discrete L2 norm sqrt(h * sum e^2) and the L_inf norm max|e|."""
    return np.sqrt(h * np.sum(e**2)), np.max(np.abs(e))


def observed_order(h, err, h_range=None):
    """Return the slope p of log(err) vs log(h) (least squares), optionally only for h in h_range."""
    h, err = np.asarray(h), np.asarray(err)
    mask = err > 0
    if h_range is not None:
        mask &= (h >= h_range[0]) & (h <= h_range[1])
    return np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)[0]


# ----------------------------------------------------------------------------
# Figure output
# ----------------------------------------------------------------------------
def save_figure(fig, name, folder):
    """Save a figure as <folder>/<name>.pdf, creating the folder if needed."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")
    return path


def save_text(text, name, folder):
    """Write `text` to <folder>/<name>.txt (tables and numbers quoted in the report) and print it."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"  saved {path}")
    return path
