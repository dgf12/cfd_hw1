"""Finite-difference functions for Section 1: stencils, truncation error, sparse operators, 1D convection."""

from fractions import Fraction
from math import factorial

import numpy as np
import scipy.sparse as sp

from .common import test_function, test_function_derivatives, solve_linear, error_norms

# ----------------------------------------------------------------------------
# First-derivative stencils:  f'_i ~ (1/h) * sum_j c_j f_{i+s_j}
# Each entry: (offsets s_j, coefficients c_j). Select by name.
# ----------------------------------------------------------------------------
STENCILS = {
    "backward-1": ([-1, 0], [Fraction(-1), Fraction(1)]),
    "forward-1":  ([0, 1], [Fraction(-1), Fraction(1)]),
    "central-2":  ([-1, 1], [Fraction(-1, 2), Fraction(1, 2)]),
    "backward-2": ([-2, -1, 0], [Fraction(1, 2), Fraction(-2), Fraction(3, 2)]),
    "forward-2":  ([0, 1, 2], [Fraction(-3, 2), Fraction(2), Fraction(-1, 2)]),
    "central-4":  ([-2, -1, 1, 2], [Fraction(1, 12), Fraction(-8, 12), Fraction(8, 12), Fraction(-1, 12)]),
}


def get_stencil(scheme):
    """Return (offsets, coefficients) of the first-derivative stencil named `scheme`."""
    if scheme not in STENCILS:
        raise ValueError(f"Unknown scheme '{scheme}'. Options: {list(STENCILS)}")
    return STENCILS[scheme]


def truncation_term(scheme, m_max=10):
    """Return (p, C, m) such that FD - exact = C h^p f^(m) + ..., from the Taylor moments of the stencil."""
    offsets, coeffs = get_stencil(scheme)
    for m in range(2, m_max + 1):
        moment = sum(c * Fraction(s) ** m for s, c in zip(offsets, coeffs))
        if moment != 0:
            return m - 1, moment / factorial(m), m
    raise ValueError(f"No truncation term found up to order {m_max} for '{scheme}'.")


def fd_formula_text(scheme):
    """Return a text description of the FD formula and its leading truncation error."""
    offsets, coeffs = get_stencil(scheme)
    terms = " ".join(f"{'+' if c > 0 else '-'} {abs(c)} f(x{s:+d}h)".replace("x+0h", "x") for s, c in zip(offsets, coeffs))
    p, C, m = truncation_term(scheme)
    q = next(n - 1 for n in range(m + 1, 20) if sum(c * Fraction(s) ** n for s, c in zip(offsets, coeffs)) != 0)
    return f"{scheme:<11s}: f'(x) ~ [{terms} ] / h ;  FD - exact = {C} h^{p} f^({m})(x) + O(h^{q})"


# ----------------------------------------------------------------------------
# Section 1.1 - pointwise accuracy
# ----------------------------------------------------------------------------
def fd_derivative(func, x0, h, scheme):
    """Return the FD approximation of func'(x0) for each spacing in h, using the named stencil."""
    offsets, coeffs = get_stencil(scheme)
    return sum(float(c) * func(x0 + s * h) for s, c in zip(offsets, coeffs)) / h


def accuracy_study(x0, k, schemes, h):
    """Return {scheme: (|FD error|, |leading truncation term|)} for f = exp(sin(k x)) at x0, for each h."""
    p_max = max(truncation_term(s)[2] for s in schemes)
    derivs = test_function_derivatives(x0, k, p_max)
    func = lambda x: test_function(x, k)
    results = {}
    for scheme in schemes:
        err = np.abs(fd_derivative(func, x0, h, scheme) - derivs[1])
        p, C, m = truncation_term(scheme)
        trunc = np.abs(float(C) * h**p * derivs[m])
        results[scheme] = (err, trunc)
    return results


def accuracy_h_values(k, h_min, n_h):
    """Return n_h log-spaced grid sizes from h_min up to the coarse value h = 2 pi / k."""
    return np.logspace(np.log10(h_min), np.log10(2 * np.pi / k), n_h)


# ----------------------------------------------------------------------------
# Section 1.2 - sparse operators and steady convection
# ----------------------------------------------------------------------------
def uniform_grid_1d(x_start, x_end, N):
    """Return the N+1 points of a uniform grid on [x_start, x_end] and the spacing h."""
    x = np.linspace(x_start, x_end, N + 1)
    return x, x[1] - x[0]


def derivative_matrix_1d(n_points, h, scheme, scheme_boundary, skip_rows=()):
    """Assemble the sparse first-derivative matrix D; rows where `scheme` does not fit use `scheme_boundary`."""
    rows, cols, vals = [], [], []
    for i in range(n_points):
        if i in skip_rows:                       # row set later by a boundary condition
            continue
        for name in (scheme, scheme_boundary):
            offsets, coeffs = get_stencil(name)
            if all(0 <= i + s < n_points for s in offsets):
                rows += [i] * len(offsets)
                cols += [i + s for s in offsets]
                vals += [float(c) / h for c in coeffs]
                break
        else:
            raise ValueError(f"Row {i}: neither '{scheme}' nor '{scheme_boundary}' fits inside the grid.")
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_points, n_points))


def convection_system_1d(x, a, s, u0, scheme, scheme_boundary):
    """Return sparse A and rhs b of a D u = s with the Dirichlet condition u(x_0) = u0 in row 0."""
    n, h = len(x), x[1] - x[0]
    D = derivative_matrix_1d(n, h, scheme, scheme_boundary, skip_rows=(0,))
    A = a * D + sp.csr_matrix(([1.0], ([0], [0])), shape=(n, n))
    b = s.copy()
    b[0] = u0
    return A.tocsr(), b


def solve_convection_1d(N, L, a, k, u0, scheme, scheme_boundary, solver):
    """Solve a du/dx = a f'(x), u(0)=u0 on [0, L] with N+1 points; return x, u, h."""
    x, h = uniform_grid_1d(0.0, L, N)
    s = a * test_function_derivatives(x, k, 1)[1]
    A, b = convection_system_1d(x, a, s, u0, scheme, scheme_boundary)
    u, _ = solve_linear(A, b, solver, np.zeros_like(b))
    return x, u, h


def convection_convergence(N_list, L, a, k, u0, scheme, scheme_boundary, solver):
    """Return arrays h, L2 and L_inf of the error u - f(x) for each N in N_list."""
    hs, L2, Linf = [], [], []
    for N in N_list:
        x, u, h = solve_convection_1d(N, L, a, k, u0, scheme, scheme_boundary, solver)
        e2, einf = error_norms(u - test_function(x, k), h)
        hs.append(h), L2.append(e2), Linf.append(einf)
    return np.array(hs), np.array(L2), np.array(Linf)
