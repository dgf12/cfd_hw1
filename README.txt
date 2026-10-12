CFM 2026/2027 - Homework 1 - Group ##
=====================================

Language and packages
---------------------
Python 3.13 (tested with Python 3.13.16)
NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 (no other packages)
>>> UPDATE these versions to the ones on the machine used for the oral discussion
    (python --version ; python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)")

Structure
---------
cfdlib/                   all functions (no numerical method in the run scripts)
    common.py             analytical test function, linear solvers (direct, Jacobi, Gauss-Seidel, SOR),
                          error norms, observed order, saving figures and text tables
    fd.py                 Section 1: FD stencils (by name), truncation error, sparse 1D operator, 1D convection
    fv.py                 Section 2: grid, airfoil, sparse FV operators (Kronecker products), boundary conditions,
                          Murman switch, Newton iteration, post-processing (eq. 4), Mach-wave angle, discretisation table
    plots.py              plotting functions
run_s1_accuracy.py        Section 1.1
run_s1_convection.py      Section 1.2
run_s2_flow.py            Section 2 (three regimes)

Every choice is set in the PARAMETERS block at the top of each run script and passed to the functions.
Schemes, boundary conditions and solvers are chosen by name (one-word change), e.g.
    Section 1:  SCHEME = "central-2" | "backward-1" | "backward-2" | "forward-1" | "forward-2" | "central-4"
    Section 2:  equation    'linear' (3a) | 'tsd' (3b)
                scheme_x    'central' | 'backward' | 'switch' (central where A > 0, backward where A < 0)
                scheme_y    'central'
                bc_inflow   'phi' (phi = 0) | 'phi+phix' (phi = 0 and dphi/dx = 0)          boundary A
                bc_outflow  'neumann' (dphi/dx = 0) | 'dirichlet' (phi = 0) | 'none'       boundary C
                bc_farfield 'neumann' (dphi/dy = 0) | 'dirichlet' (phi = 0)                boundary B
                solver      'direct' | 'jacobi' | 'gauss-seidel' | 'sor' (with tol, max_iter, omega)
                grid        kind_x / kind_y 'uniform' | 'geometric' (first cell size, constant growth ratio)
Unknown names stop the run with the list of valid options.

How to run (from this folder or with F5 in Spyder; outputs are written to figures/)
-----------------------------------------------------------------------------------
python run_s1_accuracy.py    (~2 s)  -> s1_accuracy_f1.pdf, s1_accuracy_f2.pdf
                                        console: FD formulas, truncation errors, analytical and observed orders
python run_s1_convection.py  (~2 s)  -> s1_convection_solution.pdf, s1_convection_error.pdf,
                                        s1_convection_norms.pdf (only if FULL = True); console: observed orders
python run_s2_flow.py        (~15 s) -> s2_flow_table.txt (fluxes, switch, BCs A-E, solver per regime)
                                        s2_<regime>_<field>.pdf, regime = subsonic | transonic | supersonic,
                                        field = phi | mach | rho  (same colour scale if COMMON_SCALE = True)
                                        s2_transonic_newton.pdf (Newton residual history)
                                        console: Newton steps, supersonic cells, Mach-wave angle vs arcsin(1/M)

Flags: FULL = True/False (Section 1.2 refinement study); SHOW = True/False shows the figures on screen.
In Spyder, figures open in a separate window with Preferences > IPython console > Graphics > Backend = Automatic.

Section 2 method summary
------------------------
Cell-centred FV on the rectangle; unknowns ordered row by row (k = j*nx + i). The 2D operators are Kronecker products
of sparse 1D operators: face gradients (two-point differences between centres), central cell balance
(F_e - F_w)/dx_i and backward balance (F_w - F_ww)/dx_(i-1). The x-flux is F(u) = (1 - M^2) u for eq. (3a) and
F(u) = (1 - M^2) u - (gamma+1) M^2 u^2/(2 U) for eq. (3b), so dF/du = A.
Transonic switch (Murman, conservative form): mu_i = 1 where A_i < 0; cell i uses (1 - mu_i) x central balance
+ mu_(i-1) x backward balance (shock point: both; sonic point: neither).
Boundary conditions: A phi = 0 on the face (supersonic: also dphi/dx = 0 upstream, i.e. zero ghost flux);
B dphi/dy = 0; C dphi/dx = 0 (supersonic: none, gradient extrapolated, not used by the backward balance);
D dphi/dy = 0; E dphi/dy = U dys/dx using the cell mean of the slope (exact for the diamond).
Nonlinear solver: Newton, phi <- phi - relax J^-1 R, residual ||R||/||b|| < tol, with optional continuation in M
(0.80 -> 0.85 -> 0.88 -> 0.90 -> 0.91). Each Newton step is a linear system (direct sparse LU by default).
The linear equation (3a) uses the same code and converges in one Newton step.

Use of AI tools
---------------
The numerical methods were first written by the group members. The organisation of the code into the cfdlib module
and run scripts, the sparse (Kronecker) assembly, the generic stencil/truncation-error functions, the conservative
Murman switch with Newton iteration and the iterative linear solvers were written or restructured with the help
of an AI assistant (Claude, Anthropic), and then reviewed and verified by the group.
