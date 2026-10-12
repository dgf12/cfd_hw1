"""Plotting functions (post-processing) for all sections."""

import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (6.4, 4.8)
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]


def plot_accuracy(h, results, orders, title, xlim, ylim):
    """Log-log plot of FD error (solid) and leading truncation term (dashed) vs h, one colour per scheme."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for (scheme, (err, trunc)), color in zip(results.items(), COLORS):
        ax.loglog(h, err, "-", color=color, lw=1.0, label=f"{scheme}, error ($p$ = {orders[scheme]:.2f})")
        ax.loglog(h, trunc, "--", color=color, lw=1.0, label=f"{scheme}, leading term")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
    ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax.grid(True, which="major", ls=":", alpha=0.6)
    ax.set_xlabel("$h$", fontsize=8)
    ax.set_ylabel(r"$|(\mathrm{d}f/\mathrm{d}x)_{FD} - f'(40)|$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.03, 0.95, title, transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(loc="lower right", fontsize=6, frameon=False, handlelength=2.0, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    return fig


def plot_solution_1d(x, u, x_exact, u_exact, title):
    """Plot the numerical solution (markers) against the exact solution (line)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x_exact, u_exact, "k-", lw=1.5, label="exact $f_2(x)$")
    ax.plot(x, u, "o--", color=COLORS[1], ms=4, lw=1, label=f"numerical ($N={len(x) - 1}$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    return fig


def plot_error_1d(x, e, title):
    """Plot the pointwise error e_i against x_i."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, e, ".-", color=COLORS[0])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$u_i - f_2(x_i)$")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.6)
    return fig


def plot_norms(h, L2, Linf, p2, pinf, ref_order, title):
    """Log-log plot of the L2 and L_inf error norms vs h, with an O(h^ref_order) reference line."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(h, L2, "o-", color=COLORS[0], label=f"$L_2$ (slope {p2:.2f})")
    ax.loglog(h, Linf, "s-", color=COLORS[1], label=rf"$L_\infty$ (slope {pinf:.2f})")
    ax.loglog(h, Linf[0] * (h / h[0]) ** ref_order, "k--", label=f"$O(h^{ref_order})$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("error norm")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    return fig


def finish_figures(show):
    """Show all open figures if `show` is True, otherwise close them."""
    plt.show() if show else plt.close("all")


def plot_field(g, F, vmin, vmax, label, title, airfoil_x, airfoil_y, window):
    """Filled contours of a cell-centred field on the colour scale [vmin, vmax], thin isolines on its own range, airfoil."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cf = ax.contourf(g["xc"], g["yc"], np.clip(F, vmin, vmax), levels=np.linspace(vmin, vmax, 41), cmap="viridis")
    cf.set_edgecolor("face")                                   # no white seams between levels in the pdf
    ix = (g["xc"] >= window["x"][0]) & (g["xc"] <= window["x"][1])
    iy = (g["yc"] >= window["y"][0]) & (g["yc"] <= window["y"][1])
    local = F[np.ix_(iy, ix)]
    if local.max() > local.min():
        ax.contour(g["xc"], g["yc"], F, levels=np.linspace(local.min(), local.max(), 16)[1:-1],
                   colors="k", linewidths=0.3, alpha=0.6)
    ax.fill_between(airfoil_x, 0.0, airfoil_y, color="w", edgecolor="k", lw=0.8)
    ax.set_xlim(*window["x"])
    ax.set_ylim(*window["y"])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title(title)
    cb = fig.colorbar(cf, ax=ax, label=label, shrink=0.85)
    cb.locator = plt.MaxNLocator(6)
    cb.update_ticks()
    return fig


def plot_residuals(histories, title):
    """Semilog plot of residual histories (label -> list of residuals) against the iteration number."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (label, h), color in zip(histories.items(), COLORS):
        ax.semilogy(range(1, len(h) + 1), h, "-", color=color, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|K(\phi)\phi - b\| / \|b\|$")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    return fig


def field_limits(fields_by_case, key, g, window):
    """Return (min, max) of field `key` over all cases inside the plotted window (common colour scale)."""
    ix = (g["xc"] >= window["x"][0]) & (g["xc"] <= window["x"][1])
    iy = (g["yc"] >= window["y"][0]) & (g["yc"] <= window["y"][1])
    values = np.concatenate([f[key][np.ix_(iy, ix)].ravel() for f in fields_by_case.values()])
    return values.min(), values.max()
