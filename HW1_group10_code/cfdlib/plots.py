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


def set_plot_style(font_size):
    """Common font size for all figures (readability in the report)."""
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": font_size - 1})


def plot_convection_solution(solutions, x_exact, u_exact, size):
    """u_i against x for each case (markers) and the exact solution (dashed)."""
    fig, ax = plt.subplots(figsize=size)
    one = len(solutions) == 1
    for (lab, (x, u, h, _)), col in zip(solutions.items(), COLORS):
        ax.plot(x, u, "-o", ms=3, lw=1, mfc="none", color=col,
                label="Numerical ($u_i$)" if one else f"Numerical ($u_i$), {lab}")
    ax.plot(x_exact, u_exact, "k--", lw=1, label="Exact ($f_2(x)$)")
    N = len(next(iter(solutions.values()))[0]) - 1
    ax.set(xlabel="$x$", ylabel="$u$", xlim=(x_exact[0], x_exact[-1]),
           title=f"Steady convection: solution (N = {N}, {', '.join(solutions)})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.3)
    return fig


def plot_convection_error(solutions, errors, x_pred, predictions, size):
    """Error u_i - f2(x_i) against x for each case, with the predicted error (dashed) where available."""
    fig, ax = plt.subplots(figsize=size)
    tag = (lambda lab: "") if len(solutions) == 1 else (lambda lab: f", {lab}")
    for (lab, (x, u, h, _)), col in zip(solutions.items(), COLORS):
        ax.plot(x, errors[lab], "-o", ms=2.5, lw=1, color=col, label=f"Numerical ($u_i - f_2(x_i)$){tag(lab)}")
        if predictions[lab] is not None:
            ax.plot(x_pred, predictions[lab], "--", lw=1, color="k" if len(solutions) == 1 else col,
                    label=f"estimated error{tag(lab)}")
    N = len(next(iter(solutions.values()))[0]) - 1
    ax.set(xlabel="$x$", ylabel=r"$u_i - f_2(x_i)$", xlim=(x_pred[0], x_pred[-1]),
           title=f"Steady convection: error (N = {N})")
    ax.grid(True, alpha=0.3)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(top=y1 + 0.45 * (y1 - y0))                     # free space for the legend
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    return fig


def plot_convection_norms(results, ref_orders, size):
    """Log-log L2 (filled) and L_inf (open) error norms against h, with reference slopes."""
    fig, ax = plt.subplots(figsize=size)
    one = len(results) == 1
    for (lab, r), col in zip(results.items(), COLORS):
        tag = "" if one else f"{lab} "
        ax.loglog(r["h"], r["l2"], "-o", ms=3, lw=1, color="tab:red" if one else col,
                  label=f"{tag}$L_2$ (p={r['fit_l2']:.2f})")
        ax.loglog(r["h"], r["linf"], "-s", ms=3, lw=1, mfc="none", color=col,
                  label=rf"{tag}$L_\infty$ (p={r['fit_linf']:.2f})")
    h_all = np.concatenate([r["h"] for r in results.values()])
    e_min = min(r["linf"][-1] for r in results.values())
    hr = np.array([h_all.min(), h_all.max()])
    for p, ls in zip(ref_orders, [":", "--", "-."]):
        ax.loglog(hr, 0.3 * e_min * (hr / hr[0]) ** p, "k" + ls, lw=0.8, label=f"$\\propto h^{p}$")
    ax.set(xlabel="$h$", ylabel="Error norm", title="Grid convergence")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1e4)                    # free space for the legend
    ax.legend(frameon=False, fontsize="small", loc="upper left", ncol=2)
    fig.tight_layout(pad=0.3)
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


def plot_residuals(histories, title, ylim=None):
    """Semilog plot of residual histories (label -> list of residuals) against the iteration number (optional ylim)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (label, h), color in zip(histories.items(), COLORS):
        ax.semilogy(range(1, len(h) + 1), h, "-", color=color, label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
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


def colour_limits(fields_by_case, key, regime, g, window, scale):
    """Return (vmin, vmax) of field `key` for one regime: 'common' (same range for all regimes), 'own' (own range),
    or 'span' (same range WIDTH for all regimes, centred on this regime's own range)."""
    if scale == "common":
        return field_limits(fields_by_case, key, g, window)
    if scale == "own":
        return field_limits({regime: fields_by_case[regime]}, key, g, window)
    if scale == "span":
        width = max(np.subtract(*field_limits({r: f}, key, g, window)[::-1]) for r, f in fields_by_case.items())
        lo, hi = field_limits({regime: fields_by_case[regime]}, key, g, window)
        return 0.5 * (lo + hi) - 0.5 * width, 0.5 * (lo + hi) + 0.5 * width
    raise ValueError(f"Unknown colour scale '{scale}'. Options: 'common', 'own', 'span'.")


def plot_meshes(grids, airfoil_x, airfoil_y, zoom):
    """Draw the face lines of each grid (label -> grid dict): full domain on top, zoom near the airfoil below."""
    fig, axes = plt.subplots(2, len(grids), figsize=(3.6 * len(grids), 7.5),
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for col, (label, g) in enumerate(grids.items()):
        for ax, window in ((axes[0, col], None), (axes[1, col], zoom)):
            ax.vlines(g["xf"], g["yf"][0], g["yf"][-1], colors="k", lw=0.3)
            ax.hlines(g["yf"], g["xf"][0], g["xf"][-1], colors="k", lw=0.3)
            ax.fill_between(airfoil_x, 0.0, airfoil_y, color=COLORS[1], lw=0)
            ax.plot(airfoil_x, 0.0 * airfoil_x, "-", color=COLORS[1], lw=2.0)     # condition E segment
            if window is not None:
                ax.set_xlim(*window["x"])
                ax.set_ylim(*window["y"])
            ax.set_aspect("equal")
            ax.set_xlabel("$x/c$")
            ax.set_ylabel("$y/c$")
        axes[0, col].set_title(f"{label}\n{g['nx']} x {g['ny']} = {g['nx'] * g['ny']} cells", fontsize=9)
    axes[1, 0].text(0.02, 0.92, "zoom near the airfoil", transform=axes[1, 0].transAxes, fontsize=8, va="top")
    fig.tight_layout()
    return fig


def plot_grid_study(results, ref_cells):
    """Log-log plots of the Cd error against the number of cells and of the wall-Cp error against CPU time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for (name, r), color in zip(results.items(), COLORS):
        ax1.loglog(r["cells"], r["cd_err"], "o-", color=color, label=name)
        ax2.loglog(r["cpu"], r["cp_err"], "o-", color=color, label=name)
    ax1.set_xlabel("number of cells")
    ax1.set_ylabel(r"$|C_d - C_{d,ref}| / C_{d,ref}$")
    ax1.set_title(f"(a) wave drag error (reference: {ref_cells} cells)", fontsize=9)
    ax2.set_xlabel("CPU time [s]")
    ax2.set_ylabel(r"mean $|C_p - C_{p,ref}|$ on the chord")
    ax2.set_title("(b) wall pressure error against cost", fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_mms_errors(results):
    """Log-log plot of the MMS errors against h: interior L2 (solid) and boundary-cell L_inf (dashed), per scheme."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (name, r), color in zip(results.items(), COLORS):
        ax.loglog(r["h"], r["L2_int"], "o-", color=color, label=f"{name}: interior $L_2$")
        ax.loglog(r["h"], r["Linf_bnd"], "s--", color=color, mfc="none", label=rf"{name}: boundary $L_\infty$")
    h = next(iter(results.values()))["h"]
    ax.loglog(h, 0.3 * h, "k:", lw=1.0, label="$O(h)$")
    ax.loglog(h, 0.3 * h**2, "k-.", lw=1.0, label="$O(h^2)$")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$|\phi - \phi_m|$")
    ax.set_title("Manufactured-solution verification")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    return fig
