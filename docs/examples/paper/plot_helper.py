import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

def plot_res(wavelengths, fob_init, fob_final, sum_init, sum_final,
             va, positions, radii, posf, radf, wl_at, label, fname, axs=None):

    if axs is None:
        fig, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(20, 17))
    else:
        ((ax1, ax2), (ax5, ax6)) = axs
        fig = ax1.figure  

    ax1.scatter(range(len(va)), va)
    ax1.set_yscale("log")
    ax1.set_ylabel(label)
    ax1.set_xlabel("Steps")

    plot_3d_with_depth(positions, radii, ax=ax2, color="red", label="Initial")
    plot_3d_with_depth(posf, radf, ax=ax2, color="blue", label="Optimized")

    ax5.plot(wavelengths, fob_init,  color="r", label="F/B init")
    ax5.plot(wavelengths, fob_final, color="b", label="F/B final")
    ax5.set_ylabel("F/B")
    ax5.set_xlabel(r"$\lambda$ (nm)")
    ax5.axvline(x=wl_at, linestyle="--")
    ax5.legend()

    ax6.plot(wavelengths, sum_init,  color="r", label=r"$\sigma_{\text{sca}}^{\text{init}}$")
    ax6.plot(wavelengths, sum_final, color="b", label=r"$\sigma_{\text{sca}}^{\text{final}}$")
    ax6.set_ylabel(r"$\sigma_{\text{sca}}$ (nm$^2$)")
    ax6.set_xlabel(r"$\lambda$ (nm)")
    ax6.axvline(x=wl_at, linestyle="--")
    ax6.legend()

    fig.suptitle(fname)
    return fig, ((ax1, ax2), (ax5, ax6))

def set_axes_not_equal(ax):
    """Enforces equal aspect ratio for 3D plots."""
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(-200, 200)

    # 2. compute the actual spans of your data
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)

    # 3. tell Matplotlib: “make my box shape = (sx : sy : sz)”
    ax.set_box_aspect((sx, sy, sz))

def set_axes_equal(ax):
    """Enforce equal aspect for 3D axes."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    mean_x = np.mean(x_limits)
    mean_y = np.mean(y_limits)
    mean_z = np.mean(z_limits)

    ax.set_xlim(mean_x - max_range / 2, mean_x + max_range / 2)
    ax.set_ylim(mean_y - max_range / 2, mean_y + max_range / 2)
    ax.set_zlim(mean_z - max_range / 2, mean_z + max_range / 2)

def plot_3d_with_depth(centers, radii, depth_factor=1.0, ax=None,
                       color="red", label=None, norm=None):
    """
    Plot 3D spheres with opacity increasing with height (z),
    and a semi-transparent z=0 plane.

    Parameters
    ----------
    centers : (N, 3) array-like
        Sphere centers [x, y, z].
    radii : (N,) array-like
        Sphere radii.
    depth_factor : float
        Currently unused (kept for API compatibility).
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axis. If None, a new figure/axis is created.
    color : str or color
        Surface color of the spheres.
    label : str, optional
        Kept for API compatibility (not used inside).
    norm : float, optional
       
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Axis labels and ticks
    ax.set_xlabel("x (nm)", labelpad=4)
    ax.set_ylabel("y (nm)", labelpad=4)
    ax.set_zlabel("z (nm)", labelpad=4)

    ax.xaxis.set_tick_params(pad=1)
    ax.yaxis.set_tick_params(pad=1)
    ax.zaxis.set_tick_params(pad=1)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    # z = 0 plane (same fixed extent as before)
    xx, yy = np.meshgrid(
        np.linspace(-400, 400, 10),
        np.linspace(-400, 400, 10),
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color="gray", alpha=0.3,
                    linewidth=0, zorder=0)

    # Opacity scaling along z
    if norm is None:
        # Small, safe default; if you always pass norm, this never triggers.
        if centers.size == 0:
            norm = 1.0
        else:
            norm = float(np.max(centers[:, 2]) or 1.0)

    # Precompute sphere mesh (unit sphere) once
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    sin_v = np.sin(v)
    cos_v = np.cos(v)

    # Build spheres
    for i, center in enumerate(centers):
        cx, cy, cz = center
        r = radii[i]

        x = r * np.outer(cos_u, sin_v) + cx
        y = r * np.outer(sin_u, sin_v) + cy
        z = r * np.outer(np.ones_like(u), cos_v) + cz

        # Map z position → alpha
        alpha_value = np.interp(cz, [0.0, norm], [0.4, 0.6])

        ax.plot_surface(
            x, y, z,
            color=color,
            alpha=float(alpha_value),
            edgecolor="none",
        )

    set_axes_not_equal(ax)
    return ax

def plot_opt(pos, radii, pos_opt, radii_opt, pitch, ax=None):
    """
    3D overlay of initial (red) and optimized (blue) sphere arrays.

    Parameters
    ----------
    pos, radii      : initial centers (N,3) and radii (N,)
    pos_opt, radii_opt : optimized centers (N,3) and radii (N,)
    pitch           : not used here (kept for API consistency)
    ax              : optional existing 3D axis

    Returns
    -------
    ax : matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # shared color normalization by z (for consistency if you later color by depth)
    vm = max(np.abs(c[2]) for ps in [pos_opt, pos] for c in ps)
    _ = Normalize(vmin=-vm, vmax=vm)  # kept for future use if needed

    colors = ["red", "blue"]
    labels = ["Initial", "Optimized"]

    plot_3d_with_depth(pos,     radii,     ax=ax, color=colors[0], label=labels[0])
    plot_3d_with_depth(pos_opt, radii_opt, ax=ax, color=colors[1], label=labels[1])

    # legend via proxy patches (since surfaces don't create handles)
    legend_proxy = [Patch(color=colors[i], label=labels[i]) for i in range(2)]
    ax.legend(handles=legend_proxy)
    ax.grid()
    return ax

def plot_plane(plane_mask, vfa, vba, vf2a, vb2a, x, y, z, ttl, ax):

    a_plane = x[plane_mask] if ("XZ" in ttl) else y[plane_mask]
    z_plane = z[plane_mask]

    # Extract fields on this plane
    vf  = vfa[plane_mask]
    vb  = vba[plane_mask]
    vf2 = vf2a[plane_mask]
    vb2 = vb2a[plane_mask]

    # Angle: backward is forward + π
    theta  = np.arctan2(a_plane, z_plane)
    theta2 = (theta + np.pi) % (2 * np.pi)

    # magnitudes, strictly positive for log scale
    vf_p, vb_p  = np.abs(vf),  np.abs(vb)
    vf2_p, vb2_p = np.abs(vf2), np.abs(vb2)
    vmax = np.max([vf_p.max(initial=0), vb_p.max(initial=0),
                   vf2_p.max(initial=0), vb2_p.max(initial=0)])
    eps = max(1e-15, 1e-12 * vmax) if vmax > 0 else 1e-15
    vf_p  = np.clip(vf_p,  eps, None)
    vb_p  = np.clip(vb_p,  eps, None)
    vf2_p = np.clip(vf2_p, eps, None)
    vb2_p = np.clip(vb2_p, eps, None)

    # sort for cleaner plotting 
    order = np.argsort(theta)
    th_f, th_b = theta[order], theta2[order]
    vf_p, vb_p = vf_p[order], vb_p[order]
    vf2_p, vb2_p = vf2_p[order], vb2_p[order]

    ax.scatter(th_f, vf_p,  color="r", s=8, label="Initial")
    ax.scatter(th_b, vb_p,  color="r", s=8)
    ax.scatter(th_f, vf2_p, color="b", s=8, label="Final")
    ax.scatter(th_b, vb2_p, color="b", s=8)

    ax.set_title(ttl)
    ax.set_theta_zero_location("N")
    ax.set_yscale("log")


def plot_2d_field(vfa, vba, vf2a, vb2a, x, y, z, title=None, axs=None, fig=None):
    vfa  = np.asarray(vfa).ravel()
    vba  = np.asarray(vba).ravel()
    vf2a = np.asarray(vf2a).ravel()
    vb2a = np.asarray(vb2a).ravel()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    all_vs = np.concatenate((vfa, vba, vf2a, vb2a))
    ymin = np.nanmin(all_vs)
    ymax = np.nanmax(all_vs)

    tolerance = 0.1  

    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                       subplot_kw={'projection': 'polar'})
    else:
        ax1, ax2 = axs

    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    mask_xz = (np.abs(y) < tolerance)  # XZ plane
    mask_yz = (np.abs(x) < tolerance)  # YZ plane

    plot_plane(mask_xz, vfa, vba, vf2a, vb2a, x, y, z, "XZ plane", ax=ax1)
    plot_plane(mask_yz, vfa, vba, vf2a, vb2a, x, y, z, "YZ plane", ax=ax2)

    ax1.legend(loc='upper left', markerscale=4, bbox_to_anchor=(0.95, 0.95))

    if title is not None:
        plt.title(title)
               
def plot_multipole(xss, wls, wl, ax, title):
    wls = np.asarray(wls)
    idx = np.where(np.isclose(wls, wl))[0]
    if idx.size != 1:
        raise ValueError(f"Expected exactly one match for wl={wl}, got {idx.size}")
    idx = idx[0]

    xi_names = []
    xi_vals = []

    for key, value in xss.items():
        arr = np.asarray(value)
        xi_names.append(key)
        xi_vals.append(arr[idx])

    base_colors = ["magenta", "cyan"]
    repeat = (len(xi_names) + len(base_colors) - 1) // len(base_colors)
    colors = (base_colors * repeat)[:len(xi_names)]
    ax.bar(xi_names, xi_vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel(r"$\sigma_{sc}$ (nm$^2$)")
    ax.tick_params(axis='x', labelsize=11)