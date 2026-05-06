"""
fusion_v2_dual_kernel.py
Optical-prior guided AFM planned sampling with dual-kernel GP reconstruction.

Stage 1: Optical prior segmentation (gradient-based ROI detection)
Stage 2: Planned AFM sampling (base grid + edge multi-ring path + interior)
Stage 3: Dual-kernel GP reconstruction (RBF for base, Matern-0.5 for edge)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation, label, map_coordinates
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── 1. Surface ────────────────────────────────────────────────────────────────

def make_surface(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)

    base = 0.02 * (X - 5)**2 + 0.015 * (Y - 5)**2

    # Mesa (凸台): steep tanh edge, edge_w=0.08 gives ~0.3 unit transition
    r_mesa = np.hypot(X - 3.0, Y - 5.0)
    mesa = 1.5 * 0.5 * (1 - np.tanh((r_mesa - 1.2) / 0.08))

    # Pit (凹坑)
    r_pit = np.hypot(X - 7.0, Y - 5.0)
    pit = -1.2 * 0.5 * (1 - np.tanh((r_pit - 1.0) / 0.08))

    Z_true = base + mesa + pit

    # Optical: diffraction-limit blur + sensor noise
    Z_opt = gaussian_filter(Z_true, sigma=4.5)
    Z_opt += rng.normal(0, 0.015, Z_true.shape)

    return x, y, X, Y, Z_true, Z_opt


# ── 2. Segmentation ───────────────────────────────────────────────────────────

def segment_optical(x, y, X, Y, Z_opt):
    dx, dy = x[1] - x[0], y[1] - y[0]

    Z_base_freq = gaussian_filter(Z_opt, sigma=10.0)
    Z_residual = Z_opt - Z_base_freq

    gy, gx = np.gradient(Z_opt, dy, dx)
    grad = np.hypot(gx, gy)

    # Defect seed from large residuals
    thresh = np.percentile(np.abs(Z_residual), 80)
    seed = np.abs(Z_residual) > thresh
    defect_mask = binary_dilation(seed, iterations=3)

    # Keep top-3 components by area
    lbl, n = label(defect_mask)
    sizes = sorted([(i, int(np.sum(lbl == i))) for i in range(1, n + 1)],
                   key=lambda t: t[1], reverse=True)
    clean = np.zeros_like(defect_mask)
    for cid, sz in sizes[:3]:
        if sz > 50:
            clean |= (lbl == cid)
    defect_mask = binary_dilation(clean, iterations=2)

    # Edge mask: high gradient inside dilated defect region
    roi_grad = grad[defect_mask]
    grad_thresh = np.percentile(roi_grad, 60) if len(roi_grad) else np.percentile(grad, 85)
    edge_mask = (grad > grad_thresh) & binary_dilation(defect_mask, iterations=3)
    edge_mask = binary_dilation(edge_mask, iterations=2)

    base_mask = ~binary_dilation(defect_mask, iterations=3)

    # Component descriptors
    lbl2, n2 = label(defect_mask)
    components = []
    for cid in range(1, n2 + 1):
        yy, xx = np.where(lbl2 == cid)
        if len(xx) < 30:
            continue
        w = np.abs(Z_residual[yy, xx])
        if w.sum() < 1e-12:
            w = np.ones_like(w)
        cx = float(np.sum(X[yy, xx] * w) / w.sum())
        cy = float(np.sum(Y[yy, xx] * w) / w.sum())

        comp_edge = (lbl2 == cid) & edge_mask
        ey, ex = np.where(comp_edge)
        if len(ex) >= 8:
            radius = float(np.median(np.hypot(X[ey, ex] - cx, Y[ey, ex] - cy)))
        else:
            radius = float(np.sqrt(len(xx) * dx * dy / np.pi) * 0.7)

        components.append({
            'cx': cx, 'cy': cy,
            'radius': max(radius, 3 * dx),
            'sign': float(np.sign(np.mean(Z_residual[yy, xx])))
        })

    return defect_mask, edge_mask, base_mask, components


# ── 3. Planned sampling ───────────────────────────────────────────────────────

def _snap_unique(x, y, xy, kinds):
    if len(xy) == 0:
        return xy.reshape(0, 2), kinds
    dx, dy = x[1] - x[0], y[1] - y[0]
    ix = np.clip(np.rint((xy[:, 0] - x[0]) / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.rint((xy[:, 1] - y[0]) / dy).astype(int), 0, len(y) - 1)
    seen, keep = set(), []
    for i, key in enumerate((iy * len(x) + ix).tolist()):
        if key not in seen:
            keep.append(i)
            seen.add(key)
    return np.column_stack([x[ix[keep]], y[iy[keep]]]), kinds[keep]


def plan_samples(x, y, base_mask, components,
                 base_stride=10, n_rings=6, n_angles=36):
    """
    base_stride : grid spacing for sparse base samples
    n_rings     : number of concentric rings per component (covers edge zone)
    n_angles    : angular resolution per ring
    """
    # Base: sparse grid
    base_pts = [
        [x[ix], y[iy]]
        for iy in range(2, len(y) - 2, base_stride)
        for ix in range(2, len(x) - 2, base_stride)
        if base_mask[iy, ix]
    ]

    edge_pts, interior_pts, path_pts = [], [], []
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for comp in components:
        cx, cy, r = comp['cx'], comp['cy'], comp['radius']

        # Interior: center + two sparse rings
        interior_pts.append([cx, cy])
        for r_frac in (0.35, 0.65):
            for a in theta[::4]:
                interior_pts.append([cx + r_frac * r * np.cos(a),
                                     cy + r_frac * r * np.sin(a)])

        # Edge: n_rings concentric rings spanning [-0.55r, +0.55r] around edge
        offsets = np.linspace(-0.55 * r, 0.55 * r, n_rings)
        for k, offset in enumerate(offsets):
            ring_r = max(r + offset, x[1] - x[0])
            # Alternate CW/CCW to shorten probe travel between rings
            angles = theta if k % 2 == 0 else theta[::-1]
            ring = [[cx + ring_r * np.cos(a), cy + ring_r * np.sin(a)] for a in angles]
            path_pts.extend(ring)
            edge_pts.extend(ring)

    all_pts = base_pts + interior_pts + edge_pts
    all_kinds = (['base'] * len(base_pts) +
                 ['interior'] * len(interior_pts) +
                 ['edge'] * len(edge_pts))

    xy = np.array(all_pts, dtype=float)
    kinds = np.array(all_kinds)
    xy, kinds = _snap_unique(x, y, xy, kinds)

    path_xy = np.array(path_pts, dtype=float)
    if len(path_xy):
        path_xy[:, 0] = np.clip(path_xy[:, 0], x[0], x[-1])
        path_xy[:, 1] = np.clip(path_xy[:, 1], y[0], y[-1])

    path_len = float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))) if len(path_xy) > 1 else 0.0
    return xy, kinds, path_xy, path_len


# ── 4. GP reconstruction ──────────────────────────────────────────────────────

def _sample(x, y, Z, xy):
    dx, dy = x[1] - x[0], y[1] - y[0]
    col = (xy[:, 0] - x[0]) / dx
    row = (xy[:, 1] - y[0]) / dy
    return map_coordinates(Z, [row, col], order=1, mode='nearest')


def _fit_gp(X_tr, y_tr, X_pr, kernel, alpha):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                  normalize_y=False, optimizer=None)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_pr)


def reconstruct_dual(x, y, X, Y, Z_true, Z_opt, defect_mask, xy, kinds):
    """
    Correct Z_opt residual with two GPs:
      base region  → RBF (long length-scale, smooth)
      defect region → Matern-0.5 (short length-scale, handles steep edges)
    Blend spatially using optical-prior defect weight.
    """
    X_pred = np.column_stack([X.ravel(), Y.ravel()])
    residual = _sample(x, y, Z_true, xy) - _sample(x, y, Z_opt, xy)

    is_base = kinds == 'base'
    is_local = ~is_base
    if is_base.sum() < 5:
        is_base = kinds != 'edge'
    if is_local.sum() < 5:
        is_local = ~is_base

    k_base = (C(1.0, 'fixed') * RBF(length_scale=2.5, length_scale_bounds='fixed')
              + WhiteKernel(0.02, 'fixed'))
    # Matern nu=0.5 is the exponential kernel — C^0 continuity, ideal for step edges
    k_edge = (C(1.0, 'fixed') * Matern(length_scale=0.28, nu=0.5, length_scale_bounds='fixed')
              + WhiteKernel(1e-3, 'fixed'))

    corr_base = _fit_gp(xy[is_base], residual[is_base], X_pred, k_base, alpha=0.02)
    corr_edge = _fit_gp(xy[is_local], residual[is_local], X_pred, k_edge, alpha=1e-3)

    # Smooth blending weight from optical defect mask
    w = gaussian_filter(defect_mask.astype(float), sigma=2.5)
    w = np.clip(w / (w.max() + 1e-12), 0, 1).ravel()

    correction = corr_base * (1 - 0.85 * w) + corr_edge * w
    return Z_opt + correction.reshape(Z_true.shape)


def reconstruct_single_rbf(x, y, X, Y, Z_true, Z_opt, xy):
    """Baseline: single RBF GP on random samples, no optical prior."""
    X_pred = np.column_stack([X.ravel(), Y.ravel()])
    residual = _sample(x, y, Z_true, xy) - _sample(x, y, Z_opt, xy)
    k = (C(1.0, 'fixed') * RBF(length_scale=0.9, length_scale_bounds='fixed')
         + WhiteKernel(2e-5, 'fixed'))
    corr = _fit_gp(xy, residual, X_pred, k, alpha=1e-4)
    return Z_opt + corr.reshape(Z_true.shape)


# ── 5. Metrics ────────────────────────────────────────────────────────────────

def _true_edge_mask(x, y, Z_true):
    gy, gx = np.gradient(Z_true, y[1] - y[0], x[1] - x[0])
    grad = np.hypot(gx, gy)
    return binary_dilation(grad > np.percentile(grad, 88), iterations=2)


def rmse(a, b, mask=None):
    d = (a - b)[mask] if mask is not None else (a - b)
    return float(np.sqrt(np.mean(d**2)))


def mae(a, b, mask=None):
    d = np.abs(a - b)[mask] if mask is not None else np.abs(a - b)
    return float(np.mean(d))


# ── 6. Visualisation ──────────────────────────────────────────────────────────

def plot_all(x, y, X, Y, Z_true, Z_opt, defect_mask, edge_mask,
             xy, kinds, path_xy,
             random_xy, Z_rand, Z_dual):

    extent = [x[0], x[-1], y[0], y[-1]]
    edge_truth = _true_edge_mask(x, y, Z_true)
    vmin, vmax = np.percentile(Z_true, [0.5, 99.5])
    err_vmax = np.percentile(np.abs(Z_opt - Z_true), 98)

    err_opt  = np.abs(Z_opt  - Z_true)
    err_rand = np.abs(Z_rand - Z_true)
    err_dual = np.abs(Z_dual - Z_true)

    # ── Figure 1: 2-row overview ──────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    fig1.suptitle("Optical-prior + Planned Dual-Kernel AFM Reconstruction", fontsize=13)

    specs = [
        (axes[0, 0], Z_true,  "Ground truth (AFM)",        "viridis", vmin, vmax),
        (axes[0, 1], Z_opt,   "Optical prior (blurred)",   "viridis", vmin, vmax),
        (axes[0, 2], Z_dual,  "Planned dual-kernel GP",    "viridis", vmin, vmax),
        (axes[1, 0], err_opt,  "Error: optical only",      "inferno", 0, err_vmax),
        (axes[1, 1], err_rand, "Error: random + RBF",      "inferno", 0, err_vmax),
        (axes[1, 2], err_dual, "Error: planned dual GP",   "inferno", 0, err_vmax),
    ]
    for ax, Z, title, cmap, lo, hi in specs:
        im = ax.imshow(Z, origin='lower', extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Gradient feature map
    gy, gx = np.gradient(Z_opt, y[1]-y[0], x[1]-x[0])
    grad = np.hypot(gx, gy)
    im = axes[0, 3].imshow(grad, origin='lower', extent=extent, cmap='magma')
    axes[0, 3].set_title("Optical gradient (feature map)", fontsize=10)
    axes[0, 3].set_xlabel("x"); axes[0, 3].set_ylabel("y")
    fig1.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.03)

    # Sampling plan overlay
    ax = axes[1, 3]
    ax.imshow(Z_opt, origin='lower', extent=extent, cmap='gray')
    ax.contour(X, Y, defect_mask.astype(float), levels=[0.5], colors='cyan',   linewidths=1.2)
    ax.contour(X, Y, edge_mask.astype(float),   levels=[0.5], colors='yellow', linewidths=1.0)
    ax.scatter(random_xy[:, 0], random_xy[:, 1], s=6,  c='white',   alpha=0.25, label='random')
    ax.scatter(xy[kinds=='base',     0], xy[kinds=='base',     1], s=18, c='#3ddc97', label='base')
    ax.scatter(xy[kinds=='interior', 0], xy[kinds=='interior', 1], s=22, c='#2f80ed', label='interior')
    ax.scatter(xy[kinds=='edge',     0], xy[kinds=='edge',     1], s=12, c='#ff4d4d', label='edge')
    if len(path_xy):
        ax.plot(path_xy[:, 0], path_xy[:, 1], color='#ff4d4d', lw=0.9, alpha=0.7)
    ax.set_title("ROI + planned AFM path", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc='upper right', fontsize=7, frameon=True)

    # ── Figure 2: line profiles ───────────────────────────────────────────────
    profile_y = 5.0
    row = int(np.argmin(np.abs(y - profile_y)))
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    fig2.suptitle(f"Line profiles at y = {y[row]:.2f}", fontsize=12)

    for ax2, (lo2, hi2, title2) in zip(axes2, [
        (None, None, "Full profile"),
        (1.5, 4.5,   "Zoom: mesa edge region"),
    ]):
        ax2.plot(x, Z_true[row],  'k-',  lw=2.2, label='True AFM')
        ax2.plot(x, Z_opt[row],   color='#888', lw=2.0, label='Optical prior')
        ax2.plot(x, Z_rand[row],  color='#8e44ad', lw=1.8, label='Random + single RBF')
        ax2.plot(x, Z_dual[row],  color='#d62728', lw=2.2, label='Planned dual GP')
        ax2.set_title(title2); ax2.set_xlabel("x"); ax2.set_ylabel("height")
        ax2.grid(alpha=0.25); ax2.legend(fontsize=9)
        if lo2 is not None:
            ax2.set_xlim(lo2, hi2)

    # ── Figure 3: 3-D surfaces ────────────────────────────────────────────────
    fig3 = plt.figure(figsize=(14, 4.5), constrained_layout=True)
    fig3.suptitle("3-D surface comparison", fontsize=12)
    skip = 2
    for idx, (Z3, title3) in enumerate([(Z_true, "True AFM"),
                                         (Z_opt,  "Optical prior"),
                                         (Z_dual, "Planned dual GP")], 1):
        ax3 = fig3.add_subplot(1, 3, idx, projection='3d')
        ax3.plot_surface(X[::skip, ::skip], Y[::skip, ::skip], Z3[::skip, ::skip],
                         cmap='viridis', linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)
        ax3.set_title(title3, fontsize=10)
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
        ax3.set_zlim(float(np.percentile(Z_true, 0.5)), float(np.percentile(Z_true, 99.5)))
        ax3.view_init(elev=32, azim=-55)

    return fig1, fig2, fig3


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    # Build surface
    x, y, X, Y, Z_true, Z_opt = make_surface(n=100, seed=42)

    # Optical segmentation
    defect_mask, edge_mask, base_mask, components = segment_optical(x, y, X, Y, Z_opt)

    # Planned AFM samples
    xy, kinds, path_xy, path_len = plan_samples(
        x, y, base_mask, components,
        base_stride=10, n_rings=6, n_angles=36
    )

    # Random baseline (same budget)
    rng = np.random.default_rng(99)
    random_xy = np.column_stack([
        rng.uniform(x[0], x[-1], len(xy)),
        rng.uniform(y[0], y[-1], len(xy)),
    ])
    random_xy, _ = _snap_unique(x, y, random_xy, np.full(len(xy), 'r'))

    # Reconstruct
    Z_rand = reconstruct_single_rbf(x, y, X, Y, Z_true, Z_opt, random_xy)
    Z_dual = reconstruct_dual(x, y, X, Y, Z_true, Z_opt, defect_mask, xy, kinds)

    # Metrics
    edge_truth = _true_edge_mask(x, y, Z_true)
    print("\n=== Fusion v2: optical-prior + planned dual-kernel GP ===")
    print(f"Detected components : {len(components)}")
    print(f"AFM budget          : planned={len(xy)}  random={len(random_xy)}")
    print(f"  base={np.sum(kinds=='base')}  interior={np.sum(kinds=='interior')}  edge={np.sum(kinds=='edge')}")
    print(f"Edge path length    : {path_len:.2f} units")
    print(f"\n{'Method':<30} {'RMSE(all)':>10} {'RMSE(edge)':>11} {'MAE(edge)':>10}")
    print("-" * 65)
    for name, Z in [("Optical only",            Z_opt),
                    ("Random + single RBF",      Z_rand),
                    ("Planned dual-kernel GP",   Z_dual)]:
        print(f"{name:<30} {rmse(Z, Z_true):>10.4f} {rmse(Z, Z_true, edge_truth):>11.4f} {mae(Z, Z_true, edge_truth):>10.4f}")
    print()

    plot_all(x, y, X, Y, Z_true, Z_opt, defect_mask, edge_mask,
             xy, kinds, path_xy, random_xy, Z_rand, Z_dual)
    plt.show()


if __name__ == "__main__":
    main()
