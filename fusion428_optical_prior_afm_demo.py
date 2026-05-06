from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label,
    map_coordinates,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class SurfaceData:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z_true: np.ndarray
    Z_optical: np.ndarray


@dataclass
class FeatureMasks:
    optical_base: np.ndarray
    optical_residual: np.ndarray
    gradient: np.ndarray
    defect_mask: np.ndarray
    edge_mask: np.ndarray
    base_mask: np.ndarray
    labels: np.ndarray
    components: list[dict]


@dataclass
class PlannedSamples:
    xy: np.ndarray
    kind: np.ndarray
    path_xy: np.ndarray
    path_length: float


def circular_step(
    X: np.ndarray,
    Y: np.ndarray,
    center: tuple[float, float],
    radius: float,
    height: float,
    edge_width: float,
) -> np.ndarray:
    """Smooth circular mesa/pit with a steep but finite edge."""
    r = np.hypot(X - center[0], Y - center[1])
    inside = 0.5 * (1.0 - np.tanh((r - radius) / edge_width))
    return height * inside


def generate_surface(grid_size: int = 90, seed: int = 7) -> SurfaceData:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, grid_size)
    y = np.linspace(0.0, 10.0, grid_size)
    X, Y = np.meshgrid(x, y)

    base = (
        0.018 * (X - 5.0) ** 2
        + 0.010 * (Y - 4.5) ** 2
        + 0.055 * np.sin(0.55 * X + 0.25 * Y)
        - 0.040 * np.cos(0.42 * Y)
    )
    mesa = circular_step(X, Y, center=(3.25, 5.05), radius=1.05, height=1.15, edge_width=0.075)
    pit = circular_step(X, Y, center=(7.15, 5.05), radius=0.90, height=-0.95, edge_width=0.075)
    Z_true = base + mesa + pit

    # Optical prior: the main engineering error is low-pass blur, not just noise.
    Z_optical = gaussian_filter(Z_true, sigma=4.2)
    Z_optical += rng.normal(0.0, 0.012, size=Z_true.shape)

    return SurfaceData(x=x, y=y, X=X, Y=Y, Z_true=Z_true, Z_optical=Z_optical)


def _component_summary(
    labels: np.ndarray,
    component_id: int,
    X: np.ndarray,
    Y: np.ndarray,
    residual: np.ndarray,
    edge_mask: np.ndarray,
    dx: float,
) -> dict | None:
    yy, xx = np.where(labels == component_id)
    if len(xx) < 35:
        return None

    weights = np.abs(residual[yy, xx])
    if np.sum(weights) <= 1e-12:
        weights = np.ones_like(weights)

    cx = float(np.sum(X[yy, xx] * weights) / np.sum(weights))
    cy = float(np.sum(Y[yy, xx] * weights) / np.sum(weights))

    component_edge = (labels == component_id) & edge_mask
    edge_yy, edge_xx = np.where(component_edge)
    if len(edge_xx) >= 8:
        rr = np.hypot(X[edge_yy, edge_xx] - cx, Y[edge_yy, edge_xx] - cy)
        radius = float(np.median(rr))
    else:
        area = len(xx) * dx * dx
        radius = float(np.sqrt(area / np.pi) * 0.72)

    sign = float(np.sign(np.mean(residual[yy, xx])))
    return {
        "id": component_id,
        "cx": cx,
        "cy": cy,
        "radius": max(radius, 3.0 * dx),
        "area_px": len(xx),
        "sign": sign,
    }


def optical_prior_segmentation(data: SurfaceData) -> FeatureMasks:
    dx = float(data.x[1] - data.x[0])
    dy = float(data.y[1] - data.y[0])

    optical_base = gaussian_filter(data.Z_optical, sigma=8.0)
    optical_residual = data.Z_optical - optical_base

    gy, gx = np.gradient(data.Z_optical, dy, dx)
    gradient = np.hypot(gx, gy)

    # Defect ROI: low-frequency base is removed, then large optical residuals are kept.
    defect_seed = np.abs(optical_residual) > np.percentile(np.abs(optical_residual), 82.0)
    defect_mask = binary_dilation(defect_seed, iterations=2)

    initial_labels, n_labels = label(defect_mask)
    component_sizes = [(cid, int(np.sum(initial_labels == cid))) for cid in range(1, n_labels + 1)]
    component_sizes = sorted(component_sizes, key=lambda item: item[1], reverse=True)[:3]

    clean_defect = np.zeros_like(defect_mask, dtype=bool)
    for cid, _ in component_sizes:
        clean_defect |= initial_labels == cid
    defect_mask = binary_dilation(clean_defect, iterations=1)
    labels, n_labels = label(defect_mask)

    roi_gradient = gradient[defect_mask]
    grad_threshold = np.percentile(roi_gradient, 66.0) if len(roi_gradient) else np.percentile(gradient, 88.0)
    edge_mask = (gradient > grad_threshold) & binary_dilation(defect_mask, iterations=2)
    edge_mask = binary_dilation(edge_mask, iterations=2)

    labels, n_labels = label(defect_mask)
    components: list[dict] = []
    for cid in range(1, n_labels + 1):
        summary = _component_summary(labels, cid, data.X, data.Y, optical_residual, edge_mask, dx)
        if summary is not None:
            components.append(summary)

    base_mask = ~(binary_dilation(defect_mask, iterations=2) | edge_mask)
    return FeatureMasks(
        optical_base=optical_base,
        optical_residual=optical_residual,
        gradient=gradient,
        defect_mask=defect_mask,
        edge_mask=edge_mask,
        base_mask=base_mask,
        labels=labels,
        components=components,
    )


def _snap_unique_points(data: SurfaceData, xy: np.ndarray, kind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xy) == 0:
        return xy.reshape(0, 2), kind

    dx = float(data.x[1] - data.x[0])
    dy = float(data.y[1] - data.y[0])
    ix = np.rint((xy[:, 0] - data.x[0]) / dx).astype(int)
    iy = np.rint((xy[:, 1] - data.y[0]) / dy).astype(int)
    ix = np.clip(ix, 0, len(data.x) - 1)
    iy = np.clip(iy, 0, len(data.y) - 1)

    seen: set[int] = set()
    keep: list[int] = []
    for i, key in enumerate((iy * len(data.x) + ix).tolist()):
        if key not in seen:
            keep.append(i)
            seen.add(key)

    snapped = np.column_stack([data.x[ix[keep]], data.y[iy[keep]]])
    return snapped, kind[keep]


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def plan_afm_samples(
    data: SurfaceData,
    masks: FeatureMasks,
    base_stride: int = 12,
    edge_angles: int = 24,
) -> PlannedSamples:
    base_points: list[list[float]] = []
    for iy in range(3, len(data.y) - 3, base_stride):
        for ix in range(3, len(data.x) - 3, base_stride):
            if masks.base_mask[iy, ix]:
                base_points.append([data.x[ix], data.y[iy]])

    interior_points: list[list[float]] = []
    edge_path: list[list[float]] = []
    theta = np.linspace(0.0, 2.0 * np.pi, edge_angles, endpoint=False)
    edge_offsets = np.asarray([-0.45, -0.30, -0.16, 0.0, 0.16, 0.30, 0.45])

    for comp in masks.components:
        cx = comp["cx"]
        cy = comp["cy"]
        r_edge = comp["radius"]

        interior_points.append([cx, cy])
        for r_scale in (0.25, 0.50, 0.72):
            for angle in theta[::6]:
                interior_points.append([cx + r_scale * r_edge * np.cos(angle), cy + r_scale * r_edge * np.sin(angle)])

        # Radial edge-crossing profiles are more informative than only tracing
        # one contour: each short spoke observes outside/edge/inside heights.
        for spoke_id, angle in enumerate(theta):
            local_offsets = edge_offsets if spoke_id % 2 == 0 else edge_offsets[::-1]
            for offset in local_offsets:
                radius = max(r_edge + offset, 0.0)
                edge_path.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])

    xy_parts = []
    kind_parts = []
    if base_points:
        xy_parts.append(np.asarray(base_points))
        kind_parts.append(np.full(len(base_points), "base"))
    if interior_points:
        xy_parts.append(np.asarray(interior_points))
        kind_parts.append(np.full(len(interior_points), "interior"))
    if edge_path:
        xy_parts.append(np.asarray(edge_path))
        kind_parts.append(np.full(len(edge_path), "edge"))

    xy = np.vstack(xy_parts)
    kind = np.concatenate(kind_parts)
    xy, kind = _snap_unique_points(data, xy, kind)

    path_xy = np.asarray(edge_path)
    if len(path_xy):
        path_xy = np.column_stack(
            [
                np.clip(path_xy[:, 0], data.x[0], data.x[-1]),
                np.clip(path_xy[:, 1], data.y[0], data.y[-1]),
            ]
        )

    return PlannedSamples(
        xy=xy,
        kind=kind,
        path_xy=path_xy,
        path_length=_path_length(path_xy),
    )


def sample_grid(data: SurfaceData, Z: np.ndarray, xy: np.ndarray) -> np.ndarray:
    dx = float(data.x[1] - data.x[0])
    dy = float(data.y[1] - data.y[0])
    col = (xy[:, 0] - data.x[0]) / dx
    row = (xy[:, 1] - data.y[0]) / dy
    return map_coordinates(Z, [row, col], order=1, mode="nearest")


def make_prediction_grid(data: SurfaceData) -> np.ndarray:
    return np.column_stack([data.X.ravel(), data.Y.ravel()])


def fit_gp_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    kernel,
    alpha: float = 1e-6,
    normalize_y: bool = True,
) -> np.ndarray:
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        optimizer=None,
    )
    gp.fit(X_train, y_train)
    return gp.predict(X_pred)


def reconstruct_dual_kernel(data: SurfaceData, masks: FeatureMasks, samples: PlannedSamples) -> np.ndarray:
    X_pred = make_prediction_grid(data)
    y_true = sample_grid(data, data.Z_true, samples.xy)
    y_opt = sample_grid(data, data.Z_optical, samples.xy)
    residual = y_true - y_opt

    base_train = samples.kind == "base"
    local_train = ~base_train

    if np.sum(base_train) < 8:
        base_train = samples.kind != "edge"
    if np.sum(local_train) < 10:
        local_train = ~base_train

    base_kernel = (
        C(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=2.20, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=0.03, noise_level_bounds="fixed")
    )
    local_kernel = (
        C(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=0.34, nu=1.5, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=2e-3, noise_level_bounds="fixed")
    )

    base_corr = fit_gp_predict(
        samples.xy[base_train],
        residual[base_train],
        X_pred,
        base_kernel,
        alpha=0.03,
        normalize_y=False,
    )
    local_corr = fit_gp_predict(
        samples.xy[local_train],
        residual[local_train],
        X_pred,
        local_kernel,
        alpha=2e-3,
        normalize_y=False,
    )

    # Optical ROI gates the sharp local GP, avoiding long-range GP extrapolation.
    defect_weight = gaussian_filter(masks.defect_mask.astype(float), sigma=2.0)
    defect_weight = defect_weight / (np.max(defect_weight) + 1e-12)
    defect_weight = np.clip(defect_weight, 0.0, 1.0).ravel()

    correction = base_corr * (1.0 - 0.75 * defect_weight) + local_corr * defect_weight
    return data.Z_optical + correction.reshape(data.Z_true.shape)


def random_baseline_samples(data: SurfaceData, n_samples: int, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = np.column_stack(
        [
            rng.uniform(data.x[0], data.x[-1], n_samples),
            rng.uniform(data.y[0], data.y[-1], n_samples),
        ]
    )
    kind = np.full(n_samples, "random")
    xy, _ = _snap_unique_points(data, xy, kind)
    return xy


def reconstruct_single_rbf(data: SurfaceData, xy: np.ndarray) -> np.ndarray:
    X_pred = make_prediction_grid(data)
    y_true = sample_grid(data, data.Z_true, xy)
    y_opt = sample_grid(data, data.Z_optical, xy)
    residual = y_true - y_opt

    kernel = (
        C(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=0.95, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=2e-5, noise_level_bounds="fixed")
    )
    corr = fit_gp_predict(xy, residual, X_pred, kernel)
    return data.Z_optical + corr.reshape(data.Z_true.shape)


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = a - b
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff**2)))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a - b)
    if mask is not None:
        diff = diff[mask]
    return float(np.mean(diff))


def true_edge_mask(data: SurfaceData) -> np.ndarray:
    gy, gx = np.gradient(data.Z_true, data.y[1] - data.y[0], data.x[1] - data.x[0])
    grad = np.hypot(gx, gy)
    return binary_dilation(grad > np.percentile(grad, 88.5), iterations=2)


def print_metrics(
    data: SurfaceData,
    masks: FeatureMasks,
    samples: PlannedSamples,
    Z_random: np.ndarray,
    Z_dual: np.ndarray,
    random_xy: np.ndarray,
) -> None:
    edge_truth = true_edge_mask(data)
    rows = [
        ("Optical only", data.Z_optical),
        ("Random AFM + single RBF", Z_random),
        ("Optical prior + planned dual GP", Z_dual),
    ]

    print("\n=== Fusion 4.28 optical-prior AFM demo ===")
    print(f"Detected defect components: {len(masks.components)}")
    print(f"AFM budget: planned={len(samples.xy)}, random_baseline={len(random_xy)}")
    print(
        "Planned split: "
        f"base={np.sum(samples.kind == 'base')}, "
        f"interior={np.sum(samples.kind == 'interior')}, "
        f"edge_profiles={np.sum(samples.kind == 'edge')}"
    )
    print(f"Planned local edge path length: {samples.path_length:.2f} length units")
    print("\nMethod                         RMSE(all)   RMSE(edge)   MAE(edge)")
    print("-" * 68)
    for name, Z in rows:
        print(
            f"{name:<30} "
            f"{rmse(Z, data.Z_true):>9.4f}   "
            f"{rmse(Z, data.Z_true, edge_truth):>10.4f}   "
            f"{mae(Z, data.Z_true, edge_truth):>9.4f}"
        )
    print()


def plot_results(
    data: SurfaceData,
    masks: FeatureMasks,
    samples: PlannedSamples,
    random_xy: np.ndarray,
    Z_random: np.ndarray,
    Z_dual: np.ndarray,
) -> None:
    extent = [data.x[0], data.x[-1], data.y[0], data.y[-1]]
    err_optical = np.abs(data.Z_optical - data.Z_true)
    err_random = np.abs(Z_random - data.Z_true)
    err_dual = np.abs(Z_dual - data.Z_true)
    vmax = np.percentile(data.Z_true, 99.2)
    vmin = np.percentile(data.Z_true, 0.8)
    err_vmax = np.percentile(err_optical, 98.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)
    image_specs = [
        (axes[0, 0], data.Z_true, "Ground truth AFM", "viridis", vmin, vmax),
        (axes[0, 1], data.Z_optical, "Optical prior: blurred", "viridis", vmin, vmax),
        (axes[0, 2], masks.gradient, "Optical gradient feature", "magma", None, None),
        (axes[0, 3], Z_dual, "Planned dual-kernel GP", "viridis", vmin, vmax),
        (axes[1, 0], err_optical, "Optical absolute error", "inferno", 0.0, err_vmax),
        (axes[1, 1], err_random, "Random + single RBF error", "inferno", 0.0, err_vmax),
        (axes[1, 2], err_dual, "Planned dual GP error", "inferno", 0.0, err_vmax),
    ]

    for ax, Z, title, cmap, lo, hi in image_specs:
        im = ax.imshow(Z, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 3]
    ax.imshow(data.Z_optical, origin="lower", extent=extent, cmap="gray")
    ax.contour(data.X, data.Y, masks.defect_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
    ax.contour(data.X, data.Y, masks.edge_mask.astype(float), levels=[0.5], colors="yellow", linewidths=1.1)
    ax.scatter(random_xy[:, 0], random_xy[:, 1], s=8, c="white", alpha=0.23, label="random")
    ax.scatter(samples.xy[samples.kind == "base", 0], samples.xy[samples.kind == "base", 1], s=18, c="#3ddc97", label="base")
    ax.scatter(
        samples.xy[samples.kind == "interior", 0],
        samples.xy[samples.kind == "interior", 1],
        s=22,
        c="#2f80ed",
        label="interior",
    )
    ax.scatter(samples.xy[samples.kind == "edge", 0], samples.xy[samples.kind == "edge", 1], s=14, c="#ff4d4d", label="edge")
    if len(samples.path_xy):
        ax.plot(samples.path_xy[:, 0], samples.path_xy[:, 1], color="#ff4d4d", linewidth=1.0, alpha=0.75)
    ax.set_title("Optical ROI + AFM planned path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # Line profile through the two defects.
    profile_y = 5.05
    row = int(np.argmin(np.abs(data.y - profile_y)))
    fig2, ax2 = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    ax2.plot(data.x, data.Z_true[row], "k-", linewidth=2.0, label="true AFM")
    ax2.plot(data.x, data.Z_optical[row], color="#8a8a8a", linewidth=2.0, label="optical prior")
    ax2.plot(data.x, Z_random[row], color="#8e44ad", linewidth=1.8, label="random + single RBF")
    ax2.plot(data.x, Z_dual[row], color="#d62728", linewidth=2.0, label="planned dual GP")
    ax2.set_title(f"Line profile at y = {data.y[row]:.2f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("height")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig3 = plt.figure(figsize=(13.0, 4.2), constrained_layout=True)
    surface_specs = [
        (data.Z_true, "3D true AFM"),
        (data.Z_optical, "3D optical prior"),
        (Z_dual, "3D planned dual GP"),
    ]
    z_min = float(np.percentile(data.Z_true, 0.8))
    z_max = float(np.percentile(data.Z_true, 99.2))
    skip = 2
    for index, (Z, title) in enumerate(surface_specs, start=1):
        ax3 = fig3.add_subplot(1, 3, index, projection="3d")
        ax3.plot_surface(
            data.X[::skip, ::skip],
            data.Y[::skip, ::skip],
            Z[::skip, ::skip],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            vmin=vmin,
            vmax=vmax,
        )
        ax3.set_title(title)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlim(z_min, z_max)
        ax3.view_init(elev=32, azim=-58)


def run_demo(show: bool = True) -> None:
    data = generate_surface()
    masks = optical_prior_segmentation(data)
    samples = plan_afm_samples(data, masks)
    random_xy = random_baseline_samples(data, len(samples.xy))

    Z_random = reconstruct_single_rbf(data, random_xy)
    Z_dual = reconstruct_dual_kernel(data, masks, samples)

    print_metrics(data, masks, samples, Z_random, Z_dual, random_xy)
    plot_results(data, masks, samples, random_xy, Z_random, Z_dual)

    if show:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optical-prior AFM planned sampling demo for mesa/pit reconstruction."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run the numerical demo without opening matplotlib windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(show=not args.no_show)
