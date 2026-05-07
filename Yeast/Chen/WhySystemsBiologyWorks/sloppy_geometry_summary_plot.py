"""Compact summary of Tyson sloppy-subspace and projection diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def finite_offdiag(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    vals = matrix[mask]
    return vals[np.isfinite(vals)]


def bxp_stats(values: np.ndarray, label: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "label": label,
        "whislo": float(np.quantile(values, 0.05)),
        "q1": float(np.quantile(values, 0.25)),
        "med": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "whishi": float(np.quantile(values, 0.95)),
        "fliers": [],
    }


def draw_boxes(ax, series: list[tuple[str, np.ndarray]], ylabel: str, title: str) -> None:
    stats = [bxp_stats(values, label) for label, values in series]
    ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.55)
    for patch in ax.patches:
        patch.set_facecolor("#d8e8f7")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for idx, (_, values) in enumerate(series, start=1):
        values = np.asarray(values, dtype=float)
        ax.scatter(idx, np.nanmedian(values), color="black", s=16, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)


def main() -> None:
    loose_subspace = np.load(ROOT / "results/sloppy_subspace/tyson_init_thr1_sloppy_all_k3.npz", allow_pickle=True)
    loose_projection = np.load(ROOT / "results/sloppy_projection/tyson_init_thr1_projection_k3.npz", allow_pickle=True)
    strict_projection = np.load(ROOT / "results/ellipsoid_check/tyson_init_1e6_thr005_ellipsoid_k3.npz", allow_pickle=True)

    loose_pairwise = finite_offdiag(loose_subspace["pairwise_chordal"])
    loose_wt = np.asarray(loose_subspace["wt_chordal"], dtype=float)
    loose_explained = np.asarray(loose_projection["explained_fraction"], dtype=float)
    loose_residual_norm = np.asarray(loose_projection["residual_fraction"], dtype=float)
    strict_explained = np.asarray(strict_projection["explained_fraction"], dtype=float)
    strict_residual_norm = np.asarray(strict_projection["residual_fraction"], dtype=float)
    strict_pca = np.asarray(strict_projection["pca_explained_fraction"], dtype=float)
    loose_pca = np.asarray(loose_projection["pca_explained_fraction"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)

    draw_boxes(
        axes[0, 0],
        [
            ("between\npoints", loose_pairwise),
            ("WT vs\npoint", loose_wt),
        ],
        "chordal distance between 3D sloppy subspaces",
        "Tyson local sloppy directions are stable\nf <= 1, n=948",
    )

    draw_boxes(
        axes[0, 1],
        [
            ("explained\nsquared", loose_explained),
            ("residual\nnorm", loose_residual_norm),
            ("explained\nsquared", strict_explained),
            ("residual\nnorm", strict_residual_norm),
        ],
        "fraction",
        "Projection onto WT 3D sloppy subspace",
    )
    axes[0, 1].set_xticklabels(
        ["f<=1\nexpl.", "f<=1\nresid.", "f<=0.05\nexpl.", "f<=0.05\nresid."]
    )
    axes[0, 1].text(
        0.02,
        0.02,
        "explained = ||parallel||² / ||total||²\nresidual = ||orthogonal|| / ||total||",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    x = np.arange(1, len(loose_pca) + 1)
    axes[1, 0].plot(x, np.cumsum(loose_pca), marker="o", label="f <= 1, n=948")
    axes[1, 0].plot(x, np.cumsum(strict_pca), marker="o", label="f <= 0.05, n=52")
    axes[1, 0].axhline(0.8, color="0.7", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.9, color="0.7", linestyle=":", linewidth=1)
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_xlabel("PCA dimension")
    axes[1, 0].set_ylabel("cumulative explained variance")
    axes[1, 0].set_title("Cloud low-dimensionality")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0,
        1,
        "Takeaway\n\n"
        "The very stable chordal distances show that the local Hessian\n"
        "sloppy subspace hardly rotates across Tyson f <= 1 points.\n\n"
        "But the projection boxes show that the point cloud is not fully\n"
        "contained in the WT 3D sloppy plane. The stricter f <= 0.05\n"
        "sample is more low-dimensional, but it had only n=52 points.\n\n"
        "So the safe statement is: stable local sloppy directions,\n"
        "partial WT-sloppy alignment, not proven global ellipsoid.",
        ha="left",
        va="top",
        fontsize=10,
    )

    out = ROOT / "figures/sloppy_geometry/tyson_sloppy_projection_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
