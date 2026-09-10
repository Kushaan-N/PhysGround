"""Figures (spec 15).

Every function takes already-computed arrays and returns a saved path. Nothing
here recomputes a statistic: summary numbers come from ``stats.py`` operating on
the raw ``.npz`` files each experiment wrote (spec 0.4), so a figure can always
be regenerated without re-running a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "contact_sheet",
    "decorrelation_heatmap",
    "main_result_figure",
    "occlusion_figure",
    "frame_vs_video_figure",
]

_DPI = 140


def _mpl():
    """Import matplotlib with a headless backend.

    Selected before ``pyplot`` is imported: on a compute node with no display,
    the default interactive backend fails at import time rather than at draw
    time, which turns a figure call into a job crash.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# Gate G2
# --------------------------------------------------------------------------- #

def contact_sheet(frames: Sequence[np.ndarray], captions: Sequence[str], path: Path,
                  ncols: int = 5, title: str | None = None) -> Path:
    """Tile frames with their labels printed on them (spec 9.2).

    This is the one gate a human has to execute. It exists to catch what no
    automated check thinks to look for -- objects intersecting the floor, the arm
    out of frame, the occluder in the wrong place, a degenerate camera angle --
    so the captions carry exactly the fields needed to tell whether a frame's
    label matches what the frame shows.
    """
    plt = _mpl()
    n = len(frames)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.85 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, frame, caption in zip(axes, frames, captions):
        ax.imshow(frame)
        ax.set_title(caption, fontsize=6.5, family="monospace", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------- #
# Gate G1 / appendix figure
# --------------------------------------------------------------------------- #

def decorrelation_heatmap(report: dict, path: Path) -> Path:
    """Spearman matrix over all factor pairs (spec 15, figure 1).

    Appendix material, but load-bearing for trust: it is the first thing a
    sceptical reviewer looks for, because every claim in the paper depends on
    physical factors being unpredictable from appearance.
    """
    plt = _mpl()
    rho = np.asarray(report["rho_matrix"])
    names = list(report["names"])
    groups = list(report["groups"])

    size = max(6.0, 0.28 * len(names))
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    image = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)

    # Group boundaries, so "physical vs appearance" is readable at a glance
    # rather than requiring the reader to parse variable names.
    boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="k", lw=0.9)
        ax.axvline(b - 0.5, color="k", lw=0.9)

    fig.colorbar(image, ax=ax, fraction=0.046, label="Spearman rho")
    ax.set_title(
        f"Factor decorrelation, n={report['n_scenes']} scenes\n"
        f"max |rho| over cross-group pairs = {report['max_abs_rho']:.3f} "
        f"(threshold {report['threshold']}), "
        f"{report['n_significant_after_holm']} significant after Holm",
        fontsize=9)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------- #
# Result figures
# --------------------------------------------------------------------------- #

def _target_panels(rows: list[dict]) -> list[str]:
    order, seen = [], set()
    for row in rows:
        if row["target"] not in seen:
            seen.add(row["target"])
            order.append(row["target"])
    return order


def main_result_figure(rows: list[dict], path: Path, *, view: str = "cls",
                       encoder: str = "dinov2_b", baseline_encoder: str = "random_b",
                       pixel_encoder: str = "raw_pixel") -> Path:
    """Figure 2: metric by property and layer, with both baselines overlaid.

    ``rows`` are summary records with keys ``target``, ``encoder``, ``layer``,
    ``view``, ``metric``, ``value``, ``ci_low``, ``ci_high``. Baselines are drawn
    as horizontal bands rather than as another line, because they do not vary
    with layer and drawing them as lines invites reading a slope into noise.
    """
    plt = _mpl()
    targets = _target_panels([r for r in rows if r["encoder"] == encoder])
    ncols = min(4, max(len(targets), 1))
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, target in zip(axes, targets):
        main = sorted([r for r in rows
                       if r["target"] == target and r["encoder"] == encoder and r["view"] == view],
                      key=lambda r: r["layer"])
        if main:
            layers = [r["layer"] for r in main]
            values = [r["value"] for r in main]
            lo = [r["value"] - r["ci_low"] for r in main]
            hi = [r["ci_high"] - r["value"] for r in main]
            ax.errorbar(layers, values, yerr=[lo, hi], marker="o", capsize=3,
                        lw=1.6, label=encoder)

        for name, colour, style in ((baseline_encoder, "tab:orange", "--"),
                                    (pixel_encoder, "tab:green", ":")):
            band = [r for r in rows if r["target"] == target and r["encoder"] == name]
            if not band:
                continue
            best = max(band, key=lambda r: r["value"])
            ax.axhline(best["value"], color=colour, ls=style, lw=1.3, label=name)
            ax.axhspan(best["ci_low"], best["ci_high"], color=colour, alpha=0.13, lw=0)

        chance = _chance_level(main[0]["metric"] if main else "r2")
        if chance is not None:
            ax.axhline(chance, color="grey", lw=0.9, alpha=0.7)

        ax.set_title(target, fontsize=9)
        ax.set_xlabel("block")
        ax.set_ylabel(main[0]["metric"] if main else "")
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes[len(targets):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Linearly decodable physical state by layer  (view={view})", fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _chance_level(metric: str) -> float | None:
    if metric in ("r2",):
        return 0.0
    if metric in ("balanced_accuracy", "auroc"):
        return 0.5
    return None


def occlusion_figure(bins: np.ndarray, series: dict[str, dict], path: Path) -> Path:
    """Figure 4: accuracy against occlusion fraction, contact versus position.

    The contrast is the figure. Contact degrading on its own says only that
    occlusion makes vision harder; contact degrading *while object position holds
    up* is what supports H4, so both are drawn on shared axes.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    centres = 0.5 * (np.asarray(bins)[:-1] + np.asarray(bins)[1:])

    for label, data in series.items():
        values = np.asarray(data["value"], dtype=float)
        ax.plot(centres, values, marker="o", lw=1.8, label=label)
        if "ci_low" in data and "ci_high" in data:
            ax.fill_between(centres, np.asarray(data["ci_low"], dtype=float),
                            np.asarray(data["ci_high"], dtype=float), alpha=0.15, lw=0)

    ax.axhline(1.0, color="grey", lw=0.9, alpha=0.7)
    ax.axhline(0.0, color="grey", lw=0.9, alpha=0.4, ls="--")
    ax.set_ylim(-0.15, 1.25)
    ax.set_xlabel("occlusion fraction of the target")
    ax.set_ylabel("retained skill vs. the matched unoccluded frame")
    ax.set_title("Contact state degrades under occlusion; object position does not (H4)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def frame_vs_video_figure(rows: list[dict], path: Path,
                          targets: Sequence[str] = ("log_mass", "friction_slide")) -> Path:
    """Figure 3: single-frame versus video encoders on the dynamic properties (H3)."""
    plt = _mpl()
    encoders = sorted({r["encoder"] for r in rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.6), squeeze=False)

    for ax, target in zip(axes.ravel(), targets):
        values, errs, labels = [], [[], []], []
        for encoder in encoders:
            subset = [r for r in rows if r["target"] == target and r["encoder"] == encoder]
            if not subset:
                continue
            best = max(subset, key=lambda r: r["value"])
            values.append(best["value"])
            errs[0].append(best["value"] - best["ci_low"])
            errs[1].append(best["ci_high"] - best["value"])
            labels.append(encoder)
        positions = np.arange(len(values))
        ax.bar(positions, values, yerr=errs, capsize=4,
               color=["tab:blue" if "dino" in l or "random" in l else "tab:purple" for l in labels])
        ax.axhline(0.0, color="grey", lw=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(target, fontsize=9)
        ax.set_ylabel("R^2 (best layer/view)")
        ax.grid(alpha=0.25, lw=0.5, axis="y")

    fig.suptitle("Video encoders recover dynamic properties a single frame cannot (H3)",
                 fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
