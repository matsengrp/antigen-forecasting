"""Plot how genetically distant the independent origins of a mutation are.

Consumes the outputs of ``mutation_background_distances.py`` and draws a
three-panel supplementary figure for the antigen-prime revision (Reviewer 1,
comment 1a):

- Panel A: distribution of the amino-acid distance between the backgrounds of a
  mutation's independent origins (epitope vs non-epitope), against a null of
  distances between random pairs of origin backgrounds.
- Panel B: independent-origin count versus mean background distance, sized by
  progeny; near-simultaneous origins (a likely tree-inference artifact) are
  outlined so they are visible rather than hidden.
- Panel C: mean background distance versus the time separating a mutation's
  origins, showing that the more independently a mutation recurs, the more
  genetically different the backgrounds it arises in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Origins whose descendant clades are sampled within this many years of each
# other are treated as near-simultaneous: the signature of one real mutation
# split across nearby branches by phylogenetic inference rather than a genuine
# second independent origin.
NEAR_SIMULTANEOUS_YEARS = 0.05

EPITOPE_COLOR = "#B30000"
NON_EPITOPE_COLOR = "#3498db"
NULL_COLOR = "#7f7f7f"

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
}


def _label(is_epitope: bool) -> str:
    return "epitope" if is_epitope else "non-epitope"


def panel_occurrence_counts(ax: plt.Axes, mutations: pd.DataFrame) -> None:
    """Cumulative fraction of mutations with <= X independent origins, per class.

    Unlike the distance panels, this uses *all* mutations (including
    single-origin ones), so X starts at 1: the value at X=1 is the fraction of
    mutations that arise only once.
    """
    max_occ = int(mutations["n_independent_origins"].max())
    xs = np.arange(1, max_occ + 1)
    for is_epitope, color in [(True, EPITOPE_COLOR), (False, NON_EPITOPE_COLOR)]:
        subset = mutations[mutations["is_epitope"] == is_epitope]
        counts = subset["n_independent_origins"].to_numpy()
        n = len(counts)
        assert n > 0, f"no {_label(is_epitope)} mutations to plot"
        frac = np.array([np.count_nonzero(counts <= x) / n for x in xs])
        ax.step(
            xs,
            frac,
            where="post",
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=f"{_label(is_epitope)} ({n})",
        )
    ax.set_xlabel("Independent origins (X)")
    ax.set_ylabel("Cumulative fraction of mutations $\\leq$ X")
    ax.set_xlim(left=1)
    ax.set_ylim(top=1.02)
    ax.legend(loc="lower right")


def panel_distribution(
    ax: plt.Axes, recurrent: pd.DataFrame, null: pd.DataFrame
) -> None:
    """Panel A: ECDF of background distance, per site class, versus the null."""
    for is_epitope, color in [(True, EPITOPE_COLOR), (False, NON_EPITOPE_COLOR)]:
        subset = recurrent[recurrent["is_epitope"] == is_epitope]
        sns.ecdfplot(
            data=subset,
            x="mean_origin_background_distance_aa",
            ax=ax,
            color=color,
            label=f"{_label(is_epitope)} ({len(subset)})",
            linewidth=1.8,
        )
    sns.ecdfplot(
        data=null,
        x="background_distance_aa",
        ax=ax,
        color=NULL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="null: random origin pairs",
    )
    ax.set_xlabel("AA distance between origin backgrounds")
    ax.set_ylabel("Cumulative fraction of mutations")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")


def panel_origins_vs_distance(ax: plt.Axes, recurrent: pd.DataFrame) -> None:
    """Panel B: origin count versus background distance, sized by progeny."""
    near = recurrent["min_time_between_origins"] < NEAR_SIMULTANEOUS_YEARS
    for is_epitope, color in [(True, EPITOPE_COLOR), (False, NON_EPITOPE_COLOR)]:
        subset = recurrent[(recurrent["is_epitope"] == is_epitope) & ~near]
        ax.scatter(
            subset["n_independent_origins"],
            subset["mean_origin_background_distance_aa"],
            s=20 + 40 * np.log10(subset["max_progeny"].clip(lower=1) + 1),
            c=color,
            alpha=0.6,
            linewidths=0,
            label=_label(is_epitope),
        )
    near_subset = recurrent[near]
    ax.scatter(
        near_subset["n_independent_origins"],
        near_subset["mean_origin_background_distance_aa"],
        s=20 + 40 * np.log10(near_subset["max_progeny"].clip(lower=1) + 1),
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        label=f"near-simultaneous (<{NEAR_SIMULTANEOUS_YEARS:g} yr)",
    )
    ax.set_xlabel("Number of independent origins")
    ax.set_ylabel("AA distance between origin backgrounds")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")


def panel_distance_vs_time(ax: plt.Axes, recurrent: pd.DataFrame) -> None:
    """Panel C: background distance versus time separating the origins."""
    for is_epitope, color in [(True, EPITOPE_COLOR), (False, NON_EPITOPE_COLOR)]:
        subset = recurrent[recurrent["is_epitope"] == is_epitope]
        ax.scatter(
            subset["min_time_between_origins"].clip(lower=1e-3),
            subset["mean_origin_background_distance_aa"],
            c=color,
            alpha=0.5,
            linewidths=0,
            s=18,
            label=_label(is_epitope),
        )
    # Binned medians summarise the trend across time-separation scales.
    edges = np.array([0.0, 0.05, 0.5, 2.0, 10.0, 40.0])
    centers, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        window = recurrent[
            (recurrent["min_time_between_origins"] >= low)
            & (recurrent["min_time_between_origins"] < high)
        ]
        if len(window):
            centers.append(np.sqrt(max(low, 1e-3) * high))
            medians.append(window["mean_origin_background_distance_aa"].median())
    ax.plot(
        centers,
        medians,
        color="black",
        marker="o",
        linewidth=1.5,
        label="binned median",
    )
    ax.axvspan(1e-3, NEAR_SIMULTANEOUS_YEARS, color=NULL_COLOR, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Time between origins (years)")
    ax.set_ylabel("AA distance between origin backgrounds")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")


def _progeny_size(series: pd.Series) -> np.ndarray:
    """Marker size scaled by log progeny, matching the other scatter panels."""
    return 20 + 40 * np.log10(series.clip(lower=1) + 1)


def panel_antigenic_vs_genetic(
    ax: plt.Axes, pairs: pd.DataFrame, null: pd.DataFrame
) -> None:
    """Per-pair antigenic distance vs genetic background distance (epitope only).

    Antigenic position drifts with cumulative epitope count, so a pair's antigenic
    distance is only interpretable relative to its genetic background distance: the
    dashed matched-null line is the antigenic distance random epitope-origin pairs
    show at each genetic separation. Points in the shaded small-x (similar
    background) region sitting above that line are same-mutation origins that the
    random per-event direction pulled apart antigenically despite arising in nearly
    identical backgrounds.
    """
    near = pairs["time_between_origins"] < NEAR_SIMULTANEOUS_YEARS
    resolved = pairs[~near]
    ax.scatter(
        resolved["genetic_background_distance_aa"],
        resolved["antigenic_distance"],
        s=_progeny_size(resolved["min_progeny_of_pair"]),
        c=EPITOPE_COLOR,
        alpha=0.6,
        linewidths=0,
        label="epitope origin pair",
    )
    near_pairs = pairs[near]
    ax.scatter(
        near_pairs["genetic_background_distance_aa"],
        near_pairs["antigenic_distance"],
        s=_progeny_size(near_pairs["min_progeny_of_pair"]),
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        label=f"near-simultaneous (<{NEAR_SIMULTANEOUS_YEARS:g} yr)",
    )
    # Matched-null baseline: median antigenic distance of random epitope-origin
    # pairs, binned by their genetic background distance (the drift expected at
    # each genetic separation).
    edges = np.array([0, 1, 2, 4, 8, 16, 32])
    centers, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        window = null[
            (null["genetic_background_distance_aa"] >= low)
            & (null["genetic_background_distance_aa"] < high)
        ]
        if len(window):
            centers.append((low + high) / 2)
            medians.append(window["antigenic_distance"].median())
    ax.plot(
        centers,
        medians,
        color=NULL_COLOR,
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="matched null (median)",
    )
    ax.axvspan(0, 4, color=NULL_COLOR, alpha=0.08)
    ax.set_xlabel("Genetic background distance (AA)")
    ax.set_ylabel("Antigenic distance between origins")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")


def build_figure(mutations: pd.DataFrame, null: pd.DataFrame) -> plt.Figure:
    recurrent = mutations[mutations["n_independent_origins"] >= 2].dropna(
        subset=["mean_origin_background_distance_aa"]
    )
    assert not recurrent.empty, "no recurrent mutations to plot"
    assert recurrent["min_time_between_origins"].notna().any(), (
        "min_time_between_origins is all NaN; rerun the analysis with --tips-csv"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panel_distribution(axes[0], recurrent, null)
    panel_origins_vs_distance(axes[1], recurrent)
    panel_distance_vs_time(axes[2], recurrent)
    for label, ax in zip("ABC", axes):
        ax.set_title(label, fontweight="bold", loc="left")
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def build_occurrence_figure(mutations: pd.DataFrame) -> plt.Figure:
    """Single-panel CDF of independent-origin counts, per site class."""
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    panel_occurrence_counts(ax, mutations)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def build_antigenic_figure(pairs: pd.DataFrame, null: pd.DataFrame) -> plt.Figure:
    """Single-panel antigenic-vs-genetic distance scatter for epitope pairs."""
    assert not pairs.empty, "no epitope origin pairs to plot; rerun with --tips-csv"
    assert not null.empty, "antigenic null is empty; rerun with --tips-csv"
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.6))
    panel_antigenic_vs_genetic(ax, pairs, null)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations-csv", type=Path, required=True)
    parser.add_argument("--null-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="figureS5_mutation_homoplasy_distance",
    )
    parser.add_argument(
        "--occurrence-prefix",
        type=str,
        default="figureS5_mutation_occurrence_counts",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Optional per-origin-pair CSV; enables the antigenic-distance figure.",
    )
    parser.add_argument(
        "--antigenic-null-csv",
        type=Path,
        default=None,
        help="Optional antigenic-null CSV paired with --pairs-csv.",
    )
    parser.add_argument(
        "--antigenic-prefix",
        type=str,
        default="figureS5_mutation_antigenic_distance",
    )
    return parser.parse_args()


def _save(fig: plt.Figure, output_dir: Path, prefix: str) -> None:
    for suffix in ("pdf", "png"):
        path = output_dir / f"{prefix}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mutations = pd.read_csv(args.mutations_csv)
    null = pd.read_csv(args.null_csv)

    with plt.rc_context(RC_PARAMS):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _save(build_figure(mutations, null), args.output_dir, args.output_prefix)
        _save(
            build_occurrence_figure(mutations),
            args.output_dir,
            args.occurrence_prefix,
        )
        if args.pairs_csv is not None and args.antigenic_null_csv is not None:
            pairs = pd.read_csv(args.pairs_csv)
            antigenic_null = pd.read_csv(args.antigenic_null_csv)
            _save(
                build_antigenic_figure(pairs, antigenic_null),
                args.output_dir,
                args.antigenic_prefix,
            )


if __name__ == "__main__":
    main()
