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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_background_distances import NEAR_SIMULTANEOUS_YEARS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigentools.supplement_style import (  # noqa: E402
    SUPPLEMENT_RC,
    add_panel_letters,
    hollow_boxes,
    seed_jitter,
    style_panel,
)

EPITOPE_COLOR = "#B30000"
NON_EPITOPE_COLOR = "#3498db"
NULL_COLOR = "#7f7f7f"
# Used where a statistic pools epitope and non-epitope, so neither class color
# would be honest.
POOLED_COLOR = "#4a4a6a"

# The supplement convention lives in antigentools.supplement_style so this
# figure cannot drift from S4/S5 again. Retained under the old name because
# the notebook and main() both reference it.
RC_PARAMS = SUPPLEMENT_RC


def _label(is_epitope: bool) -> str:
    return "epitope" if is_epitope else "non-epitope"


def drop_near_simultaneous(mutations: pd.DataFrame) -> pd.DataFrame:
    """Drop substitutions whose origins are too close in time to be independent.

    A substitution placed on many sibling branches within hours of one another is
    an unresolved polytomy, not repeated evolution: on the representative build
    the worst case has 53 "origins" separated by under two hours, on backgrounds
    averaging 0.68 amino acids apart. Removing them caps the epitope maximum at
    12, in line with less densely sampled builds.

    ``min_time_between_origins`` is the smallest gap between any two origins of a
    substitution, and is NaN for single-origin substitutions, which are kept.

    Only whole substitutions can be dropped, not individual origin pairs: the
    per-origin times are not carried in the mutations table, so a count cannot be
    corrected by collapsing just the offending pair.
    """
    gap = mutations["min_time_between_origins"]
    assert gap.isna().equals(mutations["n_independent_origins"] == 1), (
        "min_time_between_origins should be NaN exactly for single-origin "
        "substitutions; the filter's treatment of NaN depends on it"
    )
    return mutations[gap.isna() | (gap >= NEAR_SIMULTANEOUS_YEARS)].copy()


def panel_occurrence_counts(ax: plt.Axes, mutations: pd.DataFrame, log_x: bool) -> None:
    """Cumulative fraction of mutations with <= X independent origins, per class.

    Unlike the distance panels, this uses *all* mutations (including
    single-origin ones), so X starts at 1: the value at X=1 is the fraction of
    mutations that arise only once.

    Args:
        ax: Axes to draw on.
        mutations: Per-mutation table with ``n_independent_origins`` and
            ``is_epitope``.
        log_x: Use a log x-axis. The origin count is heavy-tailed, so on builds
            where a single mutation reaches several dozen origins a linear axis
            compresses all the structure into the leftmost fifth of the panel.
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
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
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
    panel_occurrence_counts(ax, mutations, log_x=False)
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


# ---------------------------------------------------------------------------
# Across-run panels, built from scripts/sweep_mutation_homoplasy.py output.
# ---------------------------------------------------------------------------

CLASS_COLORS = {"epitope": EPITOPE_COLOR, "non_epitope": NON_EPITOPE_COLOR}
CLASS_LABELS = {"epitope": "epitope", "non_epitope": "non-epitope"}

# The statistic the figure reports. Restricting to substitutions with exactly two
# independent origins leaves one pairwise distance, so there is no
# minimum-selection effect and the matched null degenerates to the plain
# single-pair null. See specs/mutation_homoplasy.md.
HEADLINE_STATISTIC = "two_origins"


def _rate_per_run(
    summary: pd.DataFrame, numerator: str, denominator: str
) -> pd.DataFrame:
    """Long-form per-run rate for both site classes, ready for a categorical plot."""
    frames = []
    for site_class in ("epitope", "non_epitope"):
        rate = (
            summary[f"{numerator}_{site_class}"]
            / summary[f"{denominator}_{site_class}"]
        )
        frames.append(
            pd.DataFrame(
                {
                    "site_class": CLASS_LABELS[site_class],
                    "rate": rate.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).dropna()


def _strip_panel(ax: plt.Axes, data: pd.DataFrame, ylabel: str) -> None:
    """Box plus per-run points, matching the idiom used by the other aggregates."""
    order = [CLASS_LABELS["epitope"], CLASS_LABELS["non_epitope"]]
    palette = {CLASS_LABELS[k]: v for k, v in CLASS_COLORS.items()}
    sns.boxplot(
        data=data,
        x="site_class",
        y="rate",
        order=order,
        hue="site_class",
        hue_order=order,
        palette=palette,
        ax=ax,
        fliersize=0,
        width=0.55,
        linewidth=1.5,
        legend=False,
    )
    hollow_boxes(ax)
    seed_jitter()
    sns.stripplot(
        data=data,
        x="site_class",
        y="rate",
        order=order,
        hue="site_class",
        palette=palette,
        ax=ax,
        size=4.5,
        alpha=0.6,
        jitter=0.22,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def panel_recurrence_rate(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Fraction of substitutions arising independently two or more times, per run."""
    data = _rate_per_run(summary, "n_recurrent", "n_substitutions")
    _strip_panel(ax, data, "Fraction recurring")


def panel_reversion_rate(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Fraction of substitutions in a gain-then-loss cycle on one lineage, per run."""
    data = _rate_per_run(summary, "n_lineage_cycle", "n_substitutions")
    _strip_panel(ax, data, "Fraction in a reversion cycle")


def panel_similar_background(ax: plt.Axes, similar: pd.DataFrame) -> None:
    """Similar-background recurrence rate versus radius, against the matched null.

    One faint line per run per class, with the across-run median drawn bold and
    the matched null in grey. The null is what makes the observed rate readable:
    a class tracking or sitting below its null shows no tendency to recur in
    similar backgrounds.
    """
    subset = similar[similar["statistic"] == HEADLINE_STATISTIC]
    assert not subset.empty, f"no rows with statistic == {HEADLINE_STATISTIC!r}"

    for site_class, color in CLASS_COLORS.items():
        rows = subset[subset["site_class"] == site_class].copy()
        rows["rate"] = rows["n_within_k"] / rows["n_total"]
        for _, run_rows in rows.groupby(["config", "run"]):
            line = ax.plot(
                run_rows["k"],
                run_rows["rate"],
                color=color,
                lw=0.6,
                alpha=0.15,
                zorder=1,
            )[0]
            line.set_rasterized(True)
        median = rows.groupby("k")["rate"].median()
        ax.plot(
            median.index,
            median.to_numpy(),
            color=color,
            lw=2.6,
            marker="o",
            ms=4,
            zorder=3,
            label=CLASS_LABELS[site_class],
        )

    null_rows = subset.copy()
    null_rows["rate"] = null_rows["n_null_within_k"] / null_rows["n_null_total"]
    null_median = null_rows.groupby("k")["rate"].median()
    ax.plot(
        null_median.index,
        null_median.to_numpy(),
        color=NULL_COLOR,
        lw=2.0,
        ls="--",
        marker="s",
        ms=4,
        zorder=2,
        label="matched null",
    )

    ax.set_xlabel("Background similarity radius $k$ (AA)")
    ax.set_ylabel("Fraction recurring within $k$")
    ax.set_title("Similar-background recurrence vs. matched null", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")


def panel_observed_vs_null_distance(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Median same-mutation origin distance against the null, one point per run.

    ``median_same_mutation_bg_distance`` is taken over every recurrent
    substitution, so this panel pools epitope and non-epitope. It is therefore
    drawn in a neutral color: reusing the epitope red would imply an
    epitope-only statistic and read as contradicting the epitope curve in
    :func:`panel_similar_background`.
    """
    observed = summary["median_same_mutation_bg_distance"]
    null = summary["median_null_bg_distance"]
    valid = observed.notna() & null.notna()
    ax.scatter(
        null[valid],
        observed[valid],
        s=26,
        c=POOLED_COLOR,
        alpha=0.65,
        linewidths=0,
        label="simulation run (all substitutions)",
    )
    # Bracket the data rather than forcing the origin: every run sits far from
    # zero, so anchoring at zero would squeeze the points into one corner and
    # hide how they fall relative to the diagonal, which is the whole point.
    low = float(np.nanmin([observed[valid].min(), null[valid].min()]))
    high = float(np.nanmax([observed[valid].max(), null[valid].max()]))
    pad = max((high - low) * 0.15, 0.5)
    limits = [low - pad, high + pad]
    ax.plot(limits, limits, color=NULL_COLOR, ls="--", lw=1.4, label="equal distance")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    # Kept short: with an equal aspect the y-label spans the full panel height,
    # and a longer string collides with the panel letter placed above it.
    ax.set_xlabel("Random origin-pair distance (AA)")
    ax.set_ylabel("Same-mutation origin distance (AA)")
    ax.set_title("Origin distance vs. null, per run", fontsize=10)
    ax.legend(loc="upper left")


REPRESENTATIVE_RUN = ("nonEpitopeAcceptance_0.25_epitopeAcceptance_0.75", 14)


def _bin_label(low: int, high: int) -> str:
    """Readable label for a genetic-distance bin, with the top bin left open."""
    return f"{low}+" if high >= 1000 else f"{low}\u2013{high}"


def panel_genotype_antigenic(
    ax: plt.Axes,
    summary: pd.DataFrame,
    genotype: pd.DataFrame,
    representative_run: tuple[str, int],
) -> None:
    """Antigenic distance against genetic distance, one line per simulation.

    The direct answer to the reviewer's operational concern. Antigenic position is
    the cumulative sum of every mutation along a lineage, so genetically similar
    viruses share nearly all of their displacement; the per-event random direction
    perturbs a single step rather than decoupling genotype from phenotype.

    The bold line is the representative simulation rather than the across-run
    median, matching the ECDF panel of figure S5: the representative run carries
    the main-text detailed analysis, so showing where it sits inside the ensemble
    is more useful than a median no individual simulation produced.
    """
    assert not genotype.empty, "genotype-antigenic table is empty"
    bins = (
        genotype[["genetic_bin_low", "genetic_bin_high"]]
        .drop_duplicates()
        .sort_values("genetic_bin_low")
    )
    labels = [_bin_label(int(lo), int(hi)) for lo, hi in bins.to_numpy()]
    positions = {int(lo): i for i, lo in enumerate(bins["genetic_bin_low"])}

    representative = None
    for (config, run), run_rows in genotype.groupby(["config", "run"]):
        ordered = run_rows.sort_values("genetic_bin_low")
        if (config, int(run)) == representative_run:
            representative = ordered
            continue
        line = ax.plot(
            [positions[int(lo)] for lo in ordered["genetic_bin_low"]],
            ordered["median_antigenic_distance"],
            color=EPITOPE_COLOR,
            lw=0.6,
            alpha=0.15,
            zorder=1,
        )[0]
        line.set_rasterized(True)

    if representative is not None:
        ax.plot(
            [positions[int(lo)] for lo in representative["genetic_bin_low"]],
            representative["median_antigenic_distance"],
            color=EPITOPE_COLOR,
            lw=2.8,
            marker="o",
            ms=5,
            zorder=3,
            label="representative simulation",
        )
        # Frameless, matching the bottom legends on S4 and S5.
        ax.legend(loc="upper left", frameon=False)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Genetic distance between viruses (AA)")
    ax.set_ylabel("Median antigenic distance")
    ax.set_ylim(bottom=0)


def panel_identical_sequence_spread(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Fraction of shared protein sequences occupying a single antigenic position.

    The sharpest form of the concern: if the random per-event direction decoupled
    genotype from phenotype, viruses with an identical protein would still be
    scattered. Reported as a rate rather than as spread magnitudes, because the
    magnitudes span orders of magnitude -- most groups sit at exactly one position
    while a few reach several units -- and a linear axis over them buries the
    result under its own outliers. The magnitude belongs in the caption.

    The denominator is sequences, not tips: ``assign_all_variants.py`` is handed
    the deduplicated tip table, so a method encounters each distinct sequence once,
    and weighting by tip abundance would over-count the common ones.
    """
    assert "frac_shared_sequences_spreading" in summary, (
        "summary lacks identical-sequence columns; rerun the sweep against runs "
        "that retain a full tips.csv"
    )
    spreading = summary["frac_shared_sequences_spreading"].dropna()
    assert not spreading.empty, "no simulation has identical-sequence spread data"
    single_position = 1.0 - spreading

    data = pd.DataFrame({"x": "", "fraction": single_position.to_numpy()})
    sns.boxplot(
        data=data,
        x="x",
        y="fraction",
        ax=ax,
        color=EPITOPE_COLOR,
        fliersize=0,
        width=0.35,
        linewidth=1.5,
    )
    hollow_boxes(ax)
    seed_jitter()
    sns.stripplot(
        data=data,
        x="x",
        y="fraction",
        ax=ax,
        color=EPITOPE_COLOR,
        size=5,
        alpha=0.6,
        jitter=0.18,
    )
    n = len(single_position)
    ax.set_xlabel(f"{n} simulation{'' if n == 1 else 's'}")
    ax.set_ylabel("Fraction at one position")
    ax.set_ylim(0, 1.02)


def build_across_run_figure(
    summary: pd.DataFrame, genotype: pd.DataFrame, representative_run: tuple[str, int]
) -> plt.Figure:
    """Assemble the four-panel across-run figure for Reviewer 1, comment 1a.

    Each panel is a distribution over every swept simulation rather than a single
    build, which removes the cherry-picking objection the single-build version
    invited. A and B characterize the simulation's mutational behavior; C and D
    answer the reviewer's operational concern directly, by showing that genotype
    predicts antigenic position despite the per-event random direction.

    Styling and panel-letter placement come from ``antigentools.supplement_style``
    so this figure matches S4 and S5 rather than inventing a third convention.

    ``panel_similar_background`` and ``panel_observed_vs_null_distance`` are
    retained in this module because they still back the response letter, but they
    no longer earn a panel: C and D make the argument more directly.

    ``panel_occurrence_counts`` is likewise not used here; it is rendered on its
    own as supplementary figure S7, the origin-count distribution for the
    representative simulation.
    """
    assert not summary.empty, "per-run summary is empty"
    # Wide and comparatively short: the figure is placed at 1.3\\textwidth, so a
    # taller aspect pushes the caption off the bottom of the page.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.9))
    panel_recurrence_rate(axes[0, 0], summary)
    panel_reversion_rate(axes[0, 1], summary)
    panel_genotype_antigenic(axes[1, 0], summary, genotype, representative_run)
    panel_identical_sequence_spread(axes[1, 1], summary)
    for ax in axes.flat:
        style_panel(ax)
    # h_pad opens a gap between the rows for the lower panel letters, which are
    # drawn just above each row and would otherwise land on the row above's tick
    # labels. S4 and S5 are single-row figures and never needed this, so the
    # value is tuned here rather than in the shared helper.
    fig.tight_layout(h_pad=3.0)
    add_panel_letters(fig, axes, "ABCD")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations-csv", type=Path, required=True)
    parser.add_argument("--null-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="figureS3_mutation_homoplasy_distance",
    )
    parser.add_argument(
        "--occurrence-prefix",
        type=str,
        default="figureS3_mutation_occurrence_counts",
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
        default="figureS3_mutation_antigenic_distance",
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
