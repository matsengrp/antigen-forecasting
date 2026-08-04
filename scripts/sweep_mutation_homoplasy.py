"""Measure mutation homoplasy and reversion across every run in a sweep.

Reviewer 1's comment 1a bundles two objections to ``antigen-prime``'s per-event
random antigenic vectors: that the same substitution has no repeatable antigenic
effect, and that the model is not time-reversible. Measured on a single build
those two have opposite answers -- recurrence is common, lineage reversion is
rare -- so the response needs both quantified, and quantified across every
simulation rather than one build that could be read as cherry-picked.

This driver walks each run's Auspice tree and reduces it to a handful of scalars,
concatenating them into small committable tables under
``results/aggregated/<batch>/``. Per-run trees live only on the cluster (the repo
gitignores ``data/**/*__run_*/``), so this is normally submitted through
``scripts/submit_homoplasy_sweep.sh`` and only its output CSVs come back.

Each run is processed in two tiers:

1. **Tree only.** Recurrence counts and every reversion count. Needs just the
   Auspice JSON plus the shared GenBank reference and epitope-site list, so it
   cannot be defeated by a missing per-run FASTA. A failure here is a real error
   and aborts the run's row.
2. **Full reconstruction.** Origin dates (for the near-simultaneous artifact
   correction) and background distances, via
   ``mutation_background_distances.compute_tables``. This needs the per-run FASTA
   and a derived anchor tip, so it is guarded: on failure the tier-1 numbers still
   land and the reason is recorded in ``notes``.

Emitted tables:
- ``mutation_homoplasy_by_run.csv`` -- one row per run.
- ``mutation_homoplasy_similar_background_by_k.csv`` -- long form, the count of
  recurrent substitutions whose origin backgrounds sit within ``k`` amino acids
  of each other, for each ``k``, site class, statistic, and progeny threshold.
  Every row carries its own matched null (``n_null_total``, ``n_null_within_k``),
  because the observed rate means nothing on its own: what matters is whether it
  exceeds what arbitrarily chosen origins would give.

  Three statistics are reported. ``two_origins`` is the headline: restricted to
  substitutions with exactly two independent origins, so there is a single
  pairwise distance, no minimum-selection effect, and no choice to defend.
  ``mean`` and ``min`` use all recurrent substitutions and are reported for
  completeness -- they disagree, and the disagreement is itself informative, so
  neither is silently promoted. See ``specs/mutation_homoplasy.md``.

Usage:
    python scripts/sweep_mutation_homoplasy.py --batch 2026-07-04-reviewer-runs \\
        --data-root data/ -j 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_background_distances import (  # noqa: E402
    HA1_LENGTH,
    NEAR_SIMULTANEOUS_YEARS,
    Reconstruction,
    compute_tables,
    identical_sequence_antigenic_spread,
    index_origins,
    is_epitope_mutation,
    load_epitope_sites,
    load_gene_layouts,
    load_tip_antigenic,
    load_tip_sequences,
    load_tree,
    sample_genotype_antigenic_pairs,
    summarize_reversions,
)

logger = logging.getLogger(__name__)

# Amino-acid thresholds at which the "same mutation recurred in a similar
# background" rate is reported. The old single-build figure shaded 0-4 as the
# similar-background regime; sweeping a range instead avoids resting the
# conclusion on one hand-picked cutoff.
DEFAULT_THRESHOLDS = (1, 2, 3, 4, 5, 6, 7, 8)

# Progeny thresholds the analysis is repeated at. 1 keeps every origin; 3 drops
# origins whose clade never established, which is what separates a substitution
# that genuinely recurred in a similar background from one where a single event
# was split across two branches by tree inference. Reporting only the unfiltered
# number would leave the minimum-distance statistic looking alarmingly high.
DEFAULT_PROGENY_THRESHOLDS = (1, 3)

# Minimum number of matched-null replicates drawn per (site class, statistic).
# The null must be estimated far more precisely than the observation it
# calibrates, or its own sampling scatter swamps the comparison.
NULL_REPLICATE_FLOOR = 5000

# Per-run inputs, relative to a run's data directory.
TREE_SUFFIX = Path("variant-assignment") / "phylogenetic" / "auspice.json"
FASTA_SUFFIX = Path("antigen-outputs") / "unique_sequences.fasta"
TIPS_SUFFIX = Path("antigen-outputs") / "unique_tips.csv"
# The full, non-deduplicated tip table. Needed only by the identical-sequence
# spread test, which measures repeat structure that unique_tips.csv collapses.
TIPS_FULL_SUFFIX = Path("antigen-outputs") / "tips.csv"

SUMMARY_NAME = "mutation_homoplasy_by_run.csv"
SIMILAR_NAME = "mutation_homoplasy_similar_background_by_k.csv"
GENOTYPE_NAME = "genotype_antigenic_by_distance.csv"

# Genetic-distance bins (amino acids, inclusive) for the genotype-to-antigenic
# curve. The top bin is open-ended so no pair is silently dropped.
GENETIC_DISTANCE_BINS = ((0, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10_000))

# Tip and pair subsampling for the genotype-antigenic comparison. Pair count grows
# quadratically in tips, so tips are capped first and pairs sampled from those.
GENOTYPE_MAX_TIPS = 1200
GENOTYPE_N_PAIRS = 200_000

# Antigenic units below which a group of identical sequences counts as occupying
# one position. Comparing against exactly zero would count floating-point noise
# and report 31.5% of groups as spreading where the true figure is 7.1%.
SPREAD_TOLERANCE = 0.01


def setup_logging(verbose: bool, log_file: Path | None) -> None:
    """Configure root logging to stream to stdout and, optionally, a file.

    Mirrors ``scripts/aggregate_results.py`` so the on-screen and on-disk logs
    share one timestamped format.

    Args:
        verbose: Emit DEBUG-level records when True, else INFO.
        log_file: If given, also append records to this file (parent dirs made).
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.root.setLevel(level)
    logging.root.addHandler(console_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def parse_run_identity(sim_dir: Path) -> tuple[str, int]:
    """Recover ``(config, run_number)`` from a ``<config>__run_N`` directory name.

    Splits on the last ``__``, matching ``scripts/aggregate_results.py`` (configs
    may contain single underscores but never the ``__`` delimiter).

    Args:
        sim_dir: Path whose name is ``<config>__run_N``.

    Returns:
        ``(config, run_number)``.

    Raises:
        ValueError: If the directory name does not match the expected pattern.
    """
    match = re.match(r"^(.+)__run_(\d+)$", sim_dir.name)
    if match is None:
        raise ValueError(
            f"cannot parse config/run from directory name {sim_dir.name!r}"
        )
    return match.group(1), int(match.group(2))


def discover_run_trees(data_root: Path, batch: str) -> list[tuple[Path, str, int]]:
    """Find every run in a batch that has an Auspice tree.

    Unlike ``aggregate_results.discover_runs`` this globs the *data* side, since
    trees are written under ``data/<batch>/`` rather than ``results/``. Runs whose
    pipeline never reached variant assignment simply have no tree; those are
    skipped with a warning rather than treated as an error.

    Args:
        data_root: Root data directory (e.g. ``data/``).
        batch: Batch name that namespaces the sweep.

    Returns:
        Sorted list of ``(auspice_path, config, run)``.

    Raises:
        FileNotFoundError: If the batch directory does not exist.
        ValueError: If the batch contains no run directories with a tree.
    """
    base = data_root / batch
    if not base.is_dir():
        raise FileNotFoundError(f"batch data directory not found: {base}")

    found: list[tuple[Path, str, int]] = []
    run_dirs = sorted(p for p in base.glob("*__run_*") if p.is_dir())
    for run_dir in run_dirs:
        config, run = parse_run_identity(run_dir)
        auspice = run_dir / TREE_SUFFIX
        if not auspice.is_file():
            logger.warning("%s: no auspice.json, skipping", run_dir.name)
            continue
        found.append((auspice, config, run))

    if not found:
        raise ValueError(f"no runs with an auspice.json under {base}")
    return found


def derive_anchor_tip(recon: Reconstruction, tip_nt: dict[str, str]) -> str:
    """Pick a tip present in both the tree and the alignment to anchor from.

    ``mutation_background_distances`` defaults to the flu-final-specific tip
    ``45b125db``, which does not exist in other builds. Reconstruction only needs
    *some* tip whose sequence can be translated, so take the first shared name in
    sorted order for determinism.

    Args:
        recon: The parsed tree.
        tip_nt: Tip name to nucleotide sequence, from the run's FASTA.

    Returns:
        The chosen anchor tip name.

    Raises:
        ValueError: If no tip appears in both the tree and the FASTA.
    """
    shared = sorted(set(recon.nodes) & set(tip_nt))
    if not shared:
        raise ValueError("no tip name appears in both the tree and the FASTA")
    return shared[0]


def count_by_class(
    keys: Sequence[tuple[str, int, str, str]], epitope_sites: set[int]
) -> tuple[int, int]:
    """Split substitution keys into (epitope, non-epitope) counts."""
    epitope = sum(
        1
        for gene, position, _, _ in keys
        if is_epitope_mutation(gene, position, epitope_sites)
    )
    return epitope, len(keys) - epitope


def tree_only_stats(
    auspice: Path, ref_genbank: Path, epitope_sites: set[int]
) -> dict[str, Any]:
    """Compute every statistic that needs only the tree.

    Args:
        auspice: Path to the run's Auspice v2 JSON.
        ref_genbank: Shared GenBank reference giving the CDS layout.
        epitope_sites: 1-based HA1 epitope positions.

    Returns:
        Recurrence and reversion counts, plus tree-size context.
    """
    genes = load_gene_layouts(ref_genbank)
    recon = load_tree(auspice, genes)
    origins = index_origins(recon)

    recurrent = [key for key, nodes in origins.items() if len(nodes) >= 2]
    n_substitutions_epitope, n_substitutions_non_epitope = count_by_class(
        list(origins), epitope_sites
    )
    n_recurrent_epitope, n_recurrent_non_epitope = count_by_class(
        recurrent, epitope_sites
    )

    stats: dict[str, Any] = {
        "n_tips": sum(1 for node in recon.nodes.values() if not node.children),
        "n_nodes": len(recon.nodes),
        "n_substitutions": len(origins),
        "n_mutation_events": sum(len(nodes) for nodes in origins.values()),
        "n_substitutions_epitope": n_substitutions_epitope,
        "n_substitutions_non_epitope": n_substitutions_non_epitope,
        "n_recurrent_epitope": n_recurrent_epitope,
        "n_recurrent_non_epitope": n_recurrent_non_epitope,
    }
    stats.update(summarize_reversions(recon, origins, epitope_sites))
    return stats


def sequence_stats(
    auspice: Path,
    sequences_fasta: Path,
    tips_csv: Path,
    ref_genbank: Path,
    epitope_sites_path: Path,
    thresholds: Sequence[int],
    min_origin_progeny: int,
    null_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the statistics that need reconstructed background sequences.

    Reuses ``compute_tables`` so the definitions match the per-mutation figure
    exactly rather than being reimplemented here. The near-simultaneous artifact
    correction and the background distances both come out of its ``mutations``
    table.

    Args:
        auspice: Path to the run's Auspice v2 JSON.
        sequences_fasta: The run's unique-sequence alignment.
        tips_csv: The run's unique-tip metadata, supplying sampling dates.
        ref_genbank: Shared GenBank reference.
        epitope_sites_path: Shared epitope-site list.
        thresholds: Amino-acid cutoffs for the similar-background counts.
        min_origin_progeny: Drop origins whose clade has fewer than this many
            sampled tips. Transient origins dominate the minimum-distance
            statistic, so this is the control that separates a real recurrence in
            a similar background from a tree-inference fragment.
        null_samples: Null draws for the background-distance null.
        seed: Seed for the null sampling.

    Returns:
        ``(summary_columns, similar_background_rows)``.
    """
    genes = load_gene_layouts(ref_genbank)
    recon = load_tree(auspice, genes)
    tip_nt = load_tip_sequences(sequences_fasta)
    anchor_tip = derive_anchor_tip(recon, tip_nt)

    tables = compute_tables(
        auspice_json=auspice,
        sequences_fasta=sequences_fasta,
        ref_genbank=ref_genbank,
        epitope_sites_path=epitope_sites_path,
        tips_csv=tips_csv,
        anchor_tip=anchor_tip,
        min_origin_progeny=min_origin_progeny,
        max_carriers=1500,
        null_samples=null_samples,
        seed=seed,
    )
    mutations = tables["mutations"]
    background_null = tables["background_null"]

    recurrent = mutations[mutations["n_independent_origins"] >= 2]
    resolved = recurrent[
        recurrent["min_time_between_origins"] >= NEAR_SIMULTANEOUS_YEARS
    ]

    summary: dict[str, Any] = {
        "anchor_tip": anchor_tip,
        "n_recurrent_corrected_epitope": int(resolved["is_epitope"].sum()),
        "n_recurrent_corrected_non_epitope": int((~resolved["is_epitope"]).sum()),
        "median_same_mutation_bg_distance": float(
            recurrent["mean_origin_background_distance_aa"].median()
        ),
        "median_min_same_mutation_bg_distance": float(
            recurrent["min_origin_background_distance_aa"].median()
        ),
        "median_null_bg_distance": float(
            background_null["background_distance_aa"].median()
        ),
    }

    # A recurrent substitution counts as "similar background" when its origins sit
    # within k amino acids of each other. Both the mean and the minimum pairwise
    # distance are reported: the mean asks how different the backgrounds are in
    # general, the minimum asks whether the mutation *ever* recurred in a
    # near-identical background.
    #
    # Each row carries its own matched null, and that matching is the whole point.
    # A substitution with j origins contributes C(j,2) pairwise distances, so its
    # minimum is a minimum over many draws. Comparing that against a single random
    # pair is biased downward by construction: on flu-final it makes 33.5% of
    # recurrent epitope substitutions look like they recurred within 1 residue,
    # against a 7.5% single-pair null, when the correctly matched null is 41.8% --
    # i.e. the observed rate is *below* chance. The null is therefore built by
    # drawing the same number of pairs each substitution actually has and applying
    # the same statistic to them.
    null_distances = background_null["background_distance_aa"].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    similar_rows: list[dict[str, Any]] = []
    # ``two_origins`` is the headline statistic: restricting to substitutions with
    # exactly two independent origins leaves a single pairwise distance, so the
    # mean and the minimum coincide, there is no minimum-selection effect to
    # correct for, and the matched null below degenerates to the plain
    # single-pair null. It is the version that needs no methodological defense.
    for statistic, column, two_origins_only in (
        ("two_origins", "mean_origin_background_distance_aa", True),
        ("mean", "mean_origin_background_distance_aa", False),
        ("min", "min_origin_background_distance_aa", False),
    ):
        reduce = np.min if statistic == "min" else np.mean
        for site_class, mask in (
            ("epitope", recurrent["is_epitope"]),
            ("non_epitope", ~recurrent["is_epitope"]),
        ):
            subset = recurrent[mask]
            if two_origins_only:
                subset = subset[subset["n_independent_origins"] == 2]
            distances = subset[column].dropna()
            # Pair counts drive the matching: C(j,2) for a substitution with j
            # independent origins.
            origin_counts = subset.loc[distances.index, "n_independent_origins"]
            pair_counts = (origin_counts * (origin_counts - 1) // 2).astype(int)
            # Resample the pair-count distribution up to a floor so the null curve
            # is smooth. Drawing only len(distances) replicates would leave the
            # null as noisy as the observation it is meant to calibrate -- with 50
            # epitope substitutions that scatters the null by several points and
            # makes two classes drawn from the same distribution look different.
            if len(pair_counts):
                drawn = rng.choice(
                    pair_counts.to_numpy(),
                    size=max(len(pair_counts), NULL_REPLICATE_FLOOR),
                    replace=True,
                )
                matched = np.array(
                    [
                        reduce(
                            rng.choice(
                                null_distances, size=max(int(n), 1), replace=True
                            )
                        )
                        for n in drawn
                    ]
                )
            else:
                matched = np.array([])
            for k in thresholds:
                similar_rows.append(
                    {
                        "site_class": site_class,
                        "statistic": statistic,
                        "min_origin_progeny": min_origin_progeny,
                        "k": int(k),
                        "n_total": int(len(distances)),
                        "n_within_k": int((distances <= k).sum()),
                        "n_null_total": int(len(matched)),
                        "n_null_within_k": int((matched <= k).sum()),
                    }
                )
    return summary, similar_rows


IDENTICAL_SEQUENCE_COLUMNS = (
    "n_identical_sequence_groups",
    "n_tips_in_shared_sequences",
    "n_tips_total",
    "frac_shared_sequences_spreading",
    "frac_tips_in_spreading_group",
    "identical_seq_spread_median",
    "identical_seq_spread_p90",
    "identical_seq_spread_max",
    "identical_seq_spread_median_when_spreading",
)


def identical_sequence_columns(
    recon: Reconstruction, full_tips_csv: Path | None
) -> dict[str, Any]:
    """Summarise antigenic spread among tips sharing an amino-acid sequence.

    Two rates are emitted because the choice of denominator changes the headline
    number roughly sixfold. ``frac_shared_sequences_spreading`` counts sequences,
    which is the rate that matters for benchmark fairness: the assignment methods
    are handed ``unique_tips.csv`` (``assign_all_variants.py`` asserts this), so a
    method encounters each distinct sequence exactly once. The tip-weighted
    ``frac_tips_in_spreading_group`` is reported alongside it because the spreading
    sequences tend to be the common ones, and hiding that behind a denominator
    choice would be misleading.

    Args:
        recon: Parsed tree, for the gene layout used in translation.
        full_tips_csv: Full, non-deduplicated tip table, or None when absent.

    Returns:
        The columns in ``IDENTICAL_SEQUENCE_COLUMNS``, all NaN when unavailable.
    """
    if full_tips_csv is None or not full_tips_csv.is_file():
        return dict.fromkeys(IDENTICAL_SEQUENCE_COLUMNS, np.nan)

    tips = pd.read_csv(
        full_tips_csv, usecols=["nucleotideSequence", "ag1", "ag2"]
    ).dropna()
    spread = identical_sequence_antigenic_spread(recon, tips)
    if spread.empty:
        return dict.fromkeys(IDENTICAL_SEQUENCE_COLUMNS, np.nan)

    spreading = spread[spread["max_pairwise_spread"] > SPREAD_TOLERANCE]
    return {
        "n_identical_sequence_groups": int(len(spread)),
        "n_tips_in_shared_sequences": int(spread["n_tips"].sum()),
        "n_tips_total": int(len(tips)),
        "frac_shared_sequences_spreading": float(len(spreading) / len(spread)),
        "frac_tips_in_spreading_group": float(spreading["n_tips"].sum() / len(tips)),
        # Most groups sit at exactly one position, so the median and p90 of the
        # full distribution are both essentially zero; the informative numbers are
        # the maximum and the median among groups that actually spread.
        "identical_seq_spread_median": float(spread["max_pairwise_spread"].median()),
        "identical_seq_spread_p90": float(spread["max_pairwise_spread"].quantile(0.9)),
        "identical_seq_spread_max": float(spread["max_pairwise_spread"].max()),
        "identical_seq_spread_median_when_spreading": (
            float(spreading["max_pairwise_spread"].median())
            if len(spreading)
            else np.nan
        ),
    }


def genotype_antigenic_stats(
    recon: Reconstruction,
    tip_nt: dict[str, str],
    tips_csv: Path,
    full_tips_csv: Path | None,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure how tightly genotype predicts antigenic position in one run.

    This addresses the reviewer's concern in its operational form -- whether
    genetically similar viruses can sit anywhere in antigenic space -- rather than
    in terms of a single mutation's marginal effect.

    The two halves deliberately read different files. The correlation samples the
    *unique* tips so each distinct genotype contributes one point and no single
    large clade dominates. The identical-sequence spread needs the *full* tip
    table, since deduplication collapses exactly the repeat structure it measures.

    Args:
        recon: Parsed tree, for the gene layout used in translation.
        tip_nt: Tip name to nucleotide sequence, from the unique-sequence FASTA.
        tips_csv: Unique-tip metadata carrying ``ag1``/``ag2``.
        full_tips_csv: Full tip table, or None when the run retained no such file.
        seed: Seed for tip and pair subsampling.

    Returns:
        ``(summary_columns, binned_rows)``.
    """
    tip_antigenic, _ = load_tip_antigenic(tips_csv)
    pairs = sample_genotype_antigenic_pairs(
        recon, tip_nt, tip_antigenic, GENOTYPE_MAX_TIPS, GENOTYPE_N_PAIRS, seed
    )

    summary: dict[str, Any] = {
        "genotype_antigenic_pearson": float(
            pairs["genetic_distance_aa"].corr(pairs["antigenic_distance"])
        ),
        "genotype_antigenic_spearman": float(
            pairs["genetic_distance_aa"].corr(
                pairs["antigenic_distance"], method="spearman"
            )
        ),
        "antigenic_span": float(pairs["antigenic_distance"].max()),
    }
    summary.update(identical_sequence_columns(recon, full_tips_csv))

    binned: list[dict[str, Any]] = []
    for low, high in GENETIC_DISTANCE_BINS:
        window = pairs[
            (pairs["genetic_distance_aa"] >= low)
            & (pairs["genetic_distance_aa"] <= high)
        ]
        if window.empty:
            continue
        binned.append(
            {
                "genetic_bin_low": low,
                "genetic_bin_high": high,
                "n_pairs": int(len(window)),
                "median_antigenic_distance": float(
                    window["antigenic_distance"].median()
                ),
                "q25_antigenic_distance": float(
                    window["antigenic_distance"].quantile(0.25)
                ),
                "q75_antigenic_distance": float(
                    window["antigenic_distance"].quantile(0.75)
                ),
            }
        )
    return summary, binned


def process_run(
    auspice: Path,
    config: str,
    run: int,
    batch: str,
    ref_genbank: Path,
    epitope_sites_path: Path,
    thresholds: Sequence[int],
    progeny_thresholds: Sequence[int],
    null_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reduce one run to its summary row and its two long-form tables.

    Tier 1 is unguarded: if the tree cannot be walked there is nothing to salvage.
    Tier 2 is guarded so a missing or malformed FASTA leaves the tree-only numbers
    intact, with the reason recorded in ``notes``.

    The wide summary row reports the unfiltered (first) progeny threshold; the
    long-form rows carry every threshold, so the sensitivity of the recurrence
    counts to transient origins is readable straight off that table via
    ``n_total``.

    Args:
        auspice: Path to the run's Auspice v2 JSON.
        config: Parameter-set name.
        run: Run number within the parameter set.
        batch: Batch name.
        ref_genbank: Shared GenBank reference.
        epitope_sites_path: Shared epitope-site list.
        thresholds: Amino-acid cutoffs for the similar-background counts.
        progeny_thresholds: ``min_origin_progeny`` values to report at.
        null_samples: Null draws for the background-distance null.
        seed: Seed for the null sampling.

    Returns:
        ``(summary_row, similar_background_rows)``. The rows list is empty when
        tier 2 failed.
    """
    assert progeny_thresholds, "at least one progeny threshold is required"
    identity = {"batch": batch, "config": config, "run": run}
    epitope_sites = load_epitope_sites(epitope_sites_path, HA1_LENGTH)

    row: dict[str, Any] = dict(identity)
    row.update(tree_only_stats(auspice, ref_genbank, epitope_sites))

    notes: list[str] = []
    similar_rows: list[dict[str, Any]] = []
    run_dir = auspice.parent.parent.parent
    sequences_fasta = run_dir / FASTA_SUFFIX
    tips_csv = run_dir / TIPS_SUFFIX
    try:
        if not sequences_fasta.is_file():
            raise FileNotFoundError(f"missing {sequences_fasta}")
        if not tips_csv.is_file():
            raise FileNotFoundError(f"missing {tips_csv}")
        for index, min_origin_progeny in enumerate(progeny_thresholds):
            summary, rows = sequence_stats(
                auspice,
                sequences_fasta,
                tips_csv,
                ref_genbank,
                epitope_sites_path,
                thresholds,
                min_origin_progeny,
                null_samples,
                seed,
            )
            if index == 0:
                row.update(summary)
            similar_rows.extend(dict(identity, **entry) for entry in rows)

        genes = load_gene_layouts(ref_genbank)
        recon = load_tree(auspice, genes)
        tip_nt = load_tip_sequences(sequences_fasta)
        # The identical-sequence test needs the full tip table; not every run is
        # guaranteed to have retained one, so its absence degrades that statistic
        # rather than failing the whole sequence tier.
        full_tips_csv: Path | None = run_dir / TIPS_FULL_SUFFIX
        if full_tips_csv is not None and not full_tips_csv.is_file():
            notes.append("no full tips.csv; identical-sequence spread skipped")
            full_tips_csv = None
        geno_summary, geno_rows = genotype_antigenic_stats(
            recon, tip_nt, tips_csv, full_tips_csv, seed
        )
        row.update(geno_summary)
        genotype_rows = [dict(identity, **entry) for entry in geno_rows]
    except Exception as error:  # noqa: BLE001 - one bad run must not stop the sweep.
        notes.append(f"sequence stats failed: {error}")
        # Discard any partial per-threshold rows so the long tables never hold a
        # run at some thresholds but not others.
        similar_rows = []
        genotype_rows = []
        row.update(
            {
                "anchor_tip": None,
                "n_recurrent_corrected_epitope": np.nan,
                "n_recurrent_corrected_non_epitope": np.nan,
                "median_same_mutation_bg_distance": np.nan,
                "median_min_same_mutation_bg_distance": np.nan,
                "median_null_bg_distance": np.nan,
                "genotype_antigenic_pearson": np.nan,
                "genotype_antigenic_spearman": np.nan,
                "antigenic_span": np.nan,
                **dict.fromkeys(IDENTICAL_SEQUENCE_COLUMNS, np.nan),
            }
        )

    row["notes"] = ";".join(notes)
    return row, similar_rows, genotype_rows


def _write(frame: pd.DataFrame, output_path: Path, label: str) -> None:
    """Write one table, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info("wrote %s (%d rows) -> %s", label, len(frame), output_path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch name under --data-root.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/aggregated/<batch>/.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--ref-genbank",
        type=Path,
        default=Path("data/flu-final/auspice/ref_HA.gb"),
        help="Shared GenBank reference; identical across runs.",
    )
    parser.add_argument(
        "--epitope-sites",
        type=Path,
        default=Path("../antigen-prime/src/main/resources/epitopeSites.txt"),
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Amino-acid cutoffs for the similar-background counts.",
    )
    parser.add_argument(
        "--min-origin-progeny",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROGENY_THRESHOLDS),
        help=(
            "Progeny thresholds to report at. The first is used for the wide "
            "summary row; all appear in the long-form table. Transient origins "
            "dominate the minimum-distance statistic, so a filtered threshold is "
            "the control that separates real recurrence from tree-split fragments."
        ),
    )
    parser.add_argument("--null-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir or (args.results_root / "aggregated" / args.batch)
    log_file = args.log_file or (output_dir / "sweep_mutation_homoplasy.log")
    setup_logging(args.verbose, log_file)

    trees = discover_run_trees(args.data_root, args.batch)
    logger.info("found %d runs with a tree in batch %s", len(trees), args.batch)

    summary_rows: list[dict[str, Any]] = []
    similar_rows: list[dict[str, Any]] = []
    genotype_rows: list[dict[str, Any]] = []

    def record(
        index: int,
        result: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        row, rows, geno = result
        summary_rows.append(row)
        similar_rows.extend(rows)
        genotype_rows.extend(geno)
        status = "ok" if not row["notes"] else f"partial ({row['notes']})"
        logger.info(
            "[%d/%d] %s__run_%s -> %s",
            index,
            len(trees),
            row["config"],
            row["run"],
            status,
        )

    call_args = [
        (
            auspice,
            config,
            run,
            args.batch,
            args.ref_genbank,
            args.epitope_sites,
            args.thresholds,
            args.min_origin_progeny,
            args.null_samples,
            args.seed,
        )
        for auspice, config, run in trees
    ]

    if args.jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_run, *call): call[1:3] for call in call_args
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                record(index, future.result())
    else:
        for index, call in enumerate(call_args, start=1):
            record(index, process_run(*call))

    summary = pd.DataFrame(summary_rows).sort_values(["config", "run"])
    similar = pd.DataFrame(similar_rows)
    if not similar.empty:
        similar = similar.sort_values(
            ["config", "run", "min_origin_progeny", "statistic", "site_class", "k"]
        )

    genotype = pd.DataFrame(genotype_rows)
    if not genotype.empty:
        genotype = genotype.sort_values(["config", "run", "genetic_bin_low"])

    _write(summary, output_dir / SUMMARY_NAME, "per-run summary")
    _write(similar, output_dir / SIMILAR_NAME, "similar-background counts")
    _write(genotype, output_dir / GENOTYPE_NAME, "genotype-antigenic curve")

    # Report the two degradations separately. A run missing its FASTA falls all
    # the way back to tree-only numbers; a run that merely lacks the full tip
    # table still has everything except the identical-sequence spread, and
    # conflating the two overstates how much was lost.
    notes = summary["notes"].fillna("")
    tree_only = int(notes.str.contains("sequence stats failed").sum())
    no_full_tips = int(notes.str.contains("no full tips.csv").sum())
    if tree_only:
        logger.warning(
            "%d/%d runs fell back to tree-only statistics (no usable FASTA)",
            tree_only,
            len(summary),
        )
    if no_full_tips:
        logger.warning(
            "%d/%d runs have no full tips.csv, so the identical-sequence spread "
            "is missing for them (all other statistics are present)",
            no_full_tips,
            len(summary),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
