"""Render a multi-page PDF browser of flu-like candidate runs.

A selection aid for choosing a candidate run (e.g. a "representative run"). For every
candidate in ``candidate_runs.csv`` this emits one PDF page: the run's config, run-ID,
swept parameters, TMRCA, and mutation statistics as a title, followed by the figure-2
"simulation summary" 5-panel figure (tree, case counts, antigenic space, epitope
mutations, variant-frequency stackplot) rebuilt from that run's own outputs.

This is the non-Jupyter counterpart to ``notebooks/browse-candidate-runs.ipynb`` for
clusters where the Jupyter stack is unavailable. It uses the Agg backend (no display) and
reads the per-run inputs the pipeline wrote under ``data/<batch>/<config>__run_<n>/``, so
run it on HPC where those inputs live.

Any run missing an input gets a page with its stats and a note (affected panels blank)
rather than aborting the sweep.

Usage (from the repo root, in the antigen env):
    python scripts/browse_candidate_runs.py
    python scripts/browse_candidate_runs.py --sort tmrca --out /tmp/candidates.pdf
    python scripts/browse_candidate_runs.py --filter epitopeAcceptance_1.0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import baltic as bt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent

# Running as scripts/browse_candidate_runs.py puts scripts/ (not the repo root) on
# sys.path, so import antigentools by making the repo root importable first. This works
# whether or not the package is pip-installed in the active env.
sys.path.insert(0, str(REPO_ROOT))

from antigentools.paths import SimulationPaths  # noqa: E402

OVERVIEW_COLS = [
    "config",
    "run",
    "tmrca",
    "diversity",
    "antigenic_movement_per_year",
    "epitopeAcceptance",
    "nonEpitopeAcceptance",
    "trunk_epitope_to_non-epitope_ratio",
    "side_branch_epitope_to_non-epitope_ratio",
]


def create_variant_color_map(tips_df, variant_col="variant", time_col="year"):
    """Distinct colors for temporally adjacent variants (lifted from figure-2).

    Variants are ordered by birth year and assigned golden-ratio-spaced hues so that
    adjacent variants get very different colors.
    """
    variant_birth = tips_df.groupby(variant_col)[time_col].min().sort_values()
    variants_ordered = variant_birth.index.tolist()

    golden_ratio = 0.618033988749895
    colors = []
    for i in range(len(variants_ordered)):
        hue = (i * golden_ratio) % 1.0
        if i % 3 == 0:
            sat, val = 0.9, 0.95
        elif i % 3 == 1:
            sat, val = 0.7, 0.85
        else:
            sat, val = 0.85, 0.75
        rgb = mcolors.hsv_to_rgb([hue, sat, val])
        colors.append(mcolors.rgb2hex(rgb))

    return {variant: colors[i] for i, variant in enumerate(variants_ordered)}


def config_name(row):
    """Config (param-set) directory name for a candidate row: parent of run_<n>."""
    return os.path.basename(os.path.dirname(row["path"]))


def resolve_paths(row, batch, data_root, results_root):
    """SimulationPaths for one candidate row, using its (cluster-valid) sim path."""
    return SimulationPaths.from_sim_path(
        sim_path=row["path"],
        data_root=data_root,
        results_root=results_root,
        batch_name=batch,
    )


def load_tree(tree_path):
    """Load a newick tree and rescale branch lengths x1000 (0.03 -> 30 years).

    Mirrors figure-2's tree handling exactly.
    """
    tree = bt.loadNewick(str(tree_path))
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        tree.traverse_tree()
    for node in tree.Objects:
        if node.height is not None:
            node.height *= 1000
        if getattr(node, "absoluteTime", None) is not None:
            node.absoluteTime *= 1000
        if getattr(node, "length", None) is not None:
            node.length *= 1000
        if getattr(node, "x", None) is not None:
            node.x *= 1000
    if tree.treeHeight is not None:
        tree.treeHeight *= 1000
    return tree


def _find_tree_path(paths):
    """Locate the phylogenetic tree, preferring tree_raw.nwk then any *.nwk."""
    phylo_dir = paths.variant_assignment / "phylogenetic"
    preferred = phylo_dir / "tree_raw.nwk"
    if preferred.exists():
        return preferred
    if phylo_dir.is_dir():
        nwks = sorted(phylo_dir.glob("*.nwk"))
        if nwks:
            return nwks[0]
    return None


def _find_counts(paths, kind):
    """Locate seq_counts / case_counts, trying the pipeline build-root location first
    then the flu-final-style time-stamped/truth/ location."""
    assert kind in ("seq", "case")
    primary = paths.seq_counts if kind == "seq" else paths.case_counts
    if primary.exists():
        return primary
    fallback = paths.time_stamped / "truth" / f"{kind}_counts.tsv"
    if fallback.exists():
        return fallback
    return None


def load_run_inputs(paths):
    """Load the four per-run inputs for the figure. Returns (inputs, missing).

    inputs is a dict with any of {tips, tree, seqs, cases} that were found; missing is a
    list of human-readable descriptions of inputs that were absent. Each loader fails
    loudly if a file exists but is malformed; genuinely-absent files are reported via
    ``missing`` so the caller can skip the corresponding panel.
    """
    inputs = {}
    missing = []

    tips_path = paths.tips_with_variants
    if tips_path.exists():
        tips = pd.read_csv(tips_path, sep="\t")
        assert "variant_ag" in tips.columns, f"tips missing variant_ag: {tips_path}"
        inputs["tips"] = tips
    else:
        missing.append("tips_with_variants.tsv")

    tree_path = _find_tree_path(paths)
    if tree_path is not None:
        inputs["tree"] = load_tree(tree_path)
    else:
        missing.append("phylogenetic tree (*.nwk)")

    seq_path = _find_counts(paths, "seq")
    if seq_path is not None:
        seqs = pd.read_csv(seq_path, sep="\t")
        if "country" not in seqs.columns and "location" in seqs.columns:
            seqs = seqs.rename(columns={"location": "country"})
        for col in ("date", "variant", "sequences", "country"):
            assert col in seqs.columns, f"seq_counts missing {col!r}: {seq_path}"
        seqs["date"] = pd.to_datetime(seqs["date"])
        inputs["seqs"] = seqs
    else:
        missing.append("seq_counts.tsv")

    case_path = _find_counts(paths, "case")
    if case_path is not None:
        cases = pd.read_csv(case_path, sep="\t")
        if "country" not in cases.columns and "location" in cases.columns:
            cases = cases.rename(columns={"location": "country"})
        for col in ("date", "cases", "country"):
            assert col in cases.columns, f"case_counts missing {col!r}: {case_path}"
        cases["date"] = pd.to_datetime(cases["date"])
        inputs["cases"] = cases
    else:
        missing.append("case_counts.tsv")

    return inputs, missing


def run_title(row, missing):
    """Compact multi-line page title: config, run-ID, swept params, TMRCA, mutations."""
    cfg = config_name(row)
    lines = [
        f"{cfg}  —  run {int(row['run'])}",
        (
            f"epitopeAcceptance={row['epitopeAcceptance']}  "
            f"nonEpitopeAcceptance={row['nonEpitopeAcceptance']}   "
            f"tmrca={row['tmrca']:.2f} yr   diversity={row['diversity']:.2f}   "
            f"ag-move/yr={row['antigenic_movement_per_year']:.2f}"
        ),
        (
            f"trunk epi/non-epi={row['trunk_epitope_mutations']:.0f}/"
            f"{row['trunk_non-epitope_mutations']:.0f} "
            f"(ratio {row['trunk_epitope_to_non-epitope_ratio']:.2f})   "
            f"side-branch epi/non-epi={row['side_branch_epitope_mutations']:.0f}/"
            f"{row['side_branch_non-epitope_mutations']:.0f} "
            f"(ratio {row['side_branch_epitope_to_non-epitope_ratio']:.2f})"
        ),
    ]
    if missing:
        lines.append("missing inputs: " + ", ".join(missing))
    return "\n".join(lines)


def plot_run_page(inputs, row, missing, figsize):
    """Rebuild figure-2's 5-panel figure for one run and return the Figure.

    Panels: A tree, B case counts, C antigenic space, D mean epitope mutations per
    variant, E variant-frequency stackplot. Any panel whose input is missing is left
    blank with an in-axes note. Colors are generated per run (no saved color map).
    """
    tips = inputs.get("tips")
    tree = inputs.get("tree")
    seqs = inputs.get("seqs")
    cases = inputs.get("cases")

    variant_color_map = (
        create_variant_color_map(tips, "variant_ag", "year") if tips is not None else {}
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_cases = fig.add_subplot(gs[0, 1])
    ax_ag = fig.add_subplot(gs[1, 0])
    ax_epi = fig.add_subplot(gs[1, 1])
    ax_freq = fig.add_subplot(gs[2, :])

    def _blank(ax, msg):
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray", style="italic")
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel A: tree colored by variant.
    if tree is not None and tips is not None:
        variant_map = dict(zip(tips["name"], tips["variant_ag"]))
        for node in tree.Objects:
            if node.is_leaf():
                variant = variant_map.get(node.name)
                node.traits = {"variant_ag": variant,
                               "color": variant_color_map.get(variant, "gray")}
            else:
                node.traits = {"variant_ag": None, "color": "black"}
        tree.plotTree(ax_tree, width=1.5, colour="#333333")
        tree.plotPoints(
            ax_tree,
            target=lambda k: k.is_leaf(),
            size=lambda k: 10,
            colour=lambda k: k.traits.get("color", "black"),
            alpha=0.85,
            zorder=100,
            linewidths=0.3,
        )
        ax_tree.set_xlabel("Time (years)", fontsize=11)
        ax_tree.spines["right"].set_visible(False)
        ax_tree.spines["top"].set_visible(False)
        ax_tree.set_yticks([])
        ax_tree.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.5)
    else:
        _blank(ax_tree, "tree unavailable")
    ax_tree.text(-0.1, 1.05, "A", transform=ax_tree.transAxes, fontsize=16, fontweight="bold")

    # Panel B: case counts over time (years from first case).
    if cases is not None:
        cases_start = cases["date"].min()
        cases_plot = cases.copy()
        cases_plot["years"] = (cases_plot["date"] - cases_start).dt.days / 365.25
        sns.lineplot(data=cases_plot, x="years", y="cases", hue="country",
                     errorbar=None, ax=ax_cases)
        ax_cases.set_xlabel("Time (years)", fontsize=11)
        ax_cases.set_ylabel("Number of cases", fontsize=11)
        ax_cases.spines["top"].set_visible(False)
        ax_cases.spines["right"].set_visible(False)
        ax_cases.legend(title="Region", frameon=False)
    else:
        _blank(ax_cases, "case counts unavailable")
    ax_cases.text(-0.1, 1.05, "B", transform=ax_cases.transAxes, fontsize=16, fontweight="bold")

    # Panel C: antigenic space colored by variant.
    if tips is not None:
        sns.scatterplot(data=tips, x="ag1", y="ag2", hue="variant_ag",
                        palette=variant_color_map, alpha=0.7, ax=ax_ag, legend=False)
        ax_ag.set_aspect("equal")
        ax_ag.set_xlabel("Antigenic dimension 1", fontsize=11)
        ax_ag.set_ylabel("Antigenic dimension 2", fontsize=11)
        ax_ag.spines["top"].set_visible(False)
        ax_ag.spines["right"].set_visible(False)
    else:
        _blank(ax_ag, "tips unavailable")
    ax_ag.text(-0.1, 1.05, "C", transform=ax_ag.transAxes, fontsize=16, fontweight="bold")

    # Panel D: mean epitope mutations per variant over time.
    if tips is not None:
        variant_summary = (
            tips.groupby("variant_ag")
            .agg({"year": "mean", "epitopeMutationCount": "mean"})
            .reset_index()
        )
        sns.scatterplot(data=variant_summary, x="year", y="epitopeMutationCount",
                        hue="variant_ag", palette=variant_color_map, s=100, alpha=0.7,
                        ax=ax_epi, legend=False)
        ax_epi.axline((0, 10), slope=1.0, color="red", linestyle="--", alpha=0.7)
        ax_epi.set_xlabel("Time (years)", fontsize=11)
        ax_epi.set_ylabel("Epitope mutation count", fontsize=11)
        ax_epi.spines["top"].set_visible(False)
        ax_epi.spines["right"].set_visible(False)
    else:
        _blank(ax_epi, "tips unavailable")
    ax_epi.text(-0.1, 1.05, "D", transform=ax_epi.transAxes, fontsize=16, fontweight="bold")

    # Panel E: variant-frequency stackplot (60-day smoothed, from seq counts).
    if seqs is not None:
        freq_rows = []
        for date_val, date_data in seqs.groupby("date"):
            total = date_data["sequences"].sum()
            for variant, vdata in date_data.groupby("variant"):
                count = vdata["sequences"].sum()
                freq_rows.append({"date": date_val, "variant": variant,
                                  "frequency": count / total if total > 0 else 0})
        freq_df = pd.DataFrame(freq_rows)
        freq_pivot = freq_df.pivot(index="date", columns="variant",
                                   values="frequency").fillna(0)
        freq_daily = freq_pivot.resample("D").ffill()
        freq_smooth = freq_daily.rolling(window=60, center=True, min_periods=1).mean()
        freq_smooth = freq_smooth.div(freq_smooth.sum(axis=1), axis=0)

        start_date = freq_smooth.index.min()
        years = (freq_smooth.index - start_date).days / 365.25
        variants_reversed = list(reversed(freq_smooth.columns))
        ax_freq.stackplot(
            years,
            *[freq_smooth[var].values for var in variants_reversed],
            colors=[variant_color_map.get(var, "gray") for var in variants_reversed],
            alpha=0.9,
            linewidth=0,
        )

        freq_cumsum = freq_smooth[variants_reversed].cumsum(axis=1)
        for i, variant in enumerate(variants_reversed):
            vdata = freq_smooth[variant]
            significant = vdata > 0.05
            if significant.any() and vdata.max() > 0.1:
                sig_idx = vdata[significant].index
                start_t = (sig_idx[0] - start_date).days / 365.25
                end_t = (sig_idx[-1] - start_date).days / 365.25
                mid_year = (start_t + end_t) / 2
                mid_idx = sig_idx[len(sig_idx) // 2]
                y_bottom = 0 if i == 0 else (
                    freq_cumsum.loc[mid_idx, variant] - freq_smooth.loc[mid_idx, variant]
                )
                y_middle = y_bottom + freq_smooth.loc[mid_idx, variant] / 2
                ax_freq.text(mid_year, y_middle, str(variant), color="white",
                             fontweight="bold", fontsize=10, ha="center", va="center",
                             bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2))

        ax_freq.set_xlim(0, float(years.max()))
        ax_freq.set_ylim(0, 1)
        ax_freq.set_ylabel("Frequency", fontsize=11)
        ax_freq.set_xlabel("Time (years)", fontsize=11)
        ax_freq.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y * 100)}%"))
        ax_freq.spines["top"].set_visible(False)
        ax_freq.spines["right"].set_visible(False)
        ax_freq.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    else:
        _blank(ax_freq, "seq counts unavailable")
    ax_freq.text(-0.05, 1.05, "E", transform=ax_freq.transAxes, fontsize=16, fontweight="bold")

    fig.suptitle(run_title(row, missing), fontsize=10, ha="left", x=0.02, y=0.995,
                 va="top", family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def overview_page(candidates, figsize):
    """A leading page: the full candidate table for pre-scanning."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(f"{len(candidates)} candidate runs", fontsize=13, fontweight="bold", loc="left")

    table_df = candidates[OVERVIEW_COLS].copy()
    for col in table_df.columns:
        if table_df[col].dtype.kind == "f":
            table_df[col] = table_df[col].map(lambda v: f"{v:.2f}")
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1.0, 1.1)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="2026-07-04-reviewer-runs",
                        help="Batch name (default: %(default)s).")
    parser.add_argument("--candidates", type=Path,
                        default=REPO_ROOT / "candidate_runs.csv",
                        help="candidate_runs.csv path (default: %(default)s).")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Processed-data root (default: %(default)s).")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results",
                        help="Model-output root (default: %(default)s).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "browse_candidate_runs.pdf",
                        help="Output PDF path (default: %(default)s).")
    parser.add_argument("--sort", choices=["config", "tmrca", "antigenic_movement_per_year"],
                        default="config", help="Page ordering (default: %(default)s).")
    parser.add_argument("--filter", default=None,
                        help="Only render runs whose config name contains this substring.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(13.0, 14.0),
                        metavar=("W", "H"), help="Per-page figure size in inches.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    candidates = pd.read_csv(args.candidates)
    candidates["config"] = candidates["path"].map(
        lambda p: os.path.basename(os.path.dirname(p))
    )
    if args.filter:
        candidates = candidates[candidates["config"].str.contains(args.filter)]
        assert len(candidates) > 0, f"No candidates match --filter {args.filter!r}."

    sort_cols = ["config", "run"] if args.sort == "config" else [args.sort, "config", "run"]
    candidates = candidates.sort_values(sort_cols).reset_index(drop=True)

    figsize = tuple(args.figsize)
    print(f"Rendering {len(candidates)} candidate runs -> {args.out}")

    with PdfPages(args.out) as pdf:
        overview = overview_page(candidates, figsize)
        pdf.savefig(overview)
        plt.close(overview)

        for _, row in candidates.iterrows():
            label = f"{config_name(row)} run {int(row['run'])}"
            try:
                paths = resolve_paths(row, args.batch, args.data_root, args.results_root)
                inputs, missing = load_run_inputs(paths)
                if missing:
                    print(f"  [{label}] missing: {', '.join(missing)}")
                fig = plot_run_page(inputs, row, missing, figsize)
            except FileNotFoundError as err:
                # from_sim_path asserts the run dir exists; a stale/non-HPC path fires here.
                print(f"  [{label}] {err}")
                fig = plt.figure(figsize=figsize)
                fig.text(0.5, 0.5, f"{label}\n\n{err}", ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
