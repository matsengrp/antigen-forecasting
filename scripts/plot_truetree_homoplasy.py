"""True-genealogy homoplasy and variant-confusability figure (Reviewer 1, 1a/2a).

The inferred-tree analysis (``mutation_background_distances.py``) reconstructs
ancestral sequences and reports origin counts as upper bounds, because tree
inference splits one real mutation across sibling branches. antigen emits the
*true* genealogy with the exact parent and child nucleotide sequence on every
edge (``VirusTree.printBranches`` -> ``output/run-out.branches``), so here origins
are exact and no near-simultaneous filter is needed.

The figure is a single row:

- Panel A: cumulative distribution of the number of independent origins per
  amino-acid substitution, epitope vs non-epitope.
- Panels B-D: for every pair of independent origins of one substitution, the
  probability that the two origins' descendants were assigned to the same
  variant, against the amino-acid distance between the origins' genetic
  backgrounds, one panel per assignment method, each against a matched null of
  origin pairs drawn from different substitutions.

The core ``compute_truetree_tables`` is importable so the across-run sweep runs
the identical computation per run.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.Seq import Seq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_background_distances import (  # noqa: E402
    AA_TO_CODE,
    HA1_LENGTH,
    SKIP_RESIDUES,
    GeneLayout,
    is_epitope_mutation,
    load_epitope_sites,
    load_gene_layouts,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigentools.supplement_style import (  # noqa: E402
    SUPPLEMENT_RC,
    add_panel_letters,
    style_panel,
)

# Codon -> amino acid, built once so per-codon translation is a dict lookup.
CODON_TABLE = {
    "".join(c): str(Seq("".join(c)).translate()) for c in product("ACGT", repeat=3)
}

METHODS = ["variant_ag", "variant_tsne", "variant_phylo"]
METHOD_LABELS = {
    "variant_ag": "antigenic $k$-means",
    "variant_tsne": "sequence $t$-SNE",
    "variant_phylo": "phylogenetic",
}
METHOD_COLORS = {
    "variant_ag": "#B30000",
    "variant_tsne": "#3498db",
    "variant_phylo": "#2e8b57",
}
EPITOPE_COLOR = "#B30000"
NON_EPITOPE_COLOR = "#3498db"
NULL_COLOR = "#7f7f7f"

# Background-distance bins (amino acids). The first isolates near-identical
# backgrounds, where a genuine merge would happen; the top bin is open-ended.
BINS = [0, 1, 2, 4, 8, 16, 32, 1000]
N_NULL_PAIRS = 40000
SEED = 0


# --------------------------------------------------------------------------- #
# Parsing the true genealogy.
# --------------------------------------------------------------------------- #
def parse_record(record: str) -> tuple[str, bool, str]:
    """Return (name, is_tip, nucleotide_sequence) from one ``{...}`` node record.

    Field order from ``VirusTree.printBranches``: name, birth, fitness, trunk,
    tip, marked, deme, layout, then the phenotype (sequence, ag1, ag2, ...). The
    nucleotide sequence carries no commas, so a plain split is unambiguous through
    field 8.
    """
    fields = record[1:-1].split(",")  # Strip the braces.
    name = fields[0].strip().strip('"')
    is_tip = fields[4] == "1"
    nt_seq = fields[8]
    return name, is_tip, nt_seq


def codon_substitutions(parent_nt: str, child_nt: str, genes: dict[str, GeneLayout]):
    """Yield (gene, position, from_aa, to_aa) for the AA changes on one edge.

    Only codons whose nucleotides differ are translated; synonymous changes and
    mutations touching a gap, unknown, or stop residue are dropped, matching the
    inferred-tree convention.
    """
    for gene, layout in genes.items():
        p_gene = parent_nt[layout.nt_start : layout.nt_end]
        c_gene = child_nt[layout.nt_start : layout.nt_end]
        if p_gene == c_gene:
            continue
        for codon_i in range(layout.aa_length):
            lo = codon_i * 3
            p_codon = p_gene[lo : lo + 3]
            c_codon = c_gene[lo : lo + 3]
            if p_codon == c_codon:
                continue
            from_aa = CODON_TABLE.get(p_codon, "X")
            to_aa = CODON_TABLE.get(c_codon, "X")
            if from_aa == to_aa:
                continue
            if from_aa in SKIP_RESIDUES or to_aa in SKIP_RESIDUES:
                continue
            yield gene, codon_i + 1, from_aa, to_aa


def translate_full(nt: str, genes: dict[str, GeneLayout]) -> np.ndarray:
    """Translate a full nucleotide sequence into concatenated AA codes.

    Used only on origin edges (a small fraction of the tree), so per-codon
    dictionary translation is fast enough and avoids a Bio.Seq call per node.
    """
    codes: list[int] = []
    for gene, layout in genes.items():
        gene_nt = nt[layout.nt_start : layout.nt_end]
        for i in range(layout.aa_length):
            codon = gene_nt[i * 3 : i * 3 + 3]
            codes.append(AA_TO_CODE[CODON_TABLE.get(codon, "X")])
    return np.array(codes, dtype=np.int8)


def load_seq_to_variant(tips_variants_path: Path) -> dict[str, tuple]:
    """Map each unique nucleotide sequence to its (per-method) variant labels.

    ``tips_with_variants.tsv`` is the deduplicated sequence set, so every
    true-tree tip's variant labels are looked up by its sequence.
    """
    df = pd.read_csv(
        tips_variants_path, sep="\t", usecols=["nucleotideSequence"] + METHODS
    )
    return {
        row.nucleotideSequence: tuple(getattr(row, m) for m in METHODS)
        for row in df.itertuples(index=False)
    }


# --------------------------------------------------------------------------- #
# Streaming pass + postorder.
# --------------------------------------------------------------------------- #
def stream_branches(branches_path: Path, genes, seq_to_variant):
    """One streaming pass over the genealogy.

    Returns the tree structure (integer-coded parent and child-count arrays),
    per-tip variant codes, and, for every edge carrying an amino-acid change, the
    origin's substitution, origin node id, and parent-background AA sequence.
    """
    name2id: dict[str, int] = {}
    parent: list[int] = []
    n_children: list[int] = []
    is_tip: list[bool] = []
    tip_variant: dict[int, tuple] = {}

    # Per-method label -> small integer code, so postorder counts are cheap.
    label_code: dict[str, dict] = {m: {} for m in METHODS}

    def code_for(method: str, label) -> int:
        table = label_code[method]
        c = table.get(label)
        if c is None:
            c = len(table)
            table[label] = c
        return c

    def gid(name: str) -> int:
        i = name2id.get(name)
        if i is None:
            i = len(parent)
            name2id[name] = i
            parent.append(-1)
            n_children.append(0)
            is_tip.append(False)
        return i

    origins: list[tuple] = []  # (sub_key, origin_id, background_aa).
    origins_by_sub: dict[tuple, int] = defaultdict(int)
    n_edges = n_changed = n_tips = n_tip_missing = 0

    with open(branches_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            child_name, child_is_tip, child_nt = parse_record(parts[0])
            parent_name, _, parent_nt = parse_record(parts[1])
            cid, pid = gid(child_name), gid(parent_name)
            parent[cid] = pid
            n_children[pid] += 1
            n_edges += 1
            if child_is_tip:
                is_tip[cid] = True
                n_tips += 1
                labels = seq_to_variant.get(child_nt)
                if labels is None:
                    n_tip_missing += 1
                else:
                    tip_variant[cid] = tuple(
                        code_for(m, labels[k]) for k, m in enumerate(METHODS)
                    )

            if child_nt == parent_nt:
                continue
            n_changed += 1
            subs = list(codon_substitutions(parent_nt, child_nt, genes))
            if not subs:
                continue
            background = translate_full(parent_nt, genes)
            for sub in subs:
                origins.append((sub, cid, background))
                origins_by_sub[sub] += 1

    stats = {
        "n_edges": n_edges,
        "n_tips": n_tips,
        "n_changed": n_changed,
        "n_tip_missing": n_tip_missing,
    }
    tree = {"parent": parent, "n_children": n_children, "is_tip": is_tip}
    return origins, dict(origins_by_sub), tree, tip_variant, stats


def postorder_modal(tree, tip_variant, origin_ids: set[int]) -> dict[int, tuple]:
    """Assign each origin node the modal variant label of its descendant tips.

    Bounded-memory postorder (Kahn from the leaves): a node's per-method label
    counts are merged into its parent and then freed, so peak memory is the tree
    width (coexisting lineages), not the tree size.
    """
    parent = tree["parent"]
    is_tip = tree["is_tip"]
    n = len(parent)
    pending = tree["n_children"][:]  # Mutated as children are absorbed.
    counts: dict[int, list[Counter]] = {}
    modal: dict[int, tuple] = {}

    def node_counts(i: int) -> list[Counter]:
        c = counts.get(i)
        if c is None:
            c = [Counter(), Counter(), Counter()]
            counts[i] = c
        return c

    ready = [i for i in range(n) if pending[i] == 0]
    while ready:
        i = ready.pop()
        c = counts.get(i)
        if is_tip[i] and i in tip_variant:
            if c is None:
                c = [Counter(), Counter(), Counter()]
            codes = tip_variant[i]
            for k in range(3):
                c[k][codes[k]] += 1
        if i in origin_ids and c is not None:
            modal[i] = tuple(
                cc.most_common(1)[0][0] if cc else -1 for cc in c
            )
        p = parent[i]
        if p != -1 and c is not None:
            pc = node_counts(p)
            for k in range(3):
                pc[k].update(c[k])
        if p != -1:
            pending[p] -= 1
            if pending[p] == 0:
                ready.append(p)
        counts.pop(i, None)
    return modal


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #
def compute_truetree_tables(
    branches_path: Path,
    tips_variants_path: Path | None,
    genes,
    epitope_sites: set[int],
) -> dict:
    """Compute the origin-count and confusability tables for one genealogy.

    ``tips_variants_path`` may be None, in which case the confusability tables are
    empty (recurrence and origin counts still come back), mirroring the tiered
    guard in the inferred-tree sweep.
    """
    seq_to_variant = (
        load_seq_to_variant(tips_variants_path) if tips_variants_path else {}
    )
    origins, origins_by_sub, tree, tip_variant, stats = stream_branches(
        branches_path, genes, seq_to_variant
    )

    # Per-substitution origin counts -> the ECDF table.
    occ_rows = []
    for (gene, pos, _f, _t), n in origins_by_sub.items():
        occ_rows.append(
            {
                "n_independent_origins": n,
                "is_epitope": is_epitope_mutation(gene, pos, epitope_sites),
            }
        )
    occurrence = pd.DataFrame(occ_rows)

    pairs = pd.DataFrame()
    if seq_to_variant:
        origin_ids = {oid for _sub, oid, _bg in origins}
        modal = postorder_modal(tree, tip_variant, origin_ids)
        pairs = build_pairs(origins, modal, epitope_sites)

    return {
        "occurrence": occurrence,
        "pairs": pairs,
        "stats": stats,
    }


def build_pairs(origins, modal, epitope_sites) -> pd.DataFrame:
    """Same-substitution origin pairs and a matched different-substitution null."""
    by_sub: dict[tuple, list] = defaultdict(list)
    for sub, oid, bg in origins:
        by_sub[sub].append((oid, bg))

    rows = []

    def emit(kind, sub, a, b):
        oid_a, bg_a = a
        oid_b, bg_b = b
        va, vb = modal.get(oid_a), modal.get(oid_b)
        if va is None or vb is None:
            return
        row = {
            "kind": kind,
            "background_distance_aa": int(np.count_nonzero(bg_a != bg_b)),
        }
        for k, m in enumerate(METHODS):
            row[f"same_{m}"] = int(va[k] == vb[k])
        if kind == "same substitution":
            gene, pos = sub[0], sub[1]
            row["is_epitope"] = is_epitope_mutation(gene, pos, epitope_sites)
        rows.append(row)

    for sub, recs in by_sub.items():
        if len(recs) < 2:
            continue
        for a, b in combinations(recs, 2):
            emit("same substitution", sub, a, b)

    # Null: origins from different substitutions.
    flat = [(sub, oid, bg) for sub, recs in by_sub.items() for oid, bg in recs]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(flat), size=(N_NULL_PAIRS, 2))
    for i, j in idx:
        if i == j or flat[i][0] == flat[j][0]:
            continue
        emit("random origins", None, (flat[i][1], flat[i][2]), (flat[j][1], flat[j][2]))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Panels.
# --------------------------------------------------------------------------- #
def panel_occurrence(ax, occurrence: pd.DataFrame) -> None:
    """Panel A: cumulative fraction of substitutions with <= X independent origins."""
    max_occ = int(occurrence["n_independent_origins"].max())
    xs = np.arange(1, max_occ + 1)
    for is_epitope, color in [(True, EPITOPE_COLOR), (False, NON_EPITOPE_COLOR)]:
        subset = occurrence[occurrence["is_epitope"] == is_epitope]
        counts = subset["n_independent_origins"].to_numpy()
        n = len(counts)
        assert n > 0, f"no {'epitope' if is_epitope else 'non-epitope'} substitutions"
        frac = np.array([np.count_nonzero(counts <= x) / n for x in xs])
        ax.step(
            xs,
            frac,
            where="post",
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=f"{'epitope' if is_epitope else 'non-epitope'} ({n})",
        )
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(left=1)
    ax.set_ylim(top=1.02)
    ax.set_xlabel("Independent origins (X)")
    ax.set_ylabel("Fraction of substitutions $\\leq X$")
    ax.legend(loc="lower right")


def panel_confusability(ax, pairs: pd.DataFrame, method: str) -> None:
    """Panels B-D: P(same variant) against background distance, observed vs null."""
    labels = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        labels.append(f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}")
    labels[-1] = f"{BINS[-2]}+"
    binned = pd.cut(
        pairs["background_distance_aa"], bins=BINS, right=False, labels=labels
    )
    for kind, color, style in [
        ("same substitution", METHOD_COLORS[method], "-"),
        ("random origins", NULL_COLOR, "--"),
    ]:
        subset = pairs[pairs["kind"] == kind]
        grouped = subset.groupby(binned[subset.index], observed=False)[f"same_{method}"]
        rate = grouped.mean().reindex(labels).to_numpy()
        n = grouped.size().sum()
        ax.plot(
            range(len(labels)),
            rate,
            color=color,
            marker="o",
            ms=4,
            lw=1.8,
            linestyle=style,
            label=f"{kind} ({int(n)})",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Background distance (AA)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(METHOD_LABELS[method], fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def _bin_labels() -> list[str]:
    labels = [
        f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        for lo, hi in zip(BINS[:-1], BINS[1:])
    ]
    labels[-1] = f"{BINS[-2]}+"
    return labels


def panel_difference(ax, pairs: pd.DataFrame) -> None:
    """Panel B: same-substitution minus null co-assignment, per method.

    A flat line at zero means a shared substitution adds no co-assignment beyond
    background similarity; a positive value means it adds shared-ancestry signal.

    No per-bin confidence interval is drawn: origin pairs are not independent
    (pairs sharing an origin, or on nearby branches, are correlated), so an
    analytic interval would be optimistic. Uncertainty is shown instead by the
    spread of per-run curves in the pooled across-simulation figure.
    """
    labels = _bin_labels()
    binned = pd.cut(
        pairs["background_distance_aa"], bins=BINS, right=False, labels=labels
    )
    same = pairs[pairs["kind"] == "same substitution"]
    null = pairs[pairs["kind"] == "random origins"]
    same_bin, null_bin = binned[same.index], binned[null.index]
    x = np.arange(len(labels))
    for method in METHODS:
        diffs = []
        for lab in labels:
            s = same[same_bin == lab][f"same_{method}"]
            r = null[null_bin == lab][f"same_{method}"]
            if len(s) == 0 or len(r) == 0:
                diffs.append(np.nan)
                continue
            diffs.append(s.mean() - r.mean())
        ax.plot(
            x,
            diffs,
            color=METHOD_COLORS[method],
            marker="o",
            ms=4,
            lw=1.8,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Background distance (AA)")
    ax.set_ylabel("$\\Delta$ P(same variant), obs $-$ null")
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def build_figure(occurrence: pd.DataFrame, pairs: pd.DataFrame) -> plt.Figure:
    """The 1x2 row: origin-count ECDF plus the confusability difference panel.

    ``panel_confusability`` remains available for the per-method overlay framing;
    the committed figure uses the more compact difference panel.
    """
    assert not pairs.empty, "the difference panel needs variant labels"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    panel_occurrence(axes[0], occurrence)
    panel_difference(axes[1], pairs)
    for ax in axes:
        style_panel(ax)
        sns.despine(ax=ax)
    fig.tight_layout()
    add_panel_letters(fig, axes, "AB")
    return fig


# --------------------------------------------------------------------------- #
# Pooled across-run panels (S3C grammar: faint per-run curves, bold representative).
# --------------------------------------------------------------------------- #
def _is_representative(config, run, representative_run) -> bool:
    return (config == representative_run[0]) and (int(run) == representative_run[1])


def panel_occurrence_pooled(ax, origin_counts: pd.DataFrame, representative_run) -> None:
    """Pooled panel A: one faint origin-count ECDF per run and site class.

    The representative run is drawn bold, matching ``panel_genotype_antigenic`` in
    the inferred-tree figure; the faint lines show the spread across simulations.
    """
    classes = [("epitope", EPITOPE_COLOR), ("non_epitope", NON_EPITOPE_COLOR)]
    for (config, run), run_df in origin_counts.groupby(["config", "run"]):
        rep = _is_representative(config, run, representative_run)
        for name, color in classes:
            sub = run_df[run_df["site_class"] == name].sort_values("x")
            if sub.empty:
                continue
            x = sub["x"].to_numpy()
            y = sub["cumulative_fraction"].to_numpy()
            if rep:
                ax.step(
                    x, y, where="post", color=color, lw=2.6, marker="o", ms=4,
                    zorder=3, label=name.replace("_", "-"),
                )
            else:
                line = ax.step(
                    x, y, where="post", color=color, lw=0.6, alpha=0.15, zorder=1
                )[0]
                line.set_rasterized(True)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(left=1)
    ax.set_ylim(top=1.02)
    ax.set_xlabel("Independent origins (X)")
    ax.set_ylabel("Fraction of substitutions $\\leq X$")
    ax.legend(loc="lower right")


def panel_difference_pooled(ax, confusability: pd.DataFrame, representative_run) -> None:
    """Pooled panel B: one faint obs-minus-null curve per run and method.

    The representative run is bold. The spread of the faint curves is the honest
    uncertainty, replacing the single-run figure's (dropped) analytic error bars.
    """
    labels = _bin_labels()
    pos = {lab: i for i, lab in enumerate(labels)}
    for (config, run, method), g in confusability.groupby(["config", "run", "method"]):
        rep = _is_representative(config, run, representative_run)
        g = g.sort_values("bg_bin_low")
        x = [pos[lab] for lab in g["bg_bin"]]
        diff = (g["same_rate"] - g["null_rate"]).to_numpy()
        color = METHOD_COLORS[method]
        if rep:
            ax.plot(
                x, diff, color=color, lw=2.4, marker="o", ms=4, zorder=3,
                label=METHOD_LABELS[method],
            )
        else:
            line = ax.plot(x, diff, color=color, lw=0.6, alpha=0.15, zorder=1)[0]
            line.set_rasterized(True)
    ax.axhline(0, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Background distance (AA)")
    ax.set_ylabel("$\\Delta$ P(same variant), obs $-$ null")
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def build_pooled_figure(
    origin_counts: pd.DataFrame, confusability: pd.DataFrame, representative_run
) -> plt.Figure:
    """The across-run version of the S7 figure, rendered from the sweep tables."""
    assert not origin_counts.empty, "no origin-count rows to pool"
    assert not confusability.empty, "no confusability rows to pool"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    panel_occurrence_pooled(axes[0], origin_counts, representative_run)
    panel_difference_pooled(axes[1], confusability, representative_run)
    for ax in axes:
        style_panel(ax)
        sns.despine(ax=ax)
    fig.tight_layout()
    add_panel_letters(fig, axes, "AB")
    return fig


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--tips-with-variants", type=Path, required=True)
    parser.add_argument("--ref-genbank", type=Path, required=True)
    parser.add_argument("--epitope-sites", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="figureS7_truetree_homoplasy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genes = load_gene_layouts(args.ref_genbank)
    epitope_sites = load_epitope_sites(args.epitope_sites, HA1_LENGTH)

    tables = compute_truetree_tables(
        args.branches, args.tips_with_variants, genes, epitope_sites
    )
    occurrence, pairs, stats = (
        tables["occurrence"],
        tables["pairs"],
        tables["stats"],
    )
    print(
        f"edges {stats['n_edges']:,}  tips {stats['n_tips']:,}  "
        f"AA-changing edges {stats['n_changed']:,}  "
        f"tips missing a variant label {stats['n_tip_missing']:,}"
    )
    for is_epi, name in [(True, "epitope"), (False, "non-epitope")]:
        counts = occurrence[occurrence["is_epitope"] == is_epi][
            "n_independent_origins"
        ]
        rec = (counts >= 2).mean()
        print(
            f"  {name:12s} n={len(counts):5d}  recurrent={100 * rec:5.1f}%  "
            f"max_origins={int(counts.max())}"
        )
    for m in METHODS:
        real = pairs[pairs["kind"] == "same substitution"]
        ident = real[real["background_distance_aa"] == 0][f"same_{m}"].mean()
        far = real[real["background_distance_aa"] >= 8][f"same_{m}"].mean()
        print(f"  {m:14s} identical-background {ident:.1%}  >=8 AA {far:.1%}")

    with plt.rc_context(SUPPLEMENT_RC):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fig = build_figure(occurrence, pairs)
        for suffix in ("pdf", "png"):
            path = args.output_dir / f"{args.output_prefix}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
