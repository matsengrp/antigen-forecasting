"""Quantify how often each amino-acid mutation arises independently on a tree.

This script addresses Reviewer 1's comment 1a on the antigen-prime manuscript:
the concern that a given epitope substitution, assigned a random antigenic
direction per event, ought to converge antigenically across backgrounds.
Hugh's proposed rebuttal is quantitative: show that the same mutation rarely
arises independently in genetically *similar* backgrounds, so the random
direction is not a real problem.

Given the Auspice v2 tree for a simulation (which already carries per-branch
amino-acid substitutions in ``branch_attrs.mutations``), the script:

1. reconstructs the amino-acid sequence of every tree node by walking the tree
   from a translated anchor tip and applying the per-branch HA1/HA2 mutations;
2. indexes every amino-acid substitution onto the branches where it occurs, so
   that the number of *independent origins* (homoplasy count) is known;
3. for mutations with two or more independent origins, measures the amino-acid
   Hamming distance between the genetic backgrounds in which they arose; and
4. also computes, for every mutation, the mean pairwise amino-acid distance
   among all sampled tips carrying its derived residue (the coarser, tip-based
   metric requested alongside the origin-based one).

It writes one row per amino-acid mutation to a CSV.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

# Amino-acid alphabet used to encode reconstructed sequences as small integers.
# The trailing symbols cover stop codons, unknowns, and alignment gaps so that
# any residue emitted by translation or named in a mutation string is encodable.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*X-"
AA_TO_CODE = {aa: i for i, aa in enumerate(AA_ALPHABET)}
N_CODES = len(AA_ALPHABET)

# Residues that do not represent a real substitution event; mutations to or from
# any of these are skipped, matching the convention in scripts/add_new_clades.py.
SKIP_RESIDUES = frozenset({"-", "X", "*"})

# The reference tip that antigen's auspice export injects into the tree; it is
# not a sampled virus and its branch must not be counted as a mutation event.
REFERENCE_TIP = "seq0"

# Number of HA1 residues; epitope sites are defined in HA1 numbering.
HA1_LENGTH = 345

# Origins whose descendant clades are sampled within this many years of each
# other are treated as near-simultaneous: the signature of one real mutation
# split across nearby branches by phylogenetic inference rather than a genuine
# second independent origin. Defined here rather than in the plotting script so
# the tables and the figures cannot drift apart on what "near-simultaneous" means.
NEAR_SIMULTANEOUS_YEARS = 0.05


@dataclass
class Node:
    """A single node of the Auspice tree with its reconstructed sequence."""

    name: str
    children: list[str]
    parent: str | None
    # Parsed amino-acid substitutions on the branch leading into this node, as
    # a list of (gene, position, from_aa, to_aa) tuples.
    mutations: list[tuple[str, int, str, str]]
    aa: np.ndarray | None = None  # int8 codes, length == total protein length.
    progeny: int = 0  # Number of sampled (in-FASTA) descendant tips.


@dataclass
class GeneLayout:
    """Maps a gene name to its slice of the concatenated protein."""

    aa_offset: int
    aa_length: int
    nt_start: int  # 0-based inclusive.
    nt_end: int  # exclusive.


@dataclass
class Reconstruction:
    """The reconstructed tree plus the lookups derived from it."""

    nodes: dict[str, Node]
    root: str
    genes: dict[str, GeneLayout] = field(default_factory=dict)
    total_aa_length: int = 0

    def global_index(self, gene: str, position: int) -> int:
        """Return the concatenated-protein index for a 1-based gene position."""
        layout = self.genes[gene]
        assert 1 <= position <= layout.aa_length, (
            f"position {position} out of range for gene {gene} "
            f"(length {layout.aa_length})"
        )
        return layout.aa_offset + (position - 1)


def load_epitope_sites(path: Path, ha1_length: int) -> set[int]:
    """Load the 1-based HA1 epitope site positions.

    The shipped ``epitopeSites.txt`` for this study has a formatting quirk where
    two adjacent sites (189 and 190) are concatenated into the token ``189190``.
    We split any over-long numeric token into three-digit sites and assert that
    exactly the expected 49 Luksza and Lassig sites are recovered.
    """
    raw = path.read_text()
    tokens = [t for t in re.split(r"[,\s]+", raw.strip()) if t]
    sites: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"non-numeric epitope-site token: {token!r}")
        if len(token) <= 3 and int(token) <= ha1_length:
            sites.append(int(token))
            continue
        # Split a run of concatenated three-digit positions (e.g. 189190).
        assert len(token) % 3 == 0, f"cannot split epitope token {token!r}"
        parts = [int(token[i : i + 3]) for i in range(0, len(token), 3)]
        assert all(1 <= p <= ha1_length for p in parts), (
            f"split of {token!r} produced out-of-range sites {parts}"
        )
        sites.extend(parts)
    unique = sorted(set(sites))
    assert len(unique) == 49, f"expected 49 epitope sites, got {len(unique)}"
    assert unique == sites, "epitope-site list contained duplicates"
    return set(unique)


def load_gene_layouts(genbank_path: Path) -> dict[str, GeneLayout]:
    """Read CDS features and lay the genes out end-to-end in the protein."""
    record = SeqIO.read(str(genbank_path), "genbank")
    cds = sorted(
        (f for f in record.features if f.type == "CDS"),
        key=lambda f: int(f.location.start),
    )
    layouts: dict[str, GeneLayout] = {}
    aa_offset = 0
    for feature in cds:
        (gene,) = feature.qualifiers["gene"]
        start, end = int(feature.location.start), int(feature.location.end)
        assert (end - start) % 3 == 0, f"CDS {gene} length is not a multiple of 3"
        aa_length = (end - start) // 3
        layouts[gene] = GeneLayout(aa_offset, aa_length, start, end)
        aa_offset += aa_length
    assert layouts, f"no CDS features found in {genbank_path}"
    return layouts


def parse_mutations(
    branch_mutations: dict[str, list[str]], genes: dict[str, GeneLayout]
) -> list[tuple[str, int, str, str]]:
    """Parse the HA1/HA2 substitution strings on one branch.

    Each string is like ``F50I`` (from-residue, 1-based position, to-residue).
    Mutations at gaps/unknown/stop residues are skipped.
    """
    parsed: list[tuple[str, int, str, str]] = []
    for gene in genes:
        for mutation in branch_mutations.get(gene, []):
            from_aa, to_aa = mutation[0], mutation[-1]
            if from_aa in SKIP_RESIDUES or to_aa in SKIP_RESIDUES:
                continue
            position = int(mutation[1:-1])
            parsed.append((gene, position, from_aa, to_aa))
    return parsed


def load_tree(auspice_path: Path, genes: dict[str, GeneLayout]) -> Reconstruction:
    """Parse the Auspice JSON into a flat node dict with parsed mutations."""
    data = json.loads(auspice_path.read_text())
    nodes: dict[str, Node] = {}

    def visit(raw: dict, parent: str | None) -> None:
        name = raw["name"]
        assert name not in nodes, f"duplicate node name {name}"
        children = [c["name"] for c in raw.get("children", [])]
        mutations = parse_mutations(
            raw.get("branch_attrs", {}).get("mutations", {}), genes
        )
        nodes[name] = Node(name, children, parent, mutations)
        for child in raw.get("children", []):
            visit(child, name)

    visit(data["tree"], None)
    root = data["tree"]["name"]
    total = sum(layout.aa_length for layout in genes.values())
    return Reconstruction(nodes=nodes, root=root, genes=genes, total_aa_length=total)


def encode_aa(sequence: str) -> np.ndarray:
    """Encode an amino-acid string as an int8 array of alphabet codes."""
    return np.array([AA_TO_CODE[aa] for aa in sequence], dtype=np.int8)


def translate_tip(nt_sequence: str, recon: Reconstruction) -> np.ndarray:
    """Translate a nucleotide tip sequence into concatenated protein codes."""
    pieces: list[str] = []
    for gene, layout in recon.genes.items():
        protein = str(Seq(nt_sequence[layout.nt_start : layout.nt_end]).translate())
        assert len(protein) == layout.aa_length, (
            f"translation of {gene} gave {len(protein)} residues, "
            f"expected {layout.aa_length}"
        )
        pieces.append(protein)
    return encode_aa("".join(pieces))


def _apply(
    source: np.ndarray,
    mutations: list[tuple[str, int, str, str]],
    recon: Reconstruction,
    forward: bool,
) -> np.ndarray:
    """Apply (forward) or undo (inverse) branch mutations to a sequence copy."""
    result = source.copy()
    for gene, position, from_aa, to_aa in mutations:
        gi = recon.global_index(gene, position)
        expected, replacement = (from_aa, to_aa) if forward else (to_aa, from_aa)
        assert result[gi] == AA_TO_CODE[expected], (
            f"reconstruction conflict at {gene}:{position}: sequence has "
            f"{AA_ALPHABET[result[gi]]}, mutation implies {expected}"
        )
        result[gi] = AA_TO_CODE[replacement]
    return result


def reconstruct_sequences(
    recon: Reconstruction, anchor: str, anchor_aa: np.ndarray
) -> None:
    """Fill in every node's amino-acid sequence starting from one anchor tip.

    Traverses tree edges outward from the anchor, applying branch mutations
    forward (parent to child) or inverse (child to parent) as needed. Every
    application is checked against the branch annotation, so an inconsistent
    reconstruction fails loudly.
    """
    nodes = recon.nodes
    nodes[anchor].aa = anchor_aa
    stack = [anchor]
    while stack:
        name = stack.pop()
        node = nodes[name]
        assert node.aa is not None
        parent_name = node.parent
        if parent_name is not None and nodes[parent_name].aa is None:
            # Move to the parent by undoing this node's branch mutations.
            nodes[parent_name].aa = _apply(
                node.aa, node.mutations, recon, forward=False
            )
            stack.append(parent_name)
        for child_name in node.children:
            child = nodes[child_name]
            if child.aa is None:
                child.aa = _apply(node.aa, child.mutations, recon, forward=True)
                stack.append(child_name)
    missing = [n for n, node in nodes.items() if node.aa is None]
    assert not missing, (
        f"{len(missing)} nodes were not reconstructed, e.g. {missing[:5]}"
    )


def validate_against_fasta(recon: Reconstruction, tip_nt: dict[str, str]) -> None:
    """Assert every reconstructed tip matches its translated FASTA sequence."""
    mismatches: list[str] = []
    for name, nt_sequence in tip_nt.items():
        reconstructed = recon.nodes[name].aa
        assert reconstructed is not None
        if not np.array_equal(reconstructed, translate_tip(nt_sequence, recon)):
            mismatches.append(name)
    assert not mismatches, (
        f"{len(mismatches)} tips disagree with FASTA translation, e.g. {mismatches[:5]}"
    )


def compute_progeny(recon: Reconstruction, sampled: set[str]) -> None:
    """Count sampled descendant tips for every node via a postorder pass."""
    order: list[str] = []
    stack = [recon.root]
    while stack:
        name = stack.pop()
        order.append(name)
        stack.extend(recon.nodes[name].children)
    for name in reversed(order):
        node = recon.nodes[name]
        if not node.children:
            node.progeny = 1 if name in sampled else 0
        else:
            node.progeny = sum(recon.nodes[c].progeny for c in node.children)


def collect_descendant_tips(
    recon: Reconstruction, start: str, sampled: set[str]
) -> list[str]:
    """Return the sampled tip names in the subtree rooted at ``start``."""
    tips: list[str] = []
    stack = [start]
    while stack:
        name = stack.pop()
        node = recon.nodes[name]
        if not node.children:
            if name in sampled:
                tips.append(name)
        else:
            stack.extend(node.children)
    return tips


def index_origins(
    recon: Reconstruction,
) -> dict[tuple[str, int, str, str], list[str]]:
    """Map each amino-acid substitution to the child nodes where it arose.

    The branch leading to the injected reference tip is excluded, since that
    branch encodes reference divergence rather than a simulated mutation event.
    """
    origins: dict[tuple[str, int, str, str], list[str]] = {}
    for name, node in recon.nodes.items():
        if name == REFERENCE_TIP:
            continue
        for mutation in node.mutations:
            origins.setdefault(mutation, []).append(name)
    return origins


def sample_genotype_antigenic_pairs(
    recon: Reconstruction,
    tip_nt: dict[str, str],
    tip_antigenic: dict[str, np.ndarray],
    max_tips: int,
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Pair up sampled tips and measure genetic against antigenic distance.

    This is the direct test of the reviewer's operational concern, that similar
    viruses might sit anywhere in antigenic space. Antigenic position is the
    cumulative sum of every mutation along a lineage, so two genetically similar
    viruses share nearly all of their displacement regardless of the per-event
    random direction; the randomness perturbs one step rather than decoupling
    genotype from phenotype. Tips are subsampled before pairing because the pair
    count grows quadratically.

    Args:
        recon: Parsed tree, used only for its gene layout when translating.
        tip_nt: Tip name to nucleotide sequence.
        tip_antigenic: Tip name to ``(ag1, ag2)`` position.
        max_tips: Subsample this many tips before forming pairs.
        n_pairs: Number of distinct tip pairs to sample.
        seed: Seed for both subsampling steps.

    Returns:
        One row per sampled pair with ``genetic_distance_aa`` and
        ``antigenic_distance``.
    """
    shared = sorted(set(tip_nt) & set(tip_antigenic))
    assert len(shared) >= 2, "need at least two tips with sequence and antigenic data"
    rng = np.random.default_rng(seed)
    if len(shared) > max_tips:
        shared = [shared[i] for i in rng.choice(len(shared), max_tips, replace=False)]

    aa = np.stack([translate_tip(tip_nt[name], recon) for name in shared])
    ag = np.stack([tip_antigenic[name] for name in shared])

    rows, cols = np.triu_indices(len(shared), k=1)
    if len(rows) > n_pairs:
        keep = rng.choice(len(rows), n_pairs, replace=False)
        rows, cols = rows[keep], cols[keep]
    return pd.DataFrame(
        {
            "genetic_distance_aa": (aa[rows] != aa[cols]).sum(axis=1),
            "antigenic_distance": np.linalg.norm(ag[rows] - ag[cols], axis=1),
        }
    )


def identical_sequence_antigenic_spread(
    recon: Reconstruction, tips: pd.DataFrame
) -> pd.DataFrame:
    """Antigenic spread among tips that share an identical amino-acid sequence.

    The sharpest form of the reviewer's concern: if the random per-event direction
    genuinely decoupled genotype from antigenic phenotype, viruses with the *same*
    protein sequence would still be scattered in antigenic space.

    This must be given the **full** per-run ``tips.csv``, not ``unique_tips.csv``.
    The latter is deduplicated by sequence, so it collapses precisely the repeat
    structure being measured; running it there would instead compare synonymous
    variants of the same protein, which answers a different question. Both files
    contain the same set of distinct sequences, so nothing is lost by reading the
    larger one.

    Translation is done once per distinct nucleotide sequence rather than once per
    tip, which is roughly a twentyfold saving on a full tip table.

    Args:
        recon: Parsed tree, used only for its gene layout when translating.
        tips: Tip table with ``nucleotideSequence``, ``ag1``, and ``ag2`` columns.

    Returns:
        One row per amino-acid sequence carried by two or more tips, with
        ``n_tips`` and ``max_pairwise_spread`` (the group's diameter).
    """
    required = {"nucleotideSequence", "ag1", "ag2"}
    missing = required - set(tips.columns)
    assert not missing, f"tip table is missing column(s): {sorted(missing)}"
    usable = tips.dropna(subset=["nucleotideSequence", "ag1", "ag2"])
    assert not usable.empty, "no tips with both a sequence and an antigenic position"

    # Translate each distinct nucleotide sequence once, then group by the protein.
    translations = {
        sequence: translate_tip(sequence, recon).tobytes()
        for sequence in usable["nucleotideSequence"].unique()
    }
    keys = usable["nucleotideSequence"].map(translations)

    records = []
    for _, group in usable.groupby(keys, sort=False):
        if len(group) < 2:
            continue
        positions = group[["ag1", "ag2"]].to_numpy()
        # Diameter of the group, via the largest deviation from its centroid.
        centroid = positions.mean(axis=0)
        spread = float(np.linalg.norm(positions - centroid, axis=1).max() * 2)
        records.append({"n_tips": len(group), "max_pairwise_spread": spread})
    return pd.DataFrame(records, columns=["n_tips", "max_pairwise_spread"])


def is_epitope_mutation(gene: str, position: int, epitope_sites: set[int]) -> bool:
    """Report whether a mutation falls at an epitope site.

    Epitope sites are defined in HA1 numbering, so only HA1 positions qualify.
    """
    return gene == "HA1" and position in epitope_sites


def label_subtree_intervals(recon: Reconstruction) -> dict[str, tuple[int, int]]:
    """Label every node with the pre-order interval spanned by its subtree.

    Node ``a`` is a strict descendant of node ``b`` exactly when
    ``enter[b] < enter[a] <= exit[b]``, which turns an ancestry test into two
    integer comparisons instead of a walk up the tree. The traversal is iterative
    because simulated trees are deep enough to exhaust the recursion limit.
    """
    intervals: dict[str, tuple[int, int]] = {}
    entered: dict[str, int] = {}
    counter = 0
    # Each entry is (node name, whether its children have already been queued).
    stack: list[tuple[str, bool]] = [(recon.root, False)]
    while stack:
        name, expanded = stack.pop()
        if expanded:
            # Every descendant has been entered by now, so the last number handed
            # out is the high end of this subtree's interval.
            intervals[name] = (entered[name], counter - 1)
            continue
        entered[name] = counter
        counter += 1
        stack.append((name, True))
        for child in reversed(recon.nodes[name].children):
            stack.append((child, False))
    assert len(intervals) == len(recon.nodes), (
        f"labelled {len(intervals)} nodes but the tree holds {len(recon.nodes)}; "
        "the tree is not fully connected to its root"
    )
    return intervals


def is_strict_descendant(
    intervals: dict[str, tuple[int, int]], candidate: str, ancestor: str
) -> bool:
    """Report whether ``candidate`` lies strictly below ``ancestor``."""
    candidate_enter, _ = intervals[candidate]
    ancestor_enter, ancestor_exit = intervals[ancestor]
    return ancestor_enter < candidate_enter <= ancestor_exit


def find_reversion_pairs(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
) -> list[tuple[tuple[str, int, str, str], str, str]]:
    """Find every gain-then-loss of the same substitution along a single lineage.

    Returns ``(substitution, forward origin, reverse origin)`` triples where the
    reverse substitution arises on a branch strictly below the branch carrying the
    forward substitution. This is the case Reviewer 1 raised: because
    ``antigen-prime`` draws a fresh antigenic vector per mutation event, acquiring
    a mutation and then losing it does not return the virus to its starting
    antigenic position. Reverse substitutions that merely occur somewhere else on
    the tree are a weaker, more common pattern and are counted separately by
    :func:`summarize_reversions`.
    """
    intervals = label_subtree_intervals(recon)
    pairs: list[tuple[tuple[str, int, str, str], str, str]] = []
    for substitution, forward_nodes in origins.items():
        gene, position, from_aa, to_aa = substitution
        reverse_nodes = origins.get((gene, position, to_aa, from_aa))
        if reverse_nodes is None:
            continue
        for forward in forward_nodes:
            for reverse in reverse_nodes:
                if is_strict_descendant(intervals, reverse, forward):
                    pairs.append((substitution, forward, reverse))
    return pairs


def summarize_reversions(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    epitope_sites: set[int],
) -> dict[str, int]:
    """Count reversions two ways, split by site class.

    ``reverse_anywhere`` counts substitutions whose exact reverse occurs anywhere
    on the tree, which is an upper bound. ``lineage_cycle`` counts only those with
    a reverse origin strictly below a forward origin, which is the actual
    acquire-then-lose case.

    Counts are reported both *directed* and *unordered*, because they differ and
    the distinction is easy to misquote. A substitution and its reverse are two
    directed keys but one unordered pair, so on ``flu-final`` the 54 directed
    reverse-present substitutions are 27 distinct reversible pairs. Directed
    counts share a denominator with the directed substitution total; unordered
    counts do not, so mixing them would overstate the rate twofold.
    """
    pairs = find_reversion_pairs(recon, origins)
    cycle_substitutions = {substitution for substitution, _, _ in pairs}

    def unordered(keys: set[tuple[str, int, str, str]]) -> int:
        return len(
            {
                frozenset(
                    [(gene, position, from_aa, to_aa), (gene, position, to_aa, from_aa)]
                )
                for gene, position, from_aa, to_aa in keys
            }
        )

    reverse_present = {
        key for key in origins if (key[0], key[1], key[3], key[2]) in origins
    }
    counts = {
        "n_reverse_anywhere_epitope": 0,
        "n_reverse_anywhere_non_epitope": 0,
        "n_lineage_cycle_epitope": 0,
        "n_lineage_cycle_non_epitope": 0,
        "n_reverse_anywhere_unordered_pairs": unordered(reverse_present),
        "n_lineage_cycle_unordered_pairs": unordered(cycle_substitutions),
        "n_lineage_cycle_origin_pairs": len(pairs),
    }
    for substitution in origins:
        gene, position, from_aa, to_aa = substitution
        if (gene, position, to_aa, from_aa) not in origins:
            continue
        if is_epitope_mutation(gene, position, epitope_sites):
            suffix = "epitope"
        else:
            suffix = "non_epitope"
        counts[f"n_reverse_anywhere_{suffix}"] += 1
        if substitution in cycle_substitutions:
            counts[f"n_lineage_cycle_{suffix}"] += 1
    return counts


def filter_origins_by_progeny(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    min_progeny: int,
) -> dict[tuple[str, int, str, str], list[str]]:
    """Keep only origins whose child-node clade has at least ``min_progeny`` tips.

    Restricting to origins that led to a large clade drops mutations that arose
    and died off quickly (transient or spurious events) and removes the small
    fragment of a tree-inference split, where one real origin is divided across
    two branches and only one carries the bulk of the progeny. Mutations left
    with no surviving origin are removed entirely.
    """
    filtered: dict[tuple[str, int, str, str], list[str]] = {}
    for mutation, origin_nodes in origins.items():
        kept = [o for o in origin_nodes if recon.nodes[o].progeny >= min_progeny]
        if kept:
            filtered[mutation] = kept
    return filtered


def origin_background_distances(
    recon: Reconstruction, origin_nodes: list[str]
) -> list[int]:
    """Amino-acid Hamming distance between every pair of origin backgrounds.

    The background of an origin is its parent node's reconstructed sequence: the
    genetic context the mutation arose in.
    """
    backgrounds = []
    for origin in origin_nodes:
        parent = recon.nodes[origin].parent
        assert parent is not None, f"origin {origin} has no parent background"
        background = recon.nodes[parent].aa
        assert background is not None, f"background of {origin} was not reconstructed"
        backgrounds.append(background)
    return [int(np.count_nonzero(a != b)) for a, b in combinations(backgrounds, 2)]


def mean_pairwise_background_distance(
    recon: Reconstruction, origin_nodes: list[str]
) -> float:
    """Mean amino-acid Hamming distance between the origins' parent sequences."""
    return float(np.mean(origin_background_distances(recon, origin_nodes)))


def min_pairwise_background_distance(
    recon: Reconstruction, origin_nodes: list[str]
) -> float:
    """Smallest amino-acid Hamming distance between any two origin backgrounds.

    Reported alongside the mean because the two answer different questions and
    disagree sharply. "Did this mutation ever arise twice in near-identical
    backgrounds?" is the minimum, and on ``flu-final`` the minimum is far more
    permissive than the mean (33.5% of recurrent epitope substitutions have two
    origins within 1 residue, versus 1.1% by the mean). Every one of those cases
    involves an origin with almost no surviving progeny, so the two statistics
    must be read together with the progeny filter rather than one being chosen
    silently.
    """
    return float(np.min(origin_background_distances(recon, origin_nodes)))


def mean_pairwise_hamming(
    submatrix: np.ndarray, max_rows: int, rng: np.random.Generator
) -> float:
    """Mean pairwise amino-acid Hamming distance over a set of sequences.

    Computed from per-column allele frequencies in O(rows x length) rather than
    over all pairs. Sets larger than ``max_rows`` are subsampled for speed; the
    mean pairwise distance is well estimated from a sample.
    """
    rows = submatrix.shape[0]
    if rows > max_rows:
        submatrix = submatrix[rng.choice(rows, max_rows, replace=False)]
        rows = max_rows
    differing_pairs = 0.0
    for allele in range(N_CODES):
        counts = (submatrix == allele).sum(axis=0).astype(np.float64)
        differing_pairs += float(np.sum(counts * (rows - counts)))
    # Each unordered differing pair is counted once per column above (via the
    # symmetric split), so divide by the number of unordered sequence pairs.
    return differing_pairs / (rows * (rows - 1))


def build_tip_matrix(recon: Reconstruction, tip_names: list[str]) -> np.ndarray:
    """Stack sampled tip sequences into an (n_tips, length) code matrix."""
    rows = []
    for name in tip_names:
        aa = recon.nodes[name].aa
        assert aa is not None, f"tip {name} was not reconstructed"
        rows.append(aa)
    return np.stack(rows)


def build_rows(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    epitope_sites: set[int],
    tip_names: list[str],
    tip_matrix: np.ndarray,
    tip_years: dict[str, float] | None,
    sampled: set[str],
    max_carriers: int,
    seed: int,
    tip_antigenic: dict[str, np.ndarray] | None,
    tip_epi_counts: dict[str, int] | None,
) -> pd.DataFrame:
    """Assemble the per-mutation output table."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for (gene, position, from_aa, to_aa), origin_nodes in origins.items():
        gi = recon.global_index(gene, position)
        is_epitope = is_epitope_mutation(gene, position, epitope_sites)
        n_origins = len(origin_nodes)
        progenies = [recon.nodes[o].progeny for o in origin_nodes]

        # The mean and the minimum answer different questions -- "how different
        # are the backgrounds this mutation arises in?" versus "did it ever arise
        # twice in near-identical backgrounds?" -- and disagree sharply, so both
        # are carried rather than one being picked here.
        if n_origins >= 2:
            pairwise = origin_background_distances(recon, origin_nodes)
            origin_distance = float(np.mean(pairwise))
            min_origin_distance = float(np.min(pairwise))
        else:
            origin_distance = np.nan
            min_origin_distance = np.nan

        # Per-mutation antigenic scalar: mean pairwise distance between the
        # origins' antigenic positions (see build_origin_pairs for the primary,
        # per-pair form conditioned on background). Only defined for mutations
        # with antigenic data on at least two origins.
        antigenic_distance = np.nan
        if tip_antigenic is not None and tip_epi_counts is not None and n_origins >= 2:
            ag_positions = []
            for origin in origin_nodes:
                result = origin_antigenic_position(
                    recon, origin, sampled, tip_antigenic, tip_epi_counts
                )
                if result is not None:
                    ag_positions.append(result[0])
            if len(ag_positions) >= 2:
                antigenic_distance = mean_pairwise_antigenic_distance(ag_positions)

        carrier_mask = tip_matrix[:, gi] == AA_TO_CODE[to_aa]
        n_carriers = int(carrier_mask.sum())
        if n_carriers >= 2:
            carrier_distance = mean_pairwise_hamming(
                tip_matrix[carrier_mask], max_carriers, rng
            )
        else:
            carrier_distance = np.nan

        min_time_between_origins = np.nan
        if tip_years is not None and n_origins >= 2:
            origin_years = []
            for origin in origin_nodes:
                years = [
                    tip_years[t]
                    for t in collect_descendant_tips(recon, origin, sampled)
                    if t in tip_years
                ]
                if years:
                    origin_years.append(float(np.mean(years)))
            if len(origin_years) >= 2:
                min_time_between_origins = min(
                    abs(a - b) for a, b in combinations(origin_years, 2)
                )

        records.append(
            {
                "gene": gene,
                "position": position,
                "from_aa": from_aa,
                "to_aa": to_aa,
                "mutation_label": f"{gene}:{from_aa}{position}{to_aa}",
                "is_epitope": is_epitope,
                "n_independent_origins": n_origins,
                "n_carrier_tips": n_carriers,
                "total_progeny": int(sum(progenies)),
                "max_progeny": int(max(progenies)),
                "mean_origin_background_distance_aa": origin_distance,
                "min_origin_background_distance_aa": min_origin_distance,
                "mean_carrier_tip_distance_aa": carrier_distance,
                "mean_origin_antigenic_distance": antigenic_distance,
                "min_time_between_origins": min_time_between_origins,
            }
        )

    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(
        ["n_independent_origins", "total_progeny"], ascending=False
    ).reset_index(drop=True)


def sample_null_background_distances(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample AA distances between random pairs of distinct mutation origins.

    The pool is every occurrence branch's background (its parent node), taken
    across all mutations. Comparing the same-mutation origin distances against
    this null answers the reviewer's concern directly: if two origins of the
    *same* mutation are no closer than two origins of *different* mutations,
    then a mutation does not tend to recur in similar backgrounds.
    """
    pool = []
    for origin_nodes in origins.values():
        for origin in origin_nodes:
            parent = recon.nodes[origin].parent
            assert parent is not None
            background = recon.nodes[parent].aa
            assert background is not None
            pool.append(background)
    matrix = np.stack(pool)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(matrix), n_samples)
    right = rng.integers(0, len(matrix), n_samples)
    keep = left != right
    return np.count_nonzero(matrix[left[keep]] != matrix[right[keep]], axis=1)


def origin_antigenic_position(
    recon: Reconstruction,
    origin: str,
    sampled: set[str],
    tip_antigenic: dict[str, np.ndarray],
    tip_epi_counts: dict[str, int],
) -> tuple[np.ndarray, int] | None:
    """Antigenic position of an origin from its minimal-epitope-count descendants.

    In this model only epitope mutations move the antigenic phenotype, so the
    descendant tips at the *minimum* epitope-mutation count in the origin's clade
    added no further epitope mutations below the origin and therefore sit at the
    origin's exact post-mutation antigenic phenotype. Averaging ``ag`` over just
    those tips is unbiased and near-zero variance, avoiding the downstream-drift
    bias of averaging over all descendants. Returns the (2,) mean position and
    that minimum epitope count, or ``None`` when no descendant carries antigenic
    data.
    """
    tips = [
        t for t in collect_descendant_tips(recon, origin, sampled) if t in tip_antigenic
    ]
    if not tips:
        return None
    min_count = min(tip_epi_counts[t] for t in tips)
    selected = [t for t in tips if tip_epi_counts[t] == min_count]
    position = np.stack([tip_antigenic[t] for t in selected]).mean(axis=0)
    return position, min_count


def mean_pairwise_antigenic_distance(positions: list[np.ndarray]) -> float:
    """Mean pairwise Euclidean distance among 2D antigenic positions."""
    distances = [float(np.linalg.norm(a - b)) for a, b in combinations(positions, 2)]
    return float(np.mean(distances))


ORIGIN_PAIR_COLUMNS = [
    "mutation_label",
    "origin_a",
    "origin_b",
    "genetic_background_distance_aa",
    "antigenic_distance",
    "epitope_count_diff",
    "time_between_origins",
    "min_progeny_of_pair",
]


@dataclass
class OriginInfo:
    """Per-origin quantities needed to form an origin-pair row."""

    origin: str
    ag: np.ndarray  # (2,) antigenic position.
    epi_count: int
    background: np.ndarray  # Parent-node amino-acid codes.
    progeny: int
    year: float  # Mean descendant sampling year, or NaN.


def describe_origins(
    recon: Reconstruction,
    origin_nodes: list[str],
    sampled: set[str],
    tip_antigenic: dict[str, np.ndarray],
    tip_epi_counts: dict[str, int],
    tip_years: dict[str, float] | None,
) -> list[OriginInfo]:
    """Collect antigenic position, background, and time for each usable origin."""
    described: list[OriginInfo] = []
    for origin in origin_nodes:
        result = origin_antigenic_position(
            recon, origin, sampled, tip_antigenic, tip_epi_counts
        )
        if result is None:
            continue
        ag_position, epi_count = result
        parent = recon.nodes[origin].parent
        assert parent is not None, f"origin {origin} has no parent background"
        background = recon.nodes[parent].aa
        assert background is not None, f"background of {origin} not reconstructed"
        if tip_years is not None:
            years = [
                tip_years[t]
                for t in collect_descendant_tips(recon, origin, sampled)
                if t in tip_years
            ]
        else:
            years = []
        described.append(
            OriginInfo(
                origin=origin,
                ag=ag_position,
                epi_count=epi_count,
                background=background,
                progeny=recon.nodes[origin].progeny,
                year=float(np.mean(years)) if years else float("nan"),
            )
        )
    return described


def build_origin_pairs(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    epitope_sites: set[int],
    sampled: set[str],
    tip_antigenic: dict[str, np.ndarray],
    tip_epi_counts: dict[str, int],
    tip_years: dict[str, float] | None,
) -> pd.DataFrame:
    """One row per pair of antigenic-bearing origins of each epitope mutation.

    The antigenic distance between two origins is only interpretable once we
    condition on their genetic background distance (antigenic position drifts
    with cumulative epitope count), so this emits the raw per-pair quantities and
    leaves the conditioning to the figure.
    """
    records: list[dict[str, object]] = []
    for (gene, position, from_aa, to_aa), origin_nodes in origins.items():
        if not is_epitope_mutation(gene, position, epitope_sites):
            continue
        label = f"{gene}:{from_aa}{position}{to_aa}"
        described = describe_origins(
            recon, origin_nodes, sampled, tip_antigenic, tip_epi_counts, tip_years
        )
        for a, b in combinations(described, 2):
            if np.isnan(a.year) or np.isnan(b.year):
                time_between = float("nan")
            else:
                time_between = abs(a.year - b.year)
            records.append(
                {
                    "mutation_label": label,
                    "origin_a": a.origin,
                    "origin_b": b.origin,
                    "genetic_background_distance_aa": int(
                        np.count_nonzero(a.background != b.background)
                    ),
                    "antigenic_distance": float(np.linalg.norm(a.ag - b.ag)),
                    "epitope_count_diff": int(abs(a.epi_count - b.epi_count)),
                    "time_between_origins": time_between,
                    "min_progeny_of_pair": int(min(a.progeny, b.progeny)),
                }
            )
    return pd.DataFrame.from_records(records, columns=ORIGIN_PAIR_COLUMNS)


def sample_null_antigenic_distances(
    recon: Reconstruction,
    origins: dict[tuple[str, int, str, str], list[str]],
    epitope_sites: set[int],
    sampled: set[str],
    tip_antigenic: dict[str, np.ndarray],
    tip_epi_counts: dict[str, int],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Random pairs of epitope-mutation origins, carrying both distance axes.

    Emitting each null pair's genetic background distance alongside its antigenic
    distance lets the figure form both a pooled null (all rows) and a
    background-matched null (rows binned by genetic distance), so "closer than
    random?" can be judged within the similar-background regime.
    """
    pool_ag: list[np.ndarray] = []
    pool_bg: list[np.ndarray] = []
    pool_count: list[int] = []
    for (gene, position, from_aa, to_aa), origin_nodes in origins.items():
        if not is_epitope_mutation(gene, position, epitope_sites):
            continue
        for origin in origin_nodes:
            result = origin_antigenic_position(
                recon, origin, sampled, tip_antigenic, tip_epi_counts
            )
            if result is None:
                continue
            ag_position, epi_count = result
            parent = recon.nodes[origin].parent
            assert parent is not None
            background = recon.nodes[parent].aa
            assert background is not None
            pool_ag.append(ag_position)
            pool_bg.append(background)
            pool_count.append(epi_count)
    assert pool_ag, "no epitope origins with antigenic data for the null"
    ag_matrix = np.stack(pool_ag)
    bg_matrix = np.stack(pool_bg)
    counts = np.array(pool_count)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(ag_matrix), n_samples)
    right = rng.integers(0, len(ag_matrix), n_samples)
    keep = left != right
    left, right = left[keep], right[keep]
    return pd.DataFrame(
        {
            "antigenic_distance": np.linalg.norm(
                ag_matrix[left] - ag_matrix[right], axis=1
            ),
            "genetic_background_distance_aa": np.count_nonzero(
                bg_matrix[left] != bg_matrix[right], axis=1
            ),
            "epitope_count_diff": np.abs(counts[left] - counts[right]),
        }
    )


def load_tip_sequences(fasta_path: Path) -> dict[str, str]:
    """Read the sampled tip nucleotide sequences from the alignment FASTA."""
    tips = {r.id: str(r.seq) for r in SeqIO.parse(str(fasta_path), "fasta")}
    lengths = {len(s) for s in tips.values()}
    assert len(lengths) == 1, f"FASTA sequences are not aligned: lengths {lengths}"
    return tips


def load_tip_years(tips_csv_path: Path) -> dict[str, float]:
    """Read decimal-year sampling times keyed by tip name from unique_tips.csv."""
    frame = pd.read_csv(tips_csv_path, usecols=["name", "year"])
    return dict(zip(frame["name"], frame["year"].astype(float)))


def load_tip_antigenic(
    tips_csv_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Read per-tip antigenic coordinates and epitope-mutation counts.

    Returns ``(positions, epi_counts)`` where ``positions`` maps each tip name to
    its ``(ag1, ag2)`` antigenic coordinate and ``epi_counts`` maps it to the
    number of epitope mutations on its lineage (used to place origins at their
    minimal-count post-mutation antigenic phenotype).
    """
    required = ["name", "ag1", "ag2", "epitopeMutationCount"]
    header = pd.read_csv(tips_csv_path, nrows=0)
    missing = [c for c in required if c not in header.columns]
    assert not missing, f"tips CSV {tips_csv_path} missing columns: {missing}"
    frame = pd.read_csv(tips_csv_path, usecols=required)
    positions = {
        name: np.array([ag1, ag2], dtype=np.float64)
        for name, ag1, ag2 in zip(frame["name"], frame["ag1"], frame["ag2"])
    }
    epi_counts = {
        name: int(count)
        for name, count in zip(frame["name"], frame["epitopeMutationCount"])
    }
    return positions, epi_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auspice-json", type=Path, required=True)
    parser.add_argument("--sequences-fasta", type=Path, required=True)
    parser.add_argument("--ref-genbank", type=Path, required=True)
    parser.add_argument("--epitope-sites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tips-csv",
        type=Path,
        default=None,
        help="Optional unique_tips.csv for per-origin sampling-time separation.",
    )
    parser.add_argument(
        "--anchor-tip",
        type=str,
        default="45b125db",
        help="Sampled tip used to anchor sequence reconstruction.",
    )
    parser.add_argument(
        "--max-carriers-for-distance",
        type=int,
        default=1500,
        help="Cap on carrier tips used when estimating tip-based distance.",
    )
    parser.add_argument(
        "--min-origin-progeny",
        type=int,
        default=1,
        help=(
            "Keep only origins whose clade has at least this many sampled tips; "
            "restricting to established origins drops transients and tree-split "
            "fragments. Default 1 keeps every origin."
        ),
    )
    parser.add_argument(
        "--null-output",
        type=Path,
        default=None,
        help="Optional CSV of AA distances between random origin-background pairs.",
    )
    parser.add_argument(
        "--null-samples",
        type=int,
        default=20000,
        help="Number of random background pairs to sample for the null.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV of per-origin-pair antigenic vs genetic distances for "
            "epitope mutations (requires --tips-csv)."
        ),
    )
    parser.add_argument(
        "--antigenic-null-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV of antigenic distances between random epitope-origin "
            "pairs, with their genetic distance (requires --tips-csv)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def compute_tables(
    auspice_json: Path,
    sequences_fasta: Path,
    ref_genbank: Path,
    epitope_sites_path: Path,
    tips_csv: Path | None,
    anchor_tip: str,
    min_origin_progeny: int,
    max_carriers: int,
    null_samples: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Reconstruct the tree and build every output table used by the figures.

    Returns a dict with ``"mutations"`` (per-mutation table), ``"background_null"``
    (random origin-background distances), ``"pairs"`` (per-origin-pair antigenic vs
    genetic distances for epitope mutations), and ``"antigenic_null"`` (random
    epitope-origin antigenic distances). The antigenic tables are empty when
    ``tips_csv`` is not given. Both ``main`` and the reproducibility notebook call
    this so their figures come from identical data.
    """
    genes = load_gene_layouts(ref_genbank)
    epitope_sites = load_epitope_sites(epitope_sites_path, HA1_LENGTH)
    tip_nt = load_tip_sequences(sequences_fasta)
    recon = load_tree(auspice_json, genes)

    assert anchor_tip in recon.nodes, f"anchor tip {anchor_tip} is not in the tree"
    assert anchor_tip in tip_nt, f"anchor tip {anchor_tip} is not in the FASTA"
    reconstruct_sequences(recon, anchor_tip, translate_tip(tip_nt[anchor_tip], recon))
    validate_against_fasta(recon, tip_nt)

    sampled = set(tip_nt)
    compute_progeny(recon, sampled)
    origins = index_origins(recon)
    if min_origin_progeny > 1:
        n_origins_before = sum(len(v) for v in origins.values())
        origins = filter_origins_by_progeny(recon, origins, min_origin_progeny)
        n_origins_after = sum(len(v) for v in origins.values())
        print(
            f"Progeny filter (>= {min_origin_progeny} tips): kept "
            f"{n_origins_after}/{n_origins_before} origins across "
            f"{len(origins)} mutations."
        )

    tip_names = sorted(sampled)
    tip_matrix = build_tip_matrix(recon, tip_names)
    tip_years = load_tip_years(tips_csv) if tips_csv is not None else None
    if tips_csv is not None:
        tip_antigenic, tip_epi_counts = load_tip_antigenic(tips_csv)
    else:
        tip_antigenic, tip_epi_counts = None, None

    mutations = build_rows(
        recon,
        origins,
        epitope_sites,
        tip_names,
        tip_matrix,
        tip_years,
        sampled,
        max_carriers,
        seed,
        tip_antigenic,
        tip_epi_counts,
    )

    # Report the number of unique amino-acid mutations per site class, and how
    # many of those recur, so the categories are documented alongside the table.
    for is_epitope, name in [(True, "epitope"), (False, "non-epitope")]:
        class_frame = mutations[mutations["is_epitope"] == is_epitope]
        class_recurrent = int((class_frame["n_independent_origins"] >= 2).sum())
        print(
            f"  {name}: {len(class_frame)} unique AA mutations "
            f"({class_recurrent} with >=2 independent origins)."
        )

    background_null = pd.DataFrame(
        {
            "background_distance_aa": sample_null_background_distances(
                recon, origins, null_samples, seed
            )
        }
    )

    if tip_antigenic is not None and tip_epi_counts is not None:
        pairs = build_origin_pairs(
            recon,
            origins,
            epitope_sites,
            sampled,
            tip_antigenic,
            tip_epi_counts,
            tip_years,
        )
        antigenic_null = sample_null_antigenic_distances(
            recon,
            origins,
            epitope_sites,
            sampled,
            tip_antigenic,
            tip_epi_counts,
            null_samples,
            seed,
        )
    else:
        pairs = pd.DataFrame(columns=ORIGIN_PAIR_COLUMNS)
        antigenic_null = pd.DataFrame(
            columns=[
                "antigenic_distance",
                "genetic_background_distance_aa",
                "epitope_count_diff",
            ]
        )

    return {
        "mutations": mutations,
        "background_null": background_null,
        "pairs": pairs,
        "antigenic_null": antigenic_null,
    }


def main() -> None:
    args = parse_args()

    tables = compute_tables(
        args.auspice_json,
        args.sequences_fasta,
        args.ref_genbank,
        args.epitope_sites,
        args.tips_csv,
        args.anchor_tip,
        args.min_origin_progeny,
        args.max_carriers_for_distance,
        args.null_samples,
        args.seed,
    )

    mutations = tables["mutations"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mutations.to_csv(args.output, index=False)
    recurrent = int((mutations["n_independent_origins"] >= 2).sum())
    print(
        f"Wrote {len(mutations)} mutations to {args.output} "
        f"({recurrent} with >=2 independent origins)."
    )

    if args.null_output is not None:
        args.null_output.parent.mkdir(parents=True, exist_ok=True)
        tables["background_null"].to_csv(args.null_output, index=False)
        print(
            f"Wrote {len(tables['background_null'])} null background-pair "
            f"distances to {args.null_output}."
        )

    if args.pairs_output is not None:
        assert args.tips_csv is not None, "--pairs-output requires --tips-csv"
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        tables["pairs"].to_csv(args.pairs_output, index=False)
        print(
            f"Wrote {len(tables['pairs'])} epitope origin pairs to {args.pairs_output}."
        )

    if args.antigenic_null_output is not None:
        assert args.tips_csv is not None, "--antigenic-null-output requires --tips-csv"
        args.antigenic_null_output.parent.mkdir(parents=True, exist_ok=True)
        tables["antigenic_null"].to_csv(args.antigenic_null_output, index=False)
        print(
            f"Wrote {len(tables['antigenic_null'])} null antigenic-pair distances "
            f"to {args.antigenic_null_output}."
        )


if __name__ == "__main__":
    main()
