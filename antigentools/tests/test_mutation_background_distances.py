"""Tests for scripts/mutation_background_distances.py.

These cover the load-bearing, easy-to-get-wrong pieces: the epitope-site loader
(including the ``189190`` concatenation quirk), amino-acid mutation parsing,
sequence reconstruction on a hand-built tree, and the frequency-based mean
pairwise Hamming distance checked against a brute-force pairwise computation.
"""

from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "mutation_background_distances.py"
)


@pytest.fixture(scope="module")
def mbd():
    """Import scripts/mutation_background_distances.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "mutation_background_distances", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_background_distances"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _two_gene_layout(mbd, ha1_len: int, ha2_len: int):
    """Build a minimal HA1/HA2 gene layout for reconstruction tests."""
    return {
        "HA1": mbd.GeneLayout(0, ha1_len, 0, ha1_len * 3),
        "HA2": mbd.GeneLayout(ha1_len, ha2_len, ha1_len * 3, (ha1_len + ha2_len) * 3),
    }


class TestEpitopeSiteLoader:
    def test_splits_concatenated_token_and_asserts_49(self, mbd, tmp_path):
        # The real shipped file, verbatim (note the concatenated 189190).
        content = (
            "50,53,54,121,122,124,126,131,133,135,137,142,143,144,145,146,155,156,"
            "157,158,159,160,163,164,172,173,174,186,188,189190,192,193,196,197,201,"
            "207,213,217,226,227,242,244,248,275,276,278,299,307"
        )
        path = tmp_path / "epitopeSites.txt"
        path.write_text(content)
        sites = mbd.load_epitope_sites(path, mbd.HA1_LENGTH)
        assert len(sites) == 49
        assert 189 in sites and 190 in sites
        assert 189190 not in sites

    def test_rejects_wrong_count(self, mbd, tmp_path):
        path = tmp_path / "epitopeSites.txt"
        path.write_text("50,53,54")
        with pytest.raises(AssertionError):
            mbd.load_epitope_sites(path, mbd.HA1_LENGTH)

    def test_rejects_non_numeric(self, mbd, tmp_path):
        path = tmp_path / "epitopeSites.txt"
        path.write_text("50,53,foo")
        with pytest.raises(ValueError):
            mbd.load_epitope_sites(path, mbd.HA1_LENGTH)


class TestParseMutations:
    def test_parses_and_skips_gaps(self, mbd):
        genes = _two_gene_layout(mbd, 5, 5)
        branch = {"HA1": ["F50I", "S53T"], "HA2": ["S2T", "A3-", "-4A", "X5Y"]}
        parsed = mbd.parse_mutations(branch, genes)
        assert ("HA1", 50, "F", "I") in parsed
        assert ("HA1", 53, "S", "T") in parsed
        assert ("HA2", 2, "S", "T") in parsed
        # Gap/unknown endpoints are dropped.
        assert all(
            m[2] not in mbd.SKIP_RESIDUES and m[3] not in mbd.SKIP_RESIDUES
            for m in parsed
        )
        assert len(parsed) == 3


class TestReconstruction:
    def _build(self, mbd, anchor_off_root: bool):
        """Root -> child_a (HA1:A1C) -> grandchild (HA1:C1A reversion, HA2:M1V)."""
        ha1_len, ha2_len = 3, 2
        genes = _two_gene_layout(mbd, ha1_len, ha2_len)
        nodes = {
            "root": mbd.Node("root", ["a"], None, []),
            "a": mbd.Node("a", ["g"], "root", [("HA1", 1, "A", "C")]),
            "g": mbd.Node("g", [], "a", [("HA1", 1, "C", "A"), ("HA2", 1, "M", "V")]),
        }
        recon = mbd.Reconstruction(
            nodes=nodes, root="root", genes=genes, total_aa_length=ha1_len + ha2_len
        )
        # Root protein AAAMM -> encode; anchor from a tip.
        root_aa = mbd.encode_aa("AAAMM")
        return recon, root_aa

    def test_fills_all_nodes_from_leaf_anchor(self, mbd):
        recon, root_aa = self._build(mbd, anchor_off_root=True)
        # Anchor on the grandchild leaf: root reconstructed by inverse walk.
        # grandchild protein: root AAAMM -> a: CAAMM -> g: AAAVM.
        anchor_aa = mbd.encode_aa("AAAVM")
        mbd.reconstruct_sequences(recon, "g", anchor_aa)
        assert np.array_equal(recon.nodes["root"].aa, root_aa)
        assert np.array_equal(recon.nodes["a"].aa, mbd.encode_aa("CAAMM"))
        assert np.array_equal(recon.nodes["g"].aa, mbd.encode_aa("AAAVM"))

    def test_conflicting_annotation_fails_loudly(self, mbd):
        recon, _ = self._build(mbd, anchor_off_root=True)
        # Branch a mutates HA1 site 1 from A; a root whose site 1 is G (valid
        # residue, wrong state) must trip the reconstruction consistency check.
        with pytest.raises(AssertionError):
            mbd.reconstruct_sequences(recon, "root", mbd.encode_aa("GAAMM"))


class TestMeanPairwiseHamming:
    def _brute_force(self, matrix):
        pairs = list(combinations(range(matrix.shape[0]), 2))
        total = sum(int(np.count_nonzero(matrix[i] != matrix[j])) for i, j in pairs)
        return total / len(pairs)

    def test_matches_brute_force(self, mbd):
        rng = np.random.default_rng(1)
        matrix = rng.integers(0, 4, size=(30, 12)).astype(np.int8)
        expected = self._brute_force(matrix)
        got = mbd.mean_pairwise_hamming(
            matrix, max_rows=1000, rng=np.random.default_rng(2)
        )
        assert got == pytest.approx(expected)

    def test_identical_rows_give_zero(self, mbd):
        matrix = np.tile(np.array([1, 2, 3], dtype=np.int8), (5, 1))
        got = mbd.mean_pairwise_hamming(
            matrix, max_rows=1000, rng=np.random.default_rng(0)
        )
        assert got == pytest.approx(0.0)


class TestBackgroundDistance:
    def test_distance_between_independent_origin_backgrounds(self, mbd):
        genes = _two_gene_layout(mbd, 3, 0)
        # Two origins of the same mutation whose parent backgrounds differ at 2 sites.
        nodes = {
            "p1": mbd.Node("p1", ["o1"], None, []),
            "o1": mbd.Node("o1", [], "p1", [("HA1", 1, "A", "C")]),
            "p2": mbd.Node("p2", ["o2"], None, []),
            "o2": mbd.Node("o2", [], "p2", [("HA1", 1, "A", "C")]),
        }
        recon = mbd.Reconstruction(
            nodes=nodes, root="p1", genes=genes, total_aa_length=3
        )
        recon.nodes["p1"].aa = mbd.encode_aa("AAA")
        recon.nodes["p2"].aa = mbd.encode_aa("ATT")  # differs from p1 at 2 sites.
        assert mbd.mean_pairwise_background_distance(
            recon, ["o1", "o2"]
        ) == pytest.approx(2.0)


class TestMeanPairwiseAntigenicDistance:
    def _brute_force(self, positions):
        pairs = list(combinations(positions, 2))
        total = sum(float(np.linalg.norm(a - b)) for a, b in pairs)
        return total / len(pairs)

    def test_matches_brute_force(self, mbd):
        rng = np.random.default_rng(3)
        positions = [rng.normal(size=2) for _ in range(6)]
        expected = self._brute_force(positions)
        assert mbd.mean_pairwise_antigenic_distance(positions) == pytest.approx(
            expected
        )

    def test_identical_positions_give_zero(self, mbd):
        positions = [np.array([2.0, -1.0]) for _ in range(4)]
        assert mbd.mean_pairwise_antigenic_distance(positions) == pytest.approx(0.0)


class TestOriginAntigenicPosition:
    def _recon(self, mbd):
        # Origin "o" with two descendant tips at different epitope counts.
        nodes = {
            "o": mbd.Node("o", ["t1", "t2"], None, []),
            "t1": mbd.Node("t1", [], "o", []),
            "t2": mbd.Node("t2", [], "o", []),
        }
        return mbd.Reconstruction(nodes=nodes, root="o")

    def test_uses_only_minimal_epitope_count_descendants(self, mbd):
        recon = self._recon(mbd)
        sampled = {"t1", "t2"}
        positions = {"t1": np.array([1.0, 1.0]), "t2": np.array([10.0, 10.0])}
        # t2 added a further epitope mutation, so it is excluded from the estimate.
        epi_counts = {"t1": 5, "t2": 6}
        position, min_count = mbd.origin_antigenic_position(
            recon, "o", sampled, positions, epi_counts
        )
        assert min_count == 5
        assert np.array_equal(position, np.array([1.0, 1.0]))

    def test_averages_ties_at_minimal_count(self, mbd):
        recon = self._recon(mbd)
        sampled = {"t1", "t2"}
        positions = {"t1": np.array([0.0, 0.0]), "t2": np.array([4.0, 2.0])}
        epi_counts = {"t1": 5, "t2": 5}  # tie -> average both.
        position, min_count = mbd.origin_antigenic_position(
            recon, "o", sampled, positions, epi_counts
        )
        assert min_count == 5
        assert np.array_equal(position, np.array([2.0, 1.0]))

    def test_returns_none_without_antigenic_descendants(self, mbd):
        recon = self._recon(mbd)
        assert mbd.origin_antigenic_position(recon, "o", {"t1", "t2"}, {}, {}) is None


class TestLoadTipAntigenic:
    def test_loads_positions_and_counts(self, mbd, tmp_path):
        path = tmp_path / "unique_tips.csv"
        path.write_text(
            "name,year,ag1,ag2,epitopeMutationCount\n"
            "a,2000.0,1.5,-2.0,3\n"
            "b,2001.0,4.0,5.0,7\n"
        )
        positions, epi_counts = mbd.load_tip_antigenic(path)
        assert np.array_equal(positions["a"], np.array([1.5, -2.0]))
        assert epi_counts["b"] == 7

    def test_rejects_missing_columns(self, mbd, tmp_path):
        path = tmp_path / "unique_tips.csv"
        path.write_text("name,ag1,ag2\na,1.0,2.0\n")  # no epitopeMutationCount.
        with pytest.raises(AssertionError):
            mbd.load_tip_antigenic(path)


class TestSubtreeIntervals:
    def _recon(self, mbd):
        """root -> a -> g, with b a second child of root (so b and a are siblings)."""
        genes = _two_gene_layout(mbd, 3, 2)
        nodes = {
            "root": mbd.Node("root", ["a", "b"], None, []),
            "a": mbd.Node("a", ["g"], "root", []),
            "g": mbd.Node("g", [], "a", []),
            "b": mbd.Node("b", [], "root", []),
        }
        return mbd.Reconstruction(
            nodes=nodes, root="root", genes=genes, total_aa_length=5
        )

    def test_labels_every_node(self, mbd):
        intervals = mbd.label_subtree_intervals(self._recon(mbd))
        assert set(intervals) == {"root", "a", "g", "b"}

    def test_descendant_ancestor_and_unrelated(self, mbd):
        intervals = mbd.label_subtree_intervals(self._recon(mbd))
        # g sits below a, which sits below root.
        assert mbd.is_strict_descendant(intervals, "g", "a")
        assert mbd.is_strict_descendant(intervals, "g", "root")
        assert mbd.is_strict_descendant(intervals, "a", "root")
        # The relation is directional, and siblings are unrelated either way.
        assert not mbd.is_strict_descendant(intervals, "a", "g")
        assert not mbd.is_strict_descendant(intervals, "b", "a")
        assert not mbd.is_strict_descendant(intervals, "a", "b")

    def test_node_is_not_its_own_descendant(self, mbd):
        intervals = mbd.label_subtree_intervals(self._recon(mbd))
        for name in ("root", "a", "g", "b"):
            assert not mbd.is_strict_descendant(intervals, name, name)

    def test_detects_disconnected_tree(self, mbd):
        recon = self._recon(mbd)
        # An orphan unreachable from the root must be caught rather than ignored.
        recon.nodes["orphan"] = mbd.Node("orphan", [], None, [])
        with pytest.raises(AssertionError):
            mbd.label_subtree_intervals(recon)


class TestReversions:
    def _recon(self, mbd):
        """A tree holding one true lineage cycle and one sibling-only reversal.

        HA1 site 1: A->C on ``a`` and C->A on ``g``, which is strictly below ``a``,
        so this is a genuine gain-then-loss along one lineage. HA2 site 1: M->V on
        ``b`` and V->M on ``c``, which are siblings, so the reverse exists on the
        tree but never below its forward origin.
        """
        genes = _two_gene_layout(mbd, 3, 2)
        nodes = {
            "root": mbd.Node("root", ["a", "b", "c"], None, []),
            "a": mbd.Node("a", ["g"], "root", [("HA1", 1, "A", "C")]),
            "g": mbd.Node("g", [], "a", [("HA1", 1, "C", "A")]),
            "b": mbd.Node("b", [], "root", [("HA2", 1, "M", "V")]),
            "c": mbd.Node("c", [], "root", [("HA2", 1, "V", "M")]),
        }
        return mbd.Reconstruction(
            nodes=nodes, root="root", genes=genes, total_aa_length=5
        )

    def test_finds_only_the_lineage_cycle(self, mbd):
        recon = self._recon(mbd)
        pairs = mbd.find_reversion_pairs(recon, mbd.index_origins(recon))
        assert pairs == [(("HA1", 1, "A", "C"), "a", "g")]

    def test_sibling_reversal_counts_only_toward_upper_bound(self, mbd):
        recon = self._recon(mbd)
        counts = mbd.summarize_reversions(recon, mbd.index_origins(recon), {1})
        # All four substitutions have their reverse somewhere on the tree.
        assert counts["n_reverse_anywhere_epitope"] == 2  # HA1:A1C and HA1:C1A.
        assert counts["n_reverse_anywhere_non_epitope"] == 2  # HA2:M1V and HA2:V1M.
        # Only the HA1 forward origin has its reverse strictly below it.
        assert counts["n_lineage_cycle_epitope"] == 1
        assert counts["n_lineage_cycle_non_epitope"] == 0
        assert counts["n_lineage_cycle_origin_pairs"] == 1

    def test_directed_and_unordered_counts_differ(self, mbd):
        recon = self._recon(mbd)
        counts = mbd.summarize_reversions(recon, mbd.index_origins(recon), {1})
        # Four directed substitutions have a reverse, but they are only two
        # distinct reversible pairs: HA1 {A1C, C1A} and HA2 {M1V, V1M}. Quoting a
        # directed count against an unordered denominator would double the rate.
        directed = (
            counts["n_reverse_anywhere_epitope"]
            + counts["n_reverse_anywhere_non_epitope"]
        )
        assert directed == 4
        assert counts["n_reverse_anywhere_unordered_pairs"] == 2
        # Only HA1:A1C cycles, so one directed key and one unordered pair.
        assert counts["n_lineage_cycle_unordered_pairs"] == 1

    def test_epitope_split_is_ha1_only(self, mbd):
        recon = self._recon(mbd)
        origins = mbd.index_origins(recon)
        # HA2 site 1 must not be called an epitope just because 1 is an epitope
        # site in HA1 numbering.
        counts = mbd.summarize_reversions(recon, origins, {1})
        assert counts["n_reverse_anywhere_non_epitope"] == 2
        # With no epitope sites configured, every substitution is non-epitope.
        counts = mbd.summarize_reversions(recon, origins, set())
        assert counts["n_reverse_anywhere_epitope"] == 0
        assert counts["n_reverse_anywhere_non_epitope"] == 4
        assert counts["n_lineage_cycle_epitope"] == 0
        assert counts["n_lineage_cycle_non_epitope"] == 1

    def test_counts_reverse_further_down_the_lineage(self, mbd):
        genes = _two_gene_layout(mbd, 3, 2)
        nodes = {
            "root": mbd.Node("root", ["a"], None, [("HA1", 1, "A", "C")]),
            "a": mbd.Node("a", ["g"], "root", []),
            "g": mbd.Node("g", [], "a", [("HA1", 1, "C", "A")]),
        }
        recon = mbd.Reconstruction(
            nodes=nodes, root="root", genes=genes, total_aa_length=5
        )
        pairs = mbd.find_reversion_pairs(recon, mbd.index_origins(recon))
        # The reverse is two branches below the forward origin, not adjacent.
        assert pairs == [(("HA1", 1, "A", "C"), "root", "g")]

    def test_no_reverse_gives_zero_counts(self, mbd):
        genes = _two_gene_layout(mbd, 3, 2)
        nodes = {
            "root": mbd.Node("root", ["a"], None, []),
            "a": mbd.Node("a", ["g"], "root", [("HA1", 1, "A", "C")]),
            "g": mbd.Node("g", [], "a", [("HA2", 1, "M", "V")]),
        }
        recon = mbd.Reconstruction(
            nodes=nodes, root="root", genes=genes, total_aa_length=5
        )
        origins = mbd.index_origins(recon)
        assert mbd.find_reversion_pairs(recon, origins) == []
        counts = mbd.summarize_reversions(recon, origins, {1})
        assert set(counts.values()) == {0}
