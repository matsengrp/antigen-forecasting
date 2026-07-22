"""Unit tests for antigentools/variant_agreement.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from antigentools import variant_agreement as va


class TestNormalizedInformationDistance:
    def test_identical_labels_is_zero(self):
        labels = [0, 0, 1, 1, 2, 2]
        assert va.normalized_information_distance(labels, labels) == pytest.approx(0.0)

    def test_relabeled_identical_partition_is_zero(self):
        # A pure relabeling is the same partition, so NID must be 0.
        x = [0, 0, 1, 1, 2, 2]
        y = [5, 5, 9, 9, 7, 7]
        assert va.normalized_information_distance(x, y) == pytest.approx(0.0)

    def test_independent_labels_near_one(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.integers(0, 5, n)
        y = rng.integers(0, 5, n)
        nid = va.normalized_information_distance(x, y)
        assert 0.8 <= nid <= 1.0

    def test_single_cluster_returns_zero(self):
        # Both vectors are one cluster: joint entropy is 0, partitions identical.
        x = [0, 0, 0, 0]
        y = [1, 1, 1, 1]
        assert va.normalized_information_distance(x, y) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            va.normalized_information_distance([0, 1, 2], [0, 1])

    def test_partial_agreement_between_zero_and_one(self):
        x = [0, 0, 1, 1, 2, 2, 3, 3]
        y = [0, 1, 1, 1, 2, 3, 3, 0]
        nid = va.normalized_information_distance(x, y)
        assert 0.0 < nid < 1.0


class TestVariantCountsOverTime:
    def _tips(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "year_bin": [10, 10, 10, 11, 11],
                "variant_ag": [0, 1, 2, 0, 0],
                "variant_tsne": [0, 0, 1, 5, 5],
            }
        )

    def test_counts_distinct_per_bin(self):
        out = va.variant_counts_over_time(self._tips(), ["variant_ag", "variant_tsne"])
        ag = out[(out["method"] == "variant_ag")].set_index("year_bin")["n_variants"]
        assert ag.loc[10] == 3  # labels 0,1,2
        assert ag.loc[11] == 1  # label 0 only
        tsne = out[(out["method"] == "variant_tsne")].set_index("year_bin")[
            "n_variants"
        ]
        assert tsne.loc[10] == 2  # labels 0,1
        assert tsne.loc[11] == 1  # label 5

    def test_all_nan_column_skipped(self):
        tips = self._tips()
        tips["variant_phylo"] = np.nan
        out = va.variant_counts_over_time(tips, ["variant_ag", "variant_phylo"])
        assert set(out["method"].unique()) == {"variant_ag"}

    def test_missing_column_skipped(self):
        out = va.variant_counts_over_time(
            self._tips(), ["variant_ag", "variant_absent"]
        )
        assert set(out["method"].unique()) == {"variant_ag"}


class TestNidPairs:
    def test_three_methods_three_pairs(self):
        tips = pd.DataFrame(
            {
                "variant_ag": [0, 0, 1, 1],
                "variant_tsne": [0, 0, 1, 1],
                "variant_phylo": [0, 1, 0, 1],
            }
        )
        out = va.nid_pairs(tips, ["variant_ag", "variant_tsne", "variant_phylo"])
        assert len(out) == 3
        assert list(out.columns) == ["method_x", "method_y", "nid"]
        # ag and tsne are identical partitions -> NID 0.
        ag_tsne = out[
            (out["method_x"] == "variant_ag") & (out["method_y"] == "variant_tsne")
        ]["nid"].iloc[0]
        assert ag_tsne == pytest.approx(0.0)

    def test_nan_rows_dropped(self):
        tips = pd.DataFrame(
            {
                "variant_ag": [0, 0, 1, 1, np.nan],
                "variant_tsne": [0, 0, 1, 1, 2],
            }
        )
        out = va.nid_pairs(tips, ["variant_ag", "variant_tsne"])
        assert len(out) == 1
        assert out["nid"].iloc[0] == pytest.approx(0.0)

    def test_all_nan_pair_skipped(self):
        tips = pd.DataFrame(
            {
                "variant_ag": [0.0, 1.0, 2.0],
                "variant_phylo": [np.nan, np.nan, np.nan],
            }
        )
        out = va.nid_pairs(tips, ["variant_ag", "variant_phylo"])
        assert out.empty
