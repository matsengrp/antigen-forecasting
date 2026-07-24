"""Unit tests for per-host immune-history risk of infection.

These protect two invariants. First, that the vectorized ``np.minimum.reduceat`` kernel
computes exactly what a plain per-host loop would. Second, that hosts are keyed on
``(year, deme, host_id)`` -- antigen resets ``host_id`` to 0 for every snapshot and deme,
so any coarser key silently splices unrelated hosts' immune memories together and drives
every risk to the clipped minimum.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from antigentools.analysis import calculate_fitness_of_tips
from antigentools.host_immunity import (
    HISTORIES_RAW_DTYPES,
    global_naive_fraction,
    load_raw_histories,
    pack_host_histories,
    risk_of_infection_matrix,
    risk_of_infection_over_time,
    risk_of_infection_stats,
    select_timepoints,
    variance_from_risk,
)

SCRIPT_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "calc_host_immunity_fitness.py"
)

SMITH_CONVERSION = 0.07
HOMOLOGOUS_IMMUNITY = 0.95


@pytest.fixture(scope="module")
def calc_host_immunity_fitness():
    """Load the CLI script as a module (``scripts/`` is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "calc_host_immunity_fitness", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["calc_host_immunity_fitness"] = module
    spec.loader.exec_module(module)
    return module


def make_histories(rows):
    """Build a raw-histories frame from ``(year, deme, host_id, ag1, ag2, nf)`` tuples."""
    frame = pd.DataFrame(
        rows, columns=["year", "deme", "host_id", "ag1", "ag2", "naive_fraction"]
    )
    frame.insert(3, "infection_index", frame.groupby(["year", "deme", "host_id"]).cumcount())
    return frame


def naive_risks(tips_ag, histories, smith_conversion, homologous_immunity):
    """Reference implementation: an explicit loop over tips and hosts."""
    host_keys = list(dict.fromkeys(zip(histories["deme"], histories["host_id"])))
    host_keys.sort()
    out = np.empty((len(tips_ag), len(host_keys)))
    for i, (tip_ag1, tip_ag2) in enumerate(tips_ag):
        for j, (deme, host_id) in enumerate(host_keys):
            memory = histories[
                (histories["deme"] == deme) & (histories["host_id"] == host_id)
            ]
            closest = min(
                float(np.hypot(tip_ag1 - row.ag1, tip_ag2 - row.ag2))
                for row in memory.itertuples()
            )
            out[i, j] = min(
                max(closest * smith_conversion, 1.0 - homologous_immunity), 1.0
            )
    return out


class TestSelectTimepoints:
    """Test select_timepoints."""

    def test_stride_of_two_takes_every_other_year(self):
        years = np.arange(0.0, 6.0)
        assert select_timepoints(years, 2.0).tolist() == [0.0, 2.0, 4.0]

    def test_half_year_grid_supported(self):
        years = np.arange(0.0, 3.0, 0.5)
        assert select_timepoints(years, 0.5).tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    def test_finer_than_native_spacing_raises(self):
        years = np.arange(0.0, 6.0)
        with pytest.raises(ValueError, match="finer than the histories"):
            select_timepoints(years, 0.5)

    def test_too_coarse_raises(self):
        years = np.arange(0.0, 3.0)
        with pytest.raises(ValueError, match="selected only"):
            select_timepoints(years, 100.0)

    def test_non_positive_delta_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            select_timepoints(np.arange(0.0, 3.0), 0.0)


class TestPackHostHistories:
    """Test pack_host_histories."""

    def test_segments_are_contiguous_per_host(self):
        histories = make_histories(
            [
                (0.0, "north", 0, 1.0, 0.0, 0.0),
                (0.0, "north", 1, 2.0, 0.0, 0.0),
                (0.0, "north", 0, 3.0, 0.0, 0.0),
            ]
        )
        phenotypes, starts, n_total = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )

        assert n_total == 2
        assert starts.tolist() == [0, 2]
        # Host 0's two entries come first, then host 1's single entry.
        assert sorted(phenotypes[0:2, 0].tolist()) == [1.0, 3.0]
        assert phenotypes[2, 0] == 2.0

    def test_host_id_reused_across_demes_stays_separate(self):
        histories = make_histories(
            [
                (0.0, "north", 7, 1.0, 0.0, 0.0),
                (0.0, "tropics", 7, 50.0, 0.0, 0.0),
            ]
        )
        _, starts, n_total = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )

        assert n_total == 2
        assert starts.tolist() == [0, 1]

    def test_sampling_selects_hosts_not_rows(self):
        # Host 0 has 5 memory entries, hosts 1-4 have 1 each. Sampling 2 hosts must
        # yield 2 segments regardless of how lopsided the row counts are.
        rows = [(0.0, "north", 0, float(i), 0.0, 0.0) for i in range(5)]
        rows += [(0.0, "north", h, 10.0, 0.0, 0.0) for h in range(1, 5)]
        histories = make_histories(rows)

        _, starts, n_total = pack_host_histories(
            histories, 2, np.random.default_rng(0)
        )

        assert n_total == 5
        assert starts.size == 2

    def test_sampling_is_seeded(self):
        rows = [(0.0, "north", h, float(h), 0.0, 0.0) for h in range(20)]
        histories = make_histories(rows)

        first, _, _ = pack_host_histories(histories, 5, np.random.default_rng(7))
        second, _, _ = pack_host_histories(histories, 5, np.random.default_rng(7))
        other, _, _ = pack_host_histories(histories, 5, np.random.default_rng(8))

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)

    def test_n_hosts_above_available_uses_all(self):
        histories = make_histories([(0.0, "north", h, 1.0, 0.0, 0.0) for h in range(3)])
        _, starts, n_total = pack_host_histories(
            histories, 100, np.random.default_rng(0)
        )
        assert starts.size == n_total == 3

    def test_empty_frame_raises(self):
        empty = make_histories([]).astype({"host_id": "int64"})
        with pytest.raises(ValueError, match="empty frame"):
            pack_host_histories(empty, None, np.random.default_rng(0))


class TestGlobalNaiveFraction:
    """Test global_naive_fraction."""

    def test_single_deme_returns_that_fraction(self):
        histories = make_histories([(0.0, "north", h, 1.0, 0.0, 0.2) for h in range(4)])
        assert np.isclose(global_naive_fraction(histories), 0.2)

    def test_two_demes_are_host_weighted(self):
        # north: 8 experienced at nf=0.2 -> 10 hosts, 2 naive.
        # tropics: 9 experienced at nf=0.1 -> 10 hosts, 1 naive.
        rows = [(0.0, "north", h, 1.0, 0.0, 0.2) for h in range(8)]
        rows += [(0.0, "tropics", h, 1.0, 0.0, 0.1) for h in range(9)]
        histories = make_histories(rows)

        assert np.isclose(global_naive_fraction(histories), 3.0 / 20.0)

    def test_inconsistent_fraction_within_deme_raises(self):
        histories = make_histories(
            [
                (0.0, "north", 0, 1.0, 0.0, 0.2),
                (0.0, "north", 1, 1.0, 0.0, 0.3),
            ]
        )
        with pytest.raises(ValueError, match="distinct naive_fraction"):
            global_naive_fraction(histories)

    def test_fraction_of_one_raises(self):
        histories = make_histories([(0.0, "north", 0, 1.0, 0.0, 1.0)])
        with pytest.raises(ValueError, match=r"outside \[0, 1\)"):
            global_naive_fraction(histories)

    def test_host_id_gaps_are_consistent_with_the_fraction(self):
        # Naive hosts consume a host_id, so ids 0 and 3 present with 2 experienced hosts
        # implies at least 4 slots; naive_fraction=0.5 gives exactly 4. Consistent.
        histories = make_histories(
            [
                (0.0, "north", 0, 1.0, 0.0, 0.5),
                (0.0, "north", 3, 1.0, 0.0, 0.5),
            ]
        )
        assert np.isclose(global_naive_fraction(histories), 0.5)

    def test_host_id_beyond_implied_slot_count_raises(self):
        # 2 experienced hosts at naive_fraction=0.0 implies 2 slots, so a host_id of 50
        # means naive_fraction does not describe this file's host universe.
        histories = make_histories(
            [
                (0.0, "north", 0, 1.0, 0.0, 0.0),
                (0.0, "north", 50, 1.0, 0.0, 0.0),
            ]
        )
        with pytest.raises(ValueError, match="file is inconsistent"):
            global_naive_fraction(histories)


class TestRiskOfInfectionMatrix:
    """Test risk_of_infection_matrix."""

    def test_matches_naive_loop(self):
        rng = np.random.default_rng(11)
        rows = []
        for host_id in range(12):
            for _ in range(rng.integers(1, 5)):
                rows.append(
                    (
                        0.0,
                        "north" if host_id % 2 else "tropics",
                        host_id,
                        float(rng.normal(0, 8)),
                        float(rng.normal(0, 8)),
                        0.0,
                    )
                )
        histories = make_histories(rows)
        tips_ag = rng.normal(0, 8, size=(7, 2)).astype(np.float32)

        phenotypes, starts, _ = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )
        actual = risk_of_infection_matrix(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 4
        )
        expected = naive_risks(
            tips_ag, histories, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chunking_invariance(self):
        rng = np.random.default_rng(3)
        rows = [
            (0.0, "north", h, float(rng.normal(0, 5)), float(rng.normal(0, 5)), 0.0)
            for h in range(6)
            for _ in range(2)
        ]
        histories = make_histories(rows)
        tips_ag = rng.normal(0, 5, size=(9, 2)).astype(np.float32)
        phenotypes, starts, _ = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )

        one_at_a_time = risk_of_infection_matrix(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 1
        )
        all_at_once = risk_of_infection_matrix(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 10**6
        )

        np.testing.assert_array_equal(one_at_a_time, all_at_once)

    def test_risk_is_bounded(self):
        histories = make_histories(
            [
                (0.0, "north", 0, 0.0, 0.0, 0.0),
                (0.0, "north", 1, 0.0, 0.0, 0.0),
            ]
        )
        # One tip identical to the memory, one absurdly far from it.
        tips_ag = np.array([[0.0, 0.0], [1000.0, 1000.0]], dtype=np.float32)
        phenotypes, starts, _ = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )

        risks = risk_of_infection_matrix(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 8
        )

        assert np.allclose(risks[0], 1.0 - HOMOLOGOUS_IMMUNITY)
        assert np.allclose(risks[1], 1.0)

    def test_non_positive_chunk_size_raises(self):
        with pytest.raises(ValueError, match="max_block_elements must be positive"):
            risk_of_infection_matrix(
                np.zeros((1, 2)),
                np.zeros((1, 2)),
                np.array([0]),
                SMITH_CONVERSION,
                HOMOLOGOUS_IMMUNITY,
                0,
            )

    def test_non_increasing_segment_starts_raises(self):
        # A repeated start is a zero-length segment; np.minimum.reduceat would silently
        # return the raw element there instead of a reduction.
        with pytest.raises(ValueError, match="strictly increasing"):
            risk_of_infection_matrix(
                np.zeros((1, 2)),
                np.zeros((4, 2)),
                np.array([0, 2, 2]),
                SMITH_CONVERSION,
                HOMOLOGOUS_IMMUNITY,
                8,
            )

    def test_out_of_range_segment_starts_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            risk_of_infection_matrix(
                np.zeros((1, 2)),
                np.zeros((2, 2)),
                np.array([0, 2]),
                SMITH_CONVERSION,
                HOMOLOGOUS_IMMUNITY,
                8,
            )

    def test_centering_survives_large_coordinate_offsets(self):
        # Antigenic coordinates drift far from the origin while the distances deciding a
        # min stay small; without re-centering, float32 differencing loses the signal.
        rng = np.random.default_rng(5)
        base_rows = [
            (0.0, "north", h, float(rng.normal(0, 3)), float(rng.normal(0, 3)), 0.0)
            for h in range(6)
            for _ in range(2)
        ]
        tips_ag = rng.normal(0, 3, size=(5, 2))

        offset = 5000.0
        shifted_rows = [
            (y, d, h, a1 + offset, a2 + offset, nf)
            for (y, d, h, a1, a2, nf) in base_rows
        ]

        def risks_for(rows, tips):
            histories = make_histories(rows)
            phenotypes, starts, _ = pack_host_histories(
                histories, None, np.random.default_rng(0)
            )
            return risk_of_infection_matrix(
                tips, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 4
            )

        at_origin = risks_for(base_rows, tips_ag)
        shifted = risks_for(shifted_rows, tips_ag + offset)

        np.testing.assert_allclose(at_origin, shifted, rtol=1e-6, atol=1e-9)


class TestRiskOfInfectionStats:
    """Test risk_of_infection_stats."""

    @staticmethod
    def _fixture(seed):
        rng = np.random.default_rng(seed)
        rows = [
            (0.0, "north", h, float(rng.normal(0, 6)), float(rng.normal(0, 6)), 0.0)
            for h in range(9)
            for _ in range(3)
        ]
        histories = make_histories(rows)
        tips_ag = rng.normal(0, 6, size=(11, 2))
        phenotypes, starts, _ = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )
        return tips_ag, phenotypes, starts

    def test_matches_dense_matrix_reduction(self):
        tips_ag, phenotypes, starts = self._fixture(17)

        mean, sd = risk_of_infection_stats(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 3
        )
        dense = risk_of_infection_matrix(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 3
        )

        np.testing.assert_allclose(mean, dense.mean(axis=1), rtol=1e-12)
        np.testing.assert_allclose(sd, dense.std(axis=1, ddof=1), rtol=1e-6)

    def test_chunking_invariance(self):
        tips_ag, phenotypes, starts = self._fixture(23)

        one = risk_of_infection_stats(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 1
        )
        many = risk_of_infection_stats(
            tips_ag, phenotypes, starts, SMITH_CONVERSION, HOMOLOGOUS_IMMUNITY, 10**6
        )

        np.testing.assert_array_equal(one[0], many[0])
        np.testing.assert_array_equal(one[1], many[1])

    def test_single_host_gives_nan_sd(self):
        histories = make_histories([(0.0, "north", 0, 1.0, 0.0, 0.0)])
        phenotypes, starts, _ = pack_host_histories(
            histories, None, np.random.default_rng(0)
        )

        mean, sd = risk_of_infection_stats(
            np.array([[5.0, 0.0]]),
            phenotypes,
            starts,
            SMITH_CONVERSION,
            HOMOLOGOUS_IMMUNITY,
            8,
        )

        assert np.isclose(mean[0], 4.0 * SMITH_CONVERSION)
        assert np.isnan(sd[0])


class TestRiskOfInfectionOverTime:
    """Test risk_of_infection_over_time."""

    @staticmethod
    def _tips():
        return pd.DataFrame(
            {
                "name": ["a", "b", "c"],
                "year": [0.5, 1.5, 1.5],
                "ag1": [10.0, 12.0, 14.0],
                "ag2": [0.0, 0.0, 0.0],
                "variant_test": [0, 0, 1],
            }
        )

    def test_single_shared_memory_entry_matches_centroid_fitness(self):
        # With one identical memory entry per host, the min-over-memory rule degenerates
        # to distance-from-a-point, so the centroid implementation must agree exactly.
        rows = [
            (year, "north", host, 8.0, 0.0, 0.0)
            for year in (0.0, 1.0, 2.0)
            for host in range(5)
        ]
        histories = make_histories(rows)
        tips = self._tips()

        result = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )
        expected = calculate_fitness_of_tips(
            tips,
            (8.0, 0.0),
            s=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
        )

        at_year_one = result[result["year"] == 1.0].set_index("name")
        np.testing.assert_allclose(
            at_year_one.loc[tips["name"], "mean_risk_of_infection_experienced"],
            expected["fitness"],
            rtol=1e-6,
        )
        # Every host shares the same memory, so there is no spread across hosts.
        assert np.allclose(at_year_one["sd_risk_of_infection_experienced"], 0.0)

    def test_naive_weighting(self):
        rows = [
            (year, "north", host, 8.0, 0.0, 0.25)
            for year in (0.0, 1.0)
            for host in range(4)
        ]
        histories = make_histories(rows)

        result = risk_of_infection_over_time(
            self._tips(),
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )

        np.testing.assert_allclose(
            result["mean_risk_of_infection_population"],
            0.25 + 0.75 * result["mean_risk_of_infection_experienced"],
            rtol=1e-6,
        )
        assert np.allclose(result["naive_fraction"], 0.25)

    def test_reports_sampled_and_total_hosts(self):
        rows = [
            (year, "north", host, float(host), 0.0, 0.0)
            for year in (0.0, 1.0)
            for host in range(10)
        ]
        histories = make_histories(rows)

        result = risk_of_infection_over_time(
            self._tips(),
            histories,
            delta_t=1.0,
            n_hosts=3,
            seed=1,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )

        assert set(result["n_hosts_sampled"]) == {3}
        assert set(result["n_hosts_total"]) == {10}
        assert len(result) == 3 * 2  # 3 tips x 2 timepoints.

    def test_host_ids_do_not_leak_between_years(self):
        # antigen resets host_id to 0 for every (snapshot, deme), so host 0 in year 0 and
        # host 0 in year 1 are unrelated individuals. Splicing their memories together
        # would give every tip an artificially close match and collapse risk to the
        # clipped minimum -- plausible-looking, completely wrong output.
        histories = make_histories(
            [
                (0.0, "north", 0, 100.0, 0.0, 0.0),
                (1.0, "north", 0, 10.0, 0.0, 0.0),
            ]
        )
        tips = pd.DataFrame(
            {
                "name": ["a"],
                "year": [0.5],
                "ag1": [10.0],
                "ag2": [0.0],
                "variant_test": [0],
            }
        )

        result = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )

        by_year = result.set_index("year")["mean_risk_of_infection_experienced"]
        # Year 0's only memory sits 90 units away: 90 * 0.07 clips to 1.0.
        assert np.isclose(by_year[0.0], 1.0)
        # Year 1's memory is identical to the tip, so risk clips to the homologous floor.
        assert np.isclose(by_year[1.0], 1.0 - HOMOLOGOUS_IMMUNITY)

    def test_duplicate_tip_names_raise(self):
        tips = self._tips()
        tips.loc[1, "name"] = "a"
        histories = make_histories(
            [(year, "north", 0, 8.0, 0.0, 0.0) for year in (0.0, 1.0)]
        )
        with pytest.raises(ValueError, match="duplicate 'name'"):
            risk_of_infection_over_time(
                tips,
                histories,
                delta_t=1.0,
                n_hosts=None,
                seed=0,
                smith_conversion=SMITH_CONVERSION,
                homologous_immunity=HOMOLOGOUS_IMMUNITY,
                max_block_elements=8_000_000,
            )

    def test_missing_tip_column_raises(self):
        histories = make_histories(
            [(year, "north", 0, 8.0, 0.0, 0.0) for year in (0.0, 1.0)]
        )
        with pytest.raises(ValueError, match="missing required columns"):
            risk_of_infection_over_time(
                self._tips().drop(columns=["ag2"]),
                histories,
                delta_t=1.0,
                n_hosts=None,
                seed=0,
                smith_conversion=SMITH_CONVERSION,
                homologous_immunity=HOMOLOGOUS_IMMUNITY,
                max_block_elements=8_000_000,
            )


class TestVarianceFromRisk:
    """Test variance_from_risk."""

    @staticmethod
    def _inputs():
        tips = pd.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "year": [0.5, 0.5, 1.5, 1.5],
                "ag1": [10.0, 12.0, 14.0, 16.0],
                "ag2": [0.0, 0.0, 0.0, 0.0],
                "variant_test": [0, 0, 1, 1],
            }
        )
        histories = make_histories(
            [
                (year, "north", host, 8.0, 0.0, 0.2)
                for year in (0.0, 1.0, 2.0)
                for host in range(4)
            ]
        )
        return tips, histories

    def test_schema_and_weightings(self):
        tips, histories = self._inputs()
        risk_df = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )

        result = variance_from_risk(
            risk_df,
            tips,
            ["variant_test"],
            n_variant_window=1.0,
            host_weighting="population",
        )

        # Deliberately identical to calc_variance_over_time's schema, so the new method
        # is a drop-in for the centroid one everywhere downstream.
        assert list(result.columns) == [
            "year",
            "method",
            "mean_variance",
            "n_variants",
        ]
        assert set(result["method"]) == {"test"}
        assert len(result) == 3  # 3 timepoints, one row each.

    def test_population_variance_is_scaled_by_naive_fraction(self):
        # mean_population is an affine image of mean_experienced within a timepoint, so
        # the variances differ by exactly (1 - naive_fraction) ** 2.
        tips, histories = self._inputs()
        risk_df = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )
        both = {
            weighting: variance_from_risk(
                risk_df,
                tips,
                ["variant_test"],
                n_variant_window=1.0,
                host_weighting=weighting,
            ).set_index("year")["mean_variance"]
            for weighting in ("experienced", "population")
        }

        np.testing.assert_allclose(
            both["population"], (0.8**2) * both["experienced"], rtol=1e-5
        )

    def test_n_variants_counts_only_the_window(self):
        tips, histories = self._inputs()
        risk_df = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )
        result = variance_from_risk(
            risk_df,
            tips,
            ["variant_test"],
            n_variant_window=1.0,
            host_weighting="population",
        )

        by_year = result.set_index("year")["n_variants"]
        assert by_year[0.0] == 0  # No tips in (-1, 0].
        assert by_year[1.0] == 1  # Only variant 0 in (0, 1].
        assert by_year[2.0] == 1  # Only variant 1 in (1, 2].

    def test_missing_variant_column_raises(self):
        tips, histories = self._inputs()
        risk_df = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )
        with pytest.raises(ValueError, match="missing variant columns"):
            variance_from_risk(
                risk_df,
                tips,
                ["variant_absent"],
                n_variant_window=1.0,
                host_weighting="population",
            )

    def test_unknown_host_weighting_raises(self):
        tips, histories = self._inputs()
        risk_df = risk_of_infection_over_time(
            tips,
            histories,
            delta_t=1.0,
            n_hosts=None,
            seed=0,
            smith_conversion=SMITH_CONVERSION,
            homologous_immunity=HOMOLOGOUS_IMMUNITY,
            max_block_elements=8_000_000,
        )
        with pytest.raises(ValueError, match="host_weighting must be one of"):
            variance_from_risk(
                risk_df,
                tips,
                ["variant_test"],
                n_variant_window=1.0,
                host_weighting="everyone",
            )


class TestLoadRawHistories:
    """Test load_raw_histories."""

    def test_reads_and_types_columns(self, tmp_path):
        path = tmp_path / "out.histories.raw.csv"
        make_histories(
            [
                (0.0, "north", 0, -6.0, 0.0, 0.1369),
                (0.0, "north", 0, 11.649894, -3.691925, 0.1369),
                (1.0, "north", 0, -6.0, 0.0, 0.14),
            ]
        ).to_csv(path, index=False)

        result = load_raw_histories(path)

        assert list(result.columns) == list(HISTORIES_RAW_DTYPES)
        assert str(result["deme"].dtype) == "category"
        # The (-6, 0) ancestral phenotypes are real memory entries, never filtered.
        assert (result["ag1"] == -6.0).sum() == 2

    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"year": [0.0], "deme": ["north"]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_raw_histories(path)


class TestCli:
    """Test the scripts/calc_host_immunity_fitness.py entry point."""

    @staticmethod
    def _write_inputs(tmp_path):
        tips_path = tmp_path / "tips_with_variants.tsv"
        pd.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "year": [0.5, 0.5, 1.5, 1.5],
                "ag1": [10.0, 12.0, 14.0, 16.0],
                "ag2": [0.0, 0.0, 0.0, 0.0],
                "variant_ag": [0, 0, 1, 1],
                "variant_phylo": [0, 1, 1, 1],
            }
        ).to_csv(tips_path, sep="\t", index=False)

        histories_path = tmp_path / "out.histories.raw.csv"
        make_histories(
            [
                (year, deme, host, 8.0 + host, 0.0, 0.2)
                for year in (0.0, 0.5, 1.0, 1.5, 2.0)
                for deme in ("north", "tropics")
                for host in range(6)
            ]
        ).to_csv(histories_path, index=False)
        return tips_path, histories_path

    def test_end_to_end(self, tmp_path, calc_host_immunity_fitness):
        tips_path, histories_path = self._write_inputs(tmp_path)
        variance_path = tmp_path / "out" / "variance.tsv"
        risk_path = tmp_path / "out" / "risk.tsv"

        calc_host_immunity_fitness.main(
            [
                "--tips",
                str(tips_path),
                "--histories-raw",
                str(histories_path),
                "--variance-output",
                str(variance_path),
                "--risk-output",
                str(risk_path),
                "--delta-t",
                "0.5",
                "--n-hosts",
                "4",
                "--run-id",
                "run_0",
            ]
        )

        variance = pd.read_csv(variance_path, sep="\t")
        assert list(variance.columns) == [
            "run_id",
            "year",
            "method",
            "mean_variance",
            "n_variants",
        ]
        assert len(variance) == 10  # 5 timepoints x 2 methods.
        assert set(variance["run_id"]) == {"run_0"}
        assert set(variance["method"]) == {"ag", "phylo"}

        risk = pd.read_csv(risk_path, sep="\t")
        assert set(risk["n_hosts_sampled"]) == {4}
        assert set(risk["n_hosts_total"]) == {12}
        assert len(risk) == 4 * 5

    def test_no_variant_columns_raises(self, tmp_path, calc_host_immunity_fitness):
        tips_path, histories_path = self._write_inputs(tmp_path)
        plain = pd.read_csv(tips_path, sep="\t").drop(
            columns=["variant_ag", "variant_phylo"]
        )
        plain.to_csv(tips_path, sep="\t", index=False)

        with pytest.raises(ValueError, match="no variant_\\* columns"):
            calc_host_immunity_fitness.main(
                [
                    "--tips",
                    str(tips_path),
                    "--histories-raw",
                    str(histories_path),
                    "--variance-output",
                    str(tmp_path / "variance.tsv"),
                ]
            )

    def test_validate_centroids_catches_host_count_mismatch(
        self, tmp_path, calc_host_immunity_fitness
    ):
        _, histories_path = self._write_inputs(tmp_path)
        histories = calc_host_immunity_fitness.load_raw_histories(histories_path)
        centroids_path = tmp_path / "out.histories.csv"
        pd.DataFrame(
            {
                "year": [0.0, 0.0],
                "deme": ["north", "tropics"],
                "ag1": [10.5, 10.5],
                "ag2": [0.0, 0.0],
                "naive_fraction": [0.2, 0.2],
                "experienced_hosts": [99, 6],
            }
        ).to_csv(centroids_path, index=False)

        with pytest.raises(ValueError, match="experienced_hosts=99"):
            calc_host_immunity_fitness.validate_against_centroids(
                histories, centroids_path, np.array([0.0])
            )

    def test_burn_in_shifts_both_frames_the_same_way(
        self, tmp_path, calc_host_immunity_fitness
    ):
        tips = pd.DataFrame({"year": [0.5, 1.5, 2.5]})
        histories = pd.DataFrame({"year": [0.0, 1.0, 2.0]})

        tips_out, histories_out = calc_host_immunity_fitness.apply_burn_in(
            tips, histories, 1.0
        )

        assert tips_out["year"].tolist() == [0.5, 1.5]
        assert histories_out["year"].tolist() == [0.0, 1.0]

    def test_burn_in_emptying_a_frame_raises(self, tmp_path, calc_host_immunity_fitness):
        tips = pd.DataFrame({"year": [0.5]})
        histories = pd.DataFrame({"year": [0.0, 1.0]})

        with pytest.raises(ValueError, match="nothing to compute"):
            calc_host_immunity_fitness.apply_burn_in(tips, histories, 5.0)
