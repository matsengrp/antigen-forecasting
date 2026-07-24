"""Unit tests for scripts/aggregate_results.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "aggregate_results.py"


@pytest.fixture(scope="module")
def aggregate_results():
    """Import scripts/aggregate_results.py as a module."""
    spec = importlib.util.spec_from_file_location("aggregate_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_results"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_run(
    results_root: Path,
    data_root: Path,
    batch: str,
    config: str,
    run: int,
    *,
    with_scores: bool = True,
    with_tips: bool = True,
) -> None:
    """Create a synthetic per-run results + data tree for one run."""
    sim_id = f"{config}__run_{run}"
    sim_results = results_root / batch / sim_id
    sim_results.mkdir(parents=True)
    pd.DataFrame(
        {
            "pivot_date": ["2027-04-01", "2027-10-01"],
            "model": ["FGA", "GARW"],
            "location": ["north", "south"],
            "mae": [0.1 + run * 0.01, 0.2],
        }
    ).to_csv(sim_results / "growth_rate_scores.tsv", sep="\t", index=False)
    if with_scores:
        # Per-forecast-point rows (variant x date) that the summary collapses.
        pd.DataFrame(
            {
                "model": ["FGA", "FGA", "GARW", "GARW"],
                "location": ["north", "north", "north", "north"],
                "lead": [0, 0, 30, 30],
                "variant": [1, 2, 1, 2],
                "date": ["2027-04-01"] * 4,
                "MAE": [0.04, 0.06, 0.10, 0.12],
                "coverage_predictive": [0.9, 0.9, 0.8, 0.8],
            }
        ).to_csv(sim_results / "scores.tsv", sep="\t", index=False)

    if with_tips:
        sim_data = data_root / batch / sim_id
        sim_data.mkdir(parents=True)
        pd.DataFrame(
            {
                "year_bin": [10, 10, 11, 11],
                "ag1": [0.0, 1.0, 2.0, 3.0],
                "ag2": [0.0, 1.0, 2.0, 3.0],
                "variant_ag": [0, 1, 0, 1],
                "variant_tsne": [0, 1, 0, 1],
                "variant_phylo": [0, 0, 1, 1],
            }
        ).to_csv(sim_data / "tips_with_variants.tsv", sep="\t", index=False)


class TestParseRunIdentity:
    def test_parses_config_and_run(self, aggregate_results, tmp_path):
        config, run = aggregate_results.parse_run_identity(
            tmp_path / "muPhenotype_0.1__run_7"
        )
        assert config == "muPhenotype_0.1"
        assert run == 7

    def test_config_with_underscores(self, aggregate_results, tmp_path):
        config, run = aggregate_results.parse_run_identity(
            tmp_path / "nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0__run_3"
        )
        assert config == "nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0"
        assert run == 3

    def test_missing_delimiter_raises(self, aggregate_results, tmp_path):
        with pytest.raises(ValueError, match="__"):
            aggregate_results.parse_run_identity(tmp_path / "run_0")


class TestDiscoverRuns:
    def test_finds_sorted_run_dirs(self, aggregate_results, tmp_path):
        results_root = tmp_path / "results"
        data_root = tmp_path / "data"
        for run in range(3):
            _make_run(results_root, data_root, "b1", "cfgA", run)
        found = aggregate_results.discover_runs(results_root, "b1")
        assert [p.name for p in found] == ["cfgA__run_0", "cfgA__run_1", "cfgA__run_2"]

    def test_missing_batch_raises(self, aggregate_results, tmp_path):
        with pytest.raises(FileNotFoundError, match="Batch results"):
            aggregate_results.discover_runs(tmp_path / "results", "nope")

    def test_no_runs_raises(self, aggregate_results, tmp_path):
        (tmp_path / "results" / "b1").mkdir(parents=True)
        with pytest.raises(ValueError, match="No .*run"):
            aggregate_results.discover_runs(tmp_path / "results", "b1")


class TestMain:
    def test_writes_tagged_aggregates(self, aggregate_results, tmp_path):
        results_root = tmp_path / "results"
        data_root = tmp_path / "data"
        for config in ("cfgA", "cfgB"):
            for run in range(2):
                _make_run(results_root, data_root, "b1", config, run)
        out_dir = tmp_path / "out"

        aggregate_results.main(
            [
                "--batch",
                "b1",
                "--results-root",
                str(results_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(out_dir),
            ]
        )

        counts = pd.read_csv(out_dir / "variant_counts_over_time.csv")
        assert set(["batch", "config", "run", "method", "n_variants"]).issubset(
            counts.columns
        )
        assert set(counts["config"].unique()) == {"cfgA", "cfgB"}
        # 4 runs x 3 methods x 2 year bins.
        assert len(counts) == 4 * 3 * 2

        nid = pd.read_csv(out_dir / "method_agreement_nid.csv")
        assert set(["batch", "config", "run", "method_x", "method_y", "nid"]).issubset(
            nid.columns
        )
        # ag and tsne are identical partitions in the synthetic data -> NID 0.
        ag_tsne = nid[
            (nid["method_x"] == "variant_ag") & (nid["method_y"] == "variant_tsne")
        ]
        assert (ag_tsne["nid"].abs() < 1e-9).all()

        gr = pd.read_csv(out_dir / "growth_rate_scores_all.csv")
        assert {"batch", "config", "run"}.issubset(gr.columns)
        assert len(gr) == 4 * 2  # 4 runs x 2 rows each.

        # scores are summarized (collapsed over variant/date), not concatenated:
        # each run's 4 per-point rows reduce to 2 (model, location, lead) groups.
        scores = pd.read_csv(out_dir / "scores_summary.csv")
        assert {
            "batch",
            "config",
            "run",
            "model",
            "location",
            "lead",
            "MAE",
            "n_points",
        }.issubset(scores.columns)
        assert len(scores) == 4 * 2  # 4 runs x 2 groups each.
        assert (scores["n_points"] == 2).all()
        assert not (out_dir / "scores_all.csv").exists()

    def test_missing_scores_skipped_not_fatal(self, aggregate_results, tmp_path):
        results_root = tmp_path / "results"
        data_root = tmp_path / "data"
        _make_run(results_root, data_root, "b1", "cfgA", 0, with_scores=True)
        _make_run(results_root, data_root, "b1", "cfgA", 1, with_scores=False)
        out_dir = tmp_path / "out"

        aggregate_results.main(
            [
                "--batch",
                "b1",
                "--results-root",
                str(results_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(out_dir),
            ]
        )

        # scores_summary covers only the run that had scores.tsv.
        scores = pd.read_csv(out_dir / "scores_summary.csv")
        assert set(scores["run"].unique()) == {0}
        # growth-rate scores cover both runs (aggregation did not abort).
        gr = pd.read_csv(out_dir / "growth_rate_scores_all.csv")
        assert set(gr["run"].unique()) == {0, 1}

    def test_log_file_has_progress_and_summary(self, aggregate_results, tmp_path):
        results_root = tmp_path / "results"
        data_root = tmp_path / "data"
        _make_run(results_root, data_root, "b1", "cfgA", 0)
        out_dir = tmp_path / "out"
        log_file = tmp_path / "agg.log"

        aggregate_results.main(
            [
                "--batch",
                "b1",
                "--results-root",
                str(results_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(out_dir),
                "--log-file",
                str(log_file),
            ]
        )

        log_text = log_file.read_text()
        assert "[1/1] cfgA__run_0 -> ok" in log_text
        assert "1 processed, 0 skipped" in log_text
        assert "wrote" in log_text  # per-file row-count line

    def test_run_missing_all_inputs_is_skipped(self, aggregate_results, tmp_path):
        results_root = tmp_path / "results"
        data_root = tmp_path / "data"
        # A results dir with neither tips, growth_rate_scores, nor scores.
        (results_root / "b1" / "cfgA__run_0").mkdir(parents=True)
        out_dir = tmp_path / "out"
        log_file = tmp_path / "agg.log"

        aggregate_results.main(
            [
                "--batch",
                "b1",
                "--results-root",
                str(results_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(out_dir),
                "--log-file",
                str(log_file),
            ]
        )
        log_text = log_file.read_text()
        assert "SKIP" in log_text
        assert "0 processed, 1 skipped" in log_text


def _make_histories(
    experiments_root: Path,
    experiment: str,
    config: str,
    run: int,
    deme_label: str,
) -> None:
    """Write a synthetic out.histories.csv with a population-total deme row."""
    run_dir = experiments_root / experiment / "simulations" / config / f"run_{run}"
    run_dir.mkdir(parents=True)
    rows = []
    for year in range(4):
        for deme in ["north", "tropics", "south", deme_label]:
            rows.append(
                {
                    "year": float(year),
                    "deme": deme,
                    "ag1": 6.0 + year,
                    "ag2": 3.0 + 0.1 * year,
                    "naive_fraction": 0.15,
                    "experienced_hosts": 8000,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "out.histories.csv", index=False)


def _tips_with_coords() -> pd.DataFrame:
    """Tips with ag1/ag2/year and variant columns, enough for within-variant variance."""
    return pd.DataFrame(
        {
            "year": [0.5, 0.6, 1.5, 1.6, 2.5, 2.6],
            "year_bin": [0, 0, 1, 1, 2, 2],
            "ag1": [6.1, 6.4, 7.2, 7.5, 8.1, 8.6],
            "ag2": [3.0, 3.2, 3.4, 3.1, 3.6, 3.9],
            "variant_ag": [0, 0, 1, 1, 2, 2],
            "variant_tsne": [0, 1, 1, 1, 2, 2],
            "variant_phylo": [0, 0, 0, 1, 1, 2],
        }
    )


class TestFitnessVariance:
    def test_global_deme_produces_variance(self, aggregate_results, tmp_path):
        exp_root = tmp_path / "experiments"
        _make_histories(exp_root, "expA", "cfgA", 0, "global")
        out = aggregate_results._compute_fitness_variance(
            _tips_with_coords(), exp_root, "expA", "cfgA", 0, "out.histories.csv"
        )
        assert out is not None and not out.empty
        assert {"year", "method", "mean_variance"}.issubset(out.columns)
        assert set(out["method"].unique()) == {"ag", "tsne", "phylo"}

    def test_total_deme_fallback(self, aggregate_results, tmp_path):
        exp_root = tmp_path / "experiments"
        _make_histories(exp_root, "expA", "cfgA", 0, "total")
        out = aggregate_results._compute_fitness_variance(
            _tips_with_coords(), exp_root, "expA", "cfgA", 0, "out.histories.csv"
        )
        assert out is not None and not out.empty

    def test_no_population_total_deme_returns_none(self, aggregate_results, tmp_path):
        exp_root = tmp_path / "experiments"
        # Only regional demes, no global/total aggregate row.
        _make_histories(exp_root, "expA", "cfgA", 0, "north")
        out = aggregate_results._compute_fitness_variance(
            _tips_with_coords(), exp_root, "expA", "cfgA", 0, "out.histories.csv"
        )
        assert out is None

    def test_missing_file_returns_none(self, aggregate_results, tmp_path):
        out = aggregate_results._compute_fitness_variance(
            _tips_with_coords(),
            tmp_path / "experiments",
            "expA",
            "cfgA",
            0,
            "out.histories.csv",
        )
        assert out is None


def _make_raw_histories(
    experiments_root: Path,
    experiment: str,
    config: str,
    run: int,
    filename: str = "out.histories.raw.csv",
) -> None:
    """Write a synthetic out.histories.raw.csv with per-host immune memory rows."""
    run_dir = experiments_root / experiment / "simulations" / config / f"run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(4):
        for deme in ("north", "tropics", "south"):
            for host_id in range(8):
                # Two memory entries per host, drifting with the year.
                for infection_index in range(2):
                    rows.append(
                        {
                            "year": float(year),
                            "deme": deme,
                            "host_id": host_id,
                            "infection_index": infection_index,
                            "ag1": 6.0 + year + 0.3 * host_id + infection_index,
                            "ag2": 3.0 + 0.1 * year,
                            "naive_fraction": 0.2,
                        }
                    )
    pd.DataFrame(rows).to_csv(run_dir / filename, index=False)


def _named_tips_with_coords() -> pd.DataFrame:
    """Like _tips_with_coords but with the 'name' column the host-immunity path needs."""
    tips = _tips_with_coords()
    tips.insert(0, "name", [f"tip{i}" for i in range(len(tips))])
    return tips


def _host_immunity_config(aggregate_results, **overrides):
    """Build a HostImmunityConfig with test-friendly defaults."""
    kwargs = {
        "enabled": True,
        "histories_name": "out.histories.raw.csv",
        "n_hosts": 5,
        "seed": 42,
    }
    kwargs.update(overrides)
    return aggregate_results.HostImmunityConfig(**kwargs)


class TestHostImmunityVariance:
    """The opt-in per-host immune-history variance path."""

    def test_produces_variance_matching_the_centroid_schema(
        self, aggregate_results, tmp_path
    ):
        exp_root = tmp_path / "experiments"
        _make_raw_histories(exp_root, "expA", "cfgA", 0)

        out = aggregate_results._compute_host_immunity_variance(
            _named_tips_with_coords(),
            exp_root,
            "expA",
            "cfgA",
            0,
            _host_immunity_config(aggregate_results),
        )

        assert out is not None and not out.empty
        # Identical columns to _compute_fitness_variance, so the two aggregated tables
        # can be plotted against each other without any reshaping.
        assert list(out.columns) == ["year", "method", "mean_variance", "n_variants"]
        assert set(out["method"].unique()) == {"ag", "tsne", "phylo"}

    def test_missing_raw_file_returns_none(self, aggregate_results, tmp_path):
        out = aggregate_results._compute_host_immunity_variance(
            _named_tips_with_coords(),
            tmp_path / "experiments",
            "expA",
            "cfgA",
            0,
            _host_immunity_config(aggregate_results),
        )
        assert out is None

    def test_unreadable_histories_returns_none_rather_than_raising(
        self, aggregate_results, tmp_path
    ):
        # Fitness variance is best-effort: one broken run must not abort the batch.
        exp_root = tmp_path / "experiments"
        run_dir = exp_root / "expA" / "simulations" / "cfgA" / "run_0"
        run_dir.mkdir(parents=True)
        (run_dir / "out.histories.raw.csv").write_text("year,deme\n0.0,north\n")

        out = aggregate_results._compute_host_immunity_variance(
            _named_tips_with_coords(),
            exp_root,
            "expA",
            "cfgA",
            0,
            _host_immunity_config(aggregate_results),
        )
        assert out is None

    def test_using_all_hosts_is_supported(self, aggregate_results, tmp_path):
        exp_root = tmp_path / "experiments"
        _make_raw_histories(exp_root, "expA", "cfgA", 0)

        out = aggregate_results._compute_host_immunity_variance(
            _named_tips_with_coords(),
            exp_root,
            "expA",
            "cfgA",
            0,
            _host_immunity_config(aggregate_results, n_hosts=None),
        )
        assert out is not None and not out.empty

    def test_seed_is_respected(self, aggregate_results, tmp_path):
        exp_root = tmp_path / "experiments"
        _make_raw_histories(exp_root, "expA", "cfgA", 0)
        tips = _named_tips_with_coords()

        def run_with(seed):
            return aggregate_results._compute_host_immunity_variance(
                tips,
                exp_root,
                "expA",
                "cfgA",
                0,
                _host_immunity_config(aggregate_results, seed=seed),
            )["mean_variance"].to_numpy()

        assert (run_with(42) == run_with(42)).all()
