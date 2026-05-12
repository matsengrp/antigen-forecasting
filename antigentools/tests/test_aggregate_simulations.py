"""Unit and integration tests for scripts/aggregate_simulations.py."""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "aggregate_simulations.py"
)


@pytest.fixture(scope="module")
def agg():
    """Import scripts/aggregate_simulations.py as a module."""
    spec = importlib.util.spec_from_file_location("aggregate_simulations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_simulations"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# parse_params_from_name
# ---------------------------------------------------------------------------


class TestParseParamsFromName:
    def test_float_values(self, agg):
        result = agg.parse_params_from_name(
            "nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0"
        )
        assert result == {"nonEpitopeAcceptance": 0.1, "epitopeAcceptance": 1.0}

    def test_int_value(self, agg):
        result = agg.parse_params_from_name("run_5_epitopeAcceptance_1")
        assert result == {"run": 5, "epitopeAcceptance": 1}

    def test_string_value(self, agg):
        result = agg.parse_params_from_name("model_FGA_epitopeAcceptance_0.5")
        assert result == {"model": "FGA", "epitopeAcceptance": 0.5}

    def test_odd_tokens_raises(self, agg):
        with pytest.raises(ValueError, match="odd"):
            agg.parse_params_from_name("key_0.1_orphan")


# ---------------------------------------------------------------------------
# validate_and_merge_params
# ---------------------------------------------------------------------------


class TestValidateAndMergeParams:
    def test_mismatch_raises(self, agg):
        with pytest.raises(ValueError, match="mismatch"):
            agg.validate_and_merge_params(
                {"epitopeAcceptance": 0.5},
                {"epitopeAcceptance": 0.9},
                "sim123",
            )

    def test_name_only_raises(self, agg):
        with pytest.raises(ValueError, match="stale rename"):
            agg.validate_and_merge_params(
                {"staleKey": 1.0},
                {},
                "sim123",
            )

    def test_yaml_only_warns(self, agg):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = agg.validate_and_merge_params(
                {},
                {"extraParam": 42},
                "sim123",
            )
        assert any("extraParam" in str(w.message) for w in caught)
        assert result == {"extraParam": 42}

    def test_happy_path_union(self, agg):
        result = agg.validate_and_merge_params(
            {"epitopeAcceptance": 0.5},
            {"epitopeAcceptance": 0.5, "netau": 100},
            "sim123",
        )
        assert result["epitopeAcceptance"] == 0.5
        assert result["netau"] == 100


# ---------------------------------------------------------------------------
# read_summary_stats
# ---------------------------------------------------------------------------


class TestReadSummaryStats:
    def test_reads_parameter_value_pairs(self, agg, tmp_path):
        summary = tmp_path / "run-out.summary"
        summary.write_text(
            "# header comment\ndiversity\t3.769\ntmrca\t3.6\nI\t141398.19\n"
        )
        result = agg.read_summary_stats(summary)
        assert math.isclose(result["diversity"], 3.769)
        assert math.isclose(result["tmrca"], 3.6)
        assert math.isclose(result["I"], 141398.19)

    def test_missing_file_raises(self, agg, tmp_path):
        with pytest.raises(FileNotFoundError):
            agg.read_summary_stats(tmp_path / "nonexistent.summary")


# ---------------------------------------------------------------------------
# count_unique_sequences
# ---------------------------------------------------------------------------


class TestLoadTips:
    def test_counts_rows(self, agg, tmp_path):
        build = "batch/param__run_0"
        tips_dir = tmp_path / build / "antigen-outputs"
        tips_dir.mkdir(parents=True)
        (tips_dir / "unique_tips.csv").write_text(
            "name,year\nseq1,2027.0\nseq2,2027.1\nseq3,2027.2\n"
        )
        df = agg.load_tips(tmp_path, build)
        assert len(df) == 3

    def test_missing_raises(self, agg, tmp_path):
        with pytest.raises(FileNotFoundError, match="unique_tips.csv"):
            agg.load_tips(tmp_path, "batch/missing__run_0")


# ---------------------------------------------------------------------------
# calculate_antigenic_movement_per_year
# ---------------------------------------------------------------------------


class TestCalculateAntigeicMovementPerYear:
    def test_known_linear_movement(self):
        from antigentools.antigen_reader import calculate_antigenic_movement_per_year

        # 3 viruses spaced 1 year apart, moving 1 unit/year in ag1
        df = pd.DataFrame(
            {
                "year": [2025.0, 2026.0, 2027.0],
                "ag1": [0.0, 1.0, 2.0],
                "ag2": [0.0, 0.0, 0.0],
            }
        )
        result = calculate_antigenic_movement_per_year(df)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_single_tip_returns_zero(self):
        from antigentools.antigen_reader import calculate_antigenic_movement_per_year

        df = pd.DataFrame({"year": [2026.0], "ag1": [1.0], "ag2": [0.0]})
        assert calculate_antigenic_movement_per_year(df) == 0.0

    def test_all_same_time_returns_zero(self):
        from antigentools.antigen_reader import calculate_antigenic_movement_per_year

        df = pd.DataFrame(
            {
                "year": [2026.0, 2026.0, 2026.0],
                "ag1": [0.0, 1.0, 2.0],
                "ag2": [0.0, 0.0, 0.0],
            }
        )
        assert calculate_antigenic_movement_per_year(df) == 0.0


# ---------------------------------------------------------------------------
# count_branch_mutations (from antigentools.antigen_reader)
# ---------------------------------------------------------------------------


def _make_branch_line(
    name: str,
    trunk: int,
    tip: int,
    n_epi: int,
    n_non: int,
    *,
    date: float = 2027.0,
) -> str:
    """Build one comma-separated node string matching BRANCHES_COLUMNS."""
    cols = [name, date, 1.0, trunk, tip, 0, 0, 0, "ACGT", 0.5, 0.3, n_epi, n_non]
    return ",".join(str(c) for c in cols)


def _write_branches(path: Path, entries: list[tuple]) -> None:
    """Write a .branches file; entries = (name, trunk, tip, n_epi, n_non)."""
    lines = []
    for i, (name, trunk, tip, n_epi, n_non) in enumerate(entries):
        child = _make_branch_line(
            name, trunk, tip, n_epi, n_non, date=2027.0 + i * 0.01
        )
        parent_name = f"{name}_parent"
        # parent is always non-trunk side branch for simplicity
        parent = _make_branch_line(
            parent_name, 0, 0, 0, 0, date=2027.0 + i * 0.01 - 0.5
        )
        lines.append(f"{child}\t{parent}")
    path.write_text("\n".join(lines))


class TestCountBranchMutations:
    def test_trunk_and_side_counts(self, tmp_path):
        from antigentools.antigen_reader import count_branch_mutations

        branches = tmp_path / "run-out.branches"
        # trunk: 2 epi, 1 non; side: 1 epi, 3 non
        _write_branches(
            branches,
            [
                ("trunk_node", 1, 0, 2, 1),
                ("side_node", 0, 0, 1, 3),
            ],
        )
        result = count_branch_mutations(str(branches))
        assert result["trunk_epitope_mutations"] == 2.0
        assert result["trunk_non_epitope_mutations"] == 1.0
        assert math.isclose(result["trunk_epitope_to_non-epitope_ratio"], 2.0)
        assert result["side_branch_epitope_mutations"] == 1.0
        assert result["side_branch_non_epitope_mutations"] == 3.0
        assert math.isclose(result["side_branch_epitope_to_non-epitope_ratio"], 1 / 3)

    def test_zero_denominator_gives_nan(self, tmp_path):
        from antigentools.antigen_reader import count_branch_mutations

        branches = tmp_path / "run-out.branches"
        _write_branches(branches, [("trunk_node", 1, 0, 5, 0)])
        result = count_branch_mutations(str(branches))
        assert math.isnan(result["trunk_epitope_to_non-epitope_ratio"])

    def test_missing_file_raises(self, tmp_path):
        from antigentools.antigen_reader import count_branch_mutations

        with pytest.raises(FileNotFoundError):
            count_branch_mutations(str(tmp_path / "nonexistent.branches"))


# ---------------------------------------------------------------------------
# discover_builds
# ---------------------------------------------------------------------------


class TestDiscoverBuilds:
    def test_finds_builds(self, agg, tmp_path):
        for sim_id in ["param_a__run_0", "param_b__run_1"]:
            (tmp_path / "batch" / sim_id / "antigen-outputs").mkdir(parents=True)
        result = agg.discover_builds(tmp_path, "batch")
        assert result == ["batch/param_a__run_0", "batch/param_b__run_1"]

    def test_no_builds_raises(self, agg, tmp_path):
        with pytest.raises(ValueError, match="No builds"):
            agg.discover_builds(tmp_path, "empty_batch")


# ---------------------------------------------------------------------------
# aggregate_simulations — integration
# ---------------------------------------------------------------------------


def _make_sim_tree(
    tmp_path: Path,
    batch_name: str,
    experiment: str,
    param_set: str,
    run_id: str,
    *,
    params: dict | None = None,
    n_tips: int = 5,
) -> tuple[Path, Path]:
    """Create the minimal directory tree for one simulation."""
    if params is None:
        params = {"epitopeAcceptance": 0.75, "nonEpitopeAcceptance": 0.1}

    sim_id = f"{param_set}__{run_id}"
    build = f"{batch_name}/{sim_id}"

    # experiments tree
    sim_path = tmp_path / "experiments" / experiment / param_set / run_id
    out_dir = sim_path / "output"
    out_dir.mkdir(parents=True)
    (sim_path.parent / "parameters.yml").write_text(yaml.safe_dump(params))

    # run-out.summary
    summary_lines = ["# header\n"] + [
        f"{k}\t{v}\n"
        for k, v in {
            "diversity": 3.77,
            "tmrca": 3.60,
            "I": 141398.0,
            "N": 90000000,
            "R": 0.0,
            "S": 89858602.0,
            "antigenicDiversity": 0.58,
            "cases": 198000.0,
            "endDate": 30.0,
        }.items()
    ]
    (out_dir / "run-out.summary").write_text("".join(summary_lines))

    # run-out.branches
    branches = out_dir / "run-out.branches"
    _write_branches(branches, [("trunk_n", 1, 0, 3, 2), ("side_n", 0, 0, 1, 4)])

    # data tree
    data_dir = tmp_path / "data" / build / "antigen-outputs"
    data_dir.mkdir(parents=True)
    rows = [
        {"name": f"s{i}", "year": 2027.0 + i * 0.1, "ag1": i * 0.1, "ag2": i * 0.05}
        for i in range(n_tips)
    ]
    pd.DataFrame(rows).to_csv(data_dir / "unique_tips.csv", index=False)

    return tmp_path / "experiments", tmp_path / "data"


class TestAggregateSimulations:
    def test_end_to_end(self, agg, tmp_path):
        experiments_root, data_root = _make_sim_tree(
            tmp_path,
            batch_name="2026-04-29-all-sims",
            experiment="2026-01-06-runs",
            param_set="nonEpitopeAcceptance_0.1_epitopeAcceptance_0.75",
            run_id="run_0",
        )
        df = agg.aggregate_simulations(
            experiments_root=experiments_root,
            experiment="2026-01-06-runs",
            data_root=data_root,
            batch_name="2026-04-29-all-sims",
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert (
            row["build"]
            == "2026-04-29-all-sims/nonEpitopeAcceptance_0.1_epitopeAcceptance_0.75__run_0"
        )
        assert row["batch_name"] == "2026-04-29-all-sims"
        assert row["experiment"] == "2026-01-06-runs"
        assert row["run"] == 0
        assert row["n_unique_sequences"] == 5
        assert row["epitopeAcceptance"] == 0.75
        assert "diversity" in df.columns
        assert "trunk_epitope_mutations" in df.columns
        assert row["trunk_epitope_mutations"] == 3.0

    def test_missing_summary_warns_and_skips(self, agg, tmp_path):
        experiments_root, data_root = _make_sim_tree(
            tmp_path,
            batch_name="batch",
            experiment="exp",
            param_set="epitopeAcceptance_0.5_nonEpitopeAcceptance_0.1",
            run_id="run_0",
            params={"epitopeAcceptance": 0.5, "nonEpitopeAcceptance": 0.1},
        )
        # Remove the summary file
        summary = (
            experiments_root
            / "exp"
            / "epitopeAcceptance_0.5_nonEpitopeAcceptance_0.1"
            / "run_0"
            / "output"
            / "run-out.summary"
        )
        summary.unlink()

        with pytest.warns(UserWarning, match="run-out.summary"):
            with pytest.raises(ValueError, match="No complete simulations"):
                agg.aggregate_simulations(
                    experiments_root=experiments_root,
                    experiment="exp",
                    data_root=data_root,
                    batch_name="batch",
                )

    def test_all_missing_raises(self, agg, tmp_path):
        # No builds exist
        (tmp_path / "data" / "empty_batch").mkdir(parents=True)
        with pytest.raises(ValueError):
            agg.aggregate_simulations(
                experiments_root=tmp_path / "experiments",
                experiment="exp",
                data_root=tmp_path / "data",
                batch_name="empty_batch",
            )
