"""Unit tests for antigentools.paths.SimulationPaths."""

from pathlib import Path

import pytest

from antigentools.paths import SimulationPaths


@pytest.fixture
def fake_sim_path(tmp_path: Path) -> Path:
    """Create a sim_path that exists for tests that need it."""
    sim_path = (
        tmp_path
        / "experiments"
        / "2026-01-06-mutation-bug-fix-runs"
        / "nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0"
        / "run_0"
    )
    sim_path.mkdir(parents=True)
    return sim_path


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """Return (data_root, results_root) under tmp_path."""
    return tmp_path / "data", tmp_path / "results"


class TestFromBuild:
    def test_single_dataset_resolves_paths(self, fake_sim_path, roots):
        data_root, results_root = roots
        paths = SimulationPaths.from_build(
            build="flu-final",
            sim_path=fake_sim_path,
            data_root=data_root,
            results_root=results_root,
        )
        assert paths.build == "flu-final"
        assert paths.antigen_outputs == data_root / "flu-final" / "antigen-outputs"
        assert (
            paths.unique_tips
            == data_root / "flu-final" / "antigen-outputs" / "unique_tips.csv"
        )
        assert paths.results == results_root / "flu-final"
        assert paths.scores == results_root / "flu-final" / "scores.tsv"

    def test_missing_sim_path_raises(self, tmp_path, roots):
        missing = tmp_path / "does-not-exist"
        data_root, results_root = roots
        with pytest.raises(FileNotFoundError):
            SimulationPaths.from_build(
                build="flu-final",
                sim_path=missing,
                data_root=data_root,
                results_root=results_root,
            )


class TestFromSimPath:
    def test_multi_underscore_paramset(self, fake_sim_path, roots):
        data_root, results_root = roots
        paths = SimulationPaths.from_sim_path(
            sim_path=fake_sim_path,
            data_root=data_root,
            results_root=results_root,
            batch_name="2026-04-29-all-sims",
        )
        assert paths.build == (
            "2026-04-29-all-sims/nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0__run_0"
        )

    def test_rejects_double_underscore_in_paramset(self, tmp_path, roots):
        sim_path = tmp_path / "experiments" / "exp" / "param__set" / "run_0"
        sim_path.mkdir(parents=True)
        data_root, results_root = roots
        with pytest.raises(ValueError, match="__"):
            SimulationPaths.from_sim_path(
                sim_path=sim_path,
                data_root=data_root,
                results_root=results_root,
                batch_name="batch",
            )

    def test_rejects_double_underscore_in_runid(self, tmp_path, roots):
        sim_path = tmp_path / "experiments" / "exp" / "param" / "run__0"
        sim_path.mkdir(parents=True)
        data_root, results_root = roots
        with pytest.raises(ValueError, match="__"):
            SimulationPaths.from_sim_path(
                sim_path=sim_path,
                data_root=data_root,
                results_root=results_root,
                batch_name="batch",
            )

    def test_rejects_slash_in_batch_name(self, fake_sim_path, roots):
        data_root, results_root = roots
        with pytest.raises(ValueError, match="batch_name"):
            SimulationPaths.from_sim_path(
                sim_path=fake_sim_path,
                data_root=data_root,
                results_root=results_root,
                batch_name="batch/with/slash",
            )

    def test_rejects_empty_paramset(self, tmp_path, roots, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "run_0").mkdir()
        data_root, results_root = roots
        with pytest.raises(ValueError, match="param-set"):
            SimulationPaths.from_sim_path(
                sim_path=Path("run_0"),
                data_root=data_root,
                results_root=results_root,
                batch_name="batch",
            )

    def test_missing_sim_path_raises(self, tmp_path, roots):
        missing = tmp_path / "experiments" / "exp" / "param" / "run_0"
        data_root, results_root = roots
        with pytest.raises(FileNotFoundError):
            SimulationPaths.from_sim_path(
                sim_path=missing,
                data_root=data_root,
                results_root=results_root,
                batch_name="batch",
            )


class TestPathProperties:
    @pytest.fixture
    def paths(self, fake_sim_path, roots) -> SimulationPaths:
        data_root, results_root = roots
        return SimulationPaths.from_sim_path(
            sim_path=fake_sim_path,
            data_root=data_root,
            results_root=results_root,
            batch_name="2026-04-29-all-sims",
        )

    def test_raw_inputs(self, paths, fake_sim_path):
        assert paths.raw_tips == fake_sim_path / "output" / "run-out.tips"
        assert paths.raw_fasta == fake_sim_path / "output" / "run-out.fasta"
        assert paths.raw_timeseries == fake_sim_path / "out_timeseries.csv"
        assert paths.raw_parameters_yml == fake_sim_path.parent / "parameters.yml"

    def test_parsed_outputs(self, paths):
        ao = paths.antigen_outputs
        assert paths.tips == ao / "tips.csv"
        assert paths.unique_tips == ao / "unique_tips.csv"
        assert paths.unique_sequences == ao / "unique_sequences.fasta"
        assert paths.timeseries == ao / "out_timeseries.csv"

    def test_per_build_derived(self, paths, roots):
        data_root, _ = roots
        build_dir = data_root / paths.build
        assert paths.variant_assignment == build_dir / "variant-assignment"
        assert paths.tips_with_variants == build_dir / "tips_with_variants.tsv"
        assert paths.seq_counts == build_dir / "seq_counts.tsv"
        assert paths.case_counts == build_dir / "case_counts.tsv"
        assert paths.time_stamped == build_dir / "time-stamped"
        assert (
            paths.time_stamped_manifest == build_dir / "time-stamped" / "MANIFEST.tsv"
        )

    def test_results(self, paths, roots):
        _, results_root = roots
        results_dir = results_root / paths.build
        assert paths.results == results_dir
        assert paths.estimates == results_dir / "estimates"
        assert paths.estimates_manifest == results_dir / "estimates" / "MANIFEST.tsv"
        assert paths.growth_rates == results_dir / "growth-rates"
        assert paths.scores == results_dir / "scores.tsv"
        assert paths.growth_rate_scores == results_dir / "growth_rate_scores.tsv"
        assert paths.pipeline_log == results_dir / "pipeline.log"

    def test_string_inputs_coerced_to_path(self, fake_sim_path, tmp_path):
        paths = SimulationPaths.from_build(
            build="flu-final",
            sim_path=str(fake_sim_path),
            data_root=str(tmp_path / "data"),
            results_root=str(tmp_path / "results"),
        )
        assert isinstance(paths.sim_path, Path)
        assert isinstance(paths.data_root, Path)
        assert isinstance(paths.results_root, Path)
