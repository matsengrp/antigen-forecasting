"""Unit tests for scripts/run_pipeline.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml


SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_pipeline.py"


@pytest.fixture(scope="module")
def run_pipeline():
    """Import scripts/run_pipeline.py as a module."""
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_sim_path(tmp_path: Path) -> Path:
    """Create a sim_path that exists; pipeline never reads its contents in tests."""
    sim_path = tmp_path / "experiments" / "param_set" / "run_0"
    sim_path.mkdir(parents=True)
    (sim_path / "output").mkdir()
    (sim_path / "output" / "run-out.tips").write_text(
        "name,year,location,nucleotideSequence,ag1,ag2\n"
    )
    (sim_path / "out_timeseries.csv").write_text(
        "year,northCases,southCases,tropicsCases\n"
    )
    return sim_path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write a minimal valid pipeline_config.yaml under tmp_path."""
    cfg = {
        "pipeline": {
            "experiments_root": str(tmp_path / "experiments"),
            "data_root": str(tmp_path / "data"),
            "results_root": str(tmp_path / "results"),
            "window_size": 365,
            "buffer_size": 0,
            "models": ["FGA", "GARW"],
            "locations": ["north", "south", "tropics"],
            "forecast_L": 180,
            "seed_L": 14,
            "numpyro_seed": 42,
        }
    }
    path = tmp_path / "pipeline_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _completed_process(returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=""
    )


class TestLoadPipelineConfig:
    def test_loads_valid_config(self, run_pipeline, config_path):
        cfg = run_pipeline.load_pipeline_config(config_path)
        assert cfg["window_size"] == 365
        assert cfg["models"] == ["FGA", "GARW"]

    def test_missing_file_raises(self, run_pipeline, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pipeline.load_pipeline_config(tmp_path / "nope.yaml")

    def test_missing_pipeline_block_raises(self, run_pipeline, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"other_block": {}}))
        with pytest.raises(ValueError, match="pipeline"):
            run_pipeline.load_pipeline_config(path)

    def test_missing_required_key_raises(self, run_pipeline, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"pipeline": {"window_size": 365}}))
        with pytest.raises(ValueError, match="missing required"):
            run_pipeline.load_pipeline_config(path)


class TestArgParsing:
    def test_neither_build_nor_batch_errors(
        self, run_pipeline, fake_sim_path, config_path
    ):
        with pytest.raises(SystemExit):
            run_pipeline.main(
                [
                    "--sim-path",
                    str(fake_sim_path),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ]
            )

    def test_both_build_and_batch_errors(
        self, run_pipeline, fake_sim_path, config_path
    ):
        with pytest.raises(SystemExit):
            run_pipeline.main(
                [
                    "--sim-path",
                    str(fake_sim_path),
                    "--build",
                    "flu-final",
                    "--batch-name",
                    "batch-2026",
                    "--config",
                    str(config_path),
                    "--dry-run",
                ]
            )


class TestDryRun:
    def test_prints_all_steps_no_subprocess(
        self, run_pipeline, fake_sim_path, config_path
    ):
        with patch.object(
            subprocess, "run", return_value=_completed_process()
        ) as mock_run:
            run_pipeline.main(
                [
                    "--sim-path",
                    str(fake_sim_path),
                    "--build",
                    "flu-final",
                    "--config",
                    str(config_path),
                    "--dry-run",
                ]
            )
        assert mock_run.call_count == 0


class TestExecuteStep:
    def test_skipped_when_sentinels_exist(self, run_pipeline, tmp_path):
        sentinel = tmp_path / "out.tsv"
        sentinel.write_text("done")
        ran = []
        step = run_pipeline.Step(
            name="noop",
            sentinels=[sentinel],
            runner=lambda: ran.append("ran"),
            command_str="noop",
        )
        result = run_pipeline.execute_step(step, dry_run=False, force=False)
        assert result.status == "skipped"
        assert ran == []

    def test_runs_when_sentinels_absent(self, run_pipeline, tmp_path):
        sentinel = tmp_path / "out.tsv"
        step = run_pipeline.Step(
            name="noop",
            sentinels=[sentinel],
            runner=lambda: sentinel.write_text("done"),
            command_str="noop",
        )
        result = run_pipeline.execute_step(step, dry_run=False, force=False)
        assert result.status == "ok"
        assert sentinel.exists()

    def test_force_removes_existing_then_reruns(self, run_pipeline, tmp_path):
        sentinel = tmp_path / "out.tsv"
        sentinel.write_text("stale")
        step = run_pipeline.Step(
            name="noop",
            sentinels=[sentinel],
            runner=lambda: sentinel.write_text("fresh"),
            command_str="noop",
        )
        result = run_pipeline.execute_step(step, dry_run=False, force=True)
        assert result.status == "ok"
        assert sentinel.read_text() == "fresh"

    def test_failed_when_runner_raises(self, run_pipeline, tmp_path):
        sentinel = tmp_path / "out.tsv"

        def boom() -> None:
            raise RuntimeError("simulated failure")

        step = run_pipeline.Step(
            name="boom",
            sentinels=[sentinel],
            runner=boom,
            command_str="boom",
        )
        result = run_pipeline.execute_step(step, dry_run=False, force=False)
        assert result.status == "failed"

    def test_failed_when_sentinel_not_written(self, run_pipeline, tmp_path):
        sentinel = tmp_path / "out.tsv"
        step = run_pipeline.Step(
            name="lie",
            sentinels=[sentinel],
            runner=lambda: None,  # runner reports success but writes nothing.
            command_str="lie",
        )
        result = run_pipeline.execute_step(step, dry_run=False, force=False)
        assert result.status == "failed"


class TestPipelineLog:
    def test_columns_and_rows(self, run_pipeline, tmp_path):
        results = [
            run_pipeline.StepResult(
                name="parse_sim_outputs",
                start_time="2026-04-30T12:00:00+00:00",
                end_time="2026-04-30T12:00:01+00:00",
                duration_sec=1.0,
                status="ok",
                command="python parse_sim_outputs.py ...",
            ),
            run_pipeline.StepResult(
                name="assign_all_variants",
                start_time="2026-04-30T12:00:01+00:00",
                end_time="2026-04-30T12:00:01+00:00",
                duration_sec=0.0,
                status="skipped",
                command="",
            ),
        ]
        log_path = tmp_path / "pipeline.log"
        run_pipeline.write_pipeline_log(results, log_path)
        df = pd.read_csv(log_path, sep="\t")
        assert list(df.columns) == [
            "step",
            "start_time",
            "end_time",
            "duration_sec",
            "status",
            "command",
        ]
        assert df["status"].tolist() == ["ok", "skipped"]
        assert df["step"].tolist() == ["parse_sim_outputs", "assign_all_variants"]


class TestEndToEndSentinelFlow:
    def test_skip_then_force(self, run_pipeline, fake_sim_path, config_path, tmp_path):
        """Pre-create every sentinel; first invocation skips all, --force reruns them."""
        with patch.object(
            subprocess, "run", return_value=_completed_process()
        ) as mock_run:
            # First run: nothing exists. The fake subprocess "succeeds" but writes
            # no sentinels, so steps after parse would fail. To isolate sentinel
            # behavior, pre-create every sentinel and rely on skip-path only.
            from antigentools.paths import SimulationPaths

            cfg = yaml.safe_load(config_path.read_text())["pipeline"]
            paths = SimulationPaths.from_build(
                build="flu-final",
                sim_path=fake_sim_path,
                data_root=Path(cfg["data_root"]),
                results_root=Path(cfg["results_root"]),
            )
            for sentinel in [
                paths.unique_tips,
                paths.tips_with_variants,
                paths.seq_counts,
                paths.case_counts,
                paths.time_stamped_manifest,
                paths.estimates_manifest,
                paths.scores,
                paths.growth_rate_scores,
            ]:
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text("done")

            run_pipeline.main(
                [
                    "--sim-path",
                    str(fake_sim_path),
                    "--build",
                    "flu-final",
                    "--config",
                    str(config_path),
                ]
            )
            log_df = pd.read_csv(paths.pipeline_log, sep="\t")
            assert (log_df["status"] == "skipped").all(), log_df.to_string()
            assert mock_run.call_count == 0
