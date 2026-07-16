"""Unit tests for scripts/run_all_simulations.py."""

from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "run_all_simulations.py"


@pytest.fixture(scope="module")
def run_all_sims():
    """Import scripts/run_all_simulations.py as a module."""
    spec = importlib.util.spec_from_file_location("run_all_simulations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_all_simulations"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_valid_sim(
    experiments_root: Path, experiment: str, param_set: str, run_id: str
) -> Path:
    """Create a sim directory with all required input files.

    Runs live at ``<experiments_root>/<experiment>/simulations/<config>/run_*/``.
    """
    sim_path = experiments_root / experiment / "simulations" / param_set / run_id
    sim_path.mkdir(parents=True)
    (sim_path / "output").mkdir()
    (sim_path / "output" / "run-out.tips").write_text("name\n")
    (sim_path / "output" / "run-out.fasta").write_text(">a\nACGT\n")
    (sim_path / "out_timeseries.csv").write_text("year\n")
    return sim_path


class TestDiscoverSimPaths:
    def test_finds_valid_sim_dirs(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        sim0 = _make_valid_sim(root, "exp1", "ps1", "run_0")
        sim1 = _make_valid_sim(root, "exp1", "ps1", "run_1")
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert set(found) == {sim0, sim1}
        assert skipped == []

    def test_returns_sorted_paths(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        for i in range(5):
            _make_valid_sim(root, "exp1", "ps1", f"run_{i}")
        found, _ = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == sorted(found)

    def test_no_run_dirs_raises(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        (root / "exp1" / "simulations" / "ps1").mkdir(parents=True)
        with pytest.raises(ValueError, match="No run_\\*"):
            run_all_sims.discover_sim_paths(root, "exp1", "ps1", all_configs=False)

    def test_missing_simulations_dir_raises(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        (root / "exp1").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="simulations/"):
            run_all_sims.discover_sim_paths(root, "exp1", "ps1", all_configs=False)

    def test_missing_tips_is_skipped(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        sim = _make_valid_sim(root, "exp1", "ps1", "run_0")
        (sim / "output" / "run-out.tips").unlink()
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == []
        assert len(skipped) == 1
        assert skipped[0][0] == sim
        assert "tips" in skipped[0][1]

    def test_missing_timeseries_is_skipped(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        sim = _make_valid_sim(root, "exp1", "ps1", "run_0")
        (sim / "out_timeseries.csv").unlink()
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == []
        assert len(skipped) == 1
        assert "timeseries" in skipped[0][1]

    def test_legacy_timeseries_name_accepted(self, run_all_sims, tmp_path):
        """Older runs with out.timeseries (not .csv) are still valid."""
        root = tmp_path / "experiments"
        sim = _make_valid_sim(root, "exp1", "ps1", "run_0")
        (sim / "out_timeseries.csv").unlink()
        (sim / "out.timeseries").write_text("year\n")
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == [sim]
        assert skipped == []

    def test_missing_fasta_is_skipped(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        sim = _make_valid_sim(root, "exp1", "ps1", "run_0")
        (sim / "output" / "run-out.fasta").unlink()
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == []
        assert "fasta" in skipped[0][1]

    def test_ignores_files_named_run(self, run_all_sims, tmp_path):
        """Files named run_* at the sim level are not returned."""
        root = tmp_path / "experiments"
        sim = _make_valid_sim(root, "exp1", "ps1", "run_0")
        (root / "exp1" / "simulations" / "ps1" / "run_extra.txt").write_text("")
        found, _ = run_all_sims.discover_sim_paths(
            root, "exp1", "ps1", all_configs=False
        )
        assert found == [sim]

    def test_all_configs_discovers_across_cells(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        sim_a = _make_valid_sim(root, "exp1", "muP_0.1", "run_0")
        sim_b = _make_valid_sim(root, "exp1", "muP_0.2", "run_0")
        found, skipped = run_all_sims.discover_sim_paths(
            root, "exp1", None, all_configs=True
        )
        assert set(found) == {sim_a, sim_b}
        assert skipped == []

    def test_param_set_and_all_configs_mutually_exclusive(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        with pytest.raises(ValueError, match="exactly one"):
            run_all_sims.discover_sim_paths(root, "exp1", "ps1", all_configs=True)

    def test_neither_param_set_nor_all_configs_raises(self, run_all_sims, tmp_path):
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        with pytest.raises(ValueError, match="exactly one"):
            run_all_sims.discover_sim_paths(root, "exp1", None, all_configs=False)


class TestIsSimComplete:
    def test_both_sentinels_exist_returns_true(self, run_all_sims, tmp_path):
        sim_path = tmp_path / "exp" / "ps" / "run_0"
        sim_path.mkdir(parents=True)
        results_root = tmp_path / "results"
        sim_id = "ps__run_0"
        scores_dir = results_root / "batch1" / sim_id
        scores_dir.mkdir(parents=True)
        (scores_dir / "scores.tsv").write_text("")
        (scores_dir / "growth_rate_scores.tsv").write_text("")
        assert run_all_sims.is_sim_complete(sim_path, results_root, "batch1")

    def test_missing_scores_returns_false(self, run_all_sims, tmp_path):
        sim_path = tmp_path / "exp" / "ps" / "run_0"
        sim_path.mkdir(parents=True)
        results_root = tmp_path / "results"
        scores_dir = results_root / "batch1" / "ps__run_0"
        scores_dir.mkdir(parents=True)
        (scores_dir / "growth_rate_scores.tsv").write_text("")
        assert not run_all_sims.is_sim_complete(sim_path, results_root, "batch1")

    def test_missing_growth_rate_scores_returns_false(self, run_all_sims, tmp_path):
        sim_path = tmp_path / "exp" / "ps" / "run_0"
        sim_path.mkdir(parents=True)
        results_root = tmp_path / "results"
        scores_dir = results_root / "batch1" / "ps__run_0"
        scores_dir.mkdir(parents=True)
        (scores_dir / "scores.tsv").write_text("")
        assert not run_all_sims.is_sim_complete(sim_path, results_root, "batch1")

    def test_neither_sentinel_returns_false(self, run_all_sims, tmp_path):
        sim_path = tmp_path / "exp" / "ps" / "run_0"
        sim_path.mkdir(parents=True)
        assert not run_all_sims.is_sim_complete(
            sim_path, tmp_path / "results", "batch1"
        )

    def test_sim_id_derived_correctly(self, run_all_sims, tmp_path):
        """Sim id is <param_set>__<run_id>; wrong derivation causes False."""
        param_set = "nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0"
        run_id = "run_3"
        sim_path = tmp_path / "exp" / param_set / run_id
        sim_path.mkdir(parents=True)
        results_root = tmp_path / "results"
        correct_id = f"{param_set}__{run_id}"
        scores_dir = results_root / "batch1" / correct_id
        scores_dir.mkdir(parents=True)
        (scores_dir / "scores.tsv").write_text("")
        (scores_dir / "growth_rate_scores.tsv").write_text("")
        assert run_all_sims.is_sim_complete(sim_path, results_root, "batch1")


class TestBuildRunPipelineCmd:
    def test_set_thread_caps_false_excludes_flag(self, run_all_sims, tmp_path):
        cmd = run_all_sims.build_run_pipeline_cmd(
            sim_path=tmp_path / "run_0",
            batch_name="batch1",
            config_path=tmp_path / "config.yaml",
            set_thread_caps=False,
        )
        assert "--set-thread-caps" not in cmd

    def test_set_thread_caps_true_includes_flag(self, run_all_sims, tmp_path):
        cmd = run_all_sims.build_run_pipeline_cmd(
            sim_path=tmp_path / "run_0",
            batch_name="batch1",
            config_path=tmp_path / "config.yaml",
            set_thread_caps=True,
        )
        assert "--set-thread-caps" in cmd

    def test_cmd_contains_required_args(self, run_all_sims, tmp_path):
        sim_path = tmp_path / "run_0"
        config_path = tmp_path / "config.yaml"
        cmd = run_all_sims.build_run_pipeline_cmd(
            sim_path=sim_path,
            batch_name="my-batch",
            config_path=config_path,
            set_thread_caps=False,
        )
        assert str(sim_path) in cmd
        assert "my-batch" in cmd
        assert str(config_path) in cmd
        assert "run_pipeline.py" in " ".join(cmd)


class TestLoadSlurmConfig:
    def test_none_returns_defaults(self, run_all_sims):
        cfg = run_all_sims.load_slurm_config(None)
        assert "partition" in cfg
        assert "mem_gb" in cfg
        assert "conda_env" in cfg

    def test_missing_file_raises(self, run_all_sims, tmp_path):
        with pytest.raises(FileNotFoundError, match="SLURM config not found"):
            run_all_sims.load_slurm_config(tmp_path / "nonexistent.yaml")

    def test_missing_slurm_key_raises(self, run_all_sims, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({"other": {}}))
        with pytest.raises(ValueError, match="slurm"):
            run_all_sims.load_slurm_config(cfg_path)

    def test_valid_config_returned(self, run_all_sims, tmp_path):
        cfg_path = tmp_path / "slurm_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "slurm": {
                        "partition": "my-partition",
                        "time": "04:00:00",
                        "mem_gb": 16,
                        "cpus_per_task": 2,
                        "max_concurrent": 10,
                        "conda_env": "my-env",
                        "log_dir": "logs/",
                        "project_root": "/my/project",
                        "python_bin": "python3",
                    }
                }
            )
        )
        cfg = run_all_sims.load_slurm_config(cfg_path)
        assert cfg["partition"] == "my-partition"
        assert cfg["mem_gb"] == 16
        assert cfg["conda_env"] == "my-env"

    def test_partial_config_raises_with_missing_keys(self, run_all_sims, tmp_path):
        """A config missing required sub-keys raises ValueError naming them."""
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text(yaml.safe_dump({"slurm": {"partition": "campus-new"}}))
        with pytest.raises(ValueError, match="missing required slurm keys"):
            run_all_sims.load_slurm_config(cfg_path)


class TestWriteSlurmArtifacts:
    def test_submission_dir_created_under_batch(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        sim_paths = [tmp_path / "exp" / "ps" / f"run_{i}" for i in range(3)]
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        assert submission_dir.exists()
        assert submission_dir.name.startswith("slurm_submission_")
        assert submission_dir.parent == results_root / "batch1"

    def test_sim_list_has_one_path_per_line(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        sim_paths = [tmp_path / "exp" / "ps" / f"run_{i}" for i in range(3)]
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        lines = (submission_dir / "sim_list.txt").read_text().splitlines()
        assert lines == [str(p) for p in sim_paths]

    def test_submit_array_sh_is_executable(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        submit = submission_dir / "submit_array.sh"
        assert submit.exists()
        assert submit.stat().st_mode & stat.S_IXUSR

    def test_submit_array_sh_has_correct_array_directive(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        sim_paths = [tmp_path / "exp" / "ps" / f"run_{i}" for i in range(5)]
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert "#SBATCH --array=1-5" in content

    def test_submit_array_sh_has_required_sbatch_headers(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        for header in (
            "#SBATCH --job-name=",
            "#SBATCH --output=",
            "#SBATCH --error=",
            "#SBATCH --partition=",
            "#SBATCH --time=",
            "#SBATCH --mem=",
            "#SBATCH --cpus-per-task=",
        ):
            assert header in content, f"Missing SBATCH directive: {header}"

    def test_sim_list_referenced_portably(self, run_all_sims, tmp_path):
        """submit_array.sh must not embed an absolute path to sim_list.txt."""
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert 'sim_list.txt"' in content
        assert str(results_root) not in content

    def test_submit_array_sh_has_slurm_task_id(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert "${SLURM_ARRAY_TASK_ID}" in content

    def test_submit_array_sh_no_unrendered_placeholders(self, run_all_sims, tmp_path):
        """No bare {word} template tokens should appear in the rendered script."""
        import re

        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        # Match {word} but not ${...} (SLURM bash variable references).
        bare = re.findall(r"(?<!\$)\{[A-Za-z_][A-Za-z0-9_]*\}", content)
        assert bare == [], f"Unrendered placeholders in submit_array.sh: {bare}"

    def test_respects_slurm_config_values(self, run_all_sims, tmp_path):
        results_root = tmp_path / "results"
        slurm_cfg_path = tmp_path / "slurm_config.yaml"
        slurm_cfg_path.write_text(
            yaml.safe_dump(
                {
                    "slurm": {
                        "partition": "my-partition",
                        "time": "04:00:00",
                        "mem_gb": 16,
                        "cpus_per_task": 2,
                        "max_concurrent": 10,
                        "conda_env": "my-env",
                        "log_dir": "logs/",
                        "project_root": "/my/project",
                        "python_bin": "python3",
                    }
                }
            )
        )
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=slurm_cfg_path,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert "#SBATCH --partition=my-partition" in content
        assert "#SBATCH --mem=16G" in content
        assert "source activate my-env" in content
        assert "python3" in content

    def test_template_cds_into_project_root(self, run_all_sims, tmp_path):
        """The task must cd into project_root so relative data/results resolve."""
        results_root = tmp_path / "results"
        slurm_cfg_path = tmp_path / "slurm_config.yaml"
        slurm_cfg_path.write_text(
            yaml.safe_dump(
                {
                    "slurm": {
                        "partition": "my-partition",
                        "time": "04:00:00",
                        "mem_gb": 16,
                        "cpus_per_task": 2,
                        "max_concurrent": 10,
                        "conda_env": "my-env",
                        "log_dir": "logs/",
                        "project_root": "/my/project",
                        "python_bin": "python3",
                    }
                }
            )
        )
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=slurm_cfg_path,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert 'cd "/my/project"' in content

    def test_config_path_forwarded_absolute(self, run_all_sims, tmp_path):
        """--config must be rendered as an absolute path (task cd's away)."""
        results_root = tmp_path / "results"
        rel_config = Path("config.yaml")
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=rel_config,
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert str(rel_config.resolve()) in content

    def test_max_concurrent_override_applied(self, run_all_sims, tmp_path):
        """--max-concurrent override wins over the config value in the %N throttle."""
        results_root = tmp_path / "results"
        sim_paths = [tmp_path / "exp" / "ps" / f"run_{i}" for i in range(4)]
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,  # default max_concurrent is 20
            max_concurrent_override=3,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert "#SBATCH --array=1-4%3" in content

    def test_max_concurrent_none_uses_config_value(self, run_all_sims, tmp_path):
        """Without an override, the config's max_concurrent is used."""
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,  # default max_concurrent is 20
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        assert "%20" in content

    def test_sim_list_count_matches_array_n(self, run_all_sims, tmp_path):
        """The --array=1-N directive must equal the number of lines in sim_list.txt."""
        import re

        results_root = tmp_path / "results"
        sim_paths = [tmp_path / "exp" / "ps" / f"run_{i}" for i in range(7)]
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        content = (submission_dir / "submit_array.sh").read_text()
        m = re.search(r"--array=1-(\d+)", content)
        assert m is not None, "Could not find --array= directive"
        n_in_script = int(m.group(1))
        n_in_list = len((submission_dir / "sim_list.txt").read_text().splitlines())
        assert n_in_script == n_in_list

    def test_submit_array_sh_bash_syntax_valid(self, run_all_sims, tmp_path):
        """bash -n must exit 0 on the generated submit_array.sh.

        sim_path does not need to exist on disk — bash -n checks syntax only.
        """
        import subprocess

        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=[tmp_path / "exp" / "ps" / "run_0"],
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        submit = submission_dir / "submit_array.sh"
        result = subprocess.run(["bash", "-n", str(submit)], capture_output=True)
        assert result.returncode == 0, f"bash -n failed:\n{result.stderr.decode()}"

    def test_sim_list_first_last_lines_are_valid_paths(self, run_all_sims, tmp_path):
        """First and last lines of sim_list.txt must be non-empty absolute paths
        that exist on disk and contain output/run-out.tips. Catches off-by-one
        errors in the --array=1-N range and missing-file bugs in discovery.
        """
        root = tmp_path / "experiments"
        n_sims = 5
        sim_paths = [
            _make_valid_sim(root, "exp1", "ps1", f"run_{i}") for i in range(n_sims)
        ]
        results_root = tmp_path / "results"
        submission_dir = run_all_sims.write_slurm_artifacts(
            sim_paths=sim_paths,
            batch_name="batch1",
            results_root=results_root,
            config_path=tmp_path / "config.yaml",
            slurm_config_path=None,
            max_concurrent_override=None,
        )
        lines = (submission_dir / "sim_list.txt").read_text().splitlines()
        assert len(lines) == n_sims
        for line in (lines[0], lines[-1]):
            assert line, "sim_list.txt line must not be empty"
            p = Path(line)
            assert p.is_absolute(), f"Expected absolute path, got: {line}"
            assert p.exists(), f"Path does not exist: {line}"
            assert (p / "output" / "run-out.tips").exists(), (
                f"output/run-out.tips missing in: {line}"
            )


class TestRunLocal:
    def test_sequential_calls_once_per_sim(self, run_all_sims, tmp_path):
        sim_paths = [tmp_path / f"run_{i}" for i in range(3)]
        called: list[list[str]] = []

        with patch.object(
            run_all_sims, "_run_one_sim", side_effect=lambda cmd: called.append(cmd)
        ):
            run_all_sims.run_local(
                sim_paths=sim_paths,
                batch_name="batch1",
                config_path=tmp_path / "config.yaml",
                max_parallel=1,
            )
        assert len(called) == 3

    def test_no_thread_caps_when_max_parallel_is_1(self, run_all_sims, tmp_path):
        called: list[list[str]] = []

        with patch.object(
            run_all_sims, "_run_one_sim", side_effect=lambda cmd: called.append(cmd)
        ):
            run_all_sims.run_local(
                sim_paths=[tmp_path / "run_0"],
                batch_name="batch1",
                config_path=tmp_path / "config.yaml",
                max_parallel=1,
            )
        assert "--set-thread-caps" not in called[0]

    def test_thread_caps_when_max_parallel_gt_1(self, run_all_sims, tmp_path):
        """Verify --set-thread-caps is passed when max_parallel > 1."""
        sim_paths = [tmp_path / "run_0", tmp_path / "run_1"]
        submitted_cmds: list[list[str]] = []

        mock_future = MagicMock()
        mock_future.result.return_value = None

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = lambda fn, cmd: (
            submitted_cmds.append(cmd) or mock_future
        )

        mock_executor_cls = MagicMock()
        mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch("concurrent.futures.ProcessPoolExecutor", mock_executor_cls):
            with patch(
                "concurrent.futures.as_completed", return_value=[mock_future] * 2
            ):
                run_all_sims.run_local(
                    sim_paths=sim_paths,
                    batch_name="batch1",
                    config_path=tmp_path / "config.yaml",
                    max_parallel=2,
                )

        assert all("--set-thread-caps" in cmd for cmd in submitted_cmds)
        mock_executor_cls.assert_called_once_with(max_workers=2)

    def test_empty_sim_list_runs_nothing(self, run_all_sims, tmp_path):
        called: list[list[str]] = []

        with patch.object(
            run_all_sims, "_run_one_sim", side_effect=lambda cmd: called.append(cmd)
        ):
            run_all_sims.run_local(
                sim_paths=[],
                batch_name="batch1",
                config_path=tmp_path / "config.yaml",
                max_parallel=1,
            )
        assert called == []


class TestLoadResultsRoot:
    def test_returns_absolute_path(self, run_all_sims, tmp_path):
        cfg_path = tmp_path / "pipeline_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({"pipeline": {"results_root": str(tmp_path / "results")}})
        )
        result = run_all_sims._load_results_root(cfg_path)
        assert result.is_absolute()
        assert result == (tmp_path / "results").resolve()

    def test_relative_path_resolves_against_config_dir(self, run_all_sims, tmp_path):
        sub = tmp_path / "configs"
        sub.mkdir()
        cfg_path = sub / "pipeline_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({"pipeline": {"results_root": "../results"}})
        )
        result = run_all_sims._load_results_root(cfg_path)
        assert result == (tmp_path / "results").resolve()

    def test_missing_config_raises(self, run_all_sims, tmp_path):
        with pytest.raises(FileNotFoundError, match="Pipeline config not found"):
            run_all_sims._load_results_root(tmp_path / "nonexistent.yaml")

    def test_missing_pipeline_key_raises(self, run_all_sims, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({"other": {}}))
        with pytest.raises(ValueError, match="pipeline"):
            run_all_sims._load_results_root(cfg_path)

    def test_missing_results_root_key_raises(self, run_all_sims, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({"pipeline": {"window_size": 365}}))
        with pytest.raises(ValueError, match="results_root"):
            run_all_sims._load_results_root(cfg_path)


class TestParseArgs:
    def test_all_required_args_parsed(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "local",
            ]
        )
        assert args.experiment == "exp1"
        assert args.param_set == "ps1"
        assert args.batch_name == "batch1"
        assert args.mode == "local"
        assert args.max_parallel == 1
        assert args.slurm_config is None

    def test_invalid_mode_raises(self, run_all_sims, tmp_path):
        with pytest.raises(SystemExit):
            run_all_sims._parse_args(
                [
                    "--experiments-root",
                    str(tmp_path),
                    "--experiment",
                    "exp1",
                    "--param-set",
                    "ps1",
                    "--batch-name",
                    "batch1",
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--mode",
                    "cloud",
                ]
            )

    def test_max_parallel_parsed(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "local",
                "--max-parallel",
                "8",
            ]
        )
        assert args.max_parallel == 8

    def test_all_configs_flag_parsed(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--all-configs",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "slurm",
            ]
        )
        assert args.all_configs is True
        assert args.param_set is None

    def test_param_set_and_all_configs_conflict_exits(self, run_all_sims, tmp_path):
        with pytest.raises(SystemExit):
            run_all_sims._parse_args(
                [
                    "--experiments-root",
                    str(tmp_path),
                    "--experiment",
                    "exp1",
                    "--param-set",
                    "ps1",
                    "--all-configs",
                    "--batch-name",
                    "batch1",
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--mode",
                    "slurm",
                ]
            )

    def test_neither_scope_flag_exits(self, run_all_sims, tmp_path):
        with pytest.raises(SystemExit):
            run_all_sims._parse_args(
                [
                    "--experiments-root",
                    str(tmp_path),
                    "--experiment",
                    "exp1",
                    "--batch-name",
                    "batch1",
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--mode",
                    "slurm",
                ]
            )

    def test_max_concurrent_parsed(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--all-configs",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "slurm",
                "--max-concurrent",
                "5",
            ]
        )
        assert args.max_concurrent == 5

    def test_max_concurrent_defaults_none(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--all-configs",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "slurm",
            ]
        )
        assert args.max_concurrent is None

    def test_submit_flag_parsed(self, run_all_sims, tmp_path):
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--all-configs",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "slurm",
                "--submit",
            ]
        )
        assert args.submit is True

    def test_slurm_mode_with_slurm_config(self, run_all_sims, tmp_path):
        slurm_cfg = tmp_path / "slurm_config.yaml"
        args = run_all_sims._parse_args(
            [
                "--experiments-root",
                str(tmp_path),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "batch1",
                "--config",
                str(tmp_path / "config.yaml"),
                "--mode",
                "slurm",
                "--slurm-config",
                str(slurm_cfg),
            ]
        )
        assert args.mode == "slurm"
        assert args.slurm_config == slurm_cfg


def _write_pipeline_config(path: "Path", results_root: "Path") -> None:
    """Write a minimal pipeline_config.yaml for main() integration tests."""
    import yaml as _yaml

    path.write_text(
        _yaml.safe_dump(
            {
                "pipeline": {
                    "experiments_root": str(path.parent),
                    "data_root": str(path.parent / "data"),
                    "results_root": str(results_root),
                    "window_size": 365,
                    "buffer_size": 0,
                    "models": ["FGA"],
                    "locations": ["north"],
                    "forecast_L": 180,
                    "seed_L": 14,
                    "numpyro_seed": 42,
                }
            }
        )
    )


class TestMain:
    def test_slurm_mode_writes_artifacts(self, run_all_sims, tmp_path):
        """main() in slurm mode discovers sims and writes submission artifacts."""
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        _make_valid_sim(root, "exp1", "ps1", "run_1")
        results_root = tmp_path / "results"
        cfg_path = tmp_path / "config.yaml"
        _write_pipeline_config(cfg_path, results_root)

        run_all_sims.main(
            [
                "--experiments-root",
                str(root),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "my-batch",
                "--config",
                str(cfg_path),
                "--mode",
                "slurm",
            ]
        )

        batch_dir = results_root / "my-batch"
        submission_dirs = list(batch_dir.glob("slurm_submission_*"))
        assert len(submission_dirs) == 1
        submission_dir = submission_dirs[0]
        assert (submission_dir / "sim_list.txt").exists()
        assert (submission_dir / "submit_array.sh").exists()
        lines = (submission_dir / "sim_list.txt").read_text().splitlines()
        assert len(lines) == 2

    def test_slurm_mode_skips_complete_sims(self, run_all_sims, tmp_path):
        """Sims with both sentinels pre-created are omitted from sim_list.txt."""
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        _make_valid_sim(root, "exp1", "ps1", "run_1")
        results_root = tmp_path / "results"
        cfg_path = tmp_path / "config.yaml"
        _write_pipeline_config(cfg_path, results_root)

        # Pre-create sentinels for run_0 to mark it complete.
        complete_dir = results_root / "my-batch" / "ps1__run_0"
        complete_dir.mkdir(parents=True)
        (complete_dir / "scores.tsv").write_text("")
        (complete_dir / "growth_rate_scores.tsv").write_text("")

        run_all_sims.main(
            [
                "--experiments-root",
                str(root),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "my-batch",
                "--config",
                str(cfg_path),
                "--mode",
                "slurm",
            ]
        )

        batch_dir = results_root / "my-batch"
        submission_dirs = list(batch_dir.glob("slurm_submission_*"))
        assert len(submission_dirs) == 1
        lines = (submission_dirs[0] / "sim_list.txt").read_text().splitlines()
        assert len(lines) == 1
        assert "run_1" in lines[0]

    def test_submit_flag_invokes_sbatch(self, run_all_sims, tmp_path):
        """--submit calls sbatch on the generated submit_array.sh."""
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        results_root = tmp_path / "results"
        cfg_path = tmp_path / "config.yaml"
        _write_pipeline_config(cfg_path, results_root)

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )
        with patch.object(
            run_all_sims.subprocess, "run", return_value=completed
        ) as mock_run:
            run_all_sims.main(
                [
                    "--experiments-root",
                    str(root),
                    "--experiment",
                    "exp1",
                    "--param-set",
                    "ps1",
                    "--batch-name",
                    "my-batch",
                    "--config",
                    str(cfg_path),
                    "--mode",
                    "slurm",
                    "--submit",
                ]
            )
        assert mock_run.call_count == 1
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == "sbatch"
        assert called_cmd[1].endswith("submit_array.sh")

    def test_all_configs_mode_writes_artifacts(self, run_all_sims, tmp_path):
        """--all-configs discovers runs across every sweep cell."""
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "muP_0.1", "run_0")
        _make_valid_sim(root, "exp1", "muP_0.2", "run_0")
        results_root = tmp_path / "results"
        cfg_path = tmp_path / "config.yaml"
        _write_pipeline_config(cfg_path, results_root)

        run_all_sims.main(
            [
                "--experiments-root",
                str(root),
                "--experiment",
                "exp1",
                "--all-configs",
                "--batch-name",
                "my-batch",
                "--config",
                str(cfg_path),
                "--mode",
                "slurm",
            ]
        )

        batch_dir = results_root / "my-batch"
        submission_dirs = list(batch_dir.glob("slurm_submission_*"))
        assert len(submission_dirs) == 1
        lines = (submission_dirs[0] / "sim_list.txt").read_text().splitlines()
        assert len(lines) == 2

    def test_slurm_mode_all_complete_writes_nothing(self, run_all_sims, tmp_path):
        """When all sims are complete, no submission dir is created."""
        root = tmp_path / "experiments"
        _make_valid_sim(root, "exp1", "ps1", "run_0")
        results_root = tmp_path / "results"
        cfg_path = tmp_path / "config.yaml"
        _write_pipeline_config(cfg_path, results_root)

        complete_dir = results_root / "my-batch" / "ps1__run_0"
        complete_dir.mkdir(parents=True)
        (complete_dir / "scores.tsv").write_text("")
        (complete_dir / "growth_rate_scores.tsv").write_text("")

        run_all_sims.main(
            [
                "--experiments-root",
                str(root),
                "--experiment",
                "exp1",
                "--param-set",
                "ps1",
                "--batch-name",
                "my-batch",
                "--config",
                str(cfg_path),
                "--mode",
                "slurm",
            ]
        )

        batch_dir = results_root / "my-batch"
        assert not list(batch_dir.glob("slurm_submission_*"))
