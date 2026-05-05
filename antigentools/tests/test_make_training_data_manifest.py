"""Tests for the MANIFEST.tsv emission added to scripts/make_training_data.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "make_training_data.py"


@pytest.fixture(scope="module")
def make_training_data():
    """Import scripts/make_training_data.py as a module."""
    spec = importlib.util.spec_from_file_location("make_training_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_training_data"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build seq_counts.tsv, case_counts.tsv spanning two analysis dates."""
    dates = pd.date_range("2024-09-01", "2026-04-01", freq="D")
    seq_rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "country": "north",
            "variant": "A",
            "sequences": 1,
        }
        for d in dates
    ]
    case_rows = [
        {"date": d.strftime("%Y-%m-%d"), "country": "north", "cases": 10} for d in dates
    ]
    seqs = tmp_path / "seq_counts.tsv"
    cases = tmp_path / "case_counts.tsv"
    pd.DataFrame(seq_rows).to_csv(seqs, sep="\t", index=False)
    pd.DataFrame(case_rows).to_csv(cases, sep="\t", index=False)
    out_dir = tmp_path / "time-stamped"
    cfg = tmp_path / "benchmark_config.yaml"
    return seqs, cases, out_dir, cfg


def test_raises_when_no_windows_produced(make_training_data, tmp_path):
    """No analysis date inside the input range -> raise, not header-only TSV."""
    dates = pd.date_range("2025-01-01", "2025-03-15", freq="D")
    seq_rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "country": "north",
            "variant": "A",
            "sequences": 1,
        }
        for d in dates
    ]
    case_rows = [
        {"date": d.strftime("%Y-%m-%d"), "country": "north", "cases": 10} for d in dates
    ]
    seqs = tmp_path / "seq_counts.tsv"
    cases = tmp_path / "case_counts.tsv"
    pd.DataFrame(seq_rows).to_csv(seqs, sep="\t", index=False)
    pd.DataFrame(case_rows).to_csv(cases, sep="\t", index=False)

    args = SimpleNamespace(
        sequences=str(seqs),
        cases=str(cases),
        output=str(tmp_path / "time-stamped"),
        window_size=365,
        buffer_size=0,
        config_path=str(tmp_path / "benchmark_config.yaml"),
    )
    with pytest.raises(RuntimeError, match="No analysis dates"):
        make_training_data.main(args)
    assert not (tmp_path / "time-stamped" / "MANIFEST.tsv").exists()


def test_manifest_written_with_one_row_per_date(make_training_data, synthetic_inputs):
    seqs, cases, out_dir, cfg = synthetic_inputs
    args = SimpleNamespace(
        sequences=str(seqs),
        cases=str(cases),
        output=str(out_dir),
        window_size=365,
        buffer_size=0,
        config_path=str(cfg),
    )
    make_training_data.main(args)

    manifest_path = out_dir / "MANIFEST.tsv"
    assert manifest_path.exists()
    df = pd.read_csv(manifest_path, sep="\t")
    assert list(df.columns) == ["date", "seq_counts_path", "case_counts_path"]
    # The two analysis dates April 1 and October 1 fall within the date range.
    assert sorted(df["date"].tolist()) == [
        "2024-10-01",
        "2025-04-01",
        "2025-10-01",
        "2026-04-01",
    ]
    for row in df.itertuples():
        assert Path(row.seq_counts_path).exists()
        assert Path(row.case_counts_path).exists()
