"""Tests for the dedup contract in scripts/assign_all_variants.py.

The script must reject any --tips input that is not already deduplicated on
both ``name`` and ``nucleotideSequence``. Feeding the full ``tips.csv`` would
silently pick different representative rows on ``name`` collisions and
diverge from the canonical name -> nucleotideSequence dedup performed by
``parse_sim_outputs.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "assign_all_variants.py"


@pytest.fixture(scope="module")
def assign_all_variants():
    """Import scripts/assign_all_variants.py as a module."""
    spec = importlib.util.spec_from_file_location("assign_all_variants", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["assign_all_variants"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tips(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _row(name: str, seq: str) -> dict:
    return {
        "name": name,
        "year": 2030.0,
        "location": 0,
        "country": "north",
        "nucleotideSequence": seq,
        "ag1": 0.0,
        "ag2": 0.0,
    }


def test_assertion_fires_on_name_dup(assign_all_variants, tmp_path, monkeypatch):
    tips = tmp_path / "tips.tsv"
    out = tmp_path / "out.tsv"
    _write_tips(tips, [_row("a", "AAA"), _row("a", "CCC")])  # name dup
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assign_all_variants.py",
            "--tips",
            str(tips),
            "--output",
            str(out),
            "--fast",
        ],
    )
    with pytest.raises(AssertionError, match="name"):
        assign_all_variants.main()


def test_assertion_fires_on_seq_dup(assign_all_variants, tmp_path, monkeypatch):
    tips = tmp_path / "tips.tsv"
    out = tmp_path / "out.tsv"
    _write_tips(tips, [_row("a", "AAA"), _row("b", "AAA")])  # seq dup
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assign_all_variants.py",
            "--tips",
            str(tips),
            "--output",
            str(out),
            "--fast",
        ],
    )
    with pytest.raises(AssertionError, match="nucleotideSequence"):
        assign_all_variants.main()
