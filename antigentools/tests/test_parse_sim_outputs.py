"""Unit tests for scripts/parse_sim_outputs.py."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "parse_sim_outputs.py"


@pytest.fixture(scope="module")
def parse_sim_outputs():
    """Import parse_sim_outputs.py as a module."""
    spec = importlib.util.spec_from_file_location("parse_sim_outputs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["parse_sim_outputs"] = module
    spec.loader.exec_module(module)
    return module


REQUIRED_COLUMNS = [
    "name",
    "year",
    "location",
    "nucleotideSequence",
    "ag1",
    "ag2",
]


def _make_tips_csv(path: Path, rows: list[dict]) -> None:
    """Write a synthetic run-out.tips file at ``path``."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_row(**overrides) -> dict:
    """Return a minimal valid tips row with optional column overrides."""
    row = {
        "name": "tip0",
        "year": 1.0,
        "location": 0,
        "nucleotideSequence": "ATGC",
        "ag1": 0.0,
        "ag2": 0.0,
    }
    row.update(overrides)
    return row


class TestDedupTips:
    def test_pinned_order_name_then_nucleotideseq(self, parse_sim_outputs):
        # Two rows share `name`. drop_duplicates(["name"]) keeps the first → row B
        # never gets a chance, even though it has a unique nucleotideSequence.
        # Two rows share `nucleotideSequence` (row A and row D). After name-dedup,
        # the seq-dedup keeps the first → row A wins, row D drops.
        df = pd.DataFrame(
            [
                _base_row(name="dup_name", nucleotideSequence="SEQ_A"),  # A
                _base_row(name="dup_name", nucleotideSequence="SEQ_B"),  # B
                _base_row(name="unique_c", nucleotideSequence="SEQ_C"),  # C
                _base_row(name="unique_d", nucleotideSequence="SEQ_A"),  # D (seq dup)
            ]
        )
        result = parse_sim_outputs.dedup_tips(df)
        assert len(result) == 2
        names_seqs = set(zip(result["name"], result["nucleotideSequence"]))
        assert names_seqs == {("dup_name", "SEQ_A"), ("unique_c", "SEQ_C")}

    def test_diverges_from_seq_first_order(self, parse_sim_outputs):
        # Fixture where the two orderings produce different surviving rows.
        # name-first then seq: ("dup_name", "SEQ_A") wins and ("unique_d", ...) drops.
        # seq-first then name: ("dup_name", "SEQ_A") survives, ("dup_name", "SEQ_B")
        # also survives the seq step (SEQ_B is unique), then name-step drops one.
        # The fixture below pins membership for the canonical order.
        df = pd.DataFrame(
            [
                _base_row(name="dup_name", nucleotideSequence="SEQ_A"),
                _base_row(name="dup_name", nucleotideSequence="SEQ_B"),
                _base_row(name="unique_d", nucleotideSequence="SEQ_A"),
            ]
        )
        correct = parse_sim_outputs.dedup_tips(df)
        correct_set = set(zip(correct["name"], correct["nucleotideSequence"]))
        # Canonical order keeps the FIRST row only — name dedup drops row 2,
        # then seq dedup drops row 3.
        assert correct_set == {("dup_name", "SEQ_A")}


class TestParseRunOutTips:
    def test_adds_country_with_pinned_mapping(self, parse_sim_outputs, tmp_path):
        tips_path = tmp_path / "run-out.tips"
        _make_tips_csv(
            tips_path,
            [
                _base_row(name="n0", location=0),
                _base_row(name="n1", location=1, nucleotideSequence="A"),
                _base_row(name="n2", location=2, nucleotideSequence="T"),
            ],
        )
        df = parse_sim_outputs.parse_run_out_tips(tips_path)
        assert df.loc[df["name"] == "n0", "country"].item() == "north"
        assert df.loc[df["name"] == "n1", "country"].item() == "tropics"
        assert df.loc[df["name"] == "n2", "country"].item() == "south"

    def test_missing_required_column_raises(self, parse_sim_outputs, tmp_path):
        tips_path = tmp_path / "run-out.tips"
        bad = pd.DataFrame([_base_row()]).drop(columns=["ag1"])
        bad.to_csv(tips_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            parse_sim_outputs.parse_run_out_tips(tips_path)

    def test_unknown_location_value_raises(self, parse_sim_outputs, tmp_path):
        tips_path = tmp_path / "run-out.tips"
        _make_tips_csv(tips_path, [_base_row(location=99)])
        with pytest.raises(ValueError, match="location"):
            parse_sim_outputs.parse_run_out_tips(tips_path)


class TestLinkOrCopy:
    def test_creates_symlink_when_supported(self, parse_sim_outputs, tmp_path):
        src = tmp_path / "src.csv"
        src.write_text("data\n1\n")
        dst = tmp_path / "dst.csv"
        parse_sim_outputs.link_or_copy(src, dst)
        assert dst.exists()
        assert dst.read_text() == "data\n1\n"

    def test_falls_back_to_copy_on_oserror(
        self, parse_sim_outputs, tmp_path, monkeypatch
    ):
        src = tmp_path / "src.csv"
        src.write_text("data\n1\n")
        dst = tmp_path / "dst.csv"

        def _raise_oserror(*args, **kwargs):
            raise OSError("symlinks not supported")

        monkeypatch.setattr("os.symlink", _raise_oserror)
        parse_sim_outputs.link_or_copy(src, dst)
        assert dst.exists()
        assert not dst.is_symlink()
        assert dst.read_text() == "data\n1\n"


class TestMain:
    def test_writes_all_outputs(self, parse_sim_outputs, tmp_path):
        # Build a fake run_N/ layout:
        #   sim_path/output/run-out.tips
        #   sim_path/out_timeseries.csv
        sim_path = tmp_path / "experiments" / "exp" / "param" / "run_0"
        (sim_path / "output").mkdir(parents=True)
        tips_path = sim_path / "output" / "run-out.tips"
        _make_tips_csv(
            tips_path,
            [
                _base_row(name="a", nucleotideSequence="SEQ_X"),
                _base_row(name="a", nucleotideSequence="SEQ_Y"),  # name dup
                _base_row(name="b", nucleotideSequence="SEQ_X"),  # seq dup
                _base_row(name="c", nucleotideSequence="SEQ_Z"),
            ],
        )
        timeseries_src = sim_path / "out_timeseries.csv"
        timeseries_src.write_text("date,cases\n2020-01-01,5\n")

        output_dir = tmp_path / "data" / "test-build" / "antigen-outputs"

        parse_sim_outputs.main(
            [
                "--sim-path",
                str(sim_path),
                "--output-dir",
                str(output_dir),
            ]
        )

        # All four outputs present:
        assert (output_dir / "tips.csv").exists()
        assert (output_dir / "unique_tips.csv").exists()
        assert (output_dir / "unique_sequences.fasta").exists()
        assert (output_dir / "out_timeseries.csv").exists()

        # tips.csv has all 4 rows; unique_tips.csv has 2 (a/SEQ_X and c/SEQ_Z).
        tips_df = pd.read_csv(output_dir / "tips.csv")
        unique_df = pd.read_csv(output_dir / "unique_tips.csv")
        assert len(tips_df) == 4
        assert len(unique_df) == 2
        assert set(zip(unique_df["name"], unique_df["nucleotideSequence"])) == {
            ("a", "SEQ_X"),
            ("c", "SEQ_Z"),
        }

        # FASTA contains 2 records, names match unique_tips.
        fasta_text = (output_dir / "unique_sequences.fasta").read_text()
        assert fasta_text.count(">") == 2
        assert ">a\n" in fasta_text
        assert ">c\n" in fasta_text

        # Timeseries link must be readable through the link, not just exist —
        # catches broken relative symlinks (regression for PR #22 review).
        assert (
            output_dir / "out_timeseries.csv"
        ).read_text() == "date,cases\n2020-01-01,5\n"

    def test_writes_all_outputs_with_relative_sim_path(
        self, parse_sim_outputs, tmp_path, monkeypatch
    ):
        # Regression for PR #22: a relative --sim-path used to produce a symlink
        # whose target was relative to the destination dir, not CWD, breaking
        # the timeseries link in the common cluster invocation pattern.
        monkeypatch.chdir(tmp_path)
        sim_path = Path("experiments/exp/param/run_0")
        (sim_path / "output").mkdir(parents=True)
        _make_tips_csv(
            sim_path / "output" / "run-out.tips",
            [
                _base_row(name="a", nucleotideSequence="X"),
                _base_row(name="a", nucleotideSequence="Y"),
                _base_row(name="b", nucleotideSequence="X"),
            ],
        )
        (sim_path / "out_timeseries.csv").write_text("date,cases\n2020-01-01,7\n")

        output_dir = Path("data/test-build/antigen-outputs")
        parse_sim_outputs.main(
            ["--sim-path", str(sim_path), "--output-dir", str(output_dir)]
        )

        # The link must resolve and be readable from CWD.
        assert (
            output_dir / "out_timeseries.csv"
        ).read_text() == "date,cases\n2020-01-01,7\n"

    def test_asserts_dedup_actually_drops_rows(self, parse_sim_outputs, tmp_path):
        # If every row is already unique on both keys, dedup is a no-op and the
        # script should fail loudly — that means the input is unexpected.
        sim_path = tmp_path / "experiments" / "exp" / "param" / "run_0"
        (sim_path / "output").mkdir(parents=True)
        tips_path = sim_path / "output" / "run-out.tips"
        _make_tips_csv(
            tips_path,
            [
                _base_row(name="a", nucleotideSequence="X"),
                _base_row(name="b", nucleotideSequence="Y"),
            ],
        )
        (sim_path / "out_timeseries.csv").write_text("date,cases\n")

        output_dir = tmp_path / "out"
        with pytest.raises(AssertionError, match="dedup"):
            parse_sim_outputs.main(
                [
                    "--sim-path",
                    str(sim_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
