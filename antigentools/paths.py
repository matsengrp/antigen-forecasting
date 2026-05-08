"""Path construction for the antigen-forecasting analysis pipeline.

Centralizes all per-simulation path logic so downstream scripts construct paths via
``SimulationPaths`` rather than ad hoc string joins.

Design References:
- PRIMARY: specs/analysis-pipeline.md (Issue 1, Part B)

Implementation Status: WIP
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


@dataclass
class SimulationPaths:
    """All filesystem paths associated with one simulation run.

    A simulation is identified by its ``sim_path`` in ``antigen-experiments``. The
    ``build`` string is an opaque path fragment used as the leaf under ``data_root``
    and ``results_root``; downstream scripts treat it as opaque.

    Use ``from_sim_path`` for batch runs (build derived from path + batch name) or
    ``from_build`` for single-dataset runs (e.g. ``flu-final``, build passed
    verbatim).

    Attributes:
        sim_path: Path to ``run_N/`` in ``antigen-experiments``.
        data_root: Root for processed data (typically ``antigen-forecasting/data/``).
        results_root: Root for model outputs (typically ``antigen-forecasting/results/``).
        build: Opaque path fragment, e.g. ``flu-final`` or
            ``2026-04-29-all-sims/nonEpitopeAcceptance_0.1__run_0``.
    """

    sim_path: Path
    data_root: Path
    results_root: Path
    build: str

    def __post_init__(self) -> None:
        self.sim_path = Path(self.sim_path)
        self.data_root = Path(self.data_root)
        self.results_root = Path(self.results_root)

    @classmethod
    def from_sim_path(
        cls,
        sim_path: PathLike,
        data_root: PathLike,
        results_root: PathLike,
        batch_name: str,
    ) -> "SimulationPaths":
        """Construct from a ``run_N/`` path plus a human-supplied batch name.

        ``sim_id`` is derived as ``f"{sim_path.parent.name}__{sim_path.name}"`` and
        combined with ``batch_name`` to form ``build``. The ``__`` separator is the
        parsing anchor for downstream tooling, so it must not appear in either
        segment.

        Args:
            sim_path: Path to ``run_N/`` (must exist).
            data_root: Root for processed data.
            results_root: Root for model outputs.
            batch_name: Human-supplied batch name (must not contain ``/``).

        Returns:
            A ``SimulationPaths`` with ``build = f"{batch_name}/{sim_id}"``.

        Raises:
            FileNotFoundError: If ``sim_path`` does not exist.
            ValueError: If ``__`` appears in the param-set or run-id, ``/`` appears in
                ``batch_name``, or the param-set segment is empty.
        """
        sim_path = Path(sim_path)
        if not sim_path.exists():
            raise FileNotFoundError(f"sim_path does not exist: {sim_path}")

        param_set = sim_path.parent.name
        run_id = sim_path.name

        if param_set == "":
            raise ValueError(
                f"sim_path has empty param-set segment (sim_path.parent.name); got "
                f"sim_path={sim_path!r}"
            )
        if "__" in param_set:
            raise ValueError(
                f"param-set name must not contain '__' (the sim_id separator); got "
                f"param_set={param_set!r}"
            )
        if "__" in run_id:
            raise ValueError(
                f"run-id must not contain '__' (the sim_id separator); got "
                f"run_id={run_id!r}"
            )
        if "/" in batch_name:
            raise ValueError(
                f"batch_name must not contain '/' (it would corrupt the data/<build>/ "
                f"split); got batch_name={batch_name!r}"
            )

        sim_id = f"{param_set}__{run_id}"
        build = f"{batch_name}/{sim_id}"
        return cls(
            sim_path=sim_path,
            data_root=Path(data_root),
            results_root=Path(results_root),
            build=build,
        )

    @classmethod
    def from_build(
        cls,
        build: str,
        sim_path: PathLike,
        data_root: PathLike,
        results_root: PathLike,
    ) -> "SimulationPaths":
        """Construct from a verbatim build string (single-dataset mode).

        Use this for the single-dataset ``flu-final`` development case where ``build``
        is a human-chosen name rather than a derived ``<batch>/<sim-id>``.

        Args:
            build: Verbatim build string (e.g. ``"flu-final"``).
            sim_path: Path to ``run_N/`` (must exist).
            data_root: Root for processed data.
            results_root: Root for model outputs.

        Returns:
            A ``SimulationPaths`` with ``build`` set verbatim.

        Raises:
            FileNotFoundError: If ``sim_path`` does not exist.
        """
        sim_path = Path(sim_path)
        if not sim_path.exists():
            raise FileNotFoundError(f"sim_path does not exist: {sim_path}")
        return cls(
            sim_path=sim_path,
            data_root=Path(data_root),
            results_root=Path(results_root),
            build=build,
        )

    # Raw antigen-prime outputs (read-only, in antigen-experiments) -----------

    @property
    def raw_tips(self) -> Path:
        return self.sim_path / "output" / "run-out.tips"

    @property
    def raw_fasta(self) -> Path:
        return self.sim_path / "output" / "run-out.fasta"

    @property
    def raw_timeseries(self) -> Path:
        return self.sim_path / "out_timeseries.csv"

    @property
    def raw_parameters_yml(self) -> Path:
        return self.sim_path.parent / "parameters.yml"

    # Parsed outputs in data/<build>/antigen-outputs/ -------------------------

    @property
    def antigen_outputs(self) -> Path:
        return self.data_root / self.build / "antigen-outputs"

    @property
    def tips(self) -> Path:
        return self.antigen_outputs / "tips.csv"

    @property
    def unique_tips(self) -> Path:
        return self.antigen_outputs / "unique_tips.csv"

    @property
    def unique_sequences(self) -> Path:
        return self.antigen_outputs / "unique_sequences.fasta"

    @property
    def timeseries(self) -> Path:
        return self.antigen_outputs / "out_timeseries.csv"

    # Per-build derived outputs in data/<build>/ ------------------------------

    @property
    def variant_assignment(self) -> Path:
        return self.data_root / self.build / "variant-assignment"

    @property
    def tips_with_variants(self) -> Path:
        return self.data_root / self.build / "tips_with_variants.tsv"

    @property
    def seq_counts(self) -> Path:
        return self.data_root / self.build / "seq_counts.tsv"

    @property
    def case_counts(self) -> Path:
        return self.data_root / self.build / "case_counts.tsv"

    @property
    def time_stamped(self) -> Path:
        return self.data_root / self.build / "time-stamped"

    @property
    def time_stamped_manifest(self) -> Path:
        return self.time_stamped / "MANIFEST.tsv"

    # Model outputs in results/<build>/ ---------------------------------------

    @property
    def results(self) -> Path:
        return self.results_root / self.build

    @property
    def estimates(self) -> Path:
        return self.results / "estimates"

    @property
    def estimates_manifest(self) -> Path:
        return self.estimates / "MANIFEST.tsv"

    @property
    def growth_rates(self) -> Path:
        return self.results / "growth-rates"

    @property
    def scores(self) -> Path:
        return self.results / "scores.tsv"

    @property
    def growth_rate_scores(self) -> Path:
        return self.results / "growth_rate_scores.tsv"

    @property
    def vi_convergence_diagnostics(self) -> Path:
        return self.results / "vi_convergence_diagnostics.tsv"

    @property
    def pipeline_log(self) -> Path:
        return self.results / "pipeline.log"
