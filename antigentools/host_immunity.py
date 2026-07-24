"""Per-host immune-history risk of infection for simulated antigenic phenotypes.

The centroid approach in :mod:`antigentools.analysis` collapses the whole population's
immune memory to a single ``(ag1, ag2)`` point per year and measures each tip's distance
to it. This module instead uses every sampled host's *full* immune history, mirroring
``Phenotype.riskOfInfection`` in Bedford's ``antigen``::

    risk(v, h) = clip(s * min_{p in history(h)} ||v - p||, 1 - homologous_immunity, 1.0)

A virus's fitness at time ``t`` is the mean of that risk over hosts. Hosts with no immune
history (the naive fraction) contribute risk ``1.0``; they have no rows in the raw
histories file, so their contribution is reconstructed from the ``naive_fraction`` column.

Input is ``out.histories.raw.csv`` from an antigen-prime run: one row per
``(year, deme, host, infection in that host's memory)``::

    year,deme,host_id,infection_index,ag1,ag2,naive_fraction
    0.0000,north,1,0,-6.000000,0.000000,0.1369
    0.0000,north,1,1,0.000000,0.000000,0.1369

Two properties of the writer matter (``HostPopulation.printHostImmuneHistoriesCsv`` in
antigen-prime, which resets ``int hostId = 0`` on every call and is called once per
snapshot *and* deme):

- ``host_id`` is a dense slot index unique only within a ``(year, deme)`` snapshot. Every
  function here is therefore given a single year's rows and keys hosts on
  ``(deme, host_id)`` within it. Keying on ``(deme, host_id)`` across years would splice
  unrelated hosts' memories together and drive every risk down to the clipped minimum.
- Naive hosts emit no rows but still consume a sequential ``host_id``, so the sampled slot
  count is recoverable as ``n_experienced / (1 - naive_fraction)`` and cross-checkable
  against ``max(host_id) + 1``.

Hosts are a *sample* of each deme (``hostImmunitySamplesPerDeme``, typically 10,000 per
deme), not the whole population, and snapshots are written every
``printHostImmunityStep`` days -- 365 for the reviewer runs, so the native grid is yearly.

The ``(-6.0, 0.0)`` entries are legitimate ancestral phenotypes from the simulation's
initial priming, not sentinels, and must not be filtered.

Design References:
- PRIMARY: specs/analysis-pipeline.md
- Supersedes the centroid approximation in ``antigentools.analysis.calc_variance_over_time``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


HISTORIES_RAW_DTYPES: Dict[str, str] = {
    "year": "float64",
    "deme": "category",
    "host_id": "int32",
    "infection_index": "int32",
    "ag1": "float32",
    "ag2": "float32",
    "naive_fraction": "float32",
}

REQUIRED_RAW_COLUMNS: Tuple[str, ...] = tuple(HISTORIES_RAW_DTYPES)

REQUIRED_TIPS_COLUMNS: Tuple[str, ...] = ("name", "year", "ag1", "ag2")

# Long-format ``host_weighting`` levels. "experienced" averages risk over sampled hosts
# that have immune memory; "population" additionally folds in naive hosts at risk 1.0.
HOST_WEIGHTINGS: Tuple[str, str] = ("experienced", "population")

RISK_COLUMNS: Dict[str, str] = {
    "experienced": "mean_risk_of_infection_experienced",
    "population": "mean_risk_of_infection_population",
}

# Years in the histories files are written with four decimals, so exact float equality is
# unsafe but this tolerance is far below any real timepoint spacing.
YEAR_TOL: float = 1e-6

# naive_fraction is written with four decimals, bounding how precisely the sampled slot
# count can be reconstructed from it.
NAIVE_FRACTION_PRECISION: float = 5e-5


def load_raw_histories(path: Path) -> pd.DataFrame:
    """Read ``out.histories.raw.csv`` with memory-conscious dtypes.

    Args:
        path: Path to the raw histories CSV.

    Returns:
        DataFrame with exactly the columns in :data:`REQUIRED_RAW_COLUMNS`.

    Raises:
        ValueError: If any required column is missing.
    """
    header = pd.read_csv(path, nrows=0)
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in header.columns]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}. "
            f"Got columns: {list(header.columns)}"
        )

    df = pd.read_csv(
        path,
        usecols=list(REQUIRED_RAW_COLUMNS),
        dtype=HISTORIES_RAW_DTYPES,
    )
    logger.info(
        "Read %d raw history rows from %s (%d timepoints).",
        len(df),
        path,
        df["year"].nunique(),
    )
    return df


def select_timepoints(years: np.ndarray, delta_t: float) -> np.ndarray:
    """Pick the evaluation timepoints spaced ``delta_t`` apart on the native year grid.

    Keeps every recorded year ``t`` for which ``t - min(years)`` is an integer multiple
    of ``delta_t``. This snaps to years that actually exist in the histories rather than
    inventing a grid, so a ``delta_t`` finer than the recording interval is an error
    rather than a silent no-op.

    Args:
        years: Recorded years (duplicates allowed; unsorted allowed).
        delta_t: Requested spacing between evaluated timepoints, in years.

    Returns:
        Sorted array of selected years.

    Raises:
        ValueError: If ``delta_t`` is not positive, if it is finer than the native
            recording interval, or if fewer than two timepoints survive.
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive; got {delta_t}")

    unique_years = np.unique(np.asarray(years, dtype=float))
    if unique_years.size < 2:
        raise ValueError(
            f"need at least 2 distinct years in the histories; got {unique_years.size}"
        )

    native_spacing = float(np.median(np.diff(unique_years)))
    if delta_t < native_spacing - YEAR_TOL:
        raise ValueError(
            f"delta_t={delta_t} is finer than the histories' native recording interval "
            f"({native_spacing}); the file has no records in between. Use a delta_t that "
            f"is a multiple of {native_spacing}."
        )

    offsets = unique_years - unique_years[0]
    remainder = np.abs(offsets - np.round(offsets / delta_t) * delta_t)
    selected = unique_years[remainder <= YEAR_TOL + delta_t * 1e-9]

    if selected.size < 2:
        raise ValueError(
            f"delta_t={delta_t} selected only {selected.size} timepoint(s) from a grid "
            f"spanning {unique_years[0]}..{unique_years[-1]} at spacing {native_spacing}"
        )

    logger.info(
        "Selected %d of %d timepoints at delta_t=%s (native spacing %s).",
        selected.size,
        unique_years.size,
        delta_t,
        native_spacing,
    )
    return selected


def pack_host_histories(
    year_df: pd.DataFrame,
    n_hosts: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pack a sample of hosts' immune histories into one contiguous phenotype array.

    Hosts are sampled by *identity*, never by row: history lengths vary between hosts, so
    row sampling would over-represent hosts with long memories.

    Args:
        year_df: Raw history rows for a single year (all demes pooled).
        n_hosts: Number of distinct hosts to sample, or None to use every host.
        rng: Seeded generator used for the host sample.

    Returns:
        Tuple of ``(phenotypes, segment_starts, n_hosts_total)`` where ``phenotypes`` is
        an ``(m, 2)`` float64 array of ``(ag1, ag2)`` grouped by host, ``segment_starts``
        gives each sampled host's first row in ``phenotypes`` (strictly increasing), and
        ``n_hosts_total`` is the number of distinct hosts available before sampling.

    Raises:
        ValueError: If ``year_df`` is empty or ``n_hosts`` is not positive.
    """
    if len(year_df) == 0:
        raise ValueError("cannot pack host histories from an empty frame")
    if n_hosts is not None and n_hosts <= 0:
        raise ValueError(f"n_hosts must be positive or None; got {n_hosts}")

    deme_codes = pd.factorize(year_df["deme"], sort=True)[0]
    host_ids = year_df["host_id"].to_numpy()

    # Sort by (deme, host) so every host's memory occupies one contiguous block.
    order = np.lexsort((host_ids, deme_codes))
    sorted_demes = deme_codes[order]
    sorted_hosts = host_ids[order]

    is_new_host = (sorted_demes[1:] != sorted_demes[:-1]) | (
        sorted_hosts[1:] != sorted_hosts[:-1]
    )
    block_starts = np.concatenate(([0], np.flatnonzero(is_new_host) + 1))
    block_ends = np.concatenate((block_starts[1:], [order.size]))
    n_hosts_total = int(block_starts.size)

    if n_hosts is None or n_hosts >= n_hosts_total:
        chosen = np.arange(n_hosts_total)
    else:
        chosen = np.sort(rng.choice(n_hosts_total, size=n_hosts, replace=False))

    lengths = block_ends[chosen] - block_starts[chosen]
    # np.minimum.reduceat silently returns the raw element instead of reducing when a
    # segment is empty, which would corrupt every downstream risk value.
    assert (lengths > 0).all(), "every sampled host must have at least one memory entry"

    segment_starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    # Ragged gather: expand each chosen block into its own run of row indices.
    gather = (
        np.repeat(block_starts[chosen] - segment_starts, lengths)
        + np.arange(int(lengths.sum()))
    )
    rows = order[gather]

    # float64 here: the CSV carries six decimals at coordinates of order 100, which
    # float32 cannot hold. The distance kernel re-centers and downcasts for speed.
    phenotypes = np.column_stack(
        (
            year_df["ag1"].to_numpy()[rows],
            year_df["ag2"].to_numpy()[rows],
        )
    ).astype(np.float64)

    return phenotypes, segment_starts, n_hosts_total


def global_naive_fraction(year_df: pd.DataFrame) -> float:
    """Reconstruct the pooled naive fraction across demes for a single year.

    Naive hosts have no rows in the raw histories file, so the pooled fraction cannot be
    read off directly. Per deme, the sampled slot count is recovered as
    ``n_experienced / (1 - naive_fraction)`` and cross-checked against
    ``max(host_id) + 1`` -- naive hosts consume a sequential ``host_id``, so the highest id
    present can never exceed the slot count.

    The pooled fraction weights each deme by its reconstructed slot count. Together with
    the affine correction applied in :func:`risk_of_infection_over_time` this reproduces a
    uniform-over-slots population mean exactly::

        mean_pop = sum_d (S_d / S) * [nf_d + (1 - nf_d) * mean_exp_d]
                 = nf + (1 - nf) * sum_d (E_d / E) * mean_exp_d

    where the trailing sum is precisely the mean over hosts sampled uniformly from the
    pooled experienced hosts. Sampling experienced hosts and correcting analytically is
    therefore unbiased *and* lower variance than drawing naive slots at random.

    Args:
        year_df: Raw history rows for a single year.

    Returns:
        Pooled naive fraction in ``[0, 1)``.

    Raises:
        ValueError: If a deme carries more than one ``naive_fraction`` value, if a value
            falls outside ``[0, 1)``, or if the highest ``host_id`` exceeds the slot count
            implied by that fraction.
    """
    total_hosts = 0.0
    naive_hosts = 0.0

    for deme, group in year_df.groupby("deme", observed=True):
        values = np.unique(group["naive_fraction"].to_numpy())
        if values.size != 1:
            raise ValueError(
                f"deme {deme!r} has {values.size} distinct naive_fraction values "
                f"({values.tolist()}) within a single year; expected exactly 1"
            )
        fraction = float(values[0])
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"deme {deme!r} has naive_fraction={fraction}, outside [0, 1)"
            )
        n_experienced = int(group["host_id"].nunique())
        n_total = n_experienced / (1.0 - fraction)

        # naive_fraction is written with four decimals, so the reconstructed slot count
        # carries a proportional rounding error; allow for it plus one slot.
        slack = n_total * NAIVE_FRACTION_PRECISION / (1.0 - fraction) + 1.0
        highest_slot = int(group["host_id"].max()) + 1
        if highest_slot > n_total + slack:
            raise ValueError(
                f"deme {deme!r} has host_id up to {highest_slot - 1} but "
                f"naive_fraction={fraction} with {n_experienced} experienced hosts "
                f"implies only {n_total:.1f} sampled slots; the file is inconsistent"
            )

        total_hosts += n_total
        naive_hosts += n_total * fraction

    return naive_hosts / total_hosts


def _tips_per_block(n_phenotypes: int, max_block_elements: int) -> int:
    """Tips per distance block that keeps each temporary under the element budget.

    Bounding the block *area* rather than the tip count is what makes peak memory
    independent of how many hosts were sampled: at 30,000 hosts a snapshot holds tens of
    thousands of phenotypes, so a fixed tip count would size the block an order of
    magnitude larger than at 1,000 hosts.
    """
    return max(1, max_block_elements // max(1, n_phenotypes))


def _validate_kernel_inputs(
    tips_ag: np.ndarray,
    phenotypes: np.ndarray,
    segment_starts: np.ndarray,
    max_block_elements: int,
) -> None:
    """Check the shape and ``reduceat`` contracts shared by the distance kernels.

    Raises:
        ValueError: If ``max_block_elements`` is not positive, the arrays are not
            ``(n, 2)``, or ``segment_starts`` is not a valid strictly increasing index
            into ``phenotypes``.
    """
    if max_block_elements <= 0:
        raise ValueError(
            f"max_block_elements must be positive; got {max_block_elements}"
        )
    if tips_ag.ndim != 2 or tips_ag.shape[1] != 2:
        raise ValueError(f"tips_ag must have shape (k, 2); got {tips_ag.shape}")
    if phenotypes.ndim != 2 or phenotypes.shape[1] != 2:
        raise ValueError(f"phenotypes must have shape (m, 2); got {phenotypes.shape}")
    if segment_starts.size == 0:
        raise ValueError("segment_starts must not be empty")
    # np.minimum.reduceat returns the raw element instead of a reduction when a segment is
    # empty, and raises IndexError past the end. Both would be silent or late failures.
    if segment_starts[0] != 0 or not (np.diff(segment_starts) > 0).all():
        raise ValueError("segment_starts must start at 0 and be strictly increasing")
    if segment_starts[-1] >= phenotypes.shape[0]:
        raise ValueError(
            f"segment_starts[-1]={segment_starts[-1]} is out of range for "
            f"{phenotypes.shape[0]} phenotypes"
        )


def _risk_block(
    tips_block: np.ndarray,
    pheno_centered: np.ndarray,
    center: np.ndarray,
    segment_starts: np.ndarray,
    smith_conversion: float,
    homologous_immunity: float,
) -> np.ndarray:
    """Compute the ``(block, n_hosts)`` risk matrix for one chunk of tips."""
    block = (tips_block - center).astype(np.float32)
    # Built in place: each temporary is (block, n_phenotypes), the largest array here, and
    # the naive expression would hold four of them at once.
    squared = block[:, 0, None] - pheno_centered[None, :, 0]
    np.square(squared, out=squared)
    other_axis = block[:, 1, None] - pheno_centered[None, :, 1]
    np.square(other_axis, out=other_axis)
    squared += other_axis
    del other_axis
    # sqrt is monotone, so min-then-sqrt equals sqrt-then-min but takes the square root on
    # the (block, n_hosts) result rather than the (block, n_phenotypes) input.
    closest = np.sqrt(
        np.minimum.reduceat(squared, segment_starts, axis=1), dtype=np.float64
    )
    # Clip after the per-host minimum, mirroring GeometricPhenotype.riskOfInfection.
    return np.clip(closest * smith_conversion, 1.0 - homologous_immunity, 1.0)


def _center_of(phenotypes: np.ndarray) -> np.ndarray:
    """Return the phenotype centroid used to de-bias the float32 distance kernel.

    Antigenic coordinates drift to order 100 while the distances that decide a ``min`` can
    be order 0.01. Differencing those directly in float32 loses roughly three significant
    digits; re-centering first keeps the float32 values at the scale of the spread.
    """
    return phenotypes.mean(axis=0)


def risk_of_infection_matrix(
    tips_ag: np.ndarray,
    phenotypes: np.ndarray,
    segment_starts: np.ndarray,
    smith_conversion: float,
    homologous_immunity: float,
    max_block_elements: int,
) -> np.ndarray:
    """Compute each tip's risk of infection against each host.

    For every ``(tip, host)`` pair this takes the minimum Euclidean distance from the tip
    to any phenotype in that host's memory, scales it by ``smith_conversion``, and clips
    to ``[1 - homologous_immunity, 1.0]``.

    This materializes the full matrix and is intended for inspection and tests;
    :func:`risk_of_infection_stats` is the memory-bounded path used in production.

    Args:
        tips_ag: ``(k, 2)`` array of tip ``(ag1, ag2)`` coordinates.
        phenotypes: ``(m, 2)`` array from :func:`pack_host_histories`.
        segment_starts: Strictly increasing host boundaries into ``phenotypes``.
        smith_conversion: Factor scaling antigenic distance to infection risk.
        homologous_immunity: Immunity against an identical antigen; bounds minimum risk.
        max_block_elements: Element budget for each distance block, bounding peak memory
            regardless of how many hosts were sampled.

    Returns:
        ``(k, n_hosts)`` float64 array of infection risks.

    Raises:
        ValueError: If any shape or ``reduceat`` precondition is violated.
    """
    _validate_kernel_inputs(tips_ag, phenotypes, segment_starts, max_block_elements)

    center = _center_of(phenotypes)
    pheno_centered = (phenotypes - center).astype(np.float32)
    n_tips = tips_ag.shape[0]
    block_size = _tips_per_block(phenotypes.shape[0], max_block_elements)
    out = np.empty((n_tips, segment_starts.size), dtype=np.float64)

    for start in range(0, n_tips, block_size):
        stop = min(start + block_size, n_tips)
        out[start:stop] = _risk_block(
            tips_ag[start:stop],
            pheno_centered,
            center,
            segment_starts,
            smith_conversion,
            homologous_immunity,
        )

    return out


def risk_of_infection_stats(
    tips_ag: np.ndarray,
    phenotypes: np.ndarray,
    segment_starts: np.ndarray,
    smith_conversion: float,
    homologous_immunity: float,
    max_block_elements: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each tip's risk of infection across hosts.

    Equivalent to reducing :func:`risk_of_infection_matrix` along the host axis, but
    accumulates in float64 one tip-block at a time so peak memory is bounded by
    ``max_block_elements`` rather than ``n_tips * n_hosts``.

    Args:
        tips_ag: ``(k, 2)`` array of tip ``(ag1, ag2)`` coordinates.
        phenotypes: ``(m, 2)`` array from :func:`pack_host_histories`.
        segment_starts: Strictly increasing host boundaries into ``phenotypes``.
        smith_conversion: Factor scaling antigenic distance to infection risk.
        homologous_immunity: Immunity against an identical antigen.
        max_block_elements: Element budget for each distance block.

    Returns:
        Tuple of ``(mean, sd)``, each ``(k,)`` float64. ``sd`` uses ``ddof=1`` and is NaN
        when only one host was sampled.

    Raises:
        ValueError: If any shape or ``reduceat`` precondition is violated.
    """
    _validate_kernel_inputs(tips_ag, phenotypes, segment_starts, max_block_elements)

    center = _center_of(phenotypes)
    pheno_centered = (phenotypes - center).astype(np.float32)
    n_tips = tips_ag.shape[0]
    n_hosts = int(segment_starts.size)
    block_size = _tips_per_block(phenotypes.shape[0], max_block_elements)

    total = np.empty(n_tips, dtype=np.float64)
    total_squared = np.empty(n_tips, dtype=np.float64)

    for start in range(0, n_tips, block_size):
        stop = min(start + block_size, n_tips)
        risks = _risk_block(
            tips_ag[start:stop],
            pheno_centered,
            center,
            segment_starts,
            smith_conversion,
            homologous_immunity,
        )
        total[start:stop] = risks.sum(axis=1)
        total_squared[start:stop] = np.square(risks).sum(axis=1)

    mean = total / n_hosts
    if n_hosts < 2:
        return mean, np.full(n_tips, np.nan)

    variance = (total_squared - n_hosts * mean * mean) / (n_hosts - 1)
    # Cancellation in the sum-of-squares form can push a near-zero variance negative.
    return mean, np.sqrt(np.maximum(variance, 0.0))


def risk_of_infection_over_time(
    tips_df: pd.DataFrame,
    histories_df: pd.DataFrame,
    delta_t: float,
    n_hosts: Optional[int],
    seed: int,
    smith_conversion: float,
    homologous_immunity: float,
    max_block_elements: int,
) -> pd.DataFrame:
    """Compute each tip's mean risk of infection at each evaluated timepoint.

    Every tip is scored at every timepoint, matching
    ``antigentools.analysis.calc_variance_over_time``. Demes are pooled into a single
    global host sample.

    Args:
        tips_df: Unique tips with ``name``, ``year``, ``ag1``, ``ag2``.
        histories_df: Raw histories from :func:`load_raw_histories`.
        delta_t: Spacing between evaluated timepoints, in years.
        n_hosts: Hosts to sample per timepoint, or None for all hosts.
        seed: RNG seed for host sampling.
        smith_conversion: Factor scaling antigenic distance to infection risk.
        homologous_immunity: Immunity against an identical antigen.
        max_block_elements: Element budget for each distance block.

    Returns:
        Long DataFrame with columns ``year, name, mean_risk_of_infection_experienced,
        sd_risk_of_infection_experienced, mean_risk_of_infection_population,
        n_hosts_sampled, n_hosts_total, naive_fraction``.

    Raises:
        ValueError: If required tip columns are missing or tip names are not unique.
    """
    missing = [col for col in REQUIRED_TIPS_COLUMNS if col not in tips_df.columns]
    if missing:
        raise ValueError(f"tips frame missing required columns: {missing}")
    if tips_df["name"].duplicated().any():
        raise ValueError(
            "tips frame has duplicate 'name' values; pass the deduplicated unique tips "
            "(see scripts/parse_sim_outputs.py)"
        )

    timepoints = select_timepoints(histories_df["year"].to_numpy(), delta_t)
    tips_ag = tips_df[["ag1", "ag2"]].to_numpy(dtype=np.float64)
    names = tips_df["name"].to_numpy()
    rng = np.random.default_rng(seed)

    frames: List[pd.DataFrame] = []
    for timepoint in timepoints:
        year_df = histories_df[
            np.abs(histories_df["year"].to_numpy() - timepoint) <= YEAR_TOL
        ]
        phenotypes, segment_starts, n_hosts_total = pack_host_histories(
            year_df, n_hosts, rng
        )
        naive_fraction = global_naive_fraction(year_df)

        mean_experienced, sd_experienced = risk_of_infection_stats(
            tips_ag,
            phenotypes,
            segment_starts,
            smith_conversion,
            homologous_immunity,
            max_block_elements,
        )
        # A naive host has an empty memory, so its risk clips to 1.0. Applying the
        # naive fraction analytically rather than sampling naive slots is exact -- see
        # the derivation in global_naive_fraction -- and adds no Monte Carlo variance.
        mean_population = naive_fraction + (1.0 - naive_fraction) * mean_experienced

        frames.append(
            pd.DataFrame(
                {
                    "year": timepoint,
                    "name": names,
                    "mean_risk_of_infection_experienced": mean_experienced,
                    "sd_risk_of_infection_experienced": sd_experienced,
                    "mean_risk_of_infection_population": mean_population,
                    "n_hosts_sampled": segment_starts.size,
                    "n_hosts_total": n_hosts_total,
                    "naive_fraction": naive_fraction,
                }
            )
        )
        logger.debug(
            "year %.4f: %d/%d hosts, %d memory entries, naive_fraction=%.4f",
            timepoint,
            segment_starts.size,
            n_hosts_total,
            phenotypes.shape[0],
            naive_fraction,
        )

    return pd.concat(frames, ignore_index=True)


def variance_from_risk(
    risk_df: pd.DataFrame,
    tips_df: pd.DataFrame,
    variant_cols: Sequence[str],
    n_variant_window: float,
    host_weighting: str,
) -> pd.DataFrame:
    """Collapse per-tip risk into mean within-variant fitness variance per timepoint.

    A variant's fitness spread is the variance of its members' mean risk of infection;
    the reported value averages that over variants. Variance is taken over *all* tips,
    while ``n_variants`` counts only variants sampled in ``(t - n_variant_window, t]`` --
    both matching ``antigentools.analysis.calc_variance_over_time``.

    The output columns are deliberately identical to that function's, so this is a
    drop-in replacement for the centroid method everywhere downstream. The choice of
    weighting is recorded in the caller's log, not in a column.

    Args:
        risk_df: Output of :func:`risk_of_infection_over_time`.
        tips_df: Tips frame carrying ``name``, ``year`` and the variant columns.
        variant_cols: Variant assignment columns, e.g. ``["variant_ag"]``.
        n_variant_window: Width in years of the window used for ``n_variants``.
        host_weighting: Which risk column to take the variance of; one of
            :data:`HOST_WEIGHTINGS`. Note that ``population`` is an exact affine image of
            ``experienced`` within a timepoint, so the two differ only by a factor of
            ``(1 - naive_fraction) ** 2``.

    Returns:
        Long DataFrame with columns ``year, method, mean_variance, n_variants``.

    Raises:
        ValueError: If ``variant_cols`` is empty, a column is absent from ``tips_df``, or
            ``host_weighting`` is not a known weighting.
    """
    if not variant_cols:
        raise ValueError("variant_cols must not be empty")
    absent = [col for col in variant_cols if col not in tips_df.columns]
    if absent:
        raise ValueError(f"tips frame missing variant columns: {absent}")
    if host_weighting not in RISK_COLUMNS:
        raise ValueError(
            f"host_weighting must be one of {HOST_WEIGHTINGS}; got {host_weighting!r}"
        )

    risk_column = RISK_COLUMNS[host_weighting]
    merged = risk_df.merge(
        tips_df[["name", *variant_cols]], on="name", how="left", validate="many_to_one"
    )

    tip_years = tips_df["year"].to_numpy()
    records: List[Dict[str, object]] = []

    for timepoint, group in merged.groupby("year", sort=True):
        window = (tip_years > timepoint - n_variant_window) & (tip_years <= timepoint)
        for variant_col in variant_cols:
            variances = group.groupby(variant_col)[risk_column].var()
            records.append(
                {
                    "year": timepoint,
                    "method": variant_col.replace("variant_", ""),
                    "mean_variance": variances.mean(),
                    "n_variants": int(tips_df.loc[window, variant_col].nunique()),
                }
            )

    return pd.DataFrame.from_records(records)
