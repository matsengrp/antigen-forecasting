"""Metrics comparing variant-assignment methods across their cluster labels.

These helpers are lifted from ``notebooks/manuscript-figure-3-variant-assignment
-compare.ipynb`` so the aggregation script and the across-runs notebook share one
implementation. They operate on a tips DataFrame whose ``variant_*`` columns hold
integer cluster labels from each assignment method (antigenic / sequence / phylo).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

logger = logging.getLogger(__name__)


def normalized_information_distance(
    labels_x: pd.Series | np.ndarray, labels_y: pd.Series | np.ndarray
) -> float:
    """Return the normalized information distance between two label vectors.

    NID = ``1 - MI / H(joint)`` using natural-log mutual information, so 0 means
    the two partitions are identical and 1 means they are independent. When the
    joint distribution has zero entropy (both vectors are a single cluster), the
    partitions are trivially identical and 0.0 is returned.

    Args:
        labels_x: Cluster labels from one method.
        labels_y: Cluster labels from another method (same length, same order).

    Returns:
        NID in ``[0, 1]``.

    Raises:
        ValueError: If the inputs differ in length.
    """
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)
    if labels_x.shape[0] != labels_y.shape[0]:
        raise ValueError(
            f"label vectors must be the same length; got {labels_x.shape[0]} "
            f"and {labels_y.shape[0]}"
        )

    mi = mutual_info_score(labels_x, labels_y)
    contingency = contingency_matrix(labels_x, labels_y)
    joint_probs = contingency.flatten() / contingency.sum()
    joint_entropy = scipy_entropy(joint_probs[joint_probs > 0], base=np.e)
    if joint_entropy == 0:
        return 0.0
    return float(1.0 - (mi / joint_entropy))


def variant_counts_over_time(
    tips_df: pd.DataFrame,
    method_cols: list[str],
    year_col: str = "year_bin",
) -> pd.DataFrame:
    """Count distinct variants per time bin for each assignment method.

    A method whose column is absent or entirely NaN (a failed assignment track)
    is skipped with a warning rather than producing spurious zero-variant rows.

    Args:
        tips_df: Tips table with ``year_col`` and the ``method_cols`` label columns.
        method_cols: Variant-label columns to count (e.g. ``["variant_ag", ...]``).
        year_col: Time-bin column to group by (default ``"year_bin"``).

    Returns:
        Long DataFrame with columns ``[year_bin, method, n_variants]`` (``year_bin``
        named after ``year_col``). Empty if no method column is usable.
    """
    rows: list[dict] = []
    for method in method_cols:
        if method not in tips_df.columns or tips_df[method].isna().all():
            logger.warning(
                "Skipping variant counts for missing/empty column: %s", method
            )
            continue
        usable = tips_df[[year_col, method]].dropna()
        counts = usable.groupby(year_col)[method].nunique()
        for year_bin, n_variants in counts.items():
            rows.append(
                {year_col: year_bin, "method": method, "n_variants": int(n_variants)}
            )
    return pd.DataFrame(rows, columns=[year_col, "method", "n_variants"])


def nid_pairs(tips_df: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    """Compute NID for every unordered pair of assignment methods.

    Rows with a NaN label in either method of a pair are dropped before scoring;
    a pair with no comparable rows left is skipped with a warning.

    Args:
        tips_df: Tips table containing the ``method_cols`` label columns.
        method_cols: Variant-label columns to pair up.

    Returns:
        DataFrame with columns ``[method_x, method_y, nid]``, one row per pair.
    """
    rows: list[dict] = []
    for method_x, method_y in itertools.combinations(method_cols, 2):
        if method_x not in tips_df.columns or method_y not in tips_df.columns:
            logger.warning(
                "Skipping NID for missing column pair: %s, %s", method_x, method_y
            )
            continue
        pair = tips_df[[method_x, method_y]].dropna()
        if pair.empty:
            logger.warning(
                "Skipping NID for %s vs %s: no rows with both labels present",
                method_x,
                method_y,
            )
            continue
        nid = normalized_information_distance(pair[method_x], pair[method_y])
        rows.append({"method_x": method_x, "method_y": method_y, "nid": nid})
    return pd.DataFrame(rows, columns=["method_x", "method_y", "nid"])
