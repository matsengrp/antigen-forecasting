# Empirical Growth Rates

This page documents the calculation of empirical growth rates ($r_{\text{data}}$) for model evaluation.

## Definition

The instantaneous growth rate for variant $v$ at time $t$:

$$
r_v(t) = \frac{\log I_v(t) - \log I_v(t - \Delta t)}{\Delta t}
$$

where $I_v(t)$ is variant incidence:

$$
I_v(t) = \pi_v(t) \cdot C(t)
$$

- $\pi_v(t)$: variant frequency
- $C(t)$: total case count

---

## Calculation Pipeline

```
seq_counts.tsv + case_counts.tsv
    ↓ align to weekly boundaries
variant × time frequency matrix
    ↓ spline smoothing
smoothed frequencies + incidence
    ↓ log-difference
empirical growth rates (r_data)
```

### Step 1: Date Alignment

Dates are aligned to weekly boundaries (Monday week starts) for consistent merging between sequence and case data.

### Step 2: Complete Frequency Matrix

A complete variant × time matrix is constructed:
- All variant-time combinations represented
- Missing combinations filled with 0
- Frequencies calculated as $\pi_v(t) = n_v(t) / N(t)$

### Step 3: Spline Smoothing

Raw counts and frequencies are smoothed to reduce noise.

---

## Spline Smoothing

Univariate splines minimize the penalized least squares objective:

$$
\hat{f}(t) = \text{argmin}_f \sum_i (y_i - f(t_i))^2 + s \int (f''(t))^2 dt
$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s` | 1.0 | Smoothing factor (larger = smoother) |
| `k` | 3 | Spline degree (cubic) |

### Post-Smoothing Processing

1. **Clip** smoothed frequencies to $[0, 1]$
2. **Re-normalize** frequencies to sum to 1.0 at each time point
3. **Calculate incidence**: $I_v(t) = \pi_v^{\text{smooth}}(t) \cdot C(t)$

---

## Filtering Criteria

Points are filtered before computing growth rates:

| Filter | Default | Rationale |
|--------|---------|-----------|
| `min_sequence_count` | 10 | Low counts → unstable frequencies |
| `min_variant_frequency` | 0.01 | Rare variants → noisy growth rates |
| `min_variant_incidence` | 50 | Low incidence → unreliable estimates |
| `min_segment_length` | 3 | Short segments → insufficient data |
| `skip_first_n_points` | 2 | Initial points often noisy |

### Segment Filtering

Growth rates are only trusted for contiguous segments of valid data. The `connect_gaps` option controls whether gaps in data are bridged or treated as segment breaks.

---

## Implementation

The pipeline is implemented in `antigentools/analysis.py`:

```python
from antigentools.analysis import get_filtered_growth_rates_df

growth_rates_df = get_filtered_growth_rates_df(
    build="flu-simulated-150k-samples",
    model="GARW",
    location="tropics",
    pivot_date="2040-10-01",
    spline_smoothing_factor=1.0,
    spline_order=3,
    min_sequence_count=10,
    min_variant_frequency=0.01,
    min_variant_incidence=50.0
)
```

### Output Columns

| Column | Description |
|--------|-------------|
| `variant` | Variant identifier |
| `date` | Observation date |
| `sequences` | Raw sequence count |
| `smoothed_sequences` | Spline-smoothed count |
| `variant_frequency` | Raw frequency |
| `variant_frequency_smoothed` | Smoothed + renormalized frequency |
| `cases` | Total case count |
| `variant_incidence_smoothed` | $\pi_v^{\text{smooth}} \cdot C$ |
| `growth_rate_r_data` | Empirical growth rate |
| `median_r` | Model-predicted growth rate |
| `abs_error` | $|r_{\text{data}} - r_{\text{model}}|$ |

---

## Evaluation Metrics

### Window-Level

Aggregate performance per analysis window (pivot date + location):

| Metric | Formula |
|--------|---------|
| Correlation | $\rho(r_{\text{model}}, r_{\text{data}})$ |
| MAE | $\frac{1}{n}\sum |r_{\text{model}} - r_{\text{data}}|$ |
| RMSE | $\sqrt{\frac{1}{n}\sum (r_{\text{model}} - r_{\text{data}})^2}$ |
| $R^2$ | $1 - SS_{\text{res}} / SS_{\text{tot}}$ |

### Variant-Level

Per-variant metrics within each window:

```python
from antigentools.analysis import calculate_variant_mae

variant_metrics = calculate_variant_mae(
    growth_rates_df,
    min_sequence_count=10,
    min_variant_frequency=0.01,
    min_variant_incidence=50.0
)
```

Additional variant-level outputs:
- `normalized_mae`: MAE / max($|r_{\text{data}}|$)
- `total_sequences`: Sum of sequences for variant
- `max_variant_frequency`: Peak frequency achieved

---

## Scoring Script

Batch evaluation across all windows:

```bash
python scripts/score_growth_rates.py \
    --config configs/benchmark_config.yaml \
    --build flu-simulated-150k-samples \
    --output-dir results/flu-simulated-150k-samples/
```

### Outputs

| File | Description |
|------|-------------|
| `window_growth_rates.tsv` | Window-level metrics |
| `variant_growth_rates.tsv` | Variant-level metrics |
| `vi_convergence_diagnostics.tsv` | Inference diagnostics |
| `growth-rates/{MODEL}/growth_rates_{loc}_{date}.tsv` | Per-window growth rates |
