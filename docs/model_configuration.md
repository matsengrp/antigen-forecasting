# Model Configuration

This page describes inference settings and configuration for growth advantage models.

## Inference

All models use Stochastic Variational Inference (SVI) via the `evofr` package.

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `iters` | 50,000 | Optimization iterations |
| `lr` | 0.01 | Learning rate |
| `num_samples` | 500 | Posterior samples to draw |

---

## Configuration Files

Model parameters can be customized via JSON config files:

```bash
python scripts/run_model.py \
    --data_path data/build/2040-10-01/ \
    --country tropics \
    --model GARW \
    --output_dir results/ \
    --config configs/default_config.json
```

### Schema Overview

```json
{
  "seed_L": 14,
  "forecast_L": 365,
  "inference": {
    "method": "InferFullRank",
    "settings": {
      "iters": 50000,
      "lr": 0.01,
      "num_samples": 500
    }
  },
  "generation_time": {
    "distribution": "gamma",
    "parameters": {"mean": 3.0, "std": 1.2}
  },
  "model_specific": {
    "MLR": {"tau": 3.0},
    "FGA": {...},
    "GARW": {...}
  }
}
```

### Key Sections

| Section | Purpose |
|---------|---------|
| `seed_L` | Days to seed renewal models |
| `forecast_L` | Forecast horizon in days |
| `inference` | SVI settings |
| `generation_time` | Generation interval distribution |
| `delay_distribution` | Reporting delay distribution |
| `model_specific` | Per-model hyperparameters |

---

## Model-Specific Parameters

### MLR

```json
"MLR": {
  "tau": 3.0
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 3.0 | Generation time for $R_t$ calculation |

### FGA

```json
"FGA": {
  "ga_prior": 0.1,
  "case_likelihood": {
    "type": "ZINegBinomCases",
    "concentration": 0.05
  },
  "seq_likelihood": {
    "type": "DirMultinomialSeq",
    "concentration": 100
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ga_prior` | 0.1 | Prior std on growth advantages ($\sigma_\delta$) |
| `case_likelihood.concentration` | 0.05 | NegBinom concentration |
| `seq_likelihood.concentration` | 100 | Dirichlet concentration ($\kappa$) |

### GARW

```json
"GARW": {
  "ga_prior_mean": 0.1,
  "ga_prior_std": 0.01,
  "prior_family": "Normal",
  "case_likelihood": {...},
  "seq_likelihood": {...}
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ga_prior_mean` | 0.1 | Prior mean on initial $\delta_v(0)$ |
| `ga_prior_std` | 0.01 | Random walk innovation std ($\sigma_\epsilon$) |
| `prior_family` | "Normal" | Prior distribution family |

---

## Epidemiological Parameters

### Generation Time

Time between successive infections in a transmission chain.

```json
"generation_time": {
  "distribution": "gamma",
  "parameters": {
    "mean": 3.0,
    "std": 1.2
  }
}
```

Used for:
- MLR: Converting $\beta_v$ to $R_v$
- FGA/GARW: Renewal equation convolution kernel

### Delay Distribution

Reporting delay from infection to case observation.

```json
"delay_distribution": {
  "distribution": "lognorm",
  "parameters": {
    "mean": 3.1,
    "std": 1.0
  }
}
```

---

## Convergence Diagnostics

VI convergence is tracked via ELBO trajectory. Diagnostics are saved to:

```
results/{build}/convergence-diagnostics/{MODEL}_{location}_{date}_vi_diagnostics.json
```

Key metrics:
- `initial_loss` / `final_loss`: ELBO values
- `relative_change`: Fractional improvement in final window
- `converged`: Boolean based on threshold (default: 0.5)
