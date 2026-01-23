# Growth Advantage Models

This page documents the variant frequency forecasting models implemented in `scripts/run_model.py`.

## Overview

| Model | Growth Rate | Requires Cases | Key Feature |
|-------|-------------|----------------|-------------|
| MLR | Fixed per variant | No | Linear trend in logit space |
| FGA | Fixed per variant | Yes | Renewal model with fixed growth advantage |
| GARW | Time-varying | Yes | Random walk on growth advantages |

---

## Model 1: Multinomial Logistic Regression (MLR)

The simplest model — fits a linear trend in logit-frequency space for each variant.

### Model Structure

Variant frequencies $\pi_v(t)$ are modeled via softmax over linear predictors:

$$
\pi_v(t) = \frac{\exp(\alpha_v + \beta_v t)}{\sum_{v'} \exp(\alpha_{v'} + \beta_{v'} t)}
$$

where:
- $\alpha_v$ is the intercept for variant $v$
- $\beta_v$ is the growth rate (slope) for variant $v$
- $t$ is time in days

### Likelihood

Sequence counts follow a multinomial distribution:

$$
\mathbf{n}(t) \sim \text{Multinomial}(N(t), \boldsymbol{\pi}(t))
$$

where $N(t)$ is total sequences at time $t$.

### Effective Reproduction Number

$R_t$ is derived from the growth rate using a fixed generation time $\tau$:

$$
R_v = \exp(\beta_v \cdot \tau)
$$

Default: $\tau = 3.0$ days.

### Forecasting

Future frequencies are extrapolated by extending the linear predictor:

$$
\hat{\pi}_v(t + \Delta) = \text{softmax}(\alpha_v + \beta_v(t + \Delta))
$$

---

## Model 2: Fixed Growth Advantage (FGA)

A renewal model where each variant has a fixed growth advantage relative to a baseline.

### Model Structure

The effective reproduction number for variant $v$ at time $t$ is:

$$
R_v(t) = R_{\text{base}}(t) \cdot \exp(\delta_v)
$$

where:
- $R_{\text{base}}(t)$ is the baseline reproduction number (modeled with spline basis functions)
- $\delta_v$ is the fixed growth advantage for variant $v$

### Priors

$$
\delta_v \sim \mathcal{N}(0, \sigma_\delta^2)
$$

Default: $\sigma_\delta = 0.1$

### Renewal Equation

Expected incidence for variant $v$ follows the discrete renewal equation:

$$
I_v(t) = R_v(t) \sum_{s=1}^{S} g(s) \cdot I_v(t-s)
$$

where $g(s)$ is the discretized generation interval distribution.

### Generation Interval

The generation interval is modeled as a discretized Gamma distribution:

$$
g \sim \text{Gamma}(\mu_g, \sigma_g)
$$

Default: $\mu_g = 3.0$ days, $\sigma_g = 1.2$ days

### Likelihoods

**Case counts** — Zero-inflated Negative Binomial:

$$
C(t) \sim \text{ZI-NegBinom}(\mu = \sum_v I_v(t) * d, \phi)
$$

where $d$ is the reporting delay distribution and $\phi$ is the concentration parameter.

**Sequence counts** — Dirichlet-Multinomial:

$$
\mathbf{n}(t) \sim \text{DirMult}(N(t), \kappa \cdot \boldsymbol{\pi}(t))
$$

where $\kappa$ controls overdispersion (default: 100).

---

## Model 3: Growth Advantage Random Walk (GARW)

Extends FGA by allowing growth advantages to vary over time via a random walk.

### Model Structure

$$
R_v(t) = R_{\text{base}}(t) \cdot \exp(\delta_v(t))
$$

where $\delta_v(t)$ evolves as a random walk:

$$
\delta_v(t) = \delta_v(t-1) + \epsilon_v(t), \quad \epsilon_v(t) \sim \mathcal{N}(0, \sigma_\epsilon^2)
$$

### Priors

Initial growth advantage:
$$
\delta_v(0) \sim \mathcal{N}(\mu_\delta, \sigma_\delta^2)
$$

Default: $\mu_\delta = 0.1$, $\sigma_\delta = 0.01$

### Key Difference from FGA

- **FGA**: Single $\delta_v$ per variant (time-invariant)
- **GARW**: $\delta_v(t)$ trajectory per variant (time-varying)

This allows GARW to capture changing fitness landscapes, but requires more data to estimate reliably.

---

## Basis Functions

$R_{\text{base}}(t)$ is parameterized using B-spline basis functions:

$$
\log R_{\text{base}}(t) = \sum_{j=1}^{K} c_j B_j(t)
$$

Default: order 4 splines with $K = 10$ knots.
