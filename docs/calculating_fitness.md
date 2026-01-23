# Calculating Fitness

This page describes how we calculate fitness for tips (sampled viruses) based on their antigenic distance from host immune memory.

## Overview

Fitness in this context represents the **infection risk** of a viral variant given the host population's immune memory. Variants that are antigenically distant from past infections have higher fitness (higher chance of infecting hosts).

## Single Timepoint Fitness Calculation

Given a set of $N$ tips with antigenic coordinates $(a_1^{(i)}, a_2^{(i)})$ and host immune memory coordinates $(h_1, h_2)$, the fitness $f_i$ for tip $i$ is:

### Step 1: Euclidean Distance

$$
d_i = \sqrt{(a_1^{(i)} - h_1)^2 + (a_2^{(i)} - h_2)^2}
$$

### Step 2: Raw Risk

$$
r_i = d_i \cdot s
$$

where $s$ is the Smith conversion factor (default: $s = 0.07$).

### Step 3: Bounded Fitness

$$
f_i = \max\left(1 - \rho, \min(1, r_i)\right)
$$

where $\rho$ is homologous immunity (default: $\rho = 0.95$).

This bounds fitness to $[0.05, 1.0]$: tips identical to host memory have fitness $0.05$, while maximally distant tips have fitness $1.0$.

## Evaluating Variant Assignment Methods

To assess how well a variant assignment method groups antigenically similar tips, we calculate the **average within-variant fitness variance** over time.

### Per-Timepoint Variance

For a variant assignment method $M$ that partitions tips into $K$ variants $\{V_1, \ldots, V_K\}$:

1. Compute fitness $f_i$ for all tips using host coordinates at time $t$
2. For each variant $V_k$, calculate variance:

$$
\sigma^2_k = \text{Var}(\{f_i : i \in V_k\})
$$

3. Average across variants:

$$
\bar{\sigma}^2_M(t) = \frac{1}{K} \sum_{k=1}^{K} \sigma^2_k
$$

### Interpretation

- **Low variance** $\Rightarrow$ tips within each variant have similar fitness $\Rightarrow$ good antigenic clustering
- **High variance** $\Rightarrow$ tips within variants have heterogeneous fitness $\Rightarrow$ poor antigenic clustering

This metric directly measures whether defined variants contain antigenically homogeneous populations.

## Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Smith conversion | $s$ | 0.07 | Scales antigenic distance to infection risk |
| Homologous immunity | $\rho$ | 0.95 | Immunity against identical antigens |

## Implementation

*TODO: Create standalone script for batch processing across simulation runs.*
