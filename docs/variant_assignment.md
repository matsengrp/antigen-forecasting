# Variant Assignment

This page describes three methods for assigning variant labels to sampled viral tips. Each method groups tips based on different biological features, enabling comparison of how well different data types capture antigenic similarity.

## Overview

| Method | Input | Clustering | Key Assumption |
|--------|-------|------------|----------------|
| Antigenic | ag1, ag2 coordinates | k-means | Antigenic space directly reflects immune escape |
| Sequence | nucleotide sequences | k-means on embeddings | Sequence similarity correlates with antigenic similarity |
| Phylogenetic | phylogenetic tree | clade suggestion algorithm | Tree topology + epitope mutations define variants |

## Method 1: Antigenic Clustering

**Simplest approach** — directly clusters tips in antigenic coordinate space.

### Algorithm

1. Extract antigenic coordinates $(a_1, a_2)$ for each tip
2. Apply k-means clustering with $k$ clusters
3. Assign variant labels based on cluster membership

### Inputs

- Tips DataFrame with `ag1`, `ag2` columns

### Implementation

```python
from sklearn.cluster import KMeans

def assign_antigenic_variants(tips_df, n_clusters=30, random_state=42):
    coords = tips_df[['ag1', 'ag2']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    tips_df['variant_ag'] = kmeans.fit_predict(coords)
    return tips_df
```

---

## Method 2: Sequence-Based Clustering

Uses sequence embeddings to capture genetic relationships, then clusters in embedding space.

### Pipeline

```
sequences.fasta
    ↓ augur align
alignment.fasta
    ↓ pathogen-distance
distance_matrix.csv
    ↓ pathogen-embed (t-SNE)
embeddings.csv
    ↓ k-means
variant_labels
```

### Steps

1. **Align sequences** against reference using `augur align`
2. **Compute pairwise distances** using `pathogen-distance`
3. **Learn embeddings** using `pathogen-embed` (t-SNE or other method)
4. **Cluster embeddings** using k-means

### Inputs

- `sequences.fasta` — nucleotide sequences for each tip
- `ref_HA.fasta` — reference sequence for alignment

### Usage

```bash
bash scripts/make_sequence_embeddings.sh
```

Key parameters in script:
```bash
EMBEDDING_METHOD="tsne"
N_COMPONENTS=10
N_CLUSTERS=30
```

### Outputs

- `embeddings.csv` — low-dimensional representation of sequences
- `seq-clusters.csv` — variant assignments

---

## Method 3: Phylogenetic Clade Assignment

Uses tree topology and mutation information to suggest clade boundaries. Unlike k-means methods, this produces hierarchical variant labels.

### Algorithm Overview

The clade suggestion algorithm (`add_new_clades.py`) assigns clades based on:

1. **Bushiness score** — measures downstream tip diversity (similar to LBI)
2. **Branch score** — weighted count of mutations at epitope sites
3. **Divergence score** — nucleotide divergence since last clade breakpoint

A new clade is suggested when:

$$
\text{score} + \text{div\_score} > \text{cutoff}
$$

where:

$$
\text{score} = \frac{\text{bushiness}}{\text{bushiness} + \text{scale}} + \frac{\text{mut_weight}}{\text{branch_scale} + \text{mut_weight}}
$$

### Pipeline

```
sequences.fasta
    ↓ augur align
alignment.fasta
    ↓ augur tree (IQ-TREE)
tree_raw.nwk
    ↓ augur refine
tree.nwk
    ↓ augur ancestral
nt_muts.json
    ↓ augur translate
aa_muts.json
    ↓ augur export v2
auspice_base.json
    ↓ add_new_clades.py
auspice.json (with clade labels)
```

### Inputs

- `sequences.fasta` — nucleotide sequences
- `metadata.tsv` — tip metadata (strain, date, etc.)
- `ref_HA.gb` — reference GenBank file with gene annotations
- `weights.json` — per-site mutation weights (epitope sites weighted higher)

### Usage

```bash
bash scripts/assign-tree-based-variants.sh \
    --sequences data/build/sequences.fasta \
    --output-dir data/build/phylo-variants/ \
    --metadata data/build/metadata.tsv
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cutoff` | 1.0 | Score threshold for new clade |
| `min_size` | 25 | Minimum tips for valid clade |
| `bushiness_branch_scale` | 1.0 | Controls phylo score sensitivity |
| `divergence_scale` | 2.0 | Controls divergence score sensitivity |
| `branch_length_scale` | 2.0 | Controls mutation weight sensitivity |

### Outputs

- `auspice.json` — tree with clade annotations viewable in Auspice
- Clade labels accessible via `node_attrs['clade']`

---

## Comparing Methods

See [Calculating Fitness](calculating_fitness.md) for how we evaluate variant assignment quality using within-variant fitness variance.

**Key insight**: A good variant assignment method should group tips with similar antigenic properties, resulting in low within-variant fitness variance.
