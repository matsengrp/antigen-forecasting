#!/usr/bin/env python
"""
Assign variant labels using three methods: antigenic, sequence-based, and phylogenetic.

Runs all three variant assignment pipelines and outputs a combined tips DataFrame
with variant_ag, variant_tsne, and variant_phylo columns.

Usage:
    python scripts/assign_all_variants.py \
        --tips data/build/tips.tsv \
        --fasta data/build/sequences.fasta \
        --output data/build/tips_with_variants.tsv \
        --work-dir data/build/variant_assignment/
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from antigentools.utils import extract_clade_assignments_from_auspice


def assign_antigenic_variants(tips_df: pd.DataFrame, k: int = 30, random_state: int = 42) -> pd.DataFrame:
    """Assign variants via k-means on antigenic coordinates."""
    coords = tips_df[['ag1', 'ag2']].values
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    tips_df = tips_df.copy()
    tips_df['variant_ag'] = kmeans.fit_predict(coords)
    return tips_df


def run_sequence_embedding_pipeline(
    fasta_path: str,
    ref_path: str,
    work_dir: Path,
    k: int = 30,
    embedding_method: str = 't-sne',
    n_threads: int = 128,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run pathogen-embed pipeline and cluster embeddings.

    Reuses alignment and distance matrix if they already exist.
    Returns DataFrame with strain name and variant_tsne column.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = work_dir / "sequences.alignment"
    distance_path = work_dir / "distance-matrix.csv"
    embeddings_path = work_dir / f"{embedding_method}-embeddings.csv"

    # Step 1: Align sequences (skip if exists)
    if alignment_path.exists():
        print("  [1/3] Alignment exists, skipping...")
    else:
        print("  [1/3] Aligning sequences...")
        subprocess.run([
            "augur", "align",
            "--sequences", fasta_path,
            "--reference-sequence", ref_path,
            "--output", str(alignment_path),
            "--remove-reference",
            "--fill-gaps",
            "--nthreads", str(n_threads)
        ], check=True)

    # Step 2: Calculate pairwise distances (skip if exists)
    if distance_path.exists():
        print("  [2/3] Distance matrix exists, skipping...")
    else:
        print("  [2/3] Computing pairwise distances...")
        subprocess.run([
            "pathogen-distance",
            "--alignment", str(alignment_path),
            "--output", str(distance_path)
        ], check=True)

    # Step 3: Learn embeddings
    print("  [3/3] Learning embeddings...")
    subprocess.run([
        "pathogen-embed",
        "--alignment", str(alignment_path),
        "--distance-matrix", str(distance_path),
        "--output-dataframe", str(embeddings_path),
        embedding_method
    ], check=True)

    # Load embeddings and cluster
    embeddings_df = pd.read_csv(embeddings_path)
    embeddings_df.rename(columns={'strain': 'name'}, inplace=True)

    # Get embedding columns
    if embedding_method == 't-sne':
        embed_cols = ['tsne_x', 'tsne_y']
    else:
        embed_cols = [c for c in embeddings_df.columns if c.startswith(embedding_method)]

    # K-means clustering
    X = embeddings_df[embed_cols].values
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    embeddings_df['variant_tsne'] = kmeans.fit_predict(X)

    return embeddings_df[['name', 'variant_tsne']]


def run_phylogenetic_pipeline(
    fasta_path: str,
    metadata_path: str,
    ref_genbank_path: str,
    weights_path: str,
    work_dir: Path,
    target_clades: int = 30,
    n_threads: int = 4
) -> dict:
    """
    Run augur pipeline and clade assignment algorithm.

    Adjusts parameters to achieve approximately target_clades variants.
    Returns dict mapping strain name to clade label.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = work_dir / "sequences.alignment"
    tree_raw_path = work_dir / "tree_raw.nwk"
    tree_path = work_dir / "tree.nwk"
    branch_lengths_path = work_dir / "branch_lengths.json"
    nt_muts_path = work_dir / "nt_muts.json"
    aa_muts_path = work_dir / "aa_muts.json"
    auspice_base_path = work_dir / "auspice_base.json"
    auspice_path = work_dir / "auspice.json"

    # Step 1: Align (skip if exists)
    if alignment_path.exists():
        print("  [1/7] Alignment exists, skipping...")
    else:
        print("  [1/7] Aligning sequences...")
        subprocess.run([
            "augur", "align",
            "--sequences", fasta_path,
            "--reference-sequence", ref_genbank_path,
            "--output", str(alignment_path),
            "--fill-gaps",
            "--nthreads", str(n_threads)
        ], check=True)

    # Step 2: Build tree (skip if exists)
    if tree_raw_path.exists():
        print("  [2/7] Raw tree exists, skipping...")
    else:
        print("  [2/7] Building phylogenetic tree...")
        subprocess.run([
            "augur", "tree",
            "--alignment", str(alignment_path),
            "--output", str(tree_raw_path),
            "--method", "iqtree",
            "--nthreads", str(n_threads)
        ], check=True)

    # Step 3: Refine tree (skip if exists)
    if tree_path.exists():
        print("  [3/7] Refined tree exists, skipping...")
    else:
        print("  [3/7] Refining tree...")
        subprocess.run([
            "augur", "refine",
            "--tree", str(tree_raw_path),
            "--alignment", str(alignment_path),
            "--output-tree", str(tree_path),
            "--output-node-data", str(branch_lengths_path)
        ], check=True)

    # Step 4: Ancestral reconstruction (skip if exists)
    if nt_muts_path.exists():
        print("  [4/7] Ancestral sequences exist, skipping...")
    else:
        print("  [4/7] Reconstructing ancestral sequences...")
        subprocess.run([
            "augur", "ancestral",
            "--tree", str(tree_path),
            "--alignment", str(alignment_path),
            "--output-node-data", str(nt_muts_path),
            "--inference", "joint"
        ], check=True)

    # Step 5: Translate (skip if exists)
    if aa_muts_path.exists():
        print("  [5/7] AA mutations exist, skipping...")
    else:
        print("  [5/7] Translating to amino acids...")
        subprocess.run([
            "augur", "translate",
            "--tree", str(tree_path),
            "--ancestral-sequences", str(nt_muts_path),
            "--reference-sequence", ref_genbank_path,
            "--output-node-data", str(aa_muts_path)
        ], check=True)

    # Step 6: Export to auspice (skip if exists)
    if auspice_base_path.exists():
        print("  [6/7] Auspice base JSON exists, skipping...")
    else:
        print("  [6/7] Exporting to Auspice JSON...")
        subprocess.run([
            "augur", "export", "v2",
            "--tree", str(tree_path),
            "--metadata", metadata_path,
            "--node-data", str(branch_lengths_path), str(nt_muts_path), str(aa_muts_path),
            "--output", str(auspice_base_path)
        ], check=True)

    # Step 7: Assign clades
    print("  [7/7] Assigning phylogenetic clades...")
    subprocess.run([
        "python", "scripts/add_new_clades.py",
        "--input", str(auspice_base_path),
        "--lineage", "h3n2",
        "--segment", "ha",
        "--weights", weights_path,
        "--new-key", "clade",
        "--output", str(auspice_path)
    ], check=True)

    # Extract clade assignments
    clade_map = extract_clade_assignments_from_auspice(str(auspice_path))
    print(f"  Assigned {len(set(clade_map.values()))} unique phylogenetic clades")

    return clade_map


def relabel_variants_by_temporal_order(df: pd.DataFrame, variant_col: str, time_col: str = 'year') -> pd.DataFrame:
    """Relabel variants based on mean temporal order."""
    variant_avg_time = df.groupby(variant_col)[time_col].mean().sort_values()
    label_mapping = {old: new for new, old in enumerate(variant_avg_time.index)}
    df = df.copy()
    df[variant_col] = df[variant_col].map(label_mapping)
    return df


def generate_metadata_from_tips(tips_df: pd.DataFrame, output_path: Path) -> str:
    """
    Generate augur-compatible metadata TSV from tips DataFrame.

    Generates minimal metadata (strain, date, country) for augur export.
    Does not include clade_membership since variant labels are relabeled
    after all assignments complete - use the final output file for that.
    """
    metadata_df = pd.DataFrame()
    metadata_df['strain'] = tips_df['name']
    metadata_df['date'] = tips_df['year']
    metadata_df['country'] = tips_df.get('country', tips_df.get('location', 'unknown'))

    metadata_df.to_csv(output_path, sep='\t', index=False)
    print(f"  Generated metadata: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Assign variant labels using antigenic, sequence, and phylogenetic methods."
    )
    parser.add_argument("--tips", "-t", required=True, help="Path to tips TSV (with ag1, ag2, nucleotideSequence)")
    parser.add_argument("--fasta", "-f", default=None, help="Path to sequences FASTA (required unless --fast)")
    parser.add_argument("--output", "-o", required=True, help="Output TSV path")
    parser.add_argument("--work-dir", "-w", default=None, help="Working directory for intermediate files")
    parser.add_argument("--ref-fasta", default="data/flu-final/ref_HA.fasta", help="Reference FASTA for alignment")
    parser.add_argument("--ref-genbank", default="data/flu-final/auspice/ref_HA.gb", help="Reference GenBank for augur")
    parser.add_argument("--metadata", default=None, help="Metadata TSV for augur export (auto-generated from tips if not provided)")
    parser.add_argument("--weights", default="configs/weights_per_site_for_clades.json", help="Mutation weights JSON")
    parser.add_argument("-k", type=int, default=30, help="Number of clusters/clades (default: 30)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for phylogenetic pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Fast mode: only run antigenic clustering (skip sequence and phylo)")
    args = parser.parse_args()

    # Validate fasta requirement
    if not args.fast and args.fasta is None:
        parser.error("--fasta is required unless --fast mode is used")

    # Setup work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="variant_assignment_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Working directory: {work_dir}")

    # Load tips (auto-detect CSV vs TSV)
    print("\nLoading tips data...")
    if args.tips.endswith('.csv'):
        tips_df = pd.read_csv(args.tips, sep=",")
    else:
        tips_df = pd.read_csv(args.tips, sep="\t")
    print(f"  Loaded {len(tips_df)} tips")

    # Defensive contract: --tips must be the canonical name->nucleotideSequence
    # dedup output of parse_sim_outputs.py (unique_tips.csv). Feeding the full
    # tips.csv would silently pick different representative rows on name
    # collisions versus the canonical two-step dedup. Use raise (not assert) so
    # the check survives `python -O`.
    n_name_dupes = len(tips_df) - tips_df["name"].nunique()
    if n_name_dupes != 0:
        raise ValueError(
            f"--tips must be deduplicated on 'name' (got {n_name_dupes} dupes); "
            f"pass unique_tips.csv from parse_sim_outputs.py, not the full tips.csv"
        )
    n_seq_dupes = len(tips_df) - tips_df["nucleotideSequence"].nunique()
    if n_seq_dupes != 0:
        raise ValueError(
            f"--tips must be deduplicated on 'nucleotideSequence' (got {n_seq_dupes} dupes)"
        )
    unique_tips_df = tips_df.reset_index(drop=True)
    print(f"  {len(unique_tips_df)} unique sequences")

    # === Method 1: Antigenic clustering ===
    print("\n[1/3] Running antigenic variant assignment...")
    unique_tips_df = assign_antigenic_variants(unique_tips_df, k=args.k, random_state=args.seed)
    print(f"  Assigned {unique_tips_df['variant_ag'].nunique()} antigenic variants")

    # Generate metadata from tips if not provided (needed for phylogenetic pipeline)
    if args.metadata is None:
        metadata_path = work_dir / "metadata.tsv"
        generate_metadata_from_tips(unique_tips_df, metadata_path)
    else:
        metadata_path = args.metadata

    if args.fast:
        print("\n[Fast mode] Skipping sequence and phylogenetic pipelines")
    else:
        # === Method 2: Sequence-based clustering ===
        print("\n[2/3] Running sequence-based variant assignment...")
        seq_work_dir = work_dir / "sequence"
        try:
            seq_variants_df = run_sequence_embedding_pipeline(
                fasta_path=args.fasta,
                ref_path=args.ref_fasta,
                work_dir=seq_work_dir,
                k=args.k,
                n_threads=128,
                random_state=args.seed
            )
            unique_tips_df = unique_tips_df.merge(seq_variants_df, on='name', how='left')
            print(f"  Assigned {unique_tips_df['variant_tsne'].nunique()} sequence variants")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Warning: Sequence pipeline failed ({e}), skipping variant_tsne")
            unique_tips_df['variant_tsne'] = np.nan

        # === Method 3: Phylogenetic clade assignment ===
        print("\n[3/3] Running phylogenetic variant assignment...")
        phylo_work_dir = work_dir / "phylogenetic"
        try:
            clade_map = run_phylogenetic_pipeline(
                fasta_path=args.fasta,
                metadata_path=metadata_path,
                ref_genbank_path=args.ref_genbank,
                weights_path=args.weights,
                work_dir=phylo_work_dir,
                target_clades=args.k,
                n_threads=args.threads
            )
            unique_tips_df['variant_phylo'] = unique_tips_df['name'].map(clade_map)
            unique_tips_df['variant_phylo'] = unique_tips_df['variant_phylo'].fillna(-1).astype(int)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Warning: Phylogenetic pipeline failed ({e}), skipping variant_phylo")
            unique_tips_df['variant_phylo'] = np.nan

    # Relabel by temporal order
    print("\nRelabeling variants by temporal order...")
    for col in ['variant_ag', 'variant_tsne', 'variant_phylo']:
        if col in unique_tips_df.columns and not unique_tips_df[col].isna().all():
            unique_tips_df = relabel_variants_by_temporal_order(unique_tips_df, col)

    # Add year_bin
    unique_tips_df['year_bin'] = np.floor(unique_tips_df['year']).astype(int)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write output (auto-detect CSV vs TSV)
    sep = "," if args.output.endswith('.csv') else "\t"
    unique_tips_df.to_csv(args.output, sep=sep, index=False)
    print(f"\nWrote {len(unique_tips_df)} rows to {args.output}")

    # Summary
    print("\nVariant assignment summary:")
    for col in ['variant_ag', 'variant_tsne', 'variant_phylo']:
        if col in unique_tips_df.columns:
            n = unique_tips_df[col].nunique()
            print(f"  {col}: {n} variants")


if __name__ == "__main__":
    main()
