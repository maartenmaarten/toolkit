#!/usr/bin/env python3
"""
Script to analyze MMseqs clusters and count how many contain at least one annotated member.
Creates plots showing the relationship between clustering thresholds and annotated clusters.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_annotations(annotation_file):
    """
    Load annotations from CAZy characterized TSV file.
    Uses GenBank column as protein ID and the first column as the label
    (e.g. subfamily or family name) for mono-label analysis.

    Returns:
        annotated_ids: set of protein IDs with any annotation
        protein_to_label: dict mapping protein ID -> label string
    """
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return set(), {}

    try:
        df = pd.read_csv(annotation_file, sep='\t', header=0)
        label_col = df.columns[0]
        df['_genbank'] = df['GenBank'].astype(str)
        df['_label'] = df[label_col].astype(str)
        # Drop rows with missing IDs
        df = df[~df['_genbank'].isin({'nan', ''})]
        annotated_ids = set(df['_genbank'].unique())
        protein_to_label = dict(zip(df['_genbank'], df['_label']))
        print(
            f"Loaded {len(annotated_ids)} annotated proteins "
            f"(label column: '{label_col}')"
        )
        return annotated_ids, protein_to_label
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        return set(), {}


def count_annotated_clusters(cluster_file, protein_to_label):
    """
    Count how many clusters contain at least one annotated member, and how
    many contain more than one unique annotation label (multi-label clusters).
    Cluster file format: representative_id, member_id
    """
    if not os.path.exists(cluster_file):
        return 0, 0, 0
    try:
        df = pd.read_csv(cluster_file, sep='\t', header=None)

        # Group by representative (cluster)
        clusters = df.groupby(0)[1].apply(list).to_dict()

        annotated_cluster_count = 0
        multi_label_count = 0
        for cluster_members in clusters.values():
            labels = {protein_to_label[m] for m in cluster_members if m in protein_to_label}
            if labels:
                annotated_cluster_count += 1
                if len(labels) > 1:
                    multi_label_count += 1

        total_clusters = len(clusters)
        return annotated_cluster_count, total_clusters, multi_label_count

    except Exception as e:
        print(f"Error reading cluster file {cluster_file}: {e}")
        return 0, 0, 0


def parse_cluster_filename(filename):
    """
    Parse coverage and identity values from filename.
    Expected format: clusters_cov0.30_id0.30.tsv
    """
    try:
        parts = filename.replace("clusters_cov", "").replace("_id", ",").replace(".tsv", "").split(",")
        coverage = float(parts[0])
        identity = float(parts[1])
        return coverage, identity
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None


def main():
    """Main execution function."""
    
    annotation_file = "/Users/maartenboneschansker/Documents/GH43_deep_dive/data/cazyscraper/CAZy_characterized/GH43.tsv"
    clusters_dir = "/Users/maartenboneschansker/Documents/GH43_deep_dive/results/cluster_analysis/mmseqs_clusters"
    output_dir = "cluster_annotation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {annotation_file}")
    annotated_ids, protein_to_label = load_annotations(annotation_file)

    if not annotated_ids:
        print("No annotated IDs found. Exciting!å")
        return
    
    # Find all cluster files
    if not os.path.exists(clusters_dir):
        print(f"Error: Clusters directory not found: {clusters_dir}")
        return
    
    cluster_files = sorted([f for f in os.listdir(clusters_dir) if f.endswith('.tsv')])
    
    if not cluster_files:
        print(f"No cluster files found in {clusters_dir}")
        return
    
    print(f"Found {len(cluster_files)} cluster files")
    
    # Process each cluster file
    results = []
    
    for cluster_file in tqdm(cluster_files, desc="Processing cluster files"):
        coverage, identity = parse_cluster_filename(cluster_file)
        
        if coverage is None:
            continue
        
        cluster_path = os.path.join(clusters_dir, cluster_file)
        annotated_count, total_clusters, multi_label_count = count_annotated_clusters(
            cluster_path, protein_to_label
        )

        results.append({
            'coverage': coverage,
            'identity': identity,
            'annotated_clusters': annotated_count,
            'total_clusters': total_clusters,
            'multi_label_clusters': multi_label_count,
            'percent_annotated': (annotated_count / total_clusters * 100) if total_clusters > 0 else 0,
            'percent_multi_label': (multi_label_count / total_clusters * 100) if total_clusters > 0 else 0,
        })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    print("\n=== Results Summary ===")
    print(df_results)
    
    # Save results to CSV
    output_csv = os.path.join(output_dir, "annotated_clusters_analysis.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Create plots
    print("\nCreating plots...")
    
    # Get unique coverage values
    coverages = sorted(df_results['coverage'].unique())
    
    # Plot 1: Line plot - Annotated clusters vs Identity for each Coverage
    fig, ax = plt.subplots(figsize=(12, 7))
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, coverage in enumerate(coverages):
        color = color_cycle[i % len(color_cycle)]
        subset = df_results[df_results['coverage'] == coverage].sort_values('identity')
        ax.plot(
            subset['identity'] * 100,
            subset['annotated_clusters'],
            marker='o',
            linewidth=2,
            markersize=6,
            color=color,
            label=f'Coverage: {coverage:.2f} (≥1 label)',
        )
        ax.plot(
            subset['identity'] * 100,
            subset['multi_label_clusters'],
            marker='o',
            linewidth=2,
            markersize=6,
            linestyle='--',
            color=color,
            label=f'Coverage: {coverage:.2f} (>1 label)',
        )

    ax.set_xlabel('Identity Threshold (%)', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Annotated Clusters vs Clustering Thresholds', fontsize=14, fontweight='bold')
    ax.legend(title='Settings', fontsize=9, title_fontsize=10, handlelength=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot1_file = os.path.join(output_dir, "annotated_clusters_by_identity.png")
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 1 saved to: {plot1_file}")
    plt.close()
    
    # Plot 2: Line plot - Percent of annotated clusters
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, coverage in enumerate(coverages):
        color = color_cycle[i % len(color_cycle)]
        subset = df_results[df_results['coverage'] == coverage].sort_values('identity')
        ax.plot(
            subset['identity'] * 100,
            subset['percent_annotated'],
            marker='s',
            linewidth=2,
            markersize=6,
            color=color,
            label=f'Coverage: {coverage:.2f} (≥1 label)',
        )
        ax.plot(
            subset['identity'] * 100,
            subset['percent_multi_label'],
            marker='s',
            linewidth=2,
            markersize=6,
            linestyle='--',
            color=color,
            label=f'Coverage: {coverage:.2f} (>1 label)',
        )

    ax.set_xlabel('Identity Threshold (%)', fontsize=12)
    ax.set_ylabel('Percentage of Clusters (%)', fontsize=12)
    ax.set_title('Percent of Clusters with Annotated Members vs Thresholds', fontsize=14, fontweight='bold')
    ax.legend(title='Settings', fontsize=9, title_fontsize=10, handlelength=3)
    ax.grid(True, alpha=0.3)
    y_max = max(df_results['percent_annotated'].max(), df_results['percent_multi_label'].max())
    ax.set_ylim([0, min(y_max * 1.1, 100)])
    plt.tight_layout()

    plot2_file = os.path.join(output_dir, "percent_annotated_clusters.png")
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 2 saved to: {plot2_file}")
    plt.close()

    print(f"\n✓ All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
