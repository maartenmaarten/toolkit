#!/usr/bin/env python3
"""
Plot cluster size distribution from mmseqs cluster TSV file.

Usage:
    python plot_cluster_sizes.py <cluster_tsv_file>
    python plot_cluster_sizes.py <cluster_tsv_file> --output clusters_sizes.png
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_cluster_sizes(tsv_file):
    """
    Get the size of each cluster from mmseqs output TSV file.
    The TSV file has format: representative_id, cluster_member_id
    
    Returns a Series of cluster sizes.
    """
    try:
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        # Count cluster members for each representative (first column)
        cluster_sizes = df[0].value_counts().sort_values(ascending=False)
        return cluster_sizes
    except Exception as e:
        print(f"Error reading {tsv_file}: {e}", file=sys.stderr)
        sys.exit(1)


def plot_cluster_size_distribution(cluster_sizes, output_file=None, title=None):
    """
    Plot histogram with KDE of cluster sizes.
    
    Args:
        cluster_sizes: pandas Series of cluster sizes
        output_file: Optional path to save the plot
        title: Optional custom title for the plot
    """
    if len(cluster_sizes) == 0:
        print("Error: No clusters found", file=sys.stderr)
        sys.exit(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(cluster_sizes, ax=ax, color='#1f77b4', bins=100, kde=False, stat='count')
    
    ax.set_xlabel('Cluster Size (number of members)', fontsize=12)
    ax.set_ylabel('Count (log scale)', fontsize=12)
    ax.set_yscale('log')
    
    if title is None:
        title = f'Cluster Size Distribution (n={len(cluster_sizes)} clusters)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add statistics text
    stats_text = (f"n clusters: {len(cluster_sizes)}\n"
                  f"Mean size: {cluster_sizes.mean():.1f}\n"
                  f"Median size: {cluster_sizes.median():.1f}\n"
                  f"Min size: {cluster_sizes.min()}\n"
                  f"Max size: {cluster_sizes.max()}")
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    
   #plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cluster size distribution from mmseqs cluster TSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tsv_file", help="Input mmseqs cluster TSV file")
    parser.add_argument("-o", "--output", dest="output_file", default=None,
                        help="Output PNG file (default: show plot only)")
    parser.add_argument("-t", "--title", dest="title", default=None,
                        help="Custom plot title")
    
    args = parser.parse_args()
    
    # Validate input file
    tsv_path = Path(args.tsv_file)
    if not tsv_path.is_file():
        print(f"Error: TSV file not found: {args.tsv_file}", file=sys.stderr)
        sys.exit(1)
    
    # Load cluster sizes
    print(f"Loading clusters from: {args.tsv_file}")
    cluster_sizes = get_cluster_sizes(args.tsv_file)
    print(f"Found {len(cluster_sizes)} clusters")
    
    # Generate plot
    output_file = args.output_file
    if output_file is None and args.title is None:
        # Auto-generate output filename if not provided
        output_file = tsv_path.with_suffix('.png').name
        if output_file == '.png':
            output_file = 'cluster_sizes.png'
    
    plot_cluster_size_distribution(cluster_sizes, output_file, args.title)


if __name__ == "__main__":
    main()
