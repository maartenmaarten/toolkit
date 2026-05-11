#!/usr/bin/env python3
"""
Script to run mmseqs clustering with varying coverage and identity thresholds.
Records number of clusters for each parameter combination and plots the results.
"""

import subprocess
import tempfile
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def count_clusters_from_tsv(tsv_file):
    """
    Count the number of clusters from mmseqs output TSV file.
    The TSV file has format: representative_id, cluster_member_id
    """
    if not os.path.exists(tsv_file):
        return 0
    
    try:
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        # Count unique representative IDs (first column)
        num_clusters = df[0].nunique()
        return num_clusters
    except Exception as e:
        print(f"Error reading {tsv_file}: {e}")
        return 0


def run_mmseqs_cluster(fasta_file, coverage, identity, output_prefix, tmp_dir):
    """
    Run mmseqs easy-cluster with specified coverage and identity parameters.
    Returns the number of clusters found.
    
    Args:
        fasta_file: Path to input protein FASTA file
        coverage: Coverage threshold (0.0-1.0)
        identity: Sequence identity threshold (0.0-1.0)
        output_prefix: Prefix for output files
        tmp_dir: Path to temporary directory for mmseqs internals
    
    Returns:
        Number of clusters formed
    """
    try:
        # Build mmseqs easy-cluster command
        cmd = [
            'mmseqs',
            'easy-cluster',
            fasta_file,
            output_prefix,
            tmp_dir,
            '-c', str(coverage),
            '--min-seq-id', str(identity),
            '--threads', '4',
            '--cov-mode', '0'  # Use coverage based on both seqs
        ]
        print(cmd)
        
        # Run the command with suppressed output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"mmseqs failed for coverage={coverage}, identity={identity}")
            print(f"Error: {result.stderr}")
            return 0
        
        # Count clusters from output TSV
        tsv_file = f"{output_prefix}_cluster.tsv"
        num_clusters = count_clusters_from_tsv(tsv_file)
        
        return num_clusters
        
    except subprocess.TimeoutExpired:
        print(f"mmseqs timeout for coverage={coverage}, identity={identity}")
        return 0
    except Exception as e:
        print(f"Error running mmseqs: {e}")
        return 0


def main():
    """Main execution function."""
    
    # Get FASTA file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python mmseqs_curves.py <protein_fasta_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    
    if not os.path.exists(fasta_file):
        print(f"Error: FASTA file not found: {fasta_file}")
        return
    
    print(f"Using FASTA file: {fasta_file}")
    
    # Create output directory for cluster TSV files
    cluster_output_dir = "mmseqs_clusters"
    os.makedirs(cluster_output_dir, exist_ok=True)
    print(f"Cluster TSV files will be saved to: {cluster_output_dir}")
    
    # Coverage settings: 0.3 to 0.9 in steps of 0.2 (4 values)
    coverage_values = np.arange(0.3, 1.0, 0.2)
    print(f"Coverage values: {coverage_values}")
    
    # Identity settings: 0.3 to 0.9 in steps of 0.1 (7 values)
    identity_values = np.arange(0.3, 1.0, 0.1)
    print(f"Identity values: {identity_values}")
    
    # Create a temporary directory for mmseqs intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dictionary to store results: {coverage: [cluster_counts]}
        results = {cov: [] for cov in coverage_values}
        
        # Nested loops: for each coverage, test all identity values
        total_runs = len(coverage_values) * len(identity_values)
        
        # Create outer progress bar for coverage values
        with tqdm(total=total_runs, desc="Overall Progress", unit="run") as pbar_global:
            for coverage in coverage_values:
                # Create inner progress bar for identity values
                for identity in tqdm(identity_values, desc=f"Coverage {coverage:.2f}", 
                                    leave=False, unit="param"):
                    # Create unique output prefix for this run
                    output_prefix = os.path.join(
                        temp_dir, 
                        f"mmseqs_cov{coverage:.2f}_id{identity:.2f}"
                    )
                    
                    # Run mmseqs and get cluster count
                    num_clusters = run_mmseqs_cluster(
                        fasta_file,
                        coverage,
                        identity,
                        output_prefix,
                        os.path.join(temp_dir, "mmseqs_tmp")
                    )
                    
                    # Save cluster TSV file with appropriate naming
                    tsv_source = f"{output_prefix}_cluster.tsv"
                    if os.path.exists(tsv_source):
                        tsv_filename = f"clusters_cov{coverage:.2f}_id{identity:.2f}.tsv"
                        tsv_dest = os.path.join(cluster_output_dir, tsv_filename)
                        shutil.copy(tsv_source, tsv_dest)
                    
                    results[coverage].append(num_clusters)
                    pbar_global.set_postfix({"clusters": num_clusters, "coverage": f"{coverage:.2f}", 
                                            "identity": f"{identity:.2f}"})
                    pbar_global.update(1)
        
        # Convert results to DataFrame for easier plotting
        print("\nProcessing results...", flush=True)
        df_results = pd.DataFrame(
            results,
            index=identity_values
        )
        df_results.index.name = "Identity Threshold"
        df_results.columns.name = "Coverage"
        
        print("\n=== Results Summary ===")
        print(df_results)
        
        # Create the plot
        print("Creating plot...", flush=True)
        plt.figure(figsize=(12, 7))
        
        # Plot a line for each coverage value
        for coverage in tqdm(coverage_values, desc="Plotting lines", leave=False):
            plt.plot(
                identity_values * 100,  # Convert to percentage
                results[coverage],
                marker='o',
                linewidth=2,
                markersize=6,
                label=f'Coverage: {coverage:.2f}'
            )
        
        plt.xlabel('Identity Threshold (%)', fontsize=12)
        plt.ylabel('Number of Clusters', fontsize=12)
        plt.title('MMseqs Clustering Results: Effect of Coverage and Identity Thresholds', 
                  fontsize=14, fontweight='bold')
        plt.legend(title='Settings', fontsize=10, title_fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_plot = "mmseqs_clustering_curves.png"
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_plot}")
        
        # Save results to CSV
        output_csv = "mmseqs_clustering_results.csv"
        df_results.to_csv(output_csv)
        print(f"✓ Results saved to: {output_csv}")
        
        # Summary of saved cluster files
        cluster_files = [f for f in os.listdir(cluster_output_dir) if f.endswith('.tsv')]
        print(f"✓ {len(cluster_files)} cluster TSV files saved to: {cluster_output_dir}/")
        
        plt.show()


if __name__ == "__main__":
    main()
