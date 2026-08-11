#!/usr/bin/env python3
"""
Summarize MMseqs easy-cluster *_cluster.tsv output files in a directory.

For each file, computes:
  - n_clusters
  - mean cluster size
  - stdev of cluster size
  - n_sequences (total members)
  - n_singletons (clusters of size 1)

Writes a summary TSV and a couple of diagnostic plots.

Usage:
    python summarize_mmseqs_clusters.py <input_dir> [--pattern '*_cluster.tsv'] [--out summary.tsv] [--plotdir plots]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cluster_sizes_from_file(path):
    """Read an MMseqs cluster.tsv (rep<TAB>member, no header) and return cluster sizes."""
    sizes = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep = parts[0]
            sizes[rep] = sizes.get(rep, 0) + 1
    return np.array(list(sizes.values()), dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", help="Directory containing MMseqs cluster.tsv files")
    ap.add_argument("--pattern", default="*_cluster.tsv",
                     help="Glob pattern to match cluster files (default: *_cluster.tsv)")
    ap.add_argument("--out", default="cluster_summary.tsv",
                     help="Output summary TSV path")
    ap.add_argument("--plotdir", default="plots",
                     help="Directory to save plots into")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        sys.exit(f"No files matching {args.pattern} found in {args.input_dir}")

    rows = []
    for f in files:
        sizes = cluster_sizes_from_file(f)
        if sizes.size == 0:
            print(f"WARNING: no clusters parsed from {f}, skipping", file=sys.stderr)
            continue
        rows.append({
            "file": os.path.basename(f),
            "n_clusters": len(sizes),
            "n_sequences": int(sizes.sum()),
            "mean_cluster_size": sizes.mean(),
            "stdev_cluster_size": sizes.std(ddof=1) if len(sizes) > 1 else 0.0,
            "median_cluster_size": float(np.median(sizes)),
            "max_cluster_size": int(sizes.max()),
            "n_singletons": int((sizes == 1).sum()),
            "frac_singletons": float((sizes == 1).sum() / len(sizes)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote summary for {len(df)} files to {args.out}")

    os.makedirs(args.plotdir, exist_ok=True)

    # 1. n_clusters per file
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.3), 4))
    ax.bar(df["file"], df["n_clusters"], color="steelblue")
    ax.set_ylabel("Number of clusters")
    ax.set_xlabel("File")
    ax.set_title("Number of clusters per file")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(args.plotdir, "n_clusters_per_file.png"), dpi=150)
    plt.close(fig)

    # 2. mean cluster size with stdev error bars
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.3), 4))
    ax.bar(df["file"], df["mean_cluster_size"], yerr=df["stdev_cluster_size"],
           color="darkorange", capsize=2)
    ax.set_ylabel("Mean cluster size (+/- stdev)")
    ax.set_xlabel("File")
    ax.set_title("Mean cluster size per file")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(args.plotdir, "mean_cluster_size_per_file.png"), dpi=150)
    plt.close(fig)

    # 3. scatter: n_clusters vs n_sequences (sanity/overview)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["n_sequences"], df["n_clusters"], alpha=0.6, color="firebrick")
    ax.set_xlabel("Number of sequences")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Clusters vs. sequences per file")
    plt.tight_layout()
    fig.savefig(os.path.join(args.plotdir, "n_clusters_vs_n_sequences.png"), dpi=150)
    plt.close(fig)

    # 4. distribution of fraction singletons across files
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["frac_singletons"], bins=20, color="seagreen", edgecolor="black")
    ax.set_xlabel("Fraction of clusters that are singletons")
    ax.set_ylabel("Number of files")
    ax.set_title("Distribution of singleton fraction across files")
    plt.tight_layout()
    fig.savefig(os.path.join(args.plotdir, "singleton_fraction_hist.png"), dpi=150)
    plt.close(fig)

    # 5. histogram of n_clusters across files
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_clusters"], bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Number of files")
    ax.set_title("Distribution of cluster counts across files")
    plt.tight_layout()
    fig.savefig(os.path.join(args.plotdir, "n_clusters_histogram.png"), dpi=150)
    plt.close(fig)

    print(f"Plots written to {args.plotdir}/")


if __name__ == "__main__":
    main()