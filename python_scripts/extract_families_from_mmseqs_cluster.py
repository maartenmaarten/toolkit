#!/usr/bin/env python3

import os
import sys
import argparse
from collections import defaultdict
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequence clusters from MMseqs2 clustering results"
    )
    parser.add_argument("--clusters", required=True,
                       help="MMseqs2 cluster TSV file")
    parser.add_argument("--fasta", required=True,
                       help="Original FASTA file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for cluster FASTA files")
    
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading sequences from {args.fasta}...")
    sequences = {}
    for record in SeqIO.parse(args.fasta, "fasta"):
        sequences[record.id] = record
    print(f"Loaded {len(sequences)} sequences")
    
    # Parse clusters
    print(f"Parsing clusters from {args.clusters}...")
    clusters = defaultdict(list)
    with open(args.clusters, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[representative].append(member)
    
    print(f"Found {len(clusters)} clusters")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write cluster FASTA files for all clusters
    cluster_count = 0
    
    for rep_id, members in clusters.items():
        cluster_seqs = []
        for member_id in members:
            if member_id in sequences:
                cluster_seqs.append(sequences[member_id])
        
        if cluster_seqs:  # Only write if we have sequences
            output_file = os.path.join(
                args.output_dir,
                f"cluster_{cluster_count:04d}.fasta"
            )
            SeqIO.write(cluster_seqs, output_file, "fasta")
            cluster_count += 1
    
    print(f"Created {cluster_count} cluster FASTA files")


if __name__ == "__main__":
    main()