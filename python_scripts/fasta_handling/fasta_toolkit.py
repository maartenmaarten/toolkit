#!/usr/bin/env python3

import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import statistics
from collections import Counter
import matplotlib.pyplot as plt
import os as os



"""
fasta_toolkit.py

A toolkit for working with FASTA files.

Author: Maarten Boneschansker
Date: 2024-06-09
"""
def load_fasta(filepath):
    """
    Loads sequences from a FASTA file using Biopython.

    Args:
        filepath (str): Path to the FASTA file.

    Returns:
        list: A list of SeqRecord objects.
    """
    return list(SeqIO.parse(filepath, "fasta"))


def remove_all_stars(seq_records):
    """
    Removes all '*' characters from sequences in each SeqRecord object.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.

    Returns:
        list of SeqRecord: New list with all '*' characters removed.
    """
    cleaned_records = []
    for seq_record in seq_records:
        new_seq = str(seq_record.seq).replace('*', '')
        cleaned_record = SeqRecord(
            Seq(new_seq),
            id=seq_record.id,
            name=seq_record.name,
            description=seq_record.description,
            dbxrefs=seq_record.dbxrefs,
            features=seq_record.features,
            annotations=seq_record.annotations,
            letter_annotations=seq_record.letter_annotations
        )
        cleaned_records.append(cleaned_record)
    return cleaned_records


def write_fasta(seq_records, output_filepath):
    """
    Writes a list of SeqRecord objects to a FASTA file.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects to write.
        output_filepath (str): Path to the output FASTA file.
    """
    with open(output_filepath, "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")


def print_fasta_stats(seq_records, min_length=50, max_length=1000):
    """
    Prints statistics for a list of SeqRecord objects.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.
        min_length (int): Minimum sequence length to consider.
        max_length (int): Maximum sequence length to consider.
    """
    num_seqs = len(seq_records)
    lengths = [len(record.seq) for record in seq_records]
    avg_length = statistics.mean(lengths) if lengths else 0
    stdev_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
    print(f"Number of sequences: {num_seqs}")
    print(f"Average sequence length: {avg_length:.2f}")
    print(f"Minimum sequence length: {min(lengths) if lengths else 0}")
    print(f"Maximum sequence length: {max(lengths) if lengths else 0}")
    print(f"Standard deviation of sequence lengths: {stdev_length:.2f}")
    num_stars = sum(str(record.seq).count('*') for record in seq_records)
    num_dashes = sum(str(record.seq).count('-') for record in seq_records)
    print(f"Total '*' characters: {num_stars}")
    print(f"Total '-' characters: {num_dashes}")
    invalid_chars = set("".join(str(record.seq) for record in seq_records)) - set("ACDEFGHIKLMNPQRSTVWYBXZ*-UO")
    print(f"Invalid characters (not in amino acid alphabet): {', '.join(sorted(invalid_chars)) if invalid_chars else 'None'}")
    num_not_start_with_M = sum(1 for record in seq_records if not str(record.seq).startswith('M'))
    print(f"Number of sequences not starting with 'M': {num_not_start_with_M}")

    # print the number of sequences containing internal stop codons
    num_internal_stops = sum(1 for record in seq_records if '*' in str(record.seq[:-1]))
    print(f"Number of sequences with internal stop codons: {num_internal_stops}")

    # print the number of sequences containing non-standard amino acids (B, X, Z, U, O)
    non_standard_aas = set("BXZUO")
    num_non_standard = sum(1 for record in seq_records if any(aa in non_standard_aas for aa in str(record.seq)))
    print(f"Number of sequences with non-standard amino acids (B, X, Z, U, O): {num_non_standard}")

    # print the number of sequences with length higher than 1000
    num_long_seqs = sum(1 for record in seq_records if len(record.seq) > max_length)
    print(f"Number of sequences longer than {max_length} amino acids: {num_long_seqs}")
    
    record_long = [record.id for record in seq_records if len(record.seq) > max_length]
    # print the number of sequences with length lower than 50
    num_short_seqs = sum(1 for record in seq_records if len(record.seq) < min_length)
    print(f"Number of sequences shorter than {min_length} amino acids: {num_short_seqs}")
    record_short = [record.id for record in seq_records if len(record.seq) < min_length]
    print(f"IDs of sequences longer than {max_length} amino acids: {', '.join(sorted(record_long)) if record_long else 'None'}")
    print(f"IDs of sequences shorter than {min_length} amino acids: {', '.join(sorted(record_short)) if record_short else 'None'}")
    return


def plot_length_histogram(seq_records, bins=100, show=True, save_path=None):
    """
    Plots a histogram of sequence lengths.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.
        bins (int): Number of bins for the histogram.
        show (bool): Whether to display the plot.
        save_path (str or None): If provided, saves the plot to this path.
    """
    lengths = [len(record.seq) for record in seq_records]
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=bins, color='skyblue', edgecolor='black', log=False)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_amino_acid_composition(seq_records, show=True, save_path=None):
    """
    Plots a histogram of amino acid composition for a list of SeqRecord objects.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.
        show (bool): Whether to display the plot.
        save_path (str or None): If provided, saves the plot to this path.
    """

    aa_counts = Counter()
    for record in seq_records:
        aa_counts.update(str(record.seq))

    # Only plot standard amino acids and common symbols
    amino_acids = list("ACDEFGHIKLMNPQRSTVWYBXZ*-")
    counts = [aa_counts.get(aa, 0) for aa in amino_acids]

    plt.figure(figsize=(12, 6))
    plt.bar(amino_acids, counts, color='mediumseagreen', edgecolor='black')
    plt.title('Amino Acid Composition')
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def filter_fasta_by_length(seq_records, min_length=None, max_length=None):
    """
    Filters a list of SeqRecord objects by sequence length.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.
        min_length (int or None): Minimum length to keep (inclusive). If None, no minimum filter is applied.
        max_length (int or None): Maximum length to keep (inclusive). If None, no maximum filter is applied.

    Returns:
        list of SeqRecord: Filtered list of SeqRecord objects.
    """
    filtered_records = []
    for record in seq_records:
        seq_length = len(record.seq)
        if (min_length is None or seq_length >= min_length) and (max_length is None or seq_length <= max_length):
            filtered_records.append(record)
    return filtered_records


def filter_fasta_by_aa_content(seq_records, allowed_aas="ACDEFGHIKLMNPQRSTVWY"):
    """
    Filters a list of SeqRecord objects by allowed amino acid content.

    Args:
        seq_records (list of SeqRecord): List of SeqRecord objects.
        allowed_aas (set of str): Set of allowed amino acid characters.

    Returns:
        list of SeqRecord: Filtered list of SeqRecord objects.
    """
    filtered_records = []
    for record in seq_records:
        if all(aa in allowed_aas for aa in str(record.seq)):
            filtered_records.append(record)
    return filtered_records

def afa_to_fa(afa_file, output_fasta):
    """
    Converts an AFA (aligned FASTA) file to a standard FASTA file by removing gaps.

    Args:
        afa_file (str): Path to the input AFA file.
        output_fasta (str): Path to the output FASTA file.
    """
    seq_records = load_fasta(afa_file)
    for record in seq_records:
        seq_str = str(record.seq).replace("-", "").replace(".", "")
        record.seq = Seq(seq_str.upper())
    write_fasta(seq_records, output_fasta)

__all__ = [
    "load_fasta",
    "remove_all_stars",
    "write_fasta",
    "print_fasta_stats",
    "plot_length_histogram",
    "plot_amino_acid_composition",
    "filter_fasta_by_length",
    "filter_fasta_by_aa_content",
    "afa_to_fa"
]




def main():
    # Entry point for the script
    print("FASTA Toolkit - Ready to use.")

if __name__ == "__main__":

    main()