#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Description:
    -----------
The goal of this script is to split a mmseqs all_seqs file into individual fasta files.    
    Parameters:
    -----------
None
    Returns:
    --------
writes individual fasta files to disk

"""

import sys
from Bio import SeqIO
import os


def split_mmseqs_all_seqs(mmseqs_all_seqs_file, output_dir):
    output_handle = None
    # open file and iterate over records
    with open(mmseqs_all_seqs_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # find cluster header
            if record.seq == "": # empty record is cluster name
                cluster_id = record.id
                # clean cluster_id name
                cluster_id = cluster_id.replace("|", "")

                # Close the previous output file handle if it exists
                if output_handle:
                    output_handle.close()

                # Open a new output file for the current cluster
                output_handle = open(f"{output_dir}/{cluster_id}.fasta", "w")
                continue  # skip empty cluster header record

            # write remaining sequence records to file until new header is found
            SeqIO.write(record, output_handle, "fasta")

    # Close the last output file handle
    if output_handle:
        output_handle.close()


def main():
    """Main function"""
    print("cluster_splitter")
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
    split_mmseqs_all_seqs(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()