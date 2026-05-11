#!/usr/bin/env python3
# Usage: python genbank_to_fasta.py <input.gb> [output.fasta]
#        cat records.gb | python genbank_to_fasta.py - [output.fasta]

import sys
from Bio import SeqIO


def genbank_to_fasta(in_handle, out_handle):
    count = 0
    for record in SeqIO.parse(in_handle, "genbank"):
        out_handle.write(f">{record.id} {record.description}\n{record.seq}\n")
        count += 1
    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python genbank_to_fasta.py <input.gb> [output.fasta]")
        print("       cat records.gb | python genbank_to_fasta.py - [output.fasta]")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    if input_arg == "-":
        in_handle = sys.stdin
    else:
        in_handle = open(input_arg)

    if output_arg:
        out_handle = open(output_arg, "w")
    else:
        out_handle = sys.stdout

    try:
        count = genbank_to_fasta(in_handle, out_handle)
        if output_arg:
            print(f"Written {count} records to {output_arg}", file=sys.stderr)
    finally:
        if input_arg != "-":
            in_handle.close()
        if output_arg:
            out_handle.close()


if __name__ == "__main__":
    main()
