#!/usr/bin/env python3
import sys
from Bio import SeqIO
import os
import re

def split_dbcan_fasta_by_prefix(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Regex patterns for parent families and subfamilies
    parent_family_regex = r"^(GH|GT|PL|CE|AA|CBM)\d+"
    subfamily_regex = r"^(GH|GT|PL|CE|AA|CBM)\d+_\d+"
    multiple_family_count = {}

    with open(input_file, "r") as infile:
        for record in SeqIO.parse(infile, "fasta"):
            cazyme_families = set()  # Use set to avoid duplicates
            header_parts = record.id.split("|")

            # extract all CAZyme families from the header
            for part in header_parts:
                # First check for subfamily pattern (e.g., GH13_31)
                if re.match(subfamily_regex, part):
                    # Add the subfamily
                    cazyme_families.add(part)
                    # Extract and add the parent family (e.g., GH13 from GH13_31)
                    parent_family = re.match(parent_family_regex, part).group(0)
                    cazyme_families.add(parent_family)

                if len(cazyme_families) > 6:
                    print(f"Warning: More than 6 CAZyme families found in header: {record.id}")
                
                # Then check for parent family pattern (e.g., GH13)
                elif re.match(parent_family_regex, part):
                    cazyme_families.add(part)

            if not cazyme_families:
                print(f"Warning: No CAZyme family found in header: {record.id}")
                continue

            # Convert set back to list for consistency
            cazyme_families = list(cazyme_families)

            # record the length of the families
            multiple_family_count[len(cazyme_families)] = multiple_family_count.get(len(cazyme_families), 0) + 1

            # write the record to each corresponding family file
            for family in cazyme_families:
                output_file = os.path.join(output_dir, f"{family}.fasta")
                with open(output_file, "a") as outfile:
                    SeqIO.write(record, outfile, "fasta")
    
    print("\nSummary of sequences with multiple CAZyme families:")
    for count, num_sequences in sorted(multiple_family_count.items()):
        print(f"{num_sequences} sequences with {count} CAZyme families")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_fasta_file> <output_directory>")
        sys.exit(1)
    input_fasta_file = sys.argv[1]
    output_directory = sys.argv[2]
    split_dbcan_fasta_by_prefix(input_fasta_file, output_directory)