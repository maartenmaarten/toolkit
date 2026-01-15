from Bio import SeqIO
import re
import sys
import os

# Input and output file paths
input_fasta = sys.argv[1]  # First command-line argument for input FASTA file
output_fasta = sys.argv[2]  # Second command-line argument for output FASTA file
ec_counts_file = sys.argv[3]  # Third argument for EC counts output file

# Regular expression to match EC numbers (e.g., "x.x.x.x" where x is a number)
ec_number_pattern = re.compile(r"\b\d+\.\d+\.\d+\.\d+\b")

# Dictionary to track EC numbers
ec_counts = {}
sequence_count = 0

# Open the output file for writing
with open(output_fasta, "w") as output_handle:
    # Parse the input FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        # Find all EC numbers in the header
        ec_numbers = ec_number_pattern.findall(record.description)
        
        # Check if the header contains any EC numbers
        if ec_numbers:
            # Write the sequence to the output file
            SeqIO.write(record, output_handle, "fasta")
            sequence_count += 1
            
            # Count each EC number found using dictionary
            for ec_num in ec_numbers:
                if ec_num in ec_counts:
                    ec_counts[ec_num] += 1
                else:
                    ec_counts[ec_num] = 1
            
            # Print the sequence and its EC numbers
            print(f"Sequence {sequence_count}: {record.id} - EC numbers: {', '.join(ec_numbers)}")

print(f"\nSequences with EC numbers have been written to {output_fasta}.")
print(f"Total sequences written: {sequence_count}")
print(f"Total unique EC numbers found: {len(ec_counts)}")

# Print EC number statistics - sort by count (descending)
print("\nEC Number Frequency:")
print("EC Number\tCount")
print("-" * 20)
# Sort dictionary by values (counts) in descending order
sorted_ec_counts = sorted(ec_counts.items(), key=lambda x: x[1], reverse=True)
for ec_num, count in sorted_ec_counts:
    print(f"{ec_num}\t{count}")

# Ensure the directory exists for EC counts file
os.makedirs(os.path.dirname(ec_counts_file), exist_ok=True)

# Write EC counts to the specified file
with open(ec_counts_file, 'w') as ec_handle:
    ec_handle.write("EC_Number\tCount\n")
    for ec_num, count in sorted_ec_counts:
        ec_handle.write(f"{ec_num}\t{count}\n")

print(f"EC number counts saved to: {ec_counts_file}")

