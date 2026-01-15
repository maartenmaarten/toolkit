#!/bin/bash
# This script adds taxonomy information to a CAZy file using taxonkit.
# Define input and output files
input_file="$1"
output_file="$2"

# Check if the input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Check if the output file is provided in the right format
if [ -z "$2" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi



# Run taxonkit name2taxid on the input file
cat "${input_file}" | taxonkit name2taxid -j 8 --name-field 3 > "${output_file}.tmp"

# sort the output file on the column containing the taxid
sort -t$'\t' -k6,6n "${output_file}.tmp" > "${output_file}.sorted.tmp"

# split the files, one for a direct hit and one for a not found hit
awk -F'\t' '{if ($6 == "") print $1, $2, $3, $4, $5}' OFS='\t' "${output_file}.sorted.tmp" > "${output_file}.not_found.tmp"
awk -F'\t' '{if ($6 != "" && $6 ~ /^[0-9]+$/) print $0}' "${output_file}.sorted.tmp" > "${output_file}.found.tmp"

# Count the lines of both files and print the results
found_count=$(wc -l < "${output_file}.found.tmp")
not_found_count=$(wc -l < "${output_file}.not_found.tmp")

echo "Lines in found file: $found_count"
echo "Lines in not found file: $not_found_count"

# Run taxonkit name2taxid on the input file with fuzzy matching
cat "${output_file}.not_found.tmp" | taxonkit name2taxid -j 8 --fuzzy --name-field 3 > "${output_file}.fuzzy.tmp"


# count the lines in fuzzy
fuzzy_count=$(awk -F'\t' '{if ($6 == "") print $0}' "${output_file}.fuzzy.tmp" | wc -l)
echo "Entries with no taxid found: $fuzzy_count"

# concatenate the found and fuzzy files
cat "${output_file}.found.tmp" "${output_file}.fuzzy.tmp" | sort -t$'\t' -k6,6nr > "${output_file}"

# run taxonkit lineage on it
cat "${output_file}" | taxonkit lineage --taxid-field 6 > "${output_file}_taxonomy.txt"


### note that for the missing ones, you need to run entrez directly on them to get taxonomy.
### this is doable in bash, but not ideal.
### You would need another of these split-loops with your files. Not ideal.
