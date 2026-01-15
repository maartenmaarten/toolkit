#!/bin/bash

# Basic shell script for performing operations on CAZy data

# Print a welcome message
echo "Welcome to the CAZy data operations script!"

# Check if a file is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Assign the input file to a variable
input_file=$1

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found!"
    exit 1
fi

# Write the input file name to the summary file
echo "Input file: $input_file" >> summary.txt

# Count the number of lines in the file
line_count=$(wc -l < "$input_file")
echo "The file '$input_file' contains $line_count lines."

# Write unique entries to a new file
output_file="${input_file%.txt}_uniq.txt"
sort "$input_file" | uniq > "$output_file"
output_line_count=$(wc -l < "$output_file")
# Append the output line count to the summary file
echo $output_line_count "uniq lines." >> summary.txt

# Extract unique items from the first column, count them, and write to a file
unique_ids=$(awk -F'\t' '{print $1}' "$input_file" | sort | uniq)
echo "$unique_ids" > cazy_ids.txt
unique_count=$(echo "$unique_ids" | wc -l)
echo $unique_count" unique families." >> summary.txt

# Count unique items in the second column and print their occurrences
echo "Counting unique items in the second column:"
awk -F'\t' '{print $2}' "$input_file" | sort | uniq -c >> summary.txt
# Add two whitespaces to the summary.txt file
echo \ >> summary.txt
echo \ >> summary.txt
# Print the contents of the summary file
echo "Contents of 'summary.txt':"
cat summary.txt
# End of script
echo "Basic operations completed!"