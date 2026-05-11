#!/bin/bash
# This script adds taxonomy information to a CAZy file using taxonkit.
# Define input and output files
input_file=$1

# Run taxonkit on the input file
cat "${input_file}" | taxonkit name2taxid  -j 8 --name-field 6 | \
taxonkit reformat2 --taxid-field 7 -f "d__{domain|acellular root|superkingdom}\t\
p__{phylum}\t\
c__{class}\t\
o__{order}\t\
f__{family}\t\
g__{genus}\t\
s__{species}\t\
str__{subspecies|strain|no rank}" \
> "${input_file%.tsv}.tax.tsv"
