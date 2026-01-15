from Bio import SeqIO
from matplotlib_venn import venn2
import sys

import matplotlib.pyplot as plt

cazy_file = sys.argv[1]  # cazy_data.txt file
dbcan_file = sys.argv[2]  # Protein FASTA file from dbCAN output

# Read cazy_data.txt and extract genbank IDs from second column
cazy_genbanks = []
with open(cazy_file, 'r') as f:
    for line in f:
        columns = line.strip().split('\t')
        if len(columns) > 1:
            cazy_genbanks.append(columns[3])

# Load protein FASTA and extract headers
fasta_genbanks = []
for record in SeqIO.parse(dbcan_file, 'fasta'):
    fasta_genbanks.append(record.id.split('|')[0])  # Assuming genbank ID is before first '|'

# Convert to sets for Venn diagram
set_cazy = set(cazy_genbanks)
set_fasta = set(fasta_genbanks)

# Create Venn diagram
plt.figure(figsize=(8, 6))
venn2([set_cazy, set_fasta], set_labels=('CAZy', 'dbCAN'), set_colors=("cyan",
                             "blue"),alpha=0.7)
plt.title('Genbank ID Overlap: CAZy vs dbCAN')
plt.savefig('venn_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"CAZy genbanks: {len(set_cazy)}")
print(f"dbCAN genbanks: {len(set_fasta)}")
print(f"Overlap: {len(set_cazy & set_fasta)}")