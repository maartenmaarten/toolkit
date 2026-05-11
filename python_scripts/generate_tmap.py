from tmap import TMAP
import numpy as np
from tmap.utils import read_fasta

ids_fasta, sequences = read_fasta("/Users/maartenboneschansker/Documents/GH43_deep_dive/data/genbank/GH43_characterized.fa", max_seqs=50_000)

print(f"{len(sequences)} sequences loaded")
print(f"Length range: {min(len(s) for s in sequences)} -- {max(len(s) for s in sequences)} aa")
print(f"First ID:  {ids_fasta[0]}")
print(f"First seq: {sequences[0][:60]}...")


	
from tmap.utils import sequence_properties

seq_props = sequence_properties(sequences)

for key, values in seq_props.items():
    print(f"{key:25s}  min={np.nanmin(values):8.1f}  max={np.nanmax(values):10.1f}")