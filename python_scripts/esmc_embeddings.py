from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import typer
from pathlib import Path

def main(fasta_file: str, output_file: str):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"Loaded {len(sequences)} sequences from FASTA file.")

    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    client = ESMC.from_pretrained("esmc_300m").to(device)
    
    embeddings_dict = {}

    for record in sequences:
        protein = ESMProtein(sequence=str(record.seq))
        protein_tensor = client.encode(protein)
        output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        # Mean pool over sequence dimension (dim=1) after squeezing batch
        embeddings = output.embeddings.squeeze(0).mean(dim=0)  # per-sequence embedding
        embeddings_dict[record.id] = embeddings.cpu()
    
    # Save embeddings to torch file
    torch.save(embeddings_dict, output_file)
    print(f"\nSaved {len(embeddings_dict)} embeddings to {output_file}")

if __name__ == "__main__":
    typer.run(main)