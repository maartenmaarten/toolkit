from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import typer
from pathlib import Path
from tqdm import tqdm


def main(
    fasta_file: str,
    output_file: str,
    checkpoint_every: int = 500,
):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    print(f"Loaded {len(sequences)} sequences")

    CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    MAX_LEN = 2000

    n_too_long = sum(1 for r in sequences if len(r.seq) > MAX_LEN)
    n_noncanonical = sum(
        1 for r in sequences
        if len(r.seq) <= MAX_LEN and not set(str(r.seq).upper()).issubset(CANONICAL_AAS)
    )
    sequences = [
        r for r in sequences
        if len(r.seq) <= MAX_LEN and set(str(r.seq).upper()).issubset(CANONICAL_AAS)
    ]
    print(f"Filtered {n_too_long} sequences > {MAX_LEN} AA")
    print(f"Filtered {n_noncanonical} sequences with non-canonical amino acids")
    print(f"Remaining: {len(sequences)} sequences")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    client = ESMC.from_pretrained("esmc_600m").to(device)

    # Resume from an existing checkpoint so interrupted runs don't restart from scratch
    out_path = Path(output_file)
    embeddings_dict = {}
    if out_path.exists():
        embeddings_dict = torch.load(out_path, map_location="cpu")
        print(f"Resuming — {len(embeddings_dict)} embeddings already saved")

    done_ids = set(embeddings_dict.keys())
    todo = [r for r in sequences if r.id not in done_ids]
    print(f"Sequences to embed: {len(todo)}")

    with torch.no_grad():
        for i, record in enumerate(tqdm(todo, desc="Embedding", unit="seq")):
            protein = ESMProtein(sequence=str(record.seq))
            protein_tensor = client.encode(protein)
            output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            embeddings_dict[record.id] = output.embeddings.squeeze(0).mean(dim=0).cpu()

            if (i + 1) % checkpoint_every == 0:
                torch.save(embeddings_dict, out_path)
                tqdm.write(f"Checkpoint: {len(embeddings_dict)} embeddings saved")

    torch.save(embeddings_dict, out_path)
    print(f"\nSaved {len(embeddings_dict)} embeddings → {out_path}")


if __name__ == "__main__":
    typer.run(main)
