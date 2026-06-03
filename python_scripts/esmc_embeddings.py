from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ESMProteinTensor, LogitsConfig
import torch
import typer
from pathlib import Path
from tqdm import tqdm


def main(
    fasta_file: str,
    output_file: str,
    batch_size: int = 16,
    checkpoint_every: int = 100,
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

    # Sort shortest-first — minimises padding waste within each batch
    sequences.sort(key=lambda r: len(r.seq))

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
        for batch_idx, i in enumerate(
            tqdm(range(0, len(todo), batch_size), desc="Batches", unit="batch")
        ):
            batch = todo[i : i + batch_size]
            lengths = [len(r.seq) for r in batch]

            # Tokenise individually (CPU-only, no model forward — fast)
            tokens = [
                client.encode(ESMProtein(sequence=str(r.seq))).sequence
                for r in batch
            ]  # each tensor: (L_i + 2,)  — BOS + amino acids + EOS

            # Pad to the longest sequence in this batch
            max_tok = max(t.shape[0] for t in tokens)
            seq_batch = torch.zeros(len(batch), max_tok, dtype=tokens[0].dtype)
            for j, t in enumerate(tokens):
                seq_batch[j, : t.shape[0]] = t

            protein_tensor = ESMProteinTensor(sequence=seq_batch.to(device))
            output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )

            emb = output.embeddings  # (B, max_tok, D)
            for j, (record, length) in enumerate(zip(batch, lengths)):
                # Pool BOS + amino acid tokens + EOS — matches single-sequence behaviour
                seq_emb = emb[j, : length + 2, :].mean(dim=0)
                embeddings_dict[record.id] = seq_emb.cpu()

            if (batch_idx + 1) % checkpoint_every == 0:
                torch.save(embeddings_dict, out_path)
                tqdm.write(f"Checkpoint: {len(embeddings_dict)} embeddings saved")

    torch.save(embeddings_dict, out_path)
    print(f"\nSaved {len(embeddings_dict)} embeddings → {out_path}")


if __name__ == "__main__":
    typer.run(main)
