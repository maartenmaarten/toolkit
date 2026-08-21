from Bio import SeqIO
from esm.models.esmc import ESMC
import torch
import typer
from pathlib import Path
from tqdm import tqdm


def main(
    fasta_file: str,
    output_file: str,
    checkpoint_every: int = 500,
    n_layers: int = 10,
    batch_size: int = 8,
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

    client = ESMC.from_pretrained("esmc_300m").to(device)
    pad_token_id = client.tokenizer.pad_token_id

    # Resume from an existing checkpoint so interrupted runs don't restart from scratch
    out_path = Path(output_file)
    full_dict = {}
    if out_path.exists():
        full_dict = torch.load(out_path, map_location="cpu")
        print(f"Resuming — {len(full_dict)} embeddings already saved")

    done_ids = set(full_dict.keys())
    todo = [r for r in sequences if r.id not in done_ids]
    print(f"Sequences to embed: {len(todo)}")

    # Sort by length so sequences of similar size land in the same batch,
    # minimizing padding waste.
    todo.sort(key=lambda r: len(r.seq))

    n_since_checkpoint = 0
    with torch.no_grad():
        batch_starts = range(0, len(todo), batch_size)
        for batch_start in tqdm(batch_starts, desc="Embedding", unit="batch"):
            batch = todo[batch_start : batch_start + batch_size]
            tokens = client._tokenize([str(r.seq) for r in batch])
            output = client(tokens)
            # (n_layers_total, B, L, D) — keep only the last n_layers transformer blocks
            hidden_states = output.hidden_states[-n_layers:]

            mask = tokens != pad_token_id
            for i, record in enumerate(batch):
                valid_len = int(mask[i].sum().item())
                # Trim the <cls>/<eos> special tokens, keep only real residue positions.
                # (n_layers, seq_len, hidden_dim) — full per-residue embeddings
                full_dict[record.id] = hidden_states[:, i, 1 : valid_len - 1, :].cpu()

            n_since_checkpoint += len(batch)
            if n_since_checkpoint >= checkpoint_every:
                torch.save(full_dict, out_path)
                tqdm.write(f"Checkpoint: {len(full_dict)} embeddings saved")
                n_since_checkpoint = 0

    torch.save(full_dict, out_path)
    print(f"\nSaved {len(full_dict)} full embeddings (last {n_layers} layers) → {out_path}")

    # (n_layers, hidden_dim) — mean-pooled per layer, over residue positions
    mean_dict = {k: v.mean(dim=1) for k, v in full_dict.items()}
    mean_path = out_path.with_name(f"{out_path.stem}_mean_last{n_layers}{out_path.suffix}")
    torch.save(mean_dict, mean_path)
    print(f"Saved mean-pooled embeddings (last {n_layers} layers) → {mean_path}")


if __name__ == "__main__":
    typer.run(main)
