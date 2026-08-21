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

    n_layers_total = len(client.transformer.blocks)
    layer_indices = list(range(n_layers_total - n_layers, n_layers_total))
    print(f"Model has {n_layers_total} layers; saving layers {layer_indices}")

    # One file per layer, per kind (full per-residue vs. mean-pooled), named so
    # ML_utilities.load_embeddings can read any of them as-is: a flat
    # {sequence_id: Tensor} dict.
    out_path = Path(output_file)

    def full_path(layer_idx):
        return out_path.with_name(f"{out_path.stem}_full_layer{layer_idx}{out_path.suffix}")

    def mean_path(layer_idx):
        return out_path.with_name(f"{out_path.stem}_mean_layer{layer_idx}{out_path.suffix}")

    # Resume from existing checkpoints so interrupted runs don't restart from scratch
    full_by_layer, mean_by_layer = {}, {}
    for layer_idx in layer_indices:
        full_by_layer[layer_idx] = (
            torch.load(full_path(layer_idx), map_location="cpu") if full_path(layer_idx).exists() else {}
        )
        mean_by_layer[layer_idx] = (
            torch.load(mean_path(layer_idx), map_location="cpu") if mean_path(layer_idx).exists() else {}
        )

    done_ids = set(mean_by_layer[layer_indices[-1]].keys())
    if done_ids:
        print(f"Resuming — {len(done_ids)} embeddings already saved")

    todo = [r for r in sequences if r.id not in done_ids]
    print(f"Sequences to embed: {len(todo)}")

    # Sort by length so sequences of similar size land in the same batch,
    # minimizing padding waste.
    todo.sort(key=lambda r: len(r.seq))

    def save_all():
        for layer_idx in layer_indices:
            torch.save(full_by_layer[layer_idx], full_path(layer_idx))
            torch.save(mean_by_layer[layer_idx], mean_path(layer_idx))

    n_since_checkpoint = 0
    with torch.no_grad():
        batch_starts = range(0, len(todo), batch_size)
        for batch_start in tqdm(batch_starts, desc="Embedding", unit="batch"):
            batch = todo[batch_start : batch_start + batch_size]
            tokens = client._tokenize([str(r.seq) for r in batch])
            output = client(tokens)
            hidden_states = output.hidden_states  # (n_layers_total, B, L, D)

            mask = tokens != pad_token_id
            for i, record in enumerate(batch):
                valid_len = int(mask[i].sum().item())
                for layer_idx in layer_indices:
                    # Trim the <cls>/<eos> special tokens, keep only real residue positions.
                    per_residue = hidden_states[layer_idx, i, 1 : valid_len - 1, :].cpu()
                    full_by_layer[layer_idx][record.id] = per_residue
                    mean_by_layer[layer_idx][record.id] = per_residue.mean(dim=0)

            n_since_checkpoint += len(batch)
            if n_since_checkpoint >= checkpoint_every:
                save_all()
                tqdm.write(f"Checkpoint: {len(mean_by_layer[layer_indices[-1]])} embeddings saved")
                n_since_checkpoint = 0

    save_all()
    print(f"\nSaved {len(mean_by_layer[layer_indices[-1]])} embeddings for layers {layer_indices}")
    for layer_idx in layer_indices:
        print(f"  layer {layer_idx}: {full_path(layer_idx)}  (full)   {mean_path(layer_idx)}  (mean)")


if __name__ == "__main__":
    typer.run(main)
